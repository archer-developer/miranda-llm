package claudecode

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"
)

// fakeEnv switches the test binary into a fake `claude` (see TestMain):
// Start/Provider are pointed at os.Args[0], which then speaks just enough of
// the stream-json protocol to exercise this package without the real CLI.
const fakeEnv = "CLAUDECODE_FAKE_CLI"

func TestMain(m *testing.M) {
	if os.Getenv(fakeEnv) == "1" {
		runFakeCLI()
		return
	}
	os.Exit(m.Run())
}

// runFakeCLI answers each stdin message with a text delta, a tool_use +
// tool_result round, a final text block and a result whose text is
// "echo: <text>" plus the number of attachments. With --json-schema it
// answers with structured_output {"echo": <text>} instead. Special texts:
// "hang" never answers, "crash" exits non-zero with a stderr message,
// "fail" returns an is_error result, "maxturns" an error_max_turns one.
// Exits on stdin EOF, like the real CLI.
func runFakeCLI() {
	if len(os.Args) == 3 && os.Args[1] == "auth" && os.Args[2] == "status" {
		method := "claude.ai"
		if os.Getenv("ANTHROPIC_API_KEY") != "" {
			method = "api_key"
		}
		fmt.Printf(`{"loggedIn":true,"authMethod":%q}`+"\n", method)
		return
	}
	structured := slices.Contains(os.Args, "--json-schema")
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1<<20), 1<<20)
	out := json.NewEncoder(os.Stdout)
	for in.Scan() {
		var msg struct {
			Message struct {
				Content []map[string]any `json:"content"`
			} `json:"message"`
		}
		_ = json.Unmarshal(in.Bytes(), &msg)
		var text string
		attachments := 0
		for _, c := range msg.Message.Content {
			switch c["type"] {
			case "text":
				text, _ = c["text"].(string)
			case "image", "document":
				attachments++
			}
		}
		switch text {
		case "hang":
			time.Sleep(time.Hour)
		case "crash":
			fmt.Fprintln(os.Stderr, "boom: not logged in")
			os.Exit(3)
		case "fail":
			_ = out.Encode(map[string]any{"type": "result", "subtype": "success", "is_error": true, "result": "usage limit reached"})
			continue
		case "maxturns":
			_ = out.Encode(map[string]any{"type": "result", "subtype": "error_max_turns", "is_error": true})
			continue
		}
		reply := "echo: " + text
		if attachments > 0 {
			reply += fmt.Sprintf(" +%d", attachments)
		}
		_ = out.Encode(map[string]any{"type": "system", "subtype": "init", "tools": []string{"mcp__x__a"}})
		_ = out.Encode(map[string]any{"type": "stream_event", "event": map[string]any{
			"type": "content_block_delta", "delta": map[string]any{"type": "text_delta", "text": reply}}})
		_ = out.Encode(map[string]any{"type": "assistant", "message": map[string]any{
			"id": "msg_1", "content": []any{map[string]any{"type": "tool_use", "id": "t1", "name": "mcp__x__a", "input": map[string]any{"a": 1}}}}})
		_ = out.Encode(map[string]any{"type": "user", "message": map[string]any{
			"content": []any{map[string]any{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}}}})
		_ = out.Encode(map[string]any{"type": "assistant", "message": map[string]any{
			"id": "msg_2", "content": []any{map[string]any{"type": "text", "text": reply}}}})
		res := map[string]any{"type": "result", "subtype": "success", "result": reply, "num_turns": 2, "session_id": "s1"}
		if structured {
			res["structured_output"] = map[string]any{"echo": text}
			res["result"] = ""
		}
		_ = out.Encode(res)
	}
}

func startFake(t *testing.T) *Process {
	t.Helper()
	t.Setenv(fakeEnv, "1")
	p, err := Start(Options{Binary: os.Args[0], SystemPrompt: "secret prompt"})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

func TestTurnRoundTripsAndKeepsProcess(t *testing.T) {
	p := startFake(t)
	for _, text := range []string{"привет", "ещё раз"} {
		var deltas string
		var blocks []string
		var toolResults int
		var initTools []string
		res, err := p.Turn(context.Background(), UserMessage{Text: text}, Handler{
			OnInit:        func(tools []string) { initTools = tools },
			OnTextDelta:   func(d string) { deltas += d },
			OnAssistant:   func(id string, b ContentBlock) { blocks = append(blocks, id+":"+b.Type) },
			OnToolResults: func(bs []ContentBlock) { toolResults += len(bs) },
		})
		if err != nil {
			t.Fatalf("Turn(%q): %v", text, err)
		}
		want := "echo: " + text
		if res.Text != want || deltas != want || res.Subtype != "success" || res.NumTurns != 2 {
			t.Fatalf("Turn(%q) = %+v, deltas %q", text, res, deltas)
		}
		if !slices.Equal(blocks, []string{"msg_1:tool_use", "msg_2:text"}) || toolResults != 1 || !slices.Equal(initTools, []string{"mcp__x__a"}) {
			t.Fatalf("blocks %v, toolResults %d, init %v", blocks, toolResults, initTools)
		}
	}
	if !p.Alive() {
		t.Fatal("process should survive between turns")
	}
}

func TestCloseEndsTheProcess(t *testing.T) {
	p := startFake(t)
	if _, err := p.Turn(context.Background(), UserMessage{Text: "x"}, Handler{}); err != nil {
		t.Fatal(err)
	}
	_ = p.Close()
	if p.Alive() {
		t.Fatal("Close must end the process — one-shot callers rely on it dying after answering")
	}
}

func TestTurnContextCancelKillsProcess(t *testing.T) {
	p := startFake(t)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := p.Turn(ctx, UserMessage{Text: "hang"}, Handler{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for p.Alive() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if p.Alive() {
		t.Fatal("process should be killed after a cancelled turn")
	}
}

func TestTurnReportsExitWithStderr(t *testing.T) {
	p := startFake(t)
	_, err := p.Turn(context.Background(), UserMessage{Text: "crash"}, Handler{})
	if !errors.Is(err, ErrProcessExited) || !strings.Contains(err.Error(), "not logged in") {
		t.Fatalf("err = %v, want ErrProcessExited with stderr tail", err)
	}
}

func TestCloseRemovesWorkDir(t *testing.T) {
	p := startFake(t)
	if _, err := os.Stat(p.dir); err != nil {
		t.Fatalf("work dir missing while running: %v", err)
	}
	_ = p.Close()
	if _, err := os.Stat(p.dir); !os.IsNotExist(err) {
		t.Fatalf("work dir should be removed, stat err = %v", err)
	}
}

func TestBuildArgsIsolatesTheProcess(t *testing.T) {
	args, err := buildArgs(Options{
		Model: "haiku", Effort: "low", MaxTurns: 7, BuiltinTools: []string{"WebSearch"},
		AllowedTools: []string{"mcp__x__*"}, JSONSchema: map[string]any{"type": "object"}, ExtraArgs: []string{"--x"},
	}, "/p", "/m")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--input-format stream-json", "--output-format stream-json", "--no-session-persistence",
		"--strict-mcp-config", "--mcp-config /m", "--disable-slash-commands", "--system-prompt-file /p",
		"--tools WebSearch", "--allowedTools WebSearch,mcp__x__*", "--model haiku", "--effort low", "--max-turns 7",
		`--json-schema {"type":"object"}`,
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args missing %q: %v", want, args)
		}
	}
	if i := slices.Index(args, "--setting-sources"); i < 0 || args[i+1] != "" {
		t.Errorf("--setting-sources must be followed by an empty value: %v", args)
	}
	if slices.Contains(args, "--bare") {
		t.Error("--bare must not be used: it never reads the subscription login")
	}
	if args[len(args)-1] != "--x" {
		t.Errorf("ExtraArgs must come last: %v", args)
	}

	none, _ := buildArgs(Options{}, "/p", "/m")
	if i := slices.Index(none, "--tools"); i < 0 || none[i+1] != "" {
		t.Errorf("no BuiltinTools must disable every built-in tool (--tools \"\"): %v", none)
	}
	for _, flag := range []string{"--allowedTools", "--json-schema", "--model", "--max-turns"} {
		if slices.Contains(none, flag) {
			t.Errorf("unset option must not emit %s: %v", flag, none)
		}
	}
}

func TestEncodeUserMessageAttachments(t *testing.T) {
	line, err := encodeUserMessage(UserMessage{Text: "read", Attachments: []Attachment{
		{MIMEType: "application/pdf", Base64: "UERG"}, {MIMEType: "image/png", Base64: "UE5H"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var msg struct {
		Message struct {
			Content []map[string]any `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(line, &msg); err != nil {
		t.Fatal(err)
	}
	var types []string
	for _, c := range msg.Message.Content {
		types = append(types, c["type"].(string))
	}
	if !slices.Equal(types, []string{"document", "image", "text"}) {
		t.Fatalf("content types = %v, want attachments first then the text", types)
	}
	for _, mime := range []string{"text/plain", "image/heic"} {
		if _, err := encodeUserMessage(UserMessage{Attachments: []Attachment{{MIMEType: mime}}}); err == nil {
			t.Fatalf("unsupported attachment type %s must be refused", mime)
		}
	}
}

func TestCheckAuthUsesTheTurnEnvironment(t *testing.T) {
	t.Setenv(fakeEnv, "1")
	t.Setenv("ANTHROPIC_API_KEY", "sk-test")
	st, err := CheckAuth(context.Background(), os.Args[0], false)
	if err != nil || !st.LoggedIn || st.AuthMethod != "claude.ai" {
		t.Fatalf("CheckAuth(strip) = %+v, %v; want the subscription login", st, err)
	}
	st, err = CheckAuth(context.Background(), os.Args[0], true)
	if err != nil || st.AuthMethod != "api_key" {
		t.Fatalf("CheckAuth(keep) = %+v, %v; want api_key", st, err)
	}
}

func TestChildEnvStripsAPIKeys(t *testing.T) {
	env := []string{"PATH=/bin", "ANTHROPIC_API_KEY=sk", "ANTHROPIC_AUTH_TOKEN=t", "CLAUDECODE=1", "HOME=/h"}
	if got := childEnv(env, false); !slices.Equal(got, []string{"PATH=/bin", "HOME=/h"}) {
		t.Fatalf("childEnv(false) = %v", got)
	}
	if got := childEnv(env, true); !slices.Equal(got, []string{"PATH=/bin", "ANTHROPIC_API_KEY=sk", "ANTHROPIC_AUTH_TOKEN=t", "HOME=/h"}) {
		t.Fatalf("childEnv(true) = %v", got)
	}
}

func TestResolveBinaryIsAbsolute(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(dir+"/fakeclaude", []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	got, err := ResolveBinary("./fakeclaude")
	if err != nil || !strings.HasPrefix(got, "/") {
		t.Fatalf("ResolveBinary = %q, %v; want an absolute path (Start runs in a temp dir)", got, err)
	}
	if _, err := ResolveBinary("./missing"); err == nil {
		t.Fatal("a missing binary must be an error")
	}
}
