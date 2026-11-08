// Package claudecode drives the headless Claude Code CLI (`claude -p
// --input-format stream-json --output-format stream-json`) — a backend that
// spends a Claude subscription's usage window (claude.ai login) instead of
// per-token API budget.
//
// Two layers:
//
//   - Process (this file) owns one CLI process and its stream-json wire
//     protocol: write a user message to stdin, read NDJSON events from stdout
//     until that message's "result" event. It knows nothing about any
//     consuming service — no history, no tools of its own; a caller that
//     wants the model to call its tools exposes them through an MCP server it
//     names in Options.MCPServers.
//   - Provider (provider.go) is an llm.Provider + llm.StructuredProvider on
//     top of it for one-shot calls (OCR, structured extraction, ...): every
//     Chat/Structured call starts a fresh process, sends one message and
//     lets the process exit once it has answered.
//
// Ported from miranda's internal/claudecode (the claude_code orchestrator,
// miranda/docs/adr/claude-code-orchestrator.md), which proved the flag set
// and protocol against the real CLI; generalized here (attachments beyond
// images, --json-schema structured output, no implicit ToolSearch) so
// medical-card's pipeline provider and medical.ask engine can share it. See
// miranda-medical-card/docs/adr/010-claude-code-engine.md.
package claudecode

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// MCPServer is one --mcp-config entry, in the CLI's own JSON shape. Only the
// HTTP transport is modeled — the one a consuming service's loopback MCP
// endpoint uses.
type MCPServer struct {
	Type    string            `json:"type"` // "http"
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers,omitempty"`
}

// Options configures Start.
type Options struct {
	// Binary is the CLI executable ("claude" when empty). It must be logged
	// in (`claude auth login`) as the OS user the caller runs as.
	Binary string
	// Model is passed as --model ("sonnet", "opus", "haiku" or a full model
	// id); "" keeps the CLI default.
	Model string
	// Effort is passed as --effort ("low", "medium", "high", ...); "" keeps
	// the CLI default.
	Effort string
	// SystemPrompt fully replaces Claude Code's own default system prompt
	// (--system-prompt-file) — the process acts as whatever the caller says,
	// not as a coding agent. Fixed for the process's lifetime.
	SystemPrompt string
	// MCPServers are the only MCP servers the process sees
	// (--strict-mcp-config ignores the invoking user's own ~/.claude config).
	MCPServers map[string]MCPServer
	// BuiltinTools are the Claude Code built-in tools enabled for the process
	// (--tools), exactly as listed — empty disables every built-in tool. Each
	// one is also auto-approved. A caller exposing a large MCP tool set
	// should list "ToolSearch" here: the CLI only defers MCP tool schemas
	// (names only, schema fetched on demand) when ToolSearch is available —
	// measured in miranda with 120 tools, 55,786 vs 8,949 input tokens per
	// turn. With a dozen tools it isn't worth an extra round trip.
	BuiltinTools []string
	// AllowedTools are auto-approved without a permission prompt (in -p mode
	// an unapproved tool is simply denied), e.g. "mcp__medical__*".
	AllowedTools []string
	// JSONSchema, when set, is passed as --json-schema: the CLI then makes
	// the model answer through a validated StructuredOutput tool call and
	// reports the value as Result.StructuredOutput. That tool call is a turn
	// of its own, so MaxTurns must leave room for it (>= 2).
	JSONSchema map[string]any
	// MaxTurns bounds one Turn's internal agent loop (--max-turns); 0 leaves
	// the CLI default.
	MaxTurns int
	// AllowAPIKeyEnv keeps ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN in the
	// child's environment. By default they are stripped: with either set,
	// the CLI silently prefers the API key over the subscription login —
	// and a service that also has an "anthropic" provider configured always
	// has one in its environment.
	AllowAPIKeyEnv bool
	// ExtraArgs are appended verbatim after every flag Start sets itself.
	ExtraArgs []string
}

// Attachment is one base64-encoded file attached to a user message: an
// image ("image/jpeg", "image/png", "image/gif", "image/webp") or a PDF
// ("application/pdf"). Both are passed straight through as Anthropic
// Messages content blocks — verified against the real CLI (2.1.286) for a
// multi-page scanned PDF.
type Attachment struct {
	MIMEType string
	Base64   string
}

// UserMessage is what one Turn sends.
type UserMessage struct {
	Text        string
	Attachments []Attachment
}

// ContentBlock is one Anthropic Messages API content block as it appears in
// stream-json "assistant"/"user" events — text, tool_use, tool_result,
// thinking. Only the fields callers read are modeled; Raw keeps the block
// verbatim for tracing.
type ContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	Raw       json.RawMessage `json:"-"`
}

// Handler receives a Turn's events as they stream in. Every field is
// optional. Callbacks run on the goroutine that called Turn.
type Handler struct {
	// OnInit receives the tool names the CLI offers the model for this turn
	// (its per-turn "system/init" event).
	OnInit func(tools []string)
	// OnTextDelta receives assistant text as it is generated (from
	// --include-partial-messages), for live display/streaming.
	OnTextDelta func(delta string)
	// OnAssistant receives each completed assistant content block — the CLI
	// emits one "assistant" event per block, so a response with text plus
	// two tool calls arrives as three calls sharing messageID.
	OnAssistant func(messageID string, block ContentBlock)
	// OnToolResults receives the tool results the CLI fed back to the model
	// after executing its tool calls — the boundary between two model calls
	// inside one Turn.
	OnToolResults func(blocks []ContentBlock)
}

// Result is a Turn's terminal "result" event.
type Result struct {
	// Subtype is "success", or an error kind such as "error_max_turns" /
	// "error_during_execution".
	Subtype string
	IsError bool
	// Text is the final assistant text (the "result" field).
	Text string
	// StructuredOutput is the validated value the model produced for
	// Options.JSONSchema; nil when no schema was set.
	StructuredOutput json.RawMessage
	NumTurns         int
	DurationMS       int64
	// CostUSD is the CLI's API-price estimate; under a subscription login
	// nothing is actually billed per token, but it still tracks usage.
	CostUSD   float64
	SessionID string
	Usage     json.RawMessage
}

// ErrProcessExited is returned by Turn when the process ended before
// producing that turn's result. The Process is unusable afterwards.
var ErrProcessExited = errors.New("claudecode: process exited")

// Process is one running CLI process. Turns are serialized: Turn holds an
// internal lock for its whole duration.
type Process struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	events chan json.RawMessage
	// exited is closed once the stdout reader sees EOF (the process is gone
	// or going); waitErr is valid after that.
	exited  chan struct{}
	waitErr error
	stderr  *tailBuffer
	dir     string

	turnMu    sync.Mutex
	closeOnce sync.Once
}

// Start launches the CLI. The process runs in a private temporary working
// directory holding its system prompt and MCP config (both may contain
// sensitive values — medical data in a prompt, an MCP bearer token — so the
// directory is 0700 and removed by Close). No conversation transcript is
// persisted (--no-session-persistence), none of the invoking user's own
// Claude Code settings, hooks, skills or MCP servers are loaded, and only
// opts.BuiltinTools of the built-in tools are enabled — beyond those, the
// process can only act through opts.MCPServers.
func Start(opts Options) (*Process, error) {
	binary := opts.Binary
	if binary == "" {
		binary = "claude"
	}
	dir, err := os.MkdirTemp("", "claudecode-")
	if err != nil {
		return nil, fmt.Errorf("claudecode: create work dir: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(dir) }

	promptPath := filepath.Join(dir, "system-prompt.md")
	if err := os.WriteFile(promptPath, []byte(opts.SystemPrompt), 0o600); err != nil {
		cleanup()
		return nil, fmt.Errorf("claudecode: write system prompt: %w", err)
	}
	servers := opts.MCPServers
	if servers == nil {
		servers = map[string]MCPServer{}
	}
	mcpJSON, err := json.Marshal(map[string]any{"mcpServers": servers})
	if err != nil {
		cleanup()
		return nil, fmt.Errorf("claudecode: encode mcp config: %w", err)
	}
	mcpPath := filepath.Join(dir, "mcp.json")
	if err := os.WriteFile(mcpPath, mcpJSON, 0o600); err != nil {
		cleanup()
		return nil, fmt.Errorf("claudecode: write mcp config: %w", err)
	}
	args, err := buildArgs(opts, promptPath, mcpPath)
	if err != nil {
		cleanup()
		return nil, err
	}

	cmd := exec.Command(binary, args...)
	cmd.Dir = dir
	cmd.Env = childEnv(os.Environ(), opts.AllowAPIKeyEnv)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cleanup()
		return nil, fmt.Errorf("claudecode: stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cleanup()
		return nil, fmt.Errorf("claudecode: stdout pipe: %w", err)
	}
	tail := &tailBuffer{max: 4096}
	cmd.Stderr = tail
	if err := cmd.Start(); err != nil {
		cleanup()
		return nil, fmt.Errorf("claudecode: start %s: %w", binary, err)
	}

	p := &Process{
		cmd:    cmd,
		stdin:  stdin,
		events: make(chan json.RawMessage, 64),
		exited: make(chan struct{}),
		stderr: tail,
		dir:    dir,
	}
	go p.readLoop(stdout)
	return p, nil
}

// buildArgs assembles the CLI command line — split out of Start so the exact
// flag set is unit-testable without spawning anything.
func buildArgs(opts Options, promptPath, mcpPath string) ([]string, error) {
	args := []string{
		"-p",
		"--input-format", "stream-json",
		"--output-format", "stream-json",
		"--verbose", // required by -p together with stream-json output
		"--include-partial-messages",
		"--no-session-persistence",
		// Only the listed built-in tools exist for the model ("" = none).
		// MCP tools are unaffected.
		"--tools", strings.Join(opts.BuiltinTools, ","),
		"--strict-mcp-config",
		"--mcp-config", mcpPath,
		// Ignore the invoking OS user's own user/project/local settings
		// (their hooks, permissions, plugins). --bare would be stricter, but
		// it never reads the OAuth login, i.e. the subscription.
		"--setting-sources", "",
		"--disable-slash-commands",
		"--system-prompt-file", promptPath,
	}
	if allowed := append(append([]string{}, opts.BuiltinTools...), opts.AllowedTools...); len(allowed) > 0 {
		args = append(args, "--allowedTools", strings.Join(allowed, ","))
	}
	if opts.Model != "" {
		args = append(args, "--model", opts.Model)
	}
	if opts.Effort != "" {
		args = append(args, "--effort", opts.Effort)
	}
	if opts.MaxTurns > 0 {
		args = append(args, "--max-turns", strconv.Itoa(opts.MaxTurns))
	}
	if opts.JSONSchema != nil {
		schema, err := json.Marshal(opts.JSONSchema)
		if err != nil {
			return nil, fmt.Errorf("claudecode: encode json schema: %w", err)
		}
		args = append(args, "--json-schema", string(schema))
	}
	return append(args, opts.ExtraArgs...), nil
}

// AuthStatus is the subset of `claude auth status` callers log.
type AuthStatus struct {
	LoggedIn         bool   `json:"loggedIn"`
	AuthMethod       string `json:"authMethod"`
	SubscriptionType string `json:"subscriptionType"`
}

// CheckAuth runs `claude auth status` with exactly the environment Start
// gives a process (childEnv) — otherwise an ANTHROPIC_API_KEY in the
// caller's own environment makes the CLI report "api_key" even though every
// process, run with that key stripped, actually uses the subscription login.
func CheckAuth(ctx context.Context, binary string, allowAPIKeyEnv bool) (AuthStatus, error) {
	if binary == "" {
		binary = "claude"
	}
	cmd := exec.CommandContext(ctx, binary, "auth", "status")
	cmd.Env = childEnv(os.Environ(), allowAPIKeyEnv)
	out, err := cmd.Output()
	if err != nil {
		return AuthStatus{}, fmt.Errorf("claudecode: auth status: %w", err)
	}
	var st AuthStatus
	if err := json.Unmarshal(out, &st); err != nil {
		return AuthStatus{}, fmt.Errorf("claudecode: parse auth status: %w", err)
	}
	return st, nil
}

// childEnv is the process environment minus the API-key variables (unless
// allowAPIKey), so the CLI authenticates with its own subscription login.
// CLAUDECODE is dropped too: it marks "running inside Claude Code", which is
// only ever true by accident here (a developer launching the service from a
// Claude Code session) and changes the CLI's behavior.
func childEnv(env []string, allowAPIKey bool) []string {
	out := make([]string, 0, len(env))
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		switch name {
		case "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN":
			if !allowAPIKey {
				continue
			}
		case "CLAUDECODE":
			continue
		}
		out = append(out, kv)
	}
	return out
}

// readLoop forwards stdout's NDJSON lines to p.events until EOF, then reaps
// the process. A bufio.Reader (not Scanner) since a single event carrying a
// large tool result or document has no useful upper bound.
func (p *Process) readLoop(stdout io.Reader) {
	r := bufio.NewReader(stdout)
	for {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); len(line) > 0 && json.Valid(line) {
			p.events <- json.RawMessage(line)
		}
		if err != nil {
			break
		}
	}
	p.waitErr = p.cmd.Wait()
	close(p.exited)
	close(p.events)
}

// Alive reports whether the process is still running.
func (p *Process) Alive() bool {
	select {
	case <-p.exited:
		return false
	default:
		return true
	}
}

// Turn sends msg and blocks until its "result" event, streaming intermediate
// events to h. If ctx ends first the process is killed — its conversation
// state is then unknown, so a half-finished turn can't leak into the next
// one — and ctx.Err() is returned; the caller must Start a new Process.
func (p *Process) Turn(ctx context.Context, msg UserMessage, h Handler) (Result, error) {
	p.turnMu.Lock()
	defer p.turnMu.Unlock()

	if !p.Alive() {
		return Result{}, p.exitError()
	}
	line, err := encodeUserMessage(msg)
	if err != nil {
		return Result{}, err
	}
	if _, err := p.stdin.Write(line); err != nil {
		return Result{}, fmt.Errorf("claudecode: write message: %w", err)
	}

	for {
		select {
		case <-ctx.Done():
			p.kill()
			return Result{}, ctx.Err()
		case raw, ok := <-p.events:
			if !ok {
				return Result{}, p.exitError()
			}
			if res, done := dispatch(raw, h); done {
				return res, nil
			}
		}
	}
}

// encodeUserMessage renders msg as one stream-json input line. Attachments
// go first, then the text — the order Anthropic recommends for documents
// and images referred to by the instruction that follows them.
func encodeUserMessage(msg UserMessage) ([]byte, error) {
	content := make([]map[string]any, 0, 1+len(msg.Attachments))
	for _, a := range msg.Attachments {
		block, err := attachmentBlock(a)
		if err != nil {
			return nil, err
		}
		content = append(content, block)
	}
	if msg.Text != "" || len(content) == 0 {
		content = append(content, map[string]any{"type": "text", "text": msg.Text})
	}
	line, err := json.Marshal(map[string]any{
		"type":    "user",
		"message": map[string]any{"role": "user", "content": content},
	})
	if err != nil {
		return nil, fmt.Errorf("claudecode: encode message: %w", err)
	}
	return append(line, '\n'), nil
}

// attachmentBlock maps an attachment to its Messages API content block —
// "image" for the image types the API accepts, "document" for a PDF.
// Anything else (e.g. image/heic, which phones produce and Gemini accepts) is
// refused here with a clear error, before a process is even started, rather
// than sent and rejected opaquely by the API mid-process — a caller with an
// escalation target (medical-card's OCR) then falls back to it.
func attachmentBlock(a Attachment) (map[string]any, error) {
	source := map[string]any{"type": "base64", "media_type": a.MIMEType, "data": a.Base64}
	switch a.MIMEType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return map[string]any{"type": "image", "source": source}, nil
	case "application/pdf":
		return map[string]any{"type": "document", "source": source}, nil
	default:
		return nil, fmt.Errorf("claudecode: unsupported attachment type %q (only image/jpeg|png|gif|webp and application/pdf)", a.MIMEType)
	}
}

// ResolveBinary turns binary ("claude" when empty) into an absolute path:
// Start runs the CLI with its working directory set to a fresh temp dir, so
// a relative path like "./bin/claude" that resolved fine at startup would
// fail on every call.
func ResolveBinary(binary string) (string, error) {
	if binary == "" {
		binary = "claude"
	}
	path, err := exec.LookPath(binary)
	if err != nil {
		return "", fmt.Errorf("claudecode: binary %q not found: %w", binary, err)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("claudecode: binary %q: %w", binary, err)
	}
	return abs, nil
}

// event is the union of the stream-json output event fields dispatch reads.
type event struct {
	Type    string `json:"type"`
	Subtype string `json:"subtype"`
	Message struct {
		ID      string            `json:"id"`
		Content []json.RawMessage `json:"content"`
	} `json:"message"`
	Event struct {
		Type  string `json:"type"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"delta"`
	} `json:"event"`
	Tools            []string        `json:"tools"`
	IsError          bool            `json:"is_error"`
	Result           string          `json:"result"`
	StructuredOutput json.RawMessage `json:"structured_output"`
	NumTurns         int             `json:"num_turns"`
	DurationMS       int64           `json:"duration_ms"`
	TotalCostUSD     float64         `json:"total_cost_usd"`
	SessionID        string          `json:"session_id"`
	Usage            json.RawMessage `json:"usage"`
}

// dispatch routes one output event to h, reporting done with the decoded
// Result once it sees the turn's "result" event. Events it doesn't model
// (rate_limit_event, thinking_tokens, hook events, ...) are ignored.
func dispatch(raw json.RawMessage, h Handler) (Result, bool) {
	var ev event
	if err := json.Unmarshal(raw, &ev); err != nil {
		return Result{}, false
	}
	switch ev.Type {
	case "system":
		if ev.Subtype == "init" && h.OnInit != nil {
			h.OnInit(ev.Tools)
		}
	case "stream_event":
		if ev.Event.Type == "content_block_delta" && ev.Event.Delta.Type == "text_delta" && h.OnTextDelta != nil {
			h.OnTextDelta(ev.Event.Delta.Text)
		}
	case "assistant":
		if h.OnAssistant != nil {
			for _, b := range decodeBlocks(ev.Message.Content) {
				h.OnAssistant(ev.Message.ID, b)
			}
		}
	case "user":
		// Only tool results are interesting: the CLI also echoes plain user
		// text in some modes (--replay-user-messages), which isn't news.
		var results []ContentBlock
		for _, b := range decodeBlocks(ev.Message.Content) {
			if b.Type == "tool_result" {
				results = append(results, b)
			}
		}
		if len(results) > 0 && h.OnToolResults != nil {
			h.OnToolResults(results)
		}
	case "result":
		structured := ev.StructuredOutput
		if string(structured) == "null" {
			structured = nil
		}
		return Result{
			Subtype:          ev.Subtype,
			IsError:          ev.IsError,
			Text:             ev.Result,
			StructuredOutput: structured,
			NumTurns:         ev.NumTurns,
			DurationMS:       ev.DurationMS,
			CostUSD:          ev.TotalCostUSD,
			SessionID:        ev.SessionID,
			Usage:            ev.Usage,
		}, true
	}
	return Result{}, false
}

func decodeBlocks(raws []json.RawMessage) []ContentBlock {
	out := make([]ContentBlock, 0, len(raws))
	for _, r := range raws {
		var b ContentBlock
		if json.Unmarshal(r, &b) != nil {
			continue
		}
		b.Raw = r
		out = append(out, b)
	}
	return out
}

func (p *Process) exitError() error {
	<-p.exited
	if tail := strings.TrimSpace(p.stderr.String()); tail != "" {
		return fmt.Errorf("%w (%v): %s", ErrProcessExited, p.waitErr, tail)
	}
	return fmt.Errorf("%w (%v)", ErrProcessExited, p.waitErr)
}

func (p *Process) kill() {
	if p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
}

// Close ends the process — gracefully first (closing stdin makes the CLI exit
// after any in-flight turn), killing it if it hasn't exited within a few
// seconds — and removes its working directory. Safe to call more than once
// and concurrently with a Turn (which then fails with ErrProcessExited).
// One-shot callers (Provider, a per-request engine) call it right after
// their single Turn: that is what makes the process die once it has
// answered.
func (p *Process) Close() error {
	p.closeOnce.Do(func() {
		// Drain whatever is still buffered or in flight: readLoop blocks on a
		// full p.events with nobody reading, and would then never reach the
		// Wait that closes p.exited.
		go func() {
			for range p.events {
			}
		}()
		_ = p.stdin.Close()
		select {
		case <-p.exited:
		case <-time.After(5 * time.Second):
			p.kill()
			<-p.exited
		}
		_ = os.RemoveAll(p.dir)
	})
	return nil
}

// tailBuffer keeps the last max bytes written to it — the process's stderr,
// surfaced in ErrProcessExited so a crash (or "not logged in") explains
// itself without an unbounded buffer.
type tailBuffer struct {
	mu  sync.Mutex
	max int
	buf []byte
}

func (t *tailBuffer) Write(b []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, b...)
	if over := len(t.buf) - t.max; over > 0 {
		t.buf = t.buf[over:]
	}
	return len(b), nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}
