package claudecode

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/archer-developer/miranda-llm"
)

type recordingTracer struct {
	mu     sync.Mutex
	blocks []traceRecord
}

type traceRecord struct {
	provider, request, response string
	err                         error
}

func (r *recordingTracer) Trace(_ context.Context, provider, request, response string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.blocks = append(r.blocks, traceRecord{provider, request, response, err})
}

func newFakeProvider(t *testing.T, cfg Config) (*Provider, *recordingTracer) {
	t.Helper()
	t.Setenv(fakeEnv, "1")
	cfg.Binary = os.Args[0]
	p, err := New("cc", cfg, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	tr := &recordingTracer{}
	p.SetTracer(tr)
	return p, tr
}

func collect(t *testing.T, ch <-chan llm.StreamChunk) (string, error) {
	t.Helper()
	var text strings.Builder
	for c := range ch {
		if c.Err != nil {
			return text.String(), c.Err
		}
		text.WriteString(c.TextDelta)
	}
	return text.String(), nil
}

func TestProviderChatOneShot(t *testing.T) {
	p, tr := newFakeProvider(t, Config{Model: "sonnet"})
	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{
		{Role: llm.RoleSystem, Content: "transcribe"},
		{Role: llm.RoleUser, Content: "page", Parts: []llm.ContentPart{{ImageBase64: strings.Repeat("A", 1000), MIMEType: "application/pdf"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	text, err := collect(t, ch)
	if err != nil || text != "echo: page +1" {
		t.Fatalf("Chat = %q, %v; want the reply once (not the delta plus the result)", text, err)
	}

	if len(tr.blocks) != 1 || tr.blocks[0].provider != "cc" || tr.blocks[0].err != nil {
		t.Fatalf("trace blocks = %+v, want one block per call", tr.blocks)
	}
	if strings.Contains(tr.blocks[0].request, strings.Repeat("A", 100)) || !strings.Contains(tr.blocks[0].request, "omitted: 1000 base64 bytes") {
		t.Errorf("trace must elide attachment payloads: %s", tr.blocks[0].request)
	}
	if !strings.Contains(tr.blocks[0].response, `"text":"echo: page +1"`) {
		t.Errorf("trace response = %s", tr.blocks[0].response)
	}
}

func TestProviderStructuredReturnsStructuredOutput(t *testing.T) {
	p, tr := newFakeProvider(t, Config{})
	raw, err := p.Structured(context.Background(), llm.StructuredRequest{
		Messages:   []llm.Message{{Role: llm.RoleSystem, Content: "extract"}, {Role: llm.RoleUser, Content: "doc text"}},
		Schema:     map[string]any{"type": "object"},
		SchemaName: "medical_extraction",
	})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]string
	if err := json.Unmarshal(raw, &got); err != nil || got["echo"] != "doc text" {
		t.Fatalf("Structured = %s, %v", raw, err)
	}
	if len(tr.blocks) != 1 || !strings.Contains(tr.blocks[0].request, `"schema_name":"medical_extraction"`) {
		t.Fatalf("trace = %+v", tr.blocks)
	}
}

func TestProviderRejectsUnsupportedAttachmentUpFront(t *testing.T) {
	p, tr := newFakeProvider(t, Config{})
	_, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{
		{Role: llm.RoleUser, Parts: []llm.ContentPart{{MIMEType: "image/heic", ImageBase64: "x"}}},
	}})
	if err == nil || !strings.Contains(err.Error(), "image/heic") {
		t.Fatalf("err = %v, want unsupported attachment", err)
	}
	if len(tr.blocks) != 0 {
		t.Fatal("no process must be started for an unsupported attachment")
	}
}

func TestProviderRejectsTools(t *testing.T) {
	p, _ := newFakeProvider(t, Config{})
	_, err := p.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "x"}},
		Tools:    []llm.ToolDef{{Name: "t"}},
	})
	if !errors.Is(err, ErrToolsUnsupported) {
		t.Fatalf("err = %v, want ErrToolsUnsupported", err)
	}
}

func TestProviderSurfacesCLIErrors(t *testing.T) {
	p, tr := newFakeProvider(t, Config{})
	for text, want := range map[string]string{
		"fail":     "usage limit reached",
		"maxturns": "exceeded 4 turns",
		"crash":    "not logged in",
	} {
		ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: text}}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := collect(t, ch); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("Chat(%q) err = %v, want %q", text, err, want)
		}
	}
	for _, b := range tr.blocks {
		if b.err == nil {
			t.Errorf("a failed call must be traced with its error: %+v", b)
		}
	}
}

func TestProviderTimeoutKillsTheProcess(t *testing.T) {
	p, _ := newFakeProvider(t, Config{Timeout: 300 * time.Millisecond})
	start := time.Now()
	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hang"}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collect(t, ch); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
	if time.Since(start) > 10*time.Second {
		t.Fatal("a hung call must be killed at Timeout")
	}
}

func TestRenderRequest(t *testing.T) {
	system, msg, err := renderRequest([]llm.Message{
		{Role: llm.RoleSystem, Content: "a"},
		{Role: llm.RoleSystem, Content: "b"},
		{Role: llm.RoleUser, Content: "q1", Parts: []llm.ContentPart{{MIMEType: "image/png", ImageBase64: "x"}}},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{Name: "lab_results", Arguments: `{"a":1}`}}},
		{Role: llm.RoleTool, Content: "r"},
		{Role: llm.RoleAssistant, Content: "a1"},
		{Role: llm.RoleUser, Content: "q2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if system != "a\n\nb" {
		t.Errorf("system = %q", system)
	}
	for _, want := range []string{"<previous_conversation>", "[user]: q1", `[tool call] lab_results({"a":1})`, "[tool result]: r", "[assistant]: a1", "</previous_conversation>\n\nq2"} {
		if !strings.Contains(msg.Text, want) {
			t.Errorf("text missing %q:\n%s", want, msg.Text)
		}
	}
	if len(msg.Attachments) != 1 {
		t.Errorf("attachments = %v, want the earlier image carried along", msg.Attachments)
	}

	if _, msg, _ := renderRequest([]llm.Message{{Role: llm.RoleUser, Content: "only"}}); msg.Text != "only" {
		t.Errorf("single user message must be passed verbatim, got %q", msg.Text)
	}
	if _, _, err := renderRequest([]llm.Message{{Role: llm.RoleSystem, Content: "s"}}); err == nil {
		t.Error("a request without a final user message must be refused")
	}
}

func TestRenderTranscriptTruncatesToolResults(t *testing.T) {
	got := RenderTranscript([]llm.Message{{Role: llm.RoleTool, Content: "абвгд"}}, 3)
	if got != "[tool result]: абв…\n" {
		t.Fatalf("got %q", got)
	}
}

func TestTraceBuilderSplitsModelCallsAndStripsPrefix(t *testing.T) {
	tr := &recordingTracer{}
	ctx := context.Background()
	tb := NewTraceBuilder(tr, "claude_code", "mcp__medical__", "sys", UserMessage{Text: "q"}, []llm.ToolDef{{Name: "lab_results"}})
	tb.AddResponse(ctx, "m1", ContentBlock{Type: "thinking", Raw: json.RawMessage(`{"type":"thinking"}`)})
	tb.AddResponse(ctx, "m1", ContentBlock{Type: "tool_use", Name: "mcp__medical__lab_results", Raw: json.RawMessage(`{"type":"tool_use","id":"t1","name":"mcp__medical__lab_results","input":{}}`)})
	tb.ToolResults([]ContentBlock{{Type: "tool_result", Raw: json.RawMessage(`{"type":"tool_result","tool_use_id":"t1","content":"ok"}`)}})
	tb.AddResponse(ctx, "m2", ContentBlock{Type: "text", Raw: json.RawMessage(`{"type":"text","text":"answer"}`)})
	tb.Flush(ctx, nil)

	if len(tr.blocks) != 2 {
		t.Fatalf("blocks = %d, want one per model call", len(tr.blocks))
	}
	if !strings.Contains(tr.blocks[0].response, `"name":"lab_results"`) || strings.Contains(tr.blocks[0].response, "thinking") {
		t.Errorf("first block response = %s", tr.blocks[0].response)
	}
	if !strings.Contains(tr.blocks[1].request, `"tool_use_id":"t1"`) || !strings.Contains(tr.blocks[1].response, "answer") {
		t.Errorf("second block = %+v", tr.blocks[1])
	}
	if !strings.Contains(tr.blocks[1].request, `"role":"assistant"`) {
		t.Errorf("a follow-up request must carry the previous response, or analyze.IsFirstTurn mistakes it for a new question: %s", tr.blocks[1].request)
	}
}

// TestTraceBuilderKeepsInterleavedParallelCallsTogether reproduces the real
// CLI's event order for two parallel tool calls: the first call's result
// arrives between the two tool_use blocks of the same response.
func TestTraceBuilderKeepsInterleavedParallelCallsTogether(t *testing.T) {
	tr := &recordingTracer{}
	ctx := context.Background()
	tb := NewTraceBuilder(tr, "claude_code", "", "sys", UserMessage{Text: "q"}, nil)
	tb.AddResponse(ctx, "m1", ContentBlock{Type: "tool_use", Raw: json.RawMessage(`{"type":"tool_use","id":"t1"}`)})
	tb.ToolResults([]ContentBlock{{Raw: json.RawMessage(`{"type":"tool_result","tool_use_id":"t1"}`)}})
	tb.AddResponse(ctx, "m1", ContentBlock{Type: "tool_use", Raw: json.RawMessage(`{"type":"tool_use","id":"t2"}`)})
	tb.ToolResults([]ContentBlock{{Raw: json.RawMessage(`{"type":"tool_result","tool_use_id":"t2"}`)}})
	tb.AddResponse(ctx, "m2", ContentBlock{Type: "text", Raw: json.RawMessage(`{"type":"text","text":"done"}`)})
	tb.Flush(ctx, nil)

	if len(tr.blocks) != 2 || tb.Calls() != 2 {
		t.Fatalf("blocks = %d, calls = %d, want 2 (two model calls)", len(tr.blocks), tb.Calls())
	}
	for _, id := range []string{`"id":"t1"`, `"id":"t2"`} {
		if !strings.Contains(tr.blocks[0].response, id) {
			t.Errorf("first call's response lost %s: %s", id, tr.blocks[0].response)
		}
	}
	for _, id := range []string{`"tool_use_id":"t1"`, `"tool_use_id":"t2"`} {
		if !strings.Contains(tr.blocks[1].request, id) {
			t.Errorf("follow-up request lost %s: %s", id, tr.blocks[1].request)
		}
	}
}

func TestTraceBuilderTracesAFailedTurn(t *testing.T) {
	tr := &recordingTracer{}
	tb := NewTraceBuilder(tr, "claude_code", "", "sys", UserMessage{Text: "q"}, nil)
	tb.Flush(context.Background(), context.DeadlineExceeded)
	if len(tr.blocks) != 1 || tr.blocks[0].err == nil {
		t.Fatalf("blocks = %+v, want the error traced", tr.blocks)
	}
}
