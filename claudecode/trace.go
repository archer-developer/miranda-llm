package claudecode

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	llm "github.com/archer-developer/miranda-llm"
)

// traceContent renders a user message as Messages API content blocks for a
// trace — with each attachment's base64 payload replaced by its size: a
// scanned PDF is hundreds of KB of base64 that would dwarf everything else
// in llm.log while telling a reader nothing (the source file is stored by
// the caller anyway).
func traceContent(msg UserMessage) []any {
	content := make([]any, 0, 1+len(msg.Attachments))
	for _, a := range msg.Attachments {
		kind := "document"
		if strings.HasPrefix(a.MIMEType, "image/") {
			kind = "image"
		}
		content = append(content, map[string]any{
			"type":   kind,
			"source": map[string]any{"type": "base64", "media_type": a.MIMEType, "data": fmt.Sprintf("<omitted: %d base64 bytes>", len(a.Base64))},
		})
	}
	return append(content, map[string]string{"type": "text", "text": msg.Text})
}

// TraceBuilder renders a multi-step Turn (one where the CLI calls tools) into
// llm.Tracer blocks, one per model call inside the CLI. Responses use the
// Anthropic Messages shape ({"content":[...]}), so llmtrace/analyze — and
// with it llmtrace/anomaly and each service's llm-trace CLI — read them
// unchanged.
//
// A model call is identified by its assistant message id, not by tool
// results arriving: the CLI emits one "assistant" event per content block and
// starts executing a block's tool call as soon as it arrives, so with
// parallel tool calls the results of the first calls are interleaved with the
// later tool_use blocks of the same model response (seen against the real
// CLI). Tool results are collected until the next message id appears, and
// become that call's request.
//
// Wire it as Handler{OnAssistant: tb.AddResponse-wrapper, OnToolResults:
// tb.ToolResults} and call Flush once Turn returns. Not safe for concurrent
// use; a Turn's Handler callbacks all run on one goroutine.
type TraceBuilder struct {
	provider   string
	toolPrefix string
	tracer     llm.Tracer

	request  any
	started  time.Time
	msgID    string
	response []json.RawMessage
	// results are the tool results seen since the current response began —
	// the next model call's request.
	results []json.RawMessage
	// calls counts model calls seen (distinct assistant message ids).
	calls int
}

// Calls is how many model calls the Turn made so far — what a caller reports
// as its iteration count when Turn failed (timeout, process exit) and there
// is no result event's num_turns to read.
func (tb *TraceBuilder) Calls() int { return tb.calls }

// NewTraceBuilder starts the trace of one Turn; tracer may be nil (then
// nothing is written). provider tags every block; toolPrefix is the CLI's
// MCP namespace for the caller's tools ("mcp__<server>__"), stripped from
// tool_use names so the trace names tools exactly as the request's tool list
// does — otherwise the anomaly detector would flag every call as an unknown
// tool.
func NewTraceBuilder(tracer llm.Tracer, provider, toolPrefix, system string, msg UserMessage, tools []llm.ToolDef) *TraceBuilder {
	names := make([]map[string]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, map[string]string{"name": t.Name})
	}
	return &TraceBuilder{
		provider:   provider,
		toolPrefix: toolPrefix,
		tracer:     tracer,
		request: map[string]any{
			"system":   []map[string]string{{"type": "text", "text": system}},
			"messages": []any{map[string]any{"role": "user", "content": traceContent(msg)}},
			"tools":    names,
		},
		started: time.Now(),
	}
}

// AddResponse records one assistant content block (Handler.OnAssistant). A
// block with a new message id starts the next model call: the previous one is
// written out, and the next call's request becomes [assistant: previous
// response, user: the tool results seen since] — the shape the Messages API
// itself sends. (A results-only request would also be misread:
// analyze.IsFirstTurn treats a single-message request as a conversation's
// first turn, splitting one untagged question into several.)
func (tb *TraceBuilder) AddResponse(ctx context.Context, messageID string, b ContentBlock) {
	if tb.msgID != "" && messageID != tb.msgID && len(tb.response) > 0 {
		prev := tb.response
		tb.write(ctx, nil)
		tb.request = map[string]any{"messages": []any{
			map[string]any{"role": "assistant", "content": prev},
			map[string]any{"role": "user", "content": tb.results},
		}}
		tb.results = nil
		tb.started = time.Now()
	}
	if messageID != tb.msgID {
		tb.calls++
	}
	tb.msgID = messageID
	// The CLI doesn't expose thinking text (the blocks arrive empty), so
	// they would only add noise to the trace.
	if b.Type == "thinking" || b.Type == "redacted_thinking" {
		return
	}
	tb.response = append(tb.response, tb.unprefixToolUse(b))
}

func (tb *TraceBuilder) unprefixToolUse(b ContentBlock) json.RawMessage {
	name, ok := strings.CutPrefix(b.Name, tb.toolPrefix)
	if b.Type != "tool_use" || tb.toolPrefix == "" || !ok {
		return b.Raw
	}
	var m map[string]any
	if json.Unmarshal(b.Raw, &m) != nil {
		return b.Raw
	}
	m["name"] = name
	out, err := json.Marshal(m)
	if err != nil {
		return b.Raw
	}
	return out
}

// ToolResults records the tool results the CLI fed back
// (Handler.OnToolResults); they become the next model call's request.
func (tb *TraceBuilder) ToolResults(results []ContentBlock) {
	for _, r := range results {
		tb.results = append(tb.results, r.Raw)
	}
}

// Flush writes the last model call's block. Call it once after Turn returns,
// with Turn's error (a failed Turn is traced with it even when the call
// produced no output).
func (tb *TraceBuilder) Flush(ctx context.Context, err error) {
	if len(tb.response) == 0 && err == nil {
		return
	}
	tb.write(ctx, err)
}

func (tb *TraceBuilder) write(ctx context.Context, err error) {
	if tb.tracer == nil {
		return
	}
	req, _ := json.Marshal(tb.request)
	var resp []byte
	if err == nil {
		resp, _ = json.Marshal(map[string]any{
			"content":     tb.response,
			"duration_ms": time.Since(tb.started).Milliseconds(),
		})
	}
	tb.tracer.Trace(ctx, tb.provider, string(req), string(resp), err)
	tb.response = nil
}
