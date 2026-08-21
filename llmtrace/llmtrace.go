// Package llmtrace records a human-readable trace of every request sent to
// an LLM provider and what it produced in response. Each Provider builds
// its own request/response dump (see the root llm package's Tracer
// interface) — this package only owns the shared block framing (timestamp,
// provider name, conversation id) and serializing writes to the underlying
// file, so every provider's trace looks the same at a glance no matter
// which SDK produced it, and — critically — reflects exactly what that SDK
// actually sent and received rather than a reconstruction from the
// provider-agnostic llm.ChatRequest, which can't represent provider-specific
// specifics (e.g. Anthropic's own server-side tools).
package llmtrace

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	llm "github.com/archer-developer/miranda-llm"
)

type ctxKey int

const (
	conversationIDKey ctxKey = iota
	tracerKey
)

// WithConversationID attaches a conversation id to ctx so any trace recorded
// further down this call chain can be correlated with the rest of that
// conversation's logs. Absent (e.g. a background call not tied to one
// specific conversation), the trace block just omits it.
func WithConversationID(ctx context.Context, conversationID string) context.Context {
	return context.WithValue(ctx, conversationIDKey, conversationID)
}

func conversationIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(conversationIDKey).(string)
	return id
}

// WithTracer attaches an additional tracer to ctx, scoped to whatever call
// chain uses this ctx from here on — see ContextTracer for why this exists
// (scoping a trace to a single turn/request despite the process-wide tracer
// every Provider is wired to at startup via Router.SetTracer being shared,
// global, and set exactly once).
func WithTracer(ctx context.Context, t llm.Tracer) context.Context {
	return context.WithValue(ctx, tracerKey, t)
}

func tracerFrom(ctx context.Context) llm.Tracer {
	t, _ := ctx.Value(tracerKey).(llm.Tracer)
	return t
}

// Logger writes one formatted block per traced call to an underlying
// io.Writer (typically a rotating log file — see each consuming service's
// own cmd/main.go). A nil *Logger is valid and Trace on it is a no-op, so
// tracing can be wired up as an optional dependency (see
// router.Router.SetTracer and each Provider's own SetTracer).
type Logger struct {
	mu sync.Mutex
	w  io.Writer
}

// New builds a Logger writing to w.
func New(w io.Writer) *Logger {
	return &Logger{w: w}
}

// Trace implements the root llm package's Tracer interface. request and
// response are already fully formed by the calling Provider, verbatim —
// this Logger doesn't interpret their content, just frames and writes them.
// response is empty when err is non-nil. Safe for concurrent use — writes
// are serialized so concurrent turns can't interleave mid-block.
func (l *Logger) Trace(ctx context.Context, provider, request, response string, err error) {
	if l == nil {
		return
	}

	var b strings.Builder
	fmt.Fprintf(&b, "=== %s provider=%s", time.Now().Format(time.RFC3339), provider)
	if convID := conversationIDFrom(ctx); convID != "" {
		fmt.Fprintf(&b, " conversation=%s", convID)
	}
	b.WriteString(" ===\n")

	b.WriteString("--- request ---\n")
	b.WriteString(request)
	b.WriteString("\n")

	b.WriteString("--- response ---\n")
	if err != nil {
		fmt.Fprintf(&b, "error: %v\n", err)
	} else {
		b.WriteString(response)
		b.WriteString("\n")
	}
	b.WriteString("\n")

	l.mu.Lock()
	defer l.mu.Unlock()
	_, _ = io.WriteString(l.w, b.String())
}

// ContextTracer wraps a process-wide Default tracer (typically a *Logger) and
// tees each call to whatever extra tracer, if any, was attached to ctx via
// WithTracer — without ever skipping Default. Install this in place of the
// raw Default at startup (router.Router.SetTracer(&ContextTracer{Default:
// llmtrace.New(w)})) so a caller further down the stack can scope an
// additional tracer to a single turn/request (e.g. to record that turn's
// blocks in memory for anomaly detection — see llmtrace/anomaly) despite
// Router/Provider wiring their one tracer once, globally, at process
// startup, shared by every concurrent turn: swapping Default itself would
// race every other in-flight call, but attaching a per-call tracer to ctx is
// exactly as safe as any other context value.
type ContextTracer struct {
	Default llm.Tracer
}

// Trace implements llm.Tracer. Default always sees every call, unchanged —
// this is a pure addition, never a redirect, so existing behavior (the main
// log file, the live web UI trace view) is unaffected by anything attached
// via WithTracer.
func (c *ContextTracer) Trace(ctx context.Context, provider, request, response string, err error) {
	if c.Default != nil {
		c.Default.Trace(ctx, provider, request, response, err)
	}
	if extra := tracerFrom(ctx); extra != nil {
		extra.Trace(ctx, provider, request, response, err)
	}
}
