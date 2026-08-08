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
)

type ctxKey int

const conversationIDKey ctxKey = iota

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
