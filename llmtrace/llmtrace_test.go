package llmtrace

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTrace_IncludesProviderRequestAndResponse(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)

	l.Trace(context.Background(), "claude", `{"messages":[{"role":"user","content":"hi"}]}`, `{"content":[{"type":"text","text":"hello"}]}`, nil)

	out := buf.String()
	require.Contains(t, out, "provider=claude")
	require.Contains(t, out, `{"messages":[{"role":"user","content":"hi"}]}`)
	require.Contains(t, out, `{"content":[{"type":"text","text":"hello"}]}`)
}

func TestTrace_IncludesConversationIDFromContext(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)

	ctx := WithConversationID(context.Background(), "conv-123")
	l.Trace(ctx, "local", "req", "resp", nil)

	require.Contains(t, buf.String(), "conversation=conv-123")
}

func TestTrace_OmitsConversationIDWhenAbsent(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)

	l.Trace(context.Background(), "local", "req", "resp", nil)

	require.NotContains(t, buf.String(), "conversation=")
}

func TestTrace_RendersErrorInsteadOfResponse(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)

	l.Trace(context.Background(), "local", "req", "resp-that-should-not-appear", errors.New("boom"))

	out := buf.String()
	require.Contains(t, out, "error: boom")
	require.NotContains(t, out, "resp-that-should-not-appear")
}

func TestTrace_NilLoggerIsANoOp(t *testing.T) {
	var l *Logger
	require.NotPanics(t, func() {
		l.Trace(context.Background(), "local", "req", "resp", nil)
	})
}

func TestTrace_SerializesConcurrentWrites(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)

	done := make(chan struct{})
	for i := 0; i < 20; i++ {
		go func() {
			l.Trace(context.Background(), "local", "req", "resp", nil)
			done <- struct{}{}
		}()
	}
	for i := 0; i < 20; i++ {
		<-done
	}

	// Every block must be fully written (no interleaved/corrupted output):
	// exactly 20 well-formed "=== ..." header lines.
	require.Equal(t, 20, strings.Count(buf.String(), "=== "))
}

// fakeTracer records every Trace call it receives, for asserting
// ContextTracer's tee behavior below.
type fakeTracer struct {
	calls int
}

func (f *fakeTracer) Trace(ctx context.Context, provider, request, response string, err error) {
	f.calls++
}

func TestContextTracer_AlwaysCallsDefault(t *testing.T) {
	var buf bytes.Buffer
	ct := &ContextTracer{Default: New(&buf)}

	ct.Trace(context.Background(), "local", "req", "resp", nil)

	require.Contains(t, buf.String(), "provider=local")
}

func TestContextTracer_TeesToTracerAttachedViaContext(t *testing.T) {
	var buf bytes.Buffer
	extra := &fakeTracer{}
	ct := &ContextTracer{Default: New(&buf)}

	ctx := WithTracer(context.Background(), extra)
	ct.Trace(ctx, "local", "req", "resp", nil)

	require.Contains(t, buf.String(), "provider=local", "Default must still receive the call")
	require.Equal(t, 1, extra.calls, "the ctx-attached tracer must also receive the call")
}

func TestContextTracer_NoExtraTracerIsJustDefault(t *testing.T) {
	var buf bytes.Buffer
	ct := &ContextTracer{Default: New(&buf)}

	// No WithTracer in this ctx — must not panic, must behave exactly like
	// calling Default directly.
	require.NotPanics(t, func() {
		ct.Trace(context.Background(), "local", "req", "resp", nil)
	})
	require.Contains(t, buf.String(), "provider=local")
}

func TestContextTracer_UnrelatedContextValuesDontLeakAcrossTurns(t *testing.T) {
	var buf bytes.Buffer
	extraA := &fakeTracer{}
	extraB := &fakeTracer{}
	ct := &ContextTracer{Default: New(&buf)}

	ctxA := WithTracer(context.Background(), extraA)
	ctxB := WithTracer(context.Background(), extraB)

	ct.Trace(ctxA, "local", "req", "resp", nil)
	ct.Trace(ctxB, "local", "req", "resp", nil)

	require.Equal(t, 1, extraA.calls)
	require.Equal(t, 1, extraB.calls)
}
