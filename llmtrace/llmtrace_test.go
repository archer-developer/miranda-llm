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
