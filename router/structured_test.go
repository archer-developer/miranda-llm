package router

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	llm "github.com/archer-developer/miranda-llm"
	"github.com/archer-developer/miranda-llm/llmtest"
)

func TestRouter_Structured_UsesFirstProviderThatSucceeds(t *testing.T) {
	primary := llmtest.New("primary").WithStructured(llmtest.StructuredResponse{JSON: json.RawMessage(`{"ok":true}`)})
	r, err := New([]llm.Provider{primary}, noEscalations(), "")
	require.NoError(t, err)

	out, err := r.Structured(context.Background(), llm.StructuredRequest{SchemaName: "test"})
	require.NoError(t, err)
	require.JSONEq(t, `{"ok":true}`, string(out))
}

func TestRouter_Structured_FallsBackOnError(t *testing.T) {
	broken := llmtest.New("broken").WithStructured(llmtest.StructuredResponse{Err: errors.New("rate limited")})
	backup := llmtest.New("backup").WithStructured(llmtest.StructuredResponse{JSON: json.RawMessage(`{"from":"backup"}`)})
	r, err := New([]llm.Provider{broken, backup}, noEscalations(), "")
	require.NoError(t, err)

	out, err := r.Structured(context.Background(), llm.StructuredRequest{SchemaName: "test"})
	require.NoError(t, err)
	require.JSONEq(t, `{"from":"backup"}`, string(out))
}

func TestRouter_Structured_SkipsProviderThatDoesNotImplementIt(t *testing.T) {
	// chatOnly is a valid llm.Provider (usable for Chat) but does not
	// implement llm.StructuredProvider — Structured must skip it rather
	// than treating the missing capability as a hard failure, since a
	// mixed fallback chain (e.g. a structured-capable primary with a
	// chat-only backup) is a normal configuration for Chat even though
	// Structured can't use every entry.
	chatOnly := llmtest.NewChatOnly("chat-only")
	structuredCapable := llmtest.New("structured-capable").WithStructured(llmtest.StructuredResponse{JSON: json.RawMessage(`{"ok":true}`)})

	r, err := New([]llm.Provider{chatOnly, structuredCapable}, noEscalations(), "")
	require.NoError(t, err)

	out, err := r.Structured(context.Background(), llm.StructuredRequest{SchemaName: "test"})
	require.NoError(t, err)
	require.JSONEq(t, `{"ok":true}`, string(out))
}

func TestRouter_Structured_AllProvidersFailReturnsError(t *testing.T) {
	broken1 := llmtest.New("broken1").WithStructured(llmtest.StructuredResponse{Err: errors.New("down")})
	broken2 := llmtest.New("broken2").WithStructured(llmtest.StructuredResponse{Err: errors.New("also down")})
	r, err := New([]llm.Provider{broken1, broken2}, noEscalations(), "")
	require.NoError(t, err)

	_, err = r.Structured(context.Background(), llm.StructuredRequest{SchemaName: "test"})
	require.Error(t, err)
}

func TestRouter_Structured_NoProviderImplementsItReturnsClearError(t *testing.T) {
	chatOnly := llmtest.NewChatOnly("chat-only")
	r, err := New([]llm.Provider{chatOnly}, noEscalations(), "")
	require.NoError(t, err)

	_, err = r.Structured(context.Background(), llm.StructuredRequest{SchemaName: "test"})
	require.Error(t, err)
	require.Contains(t, err.Error(), "no configured provider implements structured output")
}
