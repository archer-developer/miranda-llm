package router

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	llm "github.com/archer-developer/miranda-llm"
	"github.com/archer-developer/miranda-llm/llmtest"
	"github.com/archer-developer/miranda-llm/llmtrace"
)

func drainText(t *testing.T, ch <-chan llm.StreamChunk) string {
	t.Helper()
	var text string
	for chunk := range ch {
		require.NoError(t, chunk.Err)
		text += chunk.TextDelta
	}
	return text
}

func noEscalations() map[string]EscalationConfig {
	return nil
}

func TestRouter_UsesFirstHealthyProvider(t *testing.T) {
	primary := llmtest.New("local", llmtest.Response{Text: "hi from local"})
	r, err := New([]llm.Provider{primary}, noEscalations(), "")
	require.NoError(t, err)

	var used string
	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, func(name string) { used = name })
	require.NoError(t, err)
	require.Equal(t, "hi from local", drainText(t, ch))
	require.Equal(t, "local", used)
}

func TestRouter_FallsBackOnConnectionError(t *testing.T) {
	broken := llmtest.New("broken", llmtest.Response{Err: errors.New("connection refused")})
	backup := llmtest.New("backup", llmtest.Response{Text: "hi from backup"})
	r, err := New([]llm.Provider{broken, backup}, noEscalations(), "")
	require.NoError(t, err)

	var used string
	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, func(name string) { used = name })
	require.NoError(t, err)
	require.Equal(t, "hi from backup", drainText(t, ch))
	require.Equal(t, "backup", used)
}

func TestRouter_AllProvidersFailReturnsError(t *testing.T) {
	broken1 := llmtest.New("broken1", llmtest.Response{Err: errors.New("down")})
	broken2 := llmtest.New("broken2", llmtest.Response{Err: errors.New("down too")})
	r, err := New([]llm.Provider{broken1, broken2}, noEscalations(), "")
	require.NoError(t, err)

	_, err = r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.Error(t, err)
}

func TestRouter_DefaultProviderMovedToFront(t *testing.T) {
	a := llmtest.New("a", llmtest.Response{Text: "from a"})
	b := llmtest.New("b", llmtest.Response{Text: "from b"})
	c := llmtest.New("c", llmtest.Response{Text: "from c"})
	r, err := New([]llm.Provider{a, b, c}, noEscalations(), "c")
	require.NoError(t, err)

	var used string
	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, func(name string) { used = name })
	require.NoError(t, err)
	require.Equal(t, "from c", drainText(t, ch))
	require.Equal(t, "c", used)
}

func TestRouter_UnknownDefaultProviderErrors(t *testing.T) {
	a := llmtest.New("a", llmtest.Response{Text: "from a"})
	_, err := New([]llm.Provider{a}, noEscalations(), "does-not-exist")
	require.Error(t, err)
}

func TestRouter_EscalatesToTargetProviderOnToolCall(t *testing.T) {
	escalations := map[string]EscalationConfig{
		"local-qwen": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}

	local := llmtest.New("local-qwen", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-1", Name: "escalate_to_claude", Arguments: `{"reason":"too hard"}`},
	})
	claude := llmtest.New("claude", llmtest.Response{Text: "the sophisticated answer"})

	r, err := New([]llm.Provider{local, claude}, escalations, "")
	require.NoError(t, err)

	var used string
	ch, err := r.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "a hard question"}},
	}, func(name string) { used = name })
	require.NoError(t, err)
	require.Equal(t, "the sophisticated answer", drainText(t, ch))
	require.Equal(t, "claude", used)

	// The target provider must have received the original user turn plus the
	// escalation tool call/result so it has full context.
	require.Len(t, claude.Requests, 1)
	msgs := claude.Requests[0].Messages
	require.Len(t, msgs, 3)
	require.Equal(t, llm.RoleUser, msgs[0].Role)
	require.Equal(t, llm.RoleAssistant, msgs[1].Role)
	require.Equal(t, "escalate_to_claude", msgs[1].ToolCalls[0].Name)
	require.Equal(t, llm.RoleTool, msgs[2].Role)
	require.Equal(t, "call-1", msgs[2].ToolCallID)
}

func TestRouter_EscalatesToTargetProviderOnStreamError(t *testing.T) {
	// Models a real provider's actual failure shape (see llmtest.Response's
	// StreamErr doc comment) — e.g. a Gemini provider whose keyrotation
	// loop exhausted every key across every retry cycle — as opposed to
	// TestRouter_FallsBackOnConnectionError's synchronous Chat error, which
	// no real provider implementation ever produces.
	escalations := map[string]EscalationConfig{
		"gemini-strong": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}

	quotaExhausted := errors.New("gemini: all 3 configured key(s) exhausted across 3 cycle(s): RESOURCE_EXHAUSTED")
	strong := llmtest.New("gemini-strong", llmtest.Response{StreamErr: quotaExhausted})
	claude := llmtest.New("claude", llmtest.Response{Text: "BTC is around $60k"})

	r, err := New([]llm.Provider{strong, claude}, escalations, "")
	require.NoError(t, err)

	var used string
	ch, err := r.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "what's the price of BTC"}},
	}, func(name string) { used = name })
	require.NoError(t, err)
	require.Equal(t, "BTC is around $60k", drainText(t, ch))
	require.Equal(t, "claude", used)

	// The target still sees the original turn plus a synthetic
	// tool-call/tool-result pair, same as an explicit escalation, so it has
	// full context despite the handoff not coming from a real tool call.
	require.Len(t, claude.Requests, 1)
	msgs := claude.Requests[0].Messages
	require.Len(t, msgs, 3)
	require.Equal(t, llm.RoleAssistant, msgs[1].Role)
	require.Equal(t, "escalate_to_claude", msgs[1].ToolCalls[0].Name)
}

func TestRouter_StreamErrorEscalationArgumentsAreValidJSONEvenWithUnsafeErrorText(t *testing.T) {
	// Regression test: a real Gemini quota error embeds literal newlines
	// (and often quotes) in its message text, e.g.
	// "...billing details. \n* Quota exceeded...". escalateOnError used to
	// Sprintf that text straight into a hand-built JSON literal, producing
	// invalid JSON for the synthetic tool call's Arguments — which then
	// silently became a nil/missing "input" once converted to a target
	// provider's wire format (see anthropic.toAnthropicMessages), and
	// Anthropic's API rejects a tool_use block with no "input" outright.
	escalations := map[string]EscalationConfig{
		"gemini-strong": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}

	unsafeErr := errors.New("quota exceeded, please check your plan.\n* Retry in \"37s\".")
	strong := llmtest.New("gemini-strong", llmtest.Response{StreamErr: unsafeErr})
	claude := llmtest.New("claude", llmtest.Response{Text: "ok"})

	r, err := New([]llm.Provider{strong, claude}, escalations, "")
	require.NoError(t, err)

	ch, err := r.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}, nil)
	require.NoError(t, err)
	require.Equal(t, "ok", drainText(t, ch))

	require.Len(t, claude.Requests, 1)
	args := claude.Requests[0].Messages[1].ToolCalls[0].Arguments
	var decoded map[string]string
	require.NoError(t, json.Unmarshal([]byte(args), &decoded), "escalation Arguments must be valid JSON, got: %s", args)
	require.Contains(t, decoded["reason"], "quota exceeded")
}

func TestRouter_StreamErrorEscalationDoesNotReescalateToVisitedProvider(t *testing.T) {
	// gemini-strong is reached via an explicit tool-call escalation from
	// gemini-lite, then itself hard-fails. Its escalation target is
	// gemini-lite again (a misconfiguration, but one that must not be
	// masked as a generic "cycle detected" error) — since gemini-lite is
	// already in this turn's visited set, the original stream error must
	// surface rather than bouncing back.
	escalations := map[string]EscalationConfig{
		"gemini-lite":   {Enabled: true, ToolName: "escalate_to_strong", TargetProvider: "gemini-strong"},
		"gemini-strong": {Enabled: true, ToolName: "escalate_to_lite", TargetProvider: "gemini-lite"},
	}

	lite := llmtest.New("gemini-lite", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-1", Name: "escalate_to_strong"},
	})
	quotaExhausted := errors.New("quota exhausted")
	strong := llmtest.New("gemini-strong", llmtest.Response{StreamErr: quotaExhausted})

	r, err := New([]llm.Provider{lite, strong}, escalations, "")
	require.NoError(t, err)

	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)

	var gotErr error
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = chunk.Err
		}
	}
	require.ErrorIs(t, gotErr, quotaExhausted)
}

func TestRouter_ChainedEscalation_TwoHops(t *testing.T) {
	escalations := map[string]EscalationConfig{
		"lite":   {Enabled: true, ToolName: "escalate_to_strong", TargetProvider: "strong"},
		"strong": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}

	lite := llmtest.New("lite", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-1", Name: "escalate_to_strong"},
	})
	strong := llmtest.New("strong", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-2", Name: "escalate_to_claude"},
	})
	claude := llmtest.New("claude", llmtest.Response{Text: "the final answer"})

	r, err := New([]llm.Provider{lite, strong, claude}, escalations, "")
	require.NoError(t, err)

	var used string
	callCount := 0
	ch, err := r.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "a very hard question"}},
	}, func(name string) { used = name; callCount++ })
	require.NoError(t, err)
	require.Equal(t, "the final answer", drainText(t, ch))
	require.Equal(t, "claude", used)
	require.Equal(t, 1, callCount, "onProviderUsed must fire exactly once across the whole chain")

	require.Len(t, strong.Requests, 1)
	require.Len(t, claude.Requests, 1)
}

func TestRouter_EscalationCycleDetected(t *testing.T) {
	escalations := map[string]EscalationConfig{
		"a": {Enabled: true, ToolName: "escalate_to_b", TargetProvider: "b"},
		"b": {Enabled: true, ToolName: "escalate_to_a", TargetProvider: "a"},
	}

	a := llmtest.New("a", llmtest.Response{ToolCall: &llm.ToolCall{ID: "call-1", Name: "escalate_to_b"}})
	b := llmtest.New("b", llmtest.Response{ToolCall: &llm.ToolCall{ID: "call-2", Name: "escalate_to_a"}})

	r, err := New([]llm.Provider{a, b}, escalations, "")
	require.NoError(t, err)

	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)

	var gotErr error
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = chunk.Err
		}
	}
	require.Error(t, gotErr)
	require.Contains(t, gotErr.Error(), "cycle")
}

func TestRouter_EscalationHopCapEnforced(t *testing.T) {
	// A straight-line chain of 8 providers, each escalating to the next —
	// exceeds maxEscalationHops (6) without ever revisiting a provider, so
	// this exercises the hop cap specifically, not the cycle guard.
	escalations := map[string]EscalationConfig{}
	var providers []llm.Provider
	for i := 0; i < 8; i++ {
		name := providerName(i)
		if i < 7 {
			escalations[name] = EscalationConfig{Enabled: true, ToolName: "escalate", TargetProvider: providerName(i + 1)}
			providers = append(providers, llmtest.New(name, llmtest.Response{ToolCall: &llm.ToolCall{ID: name + "-call", Name: "escalate"}}))
		} else {
			providers = append(providers, llmtest.New(name, llmtest.Response{Text: "unreachable"}))
		}
	}

	r, err := New(providers, escalations, "")
	require.NoError(t, err)

	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)

	var gotErr error
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = chunk.Err
		}
	}
	require.Error(t, gotErr)
	require.Contains(t, gotErr.Error(), "hops")
}

func providerName(i int) string {
	return string(rune('a' + i))
}

func TestRouter_EachHopSeesOnlyItsOwnEscalationTool(t *testing.T) {
	escalations := map[string]EscalationConfig{
		"lite":   {Enabled: true, ToolName: "escalate_to_strong", TargetProvider: "strong"},
		"strong": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}

	lite := llmtest.New("lite", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-1", Name: "escalate_to_strong"},
	})
	strong := llmtest.New("strong", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-2", Name: "escalate_to_claude"},
	})
	claude := llmtest.New("claude", llmtest.Response{Text: "done"})

	r, err := New([]llm.Provider{lite, strong, claude}, escalations, "")
	require.NoError(t, err)

	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)
	drainText(t, ch)

	require.Len(t, lite.Requests, 1)
	liteTools := toolNames(lite.Requests[0].Tools)
	require.Contains(t, liteTools, "escalate_to_strong")
	require.NotContains(t, liteTools, "escalate_to_claude")

	require.Len(t, strong.Requests, 1)
	strongTools := toolNames(strong.Requests[0].Tools)
	require.Contains(t, strongTools, "escalate_to_claude")
	require.NotContains(t, strongTools, "escalate_to_strong")
}

func TestRouter_EscalationToolUsesCustomDescriptionWhenSet(t *testing.T) {
	customDesc := "escalate for anything requiring code execution, which you don't have"
	escalations := map[string]EscalationConfig{
		"gemini-lite": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude", Description: customDesc},
		"generic":     {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}

	geminiLite := llmtest.New("gemini-lite", llmtest.Response{Text: "hi"})
	generic := llmtest.New("generic", llmtest.Response{Text: "hi"})
	claude := llmtest.New("claude", llmtest.Response{Text: "hi"})

	r, err := New([]llm.Provider{geminiLite, claude}, escalations, "")
	require.NoError(t, err)
	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)
	drainText(t, ch)
	require.Len(t, geminiLite.Requests, 1)
	require.Equal(t, customDesc, geminiLite.Requests[0].Tools[0].Description)

	r2, err := New([]llm.Provider{generic, claude}, escalations, "")
	require.NoError(t, err)
	ch2, err := r2.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)
	drainText(t, ch2)
	require.Len(t, generic.Requests, 1)
	require.Equal(t, defaultEscalationDescription, generic.Requests[0].Tools[0].Description)
}

func toolNames(tools []llm.ToolDef) []string {
	var names []string
	for _, t := range tools {
		names = append(names, t.Name)
	}
	return names
}

func TestRouter_EscalationTargetNotConfiguredReturnsErrChunk(t *testing.T) {
	escalations := map[string]EscalationConfig{
		"local-qwen": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}
	local := llmtest.New("local-qwen", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-1", Name: "escalate_to_claude"},
	})
	r, err := New([]llm.Provider{local}, escalations, "")
	require.NoError(t, err)

	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)

	var gotErr error
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = chunk.Err
		}
	}
	require.Error(t, gotErr)
}

func TestRouter_ChatPinnedStartsAtPinnedProviderNotDefaultOrder(t *testing.T) {
	// Models continuing a multi-turn interaction after an earlier iteration
	// escalated: once the router reports "strong" as the provider that
	// answered, a later ChatPinned call in the same turn must go straight
	// to "strong" again, never back to "lite" — lite's single scripted
	// response would panic FakeProvider if it were called a second time.
	lite := llmtest.New("lite", llmtest.Response{Text: "should never be called again"})
	strong := llmtest.New("strong", llmtest.Response{Text: "still the strong model"})
	r, err := New([]llm.Provider{lite, strong}, noEscalations(), "")
	require.NoError(t, err)

	var used string
	ch, err := r.ChatPinned(context.Background(), llm.ChatRequest{}, "strong", func(name string) { used = name })
	require.NoError(t, err)
	require.Equal(t, "still the strong model", drainText(t, ch))
	require.Equal(t, "strong", used)
	require.Len(t, lite.Requests, 0)
	require.Len(t, strong.Requests, 1)
}

func TestRouter_ChatPinnedEmptyProviderBehavesLikeChat(t *testing.T) {
	a := llmtest.New("a", llmtest.Response{Text: "from a"})
	b := llmtest.New("b", llmtest.Response{Text: "from b"})
	r, err := New([]llm.Provider{a, b}, noEscalations(), "")
	require.NoError(t, err)

	var used string
	ch, err := r.ChatPinned(context.Background(), llm.ChatRequest{}, "", func(name string) { used = name })
	require.NoError(t, err)
	require.Equal(t, "from a", drainText(t, ch))
	require.Equal(t, "a", used)
}

func TestRouter_TracesRequestAndResponseWhenTracerSet(t *testing.T) {
	provider := llmtest.New("local", llmtest.Response{Text: "hi"})
	r, err := New([]llm.Provider{provider}, noEscalations(), "")
	require.NoError(t, err)

	var buf bytes.Buffer
	r.SetTracer(llmtrace.New(&buf))

	ch, err := r.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
	}, nil)
	require.NoError(t, err)
	require.Equal(t, "hi", drainText(t, ch))

	out := buf.String()
	require.Contains(t, out, "provider=local")
	require.Contains(t, out, "hello")
	require.Contains(t, out, "hi")
}

func TestRouter_TracesEscalationAsTwoBlocks(t *testing.T) {
	escalations := map[string]EscalationConfig{
		"local-qwen": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	}
	local := llmtest.New("local-qwen", llmtest.Response{
		ToolCall: &llm.ToolCall{ID: "call-1", Name: "escalate_to_claude", Arguments: `{"reason":"hard"}`},
	})
	claude := llmtest.New("claude", llmtest.Response{Text: "the answer"})
	r, err := New([]llm.Provider{local, claude}, escalations, "")
	require.NoError(t, err)

	var buf bytes.Buffer
	r.SetTracer(llmtrace.New(&buf))

	ch, err := r.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "a hard question"}},
	}, nil)
	require.NoError(t, err)
	require.Equal(t, "the answer", drainText(t, ch))

	out := buf.String()
	require.Contains(t, out, "provider=local-qwen")
	require.Contains(t, out, "escalate_to_claude")
	require.Contains(t, out, "provider=claude")
	require.Contains(t, out, "the answer")
}

func TestRouter_NoTracerSetIsFine(t *testing.T) {
	provider := llmtest.New("local", llmtest.Response{Text: "hi"})
	r, err := New([]llm.Provider{provider}, noEscalations(), "")
	require.NoError(t, err)

	ch, err := r.Chat(context.Background(), llm.ChatRequest{}, nil)
	require.NoError(t, err)
	require.Equal(t, "hi", drainText(t, ch))
}
