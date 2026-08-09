package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"log/slog"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	llm "github.com/archer-developer/miranda-llm"
)

// This file covers the pure-function logic (isRetryable, message/tool-call
// conversion) that doesn't require a live or mocked Gemini HTTP endpoint.
// It deliberately does not port miranda's original httptest-server-based
// end-to-end streaming test suite for Chat/attempt/pump — see this repo's
// CLAUDE.md "Known gaps versus the original" section.

func TestIsRetryable(t *testing.T) {
	require.True(t, isRetryable(genai.APIError{Code: http.StatusTooManyRequests}))
	require.True(t, isRetryable(genai.APIError{Code: http.StatusUnauthorized}))
	require.True(t, isRetryable(genai.APIError{Code: http.StatusForbidden}))
	require.True(t, isRetryable(genai.APIError{Status: "RESOURCE_EXHAUSTED"}))
	require.True(t, isRetryable(genai.APIError{Code: http.StatusInternalServerError}))
	require.True(t, isRetryable(genai.APIError{Code: http.StatusServiceUnavailable}))

	require.False(t, isRetryable(genai.APIError{Code: http.StatusBadRequest}))
	require.False(t, isRetryable(errors.New("some unrelated network error")))
}

func TestToLLMToolCall_SynthesizesIDWhenMissing(t *testing.T) {
	fc := &genai.FunctionCall{Name: "search", Args: map[string]any{"q": "aspirin"}}
	tc := toLLMToolCall(fc, nil, 2)
	require.Equal(t, "search-2", tc.ID)
	require.Equal(t, "search", tc.Name)
	require.JSONEq(t, `{"q":"aspirin"}`, tc.Arguments)
	require.Empty(t, tc.ProviderMetadata)
}

func TestToLLMToolCall_PreservesExplicitID(t *testing.T) {
	fc := &genai.FunctionCall{ID: "call-abc", Name: "search"}
	tc := toLLMToolCall(fc, nil, 0)
	require.Equal(t, "call-abc", tc.ID)
}

func TestToLLMToolCall_EncodesThoughtSignature(t *testing.T) {
	fc := &genai.FunctionCall{ID: "call-1", Name: "search"}
	tc := toLLMToolCall(fc, []byte("signature-bytes"), 0)
	decoded, err := base64.StdEncoding.DecodeString(tc.ProviderMetadata)
	require.NoError(t, err)
	require.Equal(t, "signature-bytes", string(decoded))
}

func TestToGeminiContents_SplitsSystemFromTurns(t *testing.T) {
	system, contents := toGeminiContents([]llm.Message{
		{Role: llm.RoleSystem, Content: "you are a helpful assistant"},
		{Role: llm.RoleUser, Content: "hello"},
	})
	require.NotNil(t, system)
	require.Equal(t, "you are a helpful assistant", system.Parts[0].Text)
	require.Len(t, contents, 1)
	require.Equal(t, genai.RoleUser, contents[0].Role)
	require.Equal(t, "hello", contents[0].Parts[0].Text)
}

func TestToGeminiContents_ToolResultRecoversNameFromPrecedingCall(t *testing.T) {
	_, contents := toGeminiContents([]llm.Message{
		{Role: llm.RoleUser, Content: "what's the weather"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "get_weather"}}},
		{Role: llm.RoleTool, ToolCallID: "call-1", Content: "sunny"},
	})
	require.Len(t, contents, 3)
	toolContent := contents[2]
	require.Equal(t, "get_weather", toolContent.Parts[0].FunctionResponse.Name)
	require.Equal(t, "call-1", toolContent.Parts[0].FunctionResponse.ID)
}

func TestToGeminiContents_ReplayedToolCallEchoesThoughtSignature(t *testing.T) {
	sig := base64.StdEncoding.EncodeToString([]byte("real-signature"))
	_, contents := toGeminiContents([]llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "search", ProviderMetadata: sig}}},
	})
	require.Len(t, contents, 1)
	require.Equal(t, []byte("real-signature"), contents[0].Parts[0].ThoughtSignature)
}

func TestToGeminiContents_MissingThoughtSignatureFallsBackToPlaceholder(t *testing.T) {
	_, contents := toGeminiContents([]llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "search"}}},
	})
	require.Equal(t, geminiThoughtSignaturePlaceholder, contents[0].Parts[0].ThoughtSignature)
}

func TestBuildTools_CustomToolsAndGoogleSearchAreSeparateEntries(t *testing.T) {
	p := &Provider{tools: ToolsConfig{GoogleSearch: true}}
	tools := p.buildTools([]llm.ToolDef{{Name: "search_docs", Parameters: map[string]any{}}})
	require.Len(t, tools, 2, "custom function declarations and GoogleSearch must be separate *genai.Tool entries")
	require.Len(t, tools[0].FunctionDeclarations, 1)
	require.Equal(t, "search_docs", tools[0].FunctionDeclarations[0].Name)
	require.NotNil(t, tools[1].GoogleSearch)
}

func TestNew_RejectsContextCaching(t *testing.T) {
	_, err := New(context.Background(), "test", "gemini-3.0", []string{"UNSET_ENV"}, ToolsConfig{ContextCaching: true}, RotationConfig{}, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "context_caching")
}

func TestNew_FailsFastWhenNoAPIKeyEnvResolves(t *testing.T) {
	t.Setenv("MIRANDA_LLM_TEST_UNSET_KEY", "")
	_, err := New(context.Background(), "test", "gemini-3.0", []string{"MIRANDA_LLM_TEST_UNSET_KEY"}, ToolsConfig{}, RotationConfig{}, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "none of the configured api_key_envs")
}

// TestLogStructuredFinish_AbnormalFinishReasonLogsWarn covers why
// logStructuredFinish exists: a Structured call that Gemini silently
// curtailed for safety reasons returns a normal, error-free response with
// an empty result — identical, from the caller's side, to the model
// genuinely finding nothing to extract. The only place that distinction is
// visible at all is the response's own FinishReason, which Structured
// previously never looked at.
func TestLogStructuredFinish_AbnormalFinishReasonLogsWarn(t *testing.T) {
	var buf bytes.Buffer
	p := &Provider{name: "test-provider", logger: slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))}

	p.logStructuredFinish(&genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{FinishReason: genai.FinishReasonSafety}},
	})

	require.Contains(t, buf.String(), "level=WARN")
	require.Contains(t, buf.String(), "finished abnormally")
	require.Contains(t, buf.String(), "SAFETY")
}

func TestLogStructuredFinish_NormalStopFinishLogsDebugOnly(t *testing.T) {
	var buf bytes.Buffer
	p := &Provider{name: "test-provider", logger: slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))}

	p.logStructuredFinish(&genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{FinishReason: genai.FinishReasonStop}},
	})

	require.Contains(t, buf.String(), "level=DEBUG")
	require.NotContains(t, buf.String(), "level=WARN")
}

func TestLogStructuredFinish_PromptLevelBlockLogsWarn(t *testing.T) {
	var buf bytes.Buffer
	p := &Provider{name: "test-provider", logger: slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))}

	p.logStructuredFinish(&genai.GenerateContentResponse{
		PromptFeedback: &genai.GenerateContentResponsePromptFeedback{BlockReason: genai.BlockedReasonSafety},
	})

	require.Contains(t, buf.String(), "level=WARN")
	require.Contains(t, buf.String(), "blocked at prompt level")
}
