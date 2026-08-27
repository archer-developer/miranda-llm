package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	llm "github.com/archer-developer/miranda-llm"
)

func TestIsRetryable(t *testing.T) {
	require.True(t, isRetryable(genai.APIError{Code: http.StatusTooManyRequests}))
	require.True(t, isRetryable(genai.APIError{Code: http.StatusUnauthorized}))
	require.True(t, isRetryable(genai.APIError{Code: http.StatusForbidden}))
	require.True(t, isRetryable(genai.APIError{Status: "RESOURCE_EXHAUSTED"}))

	// 5xx is deliberately NOT retryable: every configured key hits the same
	// overloaded backend, so rotating keys just repeats the identical
	// failure instead of helping — see isRetryable's doc comment.
	require.False(t, isRetryable(genai.APIError{Code: http.StatusInternalServerError}))
	require.False(t, isRetryable(genai.APIError{Code: http.StatusServiceUnavailable}))

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

// --- Below: the httptest.Server-backed end-to-end streaming suite, ported
// verbatim (behavior-wise) from miranda's original internal/llm/gemini
// package at extraction time — see this file's package doc reference in
// CLAUDE.md. Exercises Chat/pump/attempt against a fake HTTP server
// standing in for the real Gemini API, via the package's apiBaseURL
// override.

// scriptedResponse is one fake server reply, in the order Chat's rotation
// loop will request them: either a successful SSE stream (events) or an
// HTTP error status with a Gemini-shaped {"error": {...}} body.
type scriptedResponse struct {
	events     []string
	statusCode int
	errBody    string
}

// sseEvent formats one data-only SSE event exactly as
// google.golang.org/genai's stream scanner expects: a "data: <json>" line
// followed by a blank line — its scan function splits tokens on "\n\n" (see
// that package's api_client.go:scan), not on a single newline.
func sseEvent(t *testing.T, payload map[string]any) string {
	t.Helper()
	b, err := json.Marshal(payload)
	require.NoError(t, err)
	return "data: " + string(b) + "\n\n"
}

func textEvent(t *testing.T, text string) string {
	t.Helper()
	return sseEvent(t, map[string]any{
		"candidates": []map[string]any{
			{"content": map[string]any{"role": "model", "parts": []map[string]any{{"text": text}}}},
		},
	})
}

func functionCallEvent(t *testing.T, name string, args map[string]any) string {
	t.Helper()
	return sseEvent(t, map[string]any{
		"candidates": []map[string]any{
			{"content": map[string]any{"role": "model", "parts": []map[string]any{
				{"functionCall": map[string]any{"name": name, "args": args}},
			}}},
		},
	})
}

// functionCallEventWithThoughtSignature mirrors a real Gemini
// generateContent response's shape for a function call, including
// thoughtSignature as a sibling field on the same Part — confirmed against
// a real API response during manual verification, not just SDK docs (see
// toLLMToolCall/toGeminiContents's doc comments for why this matters: a
// dropped signature on a same-provider multi-turn replay is a hard 400,
// not just degraded quality).
func functionCallEventWithThoughtSignature(t *testing.T, name string, args map[string]any, signatureB64 string) string {
	t.Helper()
	return sseEvent(t, map[string]any{
		"candidates": []map[string]any{
			{"content": map[string]any{"role": "model", "parts": []map[string]any{
				{"functionCall": map[string]any{"name": name, "args": args}, "thoughtSignature": signatureB64},
			}}},
		},
	})
}

// midStreamErrorEvent mirrors how google.golang.org/genai's stream parser
// (iterateResponseStream) treats any two-newline-terminated SSE chunk that
// ISN'T prefixed with "data:" — it unmarshals the raw chunk as a Gemini
// error object ({"error": {...}}) and yields that as the stream's error.
// This is how a real 429/5xx can arrive mid-stream, after some content has
// already been sent, rather than only as an HTTP-level status before any
// content — exercised by TestGemini_DoesNotRetryAfterPartialForward.
func midStreamErrorEvent(t *testing.T, code int, status, message string) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{"error": map[string]any{"code": code, "status": status, "message": message}})
	require.NoError(t, err)
	return string(b) + "\n\n"
}

// quotaErrorBody is a Gemini generateContent 429 response body.
const quotaErrorBody = `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`

// serverErrorBody is a generic Gemini 5xx response body.
const serverErrorBody = `{"error":{"code":503,"message":"backend overloaded","status":"UNAVAILABLE"}}`

// badRequestBody is a non-retryable Gemini 400 response body.
const badRequestBody = `{"error":{"code":400,"message":"invalid argument","status":"INVALID_ARGUMENT"}}`

// authErrorBody is a Gemini 401 response body — an invalid/expired key,
// specific to that key rather than the request.
const authErrorBody = `{"error":{"code":401,"message":"API key not valid","status":"UNAUTHENTICATED"}}`

// forbiddenErrorBody is a Gemini 403 response body — a revoked/disabled
// key or an insufficient-permission key, also specific to that key.
const forbiddenErrorBody = `{"error":{"code":403,"message":"permission denied","status":"PERMISSION_DENIED"}}`

// newTestServer serves script's entries in order, one per HTTP request
// received, and points package-level apiBaseURL at it for the duration of
// the test (restored via t.Cleanup). Returns a function that reports how
// many requests have been received so far.
func newTestServer(t *testing.T, script []scriptedResponse) func() int {
	t.Helper()
	requestCount := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := requestCount
		requestCount++
		if idx >= len(script) {
			t.Fatalf("newTestServer: got more requests (%d) than scripted responses (%d)", idx+1, len(script))
		}
		resp := script[idx]
		if len(resp.events) == 0 {
			w.WriteHeader(resp.statusCode)
			_, _ = w.Write([]byte(resp.errBody))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, e := range resp.events {
			_, _ = w.Write([]byte(e))
		}
	}))
	t.Cleanup(ts.Close)

	original := apiBaseURL
	apiBaseURL = ts.URL
	t.Cleanup(func() { apiBaseURL = original })

	return func() int { return requestCount }
}

// newTestProvider builds a Provider with n fake keys (env vars set via
// t.Setenv, auto-restored) pointed at a test server serving script.
func newTestProvider(t *testing.T, n int, script []scriptedResponse, rotation RotationConfig) (*Provider, func() int) {
	t.Helper()
	requestCount := newTestServer(t, script)

	var envs []string
	for i := 0; i < n; i++ {
		env := t.Name() + "_KEY_" + string(rune('A'+i))
		t.Setenv(env, "fake-key-"+string(rune('A'+i)))
		envs = append(envs, env)
	}

	p, err := New(context.Background(), "gemini-test", "gemini-test-model", envs, ToolsConfig{}, rotation, nil)
	require.NoError(t, err)
	return p, requestCount
}

// collect drains ch into its Text (concatenated TextDelta chunks) and
// ToolCalls, failing the test on any Err chunk.
type collected struct {
	Text      string
	ToolCalls []llm.ToolCall
}

func collect(t *testing.T, ch <-chan llm.StreamChunk) collected {
	t.Helper()
	var out collected
	for chunk := range ch {
		require.NoError(t, chunk.Err)
		out.Text += chunk.TextDelta
		if chunk.ToolCall != nil {
			out.ToolCalls = append(out.ToolCalls, *chunk.ToolCall)
		}
	}
	return out
}

// collectErr drains ch and returns the first Err chunk's error, or nil if
// the stream finished cleanly.
func collectErr(t *testing.T, ch <-chan llm.StreamChunk) error {
	t.Helper()
	var gotErr error
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = chunk.Err
		}
	}
	return gotErr
}

func TestGeminiNew_NoKeysConfiguredErrors(t *testing.T) {
	_, err := New(context.Background(), "gemini-test", "gemini-test-model", []string{"GEMINI_TEST_UNSET_ENV_VAR"}, ToolsConfig{}, RotationConfig{}, nil)
	require.Error(t, err)
}

func TestGeminiNew_ContextCachingRejected(t *testing.T) {
	t.Setenv("GEMINI_TEST_KEY", "fake-key")
	_, err := New(context.Background(), "gemini-test", "gemini-test-model", []string{"GEMINI_TEST_KEY"}, ToolsConfig{ContextCaching: true}, RotationConfig{}, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "context_caching")
}

func TestGemini_RotatesOnQuotaError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{events: []string{textEvent(t, "hi from key 2")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount(), "exactly one request per key, no needless extra cycling")
}

// TestGemini_ServerErrorFailsImmediatelyWithoutRotating reproduces a
// production incident (2026-08-27): a plain 503 ("high demand") during OCR
// rotated through all 4 configured keys, getting the identical 503 from
// each one before finally failing — since the backend itself is
// overloaded, every key hits the same wall, so rotating just multiplies the
// latency of a failure that was never going to succeed on any key. A 5xx
// must fail on the first key so the caller's own cross-provider escalation
// (e.g. extraction.ocrWithEscalation) kicks in promptly instead.
func TestGemini_ServerErrorFailsImmediatelyWithoutRotating(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusServiceUnavailable, errBody: serverErrorBody},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof a 5xx
		// does not rotate to the next key.
		{events: []string{textEvent(t, "should never be reached")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "a 5xx must not rotate to the next key")
}

func TestGemini_NonRetryableErrorFailsImmediately(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusBadRequest, errBody: badRequestBody},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof the
		// second key is never tried.
		{events: []string{textEvent(t, "should never be reached")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "a non-retryable error must not rotate to the next key")
}

func TestGemini_AllKeysExhaustedAfterCycles(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{MaxRetryCycles: 2, CooldownSeconds: 0})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Contains(t, gotErr.Error(), "exhausted")
	require.Equal(t, 4, requestCount(), "2 keys x 2 cycles")
}

// TestGemini_RotationEmitsNoChunksFromFailedAttempt confirms rotation still
// forwards nothing from a key that fails before any content streams (the
// common case — an HTTP-level error on connect, before Gemini has sent a
// single event) — the caller only ever sees the succeeding key's output.
func TestGemini_RotationEmitsNoChunksFromFailedAttempt(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{events: []string{textEvent(t, "only this should appear")}},
	}
	p, _ := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)

	var chunks []llm.StreamChunk
	for c := range ch {
		chunks = append(chunks, c)
	}
	// Exactly one TextDelta chunk (key 2's) and one Done chunk — nothing
	// from key 1's failed attempt leaked through.
	var textChunks int
	for _, c := range chunks {
		if c.TextDelta != "" {
			textChunks++
			require.Equal(t, "only this should appear", c.TextDelta)
		}
	}
	require.Equal(t, 1, textChunks)
}

// TestGemini_ForwardsTextChunksLiveAsTheyArrive is the direct regression
// test for switching from full-response buffering to token-by-token
// forwarding (see attempt's doc comment): each SSE event's text must reach
// the caller as its own TextDelta chunk, not accumulated into one chunk
// only after the whole response finishes.
func TestGemini_ForwardsTextChunksLiveAsTheyArrive(t *testing.T) {
	script := []scriptedResponse{
		{events: []string{textEvent(t, "Hello, "), textEvent(t, "world!")}},
	}
	p, _ := newTestProvider(t, 1, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)

	var chunks []string
	for c := range ch {
		require.NoError(t, c.Err)
		if c.TextDelta != "" {
			chunks = append(chunks, c.TextDelta)
		}
	}
	require.Equal(t, []string{"Hello, ", "world!"}, chunks, "each event's text must arrive as its own chunk, not buffered into one")
}

// TestGemini_DoesNotRetryAfterPartialForward is the safety-rule regression
// test that makes live forwarding possible without risking duplicated or
// garbled output on rotation: once some text has already reached the
// caller for this attempt, a later error on the SAME attempt must NOT
// retry on the next key — even though the error (429/RESOURCE_EXHAUSTED)
// would otherwise be retryable — since a retry would re-run the whole
// request from scratch and duplicate what's already been forwarded (and,
// in production, already spoken via TTS by a caller like miranda).
func TestGemini_DoesNotRetryAfterPartialForward(t *testing.T) {
	script := []scriptedResponse{
		{events: []string{textEvent(t, "partial answer "), midStreamErrorEvent(t, 429, "RESOURCE_EXHAUSTED", "quota exceeded mid-stream")}},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof the
		// second key is never tried once output has been forwarded.
		{events: []string{textEvent(t, "should never be reached")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)

	var text string
	var gotErr error
	for c := range ch {
		text += c.TextDelta
		if c.Err != nil {
			gotErr = c.Err
		}
	}

	require.Equal(t, "partial answer ", text, "text forwarded before the mid-stream error must still reach the caller")
	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "must not rotate to the next key once output has already been forwarded")
}

// TestGemini_MaxTokensFinishReasonReturnsError is the regression test for
// the real OCR truncation this adapter previously missed entirely: Gemini
// stops generating once it hits its output token budget (frequently
// exhausted by internal "thinking" tokens before any of the actual answer
// is written) and reports finishReason=MAX_TOKENS on the stream's terminal
// chunk, but the stream itself still ends cleanly — no streamErr, nothing
// that isRetryable would ever see. Before this test existed, attempt()
// never looked at FinishReason at all and forwarded a Done chunk exactly as
// if the response were complete, so extraction.OCR (and anything else built
// on Chat) had no way to tell a truncated transcription apart from a whole
// one. A different API key wouldn't change the model's own output budget,
// so this must be terminal, not retried.
func TestGemini_MaxTokensFinishReasonReturnsError(t *testing.T) {
	script := []scriptedResponse{
		{events: []string{sseEvent(t, map[string]any{
			"candidates": []map[string]any{
				{
					"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": "boilerplate consent preamble, then cut off mid-"}}},
					"finishReason": "MAX_TOKENS",
				},
			},
		})}},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof a
		// truncated-but-error-free finish doesn't rotate to the next key.
		{events: []string{textEvent(t, "should never be reached")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "transcribe this"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Contains(t, gotErr.Error(), "MAX_TOKENS")
	require.Equal(t, 1, requestCount(), "must not rotate keys — a different key doesn't change the model's own output budget")
}

// TestGemini_PromptBlockedReturnsError is the regression test for a
// prompt-level safety block: Gemini refuses to generate candidates at all
// (most plausibly its safety filter tripping on sensitive content already
// accumulated in the conversation — e.g. a prior tool result that placed a
// full lab-report table into context) and reports PromptFeedback.BlockReason
// instead of any per-candidate FinishReason. Before this test existed,
// attempt() never looked at PromptFeedback and forwarded an empty Done chunk
// exactly as if the model had genuinely nothing to say, so a caller (and
// ultimately the end user) saw a silent blank reply with no error at all.
func TestGemini_PromptBlockedReturnsError(t *testing.T) {
	script := []scriptedResponse{
		{events: []string{sseEvent(t, map[string]any{
			"promptFeedback": map[string]any{"blockReason": "SAFETY"},
		})}},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof a
		// prompt-level block doesn't rotate to the next key.
		{events: []string{textEvent(t, "should never be reached")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "sensitive question"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Contains(t, gotErr.Error(), "SAFETY")
	require.Equal(t, 1, requestCount(), "must not rotate keys — a different key hits the same safety filter")
}

// TestGemini_EmptyResponseWithNoFinishReasonReturnsError is the regression
// test for the real incident this adapter previously missed entirely: a
// stream that ends with no streamErr, a candidate present but with nil
// Content, and no FinishReason ever populated on any chunk (and no
// PromptFeedback either) — observed directly on a live gemini-lite call.
// Before this test existed, attempt()'s abnormal-finish check required
// FinishReason to be non-empty to trip at all, so this exact shape fell
// straight through to the "succeeded" path with an empty Done chunk,
// reaching the end user as a blank reply with no error anywhere in the
// chain.
func TestGemini_EmptyResponseWithNoFinishReasonReturnsError(t *testing.T) {
	script := []scriptedResponse{
		{events: []string{sseEvent(t, map[string]any{
			"candidates": []map[string]any{{}},
			"usageMetadata": map[string]any{
				"promptTokenCount": 14609,
				"totalTokenCount":  14609,
			},
		})}},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof this
		// doesn't rotate to the next key (a different key won't change
		// what the model decided to generate).
		{events: []string{textEvent(t, "should never be reached")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "покажи анализ крови"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Contains(t, gotErr.Error(), "response incomplete")
	require.Equal(t, 1, requestCount())
}

// TestGemini_RotatesOnAuthError and TestGemini_RotatesOnForbiddenError are
// the regression tests for broadening isRetryable to per-key auth failures
// (401/403): a revoked/disabled/individually-restricted key is a property
// of THAT key, not the request, so a different configured key can still
// succeed — unlike a malformed request (400), which fails identically on
// every key and stays non-retryable (see TestGemini_NonRetryableErrorFailsImmediately).
func TestGemini_RotatesOnAuthError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusUnauthorized, errBody: authErrorBody},
		{events: []string{textEvent(t, "hi from key 2")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount())
}

func TestGemini_RotatesOnForbiddenError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusForbidden, errBody: forbiddenErrorBody},
		{events: []string{textEvent(t, "hi from key 2")}},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount())
}

func TestGemini_ToolCallArgumentsRoundTrip(t *testing.T) {
	args := map[string]any{
		"entity_id": "light.kitchen",
		"nested":    map[string]any{"brightness": float64(80), "on": true},
	}
	script := []scriptedResponse{
		{events: []string{functionCallEvent(t, "ha_call_service", args)}},
	}
	p, _ := newTestProvider(t, 1, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "turn on the kitchen light"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Len(t, got.ToolCalls, 1)
	require.Equal(t, "ha_call_service", got.ToolCalls[0].Name)
	require.NotEmpty(t, got.ToolCalls[0].ID)

	var roundTripped map[string]any
	require.NoError(t, json.Unmarshal([]byte(got.ToolCalls[0].Arguments), &roundTripped))
	require.Equal(t, args, roundTripped)
}

// TestGemini_CapturesThoughtSignatureIntoProviderMetadata is a regression
// test for a real bug found during manual verification against the live
// API: a function-call turn's response carries a thoughtSignature that
// MUST be echoed back on the next turn or Gemini returns a hard 400. This
// asserts the capture half (attempt/toLLMToolCall); see
// TestToGeminiContents_EchoesThoughtSignatureOnReplay for the echo half.
func TestGemini_CapturesThoughtSignatureIntoProviderMetadata(t *testing.T) {
	sigBytes := []byte("opaque-thought-signature-bytes")
	sigB64 := base64.StdEncoding.EncodeToString(sigBytes)
	script := []scriptedResponse{
		{events: []string{functionCallEventWithThoughtSignature(t, "get_weather", map[string]any{"city": "Paris"}, sigB64)}},
	}
	p, _ := newTestProvider(t, 1, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "weather in Paris?"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Len(t, got.ToolCalls, 1)
	require.Equal(t, sigB64, got.ToolCalls[0].ProviderMetadata)
}

// TestToGeminiContents_EchoesThoughtSignatureOnReplay is the echo half of
// the regression test above: a stored llm.ToolCall carrying
// ProviderMetadata (as a caller like miranda's orchestrator would replay it
// from its own history store on a later turn) must produce a genai.Part
// with ThoughtSignature set on the reconstructed FunctionCall part.
func TestToGeminiContents_EchoesThoughtSignatureOnReplay(t *testing.T) {
	sigBytes := []byte("opaque-thought-signature-bytes")
	sigB64 := base64.StdEncoding.EncodeToString(sigBytes)

	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: "weather in Paris?"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{ID: "call-1", Name: "get_weather", Arguments: `{"city":"Paris"}`, ProviderMetadata: sigB64},
		}},
		{Role: llm.RoleTool, ToolCallID: "call-1", Content: "Sunny, 22C"},
	}

	_, contents := toGeminiContents(msgs)
	require.Len(t, contents, 3)

	assistantContent := contents[1]
	require.Len(t, assistantContent.Parts, 1)
	require.NotNil(t, assistantContent.Parts[0].FunctionCall)
	require.Equal(t, sigBytes, assistantContent.Parts[0].ThoughtSignature)
}

// TestToGeminiContents_NoProviderMetadataUsesPlaceholderSignature confirms a
// tool call with no captured signature (older stored history, or a
// synthetic router-built call, e.g. router.appendEscalationTurn's handoff
// acknowledgment) falls back to Google's documented placeholder rather than
// sending an empty ThoughtSignature — an empty one is a hard 400 from the
// real API ("Function call is missing a thought_signature..."), confirmed
// in production when an escalation handoff replayed to gemini-strong.
func TestToGeminiContents_NoProviderMetadataUsesPlaceholderSignature(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{ID: "call-1", Name: "get_weather", Arguments: `{"city":"Paris"}`},
		}},
	}
	_, contents := toGeminiContents(msgs)
	require.Len(t, contents, 1)
	require.Equal(t, geminiThoughtSignaturePlaceholder, contents[0].Parts[0].ThoughtSignature)
}
