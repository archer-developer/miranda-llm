package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/require"

	llm "github.com/archer-developer/miranda-llm"
)

// --- The httptest.Server-backed key-rotation suite, mirroring
// gemini_test.go/anthropic_test.go's own — see isRetryable's doc comment
// for the reasoning this exercises (429/401/403 rotate, 5xx does not).

func TestIsRetryable(t *testing.T) {
	require.True(t, isRetryable(&openai.Error{StatusCode: http.StatusTooManyRequests}))
	require.True(t, isRetryable(&openai.Error{StatusCode: http.StatusUnauthorized}))
	require.True(t, isRetryable(&openai.Error{StatusCode: http.StatusForbidden}))

	// 5xx is deliberately NOT retryable: every configured key hits the same
	// overloaded backend, so rotating keys just repeats the identical
	// failure — see isRetryable's doc comment.
	require.False(t, isRetryable(&openai.Error{StatusCode: http.StatusInternalServerError}))
	require.False(t, isRetryable(&openai.Error{StatusCode: http.StatusServiceUnavailable}))

	require.False(t, isRetryable(&openai.Error{StatusCode: http.StatusBadRequest}))
	require.False(t, isRetryable(errors.New("some unrelated network error")))
}

// scriptedResponse is one fake server reply, in the order Chat's/
// Structured's rotation loop will request them: a successful SSE stream
// (body, for Chat's NewStreaming), a successful plain JSON completion
// (jsonBody, for Structured's non-streaming Chat.Completions.New), or an
// HTTP error status with an OpenAI-shaped {"error":{...}} body.
type scriptedResponse struct {
	body       string
	jsonBody   string
	statusCode int
	errBody    string
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return string(b)
}

// openaiSuccessBody builds a minimal well-formed Chat Completions SSE chunk
// sequence: one role-opening chunk, one content-delta chunk carrying text,
// one finish_reason:"stop" chunk, terminated by the standard "data:
// [DONE]" sentinel line the SDK's Stream.Next() specifically checks for.
func openaiSuccessBody(t *testing.T, text string) string {
	t.Helper()
	chunk := func(delta map[string]any, finishReason any) string {
		return "data: " + mustJSON(t, map[string]any{
			"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finishReason}},
		}) + "\n\n"
	}
	out := chunk(map[string]any{"role": "assistant", "content": ""}, nil)
	out += chunk(map[string]any{"content": text}, nil)
	out += chunk(map[string]any{}, "stop")
	out += "data: [DONE]\n\n"
	return out
}

// quotaErrorBody is a 429 rate-limit response body.
const quotaErrorBody = `{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}`

// serverErrorBody is a generic 5xx response body.
const serverErrorBody = `{"error":{"message":"backend overloaded","type":"server_error","code":null}}`

// badRequestBody is a non-retryable 400 response body.
const badRequestBody = `{"error":{"message":"bad request","type":"invalid_request_error","code":null}}`

// newTestServer serves script in request order and returns the server's own
// base URL (passed straight to New's baseURL parameter — unlike
// gemini/anthropic, this package's New already takes baseURL directly, no
// package-level override var needed) alongside a request-count getter.
func newTestServer(t *testing.T, script []scriptedResponse) (string, func() int) {
	t.Helper()
	requestCount := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := requestCount
		requestCount++
		if idx >= len(script) {
			t.Fatalf("newTestServer: got more requests (%d) than scripted responses (%d)", idx+1, len(script))
		}
		resp := script[idx]
		if resp.statusCode != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(resp.statusCode)
			_, _ = w.Write([]byte(resp.errBody))
			return
		}
		if resp.jsonBody != "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(resp.jsonBody))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(resp.body))
	}))
	t.Cleanup(ts.Close)
	return ts.URL, func() int { return requestCount }
}

func newTestProvider(t *testing.T, n int, script []scriptedResponse, rotation RotationConfig) (*Provider, func() int) {
	t.Helper()
	baseURL, requestCount := newTestServer(t, script)

	envs := make([]string, n)
	for i := range n {
		env := fmt.Sprintf("OPENAI_TEST_KEY_%d", i)
		t.Setenv(env, fmt.Sprintf("fake-key-%d", i))
		envs[i] = env
	}

	p := New("openaicompat-test", baseURL, "test-model", envs, rotation, nil)
	return p, requestCount
}

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

func TestOpenAICompatNew_NoKeyResolvedBuildsUnauthenticatedClient(t *testing.T) {
	// Unlike gemini.New/anthropic.New, an unresolved apiKeyEnvs is not an
	// error here — see New's own doc comment for why (local backends like
	// Ollama commonly need no auth at all).
	p := New("openaicompat-test", "", "test-model", []string{"OPENAI_TEST_UNSET_ENV_VAR"}, RotationConfig{}, nil)
	require.NotNil(t, p)
	require.Len(t, p.clients, 1)
}

func TestOpenAICompat_RotatesOnQuotaError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{body: openaiSuccessBody(t, "hi from key 2")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount(), "exactly one request per key, no needless extra cycling")
}

// TestOpenAICompat_ServerErrorFailsImmediatelyWithoutRotating is the
// OpenAI-compat counterpart to gemini's/anthropic's own — same production
// incident (2026-08-27), same fix, third provider.
func TestOpenAICompat_ServerErrorFailsImmediatelyWithoutRotating(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusServiceUnavailable, errBody: serverErrorBody},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof a 5xx
		// does not rotate to the next key.
		{body: openaiSuccessBody(t, "should never be reached")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "a 5xx must not rotate to the next key")
}

func TestOpenAICompat_NonRetryableErrorFailsImmediately(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusBadRequest, errBody: badRequestBody},
		{body: openaiSuccessBody(t, "should never be reached")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "a non-retryable error must not rotate to the next key")
}

func TestOpenAICompat_RotatesOnAuthError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusUnauthorized, errBody: `{"error":{"message":"invalid key","type":"authentication_error","code":null}}`},
		{body: openaiSuccessBody(t, "hi from key 2")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount())
}

func TestOpenAICompat_AllKeysExhaustedAfterCycles(t *testing.T) {
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

func TestOpenAICompat_StructuredRotatesOnQuotaError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{jsonBody: mustJSON(t, map[string]any{
			"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "test-model",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": `{"ok":true}`}, "finish_reason": "stop"},
			},
		})},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	raw, err := p.Structured(context.Background(), llm.StructuredRequest{
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Schema:     map[string]any{"type": "object", "properties": map[string]any{"ok": map[string]any{"type": "boolean"}}},
		SchemaName: "structured_output",
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"ok":true}`, string(raw))
	require.Equal(t, 2, requestCount())
}
