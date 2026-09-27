package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/require"

	llm "github.com/archer-developer/miranda-llm"
)

// --- The httptest.Server-backed key-rotation suite, mirroring
// gemini_test.go/anthropic_test.go's own — see isRetryable's doc comment
// for the reasoning this exercises (429/401/403/503 rotate, other 5xx does
// not).

func TestIsRetryable(t *testing.T) {
	require.True(t, isRetryable(&openai.Error{StatusCode: http.StatusTooManyRequests}))
	require.True(t, isRetryable(&openai.Error{StatusCode: http.StatusUnauthorized}))
	require.True(t, isRetryable(&openai.Error{StatusCode: http.StatusForbidden}))

	// 503 (the one HTTP status universally meaning "temporarily overloaded"
	// across REST APIs) IS retryable — a different, differently-tiered key
	// may not be hitting the same overloaded capacity. errFirstResponseTimeout
	// is treated identically.
	require.True(t, isRetryable(&openai.Error{StatusCode: http.StatusServiceUnavailable}))
	require.True(t, isRetryable(errFirstResponseTimeout))

	// Any OTHER 5xx is deliberately still NOT retryable: every configured
	// key hits the same struggling backend, so rotating keys just repeats
	// the identical failure — see isRetryable's doc comment.
	require.False(t, isRetryable(&openai.Error{StatusCode: http.StatusInternalServerError}))

	require.False(t, isRetryable(&openai.Error{StatusCode: http.StatusBadRequest}))
	require.False(t, isRetryable(errors.New("some unrelated network error")))
}

func TestBanDuration(t *testing.T) {
	quotaBan := 30 * time.Minute
	overloadBan := 45 * time.Minute

	require.Equal(t, quotaBan, banDuration(&openai.Error{StatusCode: http.StatusTooManyRequests}, quotaBan, overloadBan))
	require.Equal(t, overloadBan, banDuration(&openai.Error{StatusCode: http.StatusServiceUnavailable}, quotaBan, overloadBan))
	require.Equal(t, overloadBan, banDuration(errFirstResponseTimeout, quotaBan, overloadBan))

	require.Zero(t, banDuration(&openai.Error{StatusCode: http.StatusUnauthorized}, quotaBan, overloadBan))
	require.Zero(t, banDuration(&openai.Error{StatusCode: http.StatusForbidden}, quotaBan, overloadBan))
	require.Zero(t, banDuration(&openai.Error{StatusCode: http.StatusBadRequest}, quotaBan, overloadBan))
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

// serverErrorBody is a 503 ("overloaded") response body — retryable/
// bannable, unlike a non-503 5xx (see isRetryable's doc comment).
const serverErrorBody = `{"error":{"message":"backend overloaded","type":"server_error","code":null}}`

// genericServerErrorBody is a 5xx that is NOT 503 — must still fail
// immediately without rotating, per the 2026-08-27 incident this behavior
// is named after.
const genericServerErrorBody = `{"error":{"message":"internal error","type":"server_error","code":null}}`

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

// TestOpenAICompat_OverloadedRotatesAndBansTheKey is the OpenAI-compat
// counterpart to gemini's/anthropic's own overload test: a 503 now rotates
// to the next configured key and bans the overloaded one on p.bans.
func TestOpenAICompat_OverloadedRotatesAndBansTheKey(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusServiceUnavailable, errBody: serverErrorBody},
		{body: openaiSuccessBody(t, "hi from key 2")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount())
	require.True(t, p.bans.Banned(0), "the overloaded key must be banned")
	require.False(t, p.bans.Banned(1))
}

// TestOpenAICompat_OverloadedKeyIsSkippedOnNextCall mirrors gemini's/
// anthropic's own.
func TestOpenAICompat_OverloadedKeyIsSkippedOnNextCall(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusServiceUnavailable, errBody: serverErrorBody},
		{body: openaiSuccessBody(t, "first call, key 2")},
		{body: openaiSuccessBody(t, "second call, key 2 again")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	collect(t, ch)
	require.Equal(t, 2, requestCount())

	ch2, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi again"}}})
	require.NoError(t, err)
	got2 := collect(t, ch2)

	require.Equal(t, "second call, key 2 again", got2.Text)
	require.Equal(t, 3, requestCount(), "the banned key must be skipped outright, not requested a second time")
}

// TestOpenAICompat_GenericServerErrorFailsImmediatelyWithoutRotating is the
// OpenAI-compat counterpart to gemini's/anthropic's own — same production
// incident (2026-08-27), same fix, third provider: a non-503 5xx must still
// fail on the first key.
func TestOpenAICompat_GenericServerErrorFailsImmediatelyWithoutRotating(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusInternalServerError, errBody: genericServerErrorBody},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof this
		// does not rotate to the next key.
		{body: openaiSuccessBody(t, "should never be reached")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "a non-503 5xx must not rotate to the next key")
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

// TestOpenAICompat_AuthFailureLogsDistinctWarning mirrors gemini's/
// anthropic's own: a 401/403 is never banned (see banDuration) and so will
// repeat on every future call until a human fixes the key — this asserts
// that WARN line is distinguishable from the generic "trying next key"
// one, not just present.
func TestOpenAICompat_AuthFailureLogsDistinctWarning(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusUnauthorized, errBody: `{"error":{"message":"invalid key","type":"authentication_error","code":null}}`},
		{body: openaiSuccessBody(t, "hi from key 2")},
	}
	baseURL, requestCount := newTestServer(t, script)

	t.Setenv("OPENAI_AUTH_LOG_TEST_KEY_0", "fake-key-0")
	t.Setenv("OPENAI_AUTH_LOG_TEST_KEY_1", "fake-key-1")
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	p := New("openaicompat-test", baseURL, "test-model",
		[]string{"OPENAI_AUTH_LOG_TEST_KEY_0", "OPENAI_AUTH_LOG_TEST_KEY_1"},
		RotationConfig{}, logger)

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	collect(t, ch)
	require.Equal(t, 2, requestCount())

	require.Contains(t, buf.String(), "auth failure", "a 401/403 must log a distinct message, not the generic rotation one")
	require.Contains(t, buf.String(), "manual attention")
	require.NotContains(t, buf.String(), "key error, trying next key", "the generic rotation message must not ALSO fire for an auth failure")
}

// TestOpenAICompat_AllKeysBannedAfterFirstCycleFailsFastWithoutASecondCycle
// mirrors gemini's/anthropic's own — see their doc comments for why banning
// makes the second cycle a no-op here.
func TestOpenAICompat_AllKeysBannedAfterFirstCycleFailsFastWithoutASecondCycle(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{MaxRetryCycles: 2, CooldownSeconds: 0})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Contains(t, gotErr.Error(), "banned")
	require.Equal(t, 2, requestCount(), "cycle 2 must find every key already banned and skip straight to failing")
}

// TestOpenAICompat_QuotaErrorBansTheKeyForFutureCalls mirrors gemini's/
// anthropic's own.
func TestOpenAICompat_QuotaErrorBansTheKeyForFutureCalls(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{body: openaiSuccessBody(t, "first call, key 2")},
		{body: openaiSuccessBody(t, "second call, key 2 again")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	collect(t, ch)
	require.True(t, p.bans.Banned(0))

	ch2, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi again"}}})
	require.NoError(t, err)
	got2 := collect(t, ch2)

	require.Equal(t, "second call, key 2 again", got2.Text)
	require.Equal(t, 3, requestCount())
}

// TestOpenAICompat_FirstResponseTimeoutRotatesAndBans mirrors gemini's/
// anthropic's own: a backend that accepts the connection but never sends a
// single chunk within OverloadTimeoutSeconds must be treated as overloaded
// (ban + rotate).
func TestOpenAICompat_FirstResponseTimeoutRotatesAndBans(t *testing.T) {
	requestCount := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := requestCount
		requestCount++
		if idx == 0 {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(openaiSuccessBody(t, "hi from key 2")))
	}))
	defer ts.Close()

	t.Setenv("OPENAI_TIMEOUT_TEST_KEY_0", "fake-key-0")
	t.Setenv("OPENAI_TIMEOUT_TEST_KEY_1", "fake-key-1")
	p := New("openaicompat-test", ts.URL, "test-model",
		[]string{"OPENAI_TIMEOUT_TEST_KEY_0", "OPENAI_TIMEOUT_TEST_KEY_1"},
		RotationConfig{OverloadTimeoutSeconds: 1}, nil)

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.True(t, p.bans.Banned(0), "the slow-to-respond key must be banned")
}

// TestOpenAICompat_SlowButSuccessfulCallBansKeyAfterwards mirrors gemini's/
// anthropic's own: the key delivers its first chunk promptly (so the
// first-response watchdog never fires), then the rest of the stream arrives
// long after OverloadTimeoutSeconds. The call must still succeed on key 0,
// but key 0 must come out banned so a later call skips straight to another
// key.
func TestOpenAICompat_SlowButSuccessfulCallBansKeyAfterwards(t *testing.T) {
	full := openaiSuccessBody(t, "slow but successful")
	parts := strings.SplitN(full, "\n\n", 2)
	require.Len(t, parts, 2, "test fixture must contain more than one SSE chunk to split on")
	first, rest := parts[0]+"\n\n", parts[1]

	requestCount := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(first))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Well past OverloadTimeoutSeconds below, but the first chunk above
		// already went out before this sleep starts.
		time.Sleep(1200 * time.Millisecond)
		_, _ = w.Write([]byte(rest))
	}))
	defer ts.Close()

	t.Setenv("OPENAI_SLOW_TEST_KEY_0", "fake-key-0")
	p := New("openaicompat-test", ts.URL, "test-model",
		[]string{"OPENAI_SLOW_TEST_KEY_0"},
		RotationConfig{OverloadTimeoutSeconds: 1}, nil)

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "slow but successful", got.Text, "the slow call must still succeed and deliver its already-forwarded content")
	require.Equal(t, 1, requestCount)
	require.True(t, p.bans.Banned(0), "a key that took longer than the overload timeout end-to-end must be banned even though it succeeded")
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
