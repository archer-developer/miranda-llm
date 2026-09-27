package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"

	llm "github.com/archer-developer/miranda-llm"
)

// TestToAnthropicTools_NoPropertiesKeyStillSendsInputSchema reproduces a
// production 400 ("tools.0.custom.input_schema: Field required"): a tool
// whose JSON schema has no "properties" key at all (e.g. a no-argument tool
// like {"type":"object","additionalProperties":false}) left Properties as a
// nil interface, which made the whole anthropic.ToolInputSchemaParam struct
// the Go zero value — the SDK's `omitzero` tag then dropped "input_schema"
// from the wire request entirely instead of sending an empty one.
func TestToAnthropicTools_NoPropertiesKeyStillSendsInputSchema(t *testing.T) {
	tools := toAnthropicTools([]llm.ToolDef{
		{
			Name:        "code_exec_sandbox_create_session",
			Description: "Start a session",
			Parameters:  map[string]any{"type": "object", "additionalProperties": false},
		},
	})
	require.Len(t, tools, 1)

	raw, err := tools[0].MarshalJSON()
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))
	require.Contains(t, decoded, "input_schema", "input_schema must always be sent, even for a tool with no properties")

	schema, ok := decoded["input_schema"].(map[string]any)
	require.True(t, ok, "input_schema must be an object")
	require.Equal(t, "object", schema["type"])
}

// TestRequiredFields covers the JSON Schema "required" array extraction
// requiredFields does for both toAnthropicTools and Structured.
func TestRequiredFields(t *testing.T) {
	require.Equal(t, []string{"a", "b"}, requiredFields(map[string]any{
		"type":     "object",
		"required": []any{"a", "b"},
	}))
	require.Nil(t, requiredFields(map[string]any{"type": "object"}))
}

// TestToAnthropicMessages_ToolCallWithEmptyOrMalformedArgumentsStillSendsInput
// reproduces a production 400 ("messages.N.content.0.tool_use.input: Field
// required"): a synthetic escalation tool call (router.escalateOnError) or
// any tool call with empty/invalid Arguments used to leave input as a nil
// interface, and the SDK drops "input" from the wire request entirely for a
// nil input — Anthropic requires the field on every tool_use block, even an
// empty object, and rejects the whole request otherwise.
func TestToAnthropicMessages_ToolCallWithEmptyOrMalformedArgumentsStillSendsInput(t *testing.T) {
	for _, args := range []string{"", "not valid json", `{"reason":"has a literal` + "\n" + `newline"}`} {
		_, msgs := toAnthropicMessages([]llm.Message{
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "escalate_to_claude", Arguments: args}}},
		})
		require.Len(t, msgs, 1)

		raw, err := json.Marshal(msgs[0])
		require.NoError(t, err)

		var decoded map[string]any
		require.NoError(t, json.Unmarshal(raw, &decoded))
		content, ok := decoded["content"].([]any)
		require.True(t, ok)
		require.Len(t, content, 1)
		block, ok := content[0].(map[string]any)
		require.True(t, ok)
		require.Contains(t, block, "input", "tool_use block must always carry an input field, got Arguments=%q", args)
	}
}

// TestToAnthropicMessages_PDFPartUsesDocumentBlockNotImageBlock reproduces a
// production 400 (2026-08-27): OCR's ContentPart carries MIMEType
// "application/pdf" straight from the caller's file — valid for Gemini's
// inlineData, but Claude only accepts image/jpeg|png|gif|webp in an image
// block and hard-rejects anything else. A PDF part must become a document
// block instead, so OCR escalation to Claude actually works for PDF input.
func TestToAnthropicMessages_PDFPartUsesDocumentBlockNotImageBlock(t *testing.T) {
	_, msgs := toAnthropicMessages([]llm.Message{
		{Role: llm.RoleUser, Parts: []llm.ContentPart{{ImageBase64: "ZmFrZS1wZGY=", MIMEType: "application/pdf"}}},
	})
	require.Len(t, msgs, 1)

	raw, err := json.Marshal(msgs[0])
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))
	content, ok := decoded["content"].([]any)
	require.True(t, ok)
	require.Len(t, content, 1)
	block, ok := content[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "document", block["type"], "a PDF part must become a document block, not an image block")

	source, ok := block["source"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "application/pdf", source["media_type"])
	require.Equal(t, "ZmFrZS1wZGY=", source["data"])
}

// TestToAnthropicMessages_SingleSystemBlockGetsCacheBreakpoint covers the
// common case (one caller-supplied RoleSystem message): it must still be
// marked, since it's both first and last.
func TestToAnthropicMessages_SingleSystemBlockGetsCacheBreakpoint(t *testing.T) {
	system, _ := toAnthropicMessages([]llm.Message{
		{Role: llm.RoleSystem, Content: "You are Miranda."},
		{Role: llm.RoleUser, Content: "hi"},
	})
	require.Len(t, system, 1)
	require.Equal(t, anthropic.NewCacheControlEphemeralParam(), system[0].CacheControl)
}

// TestToAnthropicMessages_CachesFirstSystemBlockNotLast reproduces the bug
// this ADR fixes (docs/adr/system-prompt-caching.md in the miranda repo): a
// caller sending a stable prefix (persona/memory) followed by a volatile
// suffix (current time, different on every turn) needs the breakpoint on
// the STABLE block. The old "mark the last block" behavior would instead
// mark the volatile one — a breakpoint that can never hit twice, since its
// content differs on every call — silently defeating caching for any
// caller that adopts a multi-block system prompt.
func TestToAnthropicMessages_CachesFirstSystemBlockNotLast(t *testing.T) {
	system, _ := toAnthropicMessages([]llm.Message{
		{Role: llm.RoleSystem, Content: "You are Miranda. Shared memory: ..."},
		{Role: llm.RoleSystem, Content: "Current time: 2026-08-14 12:00 MSK."},
		{Role: llm.RoleUser, Content: "hi"},
	})
	require.Len(t, system, 2)
	require.Equal(t, anthropic.NewCacheControlEphemeralParam(), system[0].CacheControl,
		"the stable (first) block must carry the breakpoint")
	require.Zero(t, system[1].CacheControl,
		"the volatile (last) block must NOT carry a breakpoint — its content differs every turn, so marking it would never hit")
}

// --- Below: the httptest.Server-backed key-rotation suite, mirroring
// gemini_test.go's own — see isRetryable's doc comment for the reasoning
// this exercises (429/401/403 rotate, 5xx does not).

// overloadedTestErr builds an *anthropic.Error whose Type() reports
// "overloaded_error" — apierror.Error's errorType field backing Type() is
// unexported, so a plain struct literal can't set it directly; this goes
// through the same UnmarshalJSON path richErrorDecoder.newAPIError uses on
// a real streamed error event, which is the only way to legitimately end up
// with that state anyway.
func overloadedTestErr(t *testing.T, statusCode int) *anthropic.Error {
	t.Helper()
	err := &anthropic.Error{StatusCode: statusCode}
	require.NoError(t, err.UnmarshalJSON([]byte(serverErrorBody)))
	return err
}

func TestIsRetryable(t *testing.T) {
	require.True(t, isRetryable(&anthropic.Error{StatusCode: http.StatusTooManyRequests}))
	require.True(t, isRetryable(&anthropic.Error{StatusCode: http.StatusUnauthorized}))
	require.True(t, isRetryable(&anthropic.Error{StatusCode: http.StatusForbidden}))

	// Anthropic's specific "overloaded" shape (error.type ==
	// "overloaded_error", normally delivered as HTTP 529) IS retryable — a
	// different, differently-tiered key may not be hitting the same
	// overloaded capacity pool. errFirstResponseTimeout (attempt's
	// time-to-first-event watchdog) is treated identically.
	require.True(t, isRetryable(overloadedTestErr(t, 529)))
	require.True(t, isRetryable(errFirstResponseTimeout))

	// Any OTHER 5xx (not Anthropic's specific overloaded_error type) is
	// deliberately still NOT retryable: every configured key hits the same
	// struggling backend, so rotating keys just repeats the identical
	// failure — see isRetryable's doc comment.
	require.False(t, isRetryable(&anthropic.Error{StatusCode: http.StatusInternalServerError}))
	require.False(t, isRetryable(&anthropic.Error{StatusCode: http.StatusServiceUnavailable}))
	require.False(t, isRetryable(&anthropic.Error{StatusCode: 529}))

	require.False(t, isRetryable(&anthropic.Error{StatusCode: http.StatusBadRequest}))
	require.False(t, isRetryable(errors.New("some unrelated network error")))
}

func TestBanDuration(t *testing.T) {
	quotaBan := 30 * time.Minute
	overloadBan := 45 * time.Minute

	require.Equal(t, quotaBan, banDuration(&anthropic.Error{StatusCode: http.StatusTooManyRequests}, quotaBan, overloadBan))
	require.Equal(t, overloadBan, banDuration(overloadedTestErr(t, 529), quotaBan, overloadBan))
	require.Equal(t, overloadBan, banDuration(errFirstResponseTimeout, quotaBan, overloadBan))

	// A per-key auth failure is retryable but not banned — the key itself
	// needs fixing, waiting doesn't help.
	require.Zero(t, banDuration(&anthropic.Error{StatusCode: http.StatusUnauthorized}, quotaBan, overloadBan))
	require.Zero(t, banDuration(&anthropic.Error{StatusCode: http.StatusForbidden}, quotaBan, overloadBan))
	require.Zero(t, banDuration(&anthropic.Error{StatusCode: http.StatusBadRequest}, quotaBan, overloadBan))
}

// scriptedResponse is one fake server reply, in the order Chat's/
// Structured's rotation loop will request them: a successful SSE stream
// (body, for Chat's NewStreaming), a successful plain JSON message
// (jsonBody, for Structured's non-streaming Messages.New — needs its own
// Content-Type: application/json, unlike the SSE case), or an HTTP error
// status with an Anthropic-shaped {"type":"error","error":{...}} body.
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

// anthropicSuccessBody builds the minimal well-formed SSE event sequence
// (message_start -> content_block_start -> content_block_delta ->
// content_block_stop -> message_delta -> message_stop) Message.Accumulate
// requires — each event needs its predecessor to have run (e.g.
// content_block_delta needs a content_block_start at the same index first),
// so a partial/ad-hoc event list fails accumulation outright rather than
// just under-reporting text.
func anthropicSuccessBody(t *testing.T, text string) string {
	t.Helper()
	var sb []string
	sb = append(sb, "event: message_start\ndata: "+mustJSON(t, map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test-model",
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 1, "output_tokens": 0},
		},
	})+"\n\n")
	sb = append(sb, "event: content_block_start\ndata: "+mustJSON(t, map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})+"\n\n")
	sb = append(sb, "event: content_block_delta\ndata: "+mustJSON(t, map[string]any{
		"type": "content_block_delta", "index": 0,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})+"\n\n")
	sb = append(sb, "event: content_block_stop\ndata: "+mustJSON(t, map[string]any{"type": "content_block_stop", "index": 0})+"\n\n")
	sb = append(sb, "event: message_delta\ndata: "+mustJSON(t, map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 5},
	})+"\n\n")
	sb = append(sb, "event: message_stop\ndata: "+mustJSON(t, map[string]any{"type": "message_stop"})+"\n\n")

	out := ""
	for _, s := range sb {
		out += s
	}
	return out
}

// quotaErrorBody is a 429 rate_limit_error response body.
const quotaErrorBody = `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`

// serverErrorBody is Anthropic's own "overloaded" response body — retryable/
// bannable, unlike an unrecognized 5xx (see isRetryable's doc comment).
const serverErrorBody = `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`

// genericServerErrorBody is a 5xx that is NOT Anthropic's specific
// "overloaded" shape — must still fail immediately without rotating, per
// the 2026-08-27 incident this behavior is named after.
const genericServerErrorBody = `{"type":"error","error":{"type":"api_error","message":"internal error"}}`

// badRequestBody is a non-retryable 400 response body.
const badRequestBody = `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`

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

	original := apiBaseURL
	apiBaseURL = ts.URL
	t.Cleanup(func() { apiBaseURL = original })

	return func() int { return requestCount }
}

func newTestProvider(t *testing.T, n int, script []scriptedResponse, rotation RotationConfig) (*Provider, func() int) {
	t.Helper()
	requestCount := newTestServer(t, script)

	envs := make([]string, n)
	for i := range n {
		env := fmt.Sprintf("ANTHROPIC_TEST_KEY_%d", i)
		t.Setenv(env, fmt.Sprintf("fake-key-%d", i))
		envs[i] = env
	}

	p, err := New("anthropic-test", "claude-test-model", envs, ToolsConfig{}, rotation, nil)
	require.NoError(t, err)
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

func TestAnthropicNew_NoKeysConfiguredErrors(t *testing.T) {
	_, err := New("anthropic-test", "claude-test-model", []string{"ANTHROPIC_TEST_UNSET_ENV_VAR"}, ToolsConfig{}, RotationConfig{}, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "none of the configured api_key_envs")
}

func TestAnthropic_RotatesOnQuotaError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{body: anthropicSuccessBody(t, "hi from key 2")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount(), "exactly one request per key, no needless extra cycling")
}

// TestAnthropic_OverloadedRotatesAndBansTheKey is the Anthropic counterpart
// to gemini's TestGemini_OverloadedRotatesAndBansTheKey: Anthropic's own
// overloaded_error now rotates to the next configured key and bans the
// overloaded one on p.bans.
func TestAnthropic_OverloadedRotatesAndBansTheKey(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusServiceUnavailable, errBody: serverErrorBody},
		{body: anthropicSuccessBody(t, "hi from key 2")},
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

// TestAnthropic_OverloadedKeyIsSkippedOnNextCall mirrors gemini's
// TestGemini_OverloadedKeyIsSkippedOnNextCall.
func TestAnthropic_OverloadedKeyIsSkippedOnNextCall(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusServiceUnavailable, errBody: serverErrorBody},
		{body: anthropicSuccessBody(t, "first call, key 2")},
		{body: anthropicSuccessBody(t, "second call, key 2 again")},
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

// TestAnthropic_GenericServerErrorFailsImmediatelyWithoutRotating is the
// Anthropic counterpart to gemini's
// TestGemini_GenericServerErrorFailsImmediatelyWithoutRotating — same
// production incident (2026-08-27), same fix, different provider: an
// unrecognized 5xx (not Anthropic's specific overloaded_error type) must
// still fail on the first key.
func TestAnthropic_GenericServerErrorFailsImmediatelyWithoutRotating(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusInternalServerError, errBody: genericServerErrorBody},
		// A second scripted entry that would fail the test via
		// newTestServer's t.Fatalf if it were ever requested — proof this
		// does not rotate to the next key.
		{body: anthropicSuccessBody(t, "should never be reached")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "an unrecognized 5xx must not rotate to the next key")
}

func TestAnthropic_NonRetryableErrorFailsImmediately(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusBadRequest, errBody: badRequestBody},
		{body: anthropicSuccessBody(t, "should never be reached")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	gotErr := collectErr(t, ch)

	require.Error(t, gotErr)
	require.Equal(t, 1, requestCount(), "a non-retryable error must not rotate to the next key")
}

func TestAnthropic_RotatesOnAuthError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusUnauthorized, errBody: `{"type":"error","error":{"type":"authentication_error","message":"invalid key"}}`},
		{body: anthropicSuccessBody(t, "hi from key 2")},
	}
	p, requestCount := newTestProvider(t, 2, script, RotationConfig{})

	ch, err := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, err)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.Equal(t, 2, requestCount())
}

// TestAnthropic_AuthFailureLogsDistinctWarning mirrors gemini's
// TestGemini_AuthFailureLogsDistinctWarning: a 401/403 is never banned (see
// banDuration) and so will repeat on every future call until a human fixes
// the key — this asserts that WARN line is distinguishable from the
// generic "trying next key" one, not just present.
func TestAnthropic_AuthFailureLogsDistinctWarning(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusUnauthorized, errBody: `{"type":"error","error":{"type":"authentication_error","message":"invalid key"}}`},
		{body: anthropicSuccessBody(t, "hi from key 2")},
	}
	requestCount := newTestServer(t, script)

	t.Setenv("ANTHROPIC_AUTH_LOG_TEST_KEY_0", "fake-key-0")
	t.Setenv("ANTHROPIC_AUTH_LOG_TEST_KEY_1", "fake-key-1")
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	p, err := New("anthropic-test", "claude-test-model",
		[]string{"ANTHROPIC_AUTH_LOG_TEST_KEY_0", "ANTHROPIC_AUTH_LOG_TEST_KEY_1"},
		ToolsConfig{}, RotationConfig{}, logger)
	require.NoError(t, err)

	ch, chatErr := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, chatErr)
	collect(t, ch)
	require.Equal(t, 2, requestCount())

	require.Contains(t, buf.String(), "auth failure", "a 401/403 must log a distinct message, not the generic rotation one")
	require.Contains(t, buf.String(), "manual attention")
	require.NotContains(t, buf.String(), "key error, trying next key", "the generic rotation message must not ALSO fire for an auth failure")
}

// TestAnthropic_AllKeysBannedAfterFirstCycleFailsFastWithoutASecondCycle
// mirrors gemini's TestGemini_AllKeysBannedAfterFirstCycleFailsFastWithoutASecondCycle
// — see its doc comment for why banning makes the second cycle a no-op here.
func TestAnthropic_AllKeysBannedAfterFirstCycleFailsFastWithoutASecondCycle(t *testing.T) {
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

// TestAnthropic_QuotaErrorBansTheKeyForFutureCalls mirrors gemini's
// TestGemini_QuotaErrorBansTheKeyForFutureCalls.
func TestAnthropic_QuotaErrorBansTheKeyForFutureCalls(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{body: anthropicSuccessBody(t, "first call, key 2")},
		{body: anthropicSuccessBody(t, "second call, key 2 again")},
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

// TestAnthropic_FirstResponseTimeoutRotatesAndBans mirrors gemini's
// TestGemini_FirstResponseTimeoutRotatesAndBans: a backend that accepts the
// connection but never sends a single SSE event within
// OverloadTimeoutSeconds must be treated as overloaded (ban + rotate).
func TestAnthropic_FirstResponseTimeoutRotatesAndBans(t *testing.T) {
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
		_, _ = w.Write([]byte(anthropicSuccessBody(t, "hi from key 2")))
	}))
	defer ts.Close()
	original := apiBaseURL
	apiBaseURL = ts.URL
	defer func() { apiBaseURL = original }()

	t.Setenv("ANTHROPIC_TIMEOUT_TEST_KEY_0", "fake-key-0")
	t.Setenv("ANTHROPIC_TIMEOUT_TEST_KEY_1", "fake-key-1")
	p, err := New("anthropic-test", "claude-test-model",
		[]string{"ANTHROPIC_TIMEOUT_TEST_KEY_0", "ANTHROPIC_TIMEOUT_TEST_KEY_1"},
		ToolsConfig{}, RotationConfig{OverloadTimeoutSeconds: 1}, nil)
	require.NoError(t, err)

	ch, chatErr := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, chatErr)
	got := collect(t, ch)

	require.Equal(t, "hi from key 2", got.Text)
	require.True(t, p.bans.Banned(0), "the slow-to-respond key must be banned")
}

// TestAnthropic_SlowButSuccessfulCallBansKeyAfterwards mirrors gemini's
// TestGemini_SlowButSuccessfulCallBansKeyAfterwards: the key delivers its
// first content_block_delta promptly (so the first-response watchdog never
// fires), then the rest of the stream arrives long after
// OverloadTimeoutSeconds. The call must still succeed on key 0, but key 0
// must come out banned so a later call skips straight to another key.
func TestAnthropic_SlowButSuccessfulCallBansKeyAfterwards(t *testing.T) {
	full := anthropicSuccessBody(t, "slow but successful")
	splitAt := strings.Index(full, "event: content_block_stop")
	require.Greater(t, splitAt, 0, "test fixture must contain a content_block_stop event to split on")
	first, rest := full[:splitAt], full[splitAt:]

	requestCount := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(first))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Well past OverloadTimeoutSeconds below, but the first delta above
		// already went out before this sleep starts.
		time.Sleep(1200 * time.Millisecond)
		_, _ = w.Write([]byte(rest))
	}))
	defer ts.Close()
	original := apiBaseURL
	apiBaseURL = ts.URL
	defer func() { apiBaseURL = original }()

	t.Setenv("ANTHROPIC_SLOW_TEST_KEY_0", "fake-key-0")
	p, err := New("anthropic-test", "claude-test-model",
		[]string{"ANTHROPIC_SLOW_TEST_KEY_0"},
		ToolsConfig{}, RotationConfig{OverloadTimeoutSeconds: 1}, nil)
	require.NoError(t, err)

	ch, chatErr := p.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}})
	require.NoError(t, chatErr)
	got := collect(t, ch)

	require.Equal(t, "slow but successful", got.Text, "the slow call must still succeed and deliver its already-forwarded content")
	require.Equal(t, 1, requestCount)
	require.True(t, p.bans.Banned(0), "a key that took longer than the overload timeout end-to-end must be banned even though it succeeded")
}

func TestAnthropic_StructuredRotatesOnQuotaError(t *testing.T) {
	script := []scriptedResponse{
		{statusCode: http.StatusTooManyRequests, errBody: quotaErrorBody},
		{jsonBody: mustJSON(t, map[string]any{
			"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test-model",
			"content": []map[string]any{
				{"type": "tool_use", "id": "toolu_1", "name": "structured_output", "input": map[string]any{"ok": true}},
			},
			"stop_reason": "tool_use", "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 1, "output_tokens": 1},
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

// TestAnthropic_ChatNeverForwardsTemperature reproduces a production 400
// (2026-08-27): extraction.OCR always sets ChatRequest.Temperature
// explicitly (0.1, to keep Gemini from wandering off mid-transcription)
// with no way to know at call time whether Gemini or an escalated Claude
// call ends up handling the request — a current Claude model hard-rejects
// the `temperature` field outright ("deprecated for this model") for any
// value at all, so every OCR call that escalated to Claude 400'd. Chat must
// never put "temperature" on the wire, even when the caller explicitly
// sets one — see Chat's own doc comment for the full incident.
func TestAnthropic_ChatNeverForwardsTemperature(t *testing.T) {
	var capturedBody string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		capturedBody = string(body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(anthropicSuccessBody(t, "ok")))
	}))
	defer ts.Close()

	original := apiBaseURL
	apiBaseURL = ts.URL
	defer func() { apiBaseURL = original }()

	t.Setenv("ANTHROPIC_TEST_TEMP_KEY", "fake-key")
	p, err := New("anthropic-test", "claude-test-model", []string{"ANTHROPIC_TEST_TEMP_KEY"}, ToolsConfig{}, RotationConfig{}, nil)
	require.NoError(t, err)

	temp := 0.1
	ch, err := p.Chat(context.Background(), llm.ChatRequest{
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Temperature: &temp,
	})
	require.NoError(t, err)
	collect(t, ch)

	require.NotContains(t, capturedBody, "temperature", "Chat must never forward Temperature to the Anthropic API")
}
