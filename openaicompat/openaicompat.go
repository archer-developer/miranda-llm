// Package openaicompat implements llm.Provider (and llm.StructuredProvider)
// on top of the official github.com/openai/openai-go/v2 SDK, pointed at any
// OpenAI Chat Completions compatible backend (OpenAI itself, or a
// local/self-hosted server such as Ollama, vLLM, LM Studio, or a hosted
// router like OpenRouter) via option.WithBaseURL. This is the single client
// used for every provider that isn't native Anthropic or native Gemini. See
// the gemini package for the sibling adapter this mirrors, including the
// same api_key_envs-driven key rotation (see keyrotation, isRetryable) —
// rotates on quota (429), a 503 (the one HTTP status universally meaning
// "temporarily overloaded" regardless of which backend baseURL points at —
// see isOverloaded), a first-response timeout (see attempt), and per-key
// auth errors (401/403). A quota or overload rotation also bans that key on
// the shared keyrotation.Banlist for a configurable duration, so a
// deployment that orders a free-tier key before a paid one fails over on
// the very next request instead of re-discovering the same failure from
// scratch — see isRetryable/banDuration.
package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"

	llm "github.com/archer-developer/miranda-llm"
	"github.com/archer-developer/miranda-llm/keyrotation"
)

// defaultStructuredSchemaName is used by Structured when the caller's
// StructuredRequest.SchemaName is empty.
const defaultStructuredSchemaName = "structured_output"

// defaultStructuredTemperature is used by Structured when
// llm.StructuredRequest.Temperature is nil — see that field's doc comment.
var defaultStructuredTemperature = 0.1

// toParamOpt converts an optional float64 to the param.Opt[float64] the SDK
// wants. Returns the zero value (omitted, per the `omitzero` tag) when v is
// nil, so the API's own default applies.
func toParamOpt(v *float64) param.Opt[float64] {
	if v == nil {
		return param.Opt[float64]{}
	}
	return openai.Float(*v)
}

// RotationConfig tunes this package's key-rotation. The zero value falls
// back to keyrotation's built-in defaults (1 cycle, no cooldown) plus this
// package's own ban defaults (30min/30min/10s). Field-for-field identical
// (same names, types, order) to gemini.RotationConfig/
// anthropic.RotationConfig on purpose — see gemini.RotationConfig's doc
// comment for why.
type RotationConfig struct {
	CooldownSeconds int
	MaxRetryCycles  int
	// QuotaBanMinutes is how long a key that hit a quota/rate-limit error
	// (429) is skipped by every subsequent Chat/Structured call, not just
	// retried within the current one. <= 0 uses defaultQuotaBan.
	QuotaBanMinutes int
	// OverloadBanMinutes is the same idea for a key that looks overloaded
	// (see isOverloaded), timed out waiting for a first response, or took
	// longer than OverloadTimeoutSeconds end-to-end even though it did
	// complete (see attempt and keyrotation.BanIfSlow). <= 0 uses
	// defaultOverloadBan.
	OverloadBanMinutes int
	// OverloadTimeoutSeconds bounds two separate things attempt checks
	// against the same budget: how long it waits for the FIRST streamed
	// event before treating the key as overloaded, and — post-hoc, once the
	// call has already finished — how long the whole call was allowed to
	// take before the key gets banned for future calls anyway (see
	// keyrotation.BanIfSlow). <= 0 uses defaultOverloadTimeout.
	OverloadTimeoutSeconds int
}

// Built-in ban/timeout defaults applied whenever the corresponding
// RotationConfig field is <= 0 — mirrors gemini.RotationConfig's own
// defaults.
const (
	defaultQuotaBan        = 30 * time.Minute
	defaultOverloadBan     = 30 * time.Minute
	defaultOverloadTimeout = 10 * time.Second
)

func (r RotationConfig) quotaBan() time.Duration {
	if r.QuotaBanMinutes <= 0 {
		return defaultQuotaBan
	}
	return time.Duration(r.QuotaBanMinutes) * time.Minute
}

func (r RotationConfig) overloadBan() time.Duration {
	if r.OverloadBanMinutes <= 0 {
		return defaultOverloadBan
	}
	return time.Duration(r.OverloadBanMinutes) * time.Minute
}

func (r RotationConfig) overloadTimeout() time.Duration {
	if r.OverloadTimeoutSeconds <= 0 {
		return defaultOverloadTimeout
	}
	return time.Duration(r.OverloadTimeoutSeconds) * time.Second
}

// Provider is an llm.Provider (and llm.StructuredProvider) backed by any
// OpenAI-compatible Chat Completions endpoint. Like gemini.Provider and
// anthropic.Provider, it holds one openai.Client per configured key and
// rotates across them on a quota or per-key auth error — see pump/attempt
// below.
type Provider struct {
	name     string
	model    string
	clients  []openai.Client // one per resolved API key, in configured order
	rotation RotationConfig
	// bans is shared across every Chat/Structured call this Provider ever
	// makes (constructed once in New, never per-call) — mirrors
	// gemini.Provider.bans; see keyrotation.Banlist.
	bans   *keyrotation.Banlist
	tracer llm.Tracer
	logger *slog.Logger
}

// New builds a Provider named name, targeting model on the endpoint at
// baseURL (empty baseURL means the real OpenAI API), resolving apiKeyEnvs to
// actual key values from the process environment and building one
// openai.Client per resolved key up front — mirrors gemini.New/
// anthropic.New. Unlike those two, an empty/unresolved apiKeyEnvs is not an
// error: local backends (Ollama, vLLM, LM Studio) commonly need no
// authentication at all, so New falls back to a single unauthenticated
// client in that case rather than failing to build the provider outright.
func New(name, baseURL, model string, apiKeyEnvs []string, rotation RotationConfig, logger *slog.Logger) *Provider {
	if logger == nil {
		logger = slog.Default()
	}

	// WithMaxRetries(0): the SDK's own default (2) retries 429/5xx
	// transparently, on the SAME key, before an error ever reaches
	// attempt/isRetryable — see anthropic.New's identical option for the
	// full reasoning (both SDKs are Stainless-generated with the same
	// default retry behavior). This package's own keyrotation.Run +
	// isRetryable is the single, explicit place that decides what gets
	// retried and how.
	baseOpts := []option.RequestOption{option.WithMaxRetries(0)}
	if baseURL != "" {
		baseOpts = append(baseOpts, option.WithBaseURL(baseURL))
	}

	var clients []openai.Client
	for _, env := range apiKeyEnvs {
		key := os.Getenv(env)
		if key == "" {
			continue
		}
		opts := append(append([]option.RequestOption{}, baseOpts...), option.WithAPIKey(key))
		clients = append(clients, openai.NewClient(opts...))
	}
	if len(clients) == 0 {
		// No key resolved — build one unauthenticated client (same
		// behavior as the old New's apiKey == "" branch) rather than
		// erroring, since that's a legitimate local-backend configuration.
		clients = []openai.Client{openai.NewClient(baseOpts...)}
	}

	return &Provider{name: name, model: model, clients: clients, rotation: rotation, bans: keyrotation.NewBanlist(), logger: logger}
}

func (p *Provider) Name() string { return p.name }

// SetTracer wires up request/response tracing (see the llmtrace package) —
// optional, and left unset (nil) by default so tests and any caller that
// doesn't care about it don't need to touch this. This Provider serializes
// its own actual API request/response (see trace), so the trace reflects
// exactly what went out and came back for this SDK.
func (p *Provider) SetTracer(t llm.Tracer) {
	p.tracer = t
}

// Chat implements llm.Provider by streaming a Chat Completions response,
// rotating across configured API keys on a retryable error (see
// pump/attempt), and re-assembling incrementally-streamed tool-call
// fragments (the API sends a function name/arguments split across many
// chunks, keyed by tool_call index) into whole llm.ToolCall values before
// they're handed to the caller.
func (p *Provider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	params := openai.ChatCompletionNewParams{
		Model:       shared.ChatModel(p.model),
		Messages:    toOpenAIMessages(req.Messages),
		Tools:       toOpenAITools(req.Tools),
		Temperature: toParamOpt(req.Temperature),
	}

	out := make(chan llm.StreamChunk)
	go p.pump(ctx, params, out)
	return out, nil
}

// pendingToolCall accumulates a tool call's fragments as they stream in.
type pendingToolCall struct {
	id, name, arguments string
}

// pump rotates across p.clients (one per configured API key) via
// keyrotation.Run, classifying each error via p.classify — quota, overload
// (503, or a first-response timeout), and per-key auth failures all rotate
// to the next key; quota/overload also ban that key on p.bans for future
// calls — mirrors gemini.Provider.pump/anthropic.Provider.pump.
func (p *Provider) pump(ctx context.Context, params openai.ChatCompletionNewParams, out chan<- llm.StreamChunk) {
	defer close(out)

	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}
	err := keyrotation.Run(ctx, p.logger, "openaicompat", len(p.clients), rotCfg, p.bans, p.classify,
		func(ctx context.Context, i int) error {
			return p.attempt(ctx, i, params, out)
		},
	)
	// errAttemptWritten means attempt already wrote its own terminal Err
	// chunk before returning — writing a second one here would be a
	// duplicate. Any other non-nil error (the rotation genuinely exhausted
	// every key/cycle, or ctx was cancelled) still needs pump to report it,
	// since nothing else has.
	if err != nil && !errors.Is(err, errAttemptWritten) {
		out <- llm.StreamChunk{Err: err}
	}
}

// errAttemptWritten marks that attempt already sent its own terminal
// StreamChunk{Err: ...} to out before returning — pump checks for this
// sentinel specifically so it doesn't write a second Err chunk on top of
// the one attempt already sent. Mirrors gemini/anthropic's own sentinel of
// the same name.
var errAttemptWritten = errors.New("openaicompat: attempt already wrote its terminal error")

// errFirstResponseTimeout means attempt gave up waiting for the first
// streamed chunk within RotationConfig.OverloadTimeoutSeconds, before
// anything was forwarded to the caller — mirrors gemini/anthropic's
// sentinel of the same name/reasoning.
var errFirstResponseTimeout = errors.New("openaicompat: no response within first-response timeout")

// attempt makes exactly one Chat.Completions.NewStreaming call against one
// client (one API key), forwarding text deltas to out live as they arrive
// and, via the SDK's own ChatCompletionAccumulator, reconstructing the full
// response for tracing — mirroring what the manually-accumulated
// pendingToolCall fragments below resolve to, but as the SDK's own response
// type so the trace reflects exactly what the API returned.
//
// Once ANY text has been forwarded to out for this attempt, a later error
// on the SAME attempt must not be retried even if isRetryable(err) would
// otherwise say yes — mirrors gemini/anthropic's own forwarded-then-error
// rule; see anthropic.Provider.attempt's doc comment for the full
// reasoning.
//
// Before anything is forwarded, attempt also bounds how long it waits for
// the very FIRST chunk via RotationConfig.OverloadTimeoutSeconds — mirrors
// gemini/anthropic's own first-response watchdog; see gemini.Provider.
// attempt's doc comment for the full reasoning.
func (p *Provider) attempt(ctx context.Context, keyIndex int, params openai.ChatCompletionNewParams, out chan<- llm.StreamChunk) error {
	start := time.Now()
	attemptCtx, cancelAttempt := context.WithCancel(ctx)
	defer cancelAttempt()

	// NewStreaming performs the actual HTTP request eagerly (before this
	// call returns), so a 429/401/403/5xx from the initial request surfaces
	// via stream.Err() below exactly like a mid-stream error would.
	stream := p.clients[keyIndex].Chat.Completions.NewStreaming(attemptCtx, params)

	var acc openai.ChatCompletionAccumulator
	forwarded := false

	// Tool calls stream as fragments identified by index; accumulate until
	// the stream ends, then emit each as one complete llm.ToolCall.
	pending := map[int64]*pendingToolCall{}
	var order []int64

	first := true
	for {
		var hasNext bool
		if first {
			first = false
			var abandoned bool
			hasNext, abandoned = keyrotation.FirstStepTimeout(ctx, p.rotation.overloadTimeout(), stream.Next)
			if abandoned {
				// stream.Next() may still be running on the goroutine
				// FirstStepTimeout spawned — cancelling attemptCtx unblocks
				// it promptly; this attempt must not touch stream again
				// without waiting for that goroutine first, which it never
				// does: it just returns, leaving the abandoned call to
				// finish and be garbage collected on its own — mirrors
				// anthropic.Provider.attempt's identical handling.
				cancelAttempt()
				if ctx.Err() != nil {
					return ctx.Err()
				}
				p.logger.Warn("openaicompat: no response within first-response timeout, treating key as overloaded", "provider", p.name, "key_index", keyIndex, "timeout", p.rotation.overloadTimeout())
				return errFirstResponseTimeout
			}
		} else {
			hasNext = stream.Next()
		}
		if !hasNext {
			break
		}
		chunk := stream.Current()
		acc.AddChunk(chunk)
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta

		if delta.Content != "" {
			forwarded = true
			out <- llm.StreamChunk{TextDelta: delta.Content}
		}

		for _, tc := range delta.ToolCalls {
			pc, ok := pending[tc.Index]
			if !ok {
				pc = &pendingToolCall{}
				pending[tc.Index] = pc
				order = append(order, tc.Index)
			}
			if tc.ID != "" {
				pc.id = tc.ID
			}
			if tc.Function.Name != "" {
				pc.name += tc.Function.Name
			}
			pc.arguments += tc.Function.Arguments
		}
	}

	if streamErr := stream.Err(); streamErr != nil {
		if forwarded {
			p.logger.Error("openaicompat: request failed mid-stream after partial output, not retrying", "provider", p.name, "key_index", keyIndex, "error", streamErr)
			wrapped := fmt.Errorf("openaicompat: stream: %w", streamErr)
			p.trace(ctx, params, nil, wrapped)
			out <- llm.StreamChunk{Err: wrapped}
			return errAttemptWritten
		}
		if isRetryable(streamErr) {
			p.logKeyRotate(keyIndex, streamErr)
			return streamErr
		}
		p.logger.Error("openaicompat: request failed", "provider", p.name, "key_index", keyIndex, "error", streamErr)
		wrapped := fmt.Errorf("openaicompat: stream: %w", streamErr)
		p.trace(ctx, params, nil, wrapped)
		out <- llm.StreamChunk{Err: wrapped}
		return errAttemptWritten
	}

	for _, idx := range order {
		pc := pending[idx]
		out <- llm.StreamChunk{ToolCall: &llm.ToolCall{ID: pc.id, Name: pc.name, Arguments: pc.arguments}}
	}

	if elapsed := time.Since(start); keyrotation.BanIfSlow(p.bans, keyIndex, elapsed, p.rotation.overloadTimeout(), p.rotation.overloadBan()) {
		p.logger.Warn("openaicompat: request succeeded but took longer than the overload timeout end-to-end, banning key for future calls", "provider", p.name, "key_index", keyIndex, "duration", elapsed, "timeout", p.rotation.overloadTimeout(), "ban", p.rotation.overloadBan())
	}
	p.trace(ctx, params, &acc.ChatCompletion, nil)
	out <- llm.StreamChunk{Done: true}
	return nil
}

// isOverloaded reports whether err carries HTTP 503 — the one status code
// universally meaning "temporarily overloaded, try again" across REST APIs
// regardless of which backend baseURL points at (unlike Anthropic/Gemini,
// there's no single semantic error-type field a generic OpenAI-compatible
// backend is guaranteed to populate consistently — see isRetryable's doc
// comment). Other 5xx codes (500/502/504) are more likely to indicate a
// genuine backend bug a retry can't fix, so only 503 is treated as
// retryable/bannable here — mirrors gemini/anthropic's own "specific shape
// only" scoping.
func isOverloaded(err error) bool {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return false
	}
	return apiErr.StatusCode == http.StatusServiceUnavailable
}

// isAuthFailure reports whether err is a per-key auth failure (HTTP 401
// Unauthorized / 403 Forbidden) — a revoked, disabled, or otherwise invalid
// key. Used only to pick a more pointed log message (see logKeyRotate):
// unlike quota/overload, this condition never clears on its own, so it is
// deliberately never banned (see banDuration) — a fixed-duration ban would
// just delay, not fix, the same failure repeating on every future call.
func isAuthFailure(err error) bool {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return false
	}
	return apiErr.StatusCode == http.StatusUnauthorized || apiErr.StatusCode == http.StatusForbidden
}

// logKeyRotate logs the WARN for a retryable per-attempt error right before
// keyrotation.Run moves on to the next key — mirrors gemini/anthropic's
// own. An auth failure gets a distinct, more pointed message: unlike
// quota/overload (banned — see banDuration), an invalid/revoked key stays
// invalid and will be retried and fail again on EVERY future call until a
// human fixes or removes it from api_key_envs.
func (p *Provider) logKeyRotate(keyIndex int, err error) {
	if isAuthFailure(err) {
		p.logger.Warn("openaicompat: key auth failure (401/403) — key looks invalid or revoked and needs manual attention; not banned, so it will be retried again on every future call until fixed", "provider", p.name, "key_index", keyIndex, "error", err)
		return
	}
	p.logger.Warn("openaicompat: key error, trying next key", "provider", p.name, "key_index", keyIndex, "error", err)
}

// isRetryable reports whether err is worth rotating to the next key for —
// mirrors gemini.isRetryable/anthropic's own version, just against the
// OpenAI-compatible SDK's error shape (*openai.Error, carrying StatusCode):
//   - a quota/rate-limit error (HTTP 429).
//   - HTTP 503 (see isOverloaded) or errFirstResponseTimeout — a different
//     key (especially a differently-tiered one) may not be hitting the same
//     overloaded capacity. This used to be excluded entirely (see the
//     production-incident note below) — it's back specifically for this
//     shape, gated by keyrotation.Banlist: an overloaded key gets banned
//     for RotationConfig.OverloadBanMinutes when it happens, so a run of
//     several configured keys all hitting the same genuine outage still
//     only pays the multiplied latency once, not on every subsequent
//     request until the outage clears.
//   - a per-key auth failure (HTTP 401 Unauthorized / 403 Forbidden).
//
// Any OTHER server-side error (a 5xx that isn't 503) is deliberately still
// NOT retried — every configured key hits the same struggling backend, so
// cycling through them just repeats the identical failure. See
// gemini.isRetryable's doc comment for the production incident (2026-08-27)
// this reasoning is shared with. Note this is a generic fallback for
// whatever backend baseURL points at (OpenAI itself, or a self-hosted
// server) — a local backend's own error shape may not populate StatusCode
// the same way OpenAI's does, in which case this simply returns false and
// the first failure surfaces immediately, same as today.
func isRetryable(err error) bool {
	if errors.Is(err, errFirstResponseTimeout) || isOverloaded(err) {
		return true
	}
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return false
	}
	switch apiErr.StatusCode {
	case http.StatusTooManyRequests, http.StatusUnauthorized, http.StatusForbidden:
		return true
	}
	return false
}

// banDuration reports how long to ban the key that produced err on the
// shared keyrotation.Banlist — 0 means "retry, but don't ban" (a per-key
// auth failure is retryable but isn't a transient condition a timed ban
// would help with).
func banDuration(err error, quotaBan, overloadBan time.Duration) time.Duration {
	if errors.Is(err, errFirstResponseTimeout) || isOverloaded(err) {
		return overloadBan
	}
	var apiErr *openai.Error
	if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests {
		return quotaBan
	}
	return 0
}

// classify adapts isRetryable/banDuration to keyrotation.Run's classify
// shape, supplying this Provider's configured ban durations.
func (p *Provider) classify(err error) (retryable bool, ban time.Duration) {
	return isRetryable(err), banDuration(err, p.rotation.quotaBan(), p.rotation.overloadBan())
}

// Structured implements llm.StructuredProvider using Chat Completions'
// native response_format: json_schema mode (supported by OpenAI itself and
// by many, but not all, OpenAI-compatible backends — a backend that
// doesn't recognize response_format simply returns its own error, which
// this method wraps and returns; there is no local fallback). One
// non-streaming call (Chat.Completions.New).
func (p *Provider) Structured(ctx context.Context, req llm.StructuredRequest) (json.RawMessage, error) {
	name := req.SchemaName
	if name == "" {
		name = defaultStructuredSchemaName
	}

	temperature := req.Temperature
	if temperature == nil {
		temperature = &defaultStructuredTemperature
	}
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(p.model),
		Messages: toOpenAIMessages(req.Messages),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   name,
					Schema: req.Schema,
				},
			},
		},
		Temperature: toParamOpt(temperature),
	}

	// Rotates across configured API keys exactly like Chat's pump/attempt,
	// via the same keyrotation.Run + p.classify (sharing the same p.bans) —
	// no first-response-timeout check here since this is a single
	// non-streaming call.
	var result json.RawMessage
	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}
	err := keyrotation.Run(ctx, p.logger, "openaicompat-structured", len(p.clients), rotCfg, p.bans, p.classify,
		func(ctx context.Context, keyIndex int) error {
			resp, err := p.clients[keyIndex].Chat.Completions.New(ctx, params)
			if err != nil {
				if isRetryable(err) {
					p.logKeyRotate(keyIndex, err)
				} else {
					p.logger.Error("openaicompat: request failed", "provider", p.name, "key_index", keyIndex, "error", err)
				}
				p.trace(ctx, params, nil, fmt.Errorf("openaicompat: structured: %w", err))
				return err
			}
			p.trace(ctx, params, resp, nil)

			if len(resp.Choices) == 0 {
				return fmt.Errorf("no choices returned")
			}
			result = json.RawMessage(resp.Choices[0].Message.Content)
			return nil
		},
	)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: structured: %w", err)
	}
	return result, nil
}

// trace serializes exactly what was sent (params) and what came back (the
// fully accumulated ChatCompletion, nil on error) to p.tracer, if one is
// configured. Shared by Chat's pump and Structured.
func (p *Provider) trace(ctx context.Context, params openai.ChatCompletionNewParams, resp *openai.ChatCompletion, err error) {
	if p.tracer == nil {
		return
	}
	reqJSON, marshalErr := json.MarshalIndent(params, "", "  ")
	if marshalErr != nil {
		reqJSON = []byte(fmt.Sprintf("(failed to marshal request: %v)", marshalErr))
	}
	var respJSON []byte
	if resp != nil {
		if respJSON, marshalErr = json.MarshalIndent(resp, "", "  "); marshalErr != nil {
			respJSON = []byte(fmt.Sprintf("(failed to marshal response: %v)", marshalErr))
		}
	}
	p.tracer.Trace(ctx, p.name, string(reqJSON), string(respJSON), err)
}

func toOpenAIMessages(msgs []llm.Message) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case llm.RoleSystem:
			out = append(out, openai.SystemMessage(m.Content))
		case llm.RoleUser:
			// Only switch to array-form content when there are actual image
			// blocks to include. Many OpenAI-compat local backends (Ollama,
			// vLLM, LM Studio) only accept content as a plain string and
			// reject the array form even for text-only content — so we must
			// not change the wire format just because Parts is non-empty.
			var imageParts []openai.ChatCompletionContentPartUnionParam
			for _, p := range m.Parts {
				if p.ImageBase64 != "" {
					dataURI := "data:" + p.MIMEType + ";base64," + p.ImageBase64
					imageParts = append(imageParts, openai.ImageContentPart(
						openai.ChatCompletionContentPartImageImageURLParam{URL: dataURI},
					))
				}
			}
			if len(imageParts) > 0 {
				// Multi-modal: text first, then images.
				parts := make([]openai.ChatCompletionContentPartUnionParam, 0, 1+len(imageParts))
				if m.Content != "" {
					parts = append(parts, openai.TextContentPart(m.Content))
				}
				out = append(out, openai.UserMessage(append(parts, imageParts...)))
			} else {
				out = append(out, openai.UserMessage(m.Content))
			}
		case llm.RoleTool:
			out = append(out, openai.ToolMessage(m.Content, m.ToolCallID))
		case llm.RoleAssistant:
			msg := openai.AssistantMessage(m.Content)
			if len(m.ToolCalls) > 0 {
				calls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					calls = append(calls, openai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
							ID: tc.ID,
							Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      tc.Name,
								Arguments: tc.Arguments,
							},
						},
					})
				}
				msg.OfAssistant.ToolCalls = calls
			}
			out = append(out, msg)
		}
	}
	return out
}

func toOpenAITools(tools []llm.ToolDef) []openai.ChatCompletionToolUnionParam {
	if len(tools) == 0 {
		return nil
	}
	out := make([]openai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		out = append(out, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        t.Name,
			Description: openai.String(t.Description),
			Parameters:  shared.FunctionParameters(t.Parameters),
		}))
	}
	return out
}
