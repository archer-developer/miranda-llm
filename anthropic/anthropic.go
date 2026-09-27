// Package anthropic implements llm.Provider (and llm.StructuredProvider) on
// top of the official github.com/anthropics/anthropic-sdk-go SDK, used
// exclusively for Claude — it gets full native support for tool use,
// streaming, and prompt caching, unlike routing Claude through an
// OpenAI-compatibility shim. See the gemini package for the sibling adapter
// this mirrors, including the same api_key_envs-driven key rotation (see
// keyrotation, isRetryable) — rotates on quota (429), Anthropic's own
// "overloaded" error (error.type == "overloaded_error", however it's
// wrapped — see isOverloaded), a first-response timeout (see attempt), and
// per-key auth errors (401/403). A quota or overload rotation also bans
// that key on the shared keyrotation.Banlist for a configurable duration,
// so a deployment that orders a free-tier key before a paid one fails over
// on the very next request instead of re-discovering the same failure from
// scratch — see isRetryable/banDuration.
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/shared"

	llm "github.com/archer-developer/miranda-llm"
	"github.com/archer-developer/miranda-llm/keyrotation"
)

const defaultMaxTokens = 4096

// apiBaseURL overrides every anthropic.Client's base URL. It's a var, not a
// const, so tests can point it at an httptest.Server instead of the real
// Anthropic API — mirrors gemini.apiBaseURL. Empty (the default) means "use
// the SDK's real Anthropic API endpoint".
var apiBaseURL string

// codeExecutionCaller identifies this provider's code execution tool
// version in the AllowedCallers list of the web search/fetch tools, so
// code running in Anthropic's sandbox is permitted to invoke them as
// helpers (e.g. fetch a page, then parse/compute over it in code) instead
// of only being callable directly by the model.
const codeExecutionCaller = "code_execution_20260521"

// defaultStructuredToolName is used by Structured when the caller's
// StructuredRequest.SchemaName is empty.
const defaultStructuredToolName = "structured_output"

// llm.ChatRequest.Temperature/llm.StructuredRequest.Temperature are
// deliberately never forwarded to the API from this provider (see Chat's
// and Structured's own doc comments) — there is no toParamOpt-style
// converter here on purpose, unlike gemini/openaicompat.

// ToolsConfig toggles which of Claude's native server-side tools are sent
// on every request from this provider. All default to false (opt-in) since
// they let the model reach the open internet or run arbitrary code — leave
// disabled for providers that shouldn't have that reach.
type ToolsConfig struct {
	// WebSearch lets the model search the live web.
	WebSearch bool
	// WebFetch lets the model retrieve a specific URL's content directly.
	WebFetch bool
	// CodeExecution runs Python/bash in Anthropic's sandbox. When WebSearch
	// or WebFetch are also enabled, the sandbox is allowed to call them as
	// helpers (e.g. fetch a page, then parse/compute over it in code) — see
	// the AllowedCallers wiring in nativeTools.
	CodeExecution bool
}

// RotationConfig tunes this package's key-rotation. The zero value falls
// back to keyrotation's built-in defaults (1 cycle, no cooldown) plus this
// package's own ban defaults (30min/30min/10s). Field-for-field identical
// (same names, types, order) to gemini.RotationConfig/
// openaicompat.RotationConfig on purpose — see gemini.RotationConfig's doc
// comment for why.
type RotationConfig struct {
	CooldownSeconds int
	MaxRetryCycles  int
	// QuotaBanMinutes is how long a key that hit a quota/rate-limit error
	// (429) is skipped by every subsequent Chat/Structured call, not just
	// retried within the current one. <= 0 uses defaultQuotaBan.
	QuotaBanMinutes int
	// OverloadBanMinutes is the same idea for a key that looks like it hit
	// Anthropic's own "overloaded" state (see isOverloaded), or timed out
	// waiting for a first response (see attempt). <= 0 uses
	// defaultOverloadBan.
	OverloadBanMinutes int
	// OverloadTimeoutSeconds bounds how long attempt waits for the FIRST
	// streamed event before treating the key as overloaded. <= 0 uses
	// defaultOverloadTimeout.
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

// Provider is an llm.Provider (and llm.StructuredProvider) backed by the
// native Anthropic Messages API. Like gemini.Provider, it holds one
// anthropic.Client per configured key and rotates across them on a quota or
// per-key auth error — see pump/attempt below.
type Provider struct {
	name      string
	model     string
	maxTokens int64
	clients   []anthropic.Client // one per resolved API key, in configured order
	tools     ToolsConfig
	rotation  RotationConfig
	// bans is shared across every Chat/Structured call this Provider ever
	// makes (constructed once in New, never per-call) — mirrors
	// gemini.Provider.bans; see keyrotation.Banlist.
	bans   *keyrotation.Banlist
	tracer llm.Tracer
	logger *slog.Logger
}

// New builds a Provider named name for the given Claude model, resolving
// apiKeyEnvs to actual key values from the process environment (never
// stored in the caller's own config file) and building one anthropic.Client
// per resolved key up front — mirrors gemini.New. Fails fast if none of
// apiKeyEnvs resolve to a non-empty value, since every subsequent
// Chat/Structured call would otherwise fail identically anyway. tools
// enables Claude's own server-executed web search/fetch/code-execution
// tools on every request from this provider (see ToolsConfig).
func New(name, model string, apiKeyEnvs []string, tools ToolsConfig, rotation RotationConfig, logger *slog.Logger) (*Provider, error) {
	if logger == nil {
		logger = slog.Default()
	}

	var clients []anthropic.Client
	for _, env := range apiKeyEnvs {
		key := os.Getenv(env)
		if key == "" {
			continue
		}
		// WithMaxRetries(0): the SDK's own default (2) retries 429/5xx
		// transparently, on the SAME key, before an error ever reaches
		// attempt/isRetryable — silently defeating the "don't rotate keys
		// on a 5xx" fix below (isRetryable never even sees most 5xx
		// responses, since the SDK already retried and only surfaces the
		// error after its own retries are exhausted) and making
		// keyrotation.Run's own retry/rotation logic redundant with a
		// second, hidden retry layer operating on different rules. This
		// package's own keyrotation.Run + isRetryable is the single,
		// explicit place that decides what gets retried and how — mirrors
		// gemini.Provider, whose SDK has no such built-in retry to begin
		// with.
		opts := []option.RequestOption{option.WithAPIKey(key), option.WithMaxRetries(0)}
		if apiBaseURL != "" {
			opts = append(opts, option.WithBaseURL(apiBaseURL))
		}
		clients = append(clients, anthropic.NewClient(opts...))
	}
	if len(clients) == 0 {
		return nil, fmt.Errorf("anthropic: none of the configured api_key_envs %v are set", apiKeyEnvs)
	}

	return &Provider{
		name:      name,
		model:     model,
		maxTokens: defaultMaxTokens,
		clients:   clients,
		tools:     tools,
		rotation:  rotation,
		bans:      keyrotation.NewBanlist(),
		logger:    logger,
	}, nil
}

func (p *Provider) Name() string { return p.name }

// SetTracer wires up request/response tracing (see the llmtrace package) —
// optional, and left unset (nil) by default so tests and any caller that
// doesn't care about it don't need to touch this. This Provider serializes
// its own actual API request/response (see trace), so the trace reflects
// exactly what went out — including native tools this Provider itself
// added (see nativeTools) that never appear in the caller's own req.Tools.
func (p *Provider) SetTracer(t llm.Tracer) {
	p.tracer = t
}

// Chat implements llm.Provider by streaming a Messages API response,
// rotating across configured API keys on a retryable error (see
// pump/attempt).
//
// req.Temperature is deliberately never forwarded to the API — see
// Structured's own doc comment for the full production incident this
// mirrors (a current Claude model hard-rejects the `temperature` field
// outright, "deprecated for this model", for any value at all, not just an
// out-of-range one). Found here specifically (2026-08-27) via
// extraction.OCR: it always sets ChatRequest.Temperature explicitly
// (0.1, to keep Gemini from wandering off mid-transcription — see
// ocrTemperature's own doc comment in that package) with no way to know at
// call time which provider will actually handle the request, so every OCR
// call that escalated from Gemini to Claude 400'd outright. Structured's
// fix (stop substituting a default when nil) only ever worked because no
// caller happened to set an explicit value — this is the same bug, just
// finally hit by a caller that does.
func (p *Provider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	system, messages := toAnthropicMessages(req.Messages)

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(p.model),
		MaxTokens: p.maxTokens,
		System:    system,
		Messages:  messages,
		Tools:     p.buildTools(req.Tools),
	}

	out := make(chan llm.StreamChunk)
	go p.pump(ctx, params, out)
	return out, nil
}

// pump rotates across p.clients (one per configured API key) via
// keyrotation.Run, classifying each error via p.classify — quota, overload
// (Anthropic's own overloaded_error, or a first-response timeout), and
// per-key auth failures all rotate to the next key; quota/overload also ban
// that key on p.bans for future calls — mirrors gemini.Provider.pump.
func (p *Provider) pump(ctx context.Context, params anthropic.MessageNewParams, out chan<- llm.StreamChunk) {
	defer close(out)

	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}
	err := keyrotation.Run(ctx, p.logger, "anthropic", len(p.clients), rotCfg, p.bans, p.classify,
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
// the one attempt already sent. isRetryable naturally treats it as
// non-retryable too (it isn't an *anthropic.Error), so keyrotation.Run
// stops rotating and returns it straight back to pump. Mirrors gemini's own
// sentinel of the same name.
var errAttemptWritten = errors.New("anthropic: attempt already wrote its terminal error")

// errFirstResponseTimeout means attempt gave up waiting for the first
// streamed event within RotationConfig.OverloadTimeoutSeconds, before
// anything was forwarded to the caller — mirrors gemini's sentinel of the
// same name/reasoning.
var errFirstResponseTimeout = errors.New("anthropic: no response within first-response timeout")

// attempt makes exactly one Messages.NewStreaming call against one client
// (one API key), forwarding TextDelta chunks to out live as they stream in
// and using the SDK's built-in Message.Accumulate to reconstruct the full
// response, so tool_use blocks (which the API streams as fragmented
// partial-JSON deltas) only need to be read once, fully assembled, at the
// end of the stream. That same accumulated message — including any
// server-side tool_use/tool_result blocks Anthropic resolved on its own
// (web search/fetch, code execution) — is what gets traced, so the trace
// reflects exactly what the API did, not just the subset a caller's own
// tool loop sees.
//
// Once ANY text has been forwarded to out for this attempt, a later error
// on the SAME attempt must not be retried even if isRetryable(err) would
// otherwise say yes — rotating to another key re-runs the whole request
// from scratch, and a caller that live-forwards chunks as they arrive has
// no way to "unsay" one — mirrors gemini.Provider.attempt's own
// forwarded-then-error rule.
//
// Returns nil once the response completes cleanly (having already written
// Done to out). Returns errAttemptWritten in the two cases described above
// (forwarded-then-error, or a non-retryable error) — both already wrote
// their own Err chunk to out, so keyrotation.Run/pump must not retry or
// double-report. Returns the raw streamErr (or errFirstResponseTimeout —
// see below) only when nothing has been forwarded yet AND isRetryable of it
// is true, so keyrotation.Run's own classify check agrees and tries the
// next key.
//
// Before anything is forwarded, attempt also bounds how long it waits for
// the very FIRST event via RotationConfig.OverloadTimeoutSeconds — mirrors
// gemini.Provider.attempt's own first-response watchdog; see its doc
// comment for the full reasoning (only the first event, never a later one,
// since the forwarded-then-error rule above already forbids retrying past
// that point anyway).
func (p *Provider) attempt(ctx context.Context, keyIndex int, params anthropic.MessageNewParams, out chan<- llm.StreamChunk) error {
	attemptCtx, cancelAttempt := context.WithCancel(ctx)
	defer cancelAttempt()

	// NewStreaming performs the actual HTTP request eagerly (before this
	// call returns), so a 429/401/403/5xx from the initial request surfaces
	// via stream.Err() below exactly like a mid-stream error would — there
	// is no separate "did the request even go out" error to check first.
	stream := p.clients[keyIndex].Messages.NewStreaming(attemptCtx, params)

	var message anthropic.Message
	forwarded := false

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
				// it promptly, but this attempt must not touch stream
				// again (Current/Err/a second Next call) without waiting
				// for that goroutine first, which it never does: it just
				// returns, leaving the abandoned call to finish and be
				// garbage collected on its own.
				cancelAttempt()
				if ctx.Err() != nil {
					return ctx.Err()
				}
				p.logger.Warn("anthropic: no response within first-response timeout, treating key as overloaded", "provider", p.name, "key_index", keyIndex, "timeout", p.rotation.overloadTimeout())
				return errFirstResponseTimeout
			}
		} else {
			hasNext = stream.Next()
		}
		if !hasNext {
			break
		}
		event := stream.Current()
		if err := message.Accumulate(event); err != nil {
			wrapped := fmt.Errorf("anthropic: accumulate: %w", err)
			p.logger.Error("anthropic: request failed", "provider", p.name, "key_index", keyIndex, "error", err)
			p.trace(ctx, params, nil, wrapped)
			out <- llm.StreamChunk{Err: wrapped}
			return errAttemptWritten
		}
		if event.Type == "content_block_delta" && event.Delta.Text != "" {
			forwarded = true
			out <- llm.StreamChunk{TextDelta: event.Delta.Text}
		}
	}
	if streamErr := stream.Err(); streamErr != nil {
		if forwarded {
			p.logger.Error("anthropic: request failed mid-stream after partial output, not retrying", "provider", p.name, "key_index", keyIndex, "error", streamErr)
			wrapped := fmt.Errorf("anthropic: stream: %w", streamErr)
			p.trace(ctx, params, nil, wrapped)
			out <- llm.StreamChunk{Err: wrapped}
			return errAttemptWritten
		}
		if isRetryable(streamErr) {
			p.logKeyRotate(keyIndex, streamErr)
			return streamErr
		}
		p.logger.Error("anthropic: request failed", "provider", p.name, "key_index", keyIndex, "error", streamErr)
		wrapped := fmt.Errorf("anthropic: stream: %w", streamErr)
		p.trace(ctx, params, nil, wrapped)
		out <- llm.StreamChunk{Err: wrapped}
		return errAttemptWritten
	}

	for _, block := range message.Content {
		if block.Type == "tool_use" {
			out <- llm.StreamChunk{ToolCall: &llm.ToolCall{ID: block.ID, Name: block.Name, Arguments: string(block.Input)}}
		}
	}

	p.trace(ctx, params, &message, nil)
	out <- llm.StreamChunk{Done: true}
	return nil
}

// isOverloaded reports whether err is Anthropic's specific "overloaded"
// shape — error.type == "overloaded_error" in the response body (normally
// delivered as HTTP 529, a non-standard code, but Type() is checked instead
// of StatusCode since it's the documented, semantic signal — see
// apierror.Error.Type's doc comment). Only this specific shape is treated
// as retryable/bannable; see isRetryable's doc comment for why a bare 5xx
// otherwise still fails immediately.
func isOverloaded(err error) bool {
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		return false
	}
	return apiErr.Type() == shared.ErrorTypeOverloadedError
}

// isAuthFailure reports whether err is a per-key auth failure (HTTP 401
// Unauthorized / 403 Forbidden) — a revoked, disabled, or otherwise invalid
// key. Used only to pick a more pointed log message (see logKeyRotate):
// unlike quota/overload, this condition never clears on its own, so it is
// deliberately never banned (see banDuration) — a fixed-duration ban would
// just delay, not fix, the same failure repeating on every future call.
func isAuthFailure(err error) bool {
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		return false
	}
	return apiErr.StatusCode == http.StatusUnauthorized || apiErr.StatusCode == http.StatusForbidden
}

// logKeyRotate logs the WARN for a retryable per-attempt error right before
// keyrotation.Run moves on to the next key — mirrors gemini.Provider's own.
// An auth failure gets a distinct, more pointed message: unlike quota/
// overload (banned — see banDuration), an invalid/revoked key stays
// invalid and will be retried and fail again on EVERY future call until a
// human fixes or removes it from api_key_envs.
func (p *Provider) logKeyRotate(keyIndex int, err error) {
	if isAuthFailure(err) {
		p.logger.Warn("anthropic: key auth failure (401/403) — key looks invalid or revoked and needs manual attention; not banned, so it will be retried again on every future call until fixed", "provider", p.name, "key_index", keyIndex, "error", err)
		return
	}
	p.logger.Warn("anthropic: key error, trying next key", "provider", p.name, "key_index", keyIndex, "error", err)
}

// isRetryable reports whether err is worth rotating to the next key for —
// mirrors gemini.isRetryable's reasoning exactly, just against Anthropic's
// own error shape (*anthropic.Error, an alias for the SDK's internal
// apierror.Error, carrying StatusCode and Type()):
//   - a quota/rate-limit error (HTTP 429) — quota is tracked per key, so a
//     different configured key plausibly has budget left.
//   - Anthropic's own "overloaded" shape (see isOverloaded) or
//     errFirstResponseTimeout — a different key (especially a
//     differently-tiered one) may not be hitting the same overloaded
//     capacity pool. This used to be excluded entirely (see the
//     production-incident note below) — it's back specifically for this
//     shape, gated by keyrotation.Banlist: an overloaded key gets banned
//     for RotationConfig.OverloadBanMinutes when it happens, so a run of
//     several configured keys all hitting the same genuine outage still
//     only pays the multiplied latency once, not on every subsequent
//     request until the outage clears.
//   - a per-key auth failure (HTTP 401 Unauthorized / 403 Forbidden) — a
//     property of THAT key specifically, not of the request.
//
// Any OTHER server-side error (a bare 5xx that isn't Anthropic's specific
// overloaded_error shape) is deliberately still NOT retried: the backend
// itself is struggling, so every configured key hits the same wall and
// cycling through them just repeats the identical failure — see
// gemini.isRetryable's doc comment for the production incident
// (2026-08-27) this reasoning is shared with.
func isRetryable(err error) bool {
	if errors.Is(err, errFirstResponseTimeout) || isOverloaded(err) {
		return true
	}
	var apiErr *anthropic.Error
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
	var apiErr *anthropic.Error
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

// Structured implements llm.StructuredProvider. The Messages API has no
// native "response_format: json_schema" equivalent (unlike Gemini and
// OpenAI-compatible backends), so this forces the model to call a single,
// synthetic tool whose input schema IS req.Schema — the standard
// workaround for schema-constrained output on this API — via ToolChoice,
// then reads the tool call's Input back as the result. One non-streaming
// call (Messages.New), since a forced single tool call has no meaningful
// streaming granularity to forward.
//
// Unlike Gemini/OpenAI-compatible Structured, req.Temperature is never
// forwarded to the API at all here, not even when the caller sets one
// explicitly (see llm.StructuredRequest.Temperature's doc comment for why
// other providers do substitute a "low, for determinism" default when
// nil). Observed in production (2026-08-09): a request that set
// Temperature at all — including the low default this provider used to
// substitute internally — got a 400 from a current Claude model
// ("`temperature` is deprecated for this model"), which made every
// Structured call against that model fail outright, escalation included.
// Since the parameter is being rejected wholesale rather than validated by
// range, there's no safe non-nil value left to fall back to here — the
// field is dropped unconditionally instead (see Chat's own doc comment for
// the sibling incident this same reasoning applies to: an even simpler
// initial fix here that merely stopped substituting a default, rather than
// dropping the field outright, only ever worked because no caller happened
// to set an explicit non-nil Temperature — extraction.OCR does, and hit
// this exact 400 the moment its Chat call escalated to Claude). A caller
// that still wants deterministic output from Claude has no lever for that
// via this field anymore — accept whatever sampling behavior the model's
// own default is.
func (p *Provider) Structured(ctx context.Context, req llm.StructuredRequest) (json.RawMessage, error) {
	system, messages := toAnthropicMessages(req.Messages)

	toolName := req.SchemaName
	if toolName == "" {
		toolName = defaultStructuredToolName
	}
	properties := req.Schema["properties"]
	if properties == nil {
		properties = map[string]any{}
	}
	tool := anthropic.ToolUnionParam{
		OfTool: &anthropic.ToolParam{
			Name: toolName,
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: properties,
				Required:   requiredFields(req.Schema),
			},
		},
	}

	params := anthropic.MessageNewParams{
		Model:      anthropic.Model(p.model),
		MaxTokens:  p.maxTokens,
		System:     system,
		Messages:   messages,
		Tools:      []anthropic.ToolUnionParam{tool},
		ToolChoice: anthropic.ToolChoiceParamOfTool(toolName),
	}

	// Rotates across configured API keys exactly like Chat's pump/attempt,
	// via the same keyrotation.Run + p.classify (sharing the same p.bans) —
	// one non-streaming call per attempted key (Messages.New), since a
	// forced single tool call has no meaningful streaming granularity to
	// forward, and no first-response-timeout check for the same reason
	// Structured skips it in gemini: the request either returns within ctx
	// or it doesn't.
	var result json.RawMessage
	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}
	err := keyrotation.Run(ctx, p.logger, "anthropic-structured", len(p.clients), rotCfg, p.bans, p.classify,
		func(ctx context.Context, keyIndex int) error {
			message, err := p.clients[keyIndex].Messages.New(ctx, params)
			if err != nil {
				if isRetryable(err) {
					p.logKeyRotate(keyIndex, err)
				} else {
					p.logger.Error("anthropic: request failed", "provider", p.name, "key_index", keyIndex, "error", err)
				}
				p.trace(ctx, params, nil, fmt.Errorf("anthropic: structured: %w", err))
				return err
			}
			p.trace(ctx, params, message, nil)

			for _, block := range message.Content {
				if block.Type == "tool_use" && block.Name == toolName {
					result = json.RawMessage(block.Input)
					return nil
				}
			}
			// Not wrapped with an "anthropic: structured:" prefix here —
			// the caller below adds it once, uniformly, for every failure
			// path (this one and a raw API error alike) so the message
			// never repeats the prefix twice.
			return fmt.Errorf("model did not call the forced tool %q", toolName)
		},
	)
	if err != nil {
		return nil, fmt.Errorf("anthropic: structured: %w", err)
	}
	return result, nil
}

// trace serializes exactly what was sent (params, the same value passed to
// the SDK's own request-building code) and what came back (the fully
// accumulated message, nil on error) to p.tracer, if one is configured.
// Shared by Chat's pump and Structured.
func (p *Provider) trace(ctx context.Context, params anthropic.MessageNewParams, message *anthropic.Message, err error) {
	if p.tracer == nil {
		return
	}
	reqJSON, marshalErr := json.MarshalIndent(params, "", "  ")
	if marshalErr != nil {
		reqJSON = []byte(fmt.Sprintf("(failed to marshal request: %v)", marshalErr))
	}
	var respJSON []byte
	if message != nil {
		if respJSON, marshalErr = json.MarshalIndent(message, "", "  "); marshalErr != nil {
			respJSON = []byte(fmt.Sprintf("(failed to marshal response: %v)", marshalErr))
		}
	}
	p.tracer.Trace(ctx, p.name, string(reqJSON), string(respJSON), err)
}

// toAnthropicMessages splits llm.Message history into Anthropic's separate
// top-level system prompt and turn-by-turn message list. Tool results are
// represented as user-role messages containing a tool_result block, per the
// Anthropic Messages API convention (there is no dedicated "tool" role).
//
// A cache_control breakpoint is placed on the FIRST system block, not the
// last. This is a deliberate convention with the caller, not an arbitrary
// choice: a caller that sends more than one RoleSystem message is expected
// to put the stable, byte-identical-across-turns content first (persona,
// standing instructions, slow-changing context like remembered facts) and
// any per-turn-volatile content (e.g. the current time, which changes on
// essentially every turn) in later blocks — see
// docs/adr/system-prompt-caching.md in the miranda repo for the caller-side
// design this exists for. Marking the first block means Anthropic's
// prefix-based prompt cache still hits on every subsequent turn even though
// a later, deliberately-unmarked volatile block differs from turn to turn —
// marking the last block (the old behavior) would instead put the
// breakpoint on the one block guaranteed never to match, defeating caching
// entirely for a multi-block caller. A caller that only ever sends a single
// system message (the common case, and every caller before this) is
// unaffected: that one block is both first and last.
func toAnthropicMessages(msgs []llm.Message) ([]anthropic.TextBlockParam, []anthropic.MessageParam) {
	var system []anthropic.TextBlockParam
	var out []anthropic.MessageParam

	for _, m := range msgs {
		switch m.Role {
		case llm.RoleSystem:
			system = append(system, anthropic.TextBlockParam{Text: m.Content})

		case llm.RoleUser:
			if len(m.Parts) > 0 {
				// Multi-modal user message: text first (so the model reads
				// the question before the image), then each image block.
				// Providers that don't support vision only get the text block.
				blocks := make([]anthropic.ContentBlockParamUnion, 0, 1+len(m.Parts))
				if m.Content != "" {
					blocks = append(blocks, anthropic.NewTextBlock(m.Content))
				}
				for _, p := range m.Parts {
					if p.ImageBase64 == "" {
						continue
					}
					if p.MIMEType == "application/pdf" {
						// Claude rejects "application/pdf" from an image
						// block outright (media_type must be one of
						// image/jpeg|png|gif|webp) — a PDF needs its own
						// document block instead. Found in production
						// (2026-08-27): OCR on a PDF sends this same
						// ContentPart to Gemini (whose inlineData.mimeType
						// happily accepts application/pdf) and, on OCR
						// escalation, to Claude — which then hard-failed
						// every single time with a 400, making the
						// escalation path dead code for any PDF input.
						blocks = append(blocks, anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
							Data: p.ImageBase64,
						}))
						continue
					}
					blocks = append(blocks, anthropic.NewImageBlockBase64(p.MIMEType, p.ImageBase64))
				}
				// Anthropic requires at least one content block; guard against
				// a Parts list where every ImageBase64 was empty combined
				// with an empty Content.
				if len(blocks) == 0 {
					blocks = append(blocks, anthropic.NewTextBlock(""))
				}
				out = append(out, anthropic.NewUserMessage(blocks...))
			} else {
				out = append(out, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
			}

		case llm.RoleTool:
			out = append(out, anthropic.NewUserMessage(anthropic.NewToolResultBlock(m.ToolCallID, m.Content, false)))

		case llm.RoleAssistant:
			var blocks []anthropic.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				// input defaults to an empty object, not nil: Anthropic's API
				// requires a tool_use block's "input" field to be present
				// (even `{}` for an argumentless call), and a nil input
				// serializes with the field omitted entirely, which the API
				// rejects outright. Empty or malformed Arguments (the latter
				// best-effort — malformed JSON here shouldn't fail the whole
				// request) both fall back to this same default.
				input := any(map[string]any{})
				if tc.Arguments != "" {
					_ = json.Unmarshal([]byte(tc.Arguments), &input)
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, input, tc.Name))
			}
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		}
	}

	// Mark the first system block as a cache breakpoint — see this
	// function's doc comment for why first, not last. Anthropic renders the
	// prompt in the order tools → system → messages, so this checkpoint
	// covers the stable prefix (tools, already separately breakpointed by
	// buildTools, plus this first system block) and prevents it from being
	// re-priced on every subsequent turn, regardless of what any later
	// system block contains.
	if len(system) > 0 {
		system[0].CacheControl = anthropic.NewCacheControlEphemeralParam()
	}

	return system, out
}

// buildTools assembles the full tool list sent on every request: the
// model-invoked custom tools first (executed by the caller's own tool
// loop), followed by this provider's enabled native Anthropic tools
// (executed by Anthropic itself; see nativeTools). Anthropic's render order
// is tools → system → messages, and both halves are identical on every
// turn of a conversation, so a single cache breakpoint on the very last
// entry caches the whole list as a shared prefix and avoids re-pricing it
// on subsequent turns.
func (p *Provider) buildTools(custom []llm.ToolDef) []anthropic.ToolUnionParam {
	out := toAnthropicTools(custom)
	out = append(out, p.nativeTools()...)
	if len(out) == 0 {
		return nil
	}
	setCacheBreakpoint(&out[len(out)-1])
	return out
}

func toAnthropicTools(tools []llm.ToolDef) []anthropic.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		// Properties must never be a nil interface: anthropic.ToolInputSchemaParam
		// has no fields of its own besides Properties/Required/Type, so a tool
		// whose schema omits "properties" entirely (e.g. a tool that takes no
		// arguments, like {"type":"object","additionalProperties":false})
		// would otherwise leave every field at its Go zero value — the SDK's
		// `omitzero` tag on ToolParam.InputSchema then treats the whole struct
		// as absent and drops "input_schema" from the request entirely, which
		// Anthropic rejects with "input_schema: Field required". An empty map
		// is a non-nil interface value, so the struct is never all-zero.
		properties := t.Parameters["properties"]
		if properties == nil {
			properties = map[string]any{}
		}
		out = append(out, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: properties,
					Required:   requiredFields(t.Parameters),
				},
			},
		})
	}
	return out
}

// nativeTools builds the Anthropic-executed tools enabled on this provider
// (ToolsConfig) — unlike everything in toAnthropicTools, the model's use of
// these never round-trips through the caller's own tool-execution loop;
// Anthropic resolves them server-side within the same streamed response.
func (p *Provider) nativeTools() []anthropic.ToolUnionParam {
	var out []anthropic.ToolUnionParam

	// AllowedCallers defaults to "direct" (the model calling the tool
	// itself). Adding the code execution caller here is what lets code
	// running in Anthropic's sandbox invoke Claude's OWN native web
	// search/fetch tools (above) as a helper in one native round trip —
	// e.g. fetch a page, then parse or compute over it in code — instead
	// of only being callable by the model directly.
	callers := []string{"direct"}
	if p.tools.CodeExecution {
		callers = append(callers, codeExecutionCaller)
	}

	if p.tools.WebSearch {
		out = append(out, anthropic.ToolUnionParam{
			OfWebSearchTool20260318: &anthropic.WebSearchTool20260318Param{
				AllowedCallers: callers,
			},
		})
	}
	if p.tools.WebFetch {
		out = append(out, anthropic.ToolUnionParam{
			OfWebFetchTool20260318: &anthropic.WebFetchTool20260318Param{
				AllowedCallers: callers,
			},
		})
	}
	if p.tools.CodeExecution {
		out = append(out, anthropic.ToolUnionParam{
			OfCodeExecutionTool20260521: &anthropic.CodeExecutionTool20260521Param{},
		})
	}
	return out
}

// setCacheBreakpoint marks tool as a prompt-cache boundary. The cache
// control field lives on whichever concrete tool struct is populated
// inside the union, so this has to switch on it explicitly — there is no
// single settable field on ToolUnionParam itself.
func setCacheBreakpoint(tool *anthropic.ToolUnionParam) {
	switch {
	case tool.OfTool != nil:
		tool.OfTool.CacheControl = anthropic.NewCacheControlEphemeralParam()
	case tool.OfWebSearchTool20260318 != nil:
		tool.OfWebSearchTool20260318.CacheControl = anthropic.NewCacheControlEphemeralParam()
	case tool.OfWebFetchTool20260318 != nil:
		tool.OfWebFetchTool20260318.CacheControl = anthropic.NewCacheControlEphemeralParam()
	case tool.OfCodeExecutionTool20260521 != nil:
		tool.OfCodeExecutionTool20260521.CacheControl = anthropic.NewCacheControlEphemeralParam()
	}
}

func requiredFields(parameters map[string]any) []string {
	raw, ok := parameters["required"].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, r := range raw {
		if s, ok := r.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
