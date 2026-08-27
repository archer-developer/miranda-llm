// Package openaicompat implements llm.Provider (and llm.StructuredProvider)
// on top of the official github.com/openai/openai-go/v2 SDK, pointed at any
// OpenAI Chat Completions compatible backend (OpenAI itself, or a
// local/self-hosted server such as Ollama, vLLM, LM Studio, or a hosted
// router like OpenRouter) via option.WithBaseURL. This is the single client
// used for every provider that isn't native Anthropic or native Gemini. See
// the gemini package for the sibling adapter this mirrors, including the
// same api_key_envs-driven key rotation (see keyrotation, isRetryable) —
// rotates on quota (429) and per-key auth errors (401/403), deliberately
// NOT on 5xx server errors.
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
// back to keyrotation's built-in defaults (1 cycle, no cooldown). Mirrors
// gemini.RotationConfig/anthropic.RotationConfig — same shape, same
// meaning.
type RotationConfig struct {
	CooldownSeconds int
	MaxRetryCycles  int
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
	tracer   llm.Tracer
	logger   *slog.Logger
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

	return &Provider{name: name, model: model, clients: clients, rotation: rotation, logger: logger}
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
// keyrotation.Run using isRetryable (quota AND per-key auth failures only —
// see its doc comment for why 5xx is deliberately excluded), mirroring
// gemini.Provider.pump/anthropic.Provider.pump.
func (p *Provider) pump(ctx context.Context, params openai.ChatCompletionNewParams, out chan<- llm.StreamChunk) {
	defer close(out)

	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}
	err := keyrotation.Run(ctx, p.logger, "openaicompat", len(p.clients), rotCfg, isRetryable,
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
func (p *Provider) attempt(ctx context.Context, keyIndex int, params openai.ChatCompletionNewParams, out chan<- llm.StreamChunk) error {
	// NewStreaming performs the actual HTTP request eagerly (before this
	// call returns), so a 429/401/403/5xx from the initial request surfaces
	// via stream.Err() below exactly like a mid-stream error would.
	stream := p.clients[keyIndex].Chat.Completions.NewStreaming(ctx, params)

	var acc openai.ChatCompletionAccumulator
	forwarded := false

	// Tool calls stream as fragments identified by index; accumulate until
	// the stream ends, then emit each as one complete llm.ToolCall.
	pending := map[int64]*pendingToolCall{}
	var order []int64

	for stream.Next() {
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
			p.logger.Warn("openaicompat: key error, trying next key", "provider", p.name, "key_index", keyIndex, "error", streamErr)
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

	p.trace(ctx, params, &acc.ChatCompletion, nil)
	out <- llm.StreamChunk{Done: true}
	return nil
}

// isRetryable reports whether err is worth rotating to the next key for —
// mirrors gemini.isRetryable/anthropic's own version, just against the
// OpenAI-compatible SDK's error shape (*openai.Error, carrying StatusCode):
//   - a quota/rate-limit error (HTTP 429).
//   - a per-key auth failure (HTTP 401 Unauthorized / 403 Forbidden).
//
// A server-side error (HTTP 5xx) is deliberately NOT retried — every
// configured key hits the same struggling backend, so cycling through them
// just repeats the identical failure. See gemini.isRetryable's doc comment
// for the production incident (2026-08-27) this reasoning is shared with.
// Note this is a generic fallback for whatever backend baseURL points at
// (OpenAI itself, or a self-hosted server) — a local backend's own error
// shape may not populate StatusCode the same way OpenAI's does, in which
// case this simply returns false and the first failure surfaces
// immediately, same as today.
func isRetryable(err error) bool {
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
	// via the same keyrotation.Run + isRetryable.
	var result json.RawMessage
	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}
	err := keyrotation.Run(ctx, p.logger, "openaicompat-structured", len(p.clients), rotCfg, isRetryable,
		func(ctx context.Context, keyIndex int) error {
			resp, err := p.clients[keyIndex].Chat.Completions.New(ctx, params)
			if err != nil {
				if isRetryable(err) {
					p.logger.Warn("openaicompat: key error, trying next key", "provider", p.name, "key_index", keyIndex, "error", err)
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
