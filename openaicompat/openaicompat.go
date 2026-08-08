// Package openaicompat implements llm.Provider (and llm.StructuredProvider)
// on top of the official github.com/openai/openai-go/v2 SDK, pointed at any
// OpenAI Chat Completions compatible backend (OpenAI itself, or a
// local/self-hosted server such as Ollama, vLLM, LM Studio, or a hosted
// router like OpenRouter) via option.WithBaseURL. This is the single client
// used for every provider that isn't native Anthropic or native Gemini.
package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/openai/openai-go/v2/shared"

	llm "github.com/archer-developer/miranda-llm"
)

// defaultStructuredSchemaName is used by Structured when the caller's
// StructuredRequest.SchemaName is empty.
const defaultStructuredSchemaName = "structured_output"

// Provider is an llm.Provider (and llm.StructuredProvider) backed by any
// OpenAI-compatible Chat Completions endpoint.
type Provider struct {
	name   string
	model  string
	client openai.Client
	tracer llm.Tracer
}

// New builds a Provider named name, targeting model on the endpoint at
// baseURL (empty baseURL means the real OpenAI API). apiKey may be empty for
// local backends that don't require authentication.
func New(name, baseURL, model, apiKey string) *Provider {
	opts := []option.RequestOption{}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	return &Provider{name: name, model: model, client: openai.NewClient(opts...)}
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

// Chat implements llm.Provider by streaming a Chat Completions response and
// re-assembling incrementally-streamed tool-call fragments (the API sends a
// function name/arguments split across many chunks, keyed by tool_call
// index) into whole llm.ToolCall values before they're handed to the caller.
func (p *Provider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(p.model),
		Messages: toOpenAIMessages(req.Messages),
		Tools:    toOpenAITools(req.Tools),
	}

	stream := p.client.Chat.Completions.NewStreaming(ctx, params)

	out := make(chan llm.StreamChunk)
	go p.pump(ctx, params, stream, out)
	return out, nil
}

// pendingToolCall accumulates a tool call's fragments as they stream in.
type pendingToolCall struct {
	id, name, arguments string
}

// pump forwards text deltas as they arrive and, via the SDK's own
// ChatCompletionAccumulator, reconstructs the full response for tracing —
// mirroring what the manually-accumulated pendingToolCall fragments below
// resolve to, but as the SDK's own response type so the trace reflects
// exactly what the API returned.
func (p *Provider) pump(ctx context.Context, params openai.ChatCompletionNewParams, stream *ssestream.Stream[openai.ChatCompletionChunk], out chan<- llm.StreamChunk) {
	defer close(out)

	var acc openai.ChatCompletionAccumulator

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

	if err := stream.Err(); err != nil {
		wrapped := fmt.Errorf("openaicompat: stream: %w", err)
		out <- llm.StreamChunk{Err: wrapped}
		p.trace(ctx, params, nil, wrapped)
		return
	}

	for _, idx := range order {
		pc := pending[idx]
		out <- llm.StreamChunk{ToolCall: &llm.ToolCall{ID: pc.id, Name: pc.name, Arguments: pc.arguments}}
	}

	p.trace(ctx, params, &acc.ChatCompletion, nil)
	out <- llm.StreamChunk{Done: true}
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
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		wrapped := fmt.Errorf("openaicompat: structured: %w", err)
		p.trace(ctx, params, nil, wrapped)
		return nil, wrapped
	}
	p.trace(ctx, params, resp, nil)

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("openaicompat: structured: no choices returned")
	}
	return json.RawMessage(resp.Choices[0].Message.Content), nil
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
