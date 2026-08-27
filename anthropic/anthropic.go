// Package anthropic implements llm.Provider (and llm.StructuredProvider) on
// top of the official github.com/anthropics/anthropic-sdk-go SDK, used
// exclusively for Claude — it gets full native support for tool use,
// streaming, and prompt caching, unlike routing Claude through an
// OpenAI-compatibility shim.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"

	llm "github.com/archer-developer/miranda-llm"
)

const defaultMaxTokens = 4096

// codeExecutionCaller identifies this provider's code execution tool
// version in the AllowedCallers list of the web search/fetch tools, so
// code running in Anthropic's sandbox is permitted to invoke them as
// helpers (e.g. fetch a page, then parse/compute over it in code) instead
// of only being callable directly by the model.
const codeExecutionCaller = "code_execution_20260521"

// defaultStructuredToolName is used by Structured when the caller's
// StructuredRequest.SchemaName is empty.
const defaultStructuredToolName = "structured_output"

// toParamOpt converts an optional float64 to the param.Opt[float64] the SDK
// wants. Returns the zero value (omitted, per the `omitzero` tag on
// MessageNewParams.Temperature) when v is nil, so the API's own default
// applies — matches Chat's existing behavior for a nil
// ChatRequest.Temperature.
func toParamOpt(v *float64) param.Opt[float64] {
	if v == nil {
		return param.Opt[float64]{}
	}
	return anthropic.Float(*v)
}

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

// Provider is an llm.Provider (and llm.StructuredProvider) backed by the
// native Anthropic Messages API.
type Provider struct {
	name      string
	model     string
	maxTokens int64
	client    anthropic.Client
	tools     ToolsConfig
	tracer    llm.Tracer
}

// New builds a Provider named name for the given Claude model. tools
// enables Claude's own server-executed web search/fetch/code-execution
// tools on every request from this provider (see ToolsConfig).
func New(name, model, apiKey string, tools ToolsConfig) *Provider {
	opts := []option.RequestOption{}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	return &Provider{
		name:      name,
		model:     model,
		maxTokens: defaultMaxTokens,
		client:    anthropic.NewClient(opts...),
		tools:     tools,
	}
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

// Chat implements llm.Provider by streaming a Messages API response.
func (p *Provider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	system, messages := toAnthropicMessages(req.Messages)

	params := anthropic.MessageNewParams{
		Model:       anthropic.Model(p.model),
		MaxTokens:   p.maxTokens,
		System:      system,
		Messages:    messages,
		Tools:       p.buildTools(req.Tools),
		Temperature: toParamOpt(req.Temperature),
	}

	stream := p.client.Messages.NewStreaming(ctx, params)

	out := make(chan llm.StreamChunk)
	go p.pump(ctx, params, stream, out)
	return out, nil
}

// pump forwards text deltas as they arrive and uses the SDK's built-in
// Message.Accumulate to reconstruct the full response, so tool_use blocks
// (which the API streams as fragmented partial-JSON deltas) only need to be
// read once, fully assembled, at the end of the stream. That same
// accumulated message — including any server-side tool_use/tool_result
// blocks Anthropic resolved on its own (web search/fetch, code execution) —
// is what gets traced, so the trace reflects exactly what the API did, not
// just the subset a caller's own tool loop sees.
func (p *Provider) pump(ctx context.Context, params anthropic.MessageNewParams, stream *ssestream.Stream[anthropic.MessageStreamEventUnion], out chan<- llm.StreamChunk) {
	defer close(out)

	var message anthropic.Message
	for stream.Next() {
		event := stream.Current()
		if err := message.Accumulate(event); err != nil {
			wrapped := fmt.Errorf("anthropic: accumulate: %w", err)
			out <- llm.StreamChunk{Err: wrapped}
			p.trace(ctx, params, nil, wrapped)
			return
		}
		if event.Type == "content_block_delta" && event.Delta.Text != "" {
			out <- llm.StreamChunk{TextDelta: event.Delta.Text}
		}
	}
	if err := stream.Err(); err != nil {
		wrapped := fmt.Errorf("anthropic: stream: %w", err)
		out <- llm.StreamChunk{Err: wrapped}
		p.trace(ctx, params, nil, wrapped)
		return
	}

	for _, block := range message.Content {
		if block.Type == "tool_use" {
			out <- llm.StreamChunk{ToolCall: &llm.ToolCall{ID: block.ID, Name: block.Name, Arguments: string(block.Input)}}
		}
	}

	p.trace(ctx, params, &message, nil)
	out <- llm.StreamChunk{Done: true}
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
// Unlike Gemini/OpenAI-compatible Structured, this deliberately does NOT
// substitute a default temperature when req.Temperature is nil (see
// llm.StructuredRequest.Temperature's doc comment for why other providers
// do) — toParamOpt passes it straight through, omitted when nil, same as
// Chat. Observed in production (2026-08-09): a request that set
// Temperature at all — including the "low, for determinism" default this
// provider used to substitute — got a 400 from a current Claude model
// ("`temperature` is deprecated for this model"), which made every
// Structured call against that model fail outright, escalation included.
// Since the parameter is being rejected wholesale rather than validated by
// range, there's no safe non-nil value left to fall back to here; only
// omitting it works. A caller that still wants deterministic structured
// output from Claude has no lever for that via this field anymore — accept
// whatever sampling behavior the model's own default is.
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
		Model:       anthropic.Model(p.model),
		MaxTokens:   p.maxTokens,
		System:      system,
		Messages:    messages,
		Tools:       []anthropic.ToolUnionParam{tool},
		ToolChoice:  anthropic.ToolChoiceParamOfTool(toolName),
		Temperature: toParamOpt(req.Temperature),
	}

	message, err := p.client.Messages.New(ctx, params)
	if err != nil {
		wrapped := fmt.Errorf("anthropic: structured: %w", err)
		p.trace(ctx, params, nil, wrapped)
		return nil, wrapped
	}
	p.trace(ctx, params, message, nil)

	for _, block := range message.Content {
		if block.Type == "tool_use" && block.Name == toolName {
			return json.RawMessage(block.Input), nil
		}
	}
	return nil, fmt.Errorf("anthropic: structured: model did not call the forced tool %q", toolName)
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
