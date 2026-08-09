// Package llm defines the provider-agnostic interfaces every backend
// (Gemini, Anthropic, any OpenAI-compatible endpoint) implements, so callers
// — router.Router, and any service that talks to a Provider directly —
// never depend on a specific vendor SDK.
//
// This package was extracted from miranda's internal/llm (see that repo's
// git history for the original, in-app version) into its own module so it
// can be shared across every Miranda microservice, not copy-pasted into
// each one. See this repo's CLAUDE.md for the extraction rationale, what
// changed in the process, and — importantly — that miranda itself has not
// been migrated onto this module yet; that migration is a separate,
// still-open task.
package llm

import (
	"context"
	"encoding/json"
)

// Role identifies who authored a Message in a conversation.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a single function/tool invocation requested by the model.
type ToolCall struct {
	ID        string // provider-assigned id, echoed back in the tool result message
	Name      string
	Arguments string // raw JSON arguments, as emitted by the model
	// ProviderMetadata is an opaque, provider-specific blob (base64-encoded
	// if the provider's own data is binary) that must be echoed back
	// verbatim on a later turn for providers that require it — e.g.
	// Gemini's "thought signature" on a function-call part (see
	// gemini's toLLMToolCall/toGeminiContents): omitting it on a later
	// turn's replayed tool call is a hard 400 from the API, not just
	// degraded quality. Empty and unused by providers that don't need this
	// (Anthropic, OpenAI-compat).
	ProviderMetadata string
}

// ContentPart is one non-text content block within a user message —
// currently only image data for vision-capable models. Plain text turns
// always use Message.Content and leave Parts nil; Parts is only non-empty
// when the caller attached an image (e.g. a photographed document) that can
// be inlined for vision.
type ContentPart struct {
	// ImageBase64 is the base64-encoded image bytes. Set MIMEType to the
	// image's actual MIME type (e.g. "image/jpeg", "image/png",
	// "image/gif", "image/webp") so providers know how to decode it.
	ImageBase64 string
	// MIMEType is the MIME type of ImageBase64's data.
	MIMEType string
}

// Message is one turn in the conversation sent to/received from a Provider.
type Message struct {
	Role Role
	// Content is the text content. Empty for an assistant message that only
	// contains tool calls.
	Content string
	// Parts carries multi-modal content blocks (image data for vision
	// models) alongside Content. Non-nil only for user messages that
	// include an image; plain text turns always use Content and leave
	// Parts nil. Providers that don't support vision ignore Parts silently.
	Parts []ContentPart
	// ToolCalls is set on assistant messages that invoke one or more tools.
	ToolCalls []ToolCall
	// ToolCallID identifies which ToolCall this message answers, and is
	// only set on Role == RoleTool messages.
	ToolCallID string
}

// ToolDef describes one tool the model is allowed to call, in JSON Schema
// form (shared shape across every provider's own SDK types).
type ToolDef struct {
	Name        string
	Description string
	Parameters  map[string]any // JSON Schema object
}

// ChatRequest is a full chat turn: history plus the tools available this turn.
type ChatRequest struct {
	Messages []Message
	Tools    []ToolDef
	// Temperature overrides the provider's own default sampling
	// temperature (0.0-1.0 for Gemini/OpenAI-compatible, 0.0-1.0 for
	// Anthropic too). nil leaves the provider's own API default in place —
	// set this explicitly when the caller's own task has a documented
	// temperature requirement (e.g. low for anything feeding a downstream
	// parser, higher for open-ended generation).
	Temperature *float64
}

// StreamChunk is one increment of a streaming chat response. A stream ends
// either with Err set, or with the channel closing after a chunk carrying
// Done == true.
type StreamChunk struct {
	// TextDelta is a piece of assistant text to append, e.g. for
	// incremental display as it becomes available.
	TextDelta string
	// ToolCall is set once when the provider finishes emitting a complete
	// tool call (providers stream tool-call arguments incrementally
	// internally, but Provider implementations buffer and emit it whole
	// here since partial JSON arguments aren't independently useful
	// downstream).
	ToolCall *ToolCall
	Done     bool
	Err      error
}

// Provider is a streaming chat backend: Gemini (gemini.Provider), native
// Anthropic (anthropic.Provider), or any OpenAI-compatible endpoint
// (openaicompat.Provider — OpenAI itself, or a local/self-hosted server).
// Implementations must be safe for concurrent use.
type Provider interface {
	// Name is the provider's configured name, used in logs and router fallback.
	Name() string
	// Chat starts a streaming chat completion. The returned channel is
	// closed once the stream ends (either after a Done chunk or an Err
	// chunk). A non-nil error return means the request could not even be
	// started (e.g. auth/connection failure).
	Chat(ctx context.Context, req ChatRequest) (<-chan StreamChunk, error)
}

// StructuredRequest is a single-shot, non-streaming request for a JSON
// value that conforms to a fixed schema — used for tasks like structured
// extraction, planning, or summarization, where the caller needs exactly
// one parseable JSON result back, not an open streamed conversation.
type StructuredRequest struct {
	Messages []Message
	// Schema is the JSON Schema (same map[string]any shape as
	// ToolDef.Parameters) the response must conform to.
	Schema map[string]any
	// SchemaName labels the schema — required by providers that implement
	// structured output via a forced single-tool call (see
	// anthropic.Provider.Structured, which uses it as that tool's name),
	// and used as a human-readable label in traces for every provider.
	// Keep it a short identifier (e.g. "medical_extraction"), not a
	// sentence — anthropic.Provider.Structured also constrains it to the
	// characters Anthropic's tool-name field accepts (letters, digits,
	// underscores, hyphens).
	SchemaName string
	// Temperature overrides the sampling temperature for this call. For
	// Gemini and OpenAI-compatible providers, nil means "use
	// defaultStructuredTemperature" (see each provider's Structured
	// implementation) rather than the provider API's own default — unlike
	// ChatRequest.Temperature, those providers' Structured calls default to
	// low-temperature already, because run-to-run consistency matters far
	// more for a result a parser will consume than the provider API's own
	// default (often tuned for open-ended chat, i.e. much higher) would
	// give: the same document, same prompt, same schema was observed to
	// non-deterministically return fully-populated arrays on one call and
	// entirely empty ones on the next at the API's default temperature.
	// Set explicitly only to deliberately raise it back up.
	//
	// anthropic.Provider is the one exception: it never substitutes a
	// default, nil or not — see its Structured doc comment for why (a
	// current Claude model rejects the temperature parameter outright, not
	// just out-of-range values, so no non-nil fallback is safe there
	// anymore).
	Temperature *float64
}

// StructuredProvider is an optional capability a Provider may additionally
// implement alongside the required streaming Chat method — not every
// backend supports schema-constrained output the same way (Gemini and
// OpenAI-compatible APIs support it natively via a response-format field;
// this package's Anthropic adapter implements it via a forced single-tool
// call, since the Messages API has no native equivalent), so callers that
// need it should type-assert for this interface (see
// router.Router.Structured, which does this once across the whole fallback
// chain) rather than requiring every Provider implementation to support it.
type StructuredProvider interface {
	// Structured runs one non-streaming call and returns the raw JSON the
	// model produced, unmarshaled by neither this package nor the
	// Provider — the caller owns decoding it into their own Go type, since
	// only the caller knows what the schema actually describes.
	Structured(ctx context.Context, req StructuredRequest) (json.RawMessage, error)
}

// Tracer receives one record per provider call: the exact request and
// response payloads, already serialized (typically to JSON) by the
// Provider that built them — only the Provider knows its own SDK's wire
// format precisely enough for the trace to be trustworthy. This is what
// lets a trace capture provider-specific specifics the shared
// ChatRequest/StreamChunk types can't represent — e.g. Anthropic's own
// server-side tools, which a Provider adds to the real request itself and
// which never appear in the ChatRequest.Tools the caller built. response is
// empty when err is non-nil. Implemented by llmtrace.Logger; a Provider
// that supports tracing accepts one via an optional SetTracer(Tracer)
// method, which router.Router forwards to on Router.SetTracer.
type Tracer interface {
	Trace(ctx context.Context, provider, request, response string, err error)
}
