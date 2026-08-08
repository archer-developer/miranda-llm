// Package llmtest provides scriptable fakes implementing llm.Provider (and
// optionally llm.StructuredProvider) plus embedding.Embedder, used by unit
// tests to exercise router fallback/escalation, tool-calling, structured
// output, and embedding-consuming code without any real network/LLM
// backend.
package llmtest

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	llm "github.com/archer-developer/miranda-llm"
)

// Response is one scripted reply the fake Provider will emit, in order, to
// a Chat call.
type Response struct {
	// Text, if non-empty, is streamed as a single StreamChunk.TextDelta.
	Text string
	// ToolCall, if set, is emitted as a StreamChunk.ToolCall after Text.
	ToolCall *llm.ToolCall
	// Err, if set, is returned as an immediate error from Chat (nothing streamed).
	Err error
	// StreamErr, if set, is delivered as a StreamChunk.Err from an
	// otherwise-successful Chat call — unlike Err, this models how every
	// real provider (anthropic, gemini, openaicompat) actually reports a
	// failure: Chat itself returns (stream, nil) and streams the error
	// later from its pump goroutine, e.g. after keyrotation exhausts every
	// configured API key across every retry cycle. Router.deliver's
	// escalation-on-error path only ever sees this shape in production, not
	// Err.
	StreamErr error
}

// StructuredResponse is one scripted reply the fake Provider will emit, in
// order, to a Structured call — kept separate from Response/Chat's own
// script so a test can script Chat and Structured calls on the same
// FakeProvider independently, matching how a real Provider's two methods
// are called for entirely different purposes (a streamed conversation turn
// vs. a one-shot schema-constrained call).
type StructuredResponse struct {
	// JSON, if Err is nil, is returned verbatim as the Structured call's result.
	JSON json.RawMessage
	// Err, if set, is returned as the Structured call's error.
	Err error
}

// FakeProvider replays a fixed script of Responses, one per call to Chat
// (and, independently, a fixed script of StructuredResponses for Structured
// calls), and records every request it received so tests can assert on
// router/caller behavior.
type FakeProvider struct {
	name string

	mu       sync.Mutex
	script   []Response
	callIdx  int
	Requests []llm.ChatRequest

	structuredScript   []StructuredResponse
	structuredCallIdx  int
	StructuredRequests []llm.StructuredRequest

	tracer llm.Tracer
}

// New creates a FakeProvider named name that will return script[i] on the
// i-th call to Chat. Calling Chat more times than len(script) panics, since
// that indicates the test under-specified expected behavior. Use
// WithStructured to additionally script Structured calls.
func New(name string, script ...Response) *FakeProvider {
	return &FakeProvider{name: name, script: script}
}

// WithStructured scripts script[i] as the response to the i-th call to
// Structured, returning the same *FakeProvider for chaining onto New.
// Calling Structured more times than len(script) panics, mirroring Chat's
// own over-call guard.
func (f *FakeProvider) WithStructured(script ...StructuredResponse) *FakeProvider {
	f.structuredScript = script
	return f
}

func (f *FakeProvider) Name() string { return f.name }

// SetTracer implements the same optional tracing hook the real providers
// expose, so tests can exercise router.Router's tracer-forwarding without a
// real SDK — see Chat, which calls it exactly like a real provider would.
func (f *FakeProvider) SetTracer(t llm.Tracer) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tracer = t
}

// Chat implements llm.Provider.
func (f *FakeProvider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	f.mu.Lock()
	if f.callIdx >= len(f.script) {
		f.mu.Unlock()
		panic(fmt.Sprintf("llmtest: FakeProvider %q got more Chat calls (%d) than scripted responses (%d)", f.name, f.callIdx+1, len(f.script)))
	}
	resp := f.script[f.callIdx]
	f.callIdx++
	f.Requests = append(f.Requests, req)
	tracer := f.tracer
	f.mu.Unlock()

	if resp.Err != nil {
		if tracer != nil {
			tracer.Trace(ctx, f.name, fmt.Sprintf("%+v", req), "", resp.Err)
		}
		return nil, resp.Err
	}

	if tracer != nil {
		response := resp.Text
		if resp.ToolCall != nil {
			response += fmt.Sprintf(" tool_call:%s(%s)", resp.ToolCall.Name, resp.ToolCall.Arguments)
		}
		tracer.Trace(ctx, f.name, fmt.Sprintf("%+v", req), response, resp.StreamErr)
	}

	if resp.StreamErr != nil {
		ch := make(chan llm.StreamChunk, 1)
		ch <- llm.StreamChunk{Err: resp.StreamErr}
		close(ch)
		return ch, nil
	}

	ch := make(chan llm.StreamChunk, 3)
	if resp.Text != "" {
		ch <- llm.StreamChunk{TextDelta: resp.Text}
	}
	if resp.ToolCall != nil {
		ch <- llm.StreamChunk{ToolCall: resp.ToolCall}
	}
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

// Structured implements llm.StructuredProvider, scripted independently from
// Chat via WithStructured. A FakeProvider that never had WithStructured
// called has an empty script, so a test that calls Structured without
// scripting it panics immediately (same "under-specified" signal as Chat's
// own over-call guard) rather than silently returning a zero value.
func (f *FakeProvider) Structured(ctx context.Context, req llm.StructuredRequest) (json.RawMessage, error) {
	f.mu.Lock()
	if f.structuredCallIdx >= len(f.structuredScript) {
		f.mu.Unlock()
		panic(fmt.Sprintf("llmtest: FakeProvider %q got more Structured calls (%d) than scripted responses (%d)", f.name, f.structuredCallIdx+1, len(f.structuredScript)))
	}
	resp := f.structuredScript[f.structuredCallIdx]
	f.structuredCallIdx++
	f.StructuredRequests = append(f.StructuredRequests, req)
	f.mu.Unlock()

	if resp.Err != nil {
		return nil, resp.Err
	}
	return resp.JSON, nil
}

// ChatOnlyProvider forwards to an internal FakeProvider for Name/Chat/
// SetTracer but deliberately does NOT implement llm.StructuredProvider
// (embedding *FakeProvider would promote its Structured method too, which
// defeats the point — this type forwards explicitly instead). Use it,
// rather than FakeProvider directly, to test that router.Router.Structured
// skips a configured provider that doesn't support structured output
// instead of treating it as a hard failure.
type ChatOnlyProvider struct {
	fake *FakeProvider
}

// NewChatOnly creates a ChatOnlyProvider named name with the given scripted
// Chat responses (see New).
func NewChatOnly(name string, script ...Response) *ChatOnlyProvider {
	return &ChatOnlyProvider{fake: New(name, script...)}
}

func (c *ChatOnlyProvider) Name() string { return c.fake.Name() }

func (c *ChatOnlyProvider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	return c.fake.Chat(ctx, req)
}

func (c *ChatOnlyProvider) SetTracer(t llm.Tracer) { c.fake.SetTracer(t) }
