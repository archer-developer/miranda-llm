// Package router selects among configured llm.Providers: it retries an
// ordered fallback chain when a provider is unreachable, and transparently
// reroutes a turn to a provider's own configured escalation target when
// that provider invokes its escalation tool (e.g. a small local model
// calling escalate_to_claude on a hard question) — walking a chain of these
// hop by hop, since each provider can configure its own target (see
// EscalationConfig). A provider that hard-fails mid-turn (e.g. a Gemini
// provider that exhausted every configured API key's quota across every
// retry cycle before streaming a single chunk — see gemini and
// keyrotation) triggers that same escalation target too, not just an
// explicit tool call (see deliver's use of escalateOnError) — a provider
// outage shouldn't fail a turn a configured stronger fallback could still
// answer.
//
// Escalation is entirely opt-in per provider: a caller that doesn't want
// this machinery (e.g. single-shot Extraction/Planner-style work, where
// there's no open conversation for a model to reasonably self-assess as
// "too hard for me" mid-generation) simply passes a nil or empty
// escalations map to New — every provider then falls back only via the
// plain reliability chain (try the next configured provider if the current
// one's Chat call fails outright), with no escalation tool ever exposed to
// the model and no mid-stream reroute.
package router

import (
	"context"
	"encoding/json"
	"fmt"

	llm "github.com/archer-developer/miranda-llm"
)

// maxEscalationHops caps how many hops one turn's escalation chain can walk
// — a real ladder is 2-3 hops deep (e.g. a cheap model escalates to a
// stronger one, which escalates to a top-tier model); this exists purely to
// stop a misconfigured cycle (A escalates to B, B escalates back to A) from
// looping forever, not to cap a legitimate deep chain.
const maxEscalationHops = 6

// EscalationConfig configures the explicit escalation tool that lets one
// provider hand a hard turn off to another. Lives per-provider (passed as a
// map keyed by provider name to New) rather than as a single global
// setting, so a chain of providers can each pick their own target/tool
// name — e.g. a cheap model escalates to a stronger one, which escalates to
// a top-tier model, each hop configured independently.
type EscalationConfig struct {
	Enabled  bool
	ToolName string
	// Description overrides the escalation tool's description shown to the
	// model — the default is a generic "too complex/ambiguous/high-stakes"
	// prompt (see defaultEscalationDescription). Set this when a provider
	// has a concrete, known-missing capability worth calling out
	// explicitly (e.g. "you have no working code-execution tool"), so the
	// model reliably escalates for it instead of attempting an answer it
	// has no way to actually compute.
	Description    string
	TargetProvider string
}

// Router dispatches chat turns to the configured llm.Providers.
type Router struct {
	providers   map[string]llm.Provider
	order       []string                    // fallback chain, in the order providers were configured (or reordered by defaultProvider)
	escalations map[string]EscalationConfig // keyed by provider name; a name absent (or present with Enabled: false) has no escalation
}

// New builds a Router over providers, tried in the given slice order for
// reliability fallback, and escalations (one EscalationConfig per provider
// name — pass nil or an empty map to disable escalation entirely, see the
// package doc comment). defaultProvider, if non-empty, is moved to the
// front of the fallback order — this is the source of truth for "default",
// not Providers' list position. Errors if defaultProvider is set but
// doesn't match any configured provider's Name().
func New(providers []llm.Provider, escalations map[string]EscalationConfig, defaultProvider string) (*Router, error) {
	if len(providers) == 0 {
		return nil, fmt.Errorf("router: no providers configured")
	}

	m := make(map[string]llm.Provider, len(providers))
	order := make([]string, 0, len(providers))
	for _, p := range providers {
		m[p.Name()] = p
		order = append(order, p.Name())
	}

	if defaultProvider != "" {
		if _, ok := m[defaultProvider]; !ok {
			return nil, fmt.Errorf("router: default_provider %q not found among configured providers", defaultProvider)
		}
		order = moveToFront(order, defaultProvider)
	}

	return &Router{providers: m, order: order, escalations: escalations}, nil
}

// moveToFront returns order with name moved to index 0, preserving the
// relative order of everything else — the rest of order becomes the
// fallback chain behind the explicit default.
func moveToFront(order []string, name string) []string {
	out := make([]string, 0, len(order))
	out = append(out, name)
	for _, n := range order {
		if n != name {
			out = append(out, n)
		}
	}
	return out
}

// traceable is implemented by providers that accept a request/response
// tracer directly (see llm.Tracer). Each Provider owns producing its own
// trace content — the exact request/response it built for its own SDK —
// rather than the Router formatting one from the provider-agnostic
// ChatRequest, which can't represent provider-specific specifics like a
// provider's own server-side tools.
type traceable interface {
	SetTracer(t llm.Tracer)
}

// SetTracer wires up request/response tracing (see the llmtrace package) by
// forwarding tracer to every configured provider that accepts one —
// optional, and left unset by default so tests and any caller that doesn't
// care about it don't need to touch this.
func (r *Router) SetTracer(tracer llm.Tracer) {
	for _, p := range r.providers {
		if t, ok := p.(traceable); ok {
			t.SetTracer(tracer)
		}
	}
}

// defaultEscalationDescription is used when a provider's EscalationConfig
// doesn't set its own Description — generic enough for any provider, but
// see that field's doc comment for when a provider-specific one is worth
// setting instead.
const defaultEscalationDescription = "Hand this turn off to a more capable model when the request is too complex, ambiguous, or high-stakes for you to handle well."

// escalationToolDef builds the llm.ToolDef the active provider sees for its
// own configured escalation target. Built here, not by the caller, because
// each provider's escalation config can differ (see EscalationConfig) and
// the caller has no per-provider visibility into which one is currently
// active at any given hop.
func escalationToolDef(esc EscalationConfig) llm.ToolDef {
	description := esc.Description
	if description == "" {
		description = defaultEscalationDescription
	}
	return llm.ToolDef{
		Name:        esc.ToolName,
		Description: description,
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"reason": map[string]any{"type": "string"}},
		},
	}
}

// requestFor returns req with esc's own escalation tool appended, if
// enabled — req itself (the base tool list) is never mutated, so each hop
// starts from the same base and only its own tool differs.
func requestFor(req llm.ChatRequest, esc EscalationConfig) llm.ChatRequest {
	if !esc.Enabled {
		return req
	}
	out := req
	out.Tools = append(append([]llm.ToolDef{}, req.Tools...), escalationToolDef(esc))
	return out
}

// Chat runs req through the fallback chain and, if escalation triggers,
// reroutes to the escalation target mid-stream — walking a chain of any
// depth (see deliver/escalate). onProviderUsed, if non-nil, is called
// exactly once with the name of the provider whose text ultimately reached
// the caller.
func (r *Router) Chat(ctx context.Context, req llm.ChatRequest, onProviderUsed func(name string)) (<-chan llm.StreamChunk, error) {
	return r.chat(ctx, req, "", onProviderUsed)
}

// ChatPinned is like Chat, but starts the fallback chain at pinnedProvider
// instead of the front of the configured order (a no-op, identical to Chat,
// when pinnedProvider is "" or names a provider that isn't configured) — for
// continuing a multi-turn interaction after an earlier turn escalated
// mid-conversation. A tool call the escalated model itself requested has to
// be answered by that same model on the next iteration once the tool result
// comes back — restarting from the chain's default provider would silently
// downgrade mid-turn, discarding the escalation the moment a tool was
// involved. pinnedProvider is only moved to the front of the
// reliability-fallback order, not treated as the sole candidate: if it
// fails to even start, the rest of the chain is still tried, same as Chat's
// own default-order fallback.
func (r *Router) ChatPinned(ctx context.Context, req llm.ChatRequest, pinnedProvider string, onProviderUsed func(name string)) (<-chan llm.StreamChunk, error) {
	return r.chat(ctx, req, pinnedProvider, onProviderUsed)
}

func (r *Router) chat(ctx context.Context, req llm.ChatRequest, pinnedProvider string, onProviderUsed func(name string)) (<-chan llm.StreamChunk, error) {
	order := r.order
	if pinnedProvider != "" {
		if _, ok := r.providers[pinnedProvider]; ok {
			order = moveToFront(r.order, pinnedProvider)
		}
	}

	stream, providerName, err := r.startChat(ctx, req, order)
	if err != nil {
		return nil, err
	}

	out := make(chan llm.StreamChunk)
	report := newOnceReporter(onProviderUsed)
	go r.pump(ctx, req, stream, providerName, out, report, map[string]bool{providerName: true})
	return out, nil
}

// newOnceReporter wraps onProviderUsed (may be nil) so it fires at most
// once regardless of how many escalation hops one turn walks.
func newOnceReporter(onProviderUsed func(string)) func(string) {
	var reported bool
	return func(name string) {
		if !reported && onProviderUsed != nil {
			onProviderUsed(name)
			reported = true
		}
	}
}

// startChat tries each provider in order until one accepts the request,
// implementing reliability fallback for connection/auth failures. Each
// candidate sees its own configured escalation tool appended to req.Tools
// (see requestFor), not a global one. order is r.order by default, or
// r.order re-rooted at a pinned provider (see ChatPinned) — callers never
// pass an arbitrary order otherwise.
func (r *Router) startChat(ctx context.Context, req llm.ChatRequest, order []string) (<-chan llm.StreamChunk, string, error) {
	var lastErr error
	for _, name := range order {
		stream, err := r.providers[name].Chat(ctx, requestFor(req, r.escalations[name]))
		if err != nil {
			lastErr = fmt.Errorf("%s: %w", name, err)
			continue
		}
		return stream, name, nil
	}
	return nil, "", fmt.Errorf("router: all providers failed: %w", lastErr)
}

// pump is the entry point for one Chat() call's whole output: it closes out
// exactly once, when the entire chain (initial call plus any escalation
// hops) finishes — deliver is the recursive part escalate calls back into,
// which must never close out itself.
func (r *Router) pump(ctx context.Context, req llm.ChatRequest, stream <-chan llm.StreamChunk, providerName string, out chan<- llm.StreamChunk, report func(string), visited map[string]bool) {
	defer close(out)
	r.deliver(ctx, req, stream, providerName, out, report, visited)
}

// deliver forwards chunks from the active provider's stream to out,
// intercepting a call to THAT provider's own configured escalation tool
// (r.escalations[providerName]) and recursing into escalate for the next
// hop instead of surfacing the tool call to the caller. Every hop's stream
// is examined the same way, which is what makes a chain of any depth work
// generically. Tracing is not this function's concern: each Provider traces
// its own call internally (see SetTracer above) as soon as its own stream
// finishes, before deliver ever sees the chunks.
func (r *Router) deliver(ctx context.Context, req llm.ChatRequest, stream <-chan llm.StreamChunk, providerName string, out chan<- llm.StreamChunk, report func(string), visited map[string]bool) {
	esc := r.escalations[providerName]
	for chunk := range stream {
		if chunk.Err != nil {
			// A hard failure (e.g. a quota-exhausted Gemini provider that
			// burned through every configured key and retry cycle before
			// producing a single chunk) is treated the same as this
			// provider explicitly asking to escalate — same target, same
			// hop-cap/cycle guards in escalate — so a provider outage
			// doesn't fail turns a configured stronger fallback could
			// still answer. Only tried once per target (visited): if it's
			// already been visited this turn, escalating again would just
			// trade this real error for escalate's own cycle-detected one,
			// so fall through and surface the original failure instead.
			if esc.Enabled && !visited[esc.TargetProvider] {
				r.escalateOnError(ctx, req, chunk.Err, esc, out, report, visited)
				return
			}
			out <- chunk
			return
		}

		if chunk.ToolCall != nil && esc.Enabled && chunk.ToolCall.Name == esc.ToolName {
			r.escalate(ctx, req, *chunk.ToolCall, esc, out, report, visited)
			return
		}

		if chunk.Done {
			report(providerName)
		}
		out <- chunk
	}
}

// escalate re-issues the conversation (base req plus a synthetic tool-call/
// tool-result turn acknowledging the handoff) to esc's target provider, and
// hands its stream to deliver for the NEXT hop — recursing through the same
// interception logic rather than forwarding raw, so the target's own
// escalation tool call (if it has one configured) is caught too. Guards
// against runaway/cyclic chains with both a hop cap and a per-turn visited
// set (a provider appearing twice in one chain is always a
// misconfiguration — a legitimate ladder is a straight line, never a
// cycle). The hop count is exactly len(visited): Chat seeds visited with
// the one initial provider, and every call here that survives both guards
// adds exactly one new entry before recursing — so there's no separate hop
// counter to carry (and risk desyncing from visited) as a parameter.
func (r *Router) escalate(ctx context.Context, req llm.ChatRequest, call llm.ToolCall, esc EscalationConfig, out chan<- llm.StreamChunk, report func(string), visited map[string]bool) {
	if len(visited) >= maxEscalationHops {
		out <- llm.StreamChunk{Err: fmt.Errorf("router: escalation chain exceeded %d hops (target %q) — check for a configuration cycle", maxEscalationHops, esc.TargetProvider)}
		return
	}
	if visited[esc.TargetProvider] {
		out <- llm.StreamChunk{Err: fmt.Errorf("router: escalation cycle detected: %q already appeared earlier in this turn's chain", esc.TargetProvider)}
		return
	}
	target, ok := r.providers[esc.TargetProvider]
	if !ok {
		out <- llm.StreamChunk{Err: fmt.Errorf("router: escalation target %q not configured", esc.TargetProvider)}
		return
	}

	escReq := appendEscalationTurn(req, call)
	escStream, err := target.Chat(ctx, requestFor(escReq, r.escalations[target.Name()]))
	if err != nil {
		out <- llm.StreamChunk{Err: fmt.Errorf("router: escalation to %s failed: %w", target.Name(), err)}
		return
	}

	nextVisited := make(map[string]bool, len(visited)+1)
	for k := range visited {
		nextVisited[k] = true
	}
	nextVisited[target.Name()] = true

	r.deliver(ctx, escReq, escStream, target.Name(), out, report, nextVisited)
}

// escalateOnError is deliver's counterpart to an explicit escalation tool
// call, for a provider that hard-failed instead: it synthesizes a tool
// call/result turn just like a real one (so the target sees a consistent
// history, and escalate's own hop-cap/cycle guards apply identically) with
// providerErr's message standing in for the model's own stated reason.
func (r *Router) escalateOnError(ctx context.Context, req llm.ChatRequest, providerErr error, esc EscalationConfig, out chan<- llm.StreamChunk, report func(string), visited map[string]bool) {
	// json.Marshal, not a hand-built string: providerErr's message is
	// arbitrary provider-supplied text (a Gemini quota error embeds real
	// newlines, for example) and Sprintf-ing it straight into a JSON
	// literal produces invalid JSON whenever it contains a quote, newline,
	// or other character JSON requires escaped.
	args, err := json.Marshal(map[string]string{
		"reason": fmt.Sprintf("previous provider failed: %s", providerErr.Error()),
	})
	if err != nil {
		args = []byte(`{"reason":"previous provider failed"}`)
	}
	call := llm.ToolCall{
		ID:        "error-escalation",
		Name:      esc.ToolName,
		Arguments: string(args),
	}
	r.escalate(ctx, req, call, esc, out, report, visited)
}

// appendEscalationTurn appends the assistant's escalation tool call and a
// synthetic tool result to the conversation, so the target provider sees a
// consistent history rather than a dangling unanswered tool call.
func appendEscalationTurn(req llm.ChatRequest, call llm.ToolCall) llm.ChatRequest {
	messages := make([]llm.Message, len(req.Messages), len(req.Messages)+2)
	copy(messages, req.Messages)
	messages = append(messages,
		llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call}},
		llm.Message{Role: llm.RoleTool, ToolCallID: call.ID, Content: "escalated to a more capable model"},
	)
	return llm.ChatRequest{Messages: messages, Tools: req.Tools}
}

// Structured runs req against the fallback chain exactly like Chat's
// reliability fallback (try each configured provider in order until one
// succeeds), but for a single non-streaming schema-constrained call instead
// of a streamed conversation — so it deliberately does not walk the
// tool-call/hard-failure escalation chain Chat does: a one-shot
// structured-output call has no open conversation for a model to reasonably
// self-assess as "too hard for me" mid-generation, and escalating a
// structured task defeats the point of pinning it to a specific, tested
// model/schema pairing. A provider that doesn't implement
// llm.StructuredProvider is skipped, not treated as a hard failure, so a
// mixed fallback chain (e.g. a structured-capable primary with a
// chat-only backup) still works for Chat even though Structured can't use
// every entry — Structured only fails outright once every candidate has
// either been skipped or hard-failed.
func (r *Router) Structured(ctx context.Context, req llm.StructuredRequest) (json.RawMessage, error) {
	var lastErr error
	tried := false
	for _, name := range r.order {
		sp, ok := r.providers[name].(llm.StructuredProvider)
		if !ok {
			continue
		}
		tried = true
		out, err := sp.Structured(ctx, req)
		if err == nil {
			return out, nil
		}
		lastErr = fmt.Errorf("%s: %w", name, err)
	}
	if !tried {
		return nil, fmt.Errorf("router: no configured provider implements structured output")
	}
	return nil, fmt.Errorf("router: all providers failed: %w", lastErr)
}
