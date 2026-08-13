# miranda-llm

A shared Go library for talking to LLM providers. It gives you one
provider-agnostic chat interface, plus the plumbing that tends to get
copy-pasted between services that call LLMs: fallback and escalation
routing across multiple providers, API-key rotation, request/response
tracing, schema-constrained structured output, and text embeddings.

It's a plain Go module — no `cmd/`, no config loader, no HTTP server. You
import it, hand its constructors plain Go values, and wire it into your
own service however you like.

See `CLAUDE.md` for package-by-package implementation notes.

## Packages

```
github.com/archer-developer/miranda-llm              — Provider, ChatRequest, StreamChunk,
                                                         StructuredRequest/StructuredProvider, Tracer
github.com/archer-developer/miranda-llm/router        — fallback + escalation routing over Providers
github.com/archer-developer/miranda-llm/keyrotation   — generic "rotate across N keys" retry loop
github.com/archer-developer/miranda-llm/llmtrace      — request/response trace logging
github.com/archer-developer/miranda-llm/gemini        — llm.Provider for native Gemini
github.com/archer-developer/miranda-llm/anthropic     — llm.Provider for native Claude
github.com/archer-developer/miranda-llm/openaicompat  — llm.Provider for any OpenAI-compatible endpoint
github.com/archer-developer/miranda-llm/embedding     — Embedder interface + Gemini implementation
github.com/archer-developer/miranda-llm/llmtest       — scriptable fakes for tests
```

## Installing

```
go get github.com/archer-developer/miranda-llm@latest
```

This is a private repository, so `go get`/`go mod tidy` need
`GOPRIVATE=github.com/archer-developer/*` set in your environment —
otherwise the Go toolchain will try to fetch it through the public module
proxy and checksum database and fail.

## Getting started: a simple chat call

Everything starts with a `Provider`. Here's the smallest useful example,
using Claude directly with no routing or fallback:

```go
import (
	llm "github.com/archer-developer/miranda-llm"
	"github.com/archer-developer/miranda-llm/anthropic"
)

provider := anthropic.New("claude", "claude-sonnet-5", apiKey, anthropic.ToolsConfig{})

stream, err := provider.Chat(ctx, llm.ChatRequest{
	Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello!"}},
}, nil)
for chunk := range stream {
	// chunk.Text, chunk.ToolCalls, chunk.Err, ...
}
```

That's enough for a single-provider integration. The rest of this README
covers what to reach for once you need more than one provider, retries, or
structured JSON output.

## Talking to more than one provider: the Router

Real deployments usually want a primary provider with a fallback (or a
free-tier provider that can hand off to a paid one when it's out of
capacity). `router.Router` wraps any number of `Provider`s with two
behaviors:

- **Reliability fallback** — if the current provider's call fails outright,
  try the next one.
- **Escalation** (optional) — a provider can expose an `escalate_to_X` tool
  that the model itself may call mid-conversation to hand off to a
  stronger model. This only makes sense for an open-ended chat/agent loop;
  see below for why it's opt-in.

```go
import (
	"log/slog"

	llm "github.com/archer-developer/miranda-llm"
	"github.com/archer-developer/miranda-llm/anthropic"
	"github.com/archer-developer/miranda-llm/gemini"
	"github.com/archer-developer/miranda-llm/router"
)

geminiProvider, err := gemini.New(ctx, "gemini-flash", "gemini-3.5-flash",
	[]string{"GEMINI_API_KEY_1", "GEMINI_API_KEY_2"},
	gemini.ToolsConfig{GoogleSearch: true},
	gemini.RotationConfig{CooldownSeconds: 10, MaxRetryCycles: 3},
	logger,
)

claudeProvider := anthropic.New("claude", "claude-sonnet-5", apiKey, anthropic.ToolsConfig{})

// escalations is optional — pass nil to disable escalation entirely and
// get only plain reliability fallback (try the next provider if the
// current one's Chat call fails outright). See router's package doc
// comment.
r, err := router.New(
	[]llm.Provider{geminiProvider, claudeProvider},
	map[string]router.EscalationConfig{
		"gemini-flash": {Enabled: true, ToolName: "escalate_to_claude", TargetProvider: "claude"},
	},
	"gemini-flash", // default provider
)

stream, err := r.Chat(ctx, llm.ChatRequest{
	Messages: []llm.Message{{Role: llm.RoleUser, Content: "..."}},
}, nil)
for chunk := range stream {
	// ...
}
```

If you don't need escalation — for example, if your service only ever
makes one-off calls with no open conversation for a model to hand off
mid-generation — just pass `nil` for the escalations map. You still get
reliability fallback across providers for free.

## Getting JSON back instead of chat text: Structured output

For extraction, planning, or "summarize this into a fixed shape" calls,
you usually want JSON that matches a schema, not free-form chat text.
`llm.StructuredProvider` (implemented by all three provider packages) and
`router.Router.Structured` give you a single-shot call that enforces a
JSON Schema on the response:

```go
schema := map[string]any{
	"type": "object",
	"properties": map[string]any{
		"items": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
	},
}

result, err := r.Structured(ctx, llm.StructuredRequest{
	Messages:   []llm.Message{{Role: llm.RoleUser, Content: documentText}},
	Schema:     schema,
	SchemaName: "extraction_result",
})
// result is json.RawMessage — unmarshal into your own Go type.
```

Each provider enforces the schema in whatever way its own API supports
(native JSON-schema response modes for Gemini and OpenAI-compatible
backends; a forced single tool call for Anthropic, which has no native
equivalent). You don't need to know which — the interface is the same
either way.

Note that `Structured` only does plain reliability fallback, not
escalation — a one-shot call has no open conversation for a model to
reasonably decide it's out of its depth, and escalating away from a model
you deliberately chose and tested against a schema would defeat the
point. It also skips (rather than fails on) any configured provider that
doesn't implement `StructuredProvider` at all, so you can safely mix a
structured-capable primary with a chat-only backup — `Chat` calls still
work across the whole chain.

## Turning text into vectors: Embeddings

Embeddings are a separate, simpler concept from chat — one text in, one
vector out, no conversation or tools — so they live in their own package
independent of everything above:

```go
import "github.com/archer-developer/miranda-llm/embedding"

embedder, err := embedding.NewGemini(ctx, apiKey, "gemini-embedding-2")
vector, err := embedder.Embed(ctx, "some text")
```

## Testing code that depends on this module

Use `llmtest` instead of hitting real providers in your tests:

```go
import "github.com/archer-developer/miranda-llm/llmtest"

provider := llmtest.New("fake", llmtest.Response{Text: "hi"}).
	WithStructured(llmtest.StructuredResponse{JSON: json.RawMessage(`{"ok":true}`)})
```

- `llmtest.ChatOnlyProvider` implements `Provider` but deliberately not
  `StructuredProvider` — use it to test that your code correctly skips a
  provider that can't do structured output.
- `llmtest.FakeEmbedder` does the same for code that depends on
  `embedding.Embedder`.

## Developing this module alongside a consuming service

If you're iterating on both this module and a service that imports it at
the same time, use a [Go workspace](https://go.dev/ref/mod#workspaces)
instead of adding a `replace` directive to the consumer's `go.mod`:

```
go work init ./miranda-llm ./your-service
```

With a `go.work` file present, `go build`/`go test` in the consuming
service resolve your local, uncommitted changes to this module directly —
no need to tag a release and bump `go.mod` on every iteration. Don't
commit `go.work`; it's a local override, not part of the module's real
dependency graph.

## Building and testing

```bash
make test    # go test ./... -race
make lint    # golangci-lint run ./...
make fmt     # gofmt + goimports
make check   # fmt + lint + test — run this before every commit
```

`make tools` installs `golangci-lint` and `goimports` if they're not
already on `PATH`. There is no CI configured — `make check` is the
enforcement mechanism.

## Requires

Go 1.25+.
