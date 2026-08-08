# miranda-llm

Shared LLM plumbing for Miranda's family of services: a provider-agnostic
chat interface, multi-provider fallback/escalation routing, API-key
rotation, request/response tracing, schema-constrained structured output,
and text embeddings — extracted from [miranda](../miranda)'s
`internal/llm` (the "brain" 's own LLM integration) into its own Go module
so it can be imported by `miranda` itself and by every other Miranda
microservice (starting with [miranda-medical-card](../miranda-medical-card))
instead of being copy-pasted into each one.

**`miranda` has not been migrated onto this module yet.** This repo is a
faithful extraction of `miranda/internal/llm`, `internal/keyrotation`, and
`internal/llmtrace` as they existed at the time of extraction, decoupled
from `miranda`'s own config package (see CLAUDE.md for exactly what
changed) and extended with two capabilities `miranda` didn't have yet
(Structured Output, Embeddings). Switching `miranda` itself to depend on
this module instead of its own internal copy is a separate, independent
task — until that happens, changes here and changes in `miranda`'s
`internal/llm` can drift apart.

See `CLAUDE.md` for the full extraction rationale, what changed in the
process, and package-by-package details.

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

## Using this module from a service

```
go get github.com/archer-developer/miranda-llm@latest
```

This is a private repository — `go get`/`go mod tidy` need
`GOPRIVATE=github.com/archer-developer/*` set in the environment so the Go
toolchain fetches it directly via git instead of trying the public module
proxy/checksum database.

### Wiring a Router

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

### Structured output (Extraction / Planner / Summary-style calls)

```go
schema := map[string]any{
	"type": "object",
	"properties": map[string]any{
		"diagnoses": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
	},
}

result, err := r.Structured(ctx, llm.StructuredRequest{
	Messages:   []llm.Message{{Role: llm.RoleUser, Content: documentText}},
	Schema:     schema,
	SchemaName: "medical_extraction",
})
// result is json.RawMessage — unmarshal into your own Go type.
```

`Structured` skips any configured provider that doesn't implement
`llm.StructuredProvider` rather than treating it as a hard failure, so a
mixed fallback chain works for `Chat` even if not every provider supports
structured output.

### Embeddings

```go
import "github.com/archer-developer/miranda-llm/embedding"

embedder, err := embedding.NewGemini(ctx, apiKey, "gemini-embedding-2")
vector, err := embedder.Embed(ctx, "some text")
```

### Testing your own code against this module

```go
import "github.com/archer-developer/miranda-llm/llmtest"

provider := llmtest.New("fake", llmtest.Response{Text: "hi"}).
	WithStructured(llmtest.StructuredResponse{JSON: json.RawMessage(`{"ok":true}`)})
```

See `llmtest.ChatOnlyProvider` for testing fallback behavior against a
provider that doesn't support structured output, and
`llmtest.FakeEmbedder` for testing code that depends on `embedding.Embedder`.

## Local development against this module

While iterating on both this module and a consuming service at the same
time, use a [Go workspace](https://go.dev/ref/mod#workspaces) instead of a
committed `replace` directive in the consumer's `go.mod`:

```
go work init ./miranda-llm ./miranda-medical-card
```

With a `go.work` file present, `go build`/`go test` in the consuming
service resolve your local, uncommitted changes to this module directly —
no need to tag a release and bump `go.mod` on every iteration. Remove or
don't commit `go.work` once you're done; it's a local override, not part of
the module's real dependency graph.

## Building and testing

```bash
make test    # go test ./... -race
make lint    # golangci-lint run ./...
make fmt     # gofmt + goimports
make check   # fmt + lint + test — run this before every commit
```

`make tools` installs `golangci-lint` and `goimports` if they're not
already on `PATH`. There is no CI configured, matching every other service
in this family — `make check` is the enforcement mechanism.

## Requires

Go 1.25+.
