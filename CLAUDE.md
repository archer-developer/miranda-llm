# miranda-llm — project notes for Claude Code

Shared LLM plumbing for Miranda's family of services — a Go library module
(no `cmd/`, no HTTP server, no MCP tools of its own), imported by services
built from [miranda-service-skeleton](../miranda-service-skeleton) and by
[miranda](../miranda) itself.

## Why this exists

`miranda`'s own `internal/llm` (chat abstraction + router + Gemini/
Anthropic/OpenAI-compat provider adapters), `internal/keyrotation`, and
`internal/llmtrace` already implemented everything
[miranda-medical-card](../miranda-medical-card)'s architecture docs
(`docs/architecture/05-llm.md`) call for in a shared LLM package: multi-
provider routing, automatic escalation, API-key rotation, tracing. Rather
than copy-pasting that code into medical-card (drifts immediately) or
standing up a network LLM-gateway microservice (adds a hop, an availability
dependency, and exactly the kind of infra this home-infra family avoids —
see miranda-service-skeleton's own "no Docker, no CI" stance), it was
extracted into this standalone, non-`internal` module so it can be a normal
`go.mod` dependency for any number of services.

**This was a pure extraction of already-decoupled code, not a redesign.**
`miranda/internal/llm/llm.go` (the core `Provider`/`ChatRequest`/
`StreamChunk` interfaces) already had zero dependency on anything
Miranda-specific — no keyring, no users, no webui. `internal/keyrotation`
and `internal/llmtrace` were likewise already fully standalone. The only
real coupling to cut was each provider adapter and `router.Router` reading
config types (`config.EscalationConfig`, `config.GeminiToolsConfig`,
`config.GeminiRotationConfig`, `config.AnthropicToolsConfig`) directly out
of `miranda/internal/config` — see "What changed vs. the original" below.

**`miranda` has not been migrated onto this module yet.** `miranda`'s own
`internal/llm/*`, `internal/keyrotation`, `internal/llmtrace` still exist
and are still what `miranda` actually runs. Switching `miranda` to import
this module instead is a separate, independent task, tracked outside this
repo — until it lands, this module and `miranda`'s internal copy can drift
apart. Do not assume `miranda` already depends on this module.

## Conventions

Same as every service in this family (see
[miranda-service-skeleton](../miranda-service-skeleton)'s `CLAUDE.md`):
write explanatory comments (doc-comments on exported symbols, comments on
non-obvious logic and *why* a decision was made) — this is a small
home-infra codebase maintained intermittently, and future-you benefits more
from carried-forward reasoning than terse code. No Docker, no CI: `make
check` run locally before every commit is the actual quality gate. Module
path convention: `github.com/archer-developer/<repo-name>`.

Unlike the services built from the skeleton, this repo is a **library**,
not a deployable: no `cmd/`, no `config.Load`, no HTTP server, no
`scripts/deploy.sh`. Every constructor (`gemini.New`, `anthropic.New`,
`router.New`, ...) takes plain Go values — a consuming service owns
mapping its own YAML config into them, the same way `miranda-diary` maps
its config into `internal/diary.Store`'s constructor.

## Package-by-package

- **root (`llm`)** — `Provider` (streaming `Chat`), `StructuredProvider`
  (single-shot schema-constrained `Structured`, optional — see below),
  `Tracer`, and the shared `Message`/`ChatRequest`/`StreamChunk`/`ToolDef`
  types every provider adapter speaks. No provider-specific or
  Miranda-specific concept lives here.
- **`router`** — `Router.Chat` (streaming, with reliability fallback +
  opt-in per-provider escalation) and `Router.Structured` (single-shot,
  fallback-only, no escalation — see "Structured output" below for why
  these two differ). `EscalationConfig` lives here now, not in a shared
  config package — it's this package's own concept.
- **`keyrotation`** — the generic "try N keys, cool down, retry" loop. Used
  by `gemini` (multiple free-tier API keys); `anthropic` and `openaicompat`
  don't rotate keys (single credential per client, matching the original).
- **`llmtrace`** — writes one framed block per provider call to an
  `io.Writer` (typically a log file the consuming service opens). A `nil
  *Logger` is a valid no-op, so tracing is fully optional.
- **`gemini`, `anthropic`, `openaicompat`** — one `Provider` implementation
  each. `gemini.ToolsConfig`/`gemini.RotationConfig` and
  `anthropic.ToolsConfig` are this package's own config types now (see
  below).
- **`embedding`** — `Embedder` interface + `GeminiEmbedder`, generalized
  from `miranda-diary`'s `internal/embedding` (previously diary-only,
  duplicated nowhere else). Deliberately independent of the root `llm`
  package's chat types — an embedding call is a different shape (one text
  in, one vector out) with no conversation or tools.
- **`llmtest`** — `FakeProvider` (scriptable `Chat` + `Structured`),
  `ChatOnlyProvider` (implements `Provider` but deliberately not
  `StructuredProvider`, for testing fallback-skip behavior), and
  `FakeEmbedder`.

## What changed vs. the original (`miranda/internal/llm/*`)

- **Config types moved from `miranda/internal/config` into the package
  that actually owns them**, and were renamed to drop the now-redundant
  prefix (no more `config.` stutter once they're not shared package-wide):
  `config.EscalationConfig` → `router.EscalationConfig`;
  `config.GeminiToolsConfig` → `gemini.ToolsConfig`;
  `config.GeminiRotationConfig` → `gemini.RotationConfig`;
  `config.AnthropicToolsConfig` → `anthropic.ToolsConfig`. A consuming
  service defines its own YAML config shape (see
  `miranda-medical-card/docs/architecture/05-llm.md`) and maps it into
  these types when constructing providers/the router — this module has no
  opinion on YAML at all.
- **Doc comments were reworded** away from `miranda`-specific framing
  ("the Orchestrator's tool loop", "streamOneTurn", "internal/httpapi") to
  generic caller-agnostic language, since this code no longer knows it's
  embedded in an agent loop specifically.
- **Everything else — the actual logic** (fallback ordering, escalation
  hop-capping/cycle-detection, key rotation, SDK request/response
  shapes) — is unchanged from the original. If you're diffing behavior,
  diff against `miranda`'s git history for `internal/llm`, `internal/
  keyrotation`, `internal/llmtrace` as of the extraction date.

## New capabilities (didn't exist in `miranda/internal/llm`)

### Structured output

`llm.StructuredProvider` (optional — implemented by all three provider
packages here) and `router.Router.Structured` are new. `miranda`'s router
only ever needed streaming chat for its agent loop; a service like
medical-card needs single-shot, schema-constrained JSON for Extraction/
Planner/Summary-style calls (see `miranda-medical-card/docs/architecture/
05-llm.md` §9 "Structured Output").

Each provider implements it differently, per what its own API actually
offers:
- **`gemini`**: native `GenerateContentConfig.ResponseMIMEType =
  "application/json"` + `ResponseJsonSchema` (a raw JSON Schema passthrough
  field — same pattern as `ParametersJsonSchema` on function
  declarations), one non-streaming `GenerateContent` call.
- **`openaicompat`**: native `response_format: json_schema` (OpenAI itself
  supports this; not every OpenAI-*compatible* backend does — an
  unsupported backend just returns its own error, wrapped and returned,
  with no local fallback).
- **`anthropic`**: the Messages API has no native equivalent, so this
  forces the model to call a single synthetic tool whose input schema *is*
  `req.Schema`, via `ToolChoice`, then reads the tool call's `Input` back —
  the standard workaround for schema-constrained output on this API.

`router.Router.Structured` deliberately does **not** walk the escalation
chain `Chat` does: a one-shot structured call has no open conversation for
a model to reasonably self-assess as "too hard for me" mid-generation, and
escalating a structured task defeats the point of pinning it to a specific,
tested model/schema pairing. It does still do plain reliability fallback
(try the next configured provider on failure), and skips — rather than
hard-fails on — a provider that doesn't implement `StructuredProvider` at
all, so a mixed chain (e.g. a structured-capable primary with a chat-only
backup) still works for `Chat`.

### Embeddings

`embedding.Embedder` + `embedding.GeminiEmbedder` are new to this shared
module — generalized from `miranda-diary`'s previously diary-only
`internal/embedding` package (identical code, moved and renamed so it's no
longer duplicated per-service).

## Escalation is opt-in, not a required feature

`router.Router`'s tool-based escalation (a provider can be configured to
expose an `escalate_to_X` tool the model may call mid-conversation, and a
hard provider failure reroutes through the same mechanism) is exactly what
`miranda`'s agent loop needs and almost certainly **not** what a
single-shot-call service like medical-card wants for its own `Chat` calls
(if it uses `Chat` at all, rather than `Structured` for everything) — there's
no open conversation for a model to reasonably decide "hand this off to a
smarter model" mid-generation. This requires no code change to opt out:
pass `nil` (or an empty map) as `router.New`'s `escalations` argument, and
every provider falls back only via the plain reliability chain (try the
next configured provider if the current one's `Chat` call fails outright)
— see `router`'s package doc comment.

## Testing

```bash
make test    # go test ./... -race
```

- `testify` (`require`), matching every other service in this family.
- `keyrotation`, `llmtrace`, and `router` (including the new `Structured`
  path) have full test coverage, ported/extended from the original with no
  loss of coverage.
- `anthropic_test.go` covers the pure `toAnthropicTools`/`requiredFields`
  logic — same coverage as the original.
- `gemini_test.go` has full parity with the original
  `miranda/internal/llm/gemini/gemini_test.go`: both the pure-function
  tests (`isRetryable`, `toLLMToolCall`, `toGeminiContents`, `buildTools`,
  `New`'s fail-fast paths) and the `httptest.Server`-backed end-to-end
  suite driving `Chat`/`pump`/`attempt` (key rotation, live streaming,
  thought-signature round-tripping) via the package's `apiBaseURL`
  override — ported when `miranda` migrated onto this module.
- No CI configured, matching this family — `make check` before every
  commit is the actual gate.

## Code quality

```bash
make fmt     # gofmt + goimports
make lint    # golangci-lint run ./...
make check   # fmt + lint + test
```

`.golangci.yml` uses the same deliberately minimal linter set as every
sibling service (`errcheck`, `govet`, `ineffassign`, `staticcheck`,
`unused`).

## Consuming this module

See `README.md` for wiring examples. Two operational details worth
repeating here:

- **Private module**: `GOPRIVATE=github.com/archer-developer/*` needs to be
  set wherever `go get`/`go mod tidy`/`go build` runs against this module,
  or the toolchain will try (and fail) to resolve it through the public
  module proxy and checksum database.
- **Local multi-repo development**: use a `go.work` file (see README), not
  a committed `replace` directive — a workspace file stays a personal/local
  override that can't accidentally ship and break the build for a fresh
  clone.

## What's deliberately not here

- **YAML config loading of any kind** — every constructor takes plain Go
  values; a consuming service owns its own config schema entirely. This
  module has no opinion on config file format, merge order, or defaults.
- **A network LLM-gateway service** — the whole point of extracting this as
  a library instead of a microservice was to avoid the extra network hop,
  availability dependency, and deployment surface for a single-host
  home-infra setup. If a future service genuinely can't be Go (a rare case
  in this family), that's the point to revisit this decision, not before.
- **Vision/OCR as a distinct capability** — image input already flows
  through `llm.ContentPart`/`Message.Parts` as part of the normal chat
  interface (Gemini and Anthropic both consume it that way); there's no
  separate "OCR mode."
- **Migrating `miranda` onto this module** — see "Why this exists" above;
  explicitly out of scope for this repo, tracked as separate work.
