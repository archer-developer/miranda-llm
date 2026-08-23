// Package gemini implements llm.Provider (and llm.StructuredProvider) on
// top of the official google.golang.org/genai Go SDK, used exclusively for
// native Gemini models — it gets full native support for function calling
// combined with Grounding with Google Search in one request, unlike routing
// Gemini through an OpenAI-compatible shim (code execution is deliberately
// not wired — see nativeTools's doc comment, it's Vertex AI-only on the
// generateContent API this package calls). See the anthropic package for
// the sibling adapter this mirrors, and keyrotation for the API-key-rotation
// shape this uses — broadened here to rotate on quota, server, AND per-key
// auth errors, not just quota — see isRetryable.
package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"google.golang.org/genai"

	llm "github.com/archer-developer/miranda-llm"
	"github.com/archer-developer/miranda-llm/keyrotation"
)

// apiBaseURL overrides every genai.Client's HTTPOptions.BaseURL. It's a
// var, not a const, so tests can point it at an httptest.Server instead of
// the real Gemini API. Empty (the default) means "use the SDK's real
// Gemini API endpoint".
var apiBaseURL string

// defaultStructuredTemperature is used by Structured when
// llm.StructuredRequest.Temperature is nil — see that field's doc comment
// for why Structured doesn't just fall through to the API's own default
// the way Chat does.
var defaultStructuredTemperature = 0.1

// toFloat32Ptr converts an optional float64 (the shared llm package's
// temperature type, chosen for provider-agnosticism) to the *float32
// genai.GenerateContentConfig.Temperature actually wants. Returns nil
// unchanged so the SDK's own default applies.
func toFloat32Ptr(v *float64) *float32 {
	if v == nil {
		return nil
	}
	f := float32(*v)
	return &f
}

// ToolsConfig toggles Gemini's own native server-side tools sent on every
// request from this provider — mirrors anthropic.ToolsConfig's
// opt-in-only shape (all default to false).
type ToolsConfig struct {
	// GoogleSearch enables Grounding with Google Search — analogous to
	// anthropic.ToolsConfig.WebSearch.
	//
	// There is deliberately no CodeExecution field here, unlike
	// anthropic.ToolsConfig: Gemini's code-execution tool
	// (genai.ToolCodeExecution) only works on Vertex AI, not the plain
	// Gemini Developer API key this provider type uses — confirmed via the
	// SDK's own source comments, consistently applied across ~98 other
	// genuinely-Vertex-only fields. See nativeTools's doc comment for the
	// full reasoning, including why a working-looking public example
	// doesn't actually contradict this.
	GoogleSearch bool
	// ContextCaching is not implemented yet — New rejects startup if this
	// is true. Gemini's CachedContent is an explicit, separately-managed
	// resource (create/reference/invalidate), structurally unlike
	// Anthropic's per-request cache_control breakpoint, and needs its own
	// design pass (cache-key strategy, invalidation trigger, TTL policy)
	// that doesn't belong bolted onto this field.
	ContextCaching bool
}

// RotationConfig tunes this package's key-rotation. The zero value falls
// back to keyrotation's built-in defaults (1 cycle, no cooldown).
type RotationConfig struct {
	CooldownSeconds int
	MaxRetryCycles  int
}

// Provider is an llm.Provider (and llm.StructuredProvider) backed by the
// native Gemini API. Unlike anthropic.Provider (one API key, one client),
// it holds one *genai.Client per configured key and rotates across them on
// a quota or server error — see pump/attempt below.
type Provider struct {
	name     string
	model    string
	clients  []*genai.Client // one per resolved API key, in configured order
	tools    ToolsConfig
	rotation RotationConfig
	tracer   llm.Tracer
	logger   *slog.Logger
}

// New builds a Provider named name for the given Gemini model, resolving
// apiKeyEnvs to actual key values from the process environment (never
// stored in the caller's own config file — same convention as every other
// secret in this family of services) and building one genai.Client per
// resolved key up front. Building clients eagerly means a later rotation is
// just "try the next already-built client" — no per-request client
// reconstruction, and no risk of a client build failure surfacing mid-
// conversation instead of at startup. Fails fast if none of apiKeyEnvs
// resolve to a non-empty value, since every subsequent Chat/Structured call
// would otherwise fail identically anyway.
func New(ctx context.Context, name, model string, apiKeyEnvs []string, tools ToolsConfig, rotation RotationConfig, logger *slog.Logger) (*Provider, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if tools.ContextCaching {
		// Rejecting outright, rather than silently ignoring the flag, means
		// a config typo/aspiration doesn't silently no-op — see
		// ToolsConfig.ContextCaching's doc comment for why Gemini's
		// CachedContent (an explicit, separately-managed resource) isn't
		// implemented yet.
		return nil, fmt.Errorf("gemini: context_caching is not implemented yet (see ToolsConfig doc comment) — leave it false")
	}

	var clients []*genai.Client
	for _, env := range apiKeyEnvs {
		key := os.Getenv(env)
		if key == "" {
			continue
		}
		client, err := genai.NewClient(ctx, &genai.ClientConfig{
			APIKey:      key,
			Backend:     genai.BackendGeminiAPI,
			HTTPOptions: genai.HTTPOptions{BaseURL: apiBaseURL},
		})
		if err != nil {
			return nil, fmt.Errorf("gemini: build client for %s: %w", env, err)
		}
		clients = append(clients, client)
	}
	if len(clients) == 0 {
		return nil, fmt.Errorf("gemini: none of the configured api_key_envs %v are set", apiKeyEnvs)
	}

	return &Provider{
		name:     name,
		model:    model,
		clients:  clients,
		tools:    tools,
		rotation: rotation,
		logger:   logger,
	}, nil
}

func (p *Provider) Name() string { return p.name }

// SetTracer matches anthropic.Provider's — the router's traceable
// structural interface (see router.Router.SetTracer, which type-asserts
// for this method) forwards a tracer here.
func (p *Provider) SetTracer(t llm.Tracer) { p.tracer = t }

// Chat implements llm.Provider by streaming a GenerateContentStream
// response, forwarding text/tool-call chunks to the caller live as they
// arrive (token-by-token, like anthropic.Provider's pump) while rotating
// across configured API keys on a retryable error (see pump/attempt).
func (p *Provider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	system, contents := toGeminiContents(req.Messages)
	cfg := &genai.GenerateContentConfig{
		SystemInstruction: system,
		Tools:             p.buildTools(req.Tools),
		Temperature:       toFloat32Ptr(req.Temperature),
	}

	out := make(chan llm.StreamChunk)
	go p.pump(ctx, contents, cfg, out)
	return out, nil
}

// pump rotates across p.clients (one per configured API key) via
// keyrotation.Run; isRetryable is broad (quota AND 5xx AND per-key auth
// failures, see its doc comment) since a conversational turn can't just
// drop the whole turn on a transient upstream failure the way a single
// fire-and-forget request can afford to fail loudly.
func (p *Provider) pump(ctx context.Context, contents []*genai.Content, cfg *genai.GenerateContentConfig, out chan<- llm.StreamChunk) {
	defer close(out)

	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}
	err := keyrotation.Run(ctx, p.logger, "gemini", len(p.clients), rotCfg, isRetryable,
		func(ctx context.Context, i int) error {
			return p.attempt(ctx, p.clients[i], i, contents, cfg, out)
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
// StreamChunk{Err: ...} to out before returning (see attempt's doc
// comment for the two cases that produce it) — pump checks for this
// sentinel specifically so it doesn't write a second Err chunk on top of
// the one attempt already sent. isRetryable naturally treats it as
// non-retryable too (it isn't a genai.APIError), so keyrotation.Run stops
// rotating and returns it straight back to pump.
var errAttemptWritten = errors.New("gemini: attempt already wrote its terminal error")

// attempt makes exactly one GenerateContentStream call against one client
// (one API key), forwarding TextDelta/ToolCall chunks to out live as they
// stream in — unlike a fully-buffered design, the first output can reach
// the caller before Gemini finishes generating the rest of the reply.
//
// The one safety rule this requires: once ANY chunk has been forwarded to
// out for this attempt, a later error on the SAME attempt must not be
// retried even if isRetryable(err) would otherwise say yes — rotating to
// another key re-runs the whole request from scratch, and a caller that
// live-forwards chunks as they arrive has no way to "unsay" one, so a retry
// after partial output would duplicate or garble what was already emitted.
// attempt itself decides this (via the local forwarded bool) rather than
// pushing the decision into isRetryable, since isRetryable is a pure
// function of the error alone and has no way to know this attempt's
// forwarding state.
//
// Returns nil once the response completes cleanly (having already written
// Done to out). Returns errAttemptWritten in the two cases described above
// (forwarded-then-error, or a non-retryable error from the first event) —
// both already wrote their own Err chunk to out, so keyrotation.Run/pump
// must not retry or double-report. Returns the raw streamErr only when
// nothing has been forwarded yet AND isRetryable(streamErr) is true, so
// keyrotation.Run's own isRetryable check (the same package-level
// function, applied to the same value) agrees and tries the next key.
func (p *Provider) attempt(ctx context.Context, client *genai.Client, keyIndex int, contents []*genai.Content, cfg *genai.GenerateContentConfig, out chan<- llm.StreamChunk) error {
	start := time.Now()
	var textBuf strings.Builder // accumulated only for tracing, not for forwarding
	var toolCalls []llm.ToolCall
	var usage *genai.GenerateContentResponseUsageMetadata
	var finishReason genai.FinishReason
	var finishMessage string
	var promptFeedback *genai.GenerateContentResponsePromptFeedback
	forwarded := false

	for resp, streamErr := range client.Models.GenerateContentStream(ctx, p.model, contents, cfg) {
		if streamErr != nil {
			if forwarded {
				p.logger.Error("gemini: request failed mid-stream after partial output, not retrying", "provider", p.name, "key_index", keyIndex, "error", streamErr)
				wrapped := fmt.Errorf("gemini: %w", streamErr)
				p.trace(ctx, contents, cfg, textBuf.String(), toolCalls, time.Since(start), usage, wrapped)
				out <- llm.StreamChunk{Err: wrapped}
				return errAttemptWritten
			}
			if isRetryable(streamErr) {
				p.logger.Warn("gemini: key error, trying next key", "provider", p.name, "key_index", keyIndex, "error", streamErr)
				return streamErr
			}
			p.logger.Error("gemini: request failed", "provider", p.name, "key_index", keyIndex, "error", streamErr)
			wrapped := fmt.Errorf("gemini: %w", streamErr)
			p.trace(ctx, contents, cfg, "", nil, time.Since(start), usage, wrapped)
			out <- llm.StreamChunk{Err: wrapped}
			return errAttemptWritten
		}
		if resp.UsageMetadata != nil {
			usage = resp.UsageMetadata
		}
		if resp.PromptFeedback != nil {
			promptFeedback = resp.PromptFeedback
		}
		for _, cand := range resp.Candidates {
			// Populated only on the stream's terminal chunk for this
			// candidate, so this naturally ends up holding the final
			// finish reason once the loop exits.
			if cand.FinishReason != "" {
				finishReason = cand.FinishReason
				finishMessage = cand.FinishMessage
			}
			if cand.Content == nil {
				continue
			}
			for _, part := range cand.Content.Parts {
				switch {
				case part.Text != "":
					textBuf.WriteString(part.Text)
					out <- llm.StreamChunk{TextDelta: part.Text}
					forwarded = true
				case part.FunctionCall != nil:
					tc := toLLMToolCall(part.FunctionCall, part.ThoughtSignature, len(toolCalls))
					toolCalls = append(toolCalls, tc)
					out <- llm.StreamChunk{ToolCall: &tc}
					forwarded = true
				}
			}
		}
	}

	// A prompt-level block (PromptFeedback.BlockReason set) means Gemini
	// never got as far as generating candidates at all — most plausibly
	// its safety filter tripping on sensitive content in the accumulated
	// conversation (observed in practice on a turn whose prior tool result
	// had just placed a full lab-report table, tied to an already-known
	// pregnancy/health condition, into context). The stream still ends
	// with no streamErr and zero forwarded content, so left unchecked this
	// falls straight through to the "succeeded" path below with an empty
	// Done chunk — indistinguishable, from the caller's side, from the
	// model genuinely having nothing to say. Checked before the
	// finishReason case since a prompt-level block leaves candidates
	// empty, so finishReason would never be populated either.
	if promptFeedback != nil && promptFeedback.BlockReason != "" {
		p.logger.Error("gemini: request blocked at prompt level", "provider", p.name, "key_index", keyIndex, "blockReason", promptFeedback.BlockReason, "blockReasonMessage", promptFeedback.BlockReasonMessage)
		wrapped := fmt.Errorf("gemini: prompt blocked (blockReason=%s): %s", promptFeedback.BlockReason, promptFeedback.BlockReasonMessage)
		p.trace(ctx, contents, cfg, textBuf.String(), toolCalls, time.Since(start), usage, wrapped)
		out <- llm.StreamChunk{Err: wrapped}
		return errAttemptWritten
	}

	// A finish reason other than STOP means the response was curtailed —
	// most commonly MAX_TOKENS, observed in practice on OCR transcription
	// calls where the model's internal "thinking" tokens consumed nearly
	// the entire output budget before any of the actual transcription was
	// written, silently truncating mid-sentence with no error. Left
	// unchecked, a caller sees exactly the same TextDelta/Done sequence as
	// a genuinely complete response — there is no way for it to tell
	// partial output from whole output apart. Treating this as a terminal
	// (non-retryable: a different API key won't change the model's own
	// output budget) error lets callers like extraction.OCR's
	// ocrWithEscalation react to it — escalating to a different provider —
	// instead of unknowingly feeding truncated text into everything
	// downstream.
	//
	// An unset finishReason with nothing ever forwarded is folded into the
	// same abnormal case: a real completion always populates it (to STOP,
	// at minimum) on the stream's terminal chunk, so seeing neither text
	// nor a tool call NOR a finish reason means the stream ended having
	// produced literally nothing recognizable — observed directly on a
	// live gemini-lite call (candidates present, all with nil Content, no
	// finishReason on any of them, no PromptFeedback either) that this
	// condition used to let through as a "successful" empty reply, which
	// then reached the end user as a blank message.
	if finishReason != genai.FinishReasonStop && (finishReason != "" || !forwarded) {
		p.logger.Error("gemini: request finished abnormally, response may be incomplete", "provider", p.name, "key_index", keyIndex, "finishReason", finishReason, "finishMessage", finishMessage)
		wrapped := fmt.Errorf("gemini: response incomplete (finishReason=%q): %s", finishReason, finishMessage)
		p.trace(ctx, contents, cfg, textBuf.String(), toolCalls, time.Since(start), usage, wrapped)
		out <- llm.StreamChunk{Err: wrapped}
		return errAttemptWritten
	}

	p.logger.Info("gemini: request succeeded", "provider", p.name, "key_index", keyIndex, "duration_ms", time.Since(start).Milliseconds())
	p.trace(ctx, contents, cfg, textBuf.String(), toolCalls, time.Since(start), usage, nil)
	out <- llm.StreamChunk{Done: true}
	return nil
}

// Structured implements llm.StructuredProvider using Gemini's native
// response-schema mode (GenerateContentConfig.ResponseMIMEType +
// ResponseJsonSchema — a raw JSON Schema passthrough field, the same
// pattern buildTools already uses for function declarations via
// ParametersJsonSchema) with a single, non-streaming GenerateContent call.
// Rotates across configured API keys exactly like Chat's pump/attempt, via
// the same keyrotation.Run + isRetryable.
func (p *Provider) Structured(ctx context.Context, req llm.StructuredRequest) (json.RawMessage, error) {
	system, contents := toGeminiContents(req.Messages)
	temperature := req.Temperature
	if temperature == nil {
		temperature = &defaultStructuredTemperature
	}
	cfg := &genai.GenerateContentConfig{
		SystemInstruction:  system,
		ResponseMIMEType:   "application/json",
		ResponseJsonSchema: req.Schema,
		Temperature:        toFloat32Ptr(temperature),
	}

	rotCfg := keyrotation.Config{
		Cycles:   p.rotation.MaxRetryCycles,
		Cooldown: time.Duration(p.rotation.CooldownSeconds) * time.Second,
	}

	start := time.Now()
	var result json.RawMessage
	var usage *genai.GenerateContentResponseUsageMetadata
	err := keyrotation.Run(ctx, p.logger, "gemini-structured", len(p.clients), rotCfg, isRetryable,
		func(ctx context.Context, i int) error {
			resp, err := p.clients[i].Models.GenerateContent(ctx, p.model, contents, cfg)
			if err != nil {
				return err
			}
			p.logStructuredFinish(resp)
			usage = resp.UsageMetadata
			text := resp.Text()
			result = json.RawMessage(text)
			return nil
		},
	)
	if err != nil {
		wrapped := fmt.Errorf("gemini: structured: %w", err)
		p.trace(ctx, contents, cfg, "", nil, time.Since(start), usage, wrapped)
		return nil, wrapped
	}
	p.trace(ctx, contents, cfg, string(result), nil, time.Since(start), usage, nil)
	return result, nil
}

// logStructuredFinish surfaces *why* a Structured response looked the way
// it did — without this, a call that Gemini silently curtailed for safety
// reasons (finishReason SAFETY/PROHIBITED_CONTENT/etc., or a prompt-level
// block that leaves zero candidates) is completely indistinguishable from
// the model genuinely finding nothing to extract: both return a
// successful, valid-JSON response with every field empty, no error for
// Structured's caller to see. Logged at Warn only for a finish that isn't
// the unremarkable STOP (or empty, meaning finishReason wasn't populated),
// so an ordinary call stays at Debug and doesn't drown out the cases worth
// noticing.
func (p *Provider) logStructuredFinish(resp *genai.GenerateContentResponse) {
	if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
		p.logger.Warn("gemini: structured request blocked at prompt level", "provider", p.name, "blockReason", resp.PromptFeedback.BlockReason, "blockReasonMessage", resp.PromptFeedback.BlockReasonMessage)
	}
	if len(resp.Candidates) == 0 {
		p.logger.Warn("gemini: structured request returned no candidates", "provider", p.name)
		return
	}
	switch reason := resp.Candidates[0].FinishReason; reason {
	case "", genai.FinishReasonStop:
		p.logger.Debug("gemini: structured request finished", "provider", p.name, "finishReason", reason)
	default:
		p.logger.Warn("gemini: structured request finished abnormally", "provider", p.name, "finishReason", reason, "finishMessage", resp.Candidates[0].FinishMessage)
	}
}

// toLLMToolCall converts a Gemini FunctionCall to llm.ToolCall,
// JSON-encoding Args (a map[string]any) into the raw-JSON-string shape
// llm.ToolCall.Arguments expects. Gemini doesn't always populate
// FunctionCall.ID (it's documented as optional); a caller matching a later
// tool result back to this call by ID would otherwise silently break, so a
// deterministic one is synthesized from the call's name and position when
// Gemini leaves it blank.
//
// thoughtSignature is the sibling Part's ThoughtSignature (not a field on
// FunctionCall itself) — confirmed against a real API response, not just
// SDK docs: replaying a function-call turn without echoing this back is a
// hard 400 ("Function call is missing a thought_signature..."), not just
// degraded quality as the SDK's own doc comments elsewhere might suggest.
// Base64-encoded into ProviderMetadata since it's opaque binary and
// llm.ToolCall carries only string fields.
func toLLMToolCall(fc *genai.FunctionCall, thoughtSignature []byte, index int) llm.ToolCall {
	id := fc.ID
	if id == "" {
		id = fmt.Sprintf("%s-%d", fc.Name, index)
	}
	argsJSON, err := json.Marshal(fc.Args)
	if err != nil {
		argsJSON = []byte("{}")
	}
	var providerMetadata string
	if len(thoughtSignature) > 0 {
		providerMetadata = base64.StdEncoding.EncodeToString(thoughtSignature)
	}
	return llm.ToolCall{ID: id, Name: fc.Name, Arguments: string(argsJSON), ProviderMetadata: providerMetadata}
}

// isRetryable reports whether err is worth rotating to the next key for:
//   - a quota/rate-limit error (HTTP 429, or APIError.Status ==
//     "RESOURCE_EXHAUSTED")
//   - a server-side error (HTTP 5xx)
//   - a per-key auth failure (HTTP 401 Unauthorized / 403 Forbidden) — this
//     is a property of THAT key specifically (revoked, disabled, or a
//     per-key restriction hit independent of quota), not of the request,
//     so a different configured key can plausibly still succeed. This is
//     distinct from a malformed/bad request (HTTP 400), which would fail
//     identically no matter which key sends it and stays non-retryable
//     below — rotating keys only helps when the fault is IN the key, not
//     in what's being asked.
//
// Any other error (a malformed request, or a network failure before any
// HTTP response came back) is not retried: cycling keys can't fix a
// request that's simply wrong.
func isRetryable(err error) bool {
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	switch apiErr.Code {
	case http.StatusTooManyRequests, http.StatusUnauthorized, http.StatusForbidden:
		return true
	}
	if apiErr.Status == "RESOURCE_EXHAUSTED" {
		return true
	}
	return apiErr.Code >= 500 && apiErr.Code < 600
}

// trace serializes the actual request (contents + cfg) and response (text +
// tool calls) to p.tracer, mirroring anthropic.Provider.trace's contract —
// same pattern, just over Gemini's own types instead of Anthropic's. Used
// by both Chat's attempt and Structured.
//
// duration/usage (added 2026-08-13) are diagnostic-only extras layered on
// top of that same contract — how long this specific call took and, when
// Gemini reported it, its prompt/candidates token counts — surfaced
// because a real debugging session needed exactly this (medical.ask calls
// silently taking many seconds under Gemini free-tier rate-limit retries)
// and llm.log was otherwise the only place that could show it per-call.
// Both are dropped on the error path along with the rest of respPayload —
// llmtrace.Logger.Trace itself only ever writes the error message when err
// is non-nil (see its own doc comment), so there's nowhere for them to
// surface there regardless of what this method passes in.
func (p *Provider) trace(ctx context.Context, contents []*genai.Content, cfg *genai.GenerateContentConfig, text string, toolCalls []llm.ToolCall, duration time.Duration, usage *genai.GenerateContentResponseUsageMetadata, err error) {
	if p.tracer == nil {
		return
	}
	reqPayload := struct {
		Contents []*genai.Content             `json:"contents"`
		Config   *genai.GenerateContentConfig `json:"config"`
	}{contents, cfg}
	reqJSON, marshalErr := json.MarshalIndent(reqPayload, "", "  ")
	if marshalErr != nil {
		reqJSON = []byte(fmt.Sprintf("(failed to marshal request: %v)", marshalErr))
	}
	var respJSON []byte
	if err == nil {
		respPayload := struct {
			Text       string                                      `json:"text"`
			ToolCalls  []llm.ToolCall                              `json:"tool_calls,omitempty"`
			DurationMs int64                                       `json:"duration_ms"`
			Usage      *genai.GenerateContentResponseUsageMetadata `json:"usage,omitempty"`
		}{text, toolCalls, duration.Milliseconds(), usage}
		if respJSON, marshalErr = json.MarshalIndent(respPayload, "", "  "); marshalErr != nil {
			respJSON = []byte(fmt.Sprintf("(failed to marshal response: %v)", marshalErr))
		}
	}
	p.tracer.Trace(ctx, p.name, string(reqJSON), string(respJSON), err)
}

// toGeminiContents splits llm.Message history into Gemini's separate
// top-level system instruction and turn-by-turn Content list — the same
// split anthropic.toAnthropicMessages does for Claude's system param.
// RoleTool messages map onto Gemini's FunctionResponse part, which requires
// the original tool's Name (not just its call ID) — llm.Message{Role:
// RoleTool} only carries ToolCallID, so Name is recovered from the
// preceding assistant message's matching ToolCall as history is walked.
//
// A replayed FunctionCall part also needs its ThoughtSignature echoed back
// from tc.ProviderMetadata (base64-decoded) — confirmed against a real API
// response: Gemini returns a 400 ("Function call is missing a
// thought_signature...") on the very next turn to the SAME provider if
// it's dropped, not just degraded quality. A call with no ProviderMetadata
// to echo — older stored history predating this fix, a synthetic call
// built elsewhere (e.g. router.appendEscalationTurn's handoff
// acknowledgment), or a call replayed to a DIFFERENT Gemini provider after
// an escalation hop (a signature is only ever verified against the exact
// model that emitted it) — falls back to geminiThoughtSignaturePlaceholder
// instead of an empty value, which hits the identical 400 otherwise.
// Google's own docs for this exact case (a function call "executed
// deterministically by the client" or transferred from a different model)
// name this literal sentinel as the documented bypass — see
// https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures.
var geminiThoughtSignaturePlaceholder = []byte("context_engineering_is_the_way_to_go")

func toGeminiContents(msgs []llm.Message) (*genai.Content, []*genai.Content) {
	var systemParts []*genai.Part
	var out []*genai.Content

	toolNames := make(map[string]string) // ToolCall.ID -> ToolCall.Name
	for _, m := range msgs {
		switch m.Role {
		case llm.RoleSystem:
			systemParts = append(systemParts, &genai.Part{Text: m.Content})

		case llm.RoleUser:
			var userParts []*genai.Part
			if m.Content != "" {
				userParts = append(userParts, &genai.Part{Text: m.Content})
			}
			for _, p := range m.Parts {
				if p.ImageBase64 != "" {
					// Best-effort: a corrupt/foreign blob just means the
					// image is silently skipped rather than failing the
					// whole request.
					imgBytes, err := base64.StdEncoding.DecodeString(p.ImageBase64)
					if err == nil {
						userParts = append(userParts, &genai.Part{
							InlineData: &genai.Blob{
								MIMEType: p.MIMEType,
								Data:     imgBytes,
							},
						})
					}
				}
			}
			if len(userParts) == 0 {
				// Gemini requires at least one part; fall back to an empty
				// text part rather than producing a structurally invalid
				// Content.
				userParts = []*genai.Part{{Text: ""}}
			}
			out = append(out, &genai.Content{Role: genai.RoleUser, Parts: userParts})

		case llm.RoleTool:
			out = append(out, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					ID:       m.ToolCallID,
					Name:     toolNames[m.ToolCallID],
					Response: map[string]any{"result": m.Content},
				},
			}}})

		case llm.RoleAssistant:
			var parts []*genai.Part
			if m.Content != "" {
				parts = append(parts, &genai.Part{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				toolNames[tc.ID] = tc.Name
				var args map[string]any
				if tc.Arguments != "" {
					// Best-effort: malformed arguments just become a nil
					// map rather than failing the whole request — same
					// posture as anthropic's tool_use handling.
					_ = json.Unmarshal([]byte(tc.Arguments), &args)
				}
				thoughtSignature := geminiThoughtSignaturePlaceholder
				if tc.ProviderMetadata != "" {
					// Best-effort: a malformed/foreign-provider blob falls
					// back to the placeholder too, same as no metadata at
					// all — mirrors the Arguments unmarshal above, except
					// an empty signature is a hard 400 here, not a
					// silently degraded field, so there's no safe empty
					// fallback.
					if decoded, err := base64.StdEncoding.DecodeString(tc.ProviderMetadata); err == nil {
						thoughtSignature = decoded
					}
				}
				parts = append(parts, &genai.Part{
					FunctionCall:     &genai.FunctionCall{ID: tc.ID, Name: tc.Name, Args: args},
					ThoughtSignature: thoughtSignature,
				})
			}
			out = append(out, &genai.Content{Role: genai.RoleModel, Parts: parts})
		}
	}

	var system *genai.Content
	if len(systemParts) > 0 {
		system = &genai.Content{Parts: systemParts}
	}
	return system, out
}

// buildTools assembles native tools (GoogleSearch, its own *genai.Tool
// entry per the REST API's rule that built-in tools can't share an entry
// with function declarations or each other) plus one *genai.Tool carrying
// every custom (caller-defined) tool's FunctionDeclaration. Unlike Claude's
// SDK, genai.FunctionDeclaration exposes a raw-JSON-Schema passthrough
// field (ParametersJsonSchema) that accepts llm.ToolDef.Parameters
// directly, so no Schema-struct conversion is needed even for often-nested
// tool schemas.
func (p *Provider) buildTools(custom []llm.ToolDef) []*genai.Tool {
	var out []*genai.Tool

	if len(custom) > 0 {
		decls := make([]*genai.FunctionDeclaration, 0, len(custom))
		for _, t := range custom {
			decls = append(decls, &genai.FunctionDeclaration{
				Name:                 t.Name,
				Description:          t.Description,
				ParametersJsonSchema: t.Parameters,
			})
		}
		out = append(out, &genai.Tool{FunctionDeclarations: decls})
	}

	out = append(out, p.nativeTools()...)
	return out
}

// nativeTools builds Gemini's own server-executed tools — mirrors
// anthropic.Provider.nativeTools. Unlike everything in buildTools' custom
// half, the model's use of these never round-trips through the caller's own
// tool-execution loop; Gemini resolves them server-side within the same
// streamed response.
//
// Code execution (genai.ToolCodeExecution) is deliberately NOT wired here:
// the SDK's source comment marks it, consistently with ~98 other
// genuinely-Vertex-only fields/types in that file, as unsupported outside
// Vertex AI on the classic generateContent/streamGenerateContent API this
// Provider calls via Models.GenerateContentStream. GoogleSearch has no such
// restriction (confirmed: no matching comment on its type or the fields
// this Provider sets), so grounding is unaffected.
func (p *Provider) nativeTools() []*genai.Tool {
	var out []*genai.Tool
	if p.tools.GoogleSearch {
		out = append(out, &genai.Tool{GoogleSearch: &genai.GoogleSearch{}})
	}
	return out
}
