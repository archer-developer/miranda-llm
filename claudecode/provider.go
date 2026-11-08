package claudecode

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	llm "github.com/archer-developer/miranda-llm"
)

// ProviderType is the provider type name consuming services use for this
// backend in their own config (`type: claude_code`) and what Provider's
// trace blocks are tagged with when no Name is given.
const ProviderType = "claude_code"

// defaultCallTimeout bounds one Chat/Structured call when Config.Timeout is
// unset. Generous on purpose: a single call covers process start (~3-5s),
// and OCR of a multi-page scanned PDF on a large model takes well over a
// minute; a hung process is still killed eventually.
const defaultCallTimeout = 5 * time.Minute

// defaultMaxTurns is --max-turns for a one-shot call. Plain text needs one
// model call; --json-schema answers through a StructuredOutput tool call
// (two model calls), and the CLI retries that call itself when the value
// fails validation — 4 leaves room for one such retry.
const defaultMaxTurns = 4

// ErrToolsUnsupported is returned by Chat for a request that offers tools.
// A one-shot process can't hand a tool call back to the caller mid-stream
// the way llm.Provider's contract expects (the CLI runs its own tool loop);
// a caller that wants Claude Code to drive tools exposes them over MCP and
// drives a Process itself — see medical-card's claude_code ask engine.
var ErrToolsUnsupported = errors.New("claudecode: tool calling is not supported by the one-shot provider")

// Config configures a Provider. Mirrors Options' process-level fields; every
// field is optional.
type Config struct {
	// Binary is the CLI executable ("claude" when empty).
	Binary string
	// Model is passed as --model ("sonnet", "opus", a full id, ...).
	Model string
	// Effort is passed as --effort; "" keeps the CLI default.
	Effort string
	// Timeout is the hard ceiling for one call, process start included; the
	// process is killed when it passes. <= 0 uses defaultCallTimeout.
	Timeout time.Duration
	// MaxTurns is --max-turns per call; <= 0 uses defaultMaxTurns.
	MaxTurns int
	// AllowAPIKeyEnv — see Options.AllowAPIKeyEnv.
	AllowAPIKeyEnv bool
	// ExtraArgs — see Options.ExtraArgs.
	ExtraArgs []string
}

// Provider is an llm.Provider and llm.StructuredProvider backed by the
// headless Claude Code CLI. Every call starts a fresh process, sends the
// whole request as one message and closes the process once it has answered
// — nothing outlives the call (no session persistence, no reused context),
// so concurrent calls are fully independent and the type is safe for
// concurrent use.
//
// Limitations, by design of a one-shot CLI call:
//   - no tool calling (ErrToolsUnsupported);
//   - ChatRequest/StructuredRequest.Temperature is ignored — the CLI exposes
//     no sampling controls;
//   - Chat does not stream: the whole reply arrives as one chunk when the
//     process finishes (one-shot callers read the full text anyway).
type Provider struct {
	name   string
	cfg    Config
	logger *slog.Logger
	tracer llm.Tracer
}

// New builds a Provider. Fails if the CLI can't be found; whether it is
// logged in is not checked here (see LogAuthStatus). A nil logger falls back
// to slog.Default().
func New(name string, cfg Config, logger *slog.Logger) (*Provider, error) {
	if logger == nil {
		logger = slog.Default()
	}
	bin, err := ResolveBinary(cfg.Binary)
	if err != nil {
		return nil, fmt.Errorf("claudecode: provider %q: %w", name, err)
	}
	cfg.Binary = bin
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultCallTimeout
	}
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = defaultMaxTurns
	}
	return &Provider{name: name, cfg: cfg, logger: logger}, nil
}

// LogAuthStatus logs how the CLI is authenticated, warning when it isn't —
// every call would otherwise fail with an opaque process exit. Not run by
// New itself (it spawns `claude auth status`, ~1s): a long-running service
// calls it once at startup, a diagnostic CLI building providers on every
// invocation doesn't have to.
func LogAuthStatus(logger *slog.Logger, prefix, binary string, allowAPIKeyEnv bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	st, err := CheckAuth(ctx, binary, allowAPIKeyEnv)
	if err != nil || !st.LoggedIn {
		logger.Warn(prefix+": Claude Code CLI is not logged in — run `claude auth login` as the OS user this service runs as", "binary", binary, "error", err)
		return
	}
	logger.Info(prefix+": Claude Code CLI ready", "authMethod", st.AuthMethod, "subscription", st.SubscriptionType)
}

// Name implements llm.Provider.
func (p *Provider) Name() string { return p.name }

// SetTracer installs a tracer (router.Router.SetTracer forwards to it). Each
// call writes one block, in the Anthropic Messages shape llmtrace/analyze
// already decodes.
func (p *Provider) SetTracer(t llm.Tracer) { p.tracer = t }

// Chat implements llm.Provider — see Provider's doc comment for what it
// doesn't support.
func (p *Provider) Chat(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamChunk, error) {
	if len(req.Tools) > 0 {
		return nil, ErrToolsUnsupported
	}
	system, msg, err := renderRequest(req.Messages)
	if err != nil {
		return nil, err
	}
	if err := checkAttachments(msg); err != nil {
		return nil, err
	}
	out := make(chan llm.StreamChunk, 2)
	go func() {
		defer close(out)
		res, err := p.call(ctx, system, msg, nil, "")
		if err != nil {
			out <- llm.StreamChunk{Err: err}
			return
		}
		out <- llm.StreamChunk{TextDelta: res.Text}
		out <- llm.StreamChunk{Done: true}
	}()
	return out, nil
}

// Structured implements llm.StructuredProvider via --json-schema: the CLI
// validates the model's answer against req.Schema itself and reports the
// value as the result's structured_output.
func (p *Provider) Structured(ctx context.Context, req llm.StructuredRequest) (json.RawMessage, error) {
	if req.Schema == nil {
		return nil, fmt.Errorf("claudecode: structured: schema is required")
	}
	system, msg, err := renderRequest(req.Messages)
	if err != nil {
		return nil, err
	}
	if err := checkAttachments(msg); err != nil {
		return nil, err
	}
	res, err := p.call(ctx, system, msg, req.Schema, req.SchemaName)
	if err != nil {
		return nil, fmt.Errorf("claudecode: structured: %w", err)
	}
	if len(res.StructuredOutput) > 0 {
		return res.StructuredOutput, nil
	}
	// Not expected with --json-schema, but a bare JSON reply is still the
	// answer the caller asked for.
	if text := strings.TrimSpace(res.Text); json.Valid([]byte(text)) && strings.HasPrefix(text, "{") {
		return json.RawMessage(text), nil
	}
	return nil, fmt.Errorf("claudecode: structured: no structured output in the result (%q)", truncate(res.Text, 200))
}

// checkAttachments rejects an unsupported attachment before any process is
// started (see attachmentBlock).
func checkAttachments(msg UserMessage) error {
	for _, a := range msg.Attachments {
		if _, err := attachmentBlock(a); err != nil {
			return err
		}
	}
	return nil
}

// call runs one process for one message and closes it.
func (p *Provider) call(ctx context.Context, system string, msg UserMessage, schema map[string]any, schemaName string) (Result, error) {
	ctx, cancel := context.WithTimeout(ctx, p.cfg.Timeout)
	defer cancel()
	started := time.Now()

	proc, err := Start(Options{
		Binary:         p.cfg.Binary,
		Model:          p.cfg.Model,
		Effort:         p.cfg.Effort,
		SystemPrompt:   system,
		JSONSchema:     schema,
		MaxTurns:       p.cfg.MaxTurns,
		AllowAPIKeyEnv: p.cfg.AllowAPIKeyEnv,
		ExtraArgs:      p.cfg.ExtraArgs,
	})
	if err != nil {
		p.trace(ctx, system, msg, schemaName, Result{}, started, err)
		return Result{}, err
	}
	defer func() { _ = proc.Close() }()

	res, err := proc.Turn(ctx, msg, Handler{})
	if err == nil {
		switch {
		case res.Subtype == "error_max_turns":
			err = fmt.Errorf("claudecode: exceeded %d turns without a final answer", p.cfg.MaxTurns)
		case res.IsError:
			err = fmt.Errorf("claudecode: call failed (%s): %s", res.Subtype, truncate(res.Text, 500))
		case schema == nil && strings.TrimSpace(res.Text) == "":
			err = fmt.Errorf("claudecode: empty reply")
		}
	} else if errors.Is(err, context.DeadlineExceeded) {
		err = fmt.Errorf("claudecode: call exceeded %s: %w", p.cfg.Timeout, err)
	}
	p.trace(ctx, system, msg, schemaName, res, started, err)
	if err != nil {
		return Result{}, err
	}
	return res, nil
}

func (p *Provider) trace(ctx context.Context, system string, msg UserMessage, schemaName string, res Result, started time.Time, callErr error) {
	if p.tracer == nil {
		return
	}
	req := map[string]any{
		"model":    p.cfg.Model,
		"system":   []map[string]string{{"type": "text", "text": system}},
		"messages": []any{map[string]any{"role": "user", "content": traceContent(msg)}},
	}
	if schemaName != "" {
		req["schema_name"] = schemaName
	}
	reqJSON, _ := json.Marshal(req)
	var respJSON []byte
	if callErr == nil {
		text := res.Text
		if len(res.StructuredOutput) > 0 {
			text = string(res.StructuredOutput)
		}
		respJSON, _ = json.Marshal(map[string]any{
			"content":     []map[string]string{{"type": "text", "text": text}},
			"duration_ms": time.Since(started).Milliseconds(),
			"num_turns":   res.NumTurns,
			"cost_usd":    res.CostUSD,
			"usage":       res.Usage,
		})
	}
	p.tracer.Trace(ctx, p.name, string(reqJSON), string(respJSON), callErr)
}

// renderRequest folds an llm message list into what one CLI process takes: a
// system prompt (every RoleSystem message, in order) and a single user
// message. The common one-shot case — system + one user message — maps
// directly; anything longer (a caller replaying an earlier exchange) is
// rendered as a transcript ahead of the final user message, since a fresh
// process has no other way to see earlier turns. Attachments from every user
// message are carried along.
func renderRequest(messages []llm.Message) (string, UserMessage, error) {
	var system []string
	var rest []llm.Message
	for _, m := range messages {
		if m.Role == llm.RoleSystem {
			system = append(system, m.Content)
			continue
		}
		rest = append(rest, m)
	}
	if len(rest) == 0 || rest[len(rest)-1].Role != llm.RoleUser {
		return "", UserMessage{}, fmt.Errorf("claudecode: request must end with a user message")
	}

	var msg UserMessage
	for _, m := range rest {
		for _, part := range m.Parts {
			msg.Attachments = append(msg.Attachments, Attachment{MIMEType: part.MIMEType, Base64: part.ImageBase64})
		}
	}
	last := rest[len(rest)-1]
	if len(rest) == 1 {
		msg.Text = last.Content
	} else {
		msg.Text = "<previous_conversation>\n" + RenderTranscript(rest[:len(rest)-1], 0) + "</previous_conversation>\n\n" + last.Content
	}
	return strings.Join(system, "\n\n"), msg, nil
}

// RenderTranscript renders llm messages as plain text for a fresh process
// that must continue an earlier conversation it never saw. toolResultMax > 0
// truncates each tool result to that many runes — enough for the model to
// know what came back without re-sending a huge document.
func RenderTranscript(msgs []llm.Message, toolResultMax int) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case llm.RoleUser:
			fmt.Fprintf(&b, "[user]: %s\n", m.Content)
		case llm.RoleAssistant:
			if m.Content != "" {
				fmt.Fprintf(&b, "[assistant]: %s\n", m.Content)
			}
			for _, tc := range m.ToolCalls {
				fmt.Fprintf(&b, "[tool call] %s(%s)\n", tc.Name, tc.Arguments)
			}
		case llm.RoleTool:
			content := m.Content
			if r := []rune(content); toolResultMax > 0 && len(r) > toolResultMax {
				content = string(r[:toolResultMax]) + "…"
			}
			fmt.Fprintf(&b, "[tool result]: %s\n", content)
		}
	}
	return b.String()
}

func truncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}
