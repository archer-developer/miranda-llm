package anomaly

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/archer-developer/miranda-llm/llmtrace/analyze"
)

// Anomaly kinds Detect can report.
const (
	KindSlowCall         = "slow_call"
	KindRepeatedToolCall = "repeated_tool_call"
	KindUnknownTool      = "unknown_tool"
	KindInvalidArguments = "invalid_arguments"
	KindToolError        = "tool_error"
	KindIterationCap     = "iteration_cap"
	KindTimeout          = "timeout"
	KindEmptyResponse    = "empty_response"
)

// Anomaly describes one issue found within a turn's trace blocks.
type Anomaly struct {
	Kind   string
	Detail string
}

// Outcome carries facts about how a turn ended that can't be reliably
// derived from its trace blocks alone — only the agent loop itself
// (Asker.Ask / Orchestrator.Handle) knows these authoritatively, since
// analyze.EndsOnToolCallWithNoAnswer's "cut off mid-loop" shape is
// deliberately ambiguous between an iteration cap and a timeout.
type Outcome struct {
	HitIterationCap bool
	TimedOut        bool
	IterationCount  int
	MaxIterations   int
}

// DefaultSlowCallThreshold is used when Options.SlowCallThreshold is zero.
const DefaultSlowCallThreshold = 20 * time.Second

// Options tunes Detect's thresholds.
type Options struct {
	// SlowCallThreshold flags a block whose measured duration (see
	// durations, Detect's second parameter) exceeds this.
	SlowCallThreshold time.Duration
}

// Detect inspects one turn's blocks (in call order, as recorded by a
// Recorder) plus its Outcome and reports every anomaly found. durations, if
// non-nil, must be the same length as blocks (see Recorder.Durations) — pass
// nil to skip slow-call detection, e.g. for blocks reconstructed from disk
// rather than a live Recorder, which carry no comparable timing.
func Detect(blocks []analyze.Block, durations []time.Duration, outcome Outcome, opts Options) []Anomaly {
	var anomalies []Anomaly

	threshold := opts.SlowCallThreshold
	if threshold <= 0 {
		threshold = DefaultSlowCallThreshold
	}
	for i, d := range durations {
		if i >= len(blocks) {
			break
		}
		if d > threshold {
			anomalies = append(anomalies, Anomaly{
				Kind: KindSlowCall,
				Detail: fmt.Sprintf("call %d (%s) took %s (threshold %s)",
					i+1, blocks[i].Provider, d.Round(time.Millisecond), threshold),
			})
		}
	}

	anomalies = append(anomalies, detectRepeatedToolCalls(blocks)...)
	anomalies = append(anomalies, detectToolResultErrors(blocks)...)
	anomalies = append(anomalies, detectEmptyFinalResponse(blocks)...)

	if outcome.HitIterationCap {
		anomalies = append(anomalies, Anomaly{
			Kind:   KindIterationCap,
			Detail: fmt.Sprintf("hit the %d-iteration cap without a final reply", outcome.MaxIterations),
		})
	}
	if outcome.TimedOut {
		anomalies = append(anomalies, Anomaly{
			Kind:   KindTimeout,
			Detail: "turn's context deadline was exceeded",
		})
	}

	return anomalies
}

// detectRepeatedToolCalls flags any (tool name, canonicalized arguments)
// pair the model called more than once across the turn's blocks — the
// classic "stuck retrying the same call" shape.
func detectRepeatedToolCalls(blocks []analyze.Block) []Anomaly {
	seen := map[string]int{}
	var order []string
	for _, b := range blocks {
		for _, tc := range analyze.ExtractToolCalls(b) {
			key := tc.Name + "\x00" + canonicalizeArgs(tc.Args)
			if seen[key] == 0 {
				order = append(order, key)
			}
			seen[key]++
		}
	}

	var out []Anomaly
	for _, key := range order {
		if seen[key] <= 1 {
			continue
		}
		name, args, _ := strings.Cut(key, "\x00")
		out = append(out, Anomaly{
			Kind: KindRepeatedToolCall,
			Detail: fmt.Sprintf("%s called %d times with identical arguments: %s",
				name, seen[key], analyze.Truncate(args, 200)),
		})
	}
	return out
}

// detectEmptyFinalResponse flags a turn whose very last call returned
// neither text nor a tool call despite ending without an error — the model
// had the floor to answer and produced nothing, which the agent loop can't
// tell apart from a genuine (if unhelpful) empty reply, so it goes straight
// to the end user as silence. Only the last block is checked: runAgentLoop
// treats any response with zero tool calls as the turn's final answer and
// returns immediately (internal/agent_loop/agent_loop.go), so this shape
// can structurally only ever occur on a turn's last call — an empty
// response mid-turn (with tool calls still following) is a different,
// already-covered shape (see EndsOnToolCallWithNoAnswer/KindIterationCap).
func detectEmptyFinalResponse(blocks []analyze.Block) []Anomaly {
	if len(blocks) == 0 {
		return nil
	}
	last := blocks[len(blocks)-1]
	if !analyze.IsEmptyResponse(last) {
		return nil
	}
	return []Anomaly{{
		Kind:   KindEmptyResponse,
		Detail: fmt.Sprintf("call %d: provider returned neither text nor a tool call — the user received an empty reply", len(blocks)),
	}}
}

// canonicalizeArgs re-marshals arg JSON with sorted keys (encoding/json
// always sorts map keys when marshaling) so two semantically-identical
// argument sets that only differ in key order still compare equal. Falls
// back to the raw string for non-JSON/unparseable arguments.
func canonicalizeArgs(args string) string {
	var v any
	if json.Unmarshal([]byte(args), &v) != nil {
		return args
	}
	canon, err := json.Marshal(v)
	if err != nil {
		return args
	}
	return string(canon)
}

// detectToolResultErrors scans each block (from the second one on — the
// first block's request predates this turn's own tool calls, so it can only
// carry earlier turns' already-handled results) for the "error: ..."-prefixed
// tool-result convention both services use (internal/ask/agent_loop.go's
// executeToolCall, internal/agent_loop's tool_dispatch.go/mcp_dispatch.go),
// classifying each into the specific anomaly kind its wording indicates.
func detectToolResultErrors(blocks []analyze.Block) []Anomaly {
	var out []Anomaly
	for i := 1; i < len(blocks); i++ {
		entries, ok := analyze.DescribeIncoming(blocks[i], false)
		if !ok {
			continue
		}
		for _, entry := range entries {
			idx := strings.Index(entry, "error:")
			if idx < 0 {
				continue
			}
			errText := entry[idx:]
			out = append(out, Anomaly{
				Kind:   errorTextKind(errText),
				Detail: fmt.Sprintf("call %d: %s", i+1, analyze.Truncate(entry, 220)),
			})
		}
	}
	return out
}

// errorTextKind classifies one tool-result "error: ..." string, matching the
// specific wording each service's own tool dispatch code uses today:
// medical-card's `"error: unknown tool %q"` (internal/ask/agent_loop.go) and
// Miranda's `"mcp: no configured server matches tool %q"`
// (internal/mcp/mcp.go, surfaced via mcp_dispatch.go) both become
// KindUnknownTool; argument decode/parse failures (medical-card's
// decodeKnowledgeRequest, Miranda's built-in tools' `"invalid arguments"`)
// become KindInvalidArguments; anything else "error:"-prefixed falls back to
// the KindToolError catch-all.
func errorTextKind(text string) string {
	lower := strings.ToLower(text)
	switch {
	case strings.Contains(lower, "unknown tool"), strings.Contains(lower, "no configured server matches tool"):
		return KindUnknownTool
	case strings.Contains(lower, "invalid arguments"), strings.Contains(lower, "decode"), strings.Contains(lower, "parse from/to date"):
		return KindInvalidArguments
	default:
		return KindToolError
	}
}
