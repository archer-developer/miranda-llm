// Package anomaly detects degenerate agent-loop turns — a slow LLM call, the
// model retrying a tool with identical arguments, a call to a tool that
// doesn't exist, malformed tool arguments, a tool execution error, or the
// loop hitting its iteration cap or a timeout — from the same trace blocks
// llmtrace/analyze already knows how to parse, so a turn worth a human's
// attention can be routed to its own log for later manual review instead of
// only being discoverable by chance in the middle of every other concurrent
// turn's traffic.
//
// Recorder is the piece that gets a turn's own blocks in the first place:
// router.Router.SetTracer wires one llmtrace.Logger into every Provider
// once, globally, at process startup, shared by every concurrent turn — it
// can't be swapped per in-flight call without racing. Recorder sidesteps
// that by being attached to a single turn's ctx via llmtrace.WithTracer
// (see llmtrace.ContextTracer), which tees every call made using that ctx to
// this Recorder in addition to — never instead of — the process-wide
// Logger, so the normal log file and the live web UI trace view keep
// working exactly as before.
package anomaly

import (
	"context"
	"sync"
	"time"

	"github.com/archer-developer/miranda-llm/llmtrace/analyze"
)

// Recorder implements llm.Tracer (see the root llm package), accumulating
// one analyze.Block per Trace call plus a heuristic duration for it. Attach
// a fresh Recorder to a single turn via llmtrace.WithTracer at the top of
// that turn and discard it once the turn ends — it is not meant to
// accumulate across more than one turn. Safe for concurrent use, since the
// router's own fallback/escalation machinery can make more than one
// underlying provider call — and therefore more than one Trace call — for a
// single agent-loop iteration.
type Recorder struct {
	conversationID string

	mu     sync.Mutex
	last   time.Time
	blocks []analyze.Block
	gaps   []time.Duration
}

// NewRecorder creates a Recorder ready to receive Trace calls for one turn.
// conversationID is stamped onto every Block it builds — pass whatever was
// already given to llmtrace.WithConversationID for this turn (or "" for a
// fully stateless call), since Recorder has no way to read that back out of
// ctx itself. The first recorded block's gap is measured from this call, not
// from the first Trace — a slow first LLM call is exactly as much an anomaly
// as a slow later one.
func NewRecorder(conversationID string) *Recorder {
	return &Recorder{conversationID: conversationID, last: time.Now()}
}

// Trace implements the root llm package's Tracer interface.
func (r *Recorder) Trace(ctx context.Context, provider, request, response string, err error) {
	now := time.Now()

	r.mu.Lock()
	defer r.mu.Unlock()

	gap := now.Sub(r.last)
	r.last = now
	r.blocks = append(r.blocks, analyze.NewBlock(now, provider, r.conversationID, request, response, err))
	r.gaps = append(r.gaps, gap)
}

// Blocks returns every block recorded so far, in call order.
func (r *Recorder) Blocks() []analyze.Block {
	r.mu.Lock()
	defer r.mu.Unlock()

	out := make([]analyze.Block, len(r.blocks))
	copy(out, r.blocks)
	return out
}

// Durations returns each recorded block's heuristic duration — the
// wall-clock gap since the previous Trace call on this Recorder (or since
// NewRecorder, for the first block) — parallel to Blocks(). This gap
// includes any tool-execution time between agent-loop iterations, not pure
// LLM latency, since no provider reliably reports its own call duration (see
// package anomaly's own Detect doc comment) — an accepted trade-off given
// this feeds a manual review, not an automated alert.
func (r *Recorder) Durations() []time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()

	out := make([]time.Duration, len(r.gaps))
	copy(out, r.gaps)
	return out
}
