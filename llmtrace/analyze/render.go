package analyze

import (
	"fmt"
	"io"
	"time"
)

// RenderConversationSummary writes a one-line-per-conversation table (id,
// provider, turn count, error count, start time), plus a note about any
// untagged blocks (Document Pipeline calls and/or stateless medical.ask/
// agent-loop questions with no session/conversation id).
func RenderConversationSummary(w io.Writer, blocks []Block) {
	summaries := Summarize(blocks)
	untagged := CountUntagged(blocks)

	if len(summaries) == 0 {
		fmt.Fprintln(w, "No conversation-tagged blocks found (the agent loop wasn't called while tracing was on).")
	} else {
		fmt.Fprintf(w, "%-28s %-16s %-6s %-7s %s\n", "CONVERSATION", "PROVIDER", "TURNS", "ERRORS", "STARTED")
		for _, s := range summaries {
			fmt.Fprintf(w, "%-28s %-16s %-6d %-7d %s\n", s.ID, s.Provider, s.Turns, s.Errors, s.FirstSeen.Time.Format(time.RFC3339))
		}
	}
	if untagged > 0 {
		fmt.Fprintf(w, "\n(%d untagged block(s) — Document Pipeline calls and/or stateless agent-loop questions (no session id); pass --untagged to see them)\n", untagged)
	}
	fmt.Fprintln(w, "\nPass --conversation <id> or --latest for the turn-by-turn table.")
}

// RenderConversationDetail writes the turn-by-turn table for one tagged
// conversation.
func RenderConversationDetail(w io.Writer, blocks []Block, conversationID string) error {
	turns := ConversationBlocks(blocks, conversationID)
	if len(turns) == 0 {
		return fmt.Errorf("no blocks found for conversation %q", conversationID)
	}

	fmt.Fprintf(w, "Conversation %s — %d turn(s), provider=%s\n\n", conversationID, len(turns), turns[0].Provider)
	renderTurns(w, turns)
	return nil
}

// RenderUntagged writes the turn-by-turn table for the most recent tail
// (across every question) of untagged blocks — every call with no
// conversation id, split back into the separate stateless questions they
// actually were (see GroupByConversationStart).
func RenderUntagged(w io.Writer, blocks []Block, tail int) error {
	turns := TailUntagged(blocks, tail)
	if len(turns) == 0 {
		return fmt.Errorf("no untagged blocks found")
	}

	groups := GroupByConversationStart(turns)
	fmt.Fprintf(w, "Untagged (stateless) — last %d block(s) across %d question(s), provider=%s\n\n", len(turns), len(groups), turns[len(turns)-1].Provider)
	for i, g := range groups {
		if i > 0 {
			fmt.Fprintln(w)
		}
		fmt.Fprintf(w, "--- question at %s ---\n", g[0].Time.Format("15:04:05"))
		renderTurns(w, g)
	}
	return nil
}

// renderTurns renders turns (already sorted oldest-first) as the shared
// per-turn table both RenderConversationDetail and RenderUntagged use — one
// numbered line per turn with what it received, an indented line per tool
// call or error it produced, and a closing note if the last turn looks cut
// off (a tool call with no final answer).
func renderTurns(w io.Writer, turns []Block) {
	for i, b := range turns {
		ts := b.Time.Format("15:04:05")
		in, _ := DescribeIncoming(b, i == 0)
		if len(in) == 0 {
			fmt.Fprintf(w, "%2d %s\n", i+1, ts)
		} else {
			fmt.Fprintf(w, "%2d %s  <- %s\n", i+1, ts, in[0])
			for _, extra := range in[1:] {
				fmt.Fprintf(w, "      <- %s\n", extra)
			}
		}
		if b.ErrorText != "" {
			fmt.Fprintf(w, "      !! ERROR: %s\n", Truncate(b.ErrorText, 300))
			continue
		}
		for _, out := range DescribeOutgoing(b) {
			fmt.Fprintf(w, "      -> %s\n", out)
		}
	}

	last := turns[len(turns)-1]
	if last.ErrorText == "" && EndsOnToolCallWithNoAnswer(last) {
		fmt.Fprintln(w, "\nNote: conversation ends on a tool call with no final answer — likely hit the agent loop's iteration cap or the request was cut off before finishing.")
	}
}
