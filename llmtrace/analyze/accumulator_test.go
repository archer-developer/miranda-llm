package analyze

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestAccumulator_FeedLineByLine_MatchesParseAll covers the actual reason
// Accumulator exists as its own type rather than a private detail of
// ParseAll: Miranda's web UI backend feeds lines one at a time as they're
// published to the hub (live, not from a byte slice it can hand to
// ParseAll), and must reassemble the exact same Block ParseAll would.
func TestAccumulator_FeedLineByLine_MatchesParseAll(t *testing.T) {
	log := `=== 2026-08-12T18:14:45Z provider=gemini-agent conversation=session_1 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Вопрос"}]}]}
--- response ---
{"text":"Ответ.","tool_calls":null}

`
	want, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, want, 1)

	acc := NewAccumulator()
	var got []Block
	for _, line := range strings.Split(strings.TrimRight(log, "\n"), "\n") {
		if block, ok := acc.Feed(line); ok {
			got = append(got, *block)
		}
	}
	// The trailing blank line that terminates the block was trimmed off by
	// TrimRight above along with the log's own trailing blank line, so feed
	// it back explicitly the way a real line-oriented source (a WebSocket
	// event per line, a bufio.Scanner) always would.
	if block, ok := acc.Feed(""); ok {
		got = append(got, *block)
	}

	require.Equal(t, want, got)
}

func TestAccumulator_MidBlockLinesReturnNothing(t *testing.T) {
	acc := NewAccumulator()
	lines := []string{
		"=== 2026-08-12T18:14:45Z provider=gemini-agent ===",
		"--- request ---",
		`{"contents":[]}`,
		"--- response ---",
		`{"text":"ok"}`,
	}
	for _, line := range lines {
		block, ok := acc.Feed(line)
		require.False(t, ok, "no block should complete before the terminating blank line")
		require.Nil(t, block)
	}
}

func TestAccumulator_StrayContentLineWithNoOpenBlockIsDropped(t *testing.T) {
	acc := NewAccumulator()
	block, ok := acc.Feed(`{"leftover": "tail of a block whose header already scrolled away"}`)
	require.False(t, ok)
	require.Nil(t, block)
}

// TestAccumulator_HeaderResetsAnUnclosedBlock documents the defensive
// behavior noted on Accumulator's own doc comment: this should never happen
// in practice (Logger.Trace publishes a whole block's lines from one
// synchronous write), but a new header line always wins over an in-progress
// one rather than the accumulator getting stuck.
func TestAccumulator_HeaderResetsAnUnclosedBlock(t *testing.T) {
	acc := NewAccumulator()
	_, _ = acc.Feed("=== 2026-08-12T18:14:45Z provider=gemini-agent ===")
	_, _ = acc.Feed("--- request ---")
	_, _ = acc.Feed(`{"contents":[]}`)
	// No terminating blank line — a second header arrives instead.
	_, _ = acc.Feed("=== 2026-08-12T18:14:50Z provider=gemini-agent conversation=session_2 ===")
	_, _ = acc.Feed("--- request ---")
	_, _ = acc.Feed(`{"contents":[]}`)
	_, _ = acc.Feed("--- response ---")
	_, _ = acc.Feed(`{"text":"ok"}`)
	block, ok := acc.Feed("")
	require.True(t, ok)
	require.Equal(t, "session_2", block.Conversation)
}

func TestAccumulator_MalformedTimestampIsIgnored(t *testing.T) {
	acc := NewAccumulator()
	_, ok := acc.Feed("=== not-a-timestamp provider=gemini-agent ===")
	require.False(t, ok)
	// The malformed header must not leave a dangling open block behind —
	// content immediately after it is dropped as stray, same as if no
	// header had matched at all.
	_, ok = acc.Feed(`{"contents":[]}`)
	require.False(t, ok)
}
