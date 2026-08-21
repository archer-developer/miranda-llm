package anomaly

import (
	"bytes"
	"testing"
	"time"

	"github.com/archer-developer/miranda-llm/llmtrace/analyze"
	"github.com/stretchr/testify/require"
)

func TestFileName_IncludesDedupedKindsInFirstSeenOrder(t *testing.T) {
	ts := time.Date(2026, 8, 21, 12, 0, 0, 0, time.UTC)
	name := FileName(ts, []Anomaly{
		{Kind: KindSlowCall}, {Kind: KindToolError}, {Kind: KindSlowCall},
	})

	require.Equal(t, "20260821T120000Z_slow_call-tool_error.log", name)
}

func TestWriteFile_RoundTripsThroughParseAll(t *testing.T) {
	original := []analyze.Block{
		analyze.NewBlock(time.Now(), "gemini", "conv-1", `{"a":1}`, `{"b":2}`, nil),
	}
	anomalies := []Anomaly{{Kind: KindSlowCall, Detail: "call 1 took too long"}}

	var buf bytes.Buffer
	require.NoError(t, WriteFile(&buf, anomalies, original))

	require.Contains(t, buf.String(), "# anomalies detected in this turn:")
	require.Contains(t, buf.String(), "[slow_call] call 1 took too long")

	parsed, err := analyze.ParseAll(&buf)
	require.NoError(t, err)
	require.Len(t, parsed, 1)
	require.Equal(t, "gemini", parsed[0].Provider)
	require.Equal(t, "conv-1", parsed[0].Conversation)
	require.JSONEq(t, `{"a":1}`, string(parsed[0].RequestJSON))
	require.JSONEq(t, `{"b":2}`, string(parsed[0].ResponseJSON))
}
