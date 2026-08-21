package anomaly

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestRecorder_AccumulatesOneBlockPerTraceCall(t *testing.T) {
	r := NewRecorder("conv-1")

	r.Trace(context.Background(), "gemini", `{"contents":[]}`, `{"text":"hi"}`, nil)
	r.Trace(context.Background(), "gemini", `{"contents":[]}`, `{"text":"there"}`, nil)

	blocks := r.Blocks()
	require.Len(t, blocks, 2)
	require.Equal(t, "conv-1", blocks[0].Conversation)
	require.Equal(t, "gemini", blocks[0].Provider)
	require.Contains(t, string(blocks[0].ResponseJSON), "hi")
	require.Contains(t, string(blocks[1].ResponseJSON), "there")
}

func TestRecorder_RecordsErrorBlocks(t *testing.T) {
	r := NewRecorder("")

	r.Trace(context.Background(), "anthropic", "req", "", errors.New("boom"))

	blocks := r.Blocks()
	require.Len(t, blocks, 1)
	require.True(t, blocks[0].IsError)
	require.Equal(t, "boom", blocks[0].ErrorText)
}

func TestRecorder_DurationsParallelBlocksAndAreNonNegative(t *testing.T) {
	r := NewRecorder("")

	r.Trace(context.Background(), "gemini", "req", "resp", nil)
	time.Sleep(5 * time.Millisecond)
	r.Trace(context.Background(), "gemini", "req", "resp", nil)

	durations := r.Durations()
	require.Len(t, durations, 2)
	require.GreaterOrEqual(t, durations[1], 5*time.Millisecond)
}

func TestRecorder_SafeForConcurrentUse(t *testing.T) {
	r := NewRecorder("conv-1")

	done := make(chan struct{})
	for i := 0; i < 20; i++ {
		go func() {
			r.Trace(context.Background(), "gemini", "req", "resp", nil)
			done <- struct{}{}
		}()
	}
	for i := 0; i < 20; i++ {
		<-done
	}

	require.Len(t, r.Blocks(), 20)
	require.Len(t, r.Durations(), 20)
}
