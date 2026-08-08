package keyrotation

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

var testLogger = slog.New(slog.DiscardHandler)

var errRetryable = errors.New("retryable")
var errFatal = errors.New("fatal")

func isRetryable(err error) bool { return errors.Is(err, errRetryable) }

func TestRun_SucceedsOnFirstKey(t *testing.T) {
	var attempts []int
	err := Run(context.Background(), testLogger, "test", 3, Config{}, isRetryable,
		func(ctx context.Context, i int) error {
			attempts = append(attempts, i)
			return nil
		},
	)
	require.NoError(t, err)
	require.Equal(t, []int{0}, attempts, "must stop at the first successful key")
}

func TestRun_RotatesOnRetryableError(t *testing.T) {
	var attempts []int
	err := Run(context.Background(), testLogger, "test", 3, Config{}, isRetryable,
		func(ctx context.Context, i int) error {
			attempts = append(attempts, i)
			if i < 2 {
				return errRetryable
			}
			return nil
		},
	)
	require.NoError(t, err)
	require.Equal(t, []int{0, 1, 2}, attempts)
}

func TestRun_StopsImmediatelyOnNonRetryableError(t *testing.T) {
	var attempts []int
	err := Run(context.Background(), testLogger, "test", 3, Config{}, isRetryable,
		func(ctx context.Context, i int) error {
			attempts = append(attempts, i)
			return errFatal
		},
	)
	require.ErrorIs(t, err, errFatal)
	require.Equal(t, []int{0}, attempts, "a non-retryable error must not try the remaining keys")
}

func TestRun_RetriesAcrossCyclesWithCooldown(t *testing.T) {
	var attempts []int
	err := Run(context.Background(), testLogger, "test", 2, Config{Cycles: 2, Cooldown: 0}, isRetryable,
		func(ctx context.Context, i int) error {
			attempts = append(attempts, i)
			// Succeed only on the second cycle's first key.
			if len(attempts) == 3 {
				return nil
			}
			return errRetryable
		},
	)
	require.NoError(t, err)
	require.Equal(t, []int{0, 1, 0}, attempts, "cycle 1 tries both keys, cycle 2 succeeds on the first")
}

func TestRun_ExhaustsAllCyclesAndWrapsLastError(t *testing.T) {
	var attempts int
	err := Run(context.Background(), testLogger, "test", 2, Config{Cycles: 2, Cooldown: 0}, isRetryable,
		func(ctx context.Context, i int) error {
			attempts++
			return errRetryable
		},
	)
	require.Error(t, err)
	require.ErrorIs(t, err, errRetryable, "the wrapped exhaustion error must still satisfy errors.Is against the underlying sentinel")
	require.Contains(t, err.Error(), "exhausted")
	require.Equal(t, 4, attempts, "2 keys x 2 cycles")
}

func TestRun_ZeroOrNegativeCyclesTreatedAsOne(t *testing.T) {
	var attempts int
	err := Run(context.Background(), testLogger, "test", 1, Config{Cycles: 0}, isRetryable,
		func(ctx context.Context, i int) error {
			attempts++
			return errRetryable
		},
	)
	require.Error(t, err)
	require.Equal(t, 1, attempts, "Cycles <= 0 must mean exactly one pass, not a retry loop")
}

func TestRun_ContextCancelledDuringCooldownReturnsCtxErr(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(5 * time.Millisecond)
		cancel()
	}()

	err := Run(ctx, testLogger, "test", 1, Config{Cycles: 2, Cooldown: time.Hour}, isRetryable,
		func(ctx context.Context, i int) error {
			return errRetryable
		},
	)
	require.ErrorIs(t, err, context.Canceled)
}
