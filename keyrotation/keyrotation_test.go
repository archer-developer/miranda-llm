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

// classify wraps isRetryable into Run's new classify shape with no banning,
// for tests that only care about the retry/stop behavior Run had before
// Banlist existed.
func classify(err error) (bool, time.Duration) { return isRetryable(err), 0 }

func TestRun_SucceedsOnFirstKey(t *testing.T) {
	var attempts []int
	err := Run(context.Background(), testLogger, "test", 3, Config{}, nil, classify,
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
	err := Run(context.Background(), testLogger, "test", 3, Config{}, nil, classify,
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
	err := Run(context.Background(), testLogger, "test", 3, Config{}, nil, classify,
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
	err := Run(context.Background(), testLogger, "test", 2, Config{Cycles: 2, Cooldown: 0}, nil, classify,
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
	err := Run(context.Background(), testLogger, "test", 2, Config{Cycles: 2, Cooldown: 0}, nil, classify,
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
	err := Run(context.Background(), testLogger, "test", 1, Config{Cycles: 0}, nil, classify,
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

	err := Run(ctx, testLogger, "test", 1, Config{Cycles: 2, Cooldown: time.Hour}, nil, classify,
		func(ctx context.Context, i int) error {
			return errRetryable
		},
	)
	require.ErrorIs(t, err, context.Canceled)
}

// --- Banlist / ban-aware Run behavior ---

func TestBanlist_NilIsSafeNoOp(t *testing.T) {
	var b *Banlist
	require.False(t, b.Banned(0))
	b.Ban(0, time.Hour) // must not panic
	require.False(t, b.Banned(0))
}

func TestBanlist_BanThenBannedThenExpires(t *testing.T) {
	b := NewBanlist()
	require.False(t, b.Banned(0))
	b.Ban(0, 10*time.Millisecond)
	require.True(t, b.Banned(0))
	require.False(t, b.Banned(1), "banning key 0 must not affect key 1")
	time.Sleep(20 * time.Millisecond)
	require.False(t, b.Banned(0), "ban must lapse once its duration has passed")
}

func TestBanlist_ZeroOrNegativeDurationDoesNotBan(t *testing.T) {
	b := NewBanlist()
	b.Ban(0, 0)
	require.False(t, b.Banned(0))
	b.Ban(0, -time.Second)
	require.False(t, b.Banned(0))
}

func TestRun_SkipsBannedKeyAndTriesTheOther(t *testing.T) {
	bans := NewBanlist()
	bans.Ban(0, time.Hour)

	var attempts []int
	err := Run(context.Background(), testLogger, "test", 2, Config{}, bans, classify,
		func(ctx context.Context, i int) error {
			attempts = append(attempts, i)
			return nil
		},
	)
	require.NoError(t, err)
	require.Equal(t, []int{1}, attempts, "the banned key must never be tried")
}

func TestRun_BansKeyOnClassifiedError(t *testing.T) {
	bans := NewBanlist()
	errBannable := errors.New("bannable")
	bannableClassify := func(err error) (bool, time.Duration) {
		if errors.Is(err, errBannable) {
			return true, time.Hour
		}
		return isRetryable(err), 0
	}

	var attempts []int
	err := Run(context.Background(), testLogger, "test", 2, Config{}, bans, bannableClassify,
		func(ctx context.Context, i int) error {
			attempts = append(attempts, i)
			if i == 0 {
				return errBannable
			}
			return nil
		},
	)
	require.NoError(t, err)
	require.Equal(t, []int{0, 1}, attempts)
	require.True(t, bans.Banned(0), "an error classified with ban>0 must ban that key")
}

// --- FirstStepTimeout ---

func TestFirstStepTimeout_ReturnsPromptlyWithoutAbandoning(t *testing.T) {
	ok, abandoned := FirstStepTimeout(context.Background(), time.Second, func() bool { return true })
	require.True(t, ok)
	require.False(t, abandoned)
}

func TestFirstStepTimeout_ZeroTimeoutCallsStepDirectly(t *testing.T) {
	called := false
	ok, abandoned := FirstStepTimeout(context.Background(), 0, func() bool { called = true; return false })
	require.True(t, called)
	require.False(t, ok)
	require.False(t, abandoned)
}

func TestFirstStepTimeout_SlowStepIsAbandonedWithNilCtxErr(t *testing.T) {
	stepReturned := make(chan struct{})
	ok, abandoned := FirstStepTimeout(context.Background(), 10*time.Millisecond, func() bool {
		time.Sleep(30 * time.Millisecond)
		close(stepReturned)
		return true
	})
	require.False(t, ok)
	require.True(t, abandoned)
	require.NoError(t, context.Background().Err(), "the timeout firing must not itself cancel ctx")
	<-stepReturned // avoid leaking the goroutine past the test
}

func TestFirstStepTimeout_CtxCancelledIsAbandonedWithNonNilCtxErr(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stepReturned := make(chan struct{})
	go func() {
		time.Sleep(5 * time.Millisecond)
		cancel()
	}()
	ok, abandoned := FirstStepTimeout(ctx, time.Hour, func() bool {
		<-ctx.Done()
		close(stepReturned)
		return true
	})
	require.False(t, ok)
	require.True(t, abandoned)
	require.Error(t, ctx.Err(), "callers distinguish the two abandoned cases via ctx.Err()")
	<-stepReturned
}

func TestRun_AllKeysBannedFailsFastWithoutCooldown(t *testing.T) {
	bans := NewBanlist()
	bans.Ban(0, time.Hour)
	bans.Ban(1, time.Hour)

	start := time.Now()
	err := Run(context.Background(), testLogger, "test", 2, Config{Cycles: 3, Cooldown: time.Hour}, bans, classify,
		func(ctx context.Context, i int) error {
			t.Fatalf("attempt must not be called for a fully-banned key set")
			return nil
		},
	)
	require.Less(t, time.Since(start), time.Second, "must fail immediately, not sleep through the cooldown")
	require.Error(t, err)
	require.Contains(t, err.Error(), "banned")
}
