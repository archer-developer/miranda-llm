// Package keyrotation implements the "try each of several API keys, and if
// every key in a pass fails, sleep and retry the whole list again" loop
// shared by any provider that rotates across multiple API keys for the same
// backend (e.g. gemini.Provider, whose free-tier keys have a low per-key
// quota) — callers differ only in what one attempt actually does and which
// errors are worth rotating for.
package keyrotation

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// Config tunes how many keys get retried and how long to wait between full
// passes over the key list. Cycles <= 0 is treated as 1 (try every key
// exactly once, no retry pass).
type Config struct {
	Cycles   int
	Cooldown time.Duration
}

// Run tries n keys in order (index 0..n-1 passed to attempt), once per
// pass, for up to cfg.Cycles passes with cfg.Cooldown between them.
//
// attempt returning nil ends Run successfully right away. A non-nil error
// is passed to isRetryable: true moves on to the next key (or, if every key
// in this pass failed, sleeps and starts a new pass); false stops
// immediately and Run returns that error verbatim — unwrapped, so any
// sentinel the caller's own error wraps survives untouched. Once every
// configured cycle is exhausted without a single key succeeding, Run wraps
// the last error with the count of keys/cycles tried (still satisfying
// errors.Is against whatever sentinel lastErr itself wraps, via %w), for a
// message that names how much was attempted.
//
// label and logger are used only for the one cross-cutting log line this
// loop owns ("cooling down before retry") — per-key success/failure
// logging is each attempt closure's own responsibility, since only it
// knows enough to log a domain-specific message or fields only the caller
// has.
func Run(ctx context.Context, logger *slog.Logger, label string, n int, cfg Config, isRetryable func(error) bool, attempt func(ctx context.Context, keyIndex int) error) error {
	cycles := cfg.Cycles
	if cycles <= 0 {
		cycles = 1
	}

	var lastErr error
	for cycle := 0; cycle < cycles; cycle++ {
		for i := 0; i < n; i++ {
			err := attempt(ctx, i)
			if err == nil {
				return nil
			}
			if !isRetryable(err) {
				return err
			}
			lastErr = err
		}
		if cycle < cycles-1 {
			logger.Info(label+": all keys exhausted this pass, cooling down before retry",
				"cycle", cycle+1, "of_cycles", cycles, "cooldown", cfg.Cooldown)
			select {
			case <-time.After(cfg.Cooldown):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return fmt.Errorf("%s: all %d configured key(s) exhausted across %d cycle(s): %w", label, n, cycles, lastErr)
}
