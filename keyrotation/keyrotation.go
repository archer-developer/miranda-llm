// Package keyrotation implements the "try each of several API keys, and if
// every key in a pass fails, sleep and retry the whole list again" loop
// shared by any provider that rotates across multiple API keys for the same
// backend (e.g. gemini.Provider, whose free-tier keys have a low per-key
// quota) — callers differ only in what one attempt actually does and which
// errors are worth rotating for.
//
// On top of that, Banlist gives a provider a way to remember, ACROSS
// separate Run calls (i.e. across separate chat/TTS requests, not just
// within one), that a given key was recently seen exhausted or overloaded —
// so a deployment that deliberately orders a free-tier key before a paid
// one (or vice versa) fails over to the other key immediately on the very
// next request, instead of re-discovering the same failure from scratch
// every time.
package keyrotation

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// Config tunes how many keys get retried and how long to wait between full
// passes over the key list. Cycles <= 0 is treated as 1 (try every key
// exactly once, no retry pass).
type Config struct {
	Cycles   int
	Cooldown time.Duration
}

// Banlist tracks temporary per-key-index bans, shared by a single provider
// instance across every Run call it makes. A *Banlist is safe for
// concurrent use; a nil *Banlist is also safe to use (Ban is a no-op,
// Banned always reports false) so a caller that doesn't want banning can
// simply not construct one, rather than every call site having to special-
// case it.
type Banlist struct {
	mu    sync.Mutex
	until map[int]time.Time
}

// NewBanlist returns an empty Banlist, ready to share across every Run call
// a single provider instance makes for its lifetime.
func NewBanlist() *Banlist {
	return &Banlist{until: make(map[int]time.Time)}
}

// Ban marks keyIndex unavailable for d from now. d <= 0 is a no-op — Run's
// caller is expected to pass 0 for errors that aren't worth banning over
// (e.g. a per-key auth failure isn't a "comes back on its own" condition a
// timed ban would help with).
func (b *Banlist) Ban(keyIndex int, d time.Duration) {
	if b == nil || d <= 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.until[keyIndex] = time.Now().Add(d)
}

// Banned reports whether keyIndex is currently banned, lazily forgetting an
// expired entry so the map doesn't grow unbounded over a long-lived
// provider's lifetime.
func (b *Banlist) Banned(keyIndex int) bool {
	if b == nil {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	until, ok := b.until[keyIndex]
	if !ok {
		return false
	}
	if !time.Now().Before(until) {
		delete(b.until, keyIndex)
		return false
	}
	return true
}

// Run tries n keys in order (index 0..n-1 passed to attempt), once per
// pass, for up to cfg.Cycles passes with cfg.Cooldown between them. A key
// currently banned on bans (see Banlist) is skipped without being tried.
//
// attempt returning nil ends Run successfully right away. A non-nil error
// is passed to classify, which reports:
//   - retryable: true moves on to the next key (or, if every non-banned key
//     in this pass failed, sleeps and starts a new pass); false stops
//     immediately and Run returns that error verbatim — unwrapped, so any
//     sentinel the caller's own error wraps survives untouched.
//   - ban: if > 0, keyIndex is banned on bans for that long (0 bans no one —
//     e.g. a per-key auth failure is retryable but isn't a "wait and it'll
//     recover" condition).
//
// If every key is currently banned at the start of a pass, Run returns an
// error immediately rather than sleeping cfg.Cooldown first — a ban's own
// expiry, not this loop's cooldown, is what recovers a key, so sleeping
// here would only delay a caller's own cross-provider escalation for no
// benefit.
//
// Once every configured cycle is exhausted without a single key succeeding,
// Run wraps the last error with the count of keys/cycles tried (still
// satisfying errors.Is against whatever sentinel lastErr itself wraps, via
// %w), for a message that names how much was attempted.
//
// label and logger are used only for the one cross-cutting log line this
// loop owns ("cooling down before retry") — per-key success/failure
// logging is each attempt closure's own responsibility, since only it
// knows enough to log a domain-specific message or fields only the caller
// has.
func Run(ctx context.Context, logger *slog.Logger, label string, n int, cfg Config, bans *Banlist,
	classify func(error) (retryable bool, ban time.Duration),
	attempt func(ctx context.Context, keyIndex int) error) error {
	cycles := cfg.Cycles
	if cycles <= 0 {
		cycles = 1
	}

	var lastErr error
	for cycle := 0; cycle < cycles; cycle++ {
		triedAny := false
		for i := 0; i < n; i++ {
			if bans.Banned(i) {
				continue
			}
			triedAny = true
			err := attempt(ctx, i)
			if err == nil {
				return nil
			}
			retryable, ban := classify(err)
			bans.Ban(i, ban)
			if !retryable {
				return err
			}
			lastErr = err
		}
		if !triedAny {
			if lastErr != nil {
				return fmt.Errorf("%s: all %d configured key(s) are currently banned: %w", label, n, lastErr)
			}
			return fmt.Errorf("%s: all %d configured key(s) are currently banned", label, n)
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

// FirstStepTimeout runs step — a single blocking "advance" of a stream
// (e.g. an SDK stream's Next()) — in a goroutine and races it against
// timeout (timeout <= 0 disables the race and just calls step directly) and
// ctx.
//
// This is meant to be used only for the very FIRST step of a streaming
// attempt, before any output has been forwarded to a caller — a slow
// time-to-first-byte is a reasonable proxy for "this key's backend looks
// overloaded" precisely because nothing has been forwarded yet, so treating
// it as a failure and rotating to another key can't duplicate or garble
// output the way retrying after a mid-stream timeout could (see each
// provider's own attempt() doc comment for why a post-forward retry is
// unsafe).
//
// ok is step's own return value, and is only meaningful when abandoned is
// false. abandoned=true means step's goroutine may STILL be running when
// FirstStepTimeout returns (the timeout elapsed, or ctx was cancelled,
// before step finished) — the caller must not touch whatever state step
// reads/writes (the stream itself, an accumulator, etc.) again without
// first ensuring that goroutine has actually finished; callers that don't
// need the abandoned call's result at all (e.g. because they immediately
// return an error and never touch the stream again) don't need to wait for
// anything, since the ignored result is simply dropped, but callers that DO
// need to release resources tied to the stream (e.g. iter.Pull2's stop)
// must derive ctx so cancelling it (e.g. via a context.WithCancel wrapped
// around whatever produced the stream) unblocks step promptly, then wait
// for it in a background goroutine before calling anything else on that
// stream. Check ctx.Err() to tell the two abandoned cases apart: non-nil
// means ctx itself was cancelled (the caller should usually just propagate
// ctx.Err()); nil means the configured timeout elapsed on its own.
func FirstStepTimeout(ctx context.Context, timeout time.Duration, step func() bool) (ok, abandoned bool) {
	if timeout <= 0 {
		return step(), false
	}
	done := make(chan bool, 1)
	go func() { done <- step() }()
	select {
	case ok := <-done:
		return ok, false
	case <-time.After(timeout):
		return false, true
	case <-ctx.Done():
		return false, true
	}
}

// BanIfSlow bans keyIndex on bans for ban and reports true if elapsed
// exceeds threshold; otherwise it does nothing and reports false.
//
// This is the post-hoc counterpart to FirstStepTimeout: that guard can only
// judge (and abort) a key before anything has been forwarded to the caller,
// so a key whose backend answers within the timeout but then takes far
// longer than that same budget to actually finish streaming the rest of its
// response never trips it — see each provider's attempt() doc comment for
// why aborting/retrying a call that has already forwarded output isn't
// safe. Observed in practice (2026-09-27): a gemini-lite free-tier key
// answered its first event promptly but then took 16s and 56s end-to-end on
// two separate calls, well past the 10s overload timeout, and kept getting
// reused because nothing checked total duration once streaming had started.
//
// Called once an attempt has already completed (successfully or not), this
// still pulls a persistently slow key from rotation for future calls,
// without touching the outcome of the call that measured it — the response
// already forwarded to the caller stands as delivered either way.
func BanIfSlow(bans *Banlist, keyIndex int, elapsed, threshold, ban time.Duration) bool {
	if elapsed <= threshold {
		return false
	}
	bans.Ban(keyIndex, ban)
	return true
}
