package llmtest

import (
	"context"
	"fmt"
	"sync"
)

// FakeEmbedder implements embedding.Embedder by returning a fixed vector for
// every input, or a scripted error — no real network/API call. Kept in this
// package rather than the embedding package itself so embedding has no test
// dependency of its own (mirrors why FakeProvider doesn't live in the root
// llm package).
type FakeEmbedder struct {
	mu      sync.Mutex
	vector  []float32
	err     error
	Queries []string
}

// NewFakeEmbedder returns a FakeEmbedder whose Embed always returns vector, nil.
func NewFakeEmbedder(vector []float32) *FakeEmbedder {
	return &FakeEmbedder{vector: vector}
}

// NewFakeEmbedderError returns a FakeEmbedder whose Embed always returns err.
func NewFakeEmbedderError(err error) *FakeEmbedder {
	return &FakeEmbedder{err: err}
}

// Embed implements embedding.Embedder.
func (f *FakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Queries = append(f.Queries, text)
	if f.err != nil {
		return nil, fmt.Errorf("llmtest: fake embedder: %w", f.err)
	}
	return f.vector, nil
}
