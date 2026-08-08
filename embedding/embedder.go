// Package embedding provides the Embedder interface and its Gemini
// implementation. The interface is kept narrow — a single Embed method —
// so tests can substitute a fake (see llmtest.FakeEmbedder) without pulling
// in HTTP or any concrete SDK. Deliberately independent of the root llm
// package's Provider/ChatRequest types: embeddings are a different shape of
// call (one text in, one vector out, no conversation, no tools) and every
// consumer so far (miranda-diary's semantic search) has needed exactly this
// and nothing more.
package embedding

import "context"

// Embedder converts text into a float32 vector suitable for cosine
// similarity comparison.
type Embedder interface {
	// Embed returns the embedding for text. The returned slice length and
	// the space it lives in are determined entirely by the underlying
	// model — see GeminiEmbedder's doc comment for what that means for
	// callers that change models on a non-empty index.
	Embed(ctx context.Context, text string) ([]float32, error)
}
