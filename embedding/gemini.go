package embedding

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// GeminiEmbedder implements Embedder using the Google Gemini embedding API.
// The client is created once and reused across calls; it is safe for
// concurrent use.
//
// If you switch the embedding model, the new model's embedding space is
// generally incompatible with vectors already stored under the old one —
// similarity scores become meaningless when comparing across spaces. This
// package has no migration path built in; a caller changing
// embedding.model on a non-empty index needs to re-embed every existing
// record (or wipe and re-add) itself.
type GeminiEmbedder struct {
	client *genai.Client
	model  string
}

// NewGemini creates a GeminiEmbedder. apiKey is the Google AI API key;
// model is the embedding model name (e.g. "gemini-embedding-2").
func NewGemini(ctx context.Context, apiKey, model string) (*GeminiEmbedder, error) {
	// The SDK defaults to v1beta; some embedding models require the stable
	// v1 API — pinning it here avoids a per-model guessing game.
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			APIVersion: "v1",
		},
	})
	if err != nil {
		return nil, fmt.Errorf("embedding: create gemini client: %w", err)
	}
	return &GeminiEmbedder{client: client, model: model}, nil
}

// Embed returns the embedding vector for text.
func (g *GeminiEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	result, err := g.client.Models.EmbedContent(ctx, g.model,
		genai.Text(text), nil,
	)
	if err != nil {
		return nil, fmt.Errorf("embedding: gemini embed: %w", err)
	}
	if len(result.Embeddings) == 0 || len(result.Embeddings[0].Values) == 0 {
		return nil, fmt.Errorf("embedding: gemini returned no embeddings")
	}
	return result.Embeddings[0].Values, nil
}
