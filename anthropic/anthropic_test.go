package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	llm "github.com/archer-developer/miranda-llm"
)

// TestToAnthropicTools_NoPropertiesKeyStillSendsInputSchema reproduces a
// production 400 ("tools.0.custom.input_schema: Field required"): a tool
// whose JSON schema has no "properties" key at all (e.g. a no-argument tool
// like {"type":"object","additionalProperties":false}) left Properties as a
// nil interface, which made the whole anthropic.ToolInputSchemaParam struct
// the Go zero value — the SDK's `omitzero` tag then dropped "input_schema"
// from the wire request entirely instead of sending an empty one.
func TestToAnthropicTools_NoPropertiesKeyStillSendsInputSchema(t *testing.T) {
	tools := toAnthropicTools([]llm.ToolDef{
		{
			Name:        "code_exec_sandbox_create_session",
			Description: "Start a session",
			Parameters:  map[string]any{"type": "object", "additionalProperties": false},
		},
	})
	require.Len(t, tools, 1)

	raw, err := tools[0].MarshalJSON()
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))
	require.Contains(t, decoded, "input_schema", "input_schema must always be sent, even for a tool with no properties")

	schema, ok := decoded["input_schema"].(map[string]any)
	require.True(t, ok, "input_schema must be an object")
	require.Equal(t, "object", schema["type"])
}

// TestRequiredFields covers the JSON Schema "required" array extraction
// requiredFields does for both toAnthropicTools and Structured.
func TestRequiredFields(t *testing.T) {
	require.Equal(t, []string{"a", "b"}, requiredFields(map[string]any{
		"type":     "object",
		"required": []any{"a", "b"},
	}))
	require.Nil(t, requiredFields(map[string]any{"type": "object"}))
}
