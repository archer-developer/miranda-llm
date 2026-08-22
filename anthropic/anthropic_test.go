package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
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

// TestToAnthropicMessages_ToolCallWithEmptyOrMalformedArgumentsStillSendsInput
// reproduces a production 400 ("messages.N.content.0.tool_use.input: Field
// required"): a synthetic escalation tool call (router.escalateOnError) or
// any tool call with empty/invalid Arguments used to leave input as a nil
// interface, and the SDK drops "input" from the wire request entirely for a
// nil input — Anthropic requires the field on every tool_use block, even an
// empty object, and rejects the whole request otherwise.
func TestToAnthropicMessages_ToolCallWithEmptyOrMalformedArgumentsStillSendsInput(t *testing.T) {
	for _, args := range []string{"", "not valid json", `{"reason":"has a literal` + "\n" + `newline"}`} {
		_, msgs := toAnthropicMessages([]llm.Message{
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "escalate_to_claude", Arguments: args}}},
		})
		require.Len(t, msgs, 1)

		raw, err := json.Marshal(msgs[0])
		require.NoError(t, err)

		var decoded map[string]any
		require.NoError(t, json.Unmarshal(raw, &decoded))
		content, ok := decoded["content"].([]any)
		require.True(t, ok)
		require.Len(t, content, 1)
		block, ok := content[0].(map[string]any)
		require.True(t, ok)
		require.Contains(t, block, "input", "tool_use block must always carry an input field, got Arguments=%q", args)
	}
}

// TestToAnthropicMessages_SingleSystemBlockGetsCacheBreakpoint covers the
// common case (one caller-supplied RoleSystem message): it must still be
// marked, since it's both first and last.
func TestToAnthropicMessages_SingleSystemBlockGetsCacheBreakpoint(t *testing.T) {
	system, _ := toAnthropicMessages([]llm.Message{
		{Role: llm.RoleSystem, Content: "You are Miranda."},
		{Role: llm.RoleUser, Content: "hi"},
	})
	require.Len(t, system, 1)
	require.Equal(t, anthropic.NewCacheControlEphemeralParam(), system[0].CacheControl)
}

// TestToAnthropicMessages_CachesFirstSystemBlockNotLast reproduces the bug
// this ADR fixes (docs/adr/system-prompt-caching.md in the miranda repo): a
// caller sending a stable prefix (persona/memory) followed by a volatile
// suffix (current time, different on every turn) needs the breakpoint on
// the STABLE block. The old "mark the last block" behavior would instead
// mark the volatile one — a breakpoint that can never hit twice, since its
// content differs on every call — silently defeating caching for any
// caller that adopts a multi-block system prompt.
func TestToAnthropicMessages_CachesFirstSystemBlockNotLast(t *testing.T) {
	system, _ := toAnthropicMessages([]llm.Message{
		{Role: llm.RoleSystem, Content: "You are Miranda. Shared memory: ..."},
		{Role: llm.RoleSystem, Content: "Current time: 2026-08-14 12:00 MSK."},
		{Role: llm.RoleUser, Content: "hi"},
	})
	require.Len(t, system, 2)
	require.Equal(t, anthropic.NewCacheControlEphemeralParam(), system[0].CacheControl,
		"the stable (first) block must carry the breakpoint")
	require.Zero(t, system[1].CacheControl,
		"the volatile (last) block must NOT carry a breakpoint — its content differs every turn, so marking it would never hit")
}
