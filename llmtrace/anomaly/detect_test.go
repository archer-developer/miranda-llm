package anomaly

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/archer-developer/miranda-llm/llmtrace/analyze"
	"github.com/stretchr/testify/require"
)

// The fixture types below mirror analyze/describe.go's private Gemini
// request/response shapes field-for-field (same JSON tags), so blocks built
// here decode through the real DescribeIncoming/DescribeOutgoing/
// ExtractToolCalls exactly like a real gemini.Provider trace would.

type testToolCall struct {
	Name      string `json:"Name"`
	Arguments string `json:"Arguments"`
}

type testGeminiResponse struct {
	Text      string         `json:"text"`
	ToolCalls []testToolCall `json:"tool_calls,omitempty"`
}

type testFuncCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type testFuncResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type testGeminiPart struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *testFuncCall     `json:"functionCall,omitempty"`
	FunctionResponse *testFuncResponse `json:"functionResponse,omitempty"`
}

type testGeminiContent struct {
	Role  string           `json:"role"`
	Parts []testGeminiPart `json:"parts"`
}

type testGeminiRequest struct {
	Contents []testGeminiContent `json:"contents"`
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return string(b)
}

// questionBlock is the turn's very first LLM call: just the user's question,
// no tool history yet.
func questionBlock(t *testing.T, question string) analyze.Block {
	req := testGeminiRequest{Contents: []testGeminiContent{
		{Role: "user", Parts: []testGeminiPart{{Text: question}}},
	}}
	resp := testGeminiResponse{ToolCalls: []testToolCall{{Name: "lab_results", Arguments: `{"patient":"joe"}`}}}
	return analyze.NewBlock(time.Now(), "gemini", "conv-1", mustJSON(t, req), mustJSON(t, resp), nil)
}

// toolResultBlock is a later block whose request carries one tool result
// (from the previous block's call) and whose own response is resp.
func toolResultBlock(t *testing.T, toolName, resultText string, resp testGeminiResponse) analyze.Block {
	req := testGeminiRequest{Contents: []testGeminiContent{
		{Role: "user", Parts: []testGeminiPart{{Text: "question"}}},
		{Role: "model", Parts: []testGeminiPart{{FunctionCall: &testFuncCall{Name: toolName, Args: map[string]any{}}}}},
		{Role: "user", Parts: []testGeminiPart{{FunctionResponse: &testFuncResponse{Name: toolName, Response: map[string]any{"result": resultText}}}}},
	}}
	return analyze.NewBlock(time.Now(), "gemini", "conv-1", mustJSON(t, req), mustJSON(t, resp), nil)
}

func finalAnswerResponse(text string) testGeminiResponse {
	return testGeminiResponse{Text: text}
}

func TestDetect_NoAnomaliesOnCleanTurn(t *testing.T) {
	blocks := []analyze.Block{
		questionBlock(t, "how am I doing"),
		toolResultBlock(t, "lab_results", `{"cholesterol":"normal"}`, finalAnswerResponse("you're doing fine")),
	}
	durations := []time.Duration{time.Second, time.Second}

	got := Detect(blocks, durations, Outcome{IterationCount: 2, MaxIterations: 16}, Options{})
	require.Empty(t, got)
}

func TestDetect_SlowCall(t *testing.T) {
	blocks := []analyze.Block{questionBlock(t, "q")}
	durations := []time.Duration{45 * time.Second}

	got := Detect(blocks, durations, Outcome{}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindSlowCall, got[0].Kind)
}

func TestDetect_SlowCall_CustomThreshold(t *testing.T) {
	blocks := []analyze.Block{questionBlock(t, "q")}
	durations := []time.Duration{5 * time.Second}

	require.Empty(t, Detect(blocks, durations, Outcome{}, Options{SlowCallThreshold: 10 * time.Second}))
	require.Len(t, Detect(blocks, durations, Outcome{}, Options{SlowCallThreshold: time.Second}), 1)
}

func TestDetect_RepeatedToolCall(t *testing.T) {
	repeat := testGeminiResponse{ToolCalls: []testToolCall{{Name: "lab_results", Arguments: `{"patient":"joe"}`}}}
	blocks := []analyze.Block{
		toolResultBlock(t, "lab_results", "ok", repeat),
		toolResultBlock(t, "lab_results", "ok", repeat),
	}

	got := Detect(blocks, nil, Outcome{}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindRepeatedToolCall, got[0].Kind)
}

func TestDetect_RepeatedToolCall_DifferentArgumentsIsNotAnomaly(t *testing.T) {
	blocks := []analyze.Block{
		toolResultBlock(t, "lab_results", "ok", testGeminiResponse{ToolCalls: []testToolCall{{Name: "lab_results", Arguments: `{"patient":"joe"}`}}}),
		toolResultBlock(t, "lab_results", "ok", testGeminiResponse{ToolCalls: []testToolCall{{Name: "lab_results", Arguments: `{"patient":"anne"}`}}}),
	}

	require.Empty(t, Detect(blocks, nil, Outcome{}, Options{}))
}

func TestDetect_RepeatedToolCall_KeyOrderDoesNotMatter(t *testing.T) {
	blocks := []analyze.Block{
		toolResultBlock(t, "lab_results", "ok", testGeminiResponse{ToolCalls: []testToolCall{{Name: "lab_results", Arguments: `{"patient":"joe","from":"2020"}`}}}),
		toolResultBlock(t, "lab_results", "ok", testGeminiResponse{ToolCalls: []testToolCall{{Name: "lab_results", Arguments: `{"from":"2020","patient":"joe"}`}}}),
	}

	got := Detect(blocks, nil, Outcome{}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindRepeatedToolCall, got[0].Kind)
}

func TestDetect_UnknownTool(t *testing.T) {
	blocks := []analyze.Block{
		questionBlock(t, "q"),
		toolResultBlock(t, "no_such_tool", `error: unknown tool "no_such_tool"`, finalAnswerResponse("sorry")),
	}

	got := Detect(blocks, nil, Outcome{}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindUnknownTool, got[0].Kind)
}

func TestDetect_UnknownTool_MirandaWording(t *testing.T) {
	blocks := []analyze.Block{
		questionBlock(t, "q"),
		toolResultBlock(t, "ha_turn_on", `error: mcp: no configured server matches tool "ha_turn_on"`, finalAnswerResponse("sorry")),
	}

	got := Detect(blocks, nil, Outcome{}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindUnknownTool, got[0].Kind)
}

func TestDetect_InvalidArguments(t *testing.T) {
	blocks := []analyze.Block{
		questionBlock(t, "q"),
		toolResultBlock(t, "lab_results", `error: ask: decode lab_results tool call arguments: unexpected end of JSON input`, finalAnswerResponse("sorry")),
	}

	got := Detect(blocks, nil, Outcome{}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindInvalidArguments, got[0].Kind)
}

func TestDetect_ToolErrorCatchAll(t *testing.T) {
	blocks := []analyze.Block{
		questionBlock(t, "q"),
		toolResultBlock(t, "lab_results", `error: provider unavailable`, finalAnswerResponse("sorry")),
	}

	got := Detect(blocks, nil, Outcome{}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindToolError, got[0].Kind)
}

func TestDetect_FirstBlockToolResultIsNotScanned(t *testing.T) {
	// A single block's own request can only carry *earlier* turns' results —
	// nothing from *this* turn has executed a tool yet at that point — so it
	// must never be misattributed to this turn's own anomaly detection.
	blocks := []analyze.Block{
		toolResultBlock(t, "lab_results", `error: unknown tool "lab_results"`, finalAnswerResponse("answer")),
	}

	require.Empty(t, Detect(blocks, nil, Outcome{}, Options{}))
}

func TestDetect_IterationCap(t *testing.T) {
	got := Detect(nil, nil, Outcome{HitIterationCap: true, MaxIterations: 16}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindIterationCap, got[0].Kind)
}

func TestDetect_Timeout(t *testing.T) {
	got := Detect(nil, nil, Outcome{TimedOut: true}, Options{})
	require.Len(t, got, 1)
	require.Equal(t, KindTimeout, got[0].Kind)
}

func TestDetect_IterationCapAndTimeoutAreIndependent(t *testing.T) {
	got := Detect(nil, nil, Outcome{HitIterationCap: true, TimedOut: true, MaxIterations: 16}, Options{})
	require.Len(t, got, 2)
}
