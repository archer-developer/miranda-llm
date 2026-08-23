package analyze

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestParseAll_GeminiShape(t *testing.T) {
	log := `=== 2026-08-12T18:14:45Z provider=gemini-agent conversation=session_1 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Какие анализы за июль?"}]}]}
--- response ---
{"text":"","tool_calls":[{"ID":"c1","Name":"lab_results","Arguments":"{\"from\":\"2026-07-01\"}"}]}

=== 2026-08-12T18:14:47Z provider=gemini-agent conversation=session_1 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Какие анализы за июль?"}]},{"role":"model","parts":[{"functionCall":{"name":"lab_results","args":{"from":"2026-07-01"}}}]},{"role":"user","parts":[{"functionResponse":{"name":"lab_results","response":{"result":"- АЛТ: 63 (2026-07-24)."}}}]}]}
--- response ---
{"text":"АЛТ повышен, стоит обсудить с врачом.","tool_calls":null}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 2)
	require.Equal(t, "session_1", blocks[0].Conversation)
	require.Equal(t, "gemini-agent", blocks[0].Provider)

	in, ok := DescribeIncoming(blocks[0], true)
	require.True(t, ok)
	require.Len(t, in, 1)
	require.Contains(t, in[0], "Какие анализы за июль?")

	out := DescribeOutgoing(blocks[0])
	require.Len(t, out, 1)
	require.Contains(t, out[0], "lab_results")
	require.Contains(t, out[0], "2026-07-01")

	in2, ok := DescribeIncoming(blocks[1], false)
	require.True(t, ok)
	require.Len(t, in2, 1)
	require.Contains(t, in2[0], "lab_results ->")
	require.Contains(t, in2[0], "АЛТ: 63")

	out2 := DescribeOutgoing(blocks[1])
	require.Len(t, out2, 1)
	require.Contains(t, out2[0], "answer: АЛТ повышен")

	require.False(t, EndsOnToolCallWithNoAnswer(blocks[1]), "a final text answer must not be flagged as a cut-off conversation")
	require.True(t, EndsOnToolCallWithNoAnswer(blocks[0]), "a tool call with no text must be flagged as a cut-off conversation")
}

// TestDescribeOutgoing_ShowsDurationAndTokenUsage covers gemini.Provider's
// trace() including duration_ms/usage — added so a slow or token-heavy turn
// is visible per-call in this table, not just as an aggregate. A response
// with no duration_ms (older trace, or a provider shape that doesn't report
// it) must not print an empty stat line.
func TestDescribeOutgoing_ShowsDurationAndTokenUsage(t *testing.T) {
	log := `=== 2026-08-12T18:14:45Z provider=gemini-agent conversation=session_1 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Вопрос"}]}]}
--- response ---
{"text":"Ответ.","tool_calls":null,"duration_ms":2891,"usage":{"promptTokenCount":4812,"candidatesTokenCount":396}}

=== 2026-08-12T18:14:47Z provider=gemini-agent conversation=session_2 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Вопрос без duration"}]}]}
--- response ---
{"text":"Ответ.","tool_calls":null}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 2)

	out := DescribeOutgoing(blocks[0])
	require.Len(t, out, 2, "answer line plus a duration/usage stat line")
	require.Contains(t, out[1], "2891ms")
	require.Contains(t, out[1], "4812+396 tokens")

	out2 := DescribeOutgoing(blocks[1])
	require.Len(t, out2, 1, "no duration_ms in the trace must not print an empty/zero stat line")
}

// TestParseAll_GeminiParallelToolCalls reproduces the shape a real
// agent-loop turn actually produced: the model called two tools at once
// (profile + lab_results), and Gemini's own request-building puts each
// functionResponse in its own trailing "user" content entry — looking only
// at the last content entry silently drops the first tool's result.
func TestParseAll_GeminiParallelToolCalls(t *testing.T) {
	log := `=== 2026-08-12T19:08:01Z provider=gemini-agent conversation=session_parallel ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Рекомендации по питанию?"}]}]}
--- response ---
{"text":"","tool_calls":[{"ID":"c1","Name":"profile","Arguments":"{}"},{"ID":"c2","Name":"lab_results","Arguments":"{\"limit\":50}"}]}

=== 2026-08-12T19:08:05Z provider=gemini-agent conversation=session_parallel ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Рекомендации по питанию?"}]},{"role":"model","parts":[{"functionCall":{"name":"profile","args":{}}},{"functionCall":{"name":"lab_results","args":{"limit":50}}}]},{"role":"user","parts":[{"functionResponse":{"name":"profile","response":{"result":"Возраст 40, аллергии: нет."}}}]},{"role":"user","parts":[{"functionResponse":{"name":"lab_results","response":{"result":"АЛТ: 63 (2026-07-24)."}}}]}]}
--- response ---
{"text":"Судя по профилю и анализам...","tool_calls":null}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 2)

	out := DescribeOutgoing(blocks[0])
	require.Len(t, out, 2, "both parallel tool calls must be listed, not just one")

	in2, ok := DescribeIncoming(blocks[1], false)
	require.True(t, ok)
	require.Len(t, in2, 2, "both tool results must be listed, not just the last content entry")
	require.Contains(t, in2[0], "profile ->")
	require.Contains(t, in2[0], "Возраст 40")
	require.Contains(t, in2[1], "lab_results ->")
	require.Contains(t, in2[1], "АЛТ: 63")
}

func TestParseAll_AnthropicShape(t *testing.T) {
	log := `=== 2026-08-12T18:20:00Z provider=claude-escalation conversation=session_2 ===
--- request ---
{"model":"claude-sonnet-5","messages":[{"role":"user","content":[{"type":"text","text":"Что с холестерином?"}]}]}
--- response ---
{"id":"msg_1","content":[{"type":"tool_use","id":"t1","name":"lab_results","input":{"indicatorName":"холестерин"}}],"stop_reason":"tool_use"}

=== 2026-08-12T18:20:03Z provider=claude-escalation conversation=session_2 ===
--- request ---
{"model":"claude-sonnet-5","messages":[{"role":"user","content":[{"type":"text","text":"Что с холестерином?"}]},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"lab_results","input":{"indicatorName":"холестерин"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"- Холестерин: 5.2 (2026-07-24)."}]}]}
--- response ---
{"id":"msg_2","content":[{"type":"text","text":"Холестерин слегка повышен."}],"stop_reason":"end_turn"}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 2)

	out := DescribeOutgoing(blocks[0])
	require.Len(t, out, 1)
	require.Contains(t, out[0], "call lab_results")
	require.Contains(t, out[0], "холестерин")
	require.True(t, EndsOnToolCallWithNoAnswer(blocks[0]))

	in2, ok := DescribeIncoming(blocks[1], false)
	require.True(t, ok)
	require.Len(t, in2, 1)
	require.Contains(t, in2[0], "tool result ->")
	require.Contains(t, in2[0], "Холестерин: 5.2")

	out2 := DescribeOutgoing(blocks[1])
	require.Len(t, out2, 1)
	require.Contains(t, out2[0], "answer: Холестерин слегка повышен.")
	require.False(t, EndsOnToolCallWithNoAnswer(blocks[1]))
}

func TestParseAll_ErrorBlock(t *testing.T) {
	log := `=== 2026-08-12T18:30:00Z provider=gemini-agent conversation=session_3 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Вопрос"}]}]}
--- response ---
error: gemini: request failed: 429 quota exceeded

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 1)
	require.True(t, blocks[0].IsError)
	require.Equal(t, "gemini: request failed: 429 quota exceeded", blocks[0].ErrorText)
	require.Empty(t, blocks[0].Response)
	require.Nil(t, blocks[0].ResponseJSON)
}

// TestIsEmptyResponse_GeminiShape is the regression test for the real
// incident this check exists to catch: a gemini-shaped response with a
// non-empty JSON envelope (usage/duration_ms present, matching what
// gemini.Provider.trace always writes) but text="" and no tool_calls at
// all — the model had the floor to answer and produced nothing.
func TestIsEmptyResponse_GeminiShape(t *testing.T) {
	log := `=== 2026-08-23T11:47:11Z provider=gemini-lite conversation=session_1 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Покажи анализ крови от 15.04.2025"}]}]}
--- response ---
{"text":"","duration_ms":844,"usage":{"promptTokenCount":14609,"totalTokenCount":14609}}

=== 2026-08-23T11:47:12Z provider=gemini-lite conversation=session_1 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Покажи анализ крови от 15.04.2025"}]}]}
--- response ---
{"text":"Вот результаты...","tool_calls":null,"duration_ms":900}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 2)

	require.True(t, IsEmptyResponse(blocks[0]), "text=\"\" with no tool_calls but a non-empty envelope must be flagged as an empty response")
	require.False(t, IsEmptyResponse(blocks[1]), "a real text answer must not be flagged as empty")
}

// TestIsEmptyResponse_AnthropicShape mirrors the gemini case for the
// anthropic response shape: an empty content array.
func TestIsEmptyResponse_AnthropicShape(t *testing.T) {
	log := `=== 2026-08-23T11:47:11Z provider=claude-escalation conversation=session_2 ===
--- request ---
{"model":"claude-sonnet-5","messages":[{"role":"user","content":[{"type":"text","text":"q"}]}]}
--- response ---
{"id":"msg_1","content":[],"stop_reason":"end_turn"}

=== 2026-08-23T11:47:12Z provider=claude-escalation conversation=session_2 ===
--- request ---
{"model":"claude-sonnet-5","messages":[{"role":"user","content":[{"type":"text","text":"q"}]}]}
--- response ---
{"id":"msg_2","content":[{"type":"text","text":"answer"}],"stop_reason":"end_turn"}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 2)

	require.True(t, IsEmptyResponse(blocks[0]), "an empty content array must be flagged as an empty response")
	require.False(t, IsEmptyResponse(blocks[1]))
}

// TestIsEmptyResponse_IgnoresErrorsAndTrivialObjects covers the two cases
// IsEmptyResponse must NOT flag: a block that's already an explicit error
// (a different, already-surfaced failure mode), and a bare "{}" — which no
// real trace ever produces, but which would otherwise trivially match an
// all-zero-value geminiChatResponse.
func TestIsEmptyResponse_IgnoresErrorsAndTrivialObjects(t *testing.T) {
	log := `=== 2026-08-23T11:47:11Z provider=gemini-agent conversation=session_3 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"q"}]}]}
--- response ---
error: gemini: request failed: 429 quota exceeded

=== 2026-08-23T11:47:12Z provider=gemini-agent conversation=session_3 ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"q"}]}]}
--- response ---
{}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 2)

	require.False(t, IsEmptyResponse(blocks[0]), "an explicit error block is a different, already-surfaced failure mode")
	require.False(t, IsEmptyResponse(blocks[1]), "a bare {} never comes from a real trace and must not be misread as this shape")
}

func TestLatestConversation_IgnoresUntaggedBlocks(t *testing.T) {
	log := `=== 2026-08-12T18:00:00Z provider=gemini-document ===
--- request ---
{}
--- response ---
{}

=== 2026-08-12T18:05:00Z provider=gemini-agent conversation=session_older ===
--- request ---
{}
--- response ---
{}

=== 2026-08-12T18:10:00Z provider=gemini-agent conversation=session_newer ===
--- request ---
{}
--- response ---
{}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 3)
	require.Equal(t, "session_newer", LatestConversation(blocks))
}

// TestTailUntagged_FiltersTaggedAndCapsToRecentN covers the stateless-ask
// case: a real agent-loop call with no session id (the default for a
// one-off question) never gets a conversation id at all, unlike a tagged
// session — so llm-trace has to fall back to "the tail of whatever's
// untagged" rather than filtering by id, and must not let a tagged
// conversation's blocks leak in.
func TestTailUntagged_FiltersTaggedAndCapsToRecentN(t *testing.T) {
	log := `=== 2026-08-12T18:00:00Z provider=gemini-document ===
--- request ---
{}
--- response ---
{}

=== 2026-08-12T18:05:00Z provider=gemini-agent conversation=session_tagged ===
--- request ---
{}
--- response ---
{}

=== 2026-08-12T19:57:51Z provider=gemini-agent ===
--- request ---
{}
--- response ---
{}

=== 2026-08-12T19:57:53Z provider=gemini-agent ===
--- request ---
{}
--- response ---
{}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 4)

	tail := TailUntagged(blocks, 20)
	require.Len(t, tail, 3, "must include both untagged agent-shaped blocks and the untagged document-pipeline block, but not the tagged session")
	require.True(t, tail[0].Time.Before(tail[1].Time), "must stay oldest-first")

	capped := TailUntagged(blocks, 2)
	require.Len(t, capped, 2, "must cap to the most recent N")
	require.Equal(t, "2026-08-12T19:57:51Z", capped[0].Time.Format(time.RFC3339))
	require.Equal(t, "2026-08-12T19:57:53Z", capped[1].Time.Format(time.RFC3339))
}

// TestGroupByConversationStart_SplitsTwoBackToBackStatelessQuestions
// reproduces exactly what --untagged saw right after being added: two
// separate stateless calls close together in time, both untagged,
// concatenated by TailUntagged into one flat slice — GroupByConversationStart
// must split them back apart at each single-content request, not treat the
// second question's first turn as a continuation of the first question's
// last one.
func TestGroupByConversationStart_SplitsTwoBackToBackStatelessQuestions(t *testing.T) {
	log := `=== 2026-08-12T19:57:51Z provider=gemini-agent ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Первый вопрос"}]}]}
--- response ---
{"text":"","tool_calls":[{"ID":"c1","Name":"lab_results","Arguments":"{}"}]}

=== 2026-08-12T19:57:53Z provider=gemini-agent ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Первый вопрос"}]},{"role":"model","parts":[{"functionCall":{"name":"lab_results","args":{}}}]},{"role":"user","parts":[{"functionResponse":{"name":"lab_results","response":{"result":"..."}}}]}]}
--- response ---
{"text":"Ответ на первый вопрос.","tool_calls":null}

=== 2026-08-12T20:01:30Z provider=gemini-agent ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Второй вопрос"}]}]}
--- response ---
{"text":"","tool_calls":[{"ID":"c2","Name":"timeline","Arguments":"{}"}]}

=== 2026-08-12T20:01:32Z provider=gemini-agent ===
--- request ---
{"contents":[{"role":"user","parts":[{"text":"Второй вопрос"}]},{"role":"model","parts":[{"functionCall":{"name":"timeline","args":{}}}]},{"role":"user","parts":[{"functionResponse":{"name":"timeline","response":{"result":"..."}}}]}]}
--- response ---
{"text":"Ответ на второй вопрос.","tool_calls":null}

`
	blocks, err := ParseAll(strings.NewReader(log))
	require.NoError(t, err)
	require.Len(t, blocks, 4)

	turns := TailUntagged(blocks, 20)
	require.Len(t, turns, 4)

	groups := GroupByConversationStart(turns)
	require.Len(t, groups, 2, "must split into exactly two questions")
	require.Len(t, groups[0], 2)
	require.Len(t, groups[1], 2)

	require.True(t, IsFirstTurn(groups[0][0]))
	require.False(t, IsFirstTurn(groups[0][1]))
	require.True(t, IsFirstTurn(groups[1][0]), "the second question's first turn must be detected as a new boundary, not a continuation")
	require.False(t, IsFirstTurn(groups[1][1]))
}

func TestTruncate_RespectsRuneBoundaries(t *testing.T) {
	s := Truncate("Показатели: холестерин, мочевина, креатинин", 15)
	require.LessOrEqual(t, len([]rune(s)), 16) // 15 + the ellipsis rune
	require.Contains(t, s, "…")
}
