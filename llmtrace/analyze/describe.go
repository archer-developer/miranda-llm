package analyze

import (
	"encoding/json"
	"fmt"
	"strings"
)

// --- Gemini trace shape (gemini.Provider.trace) — the only provider shape
// fully decoded below; used by Miranda's agent loop and medical-card's
// medical.ask agent loop alike (llm.agent_provider in each service's own
// config). ---

type geminiPart struct {
	Text             string          `json:"text"`
	FunctionCall     *geminiCall     `json:"functionCall"`
	FunctionResponse *geminiResponse `json:"functionResponse"`
}
type geminiCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}
type geminiResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}
type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}
type geminiRequest struct {
	Contents []geminiContent `json:"contents"`
}
type geminiToolCall struct {
	Name      string `json:"Name"`
	Arguments string `json:"Arguments"`
}
type geminiChatResponse struct {
	Text       string           `json:"text"`
	ToolCalls  []geminiToolCall `json:"tool_calls"`
	DurationMs int64            `json:"duration_ms"`
	Usage      *geminiUsage     `json:"usage"`
}

// geminiUsage mirrors genai.GenerateContentResponseUsageMetadata's fields
// gemini.Provider's own trace() actually surfaces, so a slow or expensive
// turn is visible per-call here instead of only in aggregate.
type geminiUsage struct {
	PromptTokenCount     int32 `json:"promptTokenCount"`
	CandidatesTokenCount int32 `json:"candidatesTokenCount"`
}

// --- Anthropic trace shape (anthropic.Provider.trace) — decoded generically
// via the well-known Messages API JSON shape rather than importing
// anthropic-sdk-go's request param types, which aren't meant to be read back
// from JSON. ---

func anthropicBlockText(block map[string]any, key string) (string, bool) {
	v, ok := block[key].(string)
	return v, ok
}

// --- shared decoding ---

// DescribeIncoming summarizes every item b's request adds to the
// conversation since the previous turn — the original question (first
// block only), or one entry per tool result a parallel multi-tool-call turn
// produced (a single turn commonly calls more than one tool at once — e.g.
// profile+lab_results together — and each gets its own result; showing only
// the last would silently drop every result but one). ok is false when the
// request doesn't match any recognized shape, so the caller can fall back to
// a plain turn number.
func DescribeIncoming(b Block, isFirst bool) ([]string, bool) {
	var req geminiRequest
	if json.Unmarshal([]byte(b.Request), &req) == nil && len(req.Contents) > 0 {
		return describeIncomingGemini(req.Contents, isFirst), true
	}

	var generic struct {
		Messages []map[string]any `json:"messages"`
	}
	if json.Unmarshal([]byte(b.Request), &generic) == nil && len(generic.Messages) > 0 {
		return describeIncomingAnthropic(generic.Messages[len(generic.Messages)-1], isFirst), true
	}

	return nil, false
}

// describeIncomingGemini handles the first block (a single "user"/text
// content — the question) and every later block, where a parallel
// multi-tool-call turn adds one whole "user"/functionResponse content entry
// per tool called (see gemini.Provider's own request-building — unlike
// Anthropic, which bundles multiple tool_results into one message, see
// describeIncomingAnthropic) rather than multiple parts on a single entry.
// Walking backward from the end and collecting every consecutive
// all-functionResponse entry covers exactly that, however many tools the
// previous turn called.
func describeIncomingGemini(contents []geminiContent, isFirst bool) []string {
	if len(contents) == 0 {
		return nil
	}
	if isFirst {
		for _, p := range contents[len(contents)-1].Parts {
			if p.Text != "" {
				return []string{"question: " + Truncate(p.Text, 220)}
			}
		}
		return nil
	}

	var reversed []string
	for i := len(contents) - 1; i >= 0; i-- {
		parts := contents[i].Parts
		if len(parts) == 0 {
			break
		}
		allResponses := true
		for _, p := range parts {
			if p.FunctionResponse == nil {
				allResponses = false
				break
			}
		}
		if !allResponses {
			break
		}
		for _, p := range parts {
			reversed = append(reversed, fmt.Sprintf("%s -> %s", p.FunctionResponse.Name, Truncate(fmt.Sprint(p.FunctionResponse.Response["result"]), 220)))
		}
	}
	// Collected walking backward (most recent tool call's result first);
	// reverse to match call order.
	for l, r := 0, len(reversed)-1; l < r; l, r = l+1, r-1 {
		reversed[l], reversed[r] = reversed[r], reversed[l]
	}
	return reversed
}

// describeIncomingAnthropic handles Anthropic's own bundling: a
// multi-tool-call turn's results all land as separate tool_result blocks
// within the one following user message (unlike Gemini — see
// describeIncomingGemini), so collecting every matching block in that
// single message already covers a parallel turn.
func describeIncomingAnthropic(last map[string]any, isFirst bool) []string {
	var out []string
	content, _ := last["content"].([]any)
	for _, item := range content {
		block, ok := item.(map[string]any)
		if !ok {
			continue
		}
		switch block["type"] {
		case "tool_result":
			text, _ := anthropicBlockText(block, "content")
			out = append(out, "tool result -> "+Truncate(text, 220))
		case "text":
			if isFirst {
				if text, ok := anthropicBlockText(block, "text"); ok {
					out = append(out, "question: "+Truncate(text, 220))
				}
			}
		}
	}
	// A plain-string content (no blocks) is valid Anthropic API shape too,
	// used for simple single-turn text.
	if len(out) == 0 {
		if text, ok := last["content"].(string); ok && isFirst {
			out = append(out, "question: "+Truncate(text, 220))
		}
	}
	return out
}

// DescribeOutgoing lists what the model did this turn — one entry per tool
// call, or its final answer text, or a raw preview when the response
// doesn't match a recognized shape at all.
func DescribeOutgoing(b Block) []string {
	var resp geminiChatResponse
	if json.Unmarshal([]byte(b.Response), &resp) == nil && (resp.Text != "" || len(resp.ToolCalls) > 0) {
		var out []string
		if resp.Text != "" {
			out = append(out, "answer: "+Truncate(resp.Text, 400))
		}
		for _, tc := range resp.ToolCalls {
			out = append(out, fmt.Sprintf("call %s(%s)", tc.Name, Truncate(tc.Arguments, 200)))
		}
		if resp.DurationMs > 0 {
			stat := fmt.Sprintf("[%dms", resp.DurationMs)
			if resp.Usage != nil {
				stat += fmt.Sprintf(", %d+%d tokens", resp.Usage.PromptTokenCount, resp.Usage.CandidatesTokenCount)
			}
			out = append(out, stat+"]")
		}
		return out
	}

	var anth struct {
		Content []map[string]any `json:"content"`
	}
	if json.Unmarshal([]byte(b.Response), &anth) == nil && len(anth.Content) > 0 {
		var out []string
		for _, block := range anth.Content {
			switch block["type"] {
			case "text":
				if text, ok := anthropicBlockText(block, "text"); ok && text != "" {
					out = append(out, "answer: "+Truncate(text, 400))
				}
			case "tool_use":
				name, _ := block["name"].(string)
				input, _ := json.Marshal(block["input"])
				out = append(out, fmt.Sprintf("call %s(%s)", name, Truncate(string(input), 200)))
			}
		}
		if len(out) > 0 {
			return out
		}
	}

	return []string{"(unrecognized response shape) " + Truncate(b.Response, 200)}
}

// EndsOnToolCallWithNoAnswer reports whether b's response is entirely tool
// call(s) with no final text — the shape a conversation that got cut off
// mid-loop (iteration cap, timeout, crash) always ends on.
func EndsOnToolCallWithNoAnswer(b Block) bool {
	// json.Unmarshal into either shape below "succeeds" even on a response
	// that matches neither (unknown fields are ignored, missing ones stay
	// zero) — checking for actual signal (a non-empty ToolCalls/Content),
	// not just a nil error, is what lets an anthropic-shaped response fall
	// through to its own branch instead of being misread as an
	// empty-but-valid gemini one.
	var resp geminiChatResponse
	if json.Unmarshal([]byte(b.Response), &resp) == nil && (resp.Text != "" || len(resp.ToolCalls) > 0) {
		return resp.Text == "" && len(resp.ToolCalls) > 0
	}
	var anth struct {
		Content []map[string]any `json:"content"`
	}
	if json.Unmarshal([]byte(b.Response), &anth) == nil && len(anth.Content) > 0 {
		sawToolUse, sawText := false, false
		for _, block := range anth.Content {
			switch block["type"] {
			case "tool_use":
				sawToolUse = true
			case "text":
				sawText = true
			}
		}
		return sawToolUse && !sawText
	}
	return false
}

// IsFirstTurn reports whether b's request is a conversation's very first
// turn — exactly one content/message entry (just the question, no
// accumulated tool-call history yet) — regardless of provider shape. The
// same signal DescribeIncoming's isFirst parameter relies on
// (describeIncomingGemini/describeIncomingAnthropic), pulled out standalone
// here so a caller can test it without already knowing which turn is first.
func IsFirstTurn(b Block) bool {
	var req geminiRequest
	if json.Unmarshal([]byte(b.Request), &req) == nil && len(req.Contents) > 0 {
		return len(req.Contents) == 1
	}
	var generic struct {
		Messages []map[string]any `json:"messages"`
	}
	if json.Unmarshal([]byte(b.Request), &generic) == nil && len(generic.Messages) > 0 {
		return len(generic.Messages) == 1
	}
	return false
}

// Truncate collapses whitespace/newlines into one table-friendly line and
// caps it to n runes.
func Truncate(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
