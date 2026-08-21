// Package analyze reads back the block format internal/llmtrace's own
// Logger.Trace writes (see that package's doc comment): one "=== <RFC3339
// timestamp> provider=<name> [conversation=<id>] ===" header per traced
// call, a "--- request ---" section, a "--- response ---" section (or a
// single "error: <message>" line in place of a response), and a blank line
// terminating the block.
//
// Ported out of miranda-medical-card's cmd/medical-dev/llm_trace.go (a CLI
// built for debugging medical.ask agent-loop conversations) and Miranda's
// own internal/webui/static/js/screens/logs-trace-parser.js (an incremental
// JS reassembler for the same format, needed because Miranda's web UI
// receives llm.log one line at a time over a WebSocket) once it became clear
// both services were maintaining separate implementations of the exact same
// block-recognition algorithm. Accumulator below is that one algorithm,
// usable both to parse a whole file at once (ParseAll) and to reassemble a
// live line-by-line stream (Feed) — see each consuming service's own
// wiring for which mode it needs.
package analyze

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"
)

// Block is one fully-reassembled traced call.
type Block struct {
	Time         time.Time       `json:"timestamp"`
	Provider     string          `json:"provider"`
	Conversation string          `json:"conversationId,omitempty"`
	Request      string          `json:"requestRaw"`
	RequestJSON  json.RawMessage `json:"requestJson,omitempty"`
	Response     string          `json:"responseRaw,omitempty"`
	ResponseJSON json.RawMessage `json:"responseJson,omitempty"`
	IsError      bool            `json:"isError"`
	ErrorText    string          `json:"errorText,omitempty"`
}

var headerRe = regexp.MustCompile(`^=== (\S+) provider=(\S+)(?: conversation=(\S+))? ===$`)

const (
	requestMarker  = "--- request ---"
	responseMarker = "--- response ---"
	errorPrefix    = "error: "
)

type section int

const (
	sectionNone section = iota
	sectionRequest
	sectionResponse
)

type openBlock struct {
	time          time.Time
	provider      string
	conversation  string
	section       section
	requestLines  []string
	responseLines []string
}

func (o *openBlock) finalize() Block {
	request := strings.Join(o.requestLines, "\n")
	response := strings.Join(o.responseLines, "\n")

	b := Block{
		Time:         o.time,
		Provider:     o.provider,
		Conversation: o.conversation,
		Request:      request,
	}
	if rest, ok := strings.CutPrefix(response, errorPrefix); ok {
		b.IsError = true
		b.ErrorText = rest
	} else {
		b.Response = response
	}

	if json.Valid([]byte(request)) {
		b.RequestJSON = json.RawMessage(request)
	}
	if !b.IsError && json.Valid([]byte(response)) {
		b.ResponseJSON = json.RawMessage(response)
	}
	return b
}

// NewBlock builds a Block directly from a traced call's raw
// request/response/err — the same derivation openBlock.finalize applies to
// parsed log text, but for a caller that already has these values in hand
// without ever going through a log line (e.g. llmtrace/anomaly.Recorder,
// which taps a call via llmtrace.WithTracer before Logger.Trace serializes
// it to text at all).
func NewBlock(t time.Time, provider, conversation, request, response string, err error) Block {
	b := Block{
		Time:         t,
		Provider:     provider,
		Conversation: conversation,
		Request:      request,
	}
	if err != nil {
		b.IsError = true
		b.ErrorText = err.Error()
	} else {
		b.Response = response
	}
	if json.Valid([]byte(request)) {
		b.RequestJSON = json.RawMessage(request)
	}
	if !b.IsError && json.Valid([]byte(response)) {
		b.ResponseJSON = json.RawMessage(response)
	}
	return b
}

// FormatBlock renders b back into the exact block-text shape
// llmtrace.Logger.Trace originally wrote it in — the inverse of
// openBlock.finalize — so a block read, filtered, or built in memory here
// (e.g. llmtrace/anomaly's anomaly-file writer, or a "copy trace" UI action)
// round-trips through ParseAll/Feed identically to the original log.
func FormatBlock(b Block) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "=== %s provider=%s", b.Time.Format(time.RFC3339), b.Provider)
	if b.Conversation != "" {
		fmt.Fprintf(&sb, " conversation=%s", b.Conversation)
	}
	sb.WriteString(" ===\n")

	sb.WriteString(requestMarker + "\n")
	sb.WriteString(b.Request)
	sb.WriteString("\n")

	sb.WriteString(responseMarker + "\n")
	if b.IsError {
		sb.WriteString(errorPrefix + b.ErrorText + "\n")
	} else {
		sb.WriteString(b.Response)
		sb.WriteString("\n")
	}
	sb.WriteString("\n")
	return sb.String()
}

// WriteBlocks renders blocks back-to-back, in order, via FormatBlock.
func WriteBlocks(w io.Writer, blocks []Block) error {
	for _, b := range blocks {
		if _, err := io.WriteString(w, FormatBlock(b)); err != nil {
			return fmt.Errorf("analyze: write block: %w", err)
		}
	}
	return nil
}

// Accumulator is a stateful, incremental reassembler: call Feed once per raw
// text line, in arrival order, and it returns a finished Block the instant a
// line completes one (the blank line terminating a block), and ok=false for
// every line in between. A zero Accumulator is not usable — use
// NewAccumulator.
//
// Deliberately tolerant of malformed/partial input rather than erroring:
// content lines encountered with no block currently open (e.g. the tail of
// an older block whose header line already scrolled out of a bounded replay
// buffer — see Miranda's internal/hub.Hub) are silently dropped, and a
// header line always starts a new block even over one that never closed.
type Accumulator struct {
	open *openBlock
}

// NewAccumulator creates a ready-to-use Accumulator.
func NewAccumulator() *Accumulator {
	return &Accumulator{}
}

// Feed processes one line and reports the Block it just completed, if any.
func (a *Accumulator) Feed(line string) (*Block, bool) {
	if m := headerRe.FindStringSubmatch(line); m != nil {
		t, err := time.Parse(time.RFC3339, m[1])
		if err != nil {
			// An unparseable timestamp means this isn't really a trace
			// header — drop it the same way a header regex mismatch would
			// have, rather than opening a block we can't sort or display.
			a.open = nil
			return nil, false
		}
		a.open = &openBlock{time: t, provider: m[2], conversation: m[3]}
		return nil, false
	}

	if a.open == nil {
		return nil, false
	}

	switch line {
	case requestMarker:
		a.open.section = sectionRequest
		return nil, false
	case responseMarker:
		a.open.section = sectionResponse
		return nil, false
	case "":
		block := a.open.finalize()
		a.open = nil
		return &block, true
	}

	switch a.open.section {
	case sectionRequest:
		a.open.requestLines = append(a.open.requestLines, line)
	case sectionResponse:
		a.open.responseLines = append(a.open.responseLines, line)
	}
	return nil, false
}

// maxLineSize accommodates a single trace line holding a full request/
// response JSON payload — bufio.Scanner's 64KB default is comfortably too
// small for a multi-turn agent-loop conversation history.
const maxLineSize = 32 * 1024 * 1024

// ParseAll reads every line from r through one Accumulator and returns every
// Block it completes, in file order. Used for whole-file, offline parsing
// (each service's own CLI) — see Accumulator.Feed directly for live,
// line-at-a-time streaming (Miranda's web UI backend).
func ParseAll(r io.Reader) ([]Block, error) {
	acc := NewAccumulator()
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)

	var blocks []Block
	for scanner.Scan() {
		if block, ok := acc.Feed(scanner.Text()); ok {
			blocks = append(blocks, *block)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("analyze: scan trace log: %w", err)
	}
	return blocks, nil
}
