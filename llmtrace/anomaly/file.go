package anomaly

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/archer-developer/miranda-llm/llmtrace/analyze"
)

// FileName suggests a name for an anomaly file: a sortable timestamp plus
// the distinct anomaly kinds found, deduplicated in first-seen order, so a
// human can scan a directory listing (logs/anomalies/) without opening every
// file to know what's in it.
func FileName(t time.Time, anomalies []Anomaly) string {
	seen := map[string]bool{}
	var kinds []string
	for _, a := range anomalies {
		if !seen[a.Kind] {
			seen[a.Kind] = true
			kinds = append(kinds, a.Kind)
		}
	}
	return fmt.Sprintf("%s_%s.log", t.UTC().Format("20060102T150405Z"), strings.Join(kinds, "-"))
}

// WriteFile writes blocks (typically the whole conversation so far — see
// each service's own wiring for how it gathers those — or just the turn's
// own blocks when no conversation id is available) to w, preceded by a
// short "#"-prefixed header summarizing the anomalies found. The header
// lines are plain comments a reviewer reads first; analyze.ParseAll/
// Accumulator.Feed silently ignore any line encountered before a block's own
// "=== ... ===" header, so the file that follows still opens directly in
// the existing medical-dev/miranda "llm-trace" CLIs with no new tooling.
func WriteFile(w io.Writer, anomalies []Anomaly, blocks []analyze.Block) error {
	var header strings.Builder
	header.WriteString("# anomalies detected in this turn:\n")
	for _, a := range anomalies {
		fmt.Fprintf(&header, "#   [%s] %s\n", a.Kind, a.Detail)
	}
	header.WriteString("#\n")
	if _, err := io.WriteString(w, header.String()); err != nil {
		return fmt.Errorf("anomaly: write header: %w", err)
	}
	return analyze.WriteBlocks(w, blocks)
}
