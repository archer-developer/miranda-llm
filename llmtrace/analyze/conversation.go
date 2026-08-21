package analyze

import "sort"

// LatestConversation returns the conversation id of the most recently
// started tagged conversation among blocks, or "" if none are tagged.
func LatestConversation(blocks []Block) string {
	var id string
	var latest Block
	for _, b := range blocks {
		if b.Conversation != "" && (id == "" || b.Time.After(latest.Time)) {
			latest, id = b, b.Conversation
		}
	}
	return id
}

// ConversationSummary aggregates every block sharing one Conversation id.
type ConversationSummary struct {
	ID        string
	Provider  string
	FirstSeen Block
	LastSeen  Block
	Turns     int
	Errors    int
}

// Summarize groups blocks by Conversation id, oldest-conversation-first.
// Untagged blocks (Conversation == "") are not included — see TailUntagged
// for those.
func Summarize(blocks []Block) []ConversationSummary {
	summaries := map[string]*ConversationSummary{}
	var order []string
	for _, b := range blocks {
		if b.Conversation == "" {
			continue
		}
		s, ok := summaries[b.Conversation]
		if !ok {
			s = &ConversationSummary{ID: b.Conversation, Provider: b.Provider, FirstSeen: b, LastSeen: b}
			summaries[b.Conversation] = s
			order = append(order, b.Conversation)
		}
		s.Turns++
		if b.Time.Before(s.FirstSeen.Time) {
			s.FirstSeen = b
		}
		if b.Time.After(s.LastSeen.Time) {
			s.LastSeen = b
		}
		if b.ErrorText != "" {
			s.Errors++
		}
	}
	sort.Slice(order, func(i, j int) bool {
		return summaries[order[i]].FirstSeen.Time.Before(summaries[order[j]].FirstSeen.Time)
	})

	out := make([]ConversationSummary, len(order))
	for i, id := range order {
		out[i] = *summaries[id]
	}
	return out
}

// CountUntagged reports how many blocks carry no Conversation id.
func CountUntagged(blocks []Block) int {
	n := 0
	for _, b := range blocks {
		if b.Conversation == "" {
			n++
		}
	}
	return n
}

// ConversationBlocks returns every block tagged with conversationID,
// oldest-first.
func ConversationBlocks(blocks []Block, conversationID string) []Block {
	var turns []Block
	for _, b := range blocks {
		if b.Conversation == conversationID {
			turns = append(turns, b)
		}
	}
	sort.Slice(turns, func(i, j int) bool { return turns[i].Time.Before(turns[j].Time) })
	return turns
}

// TailUntagged returns the most recent tail (or fewer) untagged blocks,
// oldest first.
func TailUntagged(blocks []Block, tail int) []Block {
	var turns []Block
	for _, b := range blocks {
		if b.Conversation == "" {
			turns = append(turns, b)
		}
	}
	sort.Slice(turns, func(i, j int) bool { return turns[i].Time.Before(turns[j].Time) })
	if len(turns) > tail {
		turns = turns[len(turns)-tail:]
	}
	return turns
}

// GroupByConversationStart splits turns (oldest-first, all untagged) back
// into the separate stateless conversations they actually were — with no
// conversation id to group by, a flat list otherwise runs unrelated
// questions together with no boundary between them. IsFirstTurn detects the
// boundary: a request with exactly one content/message entry is a
// conversation literally starting there, no matter which provider shape it's
// in. A tail that begins mid-conversation (its own true first turn fell
// outside the tail passed in) still gets its own group — just one whose
// first turn won't look like a true start, since IsFirstTurn correctly
// reports false for it.
func GroupByConversationStart(turns []Block) [][]Block {
	var groups [][]Block
	for _, b := range turns {
		if len(groups) == 0 || IsFirstTurn(b) {
			groups = append(groups, nil)
		}
		last := len(groups) - 1
		groups[last] = append(groups[last], b)
	}
	return groups
}
