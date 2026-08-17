package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"parallax/internal/llm"
	"parallax/internal/search"
)

// SearchEnv provides dependencies for the search_footage agent tool.
type SearchEnv struct {
	Workspace   string
	SearchMgr   *search.Manager
	EmbedClient *llm.EmbedClient
}

// RegisterSearch registers the search_footage tool in the agent registry.
func RegisterSearch(reg *Registry, env SearchEnv) {
	reg.Register(llm.NewFunctionTool(
		"search_footage",
		"Search raw footage by semantic content — quotes, dialogue, people, emotions, actions, scenes — across all uploaded media in the project. Returns a ranked list of matched moments with timestamps, thumbnails, transcript/visual snippets, and relevance scores.",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "What to search for in natural language, e.g. 'person laughing', 'close-up shot', 'where they said machine learning', 'sunset scene'"
				},
				"top_k": {
					"type": "integer",
					"description": "Maximum number of results to return (default 10, max 30)"
				},
				"kind": {
					"type": "string",
					"enum": ["all", "frame", "transcript"],
					"description": "Filter by match type: 'all' (default), 'frame' (visual descriptions), or 'transcript' (spoken dialogue)"
				}
			},
			"required": ["query"]
		}`),
	), env.searchFootage)
}

func (e SearchEnv) searchFootage(ctx context.Context, raw json.RawMessage) Result {
	var in struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
		Kind  string `json:"kind"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	q := strings.TrimSpace(in.Query)
	if q == "" {
		return Result{OK: false, Error: "query is required"}
	}
	topK := in.TopK
	if topK <= 0 {
		topK = 10
	}
	if topK > 30 {
		topK = 30
	}
	kindFilter := strings.ToLower(strings.TrimSpace(in.Kind))
	if kindFilter == "" {
		kindFilter = "all"
	}

	if e.SearchMgr == nil {
		return Result{OK: false, Error: "search manager is not configured for this project"}
	}

	idx, err := e.SearchMgr.GetIndex(e.Workspace)
	if err != nil {
		return Result{OK: false, Error: fmt.Sprintf("get search index: %v", err)}
	}

	if idx.Len() == 0 {
		return Result{OK: true, Output: map[string]any{
			"query":   q,
			"count":   0,
			"results": []any{},
			"note":    "No indexed footage found. Upload media to index frames and transcripts.",
		}}
	}

	// Check if this is an exact quote search or keyword search
	var hits []search.Hit
	isQuote := (strings.HasPrefix(q, "\"") && strings.HasSuffix(q, "\"")) ||
		strings.HasPrefix(strings.ToLower(q), "said ") ||
		strings.HasPrefix(strings.ToLower(q), "says ")

	if isQuote {
		cleanQ := strings.Trim(q, `"'`)
		cleanQ = strings.TrimPrefix(cleanQ, "said ")
		cleanQ = strings.TrimPrefix(cleanQ, "says ")
		hits = idx.KeywordSearch(cleanQ, topK)
	}

	// If no keyword hits (or not an exact quote), run semantic embedding search
	if len(hits) == 0 && e.EmbedClient != nil {
		vec, err := e.EmbedClient.Embed(ctx, q)
		if err == nil {
			hits = idx.Query(vec, topK*2) // fetch extra to allow filtering
		}
	}

	// If embedding client wasn't available or errored, try keyword search as fallback
	if len(hits) == 0 {
		hits = idx.KeywordSearch(q, topK)
	}

	// Filter by kind if specified
	var filtered []map[string]any
	for _, h := range hits {
		if kindFilter != "all" && h.Meta.Kind != kindFilter {
			continue
		}
		item := map[string]any{
			"file_id":         h.Meta.FileID,
			"media_path":      h.Meta.MediaPath,
			"start_sec":       h.Meta.StartSec,
			"end_sec":         h.Meta.EndSec,
			"kind":            h.Meta.Kind,
			"text":            h.Meta.Text,
			"thumbnail":       h.Meta.ThumbPath,
			"relevance_score": h.Score,
		}
		filtered = append(filtered, item)
		if len(filtered) >= topK {
			break
		}
	}

	if filtered == nil {
		filtered = []map[string]any{}
	}

	return Result{OK: true, Output: map[string]any{
		"query":   q,
		"count":   len(filtered),
		"results": filtered,
	}}
}
