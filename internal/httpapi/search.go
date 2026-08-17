package httpapi

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"parallax/internal/config"
	"parallax/internal/llm"
	"parallax/internal/search"
)

// handleProjectSearch handles GET /v1/projects/{id}/search?q=...&top_k=...&kind=...
func (s *Server) handleProjectSearch(w http.ResponseWriter, r *http.Request) {
	projectID := r.PathValue("id")
	project, err := s.Projects.Get(projectID)
	if err != nil {
		writeProjectError(w, err)
		return
	}

	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		writeError(w, http.StatusBadRequest, "query parameter 'q' is required")
		return
	}

	topK := 10
	if rawTopK := r.URL.Query().Get("top_k"); rawTopK != "" {
		if k, err := strconv.Atoi(rawTopK); err == nil && k > 0 {
			topK = k
		}
	}
	if topK > 50 {
		topK = 50
	}

	kindFilter := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("kind")))
	if kindFilter == "" {
		kindFilter = "all"
	}

	if s.SearchMgr == nil {
		writeError(w, http.StatusInternalServerError, "search manager not configured")
		return
	}

	idx, err := s.SearchMgr.GetIndex(project.Dir)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "get search index: "+err.Error())
		return
	}

	if idx.Len() == 0 {
		writeJSON(w, http.StatusOK, map[string]any{
			"query":   q,
			"count":   0,
			"results": []any{},
			"total":   0,
		})
		return
	}

	llmCfg := s.Settings.Get()
	embedClient := s.buildEmbedClient(llmCfg)

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

	if len(hits) == 0 && embedClient != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()
		vec, err := embedClient.Embed(ctx, q)
		if err == nil {
			hits = idx.Query(vec, topK*2)
		}
	}

	if len(hits) == 0 {
		hits = idx.KeywordSearch(q, topK)
	}

	type searchResultItem struct {
		FileID         string  `json:"file_id"`
		MediaPath      string  `json:"media_path"`
		ContentURL     string  `json:"content_url"`
		ThumbnailURL   string  `json:"thumbnail_url,omitempty"`
		StartSec       float64 `json:"start_sec"`
		EndSec         float64 `json:"end_sec"`
		Kind           string  `json:"kind"`
		Text           string  `json:"text"`
		RelevanceScore float32 `json:"relevance_score"`
	}

	var results []searchResultItem
	for _, h := range hits {
		if kindFilter != "all" && h.Meta.Kind != kindFilter {
			continue
		}
		item := searchResultItem{
			FileID:         h.Meta.FileID,
			MediaPath:      h.Meta.MediaPath,
			ContentURL:     projectFileURL(projectID, h.Meta.MediaPath),
			StartSec:       h.Meta.StartSec,
			EndSec:         h.Meta.EndSec,
			Kind:           h.Meta.Kind,
			Text:           h.Meta.Text,
			RelevanceScore: h.Score,
		}
		if h.Meta.ThumbPath != "" {
			item.ThumbnailURL = projectFileURL(projectID, h.Meta.ThumbPath)
		}
		results = append(results, item)
		if len(results) >= topK {
			break
		}
	}

	if results == nil {
		results = []searchResultItem{}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"query":   q,
		"count":   len(results),
		"results": results,
		"total":   idx.Len(),
	})
}

// triggerIngestion starts background frame analysis, transcription, and embedding for a media file.
func (s *Server) triggerIngestion(projectID, mediaRelPath string) {
	if s.SearchMgr == nil || s.Projects == nil {
		return
	}
	project, err := s.Projects.Get(projectID)
	if err != nil {
		s.log().Warn("ingest: project not found", "project", projectID, "err", err)
		return
	}

	idx, err := s.SearchMgr.GetIndex(project.Dir)
	if err != nil {
		s.log().Warn("ingest: get index", "project", projectID, "err", err)
		return
	}

	llmCfg := s.Settings.Get()
	var visionClient llm.ChatProvider
	if s.NewLLM != nil {
		visionClient = s.NewLLM(llmCfg)
	}

	embedClient := s.buildEmbedClient(llmCfg)
	transcribeClient := s.buildTranscribeClient(llmCfg)

	cfg := search.IngestConfig{
		LLMClient:        visionClient,
		EmbedClient:      embedClient,
		TranscribeClient: transcribeClient,
		Bins:             s.Bins,
		Workspace:        project.Dir,
		Logger:           s.log(),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	s.log().Info("starting background footage ingestion", "project", projectID, "media", mediaRelPath)
	search.IngestMedia(ctx, cfg, mediaRelPath, idx)
}

// buildEmbedClient creates an EmbedClient from LLM config if configured.
func (s *Server) buildEmbedClient(cfg config.LLM) *llm.EmbedClient {
	baseURL := cfg.EmbeddingBaseURL
	if baseURL == "" {
		baseURL = cfg.BaseURL
	}
	model := cfg.EmbeddingModel
	if model == "" {
		model = "text-embedding-3-small"
	}
	apiKey := cfg.APIKey
	if apiKey == "" || baseURL == "" {
		return nil
	}
	return llm.NewEmbedClient(baseURL, apiKey, model)
}

// buildTranscribeClient creates a TranscribeClient from LLM config if configured.
func (s *Server) buildTranscribeClient(cfg config.LLM) *llm.TranscribeClient {
	baseURL := cfg.TranscribeBaseURL
	if baseURL == "" {
		baseURL = cfg.BaseURL
	}
	model := cfg.TranscribeModel
	if model == "" {
		model = "whisper-1"
	}
	apiKey := cfg.APIKey
	if apiKey == "" || baseURL == "" {
		return nil
	}
	return llm.NewTranscribeClient(baseURL, apiKey, model)
}
