// Package search provides the semantic footage search index for Parallax.
// It stores embeddings for both video frame descriptions and audio transcript
// segments, then answers natural-language queries via cosine similarity.
package search

// SearchMeta is the metadata stored alongside each embedding vector.
type SearchMeta struct {
	FileID    string  `json:"file_id"`
	MediaPath string  `json:"media_path"`  // workspace-relative path
	StartSec  float64 `json:"start_sec"`
	EndSec    float64 `json:"end_sec"`
	Kind      string  `json:"kind"`        // "frame" | "transcript"
	Text      string  `json:"text"`        // frame description or transcript snippet
	ThumbPath string  `json:"thumb_path,omitempty"` // workspace-relative thumbnail
}

// Hit is one search result.
type Hit struct {
	Score float32    `json:"relevance_score"`
	Meta  SearchMeta `json:"meta"`
}

// IndexEntry is the persisted form of one embedding + metadata.
type IndexEntry struct {
	ID        string      `json:"id"`
	Embedding []float32   `json:"embedding"`
	Meta      SearchMeta  `json:"meta"`
}

// IndexingProgress reports background ingestion state.
type IndexingProgress struct {
	ProjectID string  `json:"project_id"`
	MediaPath string  `json:"media_path"`
	Phase     string  `json:"phase"` // "frames" | "transcript" | "embedding" | "done" | "error"
	Done      int     `json:"done"`
	Total     int     `json:"total"`
	Error     string  `json:"error,omitempty"`
}
