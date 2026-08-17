package search

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// Index is an in-process cosine-similarity vector store.
// For project sizes up to ~100 k entries a flat linear scan is fast enough
// (< 50 ms at dim=1536). Swap to HNSW when needed without changing the API.
type Index struct {
	mu      sync.RWMutex
	entries []IndexEntry
	path    string // absolute path to the JSON persistence file
}

// NewIndex creates (or loads) the index for a project.
// indexPath is the absolute path of the JSON file used for persistence.
func NewIndex(indexPath string) (*Index, error) {
	idx := &Index{path: indexPath}
	if err := idx.load(); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("search index: load: %w", err)
	}
	return idx, nil
}

// Add upserts an entry identified by id.
func (idx *Index) Add(id string, vec []float32, meta SearchMeta) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for i, e := range idx.entries {
		if e.ID == id {
			idx.entries[i] = IndexEntry{ID: id, Embedding: vec, Meta: meta}
			return
		}
	}
	idx.entries = append(idx.entries, IndexEntry{ID: id, Embedding: vec, Meta: meta})
}

// RemoveByFile removes all entries whose MediaPath matches the given path.
func (idx *Index) RemoveByFile(mediaPath string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	out := idx.entries[:0]
	for _, e := range idx.entries {
		if e.Meta.MediaPath != mediaPath {
			out = append(out, e)
		}
	}
	idx.entries = out
}

// Query returns the top-k entries most similar to vec (cosine similarity).
func (idx *Index) Query(vec []float32, topK int) []Hit {
	idx.mu.RLock()
	entries := append([]IndexEntry(nil), idx.entries...)
	idx.mu.RUnlock()

	if len(entries) == 0 || topK <= 0 {
		return nil
	}

	type scored struct {
		score float32
		e     IndexEntry
	}
	results := make([]scored, 0, len(entries))
	qnorm := norm(vec)
	for _, e := range entries {
		if len(e.Embedding) == 0 {
			continue
		}
		s := cosine(vec, e.Embedding, qnorm)
		results = append(results, scored{s, e})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if topK > len(results) {
		topK = len(results)
	}
	hits := make([]Hit, topK)
	for i := range hits {
		hits[i] = Hit{Score: results[i].score, Meta: results[i].e.Meta}
	}
	return hits
}

// KeywordSearch performs a case-insensitive substring search on stored text.
// Useful as an exact-quote fallback: `search_footage` calls this when the
// semantic query looks like a verbatim quote (starts/ends with quotes).
func (idx *Index) KeywordSearch(query string, topK int) []Hit {
	idx.mu.RLock()
	entries := append([]IndexEntry(nil), idx.entries...)
	idx.mu.RUnlock()

	q := strings.ToLower(strings.Trim(query, `"'`))
	var hits []Hit
	for _, e := range entries {
		if strings.Contains(strings.ToLower(e.Meta.Text), q) {
			hits = append(hits, Hit{Score: 1.0, Meta: e.Meta})
		}
		if len(hits) >= topK {
			break
		}
	}
	return hits
}

// Len returns the number of entries in the index.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.entries)
}

// Save persists the index to disk atomically.
func (idx *Index) Save() error {
	idx.mu.RLock()
	entries := append([]IndexEntry(nil), idx.entries...)
	idx.mu.RUnlock()

	b, err := json.Marshal(entries)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(idx.path), 0o700); err != nil {
		return err
	}
	tmp := idx.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, idx.path)
}

func (idx *Index) load() error {
	b, err := os.ReadFile(idx.path)
	if err != nil {
		return err
	}
	var entries []IndexEntry
	if err := json.Unmarshal(b, &entries); err != nil {
		return err
	}
	idx.mu.Lock()
	idx.entries = entries
	idx.mu.Unlock()
	return nil
}

// EntryID generates a stable, deterministic ID for a (mediaPath, kind, startSec) triplet.
func EntryID(mediaPath, kind string, startSec float64) string {
	h := sha256.Sum256(fmt.Appendf(nil, "%s|%s|%.3f", mediaPath, kind, startSec))
	return fmt.Sprintf("%x", h[:8])
}

// --- vector math ---

func norm(v []float32) float32 {
	var s float32
	for _, x := range v {
		s += x * x
	}
	return float32(math.Sqrt(float64(s)))
}

func cosine(a, b []float32, anorm float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	bn := norm(b)
	if anorm == 0 || bn == 0 {
		return 0
	}
	return dot / (anorm * bn)
}
