package search_test

import (
	"os"
	"path/filepath"
	"testing"

	"parallax/internal/search"
)

func TestVectorIndexAddAndQuery(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test_index.json")

	idx, err := search.NewIndex(indexPath)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	// Add 3 vectors
	// Vector A: [1.0, 0.0, 0.0]
	// Vector B: [0.0, 1.0, 0.0]
	// Vector C: [0.707, 0.707, 0.0]
	idx.Add("entry-1", []float32{1.0, 0.0, 0.0}, search.SearchMeta{
		FileID:    "entry-1",
		MediaPath: "media/talk.mp4",
		StartSec:  0.0,
		EndSec:    5.0,
		Kind:      "frame",
		Text:      "Person giving a presentation about machine learning",
	})
	idx.Add("entry-2", []float32{0.0, 1.0, 0.0}, search.SearchMeta{
		FileID:    "entry-2",
		MediaPath: "media/nature.mp4",
		StartSec:  10.0,
		EndSec:    15.0,
		Kind:      "frame",
		Text:      "Sunset over the mountains with red and orange sky",
	})
	idx.Add("entry-3", []float32{0.7071, 0.7071, 0.0}, search.SearchMeta{
		FileID:    "entry-3",
		MediaPath: "media/talk.mp4",
		StartSec:  5.0,
		EndSec:    10.0,
		Kind:      "transcript",
		Text:      "We saw incredible results in our latest deep learning benchmark",
	})

	if idx.Len() != 3 {
		t.Fatalf("expected len 3, got %d", idx.Len())
	}

	// Query closest to Vector A [1.0, 0.0, 0.0]
	hits := idx.Query([]float32{1.0, 0.0, 0.0}, 2)
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits, got %d", len(hits))
	}
	if hits[0].Meta.FileID != "entry-1" {
		t.Fatalf("expected top hit entry-1, got %s", hits[0].Meta.FileID)
	}
	if hits[0].Score < 0.99 {
		t.Fatalf("expected score ~1.0, got %f", hits[0].Score)
	}
	if hits[1].Meta.FileID != "entry-3" {
		t.Fatalf("expected second hit entry-3, got %s", hits[1].Meta.FileID)
	}

	// Test persistence
	if err := idx.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Reload from disk
	loadedIdx, err := search.NewIndex(indexPath)
	if err != nil {
		t.Fatalf("NewIndex reload: %v", err)
	}
	if loadedIdx.Len() != 3 {
		t.Fatalf("expected loaded len 3, got %d", loadedIdx.Len())
	}

	// Test keyword search
	kwHits := loadedIdx.KeywordSearch("sunset", 5)
	if len(kwHits) != 1 {
		t.Fatalf("expected 1 keyword hit, got %d", len(kwHits))
	}
	if kwHits[0].Meta.FileID != "entry-2" {
		t.Fatalf("expected entry-2, got %s", kwHits[0].Meta.FileID)
	}
}

func TestVectorIndexRemoveByFile(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test_index2.json")
	idx, _ := search.NewIndex(indexPath)

	idx.Add("e1", []float32{1, 0}, search.SearchMeta{MediaPath: "media/a.mp4"})
	idx.Add("e2", []float32{0, 1}, search.SearchMeta{MediaPath: "media/b.mp4"})
	idx.Add("e3", []float32{1, 1}, search.SearchMeta{MediaPath: "media/a.mp4"})

	if idx.Len() != 3 {
		t.Fatalf("expected 3, got %d", idx.Len())
	}

	idx.RemoveByFile("media/a.mp4")
	if idx.Len() != 1 {
		t.Fatalf("expected 1 after removal, got %d", idx.Len())
	}
	_ = os.Remove(indexPath)
}
