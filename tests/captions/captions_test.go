package tests

import (
  "testing"

  "parallax/internal/captions"
  "parallax/internal/llm"
)

func TestChunkWordsAndASS(t *testing.T) {
  words := []llm.TranscriptWord{
    {Word: "Hello", Start: 0.0, End: 0.5},
    {Word: "world", Start: 0.5, End: 1.0},
    {Word: "this", Start: 1.2, End: 1.5},
    {Word: "is", Start: 1.6, End: 1.8},
    {Word: "a", Start: 1.9, End: 2.0},
    {Word: "test", Start: 2.1, End: 2.6},
  }

  cfg := captions.GetPresetConfig(captions.PresetSubtitle)
  chunks := captions.ChunkWords(words, cfg, 0.3)
  if len(chunks) == 0 {
    t.Fatalf("expected chunks > 0")
  }

  clips := captions.BuildCaptionClips(words, captions.PresetSubtitle, 0, 0, 3.0, 24)
  if len(clips) == 0 {
    t.Fatalf("expected caption clips generated")
  }

  ass := captions.GenerateASS(clips, 1920, 1080)
  if ass == "" {
    t.Fatalf("expected non-empty ASS output")
  }
}
