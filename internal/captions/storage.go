package captions

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"parallax/internal/ffmpeg"
	"parallax/internal/llm"
)

// TranscriptStore manages loading and caching of word-level transcript data.
type TranscriptStore struct {
	Workspace string
	Bins      ffmpeg.Bins
	Client    *llm.TranscribeClient
}

// CachedTranscript is the persistent JSON format for transcript words.
type CachedTranscript struct {
	MediaPath string               `json:"media_path"`
	Duration  float64              `json:"duration"`
	Language  string               `json:"language,omitempty"`
	Words     []llm.TranscriptWord `json:"words"`
}

// GetTranscript retrieves word-level timestamps for the given workspace-relative media path.
// It checks local cache first, then falls back to transcription if a client is available.
func (s *TranscriptStore) GetTranscript(ctx context.Context, mediaRelPath string) ([]llm.TranscriptWord, error) {
	cleanRel := filepath.ToSlash(filepath.Clean(mediaRelPath))
	cachePath := filepath.Join(s.Workspace, ".parallax", "transcripts", sanitizeFilename(cleanRel)+".json")

	// 1. Check disk cache
	if data, err := os.ReadFile(cachePath); err == nil {
		var cached CachedTranscript
		if err := json.Unmarshal(data, &cached); err == nil && len(cached.Words) > 0 {
			return cached.Words, nil
		}
	}

	// 2. If no TranscribeClient is available, return error
	if s.Client == nil {
		return nil, fmt.Errorf("no transcript cached for %s and transcription client is not configured", mediaRelPath)
	}

	absMedia := filepath.Join(s.Workspace, filepath.FromSlash(cleanRel))
	if _, err := os.Stat(absMedia); err != nil {
		return nil, fmt.Errorf("media file not found: %w", err)
	}

	// 3. Extract audio to temp WAV for Whisper transcription
	tmpAudio := filepath.Join(s.Workspace, ".scratch", "cap-transcribe-"+randHex(8)+".wav")
	if err := os.MkdirAll(filepath.Dir(tmpAudio), 0o755); err != nil {
		return nil, err
	}
	defer os.Remove(tmpAudio)

	cmd, err := ffmpeg.Validate([]string{
		"ffmpeg", "-y", "-i", cleanRel,
		"-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
		filepath.ToSlash(filepath.Join(".scratch", filepath.Base(tmpAudio))),
	}, ffmpeg.ValidateOpts{Workspace: s.Workspace})
	if err != nil {
		return nil, fmt.Errorf("validate audio extract: %w", err)
	}

	res, err := ffmpeg.Run(ctx, s.Bins, cmd, s.Workspace, 0)
	if err != nil {
		return nil, fmt.Errorf("extract audio: %w — %s", err, res.Stderr)
	}

	// 4. Run Whisper transcription
	transcript, err := s.Client.Transcribe(ctx, tmpAudio, "")
	if err != nil {
		return nil, fmt.Errorf("transcription failed: %w", err)
	}

	words := transcript.Words
	if len(words) == 0 && len(transcript.Segments) > 0 {
		// Flatten segment words if needed
		for _, seg := range transcript.Segments {
			words = append(words, seg.Words...)
		}
	}

	// 5. Cache result on disk
	if len(words) > 0 {
		_ = os.MkdirAll(filepath.Dir(cachePath), 0o755)
		cached := CachedTranscript{
			MediaPath: cleanRel,
			Duration:  transcript.Duration,
			Language:  transcript.Language,
			Words:     words,
		}
		if b, err := json.MarshalIndent(cached, "", "  "); err == nil {
			_ = os.WriteFile(cachePath, b, 0o644)
		}
	}

	return words, nil
}

func sanitizeFilename(path string) string {
	s := filepath.ToSlash(path)
	s = strings.ReplaceAll(s, "/", "_")
	s = strings.ReplaceAll(s, "\\", "_")
	s = strings.ReplaceAll(s, ":", "_")
	return strings.TrimPrefix(s, "_")
}

func randHex(n int) string {
	return fmt.Sprintf("%016x", os.Getpid())[:n]
}
