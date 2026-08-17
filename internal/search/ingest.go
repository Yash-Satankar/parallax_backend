package search

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"parallax/internal/ffmpeg"
	"parallax/internal/llm"
)

// IngestConfig holds the dependencies required by IngestMedia.
type IngestConfig struct {
	// LLMClient is used for frame description (vision-capable model).
	LLMClient llm.ChatProvider
	// EmbedClient handles text→vector conversion.
	EmbedClient *llm.EmbedClient
	// TranscribeClient handles audio→transcript conversion.
	TranscribeClient *llm.TranscribeClient
	// Bins are the ffmpeg/ffprobe binaries.
	Bins ffmpeg.Bins
	// Workspace is the absolute project directory.
	Workspace string
	// ProgressCh receives indexing progress updates; may be nil.
	ProgressCh chan<- IndexingProgress
	// Logger is optional.
	Logger *slog.Logger
}

func (c *IngestConfig) log() *slog.Logger {
	if c.Logger != nil {
		return c.Logger
	}
	return slog.Default()
}

// IngestMedia runs the full analysis pipeline for a newly uploaded media file:
//  1. Extract representative frames via analyze_video_frames.
//  2. Describe each frame via vision-LLM.
//  3. Transcribe audio with word-level timestamps.
//  4. Embed all texts in batches.
//  5. Store entries in the project's Index.
//  6. Persist the index.
//
// It is designed to be called in a goroutine; all errors are logged and
// reported via ProgressCh rather than propagated (best-effort indexing).
func IngestMedia(ctx context.Context, cfg IngestConfig, mediaRelPath string, idx *Index) {
	report := func(phase string, done, total int, errMsg string) {
		if cfg.ProgressCh == nil {
			return
		}
		select {
		case cfg.ProgressCh <- IndexingProgress{
			MediaPath: mediaRelPath,
			Phase:     phase,
			Done:      done,
			Total:     total,
			Error:     errMsg,
		}:
		default:
		}
	}

	log := cfg.log().With("media", mediaRelPath)
	absPath := filepath.Join(cfg.Workspace, filepath.FromSlash(mediaRelPath))

	// Remove stale entries for this file first (re-index on re-upload).
	idx.RemoveByFile(mediaRelPath)

	// -----------------------------------------------------------------------
	// Step 1: Extract frames via analyze_video_frames
	// -----------------------------------------------------------------------
	report("frames", 0, 0, "")

	var frameManifest []FrameInfo
	ext := strings.ToLower(filepath.Ext(absPath))
	isVideo := ext == ".mp4" || ext == ".mov" || ext == ".mkv" || ext == ".webm" ||
		ext == ".avi" || ext == ".m4v" || ext == ".ts" || ext == ".mts"

	if isVideo {
		manifest, err := ffmpeg.AnalyzeVideoFrames(ctx, cfg.Bins, mediaRelPath, cfg.Workspace, 100)
		if err != nil {
			log.Warn("frame analysis failed", "err", err)
		} else {
			frameManifest = parseFrameManifest(manifest, cfg.Workspace)
			log.Info("frames extracted", "count", len(frameManifest))
		}
	}

	// -----------------------------------------------------------------------
	// Step 2: Describe frames via vision-LLM
	// -----------------------------------------------------------------------
	var frameTexts []embeddingJob
	if cfg.LLMClient != nil && len(frameManifest) > 0 {
		report("frames", 0, len(frameManifest), "")
		for i, f := range frameManifest {
			if ctx.Err() != nil {
				break
			}
			desc, err := describeFrame(ctx, cfg.LLMClient, f.AbsPath)
			if err != nil {
				log.Warn("frame description failed", "frame", f.AbsPath, "err", err)
				continue
			}
			frameTexts = append(frameTexts, embeddingJob{
				id:        EntryID(mediaRelPath, "frame", f.TimeSec),
				text:      desc,
				startSec:  f.TimeSec,
				endSec:    f.TimeSec + f.SceneDuration,
				kind:      "frame",
				thumbPath: f.RelPath,
			})
			report("frames", i+1, len(frameManifest), "")
		}
		log.Info("frames described", "count", len(frameTexts))
	}

	// -----------------------------------------------------------------------
	// Step 3: Transcribe audio
	// -----------------------------------------------------------------------
	var transcriptTexts []embeddingJob
	isAudio := isVideo || ext == ".mp3" || ext == ".wav" || ext == ".aac" ||
		ext == ".flac" || ext == ".m4a" || ext == ".ogg" || ext == ".opus"

	if cfg.TranscribeClient != nil && isAudio {
		report("transcript", 0, 1, "")
		// Extract audio to a temp WAV if needed (ffmpeg).
		audioPath, cleanup, err := extractAudio(ctx, cfg.Bins, absPath, cfg.Workspace)
		if err != nil {
			log.Warn("audio extraction failed", "err", err)
		} else {
			defer cleanup()
			transcript, err := cfg.TranscribeClient.Transcribe(ctx, audioPath, "")
			if err != nil {
				log.Warn("transcription failed", "err", err)
			} else {
				// Group words into ~8-word phrase groups for embedding granularity.
				groups := groupWords(transcript.Words, 8)
				for _, g := range groups {
					transcriptTexts = append(transcriptTexts, embeddingJob{
						id:       EntryID(mediaRelPath, "transcript", g.start),
						text:     g.text,
						startSec: g.start,
						endSec:   g.end,
						kind:     "transcript",
					})
				}
				log.Info("transcript processed", "words", len(transcript.Words), "groups", len(transcriptTexts))
			}
		}
		report("transcript", 1, 1, "")
	}

	// -----------------------------------------------------------------------
	// Step 4: Embed all texts in batches
	// -----------------------------------------------------------------------
	if cfg.EmbedClient == nil {
		log.Warn("no embed client configured; skipping index")
		report("done", 0, 0, "embedding not configured")
		return
	}

	allJobs := append(frameTexts, transcriptTexts...)
	if len(allJobs) == 0 {
		log.Info("nothing to index")
		report("done", 0, 0, "")
		return
	}

	report("embedding", 0, len(allJobs), "")

	const batchSize = 20
	for i := 0; i < len(allJobs); i += batchSize {
		if ctx.Err() != nil {
			break
		}
		end := i + batchSize
		if end > len(allJobs) {
			end = len(allJobs)
		}
		batch := allJobs[i:end]
		texts := make([]string, len(batch))
		for j, j2 := range batch {
			texts[j] = j2.text
		}

		vecs, err := cfg.EmbedClient.EmbedBatch(ctx, texts)
		if err != nil {
			log.Warn("embedding batch failed", "err", err)
			continue
		}
		for j, vec := range vecs {
			job := batch[j]
			idx.Add(job.id, vec, SearchMeta{
				FileID:    job.id,
				MediaPath: mediaRelPath,
				StartSec:  job.startSec,
				EndSec:    job.endSec,
				Kind:      job.kind,
				Text:      job.text,
				ThumbPath: job.thumbPath,
			})
		}
		report("embedding", i+len(batch), len(allJobs), "")
	}

	// -----------------------------------------------------------------------
	// Step 5: Persist
	// -----------------------------------------------------------------------
	if err := idx.Save(); err != nil {
		log.Error("index save failed", "err", err)
		report("done", len(allJobs), len(allJobs), "index save: "+err.Error())
		return
	}
	log.Info("indexing complete", "entries", idx.Len())
	report("done", len(allJobs), len(allJobs), "")
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

type embeddingJob struct {
	id        string
	text      string
	startSec  float64
	endSec    float64
	kind      string
	thumbPath string
}

// FrameInfo is extracted from the analyze_video_frames manifest.
type FrameInfo struct {
	AbsPath       string
	RelPath       string
	TimeSec       float64
	SceneDuration float64
}

// parseFrameManifest converts an ffmpeg.FrameManifest to a FrameInfo slice.
func parseFrameManifest(manifest ffmpeg.FrameManifest, workspace string) []FrameInfo {
	var out []FrameInfo
	for _, scene := range manifest.Scenes {
		sceneDur := scene.EndSec - scene.StartSec
		for _, f := range scene.Frames {
			rel := filepath.ToSlash(f.FramePath)
			abs := filepath.Join(workspace, filepath.FromSlash(f.FramePath))
			out = append(out, FrameInfo{
				AbsPath:       abs,
				RelPath:       rel,
				TimeSec:       f.TimestampSec,
				SceneDuration: sceneDur,
			})
		}
	}
	return out
}

// describeFrame asks the vision-LLM to describe a single frame image.
func describeFrame(ctx context.Context, provider llm.ChatProvider, absPath string) (string, error) {
	// Read the image and encode as data URI to pass inline.
	data, err := os.ReadFile(absPath)
	if err != nil {
		return "", err
	}
	ext := strings.ToLower(filepath.Ext(absPath))
	mime := "image/jpeg"
	if ext == ".png" {
		mime = "image/png"
	}

	// Build a vision message. The OpenAI-compat format for image_url content parts.
	import_msg := buildVisionMessage(data, mime)
	ch, err := provider.Stream(ctx, import_msg)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for d := range ch {
		if d.Err != nil {
			return "", d.Err
		}
		sb.WriteString(d.Content)
	}
	desc := strings.TrimSpace(sb.String())
	if desc == "" {
		return "", fmt.Errorf("empty description from LLM")
	}
	return desc, nil
}

// wordGroup is a phrase group derived from word-level transcript data.
type wordGroup struct {
	text  string
	start float64
	end   float64
}

// groupWords groups transcript words into phrase-sized chunks.
func groupWords(words []llm.TranscriptWord, groupSize int) []wordGroup {
	if len(words) == 0 {
		return nil
	}
	var groups []wordGroup
	for i := 0; i < len(words); i += groupSize {
		end := i + groupSize
		if end > len(words) {
			end = len(words)
		}
		chunk := words[i:end]
		var texts []string
		for _, w := range chunk {
			texts = append(texts, w.Word)
		}
		groups = append(groups, wordGroup{
			text:  strings.Join(texts, " "),
			start: chunk[0].Start,
			end:   chunk[len(chunk)-1].End,
		})
	}
	return groups
}

// extractAudio extracts the audio track to a temp WAV file for transcription.
// Returns the absolute path of the extracted file and a cleanup function.
func extractAudio(ctx context.Context, bins ffmpeg.Bins, absPath, workspace string) (string, func(), error) {
	tmp := filepath.Join(workspace, ".scratch", "transcribe-"+randHex(8)+".wav")
	if err := os.MkdirAll(filepath.Dir(tmp), 0o755); err != nil {
		return "", func() {}, err
	}

	relInput, err := filepath.Rel(workspace, absPath)
	if err != nil {
		relInput = absPath
	}
	relOutput, err := filepath.Rel(workspace, tmp)
	if err != nil {
		relOutput = tmp
	}

	cmd, err := ffmpeg.Validate([]string{
		"ffmpeg", "-y", "-i", filepath.ToSlash(relInput),
		"-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
		filepath.ToSlash(relOutput),
	}, ffmpeg.ValidateOpts{Workspace: workspace})
	if err != nil {
		return "", func() {}, fmt.Errorf("extractAudio: validate: %w", err)
	}
	res, err := ffmpeg.Run(ctx, bins, cmd, workspace, 0)
	if err != nil {
		_ = os.Remove(tmp)
		return "", func() {}, fmt.Errorf("extractAudio: ffmpeg: %w — %s", err, res.Stderr)
	}

	cleanup := func() { _ = os.Remove(tmp) }
	return tmp, cleanup, nil
}

// buildVisionMessage constructs an llm.Request with an image_url content part.
// The image is passed as a base64 data URI to avoid needing a public URL.
func buildVisionMessage(imageData []byte, mime string) llm.Request {
	b64 := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(imageData)
	// OpenAI vision format: content is an array of parts.
	// We encode this as a JSON string in the message Content field because
	// the llm.Message struct uses a simple string; providers that support vision
	// typically accept the array-of-parts format when content is a JSON array string.
	const prompt = "Describe this video frame in detail: people visible, their actions and emotions, objects, setting, and any text on screen. Be concise but specific (2-3 sentences)."
	contentJSON := fmt.Sprintf(`[{"type":"text","text":%q},{"type":"image_url","image_url":{"url":%q,"detail":"low"}}]`,
		prompt, b64)
	return llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: contentJSON},
		},
		Temperature: llm.Ptr(0.1),
	}
}

var mu sync.Mutex
var randCounter uint64

func randHex(n int) string {
	mu.Lock()
	randCounter++
	v := randCounter
	mu.Unlock()
	return fmt.Sprintf("%016x", v)[:n]
}
