package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"parallax/internal/captions"
	"parallax/internal/collab"
	"parallax/internal/ffmpeg"
	"parallax/internal/llm"
	"parallax/internal/projects"
)

// CaptionsEnv provides dependencies for the generate_captions agent tool.
type CaptionsEnv struct {
	Transaction      *projects.TimelineTransaction
	Store            *projects.Store
	ProjectID        string
	Workspace        string
	Bins             ffmpeg.Bins
	TranscribeClient *llm.TranscribeClient
	CollabHub        *collab.Hub
}

// RegisterCaptions registers the generate_captions tool in the agent registry.
func RegisterCaptions(reg *Registry, env CaptionsEnv) {
	reg.Register(llm.NewFunctionTool(
		"generate_captions",
		"Automatically generate animated, styled captions for a video/audio clip on the timeline from speech transcript data. Creates non-destructive title clips on track V2 with word-level reveal timing. Choose between 4 presets: 'subtitle' (clean bottom-third), 'stacked' (word bursts with neon highlight), 'minimal' (understated pill), or 'serif' (editorial documentary).",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"clip_id": {
					"type": "string",
					"description": "ID of the target video or audio clip on the timeline, or media path such as media/interview.mp4. Omit to target the first active speech clip."
				},
				"style": {
					"type": "string",
					"enum": ["subtitle", "stacked", "minimal", "serif"],
					"description": "Caption style preset: 'subtitle' (default), 'stacked', 'minimal', or 'serif'"
				}
			}
		}`),
	), env.generateCaptions)
}

func (e CaptionsEnv) generateCaptions(ctx context.Context, raw json.RawMessage) Result {
	if e.Transaction == nil {
		return Result{OK: false, Error: "timeline transaction is unavailable"}
	}

	var in struct {
		ClipID string `json:"clip_id"`
		Style  string `json:"style"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	doc := e.Transaction.Get()
	if len(doc.Clips) == 0 {
		return Result{OK: false, Error: "timeline is empty; place media on the timeline first"}
	}

	// 1. Locate target clip
	var targetClip *projects.TimelineClip
	targetID := strings.TrimSpace(in.ClipID)

	if targetID != "" {
		for i := range doc.Clips {
			c := &doc.Clips[i]
			if c.ID == targetID || c.MediaPath == targetID || strings.HasSuffix(c.MediaPath, targetID) {
				targetClip = c
				break
			}
		}
	}

	// Fallback to first video or audio clip if not specified
	if targetClip == nil {
		for i := range doc.Clips {
			c := &doc.Clips[i]
			if (c.Kind == "video" || c.Kind == "audio") && c.MediaPath != "" {
				targetClip = c
				break
			}
		}
	}

	if targetClip == nil {
		return Result{OK: false, Error: "could not find target media clip on timeline"}
	}

	// 2. Fetch transcript words
	store := captions.TranscriptStore{
		Workspace: e.Workspace,
		Bins:      e.Bins,
		Client:    e.TranscribeClient,
	}

	words, err := store.GetTranscript(ctx, targetClip.MediaPath)
	if err != nil {
		return Result{OK: false, Error: fmt.Sprintf("get transcript for %s: %v", targetClip.MediaPath, err)}
	}

	if len(words) == 0 {
		return Result{OK: true, Output: map[string]any{
			"message": "No spoken dialogue detected in footage.",
			"count":   0,
		}}
	}

	// 3. Generate caption title clips
	fps := doc.FPS
	if fps < 1 {
		fps = 24
	}

	sourceInSec := float64(targetClip.SourceInFrame) / float64(fps)
	durationSec := float64(targetClip.DurationFrames) / float64(fps)
	preset := captions.NormalizePreset(in.Style)

	captionClips := captions.BuildCaptionClips(
		words,
		preset,
		targetClip.StartFrame,
		sourceInSec,
		durationSec,
		fps,
	)

	if len(captionClips) == 0 {
		return Result{OK: true, Output: map[string]any{
			"message": "No spoken dialogue in the selected range of the clip.",
			"count":   0,
		}}
	}

	// 4. Build timeline operations
	var ops []projects.TimelineOperation

	// Remove old captions in the same frame range on track V2 if any exist
	var removeIDs []string
	targetStart := targetClip.StartFrame
	targetEnd := targetClip.StartFrame + targetClip.DurationFrames
	for _, c := range doc.Clips {
		if c.Track == "V2" && c.Kind == "title" && strings.HasPrefix(c.ID, "cap-") {
			cStart := c.StartFrame
			cEnd := c.StartFrame + c.DurationFrames
			if (cStart >= targetStart && cStart < targetEnd) || (cEnd > targetStart && cEnd <= targetEnd) {
				removeIDs = append(removeIDs, c.ID)
			}
		}
	}
	if len(removeIDs) > 0 {
		ops = append(ops, projects.TimelineOperation{
			Type: "remove_items",
			IDs:  removeIDs,
		})
	}

	// Add new caption clips
	for i := range captionClips {
		item := captionClips[i]
		ops = append(ops, projects.TimelineOperation{
			Type: "add_item",
			Item: &item,
		})
	}

	// Apply operations to transaction
	result, err := e.Transaction.Apply(ops)
	if err != nil {
		return Result{OK: false, Error: "apply captions to timeline: " + err.Error()}
	}

	// Broadcast collaborative updates
	if e.CollabHub != nil && e.ProjectID != "" {
		clipByID := make(map[string]projects.TimelineClip, len(result.Timeline.Clips))
		for _, c := range result.Timeline.Clips {
			clipByID[c.ID] = c
		}
		for _, id := range result.CreatedIDs {
			if clip, ok := clipByID[id]; ok {
				e.CollabHub.PublishClipInsert(e.ProjectID, clip, collab.KeyBetween("", ""))
			}
		}
		for _, id := range result.RemovedIDs {
			e.CollabHub.PublishClipDelete(e.ProjectID, id)
		}
	}

	cfg := captions.GetPresetConfig(preset)
	return Result{OK: true, Output: map[string]any{
		"style":          cfg.Label,
		"captions_count": len(captionClips),
		"staged":         true,
		"timeline":       result.Timeline,
		"note":           fmt.Sprintf("Generated %d animated caption clips in '%s' style on track V2.", len(captionClips), cfg.Label),
	}}
}
