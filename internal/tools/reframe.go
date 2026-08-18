package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"parallax/internal/collab"
	"parallax/internal/ffmpeg"
	"parallax/internal/llm"
	"parallax/internal/projects"
	"parallax/internal/reframe"
)

// ReframeEnv provides dependencies for the reframe_clip agent tool.
type ReframeEnv struct {
	Transaction  *projects.TimelineTransaction
	Store        *projects.Store
	ProjectID    string
	Workspace    string
	Bins         ffmpeg.Bins
	VisionClient llm.ChatProvider
	CollabHub    *collab.Hub
}

// RegisterReframe registers the reframe_clip tool in the agent registry.
func RegisterReframe(reg *Registry, env ReframeEnv) {
	reg.Register(llm.NewFunctionTool(
		"reframe_clip",
		"Intelligently convert a video clip or timeline to vertical or custom aspect ratios (16:9, 9:16, 4:5, 1:1, 4:3). Uses high-speed Go face detection (Pigo) with Vision-LLM fallback to track subjects, generate optimal crops, and produce smooth keyframed pans.",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"clip_id": {
					"type": "string",
					"description": "ID of the video clip on the timeline or media path. Omit to target the primary video clip on V1."
				},
				"target_ratios": {
					"type": "array",
					"items": {"type": "string"},
					"description": "One or more target aspect ratios, e.g. ['9:16'] for TikTok/Reels, ['1:1'] for Square, ['4:5'] for Feed (default ['9:16'])"
				}
			}
		}`),
	), env.reframeClip)
}

func (e ReframeEnv) reframeClip(ctx context.Context, raw json.RawMessage) Result {
	if e.Transaction == nil {
		return Result{OK: false, Error: "timeline transaction is unavailable"}
	}

	var in struct {
		ClipID       string   `json:"clip_id"`
		TargetRatios []string `json:"target_ratios"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	doc := e.Transaction.Get()
	if len(doc.Clips) == 0 {
		return Result{OK: false, Error: "timeline is empty; place media on the timeline first"}
	}

	// 1. Find target clip
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

	if targetClip == nil {
		for i := range doc.Clips {
			c := &doc.Clips[i]
			if c.Kind == "video" && c.MediaPath != "" {
				targetClip = c
				break
			}
		}
	}

	if targetClip == nil {
		return Result{OK: false, Error: "could not find a video clip to reframe"}
	}

	// Default to 9:16 vertical if no ratios specified
	targetRatioStrs := in.TargetRatios
	if len(targetRatioStrs) == 0 {
		targetRatioStrs = []string{"9:16"}
	}

	primaryRatio := reframe.NormalizeAspectRatio(targetRatioStrs[0])

	// 2. Plan reframe for primary target ratio
	fps := doc.FPS
	if fps < 1 {
		fps = 24
	}

	plan, err := reframe.PlanClipReframe(
		ctx,
		e.Bins,
		e.Workspace,
		targetClip.MediaPath,
		targetClip.StartFrame,
		targetClip.SourceInFrame,
		targetClip.DurationFrames,
		fps,
		primaryRatio,
		e.VisionClient,
	)
	if err != nil {
		return Result{OK: false, Error: "reframe planning failed: " + err.Error()}
	}

	// 3. Build updated clip
	updatedClip := *targetClip
	if updatedClip.Transform == nil {
		updatedClip.Transform = &projects.TimelineTransform{
			X:       float64(primaryRatio.Width) / 2.0,
			Y:       float64(primaryRatio.Height) / 2.0,
			AnchorX: 0.5,
			AnchorY: 0.5,
			ScaleX:  1,
			ScaleY:  1,
			Opacity: 1,
		}
	}
	updatedClip.Transform.CropTop = plan.Transform.CropTop
	updatedClip.Transform.CropRight = plan.Transform.CropRight
	updatedClip.Transform.CropBottom = plan.Transform.CropBottom
	updatedClip.Transform.CropLeft = plan.Transform.CropLeft

	// Replace existing crop keyframes with new smoothed tracking keyframes
	var nonCropKeyframes []projects.TimelineKeyframe
	for _, k := range updatedClip.Keyframes {
		if !strings.HasPrefix(k.Property, "transform.crop") {
			nonCropKeyframes = append(nonCropKeyframes, k)
		}
	}
	updatedClip.Keyframes = append(nonCropKeyframes, plan.Keyframes...)

	// 4. Update title/caption overlays on V2 to fit new vertical/square canvas
	var titleUpdates []projects.TimelineOperation
	for i := range doc.Clips {
		c := doc.Clips[i]
		if c.Track == "V2" && c.Kind == "title" && c.Title != nil {
			modTitleClip := c
			if modTitleClip.Transform == nil {
				modTitleClip.Transform = &projects.TimelineTransform{
					ScaleX: 1, ScaleY: 1, Opacity: 1, AnchorX: 0.5, AnchorY: 1.0,
				}
			}
			modTitleClip.Transform.X = float64(primaryRatio.Width) / 2.0
			modTitleClip.Transform.Y = float64(primaryRatio.Height) * 0.85
			titleUpdates = append(titleUpdates, projects.TimelineOperation{
				Type: "update_item",
				Item: &modTitleClip,
			})
		}
	}

	// Apply operations
	ops := []projects.TimelineOperation{
		{
			Type: "update_item",
			Item: &updatedClip,
		},
	}
	ops = append(ops, titleUpdates...)

	_, err = e.Transaction.Apply(ops)
	if err != nil {
		return Result{OK: false, Error: "apply reframe to timeline: " + err.Error()}
	}

	// Set canvas resolution on timeline
	e.Transaction.SetCanvas(plan.Canvas)

	// Broadcast updates
	if e.CollabHub != nil && e.ProjectID != "" {
		e.CollabHub.PublishFieldUpdate(e.ProjectID, updatedClip.ID, map[string]any{
			"transform": updatedClip.Transform,
			"keyframes": updatedClip.Keyframes,
		})
	}

	detectorSummary := "Pigo face detection"
	if len(plan.Detections) > 0 && plan.Detections[0].Source == "vision_llm" {
		detectorSummary = "Vision-LLM subject detection"
	}

	return Result{OK: true, Output: map[string]any{
		"ratio":            primaryRatio.Name,
		"canvas":           plan.Canvas,
		"crop":             plan.Transform,
		"keyframes_count":  len(plan.Keyframes),
		"motion_delta":     plan.MotionDelta,
		"detector_used":    detectorSummary,
		"detections_count": len(plan.Detections),
		"staged":           true,
		"timeline":         e.Transaction.Get(),
		"note": fmt.Sprintf(
			"Reframed %s to %s (%dx%d) using %s with %d keyframe(s).",
			updatedClip.Name, primaryRatio.Name, plan.Canvas.Width, plan.Canvas.Height, detectorSummary, len(plan.Keyframes),
		),
	}}
}
