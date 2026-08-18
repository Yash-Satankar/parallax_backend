package reframe

import (
	"context"
	"math"
	"path/filepath"
	"strings"

	"parallax/internal/ffmpeg"
	"parallax/internal/llm"
	"parallax/internal/projects"
)

// AspectRatio describes a target video canvas dimension.
type AspectRatio struct {
	Name   string  `json:"name"`
	Label  string  `json:"label"`
	Width  int     `json:"width"`
	Height int     `json:"height"`
	Ratio  float64 `json:"ratio"`
}

var SupportedAspectRatios = map[string]AspectRatio{
	"16:9": {Name: "16:9", Label: "Landscape (16:9)", Width: 1920, Height: 1080, Ratio: 16.0 / 9.0},
	"9:16": {Name: "9:16", Label: "Vertical / Reels / TikTok (9:16)", Width: 1080, Height: 1920, Ratio: 9.0 / 16.0},
	"4:5":  {Name: "4:5", Label: "Portrait / Feed (4:5)", Width: 1080, Height: 1350, Ratio: 4.0 / 5.0},
	"1:1":  {Name: "1:1", Label: "Square (1:1)", Width: 1080, Height: 1080, Ratio: 1.0},
	"4:3":  {Name: "4:3", Label: "Classic (4:3)", Width: 1440, Height: 1080, Ratio: 4.0 / 3.0},
}

// NormalizeAspectRatio parses a target ratio string with alias support.
func NormalizeAspectRatio(s string) AspectRatio {
	clean := strings.ToLower(strings.TrimSpace(s))
	switch clean {
	case "9:16", "vertical", "reels", "tiktok", "shorts", "story":
		return SupportedAspectRatios["9:16"]
	case "4:5", "portrait", "instagram", "feed":
		return SupportedAspectRatios["4:5"]
	case "1:1", "square":
		return SupportedAspectRatios["1:1"]
	case "4:3", "classic", "standard":
		return SupportedAspectRatios["4:3"]
	default:
		return SupportedAspectRatios["16:9"]
	}
}

// CropValues contains the 4 normalized crop inset percentages (0..1).
type CropValues struct {
	CropTop    float64 `json:"crop_top"`
	CropRight  float64 `json:"crop_right"`
	CropBottom float64 `json:"crop_bottom"`
	CropLeft   float64 `json:"crop_left"`
}

// CalculateCrop computes normalized crop insets given source canvas, target ratio, and subject box.
func CalculateCrop(srcWidth, srcHeight int, target AspectRatio, subject BoundingBox) CropValues {
	if srcWidth <= 0 {
		srcWidth = 1920
	}
	if srcHeight <= 0 {
		srcHeight = 1080
	}

	srcRatio := float64(srcWidth) / float64(srcHeight)
	targetRatio := target.Ratio

	var top, right, bottom, left float64
	cx, cy := subject.Center()

	if targetRatio < srcRatio {
		// Cropping horizontally (e.g. 16:9 to 9:16)
		cropW := targetRatio / srcRatio // Fraction of width to keep
		minCX := cropW / 2.0
		maxCX := 1.0 - cropW/2.0
		clampedCX := clamp(cx, minCX, maxCX)

		left = clampedCX - cropW/2.0
		right = 1.0 - (left + cropW)
		top = 0.0
		bottom = 0.0
	} else if targetRatio > srcRatio {
		// Cropping vertically
		cropH := srcRatio / targetRatio // Fraction of height to keep
		minCY := cropH / 2.0
		maxCY := 1.0 - cropH/2.0
		clampedCY := clamp(cy, minCY, maxCY)

		top = clampedCY - cropH/2.0
		bottom = 1.0 - (top + cropH)
		left = 0.0
		right = 0.0
	}

	return CropValues{
		CropTop:    math.Round(top*10000) / 10000,
		CropRight:  math.Round(right*10000) / 10000,
		CropBottom: math.Round(bottom*10000) / 10000,
		CropLeft:   math.Round(left*10000) / 10000,
	}
}

// ReframePlan is the computed crop layout for a clip.
type ReframePlan struct {
	Ratio       AspectRatio                `json:"ratio"`
	Canvas      projects.TimelineCanvas    `json:"canvas"`
	Transform   projects.TimelineTransform `json:"transform"`
	Keyframes   []projects.TimelineKeyframe `json:"keyframes,omitempty"`
	Detections  []SubjectDetection         `json:"detections,omitempty"`
	MotionDelta float64                    `json:"motion_delta"`
}

// PlanClipReframe computes the optimal crop and keyframed pan path for a video clip.
func PlanClipReframe(
	ctx context.Context,
	bins ffmpeg.Bins,
	workspace string,
	mediaRelPath string,
	clipStartFrame int,
	clipSourceInFrame int,
	clipDurationFrames int,
	fps int,
	targetRatio AspectRatio,
	visionClient llm.ChatProvider,
) (ReframePlan, error) {
	if fps < 1 {
		fps = 24
	}

	// 1. Extract/sample scene frames via analyze_video_frames
	manifest, err := ffmpeg.AnalyzeVideoFrames(ctx, bins, mediaRelPath, workspace, 100)
	if err != nil {
		// Fallback: center crop
		fallbackBox := BoundingBox{XMin: 0.35, YMin: 0.20, XMax: 0.65, YMax: 0.80}
		crop := CalculateCrop(1920, 1080, targetRatio, fallbackBox)
		return ReframePlan{
			Ratio:  targetRatio,
			Canvas: projects.TimelineCanvas{Width: targetRatio.Width, Height: targetRatio.Height},
			Transform: projects.TimelineTransform{
				CropTop:    crop.CropTop,
				CropRight:  crop.CropRight,
				CropBottom: crop.CropBottom,
				CropLeft:   crop.CropLeft,
				ScaleX:     1,
				ScaleY:     1,
				Opacity:    1,
			},
		}, nil
	}

	// 2. Filter sampled frames within the clip's source in/duration range
	clipSourceInSec := float64(clipSourceInFrame) / float64(fps)
	clipSourceEndSec := clipSourceInSec + (float64(clipDurationFrames) / float64(fps))

	type timedDetection struct {
		timeSec float64
		box     BoundingBox
		det     SubjectDetection
	}

	var detections []timedDetection
	for _, scene := range manifest.Scenes {
		for _, frame := range scene.Frames {
			if frame.TimestampSec < clipSourceInSec || frame.TimestampSec > clipSourceEndSec {
				continue
			}
			absFrame := filepath.Join(workspace, filepath.FromSlash(frame.FramePath))
			det := DetectSubject(ctx, absFrame, visionClient)
			detections = append(detections, timedDetection{
				timeSec: frame.TimestampSec,
				box:     det.Box,
				det:     det,
			})
		}
	}

	// If no frames were sampled in range, probe and detect a single frame
	if len(detections) == 0 {
		fallbackBox := BoundingBox{XMin: 0.35, YMin: 0.20, XMax: 0.65, YMax: 0.80}
		crop := CalculateCrop(1920, 1080, targetRatio, fallbackBox)
		return ReframePlan{
			Ratio:  targetRatio,
			Canvas: projects.TimelineCanvas{Width: targetRatio.Width, Height: targetRatio.Height},
			Transform: projects.TimelineTransform{
				CropTop:    crop.CropTop,
				CropRight:  crop.CropRight,
				CropBottom: crop.CropBottom,
				CropLeft:   crop.CropLeft,
				ScaleX:     1,
				ScaleY:     1,
				Opacity:    1,
			},
		}, nil
	}

	// 3. Motion Analysis across the shot
	var minCX, maxCX float64 = 1.0, 0.0
	var sumCX, sumCY float64
	for _, d := range detections {
		cx, cy := d.box.Center()
		if cx < minCX {
			minCX = cx
		}
		if cx > maxCX {
			maxCX = cx
		}
		sumCX += cx
		sumCY += cy
	}

	motionDelta := maxCX - minCX
	avgBox := BoundingBox{
		XMin: (sumCX / float64(len(detections))) - 0.15,
		XMax: (sumCX / float64(len(detections))) + 0.15,
		YMin: (sumCY / float64(len(detections))) - 0.20,
		YMax: (sumCY / float64(len(detections))) + 0.20,
	}

	initialCrop := CalculateCrop(1920, 1080, targetRatio, avgBox)
	var keyframes []projects.TimelineKeyframe

	// 4. Deadband Filter: If motion delta < 5%, keep framing static
	if motionDelta >= 0.05 && len(detections) > 1 {
		// Generate smooth keyframes tracking the subject horizontally
		for _, d := range detections {
			relTimeSec := d.timeSec - clipSourceInSec
			keyFrame := int(relTimeSec*float64(fps) + 0.5)
			if keyFrame < 0 {
				keyFrame = 0
			}
			if keyFrame > clipDurationFrames {
				keyFrame = clipDurationFrames
			}

			frameCrop := CalculateCrop(1920, 1080, targetRatio, d.box)
			keyframes = append(keyframes,
				projects.TimelineKeyframe{
					Property: "transform.crop_left",
					Frame:    keyFrame,
					Value:    frameCrop.CropLeft,
					Easing:   "ease_in_out",
				},
				projects.TimelineKeyframe{
					Property: "transform.crop_right",
					Frame:    keyFrame,
					Value:    frameCrop.CropRight,
					Easing:   "ease_in_out",
				},
			)
		}
	}

	var rawDets []SubjectDetection
	for _, d := range detections {
		rawDets = append(rawDets, d.det)
	}

	return ReframePlan{
		Ratio:  targetRatio,
		Canvas: projects.TimelineCanvas{Width: targetRatio.Width, Height: targetRatio.Height},
		Transform: projects.TimelineTransform{
			CropTop:    initialCrop.CropTop,
			CropRight:  initialCrop.CropRight,
			CropBottom: initialCrop.CropBottom,
			CropLeft:   initialCrop.CropLeft,
			ScaleX:     1,
			ScaleY:     1,
			Opacity:    1,
		},
		Keyframes:   keyframes,
		Detections:  rawDets,
		MotionDelta: math.Round(motionDelta*1000) / 1000,
	}, nil
}
