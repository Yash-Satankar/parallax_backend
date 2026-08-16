package ffmpeg

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// FrameManifest describes the set of frames extracted from a video for analysis.
type FrameManifest struct {
	VideoPath   string       `json:"video_path"`
	TotalFrames int          `json:"total_frames"`
	Scenes      []SceneEntry `json:"scenes"`
	ExtractedAt time.Time    `json:"extracted_at"`
}

// SceneEntry is one detected scene segment with its sampled frames.
type SceneEntry struct {
	SceneID     int           `json:"scene_id"`
	StartSec    float64       `json:"start_sec"`
	EndSec      float64       `json:"end_sec"`
	MotionScore float64       `json:"motion_score"` // 0.0 (static) – 1.0 (high motion)
	Frames      []FrameEntry  `json:"frames"`
}

// FrameEntry is one sampled video frame.
type FrameEntry struct {
	TimestampSec float64 `json:"timestamp_sec"`
	FramePath    string  `json:"frame_path"` // relative to workspace
}

const (
	defaultMaxFrames      = 250
	defaultSceneThreshold = 0.3
	minFPS                = 0.2  // 1 frame per 5 s (static scenes)
	maxFPS                = 2.5  // 2.5 frames per second (high-motion scenes)
)

// sceneChange holds a single detected scene boundary.
type sceneChange struct {
	timestampSec float64
	score        float64 // ffmpeg scene-change score 0.0–1.0
}

// AnalyzeVideoFrames performs adaptive frame sampling on a workspace-relative
// video path and extracts frames to outDir (workspace-relative).
// It returns a FrameManifest describing all extracted frames.
func AnalyzeVideoFrames(ctx context.Context, bins Bins, videoPath, workspace string, maxFrames int) (FrameManifest, error) {
	if maxFrames < 1 {
		maxFrames = defaultMaxFrames
	}

	absVideo := filepath.Join(workspace, videoPath)
	if _, err := os.Stat(absVideo); err != nil {
		return FrameManifest{}, fmt.Errorf("video not found: %w", err)
	}

	// 1. Get total video duration.
	duration, err := probeVideoDuration(ctx, bins, absVideo, workspace)
	if err != nil {
		return FrameManifest{}, fmt.Errorf("probe duration: %w", err)
	}
	if duration <= 0 {
		return FrameManifest{}, fmt.Errorf("could not determine video duration")
	}

	// 2. Detect scene boundaries using FFmpeg's scdet filter.
	changes, err := detectSceneChanges(ctx, bins, absVideo, workspace)
	if err != nil {
		// Gracefully fall back: treat entire video as one scene.
		changes = nil
	}

	// 3. Build scene segments from change points.
	scenes := buildScenes(changes, duration)

	// 4. Assign sampling density per scene based on motion score.
	assignSamplingDensity(scenes, maxFrames, duration)

	// 5. Extract frames.
	outDir := filepath.Join(".parallax", "frames", sanitizeFilename(videoPath))
	absOutDir := filepath.Join(workspace, outDir)
	if err := os.MkdirAll(absOutDir, 0o755); err != nil {
		return FrameManifest{}, fmt.Errorf("create frame dir: %w", err)
	}

	manifest := FrameManifest{
		VideoPath:   videoPath,
		ExtractedAt: time.Now().UTC(),
	}

	for si := range scenes {
		scene := &scenes[si]
		for fi, ts := range scene.sampleTimestamps {
			frameName := fmt.Sprintf("s%02d_f%04d_%06d.jpg", scene.SceneID, fi, int(ts*100))
			frameRel := filepath.ToSlash(filepath.Join(outDir, frameName))
			frameAbs := filepath.Join(workspace, frameRel)

			if err := extractFrame(ctx, bins, absVideo, workspace, ts, frameAbs); err != nil {
				// Skip frame on error; don't abort entire extraction.
				continue
			}
			scene.Frames = append(scene.Frames, FrameEntry{
				TimestampSec: ts,
				FramePath:    frameRel,
			})
		}
		manifest.TotalFrames += len(scene.Frames)
		manifest.Scenes = append(manifest.Scenes, scene.SceneEntry)
	}

	return manifest, nil
}

// -----------------------------------------------------------------------
// Internal scene type with sampling state
// -----------------------------------------------------------------------

type scene struct {
	SceneEntry
	sampleTimestamps []float64
}

// -----------------------------------------------------------------------
// Scene detection
// -----------------------------------------------------------------------

// detectSceneChanges runs FFmpeg with the scdet filter and parses its output.
// Returns scene change events with timestamps and scores.
func detectSceneChanges(ctx context.Context, bins Bins, absVideo, workspace string) ([]sceneChange, error) {
	// Use null muxer; scene info written to stderr by scdet filter.
	cmd := Command{
		Kind: KindFFmpeg,
		Args: []string{
			"-hide_banner",
			"-i", absVideo,
			"-vf", fmt.Sprintf("scdet=threshold=%.2f:sc_pass=1", defaultSceneThreshold),
			"-f", "null",
			nullDevice(),
		},
	}
	res, _ := Run(ctx, bins, cmd, workspace, 5*time.Minute)
	// Parse stderr for lines like:
	// [scdet @ 0x...] lavfi.scene_score: 0.456 Parsed_scdet_0 @ ...
	// Or in newer FFmpeg: pts=NNN ts=T.T score=S.S new_scene=1
	return parseScdetOutput(res.Stderr), nil
}

// parseScdetOutput extracts scene changes from FFmpeg scdet stderr output.
func parseScdetOutput(stderr string) []sceneChange {
	var changes []sceneChange
	scanner := bufio.NewScanner(strings.NewReader(stderr))
	for scanner.Scan() {
		line := scanner.Text()
		// Match lines: "pts_time:T scene_score:S new_scene:1"
		// or newer format with commas / key=value pairs
		if !strings.Contains(line, "scene_score") && !strings.Contains(line, "lavfi.scene_score") {
			continue
		}

		var ts, score float64
		var hasScore bool

		// Try newer format: "pts_time:T ..." with "scene_score:S"
		for _, part := range strings.Fields(line) {
			if after, ok := strings.CutPrefix(part, "pts_time:"); ok {
				ts, _ = strconv.ParseFloat(after, 64)
			}
			if after, ok := strings.CutPrefix(part, "scene_score:"); ok {
				score, _ = strconv.ParseFloat(after, 64)
				hasScore = true
			}
			// Older format "lavfi.scene_score: S"
			if after, ok := strings.CutPrefix(part, "lavfi.scene_score:"); ok {
				score, _ = strconv.ParseFloat(after, 64)
				hasScore = true
			}
		}
		if hasScore && ts > 0 {
			changes = append(changes, sceneChange{timestampSec: ts, score: score})
		}
	}
	return changes
}

// -----------------------------------------------------------------------
// Scene building
// -----------------------------------------------------------------------

func buildScenes(changes []sceneChange, duration float64) []scene {
	// Always include t=0 as the start of the first scene.
	boundaries := []float64{0}
	motionAt := map[float64]float64{}
	for _, c := range changes {
		boundaries = append(boundaries, c.timestampSec)
		motionAt[c.timestampSec] = c.score
	}
	boundaries = append(boundaries, duration)

	scenes := make([]scene, 0, len(boundaries)-1)
	for i := 0; i < len(boundaries)-1; i++ {
		start := boundaries[i]
		end := boundaries[i+1]
		if end <= start {
			continue
		}
		// Motion score: average of scene-change scores within this segment.
		score := motionAt[boundaries[i+1]]

		scenes = append(scenes, scene{
			SceneEntry: SceneEntry{
				SceneID:     i,
				StartSec:    start,
				EndSec:      end,
				MotionScore: clampF(score, 0, 1),
			},
		})
	}
	if len(scenes) == 0 {
		// Fallback: whole video as one scene.
		scenes = append(scenes, scene{
			SceneEntry: SceneEntry{
				SceneID:     0,
				StartSec:    0,
				EndSec:      duration,
				MotionScore: 0.5,
			},
		})
	}
	return scenes
}

// -----------------------------------------------------------------------
// Adaptive sampling density
// -----------------------------------------------------------------------

// assignSamplingDensity maps each scene's motion score to a frames-per-second
// sampling rate using a smooth power-curve, then scales all scenes
// proportionally to stay within maxFrames.
func assignSamplingDensity(scenes []scene, maxFrames int, totalDuration float64) {
	// 1. Compute unconstrained frame counts per scene.
	const gamma = 1.5 // controls curve shape
	rawCounts := make([]float64, len(scenes))
	rawTotal := 0.0
	for i, s := range scenes {
		dur := s.EndSec - s.StartSec
		fps := minFPS + (maxFPS-minFPS)*math.Pow(s.MotionScore, gamma)
		count := math.Max(1, dur*fps)
		rawCounts[i] = count
		rawTotal += count
	}

	// 2. Scale proportionally to stay within budget.
	scale := 1.0
	if rawTotal > float64(maxFrames) {
		scale = float64(maxFrames) / rawTotal
	}

	for i := range scenes {
		count := int(math.Max(1, math.Round(rawCounts[i]*scale)))
		dur := scenes[i].EndSec - scenes[i].StartSec
		scenes[i].sampleTimestamps = evenlySpaced(scenes[i].StartSec, dur, count)
	}
}

// evenlySpaced returns n timestamps evenly spread across [start, start+dur].
func evenlySpaced(start, dur float64, n int) []float64 {
	if n <= 0 {
		return nil
	}
	if n == 1 {
		return []float64{start + dur/2}
	}
	ts := make([]float64, n)
	step := dur / float64(n-1)
	for i := range ts {
		ts[i] = start + float64(i)*step
	}
	return ts
}

// -----------------------------------------------------------------------
// Frame extraction
// -----------------------------------------------------------------------

func extractFrame(ctx context.Context, bins Bins, absVideo, workspace string, timestampSec float64, absFramePath string) error {
	cmd := Command{
		Kind: KindFFmpeg,
		Args: []string{
			"-hide_banner",
			"-loglevel", "error",
			"-ss", fmt.Sprintf("%.4f", timestampSec),
			"-i", absVideo,
			"-frames:v", "1",
			"-q:v", "3",
			"-y",
			absFramePath,
		},
	}
	_, err := Run(ctx, bins, cmd, workspace, 30*time.Second)
	return err
}

// probeVideoDuration returns the video's duration using ffprobe.
func probeVideoDuration(ctx context.Context, bins Bins, absVideo, workspace string) (float64, error) {
	cmd := Command{
		Kind: KindFFprobe,
		Args: []string{
			"-v", "error",
			"-show_entries", "format=duration",
			"-of", "default=noprint_wrappers=1:nokey=1",
			absVideo,
		},
	}
	res, err := Run(ctx, bins, cmd, workspace, 30*time.Second)
	if err != nil {
		return 0, err
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(res.Stdout), 64)
	if err != nil {
		return 0, fmt.Errorf("parse duration: %w", err)
	}
	return v, nil
}

// nullDevice returns the OS-appropriate null output device.
func nullDevice() string {
	// On Windows: NUL ; everywhere else: /dev/null
	if filepath.Separator == '\\' {
		return "NUL"
	}
	return "/dev/null"
}

func sanitizeFilename(path string) string {
	base := filepath.Base(filepath.ToSlash(path))
	result := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, base)
	return result
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
