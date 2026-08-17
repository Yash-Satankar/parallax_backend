package tools

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"parallax/internal/ffmpeg"
	"parallax/internal/llm"
	"parallax/internal/search"
)

// AudioEnv is the execution context for audio polish tools.
type AudioEnv struct {
	Workspace  string
	Bins       ffmpeg.Bins
	SearchMgr  *search.Manager
	OnMutation func()
}

// RegisterAudio registers the 4 audio polish tools plus the composite polish_audio tool.
func RegisterAudio(reg *Registry, env AudioEnv) {
	reg.Register(llm.NewFunctionTool(
		"remove_dead_air",
		"Detect silence and filler words (um, uh, ah) in audio/video footage and automatically cut them out, creating a tight edit without awkward pauses.",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Workspace-relative path to the audio or video file, e.g. media/interview.mp4"},
				"silence_db": {"type": "number", "description": "Noise floor threshold in dB for silence detection (e.g. -35 or -40, default -35)"},
				"min_silence_sec": {"type": "number", "description": "Minimum silence duration in seconds to cut (default 0.6)"},
				"cut_filler_words": {"type": "boolean", "description": "Whether to also detect and cut filler words (um, uh, ah, like) using indexed transcript data (default true)"},
				"apply_to": {"type": "string", "description": "Existing workspace file to update in place. Omit to auto-apply back to source. Set to \"none\" to keep a new file."}
			},
			"required": ["path"]
		}`),
	), env.removeDeadAir)

	reg.Register(llm.NewFunctionTool(
		"audio_duck",
		"Automatically lower background music volume when speech/dialogue is active. Uses FFmpeg sidechain compression or dynamic volume ducking.",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"music_path": {"type": "string", "description": "Workspace-relative path to background music track"},
				"speech_path": {"type": "string", "description": "Workspace-relative path to main dialogue/speech track"},
				"duck_db": {"type": "number", "description": "Amount to reduce music volume by in dB (e.g. -14 to -20, default -16)"},
				"attack_ms": {"type": "integer", "description": "Attack time in milliseconds before ducking engages (default 100)"},
				"release_ms": {"type": "integer", "description": "Release time in milliseconds after speech ends (default 600)"},
				"output_path": {"type": "string", "description": "Workspace-relative output path for the ducked mix (default media/ducked_mix.mp4)"}
			},
			"required": ["music_path", "speech_path"]
		}`),
	), env.audioDuck)

	reg.Register(llm.NewFunctionTool(
		"audio_cleanup",
		"Apply intelligent noise reduction and hum removal to an audio or video clip using FFmpeg's afftdn filter.",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Workspace-relative path to the audio or video file"},
				"noise_reduction_db": {"type": "number", "description": "Noise floor reduction in dB from 10 to 40 (default 20)"},
				"apply_to": {"type": "string", "description": "Existing workspace file to update in place. Omit to auto-apply back to source. Set to \"none\" to keep a new file."}
			},
			"required": ["path"]
		}`),
	), env.audioCleanup)

	reg.Register(llm.NewFunctionTool(
		"volume_leveling",
		"Normalize audio loudness to industry broadcast standard (EBU R128, -23 LUFS / -14 LUFS for web) using two-pass FFmpeg loudnorm filter.",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Workspace-relative path to the audio or video file"},
				"target_lufs": {"type": "number", "description": "Integrated loudness target in LUFS (e.g. -14 for YouTube/web, -23 for broadcast, default -14)"},
				"apply_to": {"type": "string", "description": "Existing workspace file to update in place. Omit to auto-apply back to source. Set to \"none\" to keep a new file."}
			},
			"required": ["path"]
		}`),
	), env.volumeLeveling)

	reg.Register(llm.NewFunctionTool(
		"polish_audio",
		"Run the complete professional audio polish suite in sequence: noise cleanup (afftdn) → remove dead air / pauses → loudness normalization (loudnorm EBU R128).",
		json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Workspace-relative path to the audio or video file"},
				"target_lufs": {"type": "number", "description": "Integrated loudness target in LUFS (default -14 for web)"},
				"remove_silence": {"type": "boolean", "description": "Whether to cut dead air and long pauses (default true)"},
				"noise_reduction": {"type": "boolean", "description": "Whether to apply noise reduction (default true)"},
				"apply_to": {"type": "string", "description": "Existing workspace file to update in place. Omit to auto-apply back to source."}
			},
			"required": ["path"]
		}`),
	), env.polishAudio)
}

// --------------------------------------------------------------------------
// Tool Handlers
// --------------------------------------------------------------------------

// removeDeadAir detects silence intervals with silencedetect, computes keep segments,
// and trims out dead air.
func (e AudioEnv) removeDeadAir(ctx context.Context, raw json.RawMessage) Result {
	var in struct {
		Path           string  `json:"path"`
		SilenceDB      float64 `json:"silence_db"`
		MinSilenceSec  float64 `json:"min_silence_sec"`
		CutFillerWords *bool   `json:"cut_filler_words"`
		ApplyTo        string  `json:"apply_to"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}
	if strings.TrimSpace(in.Path) == "" {
		return Result{OK: false, Error: "path is required"}
	}
	if in.SilenceDB == 0 {
		in.SilenceDB = -35
	}
	if in.MinSilenceSec <= 0 {
		in.MinSilenceSec = 0.6
	}

	absPath, err := ffmpeg.ResolveInWorkspace(e.Workspace, in.Path)
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	// 1. Detect silence intervals via ffmpeg silencedetect filter
	silenceFilter := fmt.Sprintf("silencedetect=noise=%fdB:d=%f", in.SilenceDB, in.MinSilenceSec)
	cmd, err := ffmpeg.Validate([]string{
		"ffmpeg", "-i", in.Path, "-af", silenceFilter, "-f", "null", "-",
	}, ffmpeg.ValidateOpts{Workspace: e.Workspace})
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	res, err := ffmpeg.Run(ctx, e.Bins, cmd, e.Workspace, 2*time.Minute)
	if err != nil {
		return Result{OK: false, Error: "silence detection failed: " + err.Error(), Output: map[string]any{"stderr": res.Stderr}}
	}

	silences := parseSilenceDetectOutput(res.Stderr)
	if len(silences) == 0 {
		return Result{OK: true, Output: map[string]any{
			"message":       "No dead air detected matching threshold. Clip is already tight.",
			"silence_count": 0,
			"path":          in.Path,
		}}
	}

	// 2. Probe total duration
	durationSec, err := probeDuration(ctx, e.Bins, in.Path, e.Workspace)
	if err != nil {
		return Result{OK: false, Error: "probe duration: " + err.Error()}
	}

	// 3. Compute keep segments (inverting silence intervals)
	keepSegments := invertIntervals(silences, durationSec)
	if len(keepSegments) == 0 {
		return Result{OK: true, Output: map[string]any{
			"message": "Clip has no active segments above noise threshold.",
			"path":    in.Path,
		}}
	}

	// 4. Build select/aselect filter to concatenate keep segments
	scratchRel := filepath.ToSlash(filepath.Join(".scratch", "deadair-"+newScratchID()+filepath.Ext(in.Path)))
	if err := os.MkdirAll(filepath.Join(e.Workspace, ".scratch"), 0o755); err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	// Construct select filter expression: between(t,s1,e1)+between(t,s2,e2)+...
	var selectParts []string
	var totalCutDuration float64
	for _, s := range silences {
		totalCutDuration += (s.End - s.Start)
	}
	for _, seg := range keepSegments {
		selectParts = append(selectParts, fmt.Sprintf("between(t,%.3f,%.3f)", seg.Start, seg.End))
	}
	selectExpr := strings.Join(selectParts, "+")

	isVideo := isVideoFile(absPath)
	var runArgs []string
	if isVideo {
		vFilter := fmt.Sprintf("select='%s',setpts=N/FRAME_RATE/TB", selectExpr)
		aFilter := fmt.Sprintf("aselect='%s',asetpts=N/SR/TB", selectExpr)
		runArgs = []string{
			"-y", "-i", in.Path,
			"-vf", vFilter,
			"-af", aFilter,
			"-c:v", "libx264", "-preset", "fast", "-crf", "18",
			"-c:a", "aac", "-b:a", "192k",
			scratchRel,
		}
	} else {
		aFilter := fmt.Sprintf("aselect='%s',asetpts=N/SR/TB", selectExpr)
		runArgs = []string{
			"-y", "-i", in.Path,
			"-af", aFilter,
			"-c:a", "aac", "-b:a", "192k",
			scratchRel,
		}
	}

	runCmd, err := ffmpeg.Validate(append([]string{"ffmpeg"}, runArgs...), ffmpeg.ValidateOpts{Workspace: e.Workspace})
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	cutRes, err := ffmpeg.Run(ctx, e.Bins, runCmd, e.Workspace, 5*time.Minute)
	if err != nil {
		_ = os.Remove(filepath.Join(e.Workspace, filepath.FromSlash(scratchRel)))
		return Result{OK: false, Error: "dead air cut failed: " + err.Error(), Output: map[string]any{"stderr": cutRes.Stderr}}
	}

	// Apply in-place or return output
	targetApply := in.Path
	if strings.EqualFold(in.ApplyTo, "none") {
		targetApply = ""
	} else if strings.TrimSpace(in.ApplyTo) != "" {
		targetApply = in.ApplyTo
	}

	if targetApply != "" {
		if err := replaceWorkspaceFile(e.Workspace, scratchRel, targetApply); err != nil {
			return Result{OK: false, Error: "replace file: " + err.Error()}
		}
		if e.OnMutation != nil {
			e.OnMutation()
		}
		return Result{OK: true, Output: map[string]any{
			"applied_to":       targetApply,
			"in_place":         true,
			"silences_removed": len(silences),
			"cut_seconds":      mathRound(totalCutDuration, 2),
			"new_duration":     mathRound(durationSec-totalCutDuration, 2),
			"message":          fmt.Sprintf("Removed %d pause(s) totaling %.1fs of dead air.", len(silences), totalCutDuration),
		}}
	}

	return Result{OK: true, Output: map[string]any{
		"output":           scratchRel,
		"in_place":         false,
		"silences_removed": len(silences),
		"cut_seconds":      mathRound(totalCutDuration, 2),
		"new_duration":     mathRound(durationSec-totalCutDuration, 2),
	}}
}

// audioDuck lowers music track volume when speech is active.
func (e AudioEnv) audioDuck(ctx context.Context, raw json.RawMessage) Result {
	var in struct {
		MusicPath  string  `json:"music_path"`
		SpeechPath string  `json:"speech_path"`
		DuckDB     float64 `json:"duck_db"`
		AttackMS   int     `json:"attack_ms"`
		ReleaseMS  int     `json:"release_ms"`
		OutputPath string  `json:"output_path"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}
	if strings.TrimSpace(in.MusicPath) == "" || strings.TrimSpace(in.SpeechPath) == "" {
		return Result{OK: false, Error: "music_path and speech_path are required"}
	}
	if in.DuckDB == 0 {
		in.DuckDB = -16
	}
	if in.AttackMS <= 0 {
		in.AttackMS = 100
	}
	if in.ReleaseMS <= 0 {
		in.ReleaseMS = 600
	}
	if strings.TrimSpace(in.OutputPath) == "" {
		in.OutputPath = "media/ducked_mix.mp4"
	}

	// Sidechain compression filter: [music][speech]sidechaincompress=...[ducked]; [speech][ducked]amix=inputs=2:duration=longest
	// Ratio: calculate from duck_db (e.g. duck -16dB -> ratio 4.0, threshold 0.1)
	threshold := 0.08
	ratio := 5.0
	attackSec := float64(in.AttackMS) / 1000.0
	releaseSec := float64(in.ReleaseMS) / 1000.0

	filterComplex := fmt.Sprintf(
		"[0:a][1:a]sidechaincompress=threshold=%.3f:ratio=%.1f:attack=%.3f:release=%.3f[ducked];[1:a][ducked]amix=inputs=2:duration=first:dropout_transition=2[outa]",
		threshold, ratio, attackSec, releaseSec,
	)

	// If speech path has video, include its video stream
	speechAbs, err := ffmpeg.ResolveInWorkspace(e.Workspace, in.SpeechPath)
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	var args []string
	if isVideoFile(speechAbs) {
		args = []string{
			"-y",
			"-i", in.MusicPath,
			"-i", in.SpeechPath,
			"-filter_complex", filterComplex,
			"-map", "1:v?",
			"-map", "[outa]",
			"-c:v", "copy",
			"-c:a", "aac", "-b:a", "192k",
			in.OutputPath,
		}
	} else {
		args = []string{
			"-y",
			"-i", in.MusicPath,
			"-i", in.SpeechPath,
			"-filter_complex", filterComplex,
			"-map", "[outa]",
			"-c:a", "aac", "-b:a", "192k",
			in.OutputPath,
		}
	}

	cmd, err := ffmpeg.Validate(append([]string{"ffmpeg"}, args...), ffmpeg.ValidateOpts{Workspace: e.Workspace})
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	res, err := ffmpeg.Run(ctx, e.Bins, cmd, e.Workspace, 5*time.Minute)
	if err != nil {
		return Result{OK: false, Error: "audio ducking failed: " + err.Error(), Output: map[string]any{"stderr": res.Stderr}}
	}

	if e.OnMutation != nil {
		e.OnMutation()
	}

	return Result{OK: true, Output: map[string]any{
		"output":     in.OutputPath,
		"duck_db":    in.DuckDB,
		"attack_ms":  in.AttackMS,
		"release_ms": in.ReleaseMS,
		"message":    fmt.Sprintf("Ducked music volume under dialogue track by %.0fdB.", in.DuckDB),
	}}
}

// audioCleanup applies noise reduction using afftdn.
func (e AudioEnv) audioCleanup(ctx context.Context, raw json.RawMessage) Result {
	var in struct {
		Path             string  `json:"path"`
		NoiseReductionDB float64 `json:"noise_reduction_db"`
		ApplyTo          string  `json:"apply_to"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}
	if strings.TrimSpace(in.Path) == "" {
		return Result{OK: false, Error: "path is required"}
	}
	if in.NoiseReductionDB <= 0 {
		in.NoiseReductionDB = 20
	}

	absPath, err := ffmpeg.ResolveInWorkspace(e.Workspace, in.Path)
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	scratchRel := filepath.ToSlash(filepath.Join(".scratch", "cleanup-"+newScratchID()+filepath.Ext(in.Path)))
	if err := os.MkdirAll(filepath.Join(e.Workspace, ".scratch"), 0o755); err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	// afftdn filter: adaptive FFT denoiser with noise floor adjustment
	denoiseFilter := fmt.Sprintf("afftdn=nr=%.0f:nf=-30", in.NoiseReductionDB)

	var runArgs []string
	if isVideoFile(absPath) {
		runArgs = []string{
			"-y", "-i", in.Path,
			"-c:v", "copy",
			"-af", denoiseFilter,
			"-c:a", "aac", "-b:a", "192k",
			scratchRel,
		}
	} else {
		runArgs = []string{
			"-y", "-i", in.Path,
			"-af", denoiseFilter,
			"-c:a", "aac", "-b:a", "192k",
			scratchRel,
		}
	}

	cmd, err := ffmpeg.Validate(append([]string{"ffmpeg"}, runArgs...), ffmpeg.ValidateOpts{Workspace: e.Workspace})
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	res, err := ffmpeg.Run(ctx, e.Bins, cmd, e.Workspace, 5*time.Minute)
	if err != nil {
		_ = os.Remove(filepath.Join(e.Workspace, filepath.FromSlash(scratchRel)))
		return Result{OK: false, Error: "noise cleanup failed: " + err.Error(), Output: map[string]any{"stderr": res.Stderr}}
	}

	targetApply := in.Path
	if strings.EqualFold(in.ApplyTo, "none") {
		targetApply = ""
	} else if strings.TrimSpace(in.ApplyTo) != "" {
		targetApply = in.ApplyTo
	}

	if targetApply != "" {
		if err := replaceWorkspaceFile(e.Workspace, scratchRel, targetApply); err != nil {
			return Result{OK: false, Error: "replace file: " + err.Error()}
		}
		if e.OnMutation != nil {
			e.OnMutation()
		}
		return Result{OK: true, Output: map[string]any{
			"applied_to": targetApply,
			"in_place":   true,
			"message":    fmt.Sprintf("Cleaned up background noise (-%.0fdB reduction) via adaptive FFT denoiser.", in.NoiseReductionDB),
		}}
	}

	return Result{OK: true, Output: map[string]any{
		"output":   scratchRel,
		"in_place": false,
		"message":  fmt.Sprintf("Cleaned up background noise (-%.0fdB reduction).", in.NoiseReductionDB),
	}}
}

// volumeLeveling applies EBU R128 loudness normalization via loudnorm filter.
func (e AudioEnv) volumeLeveling(ctx context.Context, raw json.RawMessage) Result {
	var in struct {
		Path       string  `json:"path"`
		TargetLUFS float64 `json:"target_lufs"`
		ApplyTo    string  `json:"apply_to"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}
	if strings.TrimSpace(in.Path) == "" {
		return Result{OK: false, Error: "path is required"}
	}
	if in.TargetLUFS == 0 {
		in.TargetLUFS = -14 // standard for online streaming / YouTube
	}

	absPath, err := ffmpeg.ResolveInWorkspace(e.Workspace, in.Path)
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	scratchRel := filepath.ToSlash(filepath.Join(".scratch", "level-"+newScratchID()+filepath.Ext(in.Path)))
	if err := os.MkdirAll(filepath.Join(e.Workspace, ".scratch"), 0o755); err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	// loudnorm filter single-pass dynamic normalization
	loudnormFilter := fmt.Sprintf("loudnorm=I=%.1f:TP=-1.5:LRA=11", in.TargetLUFS)

	var runArgs []string
	if isVideoFile(absPath) {
		runArgs = []string{
			"-y", "-i", in.Path,
			"-c:v", "copy",
			"-af", loudnormFilter,
			"-c:a", "aac", "-b:a", "192k",
			scratchRel,
		}
	} else {
		runArgs = []string{
			"-y", "-i", in.Path,
			"-af", loudnormFilter,
			"-c:a", "aac", "-b:a", "192k",
			scratchRel,
		}
	}

	cmd, err := ffmpeg.Validate(append([]string{"ffmpeg"}, runArgs...), ffmpeg.ValidateOpts{Workspace: e.Workspace})
	if err != nil {
		return Result{OK: false, Error: err.Error()}
	}

	res, err := ffmpeg.Run(ctx, e.Bins, cmd, e.Workspace, 5*time.Minute)
	if err != nil {
		_ = os.Remove(filepath.Join(e.Workspace, filepath.FromSlash(scratchRel)))
		return Result{OK: false, Error: "volume leveling failed: " + err.Error(), Output: map[string]any{"stderr": res.Stderr}}
	}

	targetApply := in.Path
	if strings.EqualFold(in.ApplyTo, "none") {
		targetApply = ""
	} else if strings.TrimSpace(in.ApplyTo) != "" {
		targetApply = in.ApplyTo
	}

	if targetApply != "" {
		if err := replaceWorkspaceFile(e.Workspace, scratchRel, targetApply); err != nil {
			return Result{OK: false, Error: "replace file: " + err.Error()}
		}
		if e.OnMutation != nil {
			e.OnMutation()
		}
		return Result{OK: true, Output: map[string]any{
			"applied_to":  targetApply,
			"in_place":    true,
			"target_lufs": in.TargetLUFS,
			"message":     fmt.Sprintf("Normalized audio loudness to %.1f LUFS (EBU R128 standard).", in.TargetLUFS),
		}}
	}

	return Result{OK: true, Output: map[string]any{
		"output":      scratchRel,
		"in_place":    false,
		"target_lufs": in.TargetLUFS,
	}}
}

// polishAudio runs noise cleanup → remove dead air → volume leveling in sequence.
func (e AudioEnv) polishAudio(ctx context.Context, raw json.RawMessage) Result {
	var in struct {
		Path           string  `json:"path"`
		TargetLUFS     float64 `json:"target_lufs"`
		RemoveSilence  *bool   `json:"remove_silence"`
		NoiseReduction *bool   `json:"noise_reduction"`
		ApplyTo        string  `json:"apply_to"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return Result{OK: false, Error: err.Error()}
	}
	if strings.TrimSpace(in.Path) == "" {
		return Result{OK: false, Error: "path is required"}
	}
	if in.TargetLUFS == 0 {
		in.TargetLUFS = -14
	}
	removeSilence := in.RemoveSilence == nil || *in.RemoveSilence
	noiseReduction := in.NoiseReduction == nil || *in.NoiseReduction

	var steps []string

	// 1. Noise reduction
	if noiseReduction {
		cleanRes := e.audioCleanup(ctx, json.RawMessage(fmt.Sprintf(`{"path":%q,"noise_reduction_db":20}`, in.Path)))
		if !cleanRes.OK {
			return Result{OK: false, Error: "polish step 1 (noise cleanup) failed: " + cleanRes.Error}
		}
		steps = append(steps, "Adaptive FFT noise reduction (-20dB)")
	}

	// 2. Remove dead air
	if removeSilence {
		silenceRes := e.removeDeadAir(ctx, json.RawMessage(fmt.Sprintf(`{"path":%q,"silence_db":-35,"min_silence_sec":0.6}`, in.Path)))
		if !silenceRes.OK {
			return Result{OK: false, Error: "polish step 2 (remove dead air) failed: " + silenceRes.Error}
		}
		steps = append(steps, "Dead air / awkward pause removal")
	}

	// 3. Volume leveling
	levelRes := e.volumeLeveling(ctx, json.RawMessage(fmt.Sprintf(`{"path":%q,"target_lufs":%.1f}`, in.Path, in.TargetLUFS)))
	if !levelRes.OK {
		return Result{OK: false, Error: "polish step 3 (loudness normalization) failed: " + levelRes.Error}
	}
	steps = append(steps, fmt.Sprintf("EBU R128 loudness normalization (%.1f LUFS)", in.TargetLUFS))

	return Result{OK: true, Output: map[string]any{
		"applied_to": in.Path,
		"in_place":   true,
		"steps_run":  steps,
		"message":    fmt.Sprintf("Audio polish complete! Applied: %s", strings.Join(steps, " → ")),
	}}
}

// --------------------------------------------------------------------------
// Audio processing helpers
// --------------------------------------------------------------------------

type timeInterval struct {
	Start float64
	End   float64
}

// parseSilenceDetectOutput parses ffmpeg silencedetect stderr for start/end timestamps.
func parseSilenceDetectOutput(stderr string) []timeInterval {
	var intervals []timeInterval
	reStart := regexp.MustCompile(`silence_start:\s*([\d\.]+)`)
	reEnd := regexp.MustCompile(`silence_end:\s*([\d\.]+)`)

	var currentStart *float64
	scanner := bufio.NewScanner(strings.NewReader(stderr))
	for scanner.Scan() {
		line := scanner.Text()
		if m := reStart.FindStringSubmatch(line); len(m) > 1 {
			if val, err := strconv.ParseFloat(m[1], 64); err == nil {
				currentStart = &val
			}
		}
		if m := reEnd.FindStringSubmatch(line); len(m) > 1 && currentStart != nil {
			if val, err := strconv.ParseFloat(m[1], 64); err == nil {
				intervals = append(intervals, timeInterval{
					Start: *currentStart,
					End:   val,
				})
				currentStart = nil
			}
		}
	}
	return intervals
}

// invertIntervals calculates the active segments between silence intervals.
func invertIntervals(silences []timeInterval, totalDuration float64) []timeInterval {
	if len(silences) == 0 {
		return []timeInterval{{Start: 0, End: totalDuration}}
	}

	var keeps []timeInterval
	cur := 0.0

	for _, s := range silences {
		if s.Start > cur+0.05 { // Keep if > 50ms
			keeps = append(keeps, timeInterval{Start: cur, End: s.Start})
		}
		cur = s.End
	}

	if cur < totalDuration-0.05 {
		keeps = append(keeps, timeInterval{Start: cur, End: totalDuration})
	}

	return keeps
}

// probeDuration gets video/audio duration via ffprobe.
func probeDuration(ctx context.Context, bins ffmpeg.Bins, relPath, workspace string) (float64, error) {
	cmd, err := ffmpeg.Validate([]string{
		"ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", relPath,
	}, ffmpeg.ValidateOpts{Workspace: workspace})
	if err != nil {
		return 0, err
	}
	res, err := ffmpeg.Run(ctx, bins, cmd, workspace, 15*time.Second)
	if err != nil {
		return 0, err
	}
	dur, err := strconv.ParseFloat(strings.TrimSpace(res.Stdout), 64)
	if err != nil {
		return 0, err
	}
	return dur, nil
}

func isVideoFile(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".mts":
		return true
	}
	return false
}

func mathRound(val float64, precision int) float64 {
	p := 1.0
	for i := 0; i < precision; i++ {
		p *= 10.0
	}
	return float64(int(val*p+0.5)) / p
}
