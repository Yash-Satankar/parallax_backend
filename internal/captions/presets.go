package captions

import (
	"fmt"
	"strings"

	"parallax/internal/llm"
	"parallax/internal/projects"
)

// StylePreset represents one of the 4 supported caption styles.
type StylePreset string

const (
	PresetSubtitle StylePreset = "subtitle"
	PresetStacked  StylePreset = "stacked"
	PresetMinimal  StylePreset = "minimal"
	PresetSerif    StylePreset = "serif"
)

// NormalizePreset returns a valid StylePreset, defaulting to PresetSubtitle.
func NormalizePreset(s string) StylePreset {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "stacked", "bold", "burst", "karaoke":
		return PresetStacked
	case "minimal", "clean", "subtle":
		return PresetMinimal
	case "serif", "editorial", "documentary", "classic":
		return PresetSerif
	default:
		return PresetSubtitle
	}
}

// PresetConfig holds layout and styling parameters for a caption preset.
type PresetConfig struct {
	Name           StylePreset
	Label          string
	FontFamily     string
	FontSize       float64
	FontWeight     int
	Align          string
	Fill           string
	Stroke         string
	StrokeWidth    float64
	Background     string
	HighlightColor string
	ActiveScale    float64
	TransformY     float64 // Y position on 1080p canvas (960x1080)
	AnchorY        float64
	MaxWords       int
	MaxChars       int
	Uppercase      bool
}

// GetPresetConfig returns the configuration for a given preset.
func GetPresetConfig(preset StylePreset) PresetConfig {
	switch preset {
	case PresetStacked:
		return PresetConfig{
			Name:           PresetStacked,
			Label:          "Stacked",
			FontFamily:     "Montserrat, Impact, Arial Black, sans-serif",
			FontSize:       76,
			FontWeight:     900,
			Align:          "center",
			Fill:           "#ffffff",
			Stroke:         "#000000",
			StrokeWidth:    5,
			Background:     "",
			HighlightColor: "#00f0ff", // Neon Cyan
			ActiveScale:    1.18,
			TransformY:     680,
			AnchorY:        0.5,
			MaxWords:       3,
			MaxChars:       24,
			Uppercase:      true,
		}
	case PresetMinimal:
		return PresetConfig{
			Name:           PresetMinimal,
			Label:          "Minimal",
			FontFamily:     "Inter, system-ui, -apple-system, sans-serif",
			FontSize:       38,
			FontWeight:     500,
			Align:          "center",
			Fill:           "#f0f0f0",
			Stroke:         "",
			StrokeWidth:    0,
			Background:     "#000000aa", // Translucent black pill
			HighlightColor: "#ffffff",
			ActiveScale:    1.0,
			TransformY:     960,
			AnchorY:        1.0,
			MaxWords:       6,
			MaxChars:       40,
			Uppercase:      false,
		}
	case PresetSerif:
		return PresetConfig{
			Name:           PresetSerif,
			Label:          "Serif",
			FontFamily:     "Georgia, Playfair Display, Cambria, serif",
			FontSize:       52,
			FontWeight:     600,
			Align:          "center",
			Fill:           "#f8f6f0",
			Stroke:         "#1a1a1a",
			StrokeWidth:    2,
			Background:     "",
			HighlightColor: "#f5c542", // Warm Gold
			ActiveScale:    1.06,
			TransformY:     930,
			AnchorY:        1.0,
			MaxWords:       7,
			MaxChars:       46,
			Uppercase:      false,
		}
	default: // PresetSubtitle
		return PresetConfig{
			Name:           PresetSubtitle,
			Label:          "Subtitle",
			FontFamily:     "Inter, Helvetica Neue, Arial, sans-serif",
			FontSize:       54,
			FontWeight:     700,
			Align:          "center",
			Fill:           "#ffffff",
			Stroke:         "#000000",
			StrokeWidth:    3.5,
			Background:     "",
			HighlightColor: "#ffe600", // Vibrant Yellow
			ActiveScale:    1.0,
			TransformY:     940,
			AnchorY:        1.0,
			MaxWords:       7,
			MaxChars:       45,
			Uppercase:      false,
		}
	}
}

// ChunkWords segments a list of transcript words into natural phrase blocks.
func ChunkWords(words []llm.TranscriptWord, cfg PresetConfig, minPauseSec float64) [][]llm.TranscriptWord {
	if len(words) == 0 {
		return nil
	}
	if minPauseSec <= 0 {
		minPauseSec = 0.4
	}

	var chunks [][]llm.TranscriptWord
	var current []llm.TranscriptWord
	currentChars := 0

	for i, w := range words {
		text := strings.TrimSpace(w.Word)
		if text == "" {
			continue
		}
		if cfg.Uppercase {
			text = strings.ToUpper(text)
		}
		w.Word = text

		if len(current) == 0 {
			current = append(current, w)
			currentChars = len(text)
			continue
		}

		prev := current[len(current)-1]
		pause := w.Start - prev.End
		hasEndingPunct := endsWithPunctuation(prev.Word)
		reachedMaxWords := len(current) >= cfg.MaxWords
		reachedMaxChars := currentChars+len(text)+1 > cfg.MaxChars

		// Break chunk if long pause, sentence end, or length limit exceeded
		if pause >= minPauseSec || hasEndingPunct || reachedMaxWords || reachedMaxChars {
			chunks = append(chunks, current)
			current = []llm.TranscriptWord{w}
			currentChars = len(text)
		} else {
			current = append(current, w)
			currentChars += len(text) + 1
		}

		if i == len(words)-1 && len(current) > 0 {
			chunks = append(chunks, current)
		}
	}

	if len(current) > 0 && (len(chunks) == 0 || &chunks[len(chunks)-1][0] != &current[0]) {
		chunks = append(chunks, current)
	}

	return chunks
}

func endsWithPunctuation(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return false
	}
	last := rune(s[len(s)-1])
	return last == '.' || last == '!' || last == '?' || last == '—' || last == ';'
}

// BuildCaptionClips creates timeline title clips for a target clip using transcript words.
func BuildCaptionClips(
	words []llm.TranscriptWord,
	style StylePreset,
	clipStartFrame int,
	clipSourceInSec float64,
	clipDurationSec float64,
	fps int,
) []projects.TimelineClip {
	if len(words) == 0 {
		return nil
	}
	if fps < 1 {
		fps = 24
	}

	cfg := GetPresetConfig(style)
	clipEndSec := clipSourceInSec + clipDurationSec

	// Filter words within the clip's source range
	var clipWords []llm.TranscriptWord
	for _, w := range words {
		if w.End < clipSourceInSec || w.Start > clipEndSec {
			continue
		}
		// Adjust word timestamps relative to the source clip in-point
		relWord := w
		if relWord.Start < clipSourceInSec {
			relWord.Start = clipSourceInSec
		}
		if relWord.End > clipEndSec {
			relWord.End = clipEndSec
		}
		clipWords = append(clipWords, relWord)
	}

	if len(clipWords) == 0 {
		return nil
	}

	chunks := ChunkWords(clipWords, cfg, 0.4)
	out := make([]projects.TimelineClip, 0, len(chunks))

	for idx, chunk := range chunks {
		if len(chunk) == 0 {
			continue
		}
		chunkStartSec := chunk[0].Start - clipSourceInSec
		chunkEndSec := chunk[len(chunk)-1].End - clipSourceInSec
		if chunkEndSec <= chunkStartSec {
			chunkEndSec = chunkStartSec + 0.5
		}

		// Calculate frame timings relative to the timeline
		startFrame := clipStartFrame + int(chunkStartSec*float64(fps)+0.5)
		durationFrames := int((chunkEndSec-chunkStartSec)*float64(fps) + 0.5)
		if durationFrames < 1 {
			durationFrames = 1
		}

		var wordsData []projects.TimelineCaptionWord
		var fullTextBuilder strings.Builder
		for _, cw := range chunk {
			wRelStart := cw.Start - clipSourceInSec
			wRelEnd := cw.End - clipSourceInSec
			wStartFrame := clipStartFrame + int(wRelStart*float64(fps)+0.5)
			wDurFrames := int((wRelEnd-wRelStart)*float64(fps) + 0.5)
			if wDurFrames < 1 {
				wDurFrames = 1
			}

			wText := cw.Word
			if cfg.Uppercase {
				wText = strings.ToUpper(wText)
			}
			if fullTextBuilder.Len() > 0 {
				fullTextBuilder.WriteString(" ")
			}
			fullTextBuilder.WriteString(wText)

			wordsData = append(wordsData, projects.TimelineCaptionWord{
				Word:           wText,
				StartSec:       cw.Start,
				EndSec:         cw.End,
				StartFrame:     wStartFrame,
				DurationFrames: wDurFrames,
			})
		}

		fullText := fullTextBuilder.String()
		clipID := fmt.Sprintf("cap-%s-%04d", strings.ToLower(string(cfg.Name)), idx+1)

		titleClip := projects.TimelineClip{
			ID:             clipID,
			Name:           fullText,
			Track:          "V2",
			Kind:           "title",
			StartFrame:     startFrame,
			DurationFrames: durationFrames,
			SourceInFrame:  0,
			Transform: &projects.TimelineTransform{
				X:       960,
				Y:       cfg.TransformY,
				AnchorX: 0.5,
				AnchorY: cfg.AnchorY,
				ScaleX:  1,
				ScaleY:  1,
				Opacity: 1,
			},
			Title: &projects.TimelineTitle{
				Text:           fullText,
				FontFamily:     cfg.FontFamily,
				FontSize:       cfg.FontSize,
				FontWeight:     cfg.FontWeight,
				Align:          cfg.Align,
				Fill:           cfg.Fill,
				Stroke:         cfg.Stroke,
				StrokeWidth:    cfg.StrokeWidth,
				Background:     cfg.Background,
				StylePreset:    string(cfg.Name),
				HighlightColor: cfg.HighlightColor,
				ActiveScale:    cfg.ActiveScale,
				Words:          wordsData,
			},
		}

		out = append(out, titleClip)
	}

	return out
}

// GenerateASS creates an Advanced SubStation Alpha (.ass) subtitle document with karaoke tags.
func GenerateASS(clips []projects.TimelineClip, canvasWidth, canvasHeight int) string {
	if canvasWidth <= 0 {
		canvasWidth = 1920
	}
	if canvasHeight <= 0 {
		canvasHeight = 1080
	}

	var sb strings.Builder
	sb.WriteString("[Script Info]\n")
	sb.WriteString("Title: Parallax Animated Captions\n")
	sb.WriteString("ScriptType: v4.00+\n")
	sb.WriteString(fmt.Sprintf("PlayResX: %d\n", canvasWidth))
	sb.WriteString(fmt.Sprintf("PlayResY: %d\n", canvasHeight))
	sb.WriteString("WrapStyle: 0\n")
	sb.WriteString("ScaledBorderAndShadow: yes\n\n")

	sb.WriteString("[V4+ Styles]\n")
	sb.WriteString("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")

	// Standard ASS color format: &HAABBGGRR (Alpha, Blue, Green, Red in hex)
	styles := []StylePreset{PresetSubtitle, PresetStacked, PresetMinimal, PresetSerif}
	for _, st := range styles {
		cfg := GetPresetConfig(st)
		fontName := strings.Split(cfg.FontFamily, ",")[0]
		fontName = strings.TrimSpace(fontName)

		priCol := hexToASSColor(cfg.HighlightColor, "00")
		secCol := hexToASSColor(cfg.Fill, "00")
		outCol := hexToASSColor(cfg.Stroke, "00")
		if cfg.Stroke == "" {
			outCol = "&H00000000"
		}
		backCol := "&H80000000"

		bold := -1
		if cfg.FontWeight >= 700 {
			bold = 1
		}
		align := 2 // Bottom-Center
		if cfg.AnchorY == 0.5 {
			align = 5 // Middle-Center
		}

		marginV := int(float64(canvasHeight) - cfg.TransformY)
		if marginV < 20 {
			marginV = 40
		}

		sb.WriteString(fmt.Sprintf(
			"Style: %s,%s,%.0f,%s,%s,%s,%s,%d,0,0,0,100,100,0,0,1,%.1f,1,%d,40,40,%d,1\n",
			cfg.Label, fontName, cfg.FontSize, priCol, secCol, outCol, backCol,
			bold, cfg.StrokeWidth, align, marginV,
		))
	}

	sb.WriteString("\n[Events]\n")
	sb.WriteString("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

	for _, clip := range clips {
		if clip.Kind != "title" || clip.Title == nil {
			continue
		}
		t := clip.Title
		preset := NormalizePreset(t.StylePreset)
		cfg := GetPresetConfig(preset)

		// Convert frame times to ASS timecodes (H:MM:SS.CC)
		fps := 24.0
		startSec := float64(clip.StartFrame) / fps
		endSec := float64(clip.StartFrame+clip.DurationFrames) / fps
		startTimecode := formatASSTime(startSec)
		endTimecode := formatASSTime(endSec)

		var textWithTags strings.Builder
		if len(t.Words) > 0 {
			for _, w := range t.Words {
				durCenti := int((w.EndSec-w.StartSec)*100 + 0.5)
				if durCenti < 1 {
					durCenti = 10
				}
				// Karaoke tag \k<duration_in_centisec>
				textWithTags.WriteString(fmt.Sprintf("{\\kf%d}%s ", durCenti, w.Word))
			}
		} else {
			textWithTags.WriteString(t.Text)
		}

		cleanText := strings.TrimSpace(textWithTags.String())
		sb.WriteString(fmt.Sprintf(
			"Dialogue: 0,%s,%s,%s,,0,0,0,,%s\n",
			startTimecode, endTimecode, cfg.Label, cleanText,
		))
	}

	return sb.String()
}

func formatASSTime(sec float64) string {
	if sec < 0 {
		sec = 0
	}
	h := int(sec) / 3600
	m := (int(sec) % 3600) / 60
	s := int(sec) % 60
	cs := int((sec - float64(int(sec))) * 100)
	return fmt.Sprintf("%d:%02d:%02d.%02d", h, m, s, cs)
}

func hexToASSColor(hexStr string, alphaHex string) string {
	hexStr = strings.TrimPrefix(hexStr, "#")
	if len(hexStr) == 3 {
		hexStr = fmt.Sprintf("%c%c%c%c%c%c", hexStr[0], hexStr[0], hexStr[1], hexStr[1], hexStr[2], hexStr[2])
	}
	if len(hexStr) != 6 && len(hexStr) != 8 {
		return "&H00FFFFFF"
	}
	r := hexStr[0:2]
	g := hexStr[2:4]
	b := hexStr[4:6]
	a := alphaHex
	if len(hexStr) == 8 {
		a = hexStr[6:8]
	}
	// ASS color order: &HAABBGGRR
	return fmt.Sprintf("&H%s%s%s%s", strings.ToUpper(a), strings.ToUpper(b), strings.ToUpper(g), strings.ToUpper(r))
}
