package reframe

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"regexp"
	"strings"
	"sync"

	pigo "github.com/esimov/pigo/core"
	"parallax/internal/llm"
)

//go:embed cascade/facefinder
var cascadeBytes []byte

var (
	classifierOnce sync.Once
	classifier     *pigo.Pigo
	classifierErr  error
)

func getClassifier() (*pigo.Pigo, error) {
	classifierOnce.Do(func() {
		if len(cascadeBytes) == 0 {
			classifierErr = fmt.Errorf("cascade classifier binary is empty")
			return
		}
		p := pigo.NewPigo()
		unpacked, err := p.Unpack(cascadeBytes)
		if err != nil {
			classifierErr = fmt.Errorf("unpack pigo cascade: %w", err)
			return
		}
		classifier = unpacked
	})
	return classifier, classifierErr
}

// BoundingBox is a normalized [0, 1] rectangle.
type BoundingBox struct {
	XMin float64 `json:"xmin"`
	YMin float64 `json:"ymin"`
	XMax float64 `json:"xmax"`
	YMax float64 `json:"ymax"`
}

// Center returns the normalized center coordinate (0..1, 0..1).
func (b BoundingBox) Center() (float64, float64) {
	return (b.XMin + b.XMax) / 2.0, (b.YMin + b.YMax) / 2.0
}

// Width and Height return normalized dimensions.
func (b BoundingBox) Width() float64  { return math.Max(0, b.XMax-b.XMin) }
func (b BoundingBox) Height() float64 { return math.Max(0, b.YMax-b.YMin) }

// SubjectDetection represents a detected focal area with source detector.
type SubjectDetection struct {
	Box        BoundingBox `json:"box"`
	Confidence float64     `json:"confidence"`
	Source     string      `json:"source"` // "pigo" | "vision_llm" | "center_fallback"
}

// DetectSubject detects the focal subject of a frame image.
// It tries Pigo face detection first, then falls back to Vision-LLM, then center-frame.
func DetectSubject(ctx context.Context, frameAbsPath string, visionClient llm.ChatProvider) SubjectDetection {
	// 1. Primary: Try Pigo face detection
	if box, ok := detectFacesWithPigo(frameAbsPath); ok {
		return SubjectDetection{
			Box:        box,
			Confidence: 0.95,
			Source:     "pigo",
		}
	}

	// 2. Secondary: Fall back to Vision-LLM bounding box
	if visionClient != nil {
		if box, ok := detectSubjectWithVisionLLM(ctx, frameAbsPath, visionClient); ok {
			return SubjectDetection{
				Box:        box,
				Confidence: 0.85,
				Source:     "vision_llm",
			}
		}
	}

	// 3. Fallback: Default to center region of the frame
	return SubjectDetection{
		Box: BoundingBox{
			XMin: 0.35,
			YMin: 0.20,
			XMax: 0.65,
			YMax: 0.80,
		},
		Confidence: 0.50,
		Source:     "center_fallback",
	}
}

// detectFacesWithPigo runs the embedded Pigo classifier over a local image.
func detectFacesWithPigo(absPath string) (BoundingBox, bool) {
	cls, err := getClassifier()
	if err != nil {
		return BoundingBox{}, false
	}

	file, err := os.Open(absPath)
	if err != nil {
		return BoundingBox{}, false
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return BoundingBox{}, false
	}

	bounds := img.Bounds()
	cols, rows := bounds.Dx(), bounds.Dy()
	if cols <= 0 || rows <= 0 {
		return BoundingBox{}, false
	}

	// Convert image to grayscale pixel intensity buffer
	pixels := make([]uint8, cols*rows)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			// Rec. 601 luma formula
			gray := (299*r + 587*g + 114*b) >> 16
			pixels[y*cols+x] = uint8(gray)
		}
	}

	cParams := pigo.CascadeParams{
		MinSize:     int(float64(cols) * 0.05), // min 5% of width
		MaxSize:     int(float64(cols) * 0.90), // max 90% of width
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,
		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	dets := cls.RunCascade(cParams, 0.0)
	dets = cls.ClusterDetections(dets, 0.2)

	if len(dets) == 0 {
		return BoundingBox{}, false
	}

	// Find the largest / highest confidence face
	best := dets[0]
	for _, d := range dets {
		if d.Scale > best.Scale || (d.Scale == best.Scale && d.Q > best.Q) {
			best = d
		}
	}

	// Pigo returns (row, col, scale). Convert to normalized bounding box with head/torso margins
	faceW := float64(best.Scale) / float64(cols)
	faceH := float64(best.Scale) / float64(rows)
	faceCX := float64(best.Col) / float64(cols)
	faceCY := float64(best.Row) / float64(rows)

	// Expand to include hair and shoulders
	box := BoundingBox{
		XMin: math.Max(0.0, faceCX-faceW*0.65),
		XMax: math.Min(1.0, faceCX+faceW*0.65),
		YMin: math.Max(0.0, faceCY-faceH*0.75),
		YMax: math.Min(1.0, faceCY+faceH*1.25),
	}

	return box, true
}

// detectSubjectWithVisionLLM requests subject bounding box coordinates from the Vision LLM.
func detectSubjectWithVisionLLM(ctx context.Context, absPath string, client llm.ChatProvider) (BoundingBox, bool) {
	data, err := os.ReadFile(absPath)
	if err != nil {
		return BoundingBox{}, false
	}

	ext := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(absPath[strings.LastIndex(absPath, "."):]), "."))
	mime := "image/jpeg"
	if ext == "png" {
		mime = "image/png"
	}

	b64 := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
	const prompt = "Identify the primary focal subject (person, product, vehicle, animal, or main focal object) in this video frame. Return its normalized bounding box as JSON in this exact structure: {\"ymin\": 0.15, \"xmin\": 0.35, \"ymax\": 0.85, \"xmax\": 0.65}. All coordinates must be decimal numbers between 0.0 and 1.0. Do not include markdown formatting or extra text."

	contentJSON := fmt.Sprintf(`[{"type":"text","text":%q},{"type":"image_url","image_url":{"url":%q,"detail":"low"}}]`, prompt, b64)
	req := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: contentJSON},
		},
		Temperature: llm.Ptr(0.0),
	}

	ch, err := client.Stream(ctx, req)
	if err != nil {
		return BoundingBox{}, false
	}

	var sb strings.Builder
	for d := range ch {
		if d.Err != nil {
			return BoundingBox{}, false
		}
		sb.WriteString(d.Content)
	}

	raw := strings.TrimSpace(sb.String())
	return parseBoundingBoxJSON(raw)
}

func parseBoundingBoxJSON(raw string) (BoundingBox, bool) {
	// Try standard JSON unmarshal
	var structResult struct {
		YMin float64 `json:"ymin"`
		XMin float64 `json:"xmin"`
		YMax float64 `json:"ymax"`
		XMax float64 `json:"xmax"`
	}

	clean := strings.TrimPrefix(raw, "```json")
	clean = strings.TrimPrefix(clean, "```")
	clean = strings.TrimSuffix(clean, "```")
	clean = strings.TrimSpace(clean)

	if err := json.Unmarshal([]byte(clean), &structResult); err == nil {
		if structResult.XMax > structResult.XMin && structResult.YMax > structResult.YMin {
			return BoundingBox{
				XMin: clamp(structResult.XMin, 0.0, 1.0),
				YMin: clamp(structResult.YMin, 0.0, 1.0),
				XMax: clamp(structResult.XMax, 0.0, 1.0),
				YMax: clamp(structResult.YMax, 0.0, 1.0),
			}, true
		}
	}

	// Fallback: regex search for 4 floats
	re := regexp.MustCompile(`"?(?:xmin|ymin|xmax|ymax)"?\s*:\s*([\d\.]+)`)
	matches := re.FindAllStringSubmatch(clean, -1)
	if len(matches) >= 4 {
		var vals [4]float64
		for i := 0; i < 4; i++ {
			var v float64
			fmt.Sscanf(matches[i][1], "%f", &v)
			vals[i] = v
		}
		return BoundingBox{
			YMin: clamp(vals[0], 0, 1),
			XMin: clamp(vals[1], 0, 1),
			YMax: clamp(vals[2], 0, 1),
			XMax: clamp(vals[3], 0, 1),
		}, true
	}

	return BoundingBox{}, false
}

func clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
