package tools

import (
	"testing"
)

func TestParseSilenceDetectOutput(t *testing.T) {
	sampleStderr := `
[silencedetect @ 0000021c3b1e3800] silence_start: 1.25
[silencedetect @ 0000021c3b1e3800] silence_end: 3.40 | silence_duration: 2.15
[silencedetect @ 0000021c3b1e3800] silence_start: 8.10
[silencedetect @ 0000021c3b1e3800] silence_end: 10.50 | silence_duration: 2.40
`
	intervals := parseSilenceDetectOutput(sampleStderr)
	if len(intervals) != 2 {
		t.Fatalf("expected 2 silence intervals, got %d", len(intervals))
	}
	if intervals[0].Start != 1.25 || intervals[0].End != 3.40 {
		t.Errorf("expected [1.25, 3.40], got [%f, %f]", intervals[0].Start, intervals[0].End)
	}
	if intervals[1].Start != 8.10 || intervals[1].End != 10.50 {
		t.Errorf("expected [8.10, 10.50], got [%f, %f]", intervals[1].Start, intervals[1].End)
	}
}

func TestInvertIntervals(t *testing.T) {
	silences := []timeInterval{
		{Start: 2.0, End: 4.0},
		{Start: 7.0, End: 9.0},
	}
	totalDuration := 12.0

	keeps := invertIntervals(silences, totalDuration)
	if len(keeps) != 3 {
		t.Fatalf("expected 3 keep intervals, got %d", len(keeps))
	}
	// Segment 1: [0.0, 2.0]
	if keeps[0].Start != 0.0 || keeps[0].End != 2.0 {
		t.Errorf("segment 0: expected [0.0, 2.0], got [%f, %f]", keeps[0].Start, keeps[0].End)
	}
	// Segment 2: [4.0, 7.0]
	if keeps[1].Start != 4.0 || keeps[1].End != 7.0 {
		t.Errorf("segment 1: expected [4.0, 7.0], got [%f, %f]", keeps[1].Start, keeps[1].End)
	}
	// Segment 3: [9.0, 12.0]
	if keeps[2].Start != 9.0 || keeps[2].End != 12.0 {
		t.Errorf("segment 2: expected [9.0, 12.0], got [%f, %f]", keeps[2].Start, keeps[2].End)
	}
}
