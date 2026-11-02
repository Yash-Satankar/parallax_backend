package tests

import (
  "testing"

  "parallax/internal/reframe"
)

func TestAspectRatioAndCrop(t *testing.T) {
  r := reframe.NormalizeAspectRatio("9:16")
  if r.Name != "9:16" {
    t.Fatalf("expected 9:16, got %s", r.Name)
  }

  // Subject centered, expect crop values within 0..1
  sub := reframe.BoundingBox{XMin: 0.4, YMin: 0.2, XMax: 0.6, YMax: 0.8}
  crop := reframe.CalculateCrop(1920, 1080, r, sub)
  if crop.CropLeft < 0 || crop.CropLeft > 1 {
    t.Fatalf("crop left out of range: %v", crop.CropLeft)
  }
}
