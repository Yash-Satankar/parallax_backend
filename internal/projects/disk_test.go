package projects

import (
	"errors"
	"testing"
)

func TestFreeBytes(t *testing.T) {
	bytes, err := FreeBytes(".")
	if err != nil {
		t.Fatalf("FreeBytes failed: %v", err)
	}
	if bytes <= 0 {
		t.Fatalf("expected FreeBytes > 0, got %d", bytes)
	}
}

func TestEnsureDiskSpace(t *testing.T) {
	if err := EnsureDiskSpace(".", 1024); err != nil {
		t.Fatalf("EnsureDiskSpace failed for small allocation: %v", err)
	}
	if err := EnsureDiskSpace("", 1024); err != nil {
		t.Fatalf("EnsureDiskSpace should succeed for empty path: %v", err)
	}
	if err := EnsureDiskSpace(".", -100); err != nil {
		t.Fatalf("EnsureDiskSpace should succeed for negative bytes: %v", err)
	}
}

func TestIsNoSpace(t *testing.T) {
	if IsNoSpace(nil) {
		t.Fatalf("expected IsNoSpace(nil) to be false")
	}
	if !IsNoSpace(errors.New("no space left on device")) {
		t.Fatalf("expected string match for 'no space left'")
	}
	if !IsNoSpace(errors.New("not enough disk space (10 MB free, need about 100 MB)")) {
		t.Fatalf("expected string match for 'not enough disk space'")
	}
}
