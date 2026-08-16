package collab_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"parallax/internal/collab"
	"parallax/internal/projects"
)

func TestFractionalIndexing(t *testing.T) {
	// Test generating key between empty bounds
	k1 := collab.KeyBetween("", "")
	if k1 == "" {
		t.Fatalf("expected non-empty key")
	}

	// Test generating key before k1
	k0 := collab.KeyBetween("", k1)
	if k0 >= k1 {
		t.Fatalf("expected k0 (%q) < k1 (%q)", k0, k1)
	}

	// Test generating key after k1
	k2 := collab.KeyBetween(k1, "")
	if k1 >= k2 {
		t.Fatalf("expected k1 (%q) < k2 (%q)", k1, k2)
	}

	// Test generating key between k1 and k2
	kMid := collab.KeyBetween(k1, k2)
	if !(k1 < kMid && kMid < k2) {
		t.Fatalf("expected k1 (%q) < kMid (%q) < k2 (%q)", k1, kMid, k2)
	}

	// Test initial keys are unique and strictly sorted
	keys := collab.InitialKeys(5)
	if len(keys) != 5 {
		t.Fatalf("expected 5 keys, got %d", len(keys))
	}
	for i := 0; i < len(keys)-1; i++ {
		if keys[i] >= keys[i+1] {
			t.Fatalf("keys[%d]=%q should be < keys[%d]=%q", i, keys[i], i+1, keys[i+1])
		}
	}
}

func TestLWWTimestamp(t *testing.T) {
	ts1 := collab.Timestamp{Seq: 1, ClientID: "alice"}
	ts2 := collab.Timestamp{Seq: 2, ClientID: "bob"}
	ts3 := collab.Timestamp{Seq: 2, ClientID: "charlie"}

	if !ts2.After(ts1) {
		t.Fatalf("expected ts2 > ts1 based on seq")
	}
	if !ts3.After(ts2) {
		t.Fatalf("expected ts3 > ts2 based on client_id tiebreaker")
	}
	if ts1.After(ts2) {
		t.Fatalf("expected ts1 not after ts2")
	}

	// Test LWWMap Set and Get
	m := make(collab.LWWMap)
	changed := m.Set("start_frame", 100, ts1)
	if !changed {
		t.Fatalf("expected first write to succeed")
	}

	// Earlier timestamp should not overwrite
	changed = m.Set("start_frame", 50, collab.Timestamp{Seq: 0, ClientID: "old"})
	if changed {
		t.Fatalf("earlier timestamp should not overwrite")
	}
	val, ok := m.Get("start_frame")
	if !ok || val != 100 {
		t.Fatalf("expected value to remain 100, got %v", val)
	}

	// Later timestamp should overwrite
	changed = m.Set("start_frame", 200, ts2)
	if !changed {
		t.Fatalf("later timestamp should overwrite")
	}
	val, ok = m.Get("start_frame")
	if !ok || val != 200 {
		t.Fatalf("expected value to be 200, got %v", val)
	}
}

func TestHubWebSocketSyncAndBroadcast(t *testing.T) {
	tmpDir := t.TempDir()
	store, err := projects.NewStore(tmpDir)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	p, err := store.Create("Test Collab Project")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	hub := collab.NewHub(store, nil)

	// Create test HTTP server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hub.ServeWS(w, r, p.ID)
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	// Connect Client 1
	c1, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Dial c1: %v", err)
	}
	defer c1.Close()

	// Client 1 should receive timeline_sync
	var syncMsg collab.Msg
	if err := c1.ReadJSON(&syncMsg); err != nil {
		t.Fatalf("c1 ReadJSON sync: %v", err)
	}
	if syncMsg.Type != collab.MsgTypeSync {
		t.Fatalf("expected %q, got %q", collab.MsgTypeSync, syncMsg.Type)
	}

	// Connect Client 2
	c2, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Dial c2: %v", err)
	}
	defer c2.Close()

	// Client 2 should receive timeline_sync
	if err := c2.ReadJSON(&syncMsg); err != nil {
		t.Fatalf("c2 ReadJSON sync: %v", err)
	}

	// Client 1 sends clip_insert
	clip := collab.LiveClip{
		TimelineClip: projects.TimelineClip{
			ID:             "clip-test-1",
			Name:           "Test Clip",
			Track:          "V1",
			Kind:           "video",
			StartFrame:     0,
			DurationFrames: 120,
			MediaPath:      "media/test.mp4",
			MediaType:      "video",
		},
		Rank: "na",
	}
	insertPayload, _ := json.Marshal(collab.ClipInsertPayload{Clip: clip})
	insertMsg := collab.Msg{Type: collab.MsgTypeClipInsert, Payload: insertPayload}
	if err := c1.WriteJSON(insertMsg); err != nil {
		t.Fatalf("c1 WriteJSON insert: %v", err)
	}

	// Client 2 should receive the broadcast
	_ = c2.SetReadDeadline(time.Now().Add(2 * time.Second))
	var receivedMsg collab.Msg
	if err := c2.ReadJSON(&receivedMsg); err != nil {
		t.Fatalf("c2 ReadJSON broadcast: %v", err)
	}
	if receivedMsg.Type != collab.MsgTypeClipInsert {
		t.Fatalf("expected clip_insert broadcast, got %q", receivedMsg.Type)
	}
}
