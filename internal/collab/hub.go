package collab

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"parallax/internal/projects"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin:     func(*http.Request) bool { return true }, // CORS handled upstream
}

// Hub manages one WebSocket room per project.
type Hub struct {
	mu     sync.RWMutex
	rooms  map[string]*Room
	store  *projects.Store
	logger *slog.Logger

	// Global monotonic server sequence — all op timestamps share this clock.
	seq atomic.Int64
}

// NewHub creates a Hub backed by the given project store.
func NewHub(store *projects.Store, logger *slog.Logger) *Hub {
	if logger == nil {
		logger = slog.Default()
	}
	return &Hub{
		rooms:  make(map[string]*Room),
		store:  store,
		logger: logger,
	}
}

func (h *Hub) nextSeq() int64 { return h.seq.Add(1) }

// room returns (creating if necessary) the Room for a project.
func (h *Hub) room(projectID string) *Room {
	h.mu.Lock()
	defer h.mu.Unlock()
	r, ok := h.rooms[projectID]
	if !ok {
		r = newRoom(projectID, h)
		h.rooms[projectID] = r
		go r.run()
	}
	return r
}

// dropRoom removes an empty room.
func (h *Hub) dropRoom(projectID string) {
	h.mu.Lock()
	delete(h.rooms, projectID)
	h.mu.Unlock()
}

// ServeWS upgrades the HTTP connection to WebSocket for a project room.
func (h *Hub) ServeWS(w http.ResponseWriter, r *http.Request, projectID string) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.Warn("collab: ws upgrade failed", "project", projectID, "err", err)
		return
	}

	clientID := newClientID()
	c := &Client{
		hub:      h,
		room:     h.room(projectID),
		clientID: clientID,
		conn:     conn,
		send:     make(chan []byte, 64),
	}
	c.room.join <- c

	go c.writePump()
	go c.readPump()
}

// BroadcastAgentOp lets Director tool handlers push ops into the live room.
// If no room exists for the project, the call is a no-op (no clients connected).
func (h *Hub) BroadcastAgentOp(projectID string, msgType MsgType, payload any) {
	h.mu.RLock()
	r, ok := h.rooms[projectID]
	h.mu.RUnlock()
	if !ok {
		return
	}

	b, err := NewMsg(msgType, payload)
	if err != nil {
		h.logger.Warn("collab: marshal agent op", "err", err)
		return
	}
	r.broadcast <- broadcastMsg{data: b, exclude: ""}
}

// FlushToRevision serialises the room's live state into a new immutable revision.
// Called by the checkpoint tool and on room teardown.
func (h *Hub) FlushToRevision(projectID string, meta projects.CommitMeta) {
	h.mu.RLock()
	r, ok := h.rooms[projectID]
	h.mu.RUnlock()
	if !ok {
		return
	}
	r.flush(meta)
}

// -----------------------------------------------------------------------
// Room
// -----------------------------------------------------------------------

type broadcastMsg struct {
	data    []byte
	exclude string // clientID to skip (the sender)
}

// Room is a live collaborative session for one project.
type Room struct {
	projectID string
	hub       *Hub

	join      chan *Client
	leave     chan *Client
	broadcast chan broadcastMsg

	mu       sync.RWMutex
	clients  map[string]*Client  // clientID → client
	presence map[string]PresenceState

	// Live resolved state (post-LWW, post-fractional-index)
	clips       map[string]*LiveClip // clipID → current state
	lwwFields   map[string]LWWMap   // clipID → field → LWWField
	canvas      projects.TimelineCanvas
	transitions []projects.TimelineTransition
	fps         int

	// Opt-in synchronized playback: tracks which client is the current leader.
	// Empty string means no synchronized playback is active.
	playbackLeader string

	// Flush debounce
	debounceMu    sync.Mutex
	debounceTimer *time.Timer
}

func newRoom(projectID string, hub *Hub) *Room {
	return &Room{
		projectID: projectID,
		hub:       hub,
		join:      make(chan *Client, 8),
		leave:     make(chan *Client, 8),
		broadcast: make(chan broadcastMsg, 64),
		clients:   make(map[string]*Client),
		presence:  make(map[string]PresenceState),
		clips:     make(map[string]*LiveClip),
		lwwFields: make(map[string]LWWMap),
		fps:       24,
	}
}

func (r *Room) run() {
	defer func() {
		if rec := recover(); rec != nil {
			r.hub.logger.Error("collab: room panic recovered", "project", r.projectID, "err", rec)
		}
	}()

	// Load initial state from persistent store.
	r.loadFromStore()

	for {
		select {
		case c := <-r.join:
			r.mu.Lock()
			r.clients[c.clientID] = c
			r.mu.Unlock()
			r.sendSync(c)
			r.hub.logger.Info("collab: client joined", "project", r.projectID, "client", c.clientID)

		case c := <-r.leave:
			r.mu.Lock()
			delete(r.clients, c.clientID)
			delete(r.presence, c.clientID)
			empty := len(r.clients) == 0
			r.mu.Unlock()

			leave, _ := NewMsg(MsgTypePresenceLeave, PresenceLeavePayload{ClientID: c.clientID})
			r.broadcastToAll(leave, c.clientID)
			r.hub.logger.Info("collab: client left", "project", r.projectID, "client", c.clientID)

			if empty {
				r.flush(projects.CommitMeta{Actor: "system", Summary: "Auto-save collaborative session"})
				r.hub.dropRoom(r.projectID)
				return
			}

		case msg := <-r.broadcast:
			r.broadcastToAll(msg.data, msg.exclude)
		}
	}
}

func (r *Room) broadcastToAll(data []byte, exclude string) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for id, c := range r.clients {
		if id == exclude {
			continue
		}
		select {
		case c.send <- data:
		default:
			// Slow client — drop the message rather than blocking.
		}
	}
}

// sendSync delivers the full current state to a freshly-joined client.
func (r *Room) sendSync(c *Client) {
	r.mu.RLock()
	clips := r.sortedClips()
	pres := r.presenceList()
	canvas := r.canvas
	trans := append([]projects.TimelineTransition(nil), r.transitions...)
	fps := r.fps
	r.mu.RUnlock()

	payload := SyncPayload{
		ProjectID:   r.projectID,
		Clips:       clips,
		Canvas:      canvas,
		Transitions: trans,
		FPS:         fps,
		Presence:    pres,
		ServerSeq:   r.hub.seq.Load(),
	}
	b, err := NewMsg(MsgTypeSync, payload)
	if err != nil {
		return
	}
	select {
	case c.send <- b:
	default:
	}
}

// sortedClips returns clips sorted by fractional Rank then ID.
func (r *Room) sortedClips() []LiveClip {
	out := make([]LiveClip, 0, len(r.clips))
	for _, c := range r.clips {
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Rank != out[j].Rank {
			return out[i].Rank < out[j].Rank
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func (r *Room) presenceList() []PresenceState {
	out := make([]PresenceState, 0, len(r.presence))
	for _, p := range r.presence {
		out = append(out, p)
	}
	return out
}

// -----------------------------------------------------------------------
// Incoming message dispatch
// -----------------------------------------------------------------------

func (r *Room) handleMsg(c *Client, raw []byte) {
	var msg Msg
	if err := json.Unmarshal(raw, &msg); err != nil {
		return
	}

	ts := Timestamp{Seq: r.hub.nextSeq(), ClientID: c.clientID}

	switch msg.Type {
	case MsgTypeClipInsert:
		var p ClipInsertPayload
		if err := json.Unmarshal(msg.Payload, &p); err != nil {
			return
		}
		p.Timestamp = ts
		r.applyClipInsert(p)
		b, _ := NewMsg(MsgTypeClipInsert, p)
		r.broadcast <- broadcastMsg{data: b, exclude: c.clientID}
		r.scheduleDebouncedFlush()

	case MsgTypeClipDelete:
		var p ClipDeletePayload
		if err := json.Unmarshal(msg.Payload, &p); err != nil {
			return
		}
		p.Timestamp = ts
		r.applyClipDelete(p)
		b, _ := NewMsg(MsgTypeClipDelete, p)
		r.broadcast <- broadcastMsg{data: b, exclude: c.clientID}
		r.scheduleDebouncedFlush()

	case MsgTypeClipFieldUpdate:
		var p ClipFieldUpdatePayload
		if err := json.Unmarshal(msg.Payload, &p); err != nil {
			return
		}
		p.Timestamp = ts
		changed := r.applyFieldUpdate(p)
		if changed {
			b, _ := NewMsg(MsgTypeClipFieldUpdate, p)
			r.broadcast <- broadcastMsg{data: b, exclude: c.clientID}
			r.scheduleDebouncedFlush()
		}

	case MsgTypeClipReorder:
		var p ClipReorderPayload
		if err := json.Unmarshal(msg.Payload, &p); err != nil {
			return
		}
		p.Timestamp = ts
		r.applyClipReorder(p)
		b, _ := NewMsg(MsgTypeClipReorder, p)
		r.broadcast <- broadcastMsg{data: b, exclude: c.clientID}
		r.scheduleDebouncedFlush()

	case MsgTypePresenceUpdate:
		var p PresenceState
		if err := json.Unmarshal(msg.Payload, &p); err != nil {
			return
		}
		p.ClientID = c.clientID
		r.mu.Lock()
		r.presence[c.clientID] = p
		r.mu.Unlock()
		b, _ := NewMsg(MsgTypePresenceUpdate, p)
		r.broadcast <- broadcastMsg{data: b, exclude: c.clientID}

	case MsgTypePlaybackSync:
		// Pure relay: inject server wall time so receivers can correct for latency.
		var p PlaybackSyncPayload
		if err := json.Unmarshal(msg.Payload, &p); err != nil {
			return
		}
		p.ClientID = c.clientID
		p.ServerTimeMs = time.Now().UnixMilli()
		// Record this client as the playback leader.
		r.mu.Lock()
		r.playbackLeader = c.clientID
		r.mu.Unlock()
		b, _ := NewMsg(MsgTypePlaybackSync, p)
		r.broadcast <- broadcastMsg{data: b, exclude: c.clientID}

	case MsgTypeFrameChunk:
		// Phase 5 — pure relay of EncodedVideoChunk data between peers.
		// Server never decodes or inspects the chunk payload.
		r.broadcast <- broadcastMsg{data: raw, exclude: c.clientID}
	}
}

// -----------------------------------------------------------------------
// State mutations
// -----------------------------------------------------------------------

func (r *Room) applyClipInsert(p ClipInsertPayload) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.clips[p.Clip.ID]; exists {
		return // idempotent
	}
	clip := p.Clip
	clip.FieldSeq = p.Timestamp.Seq
	r.clips[clip.ID] = &clip
	if r.lwwFields[clip.ID] == nil {
		r.lwwFields[clip.ID] = make(LWWMap)
	}
}

func (r *Room) applyClipDelete(p ClipDeletePayload) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.clips, p.ID)
	delete(r.lwwFields, p.ID)
}

func (r *Room) applyFieldUpdate(p ClipFieldUpdatePayload) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	clip, ok := r.clips[p.ClipID]
	if !ok {
		return false
	}
	if r.lwwFields[p.ClipID] == nil {
		r.lwwFields[p.ClipID] = make(LWWMap)
	}
	changed := false
	for field, value := range p.Fields {
		if r.lwwFields[p.ClipID].Set(field, value, p.Timestamp) {
			applyFieldToClip(&clip.TimelineClip, field, value)
			changed = true
		}
	}
	if changed && p.Timestamp.Seq > clip.FieldSeq {
		clip.FieldSeq = p.Timestamp.Seq
	}
	return changed
}

func (r *Room) applyClipReorder(p ClipReorderPayload) {
	r.mu.Lock()
	defer r.mu.Unlock()
	clip, ok := r.clips[p.ID]
	if !ok {
		return
	}
	clip.Rank = p.Rank
}

// applyFieldToClip maps a string field path to the concrete TimelineClip field.
// Only the most-commonly-edited fields are handled; unknown paths are silently ignored.
func applyFieldToClip(c *projects.TimelineClip, field string, value any) {
	asFloat := func() float64 {
		switch v := value.(type) {
		case float64:
			return v
		case int:
			return float64(v)
		case int64:
			return float64(v)
		}
		return 0
	}
	asInt := func() int { return int(asFloat()) }
	asBool := func() bool {
		if b, ok := value.(bool); ok {
			return b
		}
		return false
	}
	asString := func() string {
		if s, ok := value.(string); ok {
			return s
		}
		return ""
	}

	switch field {
	case "start_frame":
		c.StartFrame = asInt()
	case "duration_frames":
		c.DurationFrames = asInt()
	case "source_in_frame":
		c.SourceInFrame = asInt()
	case "name":
		c.Name = asString()
	case "enabled":
		b := asBool()
		c.Enabled = &b
	// Audio
	case "audio.volume_db":
		if c.Audio == nil {
			c.Audio = &projects.TimelineAudio{}
		}
		c.Audio.VolumeDB = asFloat()
	case "audio.muted":
		if c.Audio == nil {
			c.Audio = &projects.TimelineAudio{}
		}
		c.Audio.Muted = asBool()
	case "audio.pan":
		if c.Audio == nil {
			c.Audio = &projects.TimelineAudio{}
		}
		c.Audio.Pan = asFloat()
	// Grade
	case "grade.exposure":
		if c.Grade == nil {
			c.Grade = &projects.TimelineColor{}
		}
		c.Grade.Exposure = asFloat()
	case "grade.contrast":
		if c.Grade == nil {
			c.Grade = &projects.TimelineColor{}
		}
		c.Grade.Contrast = asFloat()
	case "grade.saturation":
		if c.Grade == nil {
			c.Grade = &projects.TimelineColor{}
		}
		c.Grade.Saturation = asFloat()
	case "grade.temperature":
		if c.Grade == nil {
			c.Grade = &projects.TimelineColor{}
		}
		c.Grade.Temperature = asFloat()
	case "grade.tint":
		if c.Grade == nil {
			c.Grade = &projects.TimelineColor{}
		}
		c.Grade.Tint = asFloat()
	// Transform
	case "transform.x":
		if c.Transform == nil {
			c.Transform = &projects.TimelineTransform{ScaleX: 1, ScaleY: 1, Opacity: 1}
		}
		c.Transform.X = asFloat()
	case "transform.y":
		if c.Transform == nil {
			c.Transform = &projects.TimelineTransform{ScaleX: 1, ScaleY: 1, Opacity: 1}
		}
		c.Transform.Y = asFloat()
	case "transform.opacity":
		if c.Transform == nil {
			c.Transform = &projects.TimelineTransform{ScaleX: 1, ScaleY: 1, Opacity: 1}
		}
		c.Transform.Opacity = asFloat()
	case "transform.scale_x":
		if c.Transform == nil {
			c.Transform = &projects.TimelineTransform{ScaleX: 1, ScaleY: 1, Opacity: 1}
		}
		c.Transform.ScaleX = asFloat()
	case "transform.scale_y":
		if c.Transform == nil {
			c.Transform = &projects.TimelineTransform{ScaleX: 1, ScaleY: 1, Opacity: 1}
		}
		c.Transform.ScaleY = asFloat()
	// Title
	case "title.text":
		if c.Title == nil {
			c.Title = &projects.TimelineTitle{}
		}
		c.Title.Text = asString()
	case "title.font_size":
		if c.Title == nil {
			c.Title = &projects.TimelineTitle{}
		}
		c.Title.FontSize = asFloat()
	}
}

// -----------------------------------------------------------------------
// Load from / flush to persistent store
// -----------------------------------------------------------------------

func (r *Room) loadFromStore() {
	if r.hub.store == nil {
		return
	}
	tl, err := r.hub.store.GetTimeline(r.projectID)
	if err != nil {
		r.hub.logger.Warn("collab: load timeline", "project", r.projectID, "err", err)
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	r.canvas = tl.Canvas
	r.transitions = tl.Transitions
	r.fps = tl.FPS

	// Assign initial fractional ranks if not present.
	keys := InitialKeys(len(tl.Clips))
	for i, clip := range tl.Clips {
		lc := &LiveClip{TimelineClip: clip}
		if i < len(keys) {
			lc.Rank = keys[i]
		} else {
			lc.Rank = "n"
		}
		r.clips[clip.ID] = lc
		r.lwwFields[clip.ID] = make(LWWMap)
	}
}

// flush serialises the live state into the project's revision history.
func (r *Room) flush(meta projects.CommitMeta) {
	if r.hub.store == nil {
		return
	}
	r.mu.RLock()
	sorted := r.sortedClips()
	canvas := r.canvas
	trans := append([]projects.TimelineTransition(nil), r.transitions...)
	fps := r.fps
	r.mu.RUnlock()

	clips := make([]projects.TimelineClip, len(sorted))
	for i, lc := range sorted {
		clips[i] = lc.TimelineClip
	}

	tl := projects.Timeline{
		Schema:    2,
		FPS:       fps,
		Canvas:    canvas,
		Clips:     clips,
		Transitions: trans,
	}

	if _, err := r.hub.store.SaveTimeline(r.projectID, tl); err != nil {
		r.hub.logger.Warn("collab: flush timeline", "project", r.projectID, "err", err)
	}
}

// scheduleDebouncedFlush auto-saves the live state 3 s after the last op.
func (r *Room) scheduleDebouncedFlush() {
	r.debounceMu.Lock()
	defer r.debounceMu.Unlock()
	if r.debounceTimer != nil {
		r.debounceTimer.Reset(3 * time.Second)
		return
	}
	r.debounceTimer = time.AfterFunc(3*time.Second, func() {
		r.flush(projects.CommitMeta{Actor: "system", Summary: "Auto-save collaborative edits"})
		r.debounceMu.Lock()
		r.debounceTimer = nil
		r.debounceMu.Unlock()
	})
}

// -----------------------------------------------------------------------
// PublishClipFromAgent sends an agent-originated clip insert to all clients.
// -----------------------------------------------------------------------

func (h *Hub) PublishClipInsert(projectID string, clip projects.TimelineClip, rank string) {
	ts := Timestamp{Seq: h.nextSeq(), ClientID: "agent"}
	lc := LiveClip{TimelineClip: clip, Rank: rank}

	h.mu.RLock()
	r, ok := h.rooms[projectID]
	h.mu.RUnlock()
	if ok {
		r.applyClipInsert(ClipInsertPayload{Clip: lc, ClientID: "agent", Timestamp: ts})
	}

	h.BroadcastAgentOp(projectID, MsgTypeClipInsert, ClipInsertPayload{
		Clip: lc, ClientID: "agent", Timestamp: ts,
	})
}

func (h *Hub) PublishClipDelete(projectID, clipID string) {
	ts := Timestamp{Seq: h.nextSeq(), ClientID: "agent"}
	h.mu.RLock()
	r, ok := h.rooms[projectID]
	h.mu.RUnlock()
	if ok {
		r.applyClipDelete(ClipDeletePayload{ID: clipID, ClientID: "agent", Timestamp: ts})
	}
	h.BroadcastAgentOp(projectID, MsgTypeClipDelete, ClipDeletePayload{
		ID: clipID, ClientID: "agent", Timestamp: ts,
	})
}

func (h *Hub) PublishFieldUpdate(projectID, clipID string, fields map[string]any) {
	ts := Timestamp{Seq: h.nextSeq(), ClientID: "agent"}
	p := ClipFieldUpdatePayload{ClipID: clipID, Fields: fields, ClientID: "agent", Timestamp: ts}
	h.mu.RLock()
	r, ok := h.rooms[projectID]
	h.mu.RUnlock()
	if ok {
		r.applyFieldUpdate(p)
	}
	h.BroadcastAgentOp(projectID, MsgTypeClipFieldUpdate, p)
}

// -----------------------------------------------------------------------
// Client
// -----------------------------------------------------------------------

const (
	writeWait      = 10 * time.Second
	pongWait       = 60 * time.Second
	pingPeriod     = (pongWait * 9) / 10
	// Raised from 64 KB to 256 KB to support Phase 5 frame_chunk relay.
	// EncodedVideoChunks (base64-encoded) can approach 80–100 KB for I-frames.
	maxMessageSize = 256 * 1024
)

// Client is one WebSocket connection.
type Client struct {
	hub      *Hub
	room     *Room
	clientID string
	conn     *websocket.Conn
	send     chan []byte
}

func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		if rec := recover(); rec != nil {
			c.hub.logger.Error("collab: client writePump panic", "client", c.clientID, "err", rec)
		}
		ticker.Stop()
		c.conn.Close()
	}()
	for {
		select {
		case msg, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (c *Client) readPump() {
	defer func() {
		if rec := recover(); rec != nil {
			c.hub.logger.Error("collab: client readPump panic", "client", c.clientID, "err", rec)
		}
		c.room.leave <- c
		c.conn.Close()
	}()
	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})
	for {
		_, msg, err := c.conn.ReadMessage()
		if err != nil {
			break
		}
		c.room.handleMsg(c, msg)
	}
}

func newClientID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "c-" + hex.EncodeToString(b[:])
}
