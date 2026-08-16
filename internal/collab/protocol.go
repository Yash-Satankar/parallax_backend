package collab

import (
	"encoding/json"
	"parallax/internal/projects"
)

// MsgType identifies the purpose of a WebSocket message.
type MsgType string

const (
	// Sent by server to a newly-connected client: full current state.
	MsgTypeSync MsgType = "timeline_sync"
	// Broadcast when a clip is inserted (either by a user or the agent).
	MsgTypeClipInsert MsgType = "clip_insert"
	// Broadcast when a clip is removed.
	MsgTypeClipDelete MsgType = "clip_delete"
	// Broadcast when one or more clip fields are updated (LWW).
	MsgTypeClipFieldUpdate MsgType = "clip_field_update"
	// Broadcast when a clip's fractional rank changes.
	MsgTypeClipReorder MsgType = "clip_reorder"
	// Broadcast on interval or user action for cursor/selection presence.
	MsgTypePresenceUpdate MsgType = "presence_update"
	// Broadcast when a client disconnects.
	MsgTypePresenceLeave MsgType = "presence_leave"
	// Broadcast when a named checkpoint is committed to the revision history.
	MsgTypeCheckpointCreated MsgType = "checkpoint_created"
	// Sent by a client to express its intent (same types as above).
	// The server re-broadcasts accepted ops to all other clients.
)

// Msg is the envelope for all WebSocket messages.
type Msg struct {
	Type    MsgType         `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// NewMsg serializes any payload into a Msg.
func NewMsg(t MsgType, payload any) ([]byte, error) {
	pb, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return json.Marshal(Msg{Type: t, Payload: pb})
}

// -----------------------------------------------------------------------
// Payload types
// -----------------------------------------------------------------------

// SyncPayload carries the full live timeline state sent to a new client.
type SyncPayload struct {
	ProjectID   string           `json:"project_id"`
	Clips       []LiveClip       `json:"clips"`
	Canvas      projects.TimelineCanvas       `json:"canvas"`
	Transitions []projects.TimelineTransition `json:"transitions,omitempty"`
	FPS         int              `json:"fps"`
	Presence    []PresenceState  `json:"presence,omitempty"`
	ServerSeq   int64            `json:"server_seq"`
}

// LiveClip extends TimelineClip with fractional rank and per-field timestamps.
type LiveClip struct {
	projects.TimelineClip
	Rank     string  `json:"rank"`
	FieldSeq int64   `json:"field_seq"` // max server seq of any field update
}

// ClipInsertPayload carries the new clip with its initial fractional rank.
type ClipInsertPayload struct {
	Clip      LiveClip  `json:"clip"`
	ClientID  string    `json:"client_id"`
	Timestamp Timestamp `json:"ts"`
}

// ClipDeletePayload identifies the clip to remove.
type ClipDeletePayload struct {
	ID        string    `json:"id"`
	ClientID  string    `json:"client_id"`
	Timestamp Timestamp `json:"ts"`
}

// ClipFieldUpdatePayload updates one or more fields on an existing clip.
// Fields is a map of JSON field-path to new value (e.g. "start_frame": 120).
type ClipFieldUpdatePayload struct {
	ClipID    string         `json:"clip_id"`
	Fields    map[string]any `json:"fields"`
	ClientID  string         `json:"client_id"`
	Timestamp Timestamp      `json:"ts"`
}

// ClipReorderPayload reassigns the fractional rank (position) of a clip.
type ClipReorderPayload struct {
	ID        string    `json:"id"`
	Rank      string    `json:"rank"`
	ClientID  string    `json:"client_id"`
	Timestamp Timestamp `json:"ts"`
}

// PresenceState is the presence record for one connected peer.
type PresenceState struct {
	ClientID       string `json:"client_id"`
	Name           string `json:"name"`
	Color          string `json:"color"`
	PlayheadFrame  int    `json:"playhead_frame"`
	SelectedClipID string `json:"selected_clip_id,omitempty"`
}

// PresenceUpdatePayload is sent by clients on interval or state change.
type PresenceUpdatePayload = PresenceState

// PresenceLeavePayload identifies the disconnecting client.
type PresenceLeavePayload struct {
	ClientID string `json:"client_id"`
}

// CheckpointCreatedPayload announces a new immutable revision.
type CheckpointCreatedPayload struct {
	ProjectID  string `json:"project_id"`
	RevisionID int    `json:"revision_id"`
	Name       string `json:"name,omitempty"`
}
