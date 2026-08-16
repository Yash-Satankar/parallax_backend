package collab

import "strings"

// Timestamp is a Lamport-style logical clock value used for LWW (last-write-wins)
// conflict resolution. When two clients concurrently update the same clip field,
// the one with the higher Timestamp wins. On a tie, the lexicographically
// greater ClientID wins, giving a deterministic total order.
type Timestamp struct {
	Seq      int64  `json:"seq"`       // server-assigned monotonic sequence number
	ClientID string `json:"client_id"` // stable per-connection ID
}

// After reports whether t is strictly greater than other.
func (t Timestamp) After(other Timestamp) bool {
	if t.Seq != other.Seq {
		return t.Seq > other.Seq
	}
	return strings.Compare(t.ClientID, other.ClientID) > 0
}

// Zero is the zero value (less than any real timestamp).
var Zero Timestamp

// LWWField holds one field's current value and the timestamp that last wrote it.
type LWWField struct {
	Value any       `json:"value"`
	TS    Timestamp `json:"ts"`
}

// Apply returns the winning field given a proposed new value and its timestamp.
// If the proposal has an earlier timestamp, the existing field wins.
func (f LWWField) Apply(value any, ts Timestamp) LWWField {
	if ts.After(f.TS) {
		return LWWField{Value: value, TS: ts}
	}
	return f
}

// LWWMap is a map of named fields to their LWW-tracked values.
type LWWMap map[string]LWWField

// Set merges a proposed update into the map.
// Returns true if the value was actually changed.
func (m LWWMap) Set(field string, value any, ts Timestamp) bool {
	existing, ok := m[field]
	if !ok || ts.After(existing.TS) {
		m[field] = LWWField{Value: value, TS: ts}
		return true
	}
	return false
}

// Get returns the current value for a field and whether it exists.
func (m LWWMap) Get(field string) (any, bool) {
	f, ok := m[field]
	if !ok {
		return nil, false
	}
	return f.Value, true
}
