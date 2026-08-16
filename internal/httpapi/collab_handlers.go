package httpapi

import (
	"net/http"
)

// handleCollab upgrades the HTTP connection to a WebSocket for the collab hub.
func (s *Server) handleCollab(w http.ResponseWriter, r *http.Request) {
	projectID := r.PathValue("id")
	if s.Projects == nil || s.CollabHub == nil {
		writeError(w, http.StatusServiceUnavailable, "collaboration is not configured")
		return
	}
	if _, err := s.Projects.Get(projectID); err != nil {
		writeProjectError(w, err)
		return
	}
	s.CollabHub.ServeWS(w, r, projectID)
}

