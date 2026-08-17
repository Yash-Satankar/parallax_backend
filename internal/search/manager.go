package search

import (
	"fmt"
	"path/filepath"
	"sync"
)

// Manager caches and coordinates *search.Index instances for projects.
type Manager struct {
	mu      sync.Mutex
	indices map[string]*Index
}

// NewManager creates a search index manager.
func NewManager() *Manager {
	return &Manager{
		indices: make(map[string]*Index),
	}
}

// GetIndex returns (or loads) the Index for a given project directory.
func (m *Manager) GetIndex(projectDir string) (*Index, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	indexPath := filepath.Join(projectDir, ".parallax", "search_index.json")
	if idx, ok := m.indices[indexPath]; ok {
		return idx, nil
	}

	idx, err := NewIndex(indexPath)
	if err != nil {
		return nil, fmt.Errorf("search manager: load index: %w", err)
	}

	m.indices[indexPath] = idx
	return idx, nil
}
