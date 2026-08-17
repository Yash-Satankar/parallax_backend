package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// EmbedClient calls an OpenAI-compatible /v1/embeddings endpoint.
type EmbedClient struct {
	BaseURL    string
	APIKey     string
	Model      string
	HTTPClient *http.Client
}

func NewEmbedClient(baseURL, apiKey, model string) *EmbedClient {
	return &EmbedClient{
		BaseURL: baseURL,
		APIKey:  apiKey,
		Model:   model,
		HTTPClient: &http.Client{
			Timeout: 60 * time.Second,
			Transport: &http.Transport{
				Proxy:               http.ProxyFromEnvironment,
				ForceAttemptHTTP2:   true,
				MaxIdleConns:        8,
				IdleConnTimeout:     90 * time.Second,
				TLSHandshakeTimeout: 15 * time.Second,
			},
		},
	}
}

// EmbedURL constructs the embedding endpoint URL from the base URL.
func EmbedURL(base string) string {
	if base == "" {
		return ""
	}
	for len(base) > 0 && base[len(base)-1] == '/' {
		base = base[:len(base)-1]
	}
	const suffix = "/embeddings"
	if strings.HasSuffix(base, suffix) {
		return base
	}
	return base + suffix
}

type embedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"` // string or []string
}

type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Embed returns a single embedding vector for the given text.
func (c *EmbedClient) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := c.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("embed: no vectors returned")
	}
	return vecs[0], nil
}

// EmbedBatch returns embedding vectors for a batch of texts.
func (c *EmbedClient) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if strings.TrimSpace(c.APIKey) == "" {
		return nil, fmt.Errorf("embed: api key is not configured")
	}
	url := EmbedURL(c.BaseURL)
	if url == "" {
		return nil, fmt.Errorf("embed: base_url is not configured")
	}

	body, err := json.Marshal(embedRequest{Model: c.Model, Input: texts})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode >= 300 {
		var we wireError
		if json.Unmarshal(raw, &we) == nil && we.Error.Message != "" {
			return nil, fmt.Errorf("embed: %s (http %d)", we.Error.Message, resp.StatusCode)
		}
		return nil, fmt.Errorf("embed: provider returned http %d", resp.StatusCode)
	}

	var er embedResponse
	if err := json.Unmarshal(raw, &er); err != nil {
		return nil, fmt.Errorf("embed: decode response: %w", err)
	}
	if er.Error != nil && er.Error.Message != "" {
		return nil, fmt.Errorf("embed: %s", er.Error.Message)
	}

	out := make([][]float32, len(er.Data))
	for i, d := range er.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// --------------------------------------------------------------------------
// Transcription (OpenAI-compat /v1/audio/transcriptions, verbose_json)
// --------------------------------------------------------------------------

// TranscribeClient calls an OpenAI-compatible /v1/audio/transcriptions endpoint
// and requests word-level timestamps via response_format=verbose_json.
type TranscribeClient struct {
	BaseURL    string
	APIKey     string
	Model      string
	HTTPClient *http.Client
}

func NewTranscribeClient(baseURL, apiKey, model string) *TranscribeClient {
	if model == "" {
		model = "whisper-1"
	}
	return &TranscribeClient{
		BaseURL: baseURL,
		APIKey:  apiKey,
		Model:   model,
		HTTPClient: &http.Client{
			Timeout: 5 * time.Minute, // transcription can be slow for long files
			Transport: &http.Transport{
				Proxy:               http.ProxyFromEnvironment,
				ForceAttemptHTTP2:   true,
				MaxIdleConns:        4,
				IdleConnTimeout:     90 * time.Second,
				TLSHandshakeTimeout: 15 * time.Second,
			},
		},
	}
}

// TranscribeURL constructs the transcription endpoint URL from the base URL.
func TranscribeURL(base string) string {
	if base == "" {
		return ""
	}
	for len(base) > 0 && base[len(base)-1] == '/' {
		base = base[:len(base)-1]
	}
	const suffix = "/audio/transcriptions"
	if strings.HasSuffix(base, suffix) {
		return base
	}
	return base + suffix
}

// TranscriptWord is one word with its start/end timestamps from Whisper.
type TranscriptWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// TranscriptSegment is a phrase-level segment from Whisper verbose_json.
type TranscriptSegment struct {
	ID    int              `json:"id"`
	Start float64          `json:"start"`
	End   float64          `json:"end"`
	Text  string           `json:"text"`
	Words []TranscriptWord `json:"words,omitempty"`
}

// Transcript is the full transcription result.
type Transcript struct {
	Language string              `json:"language"`
	Duration float64             `json:"duration"`
	Text     string              `json:"text"`
	Segments []TranscriptSegment `json:"segments"`
	Words    []TranscriptWord    `json:"words,omitempty"` // flattened across all segments
}

type transcribeResponse struct {
	Task     string              `json:"task"`
	Language string              `json:"language"`
	Duration float64             `json:"duration"`
	Text     string              `json:"text"`
	Segments []TranscriptSegment `json:"segments"`
	Words    []TranscriptWord    `json:"words"`
	Error    *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Transcribe uploads an audio file and returns the transcript with word-level timestamps.
// audioPath is the absolute path to the audio file; lang is an optional language hint (e.g. "en").
func (c *TranscribeClient) Transcribe(ctx context.Context, audioPath, lang string) (*Transcript, error) {
	if strings.TrimSpace(c.APIKey) == "" {
		return nil, fmt.Errorf("transcribe: api key is not configured")
	}
	url := TranscribeURL(c.BaseURL)
	if url == "" {
		return nil, fmt.Errorf("transcribe: base_url is not configured")
	}

	// Build multipart form: file + model + response_format
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	fw, err := mw.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		return nil, err
	}
	// Read the audio file; caller must ensure path is valid.
	// We stream in chunks rather than loading all into memory.
	audioData, err := io.ReadAll(openFile(audioPath))
	if err != nil {
		return nil, fmt.Errorf("transcribe: read audio: %w", err)
	}
	if _, err := fw.Write(audioData); err != nil {
		return nil, err
	}

	_ = mw.WriteField("model", c.Model)
	_ = mw.WriteField("response_format", "verbose_json")
	_ = mw.WriteField("timestamp_granularities[]", "word")
	_ = mw.WriteField("timestamp_granularities[]", "segment")
	if lang != "" {
		_ = mw.WriteField("language", lang)
	}
	mw.Close()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("transcribe request: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if resp.StatusCode >= 300 {
		var we wireError
		if json.Unmarshal(raw, &we) == nil && we.Error.Message != "" {
			return nil, fmt.Errorf("transcribe: %s (http %d)", we.Error.Message, resp.StatusCode)
		}
		return nil, fmt.Errorf("transcribe: provider returned http %d", resp.StatusCode)
	}

	var tr transcribeResponse
	if err := json.Unmarshal(raw, &tr); err != nil {
		return nil, fmt.Errorf("transcribe: decode response: %w", err)
	}
	if tr.Error != nil && tr.Error.Message != "" {
		return nil, fmt.Errorf("transcribe: %s", tr.Error.Message)
	}

	// Flatten words from segments when top-level words array is missing.
	words := tr.Words
	if len(words) == 0 {
		for _, seg := range tr.Segments {
			words = append(words, seg.Words...)
		}
	}

	return &Transcript{
		Language: tr.Language,
		Duration: tr.Duration,
		Text:     tr.Text,
		Segments: tr.Segments,
		Words:    words,
	}, nil
}

// openFileInternal opens a file and returns an io.Reader; errors propagate via ReadAll.
func openFileInternal(path string) (io.Reader, error) {
	return os.Open(path)
}

// openFile wraps os.Open; read errors surface via io.ReadAll returning an error.
func openFile(path string) io.Reader {
	f, err := os.Open(path)
	if err != nil {
		// Return an error reader so io.ReadAll propagates the error.
		return errorReader{err}
	}
	return f
}

type errorReader struct{ err error }

func (e errorReader) Read(_ []byte) (int, error) { return 0, e.err }

