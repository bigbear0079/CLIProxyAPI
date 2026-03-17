package amp

import (
	"bytes"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	log "github.com/sirupsen/logrus"
)

// ResponseRewriter wraps a gin.ResponseWriter to intercept and modify the response body
// It's used to rewrite model names in responses when model mapping is used
type ResponseRewriter struct {
	gin.ResponseWriter
	body          *bytes.Buffer
	originalModel string
	isStreaming   bool
}

// NewResponseRewriter creates a new response rewriter for model name substitution
func NewResponseRewriter(w gin.ResponseWriter, originalModel string) *ResponseRewriter {
	return &ResponseRewriter{
		ResponseWriter: w,
		body:           &bytes.Buffer{},
		originalModel:  originalModel,
	}
}

// Write intercepts response writes and buffers them for model name replacement
func (rw *ResponseRewriter) Write(data []byte) (int, error) {
	// Detect streaming on first write
	if rw.body.Len() == 0 && !rw.isStreaming {
		contentType := rw.Header().Get("Content-Type")
		rw.isStreaming = strings.Contains(contentType, "text/event-stream") ||
			strings.Contains(contentType, "stream")
	}

	if rw.isStreaming {
		n, err := rw.ResponseWriter.Write(rw.rewriteStreamChunk(data))
		if err == nil {
			if flusher, ok := rw.ResponseWriter.(http.Flusher); ok {
				flusher.Flush()
			}
		}
		return n, err
	}
	return rw.body.Write(data)
}

// Flush writes the buffered response with model names rewritten
func (rw *ResponseRewriter) Flush() {
	if rw.isStreaming {
		if flusher, ok := rw.ResponseWriter.(http.Flusher); ok {
			flusher.Flush()
		}
		return
	}
	if rw.body.Len() > 0 {
		if _, err := rw.ResponseWriter.Write(rw.rewriteModelInResponse(rw.body.Bytes())); err != nil {
			log.Warnf("amp response rewriter: failed to write rewritten response: %v", err)
		}
	}
}

// modelFieldPaths lists all JSON paths where model name may appear
var modelFieldPaths = []string{"message.model", "model", "modelVersion", "response.model", "response.modelVersion"}

// rewriteModelInResponse replaces all occurrences of the mapped model with the original model in JSON
// It also suppresses "thinking" blocks if "tool_use" is present to ensure Amp client compatibility
func (rw *ResponseRewriter) rewriteModelInResponse(data []byte) []byte {
	root, errParse := jsonutil.ParseObjectBytes(data)
	if errParse != nil {
		return data
	}
	modified := false

	// 1. Amp Compatibility: Suppress thinking blocks if tool use is detected
	// The Amp client struggles when both thinking and tool_use blocks are present
	contentValue, okContent := jsonutil.Get(root, "content")
	if okContent {
		if contentArray, okArray := contentValue.([]any); okArray {
			hasToolUse := false
			filtered := make([]any, 0, len(contentArray))
			for _, itemValue := range contentArray {
				item, okItem := itemValue.(map[string]any)
				if !okItem {
					filtered = append(filtered, itemValue)
					continue
				}
				itemType, _ := item["type"].(string)
				if itemType == "tool_use" {
					hasToolUse = true
				}
				if itemType != "thinking" {
					filtered = append(filtered, itemValue)
				}
			}

			if hasToolUse && len(contentArray) > len(filtered) {
				if errSet := jsonutil.Set(root, "content", filtered); errSet != nil {
					log.Warnf("Amp ResponseRewriter: failed to suppress thinking blocks: %v", errSet)
				} else {
					modified = true
					log.Debugf("Amp ResponseRewriter: Suppressed %d thinking blocks due to tool usage", len(contentArray)-len(filtered))
				}
			}
		}
	}

	if rw.originalModel == "" {
		if !modified {
			return data
		}
		return jsonutil.MarshalOrOriginal(data, root)
	}
	for _, path := range modelFieldPaths {
		if jsonutil.Exists(root, path) {
			if errSet := jsonutil.Set(root, path, rw.originalModel); errSet == nil {
				modified = true
			}
		}
	}
	if !modified {
		return data
	}
	return jsonutil.MarshalOrOriginal(data, root)
}

// rewriteStreamChunk rewrites model names in SSE stream chunks
func (rw *ResponseRewriter) rewriteStreamChunk(chunk []byte) []byte {
	if rw.originalModel == "" {
		return chunk
	}

	// SSE format: "data: {json}\n\n"
	lines := bytes.Split(chunk, []byte("\n"))
	for i, line := range lines {
		if bytes.HasPrefix(line, []byte("data: ")) {
			jsonData := bytes.TrimPrefix(line, []byte("data: "))
			if len(jsonData) > 0 && jsonData[0] == '{' {
				// Rewrite JSON in the data line
				rewritten := rw.rewriteModelInResponse(jsonData)
				lines[i] = append([]byte("data: "), rewritten...)
			}
		}
	}

	return bytes.Join(lines, []byte("\n"))
}
