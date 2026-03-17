// Package openai provides HTTP handlers for OpenAI API endpoints.
// This package implements the OpenAI-compatible API interface, including model listing
// and chat completion functionality. It supports both streaming and non-streaming responses,
// and manages a pool of clients to interact with backend services.
// The handlers translate OpenAI API requests to the appropriate backend format and
// convert responses back to OpenAI-compatible format.
package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
	. "github.com/router-for-me/CLIProxyAPI/v6/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	responsesconverter "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/openai/openai/responses"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/api/handlers"
)

// OpenAIAPIHandler contains the handlers for OpenAI API endpoints.
// It holds a pool of clients to interact with the backend service.
type OpenAIAPIHandler struct {
	*handlers.BaseAPIHandler
}

// NewOpenAIAPIHandler creates a new OpenAI API handlers instance.
// It takes an BaseAPIHandler instance as input and returns an OpenAIAPIHandler.
//
// Parameters:
//   - apiHandlers: The base API handlers instance
//
// Returns:
//   - *OpenAIAPIHandler: A new OpenAI API handlers instance
func NewOpenAIAPIHandler(apiHandlers *handlers.BaseAPIHandler) *OpenAIAPIHandler {
	return &OpenAIAPIHandler{
		BaseAPIHandler: apiHandlers,
	}
}

// HandlerType returns the identifier for this handler implementation.
func (h *OpenAIAPIHandler) HandlerType() string {
	return OpenAI
}

// Models returns the OpenAI-compatible model metadata supported by this handler.
func (h *OpenAIAPIHandler) Models() []map[string]any {
	// Get dynamic models from the global registry
	modelRegistry := registry.GetGlobalRegistry()
	return modelRegistry.GetAvailableModels("openai")
}

// OpenAIModels handles the /v1/models endpoint.
// It returns a list of available AI models with their capabilities
// and specifications in OpenAI-compatible format.
func (h *OpenAIAPIHandler) OpenAIModels(c *gin.Context) {
	// Get all available models
	allModels := h.Models()

	// Filter to only include the 4 required fields: id, object, created, owned_by
	filteredModels := make([]map[string]any, len(allModels))
	for i, model := range allModels {
		filteredModel := map[string]any{
			"id":     model["id"],
			"object": model["object"],
		}

		// Add created field if it exists
		if created, exists := model["created"]; exists {
			filteredModel["created"] = created
		}

		// Add owned_by field if it exists
		if ownedBy, exists := model["owned_by"]; exists {
			filteredModel["owned_by"] = ownedBy
		}

		filteredModels[i] = filteredModel
	}

	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   filteredModels,
	})
}

// ChatCompletions handles the /v1/chat/completions endpoint.
// It determines whether the request is for a streaming or non-streaming response
// and calls the appropriate handler based on the model provider.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
func (h *OpenAIAPIHandler) ChatCompletions(c *gin.Context) {
	rawJSON, err := c.GetRawData()
	// If data retrieval fails, return a 400 Bad Request error.
	if err != nil {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	stream := false
	modelName := ""
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse == nil {
		stream = jsonBoolField(root, "stream")
		modelName = jsonStringField(root, "model")

		// Some clients send OpenAI Responses-format payloads to /v1/chat/completions.
		// Convert them to Chat Completions so downstream translators preserve tool metadata.
		if shouldTreatAsResponsesFormat(root) {
			rawJSON = responsesconverter.ConvertOpenAIResponsesRequestObjectToOpenAIChatCompletions(modelName, root, stream)
		}
	}

	if stream {
		h.handleStreamingResponse(c, rawJSON, modelName)
	} else {
		h.handleNonStreamingResponse(c, rawJSON, modelName)
	}

}

// shouldTreatAsResponsesFormat detects OpenAI Responses-style payloads that are
// accidentally sent to the Chat Completions endpoint.
func shouldTreatAsResponsesFormat(root map[string]any) bool {
	if root == nil {
		return false
	}
	if jsonutil.Exists(root, "messages") {
		return false
	}
	if jsonutil.Exists(root, "input") {
		return true
	}
	if jsonutil.Exists(root, "instructions") {
		return true
	}
	return false
}

// Completions handles the /v1/completions endpoint.
// It determines whether the request is for a streaming or non-streaming response
// and calls the appropriate handler based on the model provider.
// This endpoint follows the OpenAI completions API specification.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
func (h *OpenAIAPIHandler) Completions(c *gin.Context) {
	rawJSON, err := c.GetRawData()
	// If data retrieval fails, return a 400 Bad Request error.
	if err != nil {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	stream := false
	chatCompletionsJSON, modelName := convertCompletionsRequestToChatCompletions(root)
	if errParse == nil {
		stream = jsonBoolField(root, "stream")
	}

	if stream {
		h.handleCompletionsStreamingResponse(c, chatCompletionsJSON, modelName)
	} else {
		h.handleCompletionsNonStreamingResponse(c, chatCompletionsJSON, modelName)
	}

}

// convertCompletionsRequestToChatCompletions converts OpenAI completions API request to chat completions format.
// This allows the completions endpoint to use the existing chat completions infrastructure.
//
// Parameters:
//   - rawJSON: The raw JSON bytes of the completions request
//
// Returns:
//   - []byte: The converted chat completions request
func convertCompletionsRequestToChatCompletions(root map[string]any) ([]byte, string) {
	// Extract prompt from completions request.
	prompt := jsonStringField(root, "prompt")
	if prompt == "" {
		prompt = "Complete this:"
	}

	modelName := jsonStringField(root, "model")
	out := map[string]any{
		"model": modelName,
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": prompt,
			},
		},
	}

	for _, key := range []string{
		"max_tokens",
		"temperature",
		"top_p",
		"frequency_penalty",
		"presence_penalty",
		"stop",
		"stream",
		"logprobs",
		"top_logprobs",
		"echo",
	} {
		if value, ok := root[key]; ok {
			out[key] = value
		}
	}

	payload, errMarshal := json.Marshal(out)
	if errMarshal != nil {
		return []byte(`{"model":"","messages":[{"role":"user","content":"Complete this:"}]}`), modelName
	}
	return payload, modelName
}

// convertChatCompletionsResponseToCompletions converts chat completions API response back to completions format.
// This ensures the completions endpoint returns data in the expected format.
//
// Parameters:
//   - rawJSON: The raw JSON bytes of the chat completions response
//
// Returns:
//   - []byte: The converted completions response
func convertChatCompletionsResponseToCompletions(rawJSON []byte) []byte {
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	out := map[string]any{
		"id":      "",
		"object":  "text_completion",
		"created": int64(0),
		"model":   "",
		"choices": make([]any, 0),
	}
	if errParse != nil {
		return marshalOpenAICompletionsPayload(rawJSON, out)
	}

	if value, ok := root["id"]; ok {
		out["id"] = value
	}
	if value, ok := root["created"]; ok {
		out["created"] = value
	}
	if value, ok := root["model"]; ok {
		out["model"] = value
	}
	if value, ok := root["usage"]; ok {
		out["usage"] = value
	}

	choices := make([]any, 0)
	if chatChoices, ok := root["choices"].([]any); ok {
		for _, choiceValue := range chatChoices {
			choice, ok := choiceValue.(map[string]any)
			if !ok {
				continue
			}

			completionsChoice := map[string]any{
				"index": choice["index"],
			}

			if message, ok := choice["message"].(map[string]any); ok {
				if content, ok := message["content"]; ok {
					completionsChoice["text"] = jsonStringValue(content)
				}
			} else if delta, ok := choice["delta"].(map[string]any); ok {
				if content, ok := delta["content"]; ok {
					completionsChoice["text"] = jsonStringValue(content)
				}
			}

			if finishReason, ok := choice["finish_reason"]; ok && finishReason != nil {
				completionsChoice["finish_reason"] = jsonStringValue(finishReason)
			}
			if logprobs, ok := choice["logprobs"]; ok {
				completionsChoice["logprobs"] = logprobs
			}

			choices = append(choices, completionsChoice)
		}
	}
	out["choices"] = choices
	return marshalOpenAICompletionsPayload(rawJSON, out)
}

// convertChatCompletionsStreamChunkToCompletions converts a streaming chat completions chunk to completions format.
// This handles the real-time conversion of streaming response chunks and filters out empty text responses.
//
// Parameters:
//   - chunkData: The raw JSON bytes of a single chat completions stream chunk
//
// Returns:
//   - []byte: The converted completions stream chunk, or nil if should be filtered out
func convertChatCompletionsStreamChunkToCompletions(chunkData []byte) []byte {
	root, errParse := jsonutil.ParseObjectBytes(chunkData)
	if errParse != nil {
		return nil
	}

	chatChoices, _ := root["choices"].([]any)
	hasUsage := jsonutil.Exists(root, "usage")
	hasContent := false
	for _, choiceValue := range chatChoices {
		choice, ok := choiceValue.(map[string]any)
		if !ok {
			continue
		}

		if delta, ok := choice["delta"].(map[string]any); ok {
			if content := jsonStringValue(delta["content"]); content != "" {
				hasContent = true
				break
			}
		}

		finishReason := jsonStringValue(choice["finish_reason"])
		if finishReason != "" && finishReason != "null" {
			hasContent = true
			break
		}
	}
	if !hasContent && !hasUsage {
		return nil
	}

	out := map[string]any{
		"id":      "",
		"object":  "text_completion",
		"created": int64(0),
		"model":   "",
		"choices": make([]any, 0),
	}
	if value, ok := root["id"]; ok {
		out["id"] = value
	}
	if value, ok := root["created"]; ok {
		out["created"] = value
	}
	if value, ok := root["model"]; ok {
		out["model"] = value
	}

	choices := make([]any, 0, len(chatChoices))
	for _, choiceValue := range chatChoices {
		choice, ok := choiceValue.(map[string]any)
		if !ok {
			continue
		}

		completionsChoice := map[string]any{
			"index": choice["index"],
			"text":  "",
		}
		if delta, ok := choice["delta"].(map[string]any); ok {
			completionsChoice["text"] = jsonStringValue(delta["content"])
		}

		finishReason := jsonStringValue(choice["finish_reason"])
		if finishReason != "" && finishReason != "null" {
			completionsChoice["finish_reason"] = finishReason
		}
		if logprobs, ok := choice["logprobs"]; ok {
			completionsChoice["logprobs"] = logprobs
		}

		choices = append(choices, completionsChoice)
	}
	out["choices"] = choices
	if usage, ok := root["usage"]; ok {
		out["usage"] = usage
	}

	return marshalOpenAICompletionsPayload(chunkData, out)
}

func marshalOpenAICompletionsPayload(original []byte, payload map[string]any) []byte {
	out, errMarshal := json.Marshal(payload)
	if errMarshal != nil {
		return original
	}
	return out
}

func jsonStringValue(value any) string {
	if value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return typed
	case json.Number:
		return typed.String()
	case bool:
		if typed {
			return "true"
		}
		return "false"
	default:
		out, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return ""
		}
		return string(out)
	}
}

// handleNonStreamingResponse handles non-streaming chat completion responses
// for Gemini models. It selects a client from the pool, sends the request, and
// aggregates the response before sending it back to the client in OpenAI format.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
//   - rawJSON: The raw JSON bytes of the OpenAI-compatible request
func (h *OpenAIAPIHandler) handleNonStreamingResponse(c *gin.Context, rawJSON []byte, modelName string) {
	c.Header("Content-Type", "application/json")

	if modelName == "" {
		modelName = jsonStringFieldBytes(rawJSON, "model")
	}
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, h.GetAlt(c))
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

// handleStreamingResponse handles streaming responses for Gemini models.
// It establishes a streaming connection with the backend service and forwards
// the response chunks to the client in real-time using Server-Sent Events.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
//   - rawJSON: The raw JSON bytes of the OpenAI-compatible request
func (h *OpenAIAPIHandler) handleStreamingResponse(c *gin.Context, rawJSON []byte, modelName string) {
	// Get the http.Flusher interface to manually flush the response.
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Streaming not supported",
				Type:    "server_error",
			},
		})
		return
	}

	if modelName == "" {
		modelName = jsonStringFieldBytes(rawJSON, "model")
	}
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	dataChan, upstreamHeaders, errChan := h.ExecuteStreamWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, h.GetAlt(c))

	setSSEHeaders := func() {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("Access-Control-Allow-Origin", "*")
	}

	// Peek at the first chunk to determine success or failure before setting headers
	for {
		select {
		case <-c.Request.Context().Done():
			cliCancel(c.Request.Context().Err())
			return
		case errMsg, ok := <-errChan:
			if !ok {
				// Err channel closed cleanly; wait for data channel.
				errChan = nil
				continue
			}
			// Upstream failed immediately. Return proper error status and JSON.
			h.WriteErrorResponse(c, errMsg)
			if errMsg != nil {
				cliCancel(errMsg.Error)
			} else {
				cliCancel(nil)
			}
			return
		case chunk, ok := <-dataChan:
			if !ok {
				// Stream closed without data? Send DONE or just headers.
				setSSEHeaders()
				handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
				_, _ = fmt.Fprintf(c.Writer, "data: [DONE]\n\n")
				flusher.Flush()
				cliCancel(nil)
				return
			}

			// Success! Commit to streaming headers.
			setSSEHeaders()
			handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)

			_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", string(chunk))
			flusher.Flush()

			// Continue streaming the rest
			h.handleStreamResult(c, flusher, func(err error) { cliCancel(err) }, dataChan, errChan)
			return
		}
	}
}

// handleCompletionsNonStreamingResponse handles non-streaming completions responses.
// It converts completions request to chat completions format, sends to backend,
// then converts the response back to completions format before sending to client.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
//   - rawJSON: The raw JSON bytes of the OpenAI-compatible completions request
func (h *OpenAIAPIHandler) handleCompletionsNonStreamingResponse(c *gin.Context, chatCompletionsJSON []byte, modelName string) {
	c.Header("Content-Type", "application/json")

	if modelName == "" {
		modelName = jsonStringFieldBytes(chatCompletionsJSON, "model")
	}
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)
	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, chatCompletionsJSON, "")
	stopKeepAlive()
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	completionsResp := convertChatCompletionsResponseToCompletions(resp)
	_, _ = c.Writer.Write(completionsResp)
	cliCancel()
}

// handleCompletionsStreamingResponse handles streaming completions responses.
// It converts completions request to chat completions format, streams from backend,
// then converts each response chunk back to completions format before sending to client.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
//   - rawJSON: The raw JSON bytes of the OpenAI-compatible completions request
func (h *OpenAIAPIHandler) handleCompletionsStreamingResponse(c *gin.Context, chatCompletionsJSON []byte, modelName string) {
	// Get the http.Flusher interface to manually flush the response.
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Streaming not supported",
				Type:    "server_error",
			},
		})
		return
	}

	if modelName == "" {
		modelName = jsonStringFieldBytes(chatCompletionsJSON, "model")
	}
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	dataChan, upstreamHeaders, errChan := h.ExecuteStreamWithAuthManager(cliCtx, h.HandlerType(), modelName, chatCompletionsJSON, "")

	setSSEHeaders := func() {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("Access-Control-Allow-Origin", "*")
	}

	// Peek at the first chunk
	for {
		select {
		case <-c.Request.Context().Done():
			cliCancel(c.Request.Context().Err())
			return
		case errMsg, ok := <-errChan:
			if !ok {
				// Err channel closed cleanly; wait for data channel.
				errChan = nil
				continue
			}
			h.WriteErrorResponse(c, errMsg)
			if errMsg != nil {
				cliCancel(errMsg.Error)
			} else {
				cliCancel(nil)
			}
			return
		case chunk, ok := <-dataChan:
			if !ok {
				setSSEHeaders()
				handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
				_, _ = fmt.Fprintf(c.Writer, "data: [DONE]\n\n")
				flusher.Flush()
				cliCancel(nil)
				return
			}

			// Success! Set headers.
			setSSEHeaders()
			handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)

			// Write the first chunk
			converted := convertChatCompletionsStreamChunkToCompletions(chunk)
			if converted != nil {
				_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", string(converted))
				flusher.Flush()
			}

			done := make(chan struct{})
			var doneOnce sync.Once
			stop := func() { doneOnce.Do(func() { close(done) }) }

			convertedChan := make(chan []byte)
			go func() {
				defer close(convertedChan)
				for {
					select {
					case <-done:
						return
					case chunk, ok := <-dataChan:
						if !ok {
							return
						}
						converted := convertChatCompletionsStreamChunkToCompletions(chunk)
						if converted == nil {
							continue
						}
						select {
						case <-done:
							return
						case convertedChan <- converted:
						}
					}
				}
			}()

			h.handleStreamResult(c, flusher, func(err error) {
				stop()
				cliCancel(err)
			}, convertedChan, errChan)
			return
		}
	}
}
func (h *OpenAIAPIHandler) handleStreamResult(c *gin.Context, flusher http.Flusher, cancel func(error), data <-chan []byte, errs <-chan *interfaces.ErrorMessage) {
	h.ForwardStream(c, flusher, cancel, data, errs, handlers.StreamForwardOptions{
		WriteChunk: func(chunk []byte) {
			_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", string(chunk))
		},
		WriteTerminalError: func(errMsg *interfaces.ErrorMessage) {
			if errMsg == nil {
				return
			}
			status := http.StatusInternalServerError
			if errMsg.StatusCode > 0 {
				status = errMsg.StatusCode
			}
			errText := http.StatusText(status)
			if errMsg.Error != nil && errMsg.Error.Error() != "" {
				errText = errMsg.Error.Error()
			}
			body := handlers.BuildErrorResponseBody(status, errText)
			_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", string(body))
		},
		WriteDone: func() {
			_, _ = fmt.Fprint(c.Writer, "data: [DONE]\n\n")
		},
	})
}
