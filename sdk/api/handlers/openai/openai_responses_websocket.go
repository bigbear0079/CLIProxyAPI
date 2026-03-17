package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	log "github.com/sirupsen/logrus"
)

const (
	wsRequestTypeCreate  = "response.create"
	wsRequestTypeAppend  = "response.append"
	wsEventTypeError     = "error"
	wsEventTypeCompleted = "response.completed"
	wsDoneMarker         = "[DONE]"
	wsTurnStateHeader    = "x-codex-turn-state"
	wsRequestBodyKey     = "REQUEST_BODY_OVERRIDE"
	wsPayloadLogMaxSize  = 2048
	wsBodyLogMaxSize     = 64 * 1024
	wsBodyLogTruncated   = "\n[websocket log truncated]\n"
)

var responsesWebsocketUpgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

// ResponsesWebsocket handles websocket requests for /v1/responses.
// It accepts `response.create` and `response.append` requests and streams
// response events back as JSON websocket text messages.
func (h *OpenAIResponsesAPIHandler) ResponsesWebsocket(c *gin.Context) {
	conn, err := responsesWebsocketUpgrader.Upgrade(c.Writer, c.Request, websocketUpgradeHeaders(c.Request))
	if err != nil {
		return
	}
	passthroughSessionID := uuid.NewString()
	clientRemoteAddr := ""
	if c != nil && c.Request != nil {
		clientRemoteAddr = strings.TrimSpace(c.Request.RemoteAddr)
	}
	log.Infof("responses websocket: client connected id=%s remote=%s", passthroughSessionID, clientRemoteAddr)
	var wsTerminateErr error
	var wsBodyLog strings.Builder
	defer func() {
		if wsTerminateErr != nil {
			// log.Infof("responses websocket: session closing id=%s reason=%v", passthroughSessionID, wsTerminateErr)
		} else {
			log.Infof("responses websocket: session closing id=%s", passthroughSessionID)
		}
		if h != nil && h.AuthManager != nil {
			h.AuthManager.CloseExecutionSession(passthroughSessionID)
			log.Infof("responses websocket: upstream execution session closed id=%s", passthroughSessionID)
		}
		setWebsocketRequestBody(c, wsBodyLog.String())
		if errClose := conn.Close(); errClose != nil {
			log.Warnf("responses websocket: close connection error: %v", errClose)
		}
	}()

	var lastRequest []byte
	lastResponseOutput := []byte("[]")
	pinnedAuthID := ""

	for {
		msgType, payload, errReadMessage := conn.ReadMessage()
		if errReadMessage != nil {
			wsTerminateErr = errReadMessage
			appendWebsocketEvent(&wsBodyLog, "disconnect", []byte(errReadMessage.Error()))
			if websocket.IsCloseError(errReadMessage, websocket.CloseNormalClosure, websocket.CloseGoingAway, websocket.CloseNoStatusReceived) {
				log.Infof("responses websocket: client disconnected id=%s error=%v", passthroughSessionID, errReadMessage)
			} else {
				// log.Warnf("responses websocket: read message failed id=%s error=%v", passthroughSessionID, errReadMessage)
			}
			return
		}
		if msgType != websocket.TextMessage && msgType != websocket.BinaryMessage {
			continue
		}
		// log.Infof(
		// 	"responses websocket: downstream_in id=%s type=%d event=%s payload=%s",
		// 	passthroughSessionID,
		// 	msgType,
		// 	websocketPayloadEventType(payload),
		// 	websocketPayloadPreview(payload),
		// )
		appendWebsocketEvent(&wsBodyLog, "request", payload)

		allowIncrementalInputWithPreviousResponseID := false
		if pinnedAuthID != "" && h != nil && h.AuthManager != nil {
			if pinnedAuth, ok := h.AuthManager.GetByID(pinnedAuthID); ok && pinnedAuth != nil {
				allowIncrementalInputWithPreviousResponseID = websocketUpstreamSupportsIncrementalInput(pinnedAuth.Attributes, pinnedAuth.Metadata)
			}
		} else {
			requestModelName := strings.TrimSpace(jsonStringFieldBytes(payload, "model"))
			if requestModelName == "" {
				requestModelName = strings.TrimSpace(jsonStringFieldBytes(lastRequest, "model"))
			}
			allowIncrementalInputWithPreviousResponseID = h.websocketUpstreamSupportsIncrementalInputForModel(requestModelName)
		}

		var requestJSON []byte
		var updatedLastRequest []byte
		var errMsg *interfaces.ErrorMessage
		requestJSON, updatedLastRequest, errMsg = normalizeResponsesWebsocketRequestWithMode(
			payload,
			lastRequest,
			lastResponseOutput,
			allowIncrementalInputWithPreviousResponseID,
		)
		if errMsg != nil {
			h.LoggingAPIResponseError(context.WithValue(context.Background(), "gin", c), errMsg)
			markAPIResponseTimestamp(c)
			errorPayload, errWrite := writeResponsesWebsocketError(conn, errMsg)
			appendWebsocketEvent(&wsBodyLog, "response", errorPayload)
			log.Infof(
				"responses websocket: downstream_out id=%s type=%d event=%s payload=%s",
				passthroughSessionID,
				websocket.TextMessage,
				websocketPayloadEventType(errorPayload),
				websocketPayloadPreview(errorPayload),
			)
			if errWrite != nil {
				log.Warnf(
					"responses websocket: downstream_out write failed id=%s event=%s error=%v",
					passthroughSessionID,
					websocketPayloadEventType(errorPayload),
					errWrite,
				)
				return
			}
			continue
		}
		if shouldHandleResponsesWebsocketPrewarmLocally(payload, lastRequest, allowIncrementalInputWithPreviousResponseID) {
			requestJSON = deleteJSONPathBytes(requestJSON, "generate")
			updatedLastRequest = deleteJSONPathBytes(updatedLastRequest, "generate")
			lastRequest = updatedLastRequest
			lastResponseOutput = []byte("[]")
			if errWrite := writeResponsesWebsocketSyntheticPrewarm(c, conn, requestJSON, &wsBodyLog, passthroughSessionID); errWrite != nil {
				wsTerminateErr = errWrite
				appendWebsocketEvent(&wsBodyLog, "disconnect", []byte(errWrite.Error()))
				return
			}
			continue
		}
		lastRequest = updatedLastRequest

		modelName := jsonStringFieldBytes(requestJSON, "model")
		cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
		cliCtx = cliproxyexecutor.WithDownstreamWebsocket(cliCtx)
		cliCtx = handlers.WithExecutionSessionID(cliCtx, passthroughSessionID)
		if pinnedAuthID != "" {
			cliCtx = handlers.WithPinnedAuthID(cliCtx, pinnedAuthID)
		} else {
			cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, func(authID string) {
				authID = strings.TrimSpace(authID)
				if authID == "" || h == nil || h.AuthManager == nil {
					return
				}
				selectedAuth, ok := h.AuthManager.GetByID(authID)
				if !ok || selectedAuth == nil {
					return
				}
				if websocketUpstreamSupportsIncrementalInput(selectedAuth.Attributes, selectedAuth.Metadata) {
					pinnedAuthID = authID
				}
			})
		}
		dataChan, _, errChan := h.ExecuteStreamWithAuthManager(cliCtx, h.HandlerType(), modelName, requestJSON, "")

		completedOutput, errForward := h.forwardResponsesWebsocket(c, conn, cliCancel, dataChan, errChan, &wsBodyLog, passthroughSessionID)
		if errForward != nil {
			wsTerminateErr = errForward
			appendWebsocketEvent(&wsBodyLog, "disconnect", []byte(errForward.Error()))
			log.Warnf("responses websocket: forward failed id=%s error=%v", passthroughSessionID, errForward)
			return
		}
		lastResponseOutput = completedOutput
	}
}

func websocketUpgradeHeaders(req *http.Request) http.Header {
	headers := http.Header{}
	if req == nil {
		return headers
	}

	// Keep the same sticky turn-state across reconnects when provided by the client.
	turnState := strings.TrimSpace(req.Header.Get(wsTurnStateHeader))
	if turnState != "" {
		headers.Set(wsTurnStateHeader, turnState)
	}
	return headers
}

func normalizeResponsesWebsocketRequest(rawJSON []byte, lastRequest []byte, lastResponseOutput []byte) ([]byte, []byte, *interfaces.ErrorMessage) {
	return normalizeResponsesWebsocketRequestWithMode(rawJSON, lastRequest, lastResponseOutput, true)
}

func normalizeResponsesWebsocketRequestWithMode(rawJSON []byte, lastRequest []byte, lastResponseOutput []byte, allowIncrementalInputWithPreviousResponseID bool) ([]byte, []byte, *interfaces.ErrorMessage) {
	requestType := strings.TrimSpace(jsonStringFieldBytes(rawJSON, "type"))
	switch requestType {
	case wsRequestTypeCreate:
		// log.Infof("responses websocket: response.create request")
		if len(lastRequest) == 0 {
			return normalizeResponseCreateRequest(rawJSON)
		}
		return normalizeResponseSubsequentRequest(rawJSON, lastRequest, lastResponseOutput, allowIncrementalInputWithPreviousResponseID)
	case wsRequestTypeAppend:
		// log.Infof("responses websocket: response.append request")
		return normalizeResponseSubsequentRequest(rawJSON, lastRequest, lastResponseOutput, allowIncrementalInputWithPreviousResponseID)
	default:
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("unsupported websocket request type: %s", requestType),
		}
	}
}

func normalizeResponseCreateRequest(rawJSON []byte) ([]byte, []byte, *interfaces.ErrorMessage) {
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return nil, nil, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("invalid websocket request body: %w", errParse),
		}
	}
	delete(root, "type")
	root["stream"] = true
	if !jsonutil.Exists(root, "input") {
		root["input"] = []any{}
	}

	modelName := strings.TrimSpace(jsonStringField(root, "model"))
	if modelName == "" {
		return nil, nil, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("missing model in response.create request"),
		}
	}
	normalized, errMarshal := json.Marshal(root)
	if errMarshal != nil {
		return nil, nil, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("failed to normalize websocket request: %w", errMarshal),
		}
	}
	return normalized, bytes.Clone(normalized), nil
}

func normalizeResponseSubsequentRequest(rawJSON []byte, lastRequest []byte, lastResponseOutput []byte, allowIncrementalInputWithPreviousResponseID bool) ([]byte, []byte, *interfaces.ErrorMessage) {
	if len(lastRequest) == 0 {
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("websocket request received before response.create"),
		}
	}

	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("invalid websocket request body: %w", errParse),
		}
	}
	lastRoot, errParseLast := jsonutil.ParseObjectBytes(lastRequest)
	if errParseLast != nil {
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("invalid previous websocket request: %w", errParseLast),
		}
	}

	nextInputValue, okNextInput := jsonutil.Get(root, "input")
	nextInput, okNextInputCast := nextInputValue.([]any)
	if !okNextInput || !okNextInputCast {
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("websocket request requires array field: input"),
		}
	}

	// Websocket v2 mode uses response.create with previous_response_id + incremental input.
	// Do not expand it into a full input transcript; upstream expects the incremental payload.
	if allowIncrementalInputWithPreviousResponseID {
		if prev := strings.TrimSpace(jsonStringField(root, "previous_response_id")); prev != "" {
			delete(root, "type")
			if !jsonutil.Exists(root, "model") {
				modelName := strings.TrimSpace(jsonStringField(lastRoot, "model"))
				if modelName != "" {
					root["model"] = modelName
				}
			}
			if !jsonutil.Exists(root, "instructions") {
				if instructions, okInstructions := jsonutil.Get(lastRoot, "instructions"); okInstructions {
					root["instructions"] = instructions
				}
			}
			root["stream"] = true
			normalized, errMarshal := json.Marshal(root)
			if errMarshal != nil {
				return nil, lastRequest, &interfaces.ErrorMessage{
					StatusCode: http.StatusBadRequest,
					Error:      fmt.Errorf("failed to normalize websocket request: %w", errMarshal),
				}
			}
			return normalized, bytes.Clone(normalized), nil
		}
	}

	existingInputValue, okExistingInput := jsonutil.Get(lastRoot, "input")
	existingInput, okExistingInputCast := existingInputValue.([]any)
	if !okExistingInput || !okExistingInputCast {
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("previous websocket request requires array field: input"),
		}
	}
	previousOutput, errNormalize := jsonutil.NormalizeJSONArrayBytes(lastResponseOutput)
	if errNormalize != nil {
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("invalid previous response output: %w", errNormalize),
		}
	}
	mergedInput := jsonutil.MergeArrays(existingInput, previousOutput)
	mergedInput = jsonutil.MergeArrays(mergedInput, nextInput)

	delete(root, "type")
	delete(root, "previous_response_id")
	root["input"] = mergedInput
	if !jsonutil.Exists(root, "model") {
		modelName := strings.TrimSpace(jsonStringField(lastRoot, "model"))
		if modelName != "" {
			root["model"] = modelName
		}
	}
	if !jsonutil.Exists(root, "instructions") {
		if instructions, okInstructions := jsonutil.Get(lastRoot, "instructions"); okInstructions {
			root["instructions"] = instructions
		}
	}
	root["stream"] = true
	normalized, errMarshal := json.Marshal(root)
	if errMarshal != nil {
		return nil, lastRequest, &interfaces.ErrorMessage{
			StatusCode: http.StatusBadRequest,
			Error:      fmt.Errorf("failed to merge websocket input: %w", errMarshal),
		}
	}
	return normalized, bytes.Clone(normalized), nil
}

func websocketUpstreamSupportsIncrementalInput(attributes map[string]string, metadata map[string]any) bool {
	if len(attributes) > 0 {
		if raw := strings.TrimSpace(attributes["websockets"]); raw != "" {
			parsed, errParse := strconv.ParseBool(raw)
			if errParse == nil {
				return parsed
			}
		}
	}
	if len(metadata) == 0 {
		return false
	}
	raw, ok := metadata["websockets"]
	if !ok || raw == nil {
		return false
	}
	switch value := raw.(type) {
	case bool:
		return value
	case string:
		parsed, errParse := strconv.ParseBool(strings.TrimSpace(value))
		if errParse == nil {
			return parsed
		}
	default:
	}
	return false
}

func (h *OpenAIResponsesAPIHandler) websocketUpstreamSupportsIncrementalInputForModel(modelName string) bool {
	if h == nil || h.AuthManager == nil {
		return false
	}

	resolvedModelName := modelName
	initialSuffix := thinking.ParseSuffix(modelName)
	if initialSuffix.ModelName == "auto" {
		resolvedBase := util.ResolveAutoModel(initialSuffix.ModelName)
		if initialSuffix.HasSuffix {
			resolvedModelName = fmt.Sprintf("%s(%s)", resolvedBase, initialSuffix.RawSuffix)
		} else {
			resolvedModelName = resolvedBase
		}
	} else {
		resolvedModelName = util.ResolveAutoModel(modelName)
	}

	parsed := thinking.ParseSuffix(resolvedModelName)
	baseModel := strings.TrimSpace(parsed.ModelName)
	providers := util.GetProviderName(baseModel)
	if len(providers) == 0 && baseModel != resolvedModelName {
		providers = util.GetProviderName(resolvedModelName)
	}
	if len(providers) == 0 {
		return false
	}

	providerSet := make(map[string]struct{}, len(providers))
	for i := 0; i < len(providers); i++ {
		providerKey := strings.TrimSpace(strings.ToLower(providers[i]))
		if providerKey == "" {
			continue
		}
		providerSet[providerKey] = struct{}{}
	}
	if len(providerSet) == 0 {
		return false
	}

	modelKey := baseModel
	if modelKey == "" {
		modelKey = strings.TrimSpace(resolvedModelName)
	}
	registryRef := registry.GetGlobalRegistry()
	now := time.Now()
	auths := h.AuthManager.List()
	for i := 0; i < len(auths); i++ {
		auth := auths[i]
		if auth == nil {
			continue
		}
		providerKey := strings.TrimSpace(strings.ToLower(auth.Provider))
		if _, ok := providerSet[providerKey]; !ok {
			continue
		}
		if modelKey != "" && registryRef != nil && !registryRef.ClientSupportsModel(auth.ID, modelKey) {
			continue
		}
		if !responsesWebsocketAuthAvailableForModel(auth, modelKey, now) {
			continue
		}
		if websocketUpstreamSupportsIncrementalInput(auth.Attributes, auth.Metadata) {
			return true
		}
	}
	return false
}

func responsesWebsocketAuthAvailableForModel(auth *coreauth.Auth, modelName string, now time.Time) bool {
	if auth == nil {
		return false
	}
	if auth.Disabled || auth.Status == coreauth.StatusDisabled {
		return false
	}
	if modelName != "" && len(auth.ModelStates) > 0 {
		state, ok := auth.ModelStates[modelName]
		if (!ok || state == nil) && modelName != "" {
			baseModel := strings.TrimSpace(thinking.ParseSuffix(modelName).ModelName)
			if baseModel != "" && baseModel != modelName {
				state, ok = auth.ModelStates[baseModel]
			}
		}
		if ok && state != nil {
			if state.Status == coreauth.StatusDisabled {
				return false
			}
			if state.Unavailable && !state.NextRetryAfter.IsZero() && state.NextRetryAfter.After(now) {
				return false
			}
			return true
		}
	}
	if auth.Unavailable && !auth.NextRetryAfter.IsZero() && auth.NextRetryAfter.After(now) {
		return false
	}
	return true
}

func shouldHandleResponsesWebsocketPrewarmLocally(rawJSON []byte, lastRequest []byte, allowIncrementalInputWithPreviousResponseID bool) bool {
	if allowIncrementalInputWithPreviousResponseID || len(lastRequest) != 0 {
		return false
	}
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return false
	}
	if strings.TrimSpace(jsonStringField(root, "type")) != wsRequestTypeCreate {
		return false
	}
	generateValue, okGenerate := jsonutil.Get(root, "generate")
	if !okGenerate {
		return false
	}
	generate, okGenerateCast := generateValue.(bool)
	return okGenerateCast && !generate
}

func writeResponsesWebsocketSyntheticPrewarm(
	c *gin.Context,
	conn *websocket.Conn,
	requestJSON []byte,
	wsBodyLog *strings.Builder,
	sessionID string,
) error {
	payloads, errPayloads := syntheticResponsesWebsocketPrewarmPayloads(requestJSON)
	if errPayloads != nil {
		return errPayloads
	}
	for i := 0; i < len(payloads); i++ {
		markAPIResponseTimestamp(c)
		appendWebsocketEvent(wsBodyLog, "response", payloads[i])
		// log.Infof(
		// 	"responses websocket: downstream_out id=%s type=%d event=%s payload=%s",
		// 	sessionID,
		// 	websocket.TextMessage,
		// 	websocketPayloadEventType(payloads[i]),
		// 	websocketPayloadPreview(payloads[i]),
		// )
		if errWrite := conn.WriteMessage(websocket.TextMessage, payloads[i]); errWrite != nil {
			log.Warnf(
				"responses websocket: downstream_out write failed id=%s event=%s error=%v",
				sessionID,
				websocketPayloadEventType(payloads[i]),
				errWrite,
			)
			return errWrite
		}
	}
	return nil
}

func syntheticResponsesWebsocketPrewarmPayloads(requestJSON []byte) ([][]byte, error) {
	responseID := "resp_prewarm_" + uuid.NewString()
	createdAt := time.Now().Unix()
	modelName := strings.TrimSpace(jsonStringFieldBytes(requestJSON, "model"))

	createdPayloadValue := map[string]any{
		"type":            "response.created",
		"sequence_number": 0,
		"response": map[string]any{
			"id":         responseID,
			"object":     "response",
			"created_at": createdAt,
			"status":     "in_progress",
			"background": false,
			"error":      nil,
			"output":     []any{},
		},
	}
	if modelName != "" {
		createdPayloadValue["response"].(map[string]any)["model"] = modelName
	}
	createdPayload, errMarshalCreated := json.Marshal(createdPayloadValue)
	if errMarshalCreated != nil {
		return nil, errMarshalCreated
	}

	completedPayloadValue := map[string]any{
		"type":            "response.completed",
		"sequence_number": 1,
		"response": map[string]any{
			"id":         responseID,
			"object":     "response",
			"created_at": createdAt,
			"status":     "completed",
			"background": false,
			"error":      nil,
			"output":     []any{},
			"usage": map[string]any{
				"input_tokens":  0,
				"output_tokens": 0,
				"total_tokens":  0,
			},
		},
	}
	if modelName != "" {
		completedPayloadValue["response"].(map[string]any)["model"] = modelName
	}
	completedPayload, errMarshalCompleted := json.Marshal(completedPayloadValue)
	if errMarshalCompleted != nil {
		return nil, errMarshalCompleted
	}

	return [][]byte{createdPayload, completedPayload}, nil
}

func mergeJSONArrayRaw(existingRaw, appendRaw string) (string, error) {
	existingRaw = strings.TrimSpace(existingRaw)
	appendRaw = strings.TrimSpace(appendRaw)
	if existingRaw == "" {
		existingRaw = "[]"
	}
	if appendRaw == "" {
		appendRaw = "[]"
	}

	var existing []json.RawMessage
	if err := json.Unmarshal([]byte(existingRaw), &existing); err != nil {
		return "", err
	}
	var appendItems []json.RawMessage
	if err := json.Unmarshal([]byte(appendRaw), &appendItems); err != nil {
		return "", err
	}

	merged := append(existing, appendItems...)
	out, err := json.Marshal(merged)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func normalizeJSONArrayRaw(raw []byte) string {
	array, errNormalize := jsonutil.NormalizeJSONArrayBytes(raw)
	if errNormalize != nil {
		return "[]"
	}
	normalized, errMarshal := json.Marshal(array)
	if errMarshal != nil {
		return "[]"
	}
	return string(normalized)
}

func (h *OpenAIResponsesAPIHandler) forwardResponsesWebsocket(
	c *gin.Context,
	conn *websocket.Conn,
	cancel handlers.APIHandlerCancelFunc,
	data <-chan []byte,
	errs <-chan *interfaces.ErrorMessage,
	wsBodyLog *strings.Builder,
	sessionID string,
) ([]byte, error) {
	completed := false
	completedOutput := []byte("[]")

	for {
		select {
		case <-c.Request.Context().Done():
			cancel(c.Request.Context().Err())
			return completedOutput, c.Request.Context().Err()
		case errMsg, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			if errMsg != nil {
				h.LoggingAPIResponseError(context.WithValue(context.Background(), "gin", c), errMsg)
				markAPIResponseTimestamp(c)
				errorPayload, errWrite := writeResponsesWebsocketError(conn, errMsg)
				appendWebsocketEvent(wsBodyLog, "response", errorPayload)
				log.Infof(
					"responses websocket: downstream_out id=%s type=%d event=%s payload=%s",
					sessionID,
					websocket.TextMessage,
					websocketPayloadEventType(errorPayload),
					websocketPayloadPreview(errorPayload),
				)
				if errWrite != nil {
					// log.Warnf(
					// 	"responses websocket: downstream_out write failed id=%s event=%s error=%v",
					// 	sessionID,
					// 	websocketPayloadEventType(errorPayload),
					// 	errWrite,
					// )
					cancel(errMsg.Error)
					return completedOutput, errWrite
				}
			}
			if errMsg != nil {
				cancel(errMsg.Error)
			} else {
				cancel(nil)
			}
			return completedOutput, nil
		case chunk, ok := <-data:
			if !ok {
				if !completed {
					errMsg := &interfaces.ErrorMessage{
						StatusCode: http.StatusRequestTimeout,
						Error:      fmt.Errorf("stream closed before response.completed"),
					}
					h.LoggingAPIResponseError(context.WithValue(context.Background(), "gin", c), errMsg)
					markAPIResponseTimestamp(c)
					errorPayload, errWrite := writeResponsesWebsocketError(conn, errMsg)
					appendWebsocketEvent(wsBodyLog, "response", errorPayload)
					log.Infof(
						"responses websocket: downstream_out id=%s type=%d event=%s payload=%s",
						sessionID,
						websocket.TextMessage,
						websocketPayloadEventType(errorPayload),
						websocketPayloadPreview(errorPayload),
					)
					if errWrite != nil {
						log.Warnf(
							"responses websocket: downstream_out write failed id=%s event=%s error=%v",
							sessionID,
							websocketPayloadEventType(errorPayload),
							errWrite,
						)
						cancel(errMsg.Error)
						return completedOutput, errWrite
					}
					cancel(errMsg.Error)
					return completedOutput, nil
				}
				cancel(nil)
				return completedOutput, nil
			}

			payloads := websocketJSONPayloadsFromChunk(chunk)
			for i := range payloads {
				payloadRoot, _ := jsonutil.ParseObjectBytes(payloads[i])
				eventType := jsonStringField(payloadRoot, "type")
				if eventType == wsEventTypeCompleted {
					completed = true
					completedOutput = responseCompletedOutputFromObject(payloadRoot)
				}
				markAPIResponseTimestamp(c)
				appendWebsocketEvent(wsBodyLog, "response", payloads[i])
				// log.Infof(
				// 	"responses websocket: downstream_out id=%s type=%d event=%s payload=%s",
				// 	sessionID,
				// 	websocket.TextMessage,
				// 	websocketPayloadEventType(payloads[i]),
				// 	websocketPayloadPreview(payloads[i]),
				// )
				if errWrite := conn.WriteMessage(websocket.TextMessage, payloads[i]); errWrite != nil {
					log.Warnf(
						"responses websocket: downstream_out write failed id=%s event=%s error=%v",
						sessionID,
						websocketPayloadEventType(payloads[i]),
						errWrite,
					)
					cancel(errWrite)
					return completedOutput, errWrite
				}
			}
		}
	}
}

func responseCompletedOutputFromPayload(payload []byte) []byte {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return []byte("[]")
	}
	return responseCompletedOutputFromObject(root)
}

func websocketJSONPayloadsFromChunk(chunk []byte) [][]byte {
	payloads := make([][]byte, 0, 2)
	lines := bytes.Split(chunk, []byte("\n"))
	for i := range lines {
		line := bytes.TrimSpace(lines[i])
		if len(line) == 0 || bytes.HasPrefix(line, []byte("event:")) {
			continue
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			line = bytes.TrimSpace(line[len("data:"):])
		}
		if len(line) == 0 || bytes.Equal(line, []byte(wsDoneMarker)) {
			continue
		}
		if json.Valid(line) {
			payloads = append(payloads, bytes.Clone(line))
		}
	}

	if len(payloads) > 0 {
		return payloads
	}

	trimmed := bytes.TrimSpace(chunk)
	if bytes.HasPrefix(trimmed, []byte("data:")) {
		trimmed = bytes.TrimSpace(trimmed[len("data:"):])
	}
	if len(trimmed) > 0 && !bytes.Equal(trimmed, []byte(wsDoneMarker)) && json.Valid(trimmed) {
		payloads = append(payloads, bytes.Clone(trimmed))
	}
	return payloads
}

func writeResponsesWebsocketError(conn *websocket.Conn, errMsg *interfaces.ErrorMessage) ([]byte, error) {
	status := http.StatusInternalServerError
	errText := http.StatusText(status)
	if errMsg != nil {
		if errMsg.StatusCode > 0 {
			status = errMsg.StatusCode
			errText = http.StatusText(status)
		}
		if errMsg.Error != nil && strings.TrimSpace(errMsg.Error.Error()) != "" {
			errText = errMsg.Error.Error()
		}
	}

	body := handlers.BuildErrorResponseBody(status, errText)
	payloadValue := map[string]any{
		"type":   wsEventTypeError,
		"status": status,
	}

	if errMsg != nil && errMsg.Addon != nil {
		headers := make(map[string]any)
		hasHeaders := false
		for key, values := range errMsg.Addon {
			if len(values) == 0 {
				continue
			}
			headers[key] = values[0]
			hasHeaders = true
		}
		if hasHeaders {
			payloadValue["headers"] = headers
		}
	}

	if len(body) > 0 && json.Valid(body) {
		bodyRoot, errParse := jsonutil.ParseObjectBytes(body)
		if errParse == nil {
			if errorNode, okError := bodyRoot["error"]; okError {
				payloadValue["error"] = errorNode
			} else {
				payloadValue["error"] = bodyRoot
			}
		}
	}

	if _, ok := payloadValue["error"]; !ok {
		payloadValue["error"] = map[string]any{
			"type":    "server_error",
			"message": errText,
		}
	}

	payload, errMarshal := json.Marshal(payloadValue)
	if errMarshal != nil {
		return nil, errMarshal
	}
	return payload, conn.WriteMessage(websocket.TextMessage, payload)
}

func appendWebsocketEvent(builder *strings.Builder, eventType string, payload []byte) {
	if builder == nil {
		return
	}
	if builder.Len() >= wsBodyLogMaxSize {
		return
	}
	trimmedPayload := bytes.TrimSpace(payload)
	if len(trimmedPayload) == 0 {
		return
	}
	if builder.Len() > 0 {
		if !appendWebsocketLogString(builder, "\n") {
			return
		}
	}
	if !appendWebsocketLogString(builder, "websocket.") {
		return
	}
	if !appendWebsocketLogString(builder, eventType) {
		return
	}
	if !appendWebsocketLogString(builder, "\n") {
		return
	}
	if !appendWebsocketLogBytes(builder, trimmedPayload, len(wsBodyLogTruncated)) {
		appendWebsocketLogString(builder, wsBodyLogTruncated)
		return
	}
	appendWebsocketLogString(builder, "\n")
}

func appendWebsocketLogString(builder *strings.Builder, value string) bool {
	if builder == nil {
		return false
	}
	remaining := wsBodyLogMaxSize - builder.Len()
	if remaining <= 0 {
		return false
	}
	if len(value) <= remaining {
		builder.WriteString(value)
		return true
	}
	builder.WriteString(value[:remaining])
	return false
}

func appendWebsocketLogBytes(builder *strings.Builder, value []byte, reserveForSuffix int) bool {
	if builder == nil {
		return false
	}
	remaining := wsBodyLogMaxSize - builder.Len()
	if remaining <= 0 {
		return false
	}
	if len(value) <= remaining {
		builder.Write(value)
		return true
	}
	limit := remaining - reserveForSuffix
	if limit < 0 {
		limit = 0
	}
	if limit > len(value) {
		limit = len(value)
	}
	builder.Write(value[:limit])
	return false
}

func websocketPayloadEventType(payload []byte) string {
	eventType := strings.TrimSpace(jsonStringFieldBytes(payload, "type"))
	if eventType == "" {
		return "-"
	}
	return eventType
}

func responseCompletedOutputFromObject(root map[string]any) []byte {
	if root == nil {
		return []byte("[]")
	}
	output, ok := jsonutil.Get(root, "response.output")
	if !ok {
		return []byte("[]")
	}
	array, ok := output.([]any)
	if !ok {
		return []byte("[]")
	}
	out, errMarshal := json.Marshal(array)
	if errMarshal != nil {
		return []byte("[]")
	}
	return out
}

func jsonStringField(root map[string]any, path string) string {
	if root == nil {
		return ""
	}
	value, ok := jsonutil.Get(root, path)
	if !ok || value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return typed
	case json.Number:
		return typed.String()
	default:
		return fmt.Sprint(typed)
	}
}

func jsonBoolField(root map[string]any, path string) bool {
	if root == nil {
		return false
	}
	value, ok := jsonutil.Get(root, path)
	if !ok {
		return false
	}
	typed, ok := value.(bool)
	return ok && typed
}

func jsonStringFieldBytes(payload []byte, path string) string {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return ""
	}
	return jsonStringField(root, path)
}

func jsonBoolFieldBytes(payload []byte, path string) bool {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return false
	}
	return jsonBoolField(root, path)
}

func deleteJSONPathBytes(payload []byte, path string) []byte {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return payload
	}
	if errDelete := jsonutil.Delete(root, path); errDelete != nil {
		return payload
	}
	return jsonutil.MarshalOrOriginal(payload, root)
}

func websocketPayloadPreview(payload []byte) string {
	trimmedPayload := bytes.TrimSpace(payload)
	if len(trimmedPayload) == 0 {
		return "<empty>"
	}
	preview := trimmedPayload
	if len(preview) > wsPayloadLogMaxSize {
		preview = preview[:wsPayloadLogMaxSize]
	}
	previewText := strings.ReplaceAll(string(preview), "\n", "\\n")
	previewText = strings.ReplaceAll(previewText, "\r", "\\r")
	if len(trimmedPayload) > wsPayloadLogMaxSize {
		return fmt.Sprintf("%s...(truncated,total=%d)", previewText, len(trimmedPayload))
	}
	return previewText
}

func setWebsocketRequestBody(c *gin.Context, body string) {
	if c == nil {
		return
	}
	trimmedBody := strings.TrimSpace(body)
	if trimmedBody == "" {
		return
	}
	c.Set(wsRequestBodyKey, []byte(trimmedBody))
}

func markAPIResponseTimestamp(c *gin.Context) {
	if c == nil {
		return
	}
	if _, exists := c.Get("API_RESPONSE_TIMESTAMP"); exists {
		return
	}
	c.Set("API_RESPONSE_TIMESTAMP", time.Now())
}
