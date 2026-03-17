package responses

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

type geminiToResponsesState struct {
	Seq        int
	ResponseID string
	CreatedAt  int64
	Started    bool

	// message aggregation
	MsgOpened    bool
	MsgClosed    bool
	MsgIndex     int
	CurrentMsgID string
	TextBuf      strings.Builder
	ItemTextBuf  strings.Builder

	// reasoning aggregation
	ReasoningOpened bool
	ReasoningIndex  int
	ReasoningItemID string
	ReasoningEnc    string
	ReasoningBuf    strings.Builder
	ReasoningClosed bool

	// function call aggregation (keyed by output_index)
	NextIndex   int
	FuncArgsBuf map[int]*strings.Builder
	FuncNames   map[int]string
	FuncCallIDs map[int]string
	FuncDone    map[int]bool
}

// responseIDCounter provides a process-wide unique counter for synthesized response identifiers.
var responseIDCounter uint64

// funcCallIDCounter provides a process-wide unique counter for function call identifiers.
var funcCallIDCounter uint64

func pickRequestJSON(originalRequestRawJSON, requestRawJSON []byte) []byte {
	if len(originalRequestRawJSON) > 0 && json.Valid(originalRequestRawJSON) {
		return originalRequestRawJSON
	}
	if len(requestRawJSON) > 0 && json.Valid(requestRawJSON) {
		return requestRawJSON
	}
	return nil
}

func unwrapRequestRoot(root map[string]any) map[string]any {
	request, ok := jsonutil.Object(root, "request")
	if !ok {
		return root
	}
	if jsonutil.Exists(request, "model") || jsonutil.Exists(request, "input") || jsonutil.Exists(request, "instructions") {
		return request
	}
	return root
}

func unwrapGeminiResponseRoot(root map[string]any) map[string]any {
	response, ok := jsonutil.Object(root, "response")
	if !ok {
		return root
	}
	if jsonutil.Exists(response, "candidates") || jsonutil.Exists(response, "responseId") || jsonutil.Exists(response, "usageMetadata") {
		return response
	}
	return root
}

func emitEvent(event string, payload string) string {
	return fmt.Sprintf("event: %s\ndata: %s", event, payload)
}

func emitJSONEvent(event string, payload any) string {
	return emitEvent(event, string(jsonutil.MarshalOrOriginal(nil, payload)))
}

func responseMessageItemID(responseID string, index int) string {
	return fmt.Sprintf("msg_%s_%d", responseID, index)
}

func responseReasoningItemID(responseID string, index int) string {
	return fmt.Sprintf("rs_%s_%d", responseID, index)
}

func responseFunctionItemID(callID string) string {
	return fmt.Sprintf("fc_%s", callID)
}

func outputTextPart(text string) map[string]any {
	return map[string]any{
		"type":        "output_text",
		"annotations": []any{},
		"logprobs":    []any{},
		"text":        text,
	}
}

func summaryTextPart(text string) map[string]any {
	return map[string]any{
		"type": "summary_text",
		"text": text,
	}
}

func assistantMessageOutputItemInProgress(itemID string) map[string]any {
	return map[string]any{
		"id":      itemID,
		"type":    "message",
		"status":  "in_progress",
		"content": []any{},
		"role":    "assistant",
	}
}

func assistantMessageOutputItemCompleted(itemID string, text string) map[string]any {
	return map[string]any{
		"id":     itemID,
		"type":   "message",
		"status": "completed",
		"content": []any{
			outputTextPart(text),
		},
		"role": "assistant",
	}
}

func reasoningOutputItemInProgress(itemID string, encryptedContent string) map[string]any {
	return map[string]any{
		"id":                itemID,
		"type":              "reasoning",
		"status":            "in_progress",
		"encrypted_content": encryptedContent,
		"summary":           []any{},
	}
}

func reasoningOutputItemCompleted(itemID string, text string, encryptedContent string) map[string]any {
	return map[string]any{
		"id":                itemID,
		"type":              "reasoning",
		"status":            "completed",
		"encrypted_content": encryptedContent,
		"summary": []any{
			summaryTextPart(text),
		},
	}
}

func reasoningOutputItemNonStream(itemID string, text string, encryptedContent string) map[string]any {
	item := map[string]any{
		"id":                itemID,
		"type":              "reasoning",
		"encrypted_content": encryptedContent,
		"summary":           []any{},
	}
	if text != "" {
		item["summary"] = []any{summaryTextPart(text)}
	}
	return item
}

func functionCallOutputItem(callID string, name string, args string, status string) map[string]any {
	return map[string]any{
		"id":        responseFunctionItemID(callID),
		"type":      "function_call",
		"status":    status,
		"arguments": args,
		"call_id":   callID,
		"name":      name,
	}
}

func normalizeGeminiRequestInput(rawJSON []byte) map[string]any {
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return nil
	}
	return unwrapRequestRoot(root)
}

func normalizeGeminiResponseInput(rawJSON []byte) map[string]any {
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return nil
	}
	return unwrapGeminiResponseRoot(root)
}

func jsonStringField(root map[string]any, path string) string {
	value, ok := jsonutil.String(root, path)
	if !ok {
		return ""
	}
	return value
}

func normalizeResponseID(id string) string {
	if id == "" {
		id = fmt.Sprintf("resp_%x_%d", time.Now().UnixNano(), atomic.AddUint64(&responseIDCounter, 1))
	}
	if !strings.HasPrefix(id, "resp_") {
		id = fmt.Sprintf("resp_%s", id)
	}
	return id
}

func responseCreatedAt(root map[string]any) int64 {
	createdAt := time.Now().Unix()
	if value, ok := jsonutil.String(root, "createTime"); ok {
		if parsed, errParseCreateTime := time.Parse(time.RFC3339Nano, value); errParseCreateTime == nil {
			createdAt = parsed.Unix()
		}
	}
	return createdAt
}

func applyGeminiResponsesRequestFields(response map[string]any, requestRoot map[string]any, responseRoot map[string]any) {
	if requestRoot != nil {
		copyField := func(source string, target string) {
			if value, ok := jsonutil.Get(requestRoot, source); ok {
				_ = jsonutil.Set(response, target, value)
			}
		}

		copyField("instructions", "instructions")
		copyField("max_output_tokens", "max_output_tokens")
		copyField("max_tool_calls", "max_tool_calls")
		copyField("parallel_tool_calls", "parallel_tool_calls")
		copyField("previous_response_id", "previous_response_id")
		copyField("prompt_cache_key", "prompt_cache_key")
		copyField("reasoning", "reasoning")
		copyField("safety_identifier", "safety_identifier")
		copyField("service_tier", "service_tier")
		copyField("store", "store")
		copyField("temperature", "temperature")
		copyField("text", "text")
		copyField("tool_choice", "tool_choice")
		copyField("tools", "tools")
		copyField("top_logprobs", "top_logprobs")
		copyField("top_p", "top_p")
		copyField("truncation", "truncation")
		copyField("user", "user")
		copyField("metadata", "metadata")

		if model, ok := jsonutil.String(requestRoot, "model"); ok && model != "" {
			response["model"] = model
		}
	}

	if _, ok := response["model"]; !ok && responseRoot != nil {
		if model, ok := jsonutil.String(responseRoot, "modelVersion"); ok && model != "" {
			response["model"] = model
		}
	}
}

func appendGeminiUsage(response map[string]any, root map[string]any) {
	usageMetadata, ok := jsonutil.Object(root, "usageMetadata")
	if !ok {
		return
	}

	inputTokens, _ := jsonutil.Int64(usageMetadata, "promptTokenCount")
	cachedTokens, _ := jsonutil.Int64(usageMetadata, "cachedContentTokenCount")
	outputTokens, _ := jsonutil.Int64(usageMetadata, "candidatesTokenCount")
	reasoningTokens, _ := jsonutil.Int64(usageMetadata, "thoughtsTokenCount")
	totalTokens, _ := jsonutil.Int64(usageMetadata, "totalTokenCount")

	response["usage"] = map[string]any{
		"input_tokens": inputTokens,
		"input_tokens_details": map[string]any{
			"cached_tokens": cachedTokens,
		},
		"output_tokens": outputTokens,
		"output_tokens_details": map[string]any{
			"reasoning_tokens": reasoningTokens,
		},
		"total_tokens": totalTokens,
	}
}

func geminiFunctionCallArgsJSON(functionCall map[string]any) string {
	argsValue, ok := jsonutil.Get(functionCall, "args")
	if !ok {
		return "{}"
	}
	argsJSON, errMarshal := jsonutil.MarshalAny(argsValue)
	if errMarshal != nil {
		return "{}"
	}
	return string(argsJSON)
}

// ConvertGeminiResponseToOpenAIResponses converts Gemini SSE chunks into OpenAI Responses SSE events.
func ConvertGeminiResponseToOpenAIResponses(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &geminiToResponsesState{
			FuncArgsBuf: make(map[int]*strings.Builder),
			FuncNames:   make(map[int]string),
			FuncCallIDs: make(map[int]string),
			FuncDone:    make(map[int]bool),
		}
	}
	st := (*param).(*geminiToResponsesState)
	if st.FuncArgsBuf == nil {
		st.FuncArgsBuf = make(map[int]*strings.Builder)
	}
	if st.FuncNames == nil {
		st.FuncNames = make(map[int]string)
	}
	if st.FuncCallIDs == nil {
		st.FuncCallIDs = make(map[int]string)
	}
	if st.FuncDone == nil {
		st.FuncDone = make(map[int]bool)
	}

	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}

	rawJSON = bytes.TrimSpace(rawJSON)
	if len(rawJSON) == 0 || bytes.Equal(rawJSON, []byte("[DONE]")) {
		return []string{}
	}

	root := normalizeGeminiResponseInput(rawJSON)
	if root == nil {
		return []string{}
	}

	var out []string
	nextSeq := func() int {
		st.Seq++
		return st.Seq
	}

	finalizeReasoning := func() {
		if !st.ReasoningOpened || st.ReasoningClosed {
			return
		}
		full := st.ReasoningBuf.String()

		out = append(out, emitJSONEvent("response.reasoning_summary_text.done", map[string]any{
			"type":            "response.reasoning_summary_text.done",
			"sequence_number": nextSeq(),
			"item_id":         st.ReasoningItemID,
			"output_index":    st.ReasoningIndex,
			"summary_index":   0,
			"text":            full,
		}))

		out = append(out, emitJSONEvent("response.reasoning_summary_part.done", map[string]any{
			"type":            "response.reasoning_summary_part.done",
			"sequence_number": nextSeq(),
			"item_id":         st.ReasoningItemID,
			"output_index":    st.ReasoningIndex,
			"summary_index":   0,
			"part":            summaryTextPart(full),
		}))

		out = append(out, emitJSONEvent("response.output_item.done", map[string]any{
			"type":            "response.output_item.done",
			"sequence_number": nextSeq(),
			"output_index":    st.ReasoningIndex,
			"item":            reasoningOutputItemCompleted(st.ReasoningItemID, full, st.ReasoningEnc),
		}))

		st.ReasoningClosed = true
	}

	finalizeMessage := func() {
		if !st.MsgOpened || st.MsgClosed {
			return
		}
		fullText := st.ItemTextBuf.String()

		out = append(out, emitJSONEvent("response.output_text.done", map[string]any{
			"type":            "response.output_text.done",
			"sequence_number": nextSeq(),
			"item_id":         st.CurrentMsgID,
			"output_index":    st.MsgIndex,
			"content_index":   0,
			"text":            fullText,
			"logprobs":        []any{},
		}))

		out = append(out, emitJSONEvent("response.content_part.done", map[string]any{
			"type":            "response.content_part.done",
			"sequence_number": nextSeq(),
			"item_id":         st.CurrentMsgID,
			"output_index":    st.MsgIndex,
			"content_index":   0,
			"part":            outputTextPart(fullText),
		}))

		out = append(out, emitJSONEvent("response.output_item.done", map[string]any{
			"type":            "response.output_item.done",
			"sequence_number": nextSeq(),
			"output_index":    st.MsgIndex,
			"item":            assistantMessageOutputItemCompleted(st.CurrentMsgID, fullText),
		}))

		st.MsgClosed = true
	}

	if !st.Started {
		st.ResponseID = normalizeResponseID(jsonStringField(root, "responseId"))
		st.CreatedAt = responseCreatedAt(root)

		out = append(out, emitJSONEvent("response.created", map[string]any{
			"type":            "response.created",
			"sequence_number": nextSeq(),
			"response": map[string]any{
				"id":         st.ResponseID,
				"object":     "response",
				"created_at": st.CreatedAt,
				"status":     "in_progress",
				"background": false,
				"error":      nil,
				"output":     []any{},
			},
		}))

		out = append(out, emitJSONEvent("response.in_progress", map[string]any{
			"type":            "response.in_progress",
			"sequence_number": nextSeq(),
			"response": map[string]any{
				"id":         st.ResponseID,
				"object":     "response",
				"created_at": st.CreatedAt,
				"status":     "in_progress",
			},
		}))

		st.Started = true
		st.NextIndex = 0
	}

	if parts, ok := jsonutil.Array(root, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
				if st.ReasoningClosed {
					continue
				}
				if signature, ok := jsonutil.String(part, "thoughtSignature"); ok && signature != "" && signature != geminiResponsesThoughtSignature {
					st.ReasoningEnc = signature
				} else if signature, ok := jsonutil.String(part, "thought_signature"); ok && signature != "" && signature != geminiResponsesThoughtSignature {
					st.ReasoningEnc = signature
				}
				if !st.ReasoningOpened {
					st.ReasoningOpened = true
					st.ReasoningIndex = st.NextIndex
					st.NextIndex++
					st.ReasoningItemID = responseReasoningItemID(st.ResponseID, st.ReasoningIndex)

					out = append(out, emitJSONEvent("response.output_item.added", map[string]any{
						"type":            "response.output_item.added",
						"sequence_number": nextSeq(),
						"output_index":    st.ReasoningIndex,
						"item":            reasoningOutputItemInProgress(st.ReasoningItemID, st.ReasoningEnc),
					}))

					out = append(out, emitJSONEvent("response.reasoning_summary_part.added", map[string]any{
						"type":            "response.reasoning_summary_part.added",
						"sequence_number": nextSeq(),
						"item_id":         st.ReasoningItemID,
						"output_index":    st.ReasoningIndex,
						"summary_index":   0,
						"part":            summaryTextPart(""),
					}))
				}
				if text, ok := jsonutil.String(part, "text"); ok && text != "" {
					st.ReasoningBuf.WriteString(text)
					out = append(out, emitJSONEvent("response.reasoning_summary_text.delta", map[string]any{
						"type":            "response.reasoning_summary_text.delta",
						"sequence_number": nextSeq(),
						"item_id":         st.ReasoningItemID,
						"output_index":    st.ReasoningIndex,
						"summary_index":   0,
						"delta":           text,
					}))
				}
				continue
			}

			if text, ok := jsonutil.String(part, "text"); ok && text != "" {
				finalizeReasoning()
				if !st.MsgOpened {
					st.MsgOpened = true
					st.MsgIndex = st.NextIndex
					st.NextIndex++
					st.CurrentMsgID = responseMessageItemID(st.ResponseID, 0)
					st.ItemTextBuf.Reset()

					out = append(out, emitJSONEvent("response.output_item.added", map[string]any{
						"type":            "response.output_item.added",
						"sequence_number": nextSeq(),
						"output_index":    st.MsgIndex,
						"item":            assistantMessageOutputItemInProgress(st.CurrentMsgID),
					}))

					out = append(out, emitJSONEvent("response.content_part.added", map[string]any{
						"type":            "response.content_part.added",
						"sequence_number": nextSeq(),
						"item_id":         st.CurrentMsgID,
						"output_index":    st.MsgIndex,
						"content_index":   0,
						"part":            outputTextPart(""),
					}))
				}
				st.TextBuf.WriteString(text)
				st.ItemTextBuf.WriteString(text)

				out = append(out, emitJSONEvent("response.output_text.delta", map[string]any{
					"type":            "response.output_text.delta",
					"sequence_number": nextSeq(),
					"item_id":         st.CurrentMsgID,
					"output_index":    st.MsgIndex,
					"content_index":   0,
					"delta":           text,
					"logprobs":        []any{},
				}))
				continue
			}

			functionCall, ok := jsonutil.Object(part, "functionCall")
			if !ok {
				continue
			}

			finalizeReasoning()
			finalizeMessage()

			name, _ := jsonutil.String(functionCall, "name")
			index := st.NextIndex
			st.NextIndex++

			if st.FuncArgsBuf[index] == nil {
				st.FuncArgsBuf[index] = &strings.Builder{}
			}
			if st.FuncCallIDs[index] == "" {
				st.FuncCallIDs[index] = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), atomic.AddUint64(&funcCallIDCounter, 1))
			}
			st.FuncNames[index] = name

			argsJSON := geminiFunctionCallArgsJSON(functionCall)
			if st.FuncArgsBuf[index].Len() == 0 && argsJSON != "" {
				st.FuncArgsBuf[index].WriteString(argsJSON)
			}

			callID := st.FuncCallIDs[index]
			itemID := responseFunctionItemID(callID)

			out = append(out, emitJSONEvent("response.output_item.added", map[string]any{
				"type":            "response.output_item.added",
				"sequence_number": nextSeq(),
				"output_index":    index,
				"item":            functionCallOutputItem(callID, name, "", "in_progress"),
			}))

			out = append(out, emitJSONEvent("response.function_call_arguments.delta", map[string]any{
				"type":            "response.function_call_arguments.delta",
				"sequence_number": nextSeq(),
				"item_id":         itemID,
				"output_index":    index,
				"delta":           argsJSON,
			}))

			if !st.FuncDone[index] {
				out = append(out, emitJSONEvent("response.function_call_arguments.done", map[string]any{
					"type":            "response.function_call_arguments.done",
					"sequence_number": nextSeq(),
					"item_id":         itemID,
					"output_index":    index,
					"arguments":       argsJSON,
				}))

				out = append(out, emitJSONEvent("response.output_item.done", map[string]any{
					"type":            "response.output_item.done",
					"sequence_number": nextSeq(),
					"output_index":    index,
					"item":            functionCallOutputItem(callID, st.FuncNames[index], argsJSON, "completed"),
				}))

				st.FuncDone[index] = true
			}
		}
	}

	if finishReason, ok := jsonutil.String(root, "candidates.0.finishReason"); ok && finishReason != "" {
		finalizeReasoning()
		finalizeMessage()

		if len(st.FuncArgsBuf) > 0 {
			indices := make([]int, 0, len(st.FuncArgsBuf))
			for index := range st.FuncArgsBuf {
				indices = append(indices, index)
			}
			sort.Ints(indices)
			for _, index := range indices {
				if st.FuncDone[index] {
					continue
				}
				args := "{}"
				if buffer := st.FuncArgsBuf[index]; buffer != nil && buffer.Len() > 0 {
					args = buffer.String()
				}

				callID := st.FuncCallIDs[index]
				itemID := responseFunctionItemID(callID)

				out = append(out, emitJSONEvent("response.function_call_arguments.done", map[string]any{
					"type":            "response.function_call_arguments.done",
					"sequence_number": nextSeq(),
					"item_id":         itemID,
					"output_index":    index,
					"arguments":       args,
				}))

				out = append(out, emitJSONEvent("response.output_item.done", map[string]any{
					"type":            "response.output_item.done",
					"sequence_number": nextSeq(),
					"output_index":    index,
					"item":            functionCallOutputItem(callID, st.FuncNames[index], args, "completed"),
				}))

				st.FuncDone[index] = true
			}
		}

		response := map[string]any{
			"id":                 st.ResponseID,
			"object":             "response",
			"created_at":         st.CreatedAt,
			"status":             "completed",
			"background":         false,
			"error":              nil,
			"incomplete_details": nil,
		}

		requestRoot := normalizeGeminiRequestInput(pickRequestJSON(originalRequestRawJSON, requestRawJSON))
		applyGeminiResponsesRequestFields(response, requestRoot, root)

		output := make([]any, 0, st.NextIndex)
		for index := 0; index < st.NextIndex; index++ {
			if st.ReasoningOpened && index == st.ReasoningIndex {
				output = append(output, reasoningOutputItemCompleted(st.ReasoningItemID, st.ReasoningBuf.String(), st.ReasoningEnc))
				continue
			}
			if st.MsgOpened && index == st.MsgIndex {
				output = append(output, assistantMessageOutputItemCompleted(st.CurrentMsgID, st.TextBuf.String()))
				continue
			}
			if callID, ok := st.FuncCallIDs[index]; ok && callID != "" {
				args := "{}"
				if buffer := st.FuncArgsBuf[index]; buffer != nil && buffer.Len() > 0 {
					args = buffer.String()
				}
				output = append(output, functionCallOutputItem(callID, st.FuncNames[index], args, "completed"))
			}
		}
		if len(output) > 0 {
			response["output"] = output
		}

		appendGeminiUsage(response, root)

		out = append(out, emitJSONEvent("response.completed", map[string]any{
			"type":            "response.completed",
			"sequence_number": nextSeq(),
			"response":        response,
		}))
	}

	return out
}

// ConvertGeminiResponseToOpenAIResponsesNonStream aggregates Gemini response JSON into a single OpenAI Responses JSON object.
func ConvertGeminiResponseToOpenAIResponsesNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	root := normalizeGeminiResponseInput(rawJSON)
	if root == nil {
		return ""
	}

	responseID := normalizeResponseID(jsonStringField(root, "responseId"))
	response := map[string]any{
		"id":                 responseID,
		"object":             "response",
		"created_at":         responseCreatedAt(root),
		"status":             "completed",
		"background":         false,
		"error":              nil,
		"incomplete_details": nil,
	}

	requestRoot := normalizeGeminiRequestInput(pickRequestJSON(originalRequestRawJSON, requestRawJSON))
	applyGeminiResponsesRequestFields(response, requestRoot, root)

	type outputSlot struct {
		index int
	}

	var (
		reasoningText      strings.Builder
		reasoningEncrypted string
		messageText        strings.Builder
		reasoningIndex     = -1
		messageIndex       = -1
		nextIndex          = 0
		output             = make([]any, 0)
	)

	if parts, ok := jsonutil.Array(root, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
				if reasoningIndex == -1 {
					reasoningIndex = nextIndex
					nextIndex++
					output = append(output, outputSlot{index: reasoningIndex})
				}
				if text, ok := jsonutil.String(part, "text"); ok {
					reasoningText.WriteString(text)
				}
				if signature, ok := jsonutil.String(part, "thoughtSignature"); ok && signature != "" && signature != geminiResponsesThoughtSignature {
					reasoningEncrypted = signature
				} else if signature, ok := jsonutil.String(part, "thought_signature"); ok && signature != "" && signature != geminiResponsesThoughtSignature {
					reasoningEncrypted = signature
				}
				continue
			}

			if text, ok := jsonutil.String(part, "text"); ok && text != "" {
				if messageIndex == -1 {
					messageIndex = nextIndex
					nextIndex++
					output = append(output, outputSlot{index: messageIndex})
				}
				messageText.WriteString(text)
				continue
			}

			functionCall, ok := jsonutil.Object(part, "functionCall")
			if !ok {
				continue
			}

			name, _ := jsonutil.String(functionCall, "name")
			callID := fmt.Sprintf("call_%x_%d", time.Now().UnixNano(), atomic.AddUint64(&funcCallIDCounter, 1))
			argsJSON := geminiFunctionCallArgsJSON(functionCall)
			output = append(output, functionCallOutputItem(callID, name, argsJSON, "completed"))
			nextIndex++
		}
	}

	if reasoningIndex != -1 {
		itemID := fmt.Sprintf("rs_%s", strings.TrimPrefix(responseID, "resp_"))
		output[reasoningIndex] = reasoningOutputItemNonStream(itemID, reasoningText.String(), reasoningEncrypted)
	}
	if messageIndex != -1 {
		itemID := responseMessageItemID(strings.TrimPrefix(responseID, "resp_"), 0)
		output[messageIndex] = assistantMessageOutputItemCompleted(itemID, messageText.String())
	}

	finalOutput := make([]any, 0, len(output))
	for _, item := range output {
		if _, ok := item.(outputSlot); ok {
			continue
		}
		finalOutput = append(finalOutput, item)
	}
	if len(finalOutput) > 0 {
		response["output"] = finalOutput
	}

	appendGeminiUsage(response, root)

	return string(jsonutil.MarshalOrOriginal(rawJSON, response))
}
