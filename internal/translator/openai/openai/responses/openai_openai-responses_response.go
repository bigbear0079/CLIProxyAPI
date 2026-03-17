package responses

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

type oaiToResponsesStateReasoning struct {
	ReasoningID   string
	ReasoningData string
}

type oaiToResponsesState struct {
	Seq            int
	ResponseID     string
	Created        int64
	Started        bool
	ReasoningID    string
	ReasoningIndex int
	// aggregation buffers for response.output
	// Per-output message text buffers by index
	MsgTextBuf   map[int]*strings.Builder
	ReasoningBuf strings.Builder
	Reasonings   []oaiToResponsesStateReasoning
	FuncArgsBuf  map[int]*strings.Builder // index -> args
	FuncNames    map[int]string           // index -> name
	FuncCallIDs  map[int]string           // index -> call_id
	// message item state per output index
	MsgItemAdded    map[int]bool // whether response.output_item.added emitted for message
	MsgContentAdded map[int]bool // whether response.content_part.added emitted for message
	MsgItemDone     map[int]bool // whether message done events were emitted
	// function item done state
	FuncArgsDone map[int]bool
	FuncItemDone map[int]bool
	// usage aggregation
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	TotalTokens      int64
	ReasoningTokens  int64
	UsageSeen        bool
}

// responseIDCounter provides a process-wide unique counter for synthesized response identifiers.
var responseIDCounter uint64

func emitRespEvent(event string, payload string) string {
	return fmt.Sprintf("event: %s\ndata: %s", event, payload)
}

func emitRespJSONEvent(event string, payload any) string {
	return emitRespEvent(event, string(jsonutil.MarshalOrOriginal(nil, payload)))
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

func reasoningOutputItemInProgress(itemID string) map[string]any {
	return map[string]any{
		"id":      itemID,
		"type":    "reasoning",
		"status":  "in_progress",
		"summary": []any{},
	}
}

func reasoningOutputItemCompleted(itemID string, text string) map[string]any {
	return map[string]any{
		"id":                itemID,
		"type":              "reasoning",
		"status":            "completed",
		"encrypted_content": "",
		"summary": []any{
			summaryTextPart(text),
		},
	}
}

func reasoningOutputItemNonStream(itemID string, text string) map[string]any {
	item := map[string]any{
		"id":                itemID,
		"type":              "reasoning",
		"encrypted_content": "",
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

func newStreamingResponsesState() *oaiToResponsesState {
	return &oaiToResponsesState{
		FuncArgsBuf:     make(map[int]*strings.Builder),
		FuncNames:       make(map[int]string),
		FuncCallIDs:     make(map[int]string),
		MsgTextBuf:      make(map[int]*strings.Builder),
		MsgItemAdded:    make(map[int]bool),
		MsgContentAdded: make(map[int]bool),
		MsgItemDone:     make(map[int]bool),
		FuncArgsDone:    make(map[int]bool),
		FuncItemDone:    make(map[int]bool),
		Reasonings:      make([]oaiToResponsesStateReasoning, 0),
	}
}

func resetStreamingResponsesState(st *oaiToResponsesState, responseID string, created int64) {
	st.ResponseID = responseID
	st.Created = created
	st.Started = true
	st.ReasoningID = ""
	st.ReasoningIndex = 0
	st.MsgTextBuf = make(map[int]*strings.Builder)
	st.ReasoningBuf.Reset()
	st.Reasonings = make([]oaiToResponsesStateReasoning, 0)
	st.FuncArgsBuf = make(map[int]*strings.Builder)
	st.FuncNames = make(map[int]string)
	st.FuncCallIDs = make(map[int]string)
	st.MsgItemAdded = make(map[int]bool)
	st.MsgContentAdded = make(map[int]bool)
	st.MsgItemDone = make(map[int]bool)
	st.FuncArgsDone = make(map[int]bool)
	st.FuncItemDone = make(map[int]bool)
	st.PromptTokens = 0
	st.CachedTokens = 0
	st.CompletionTokens = 0
	st.TotalTokens = 0
	st.ReasoningTokens = 0
	st.UsageSeen = false
}

func sortedTrueIndices(values map[int]bool) []int {
	indices := make([]int, 0, len(values))
	for index, included := range values {
		if included {
			indices = append(indices, index)
		}
	}
	sort.Ints(indices)
	return indices
}

func sortedStringIndices(values map[int]string) []int {
	indices := make([]int, 0, len(values))
	for index := range values {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	return indices
}

func updateResponsesUsageState(root map[string]any, st *oaiToResponsesState) {
	if usage, ok := jsonutil.Object(root, "usage"); ok {
		if value, ok := jsonutil.Int64(usage, "prompt_tokens"); ok {
			st.PromptTokens = value
			st.UsageSeen = true
		}
		if value, ok := jsonutil.Int64(usage, "prompt_tokens_details.cached_tokens"); ok {
			st.CachedTokens = value
			st.UsageSeen = true
		}
		if value, ok := jsonutil.Int64(usage, "completion_tokens"); ok {
			st.CompletionTokens = value
			st.UsageSeen = true
		} else if value, ok := jsonutil.Int64(usage, "output_tokens"); ok {
			st.CompletionTokens = value
			st.UsageSeen = true
		}
		if value, ok := jsonutil.Int64(usage, "output_tokens_details.reasoning_tokens"); ok {
			st.ReasoningTokens = value
			st.UsageSeen = true
		} else if value, ok := jsonutil.Int64(usage, "completion_tokens_details.reasoning_tokens"); ok {
			st.ReasoningTokens = value
			st.UsageSeen = true
		}
		if value, ok := jsonutil.Int64(usage, "total_tokens"); ok {
			st.TotalTokens = value
			st.UsageSeen = true
		}
	}
}

func applyResponsesRequestFields(response map[string]any, requestRoot map[string]any, fallbackModelRoot map[string]any, allowMaxTokensFallback bool) {
	if requestRoot != nil {
		copyField := func(source string, target string) {
			if value, ok := jsonutil.Get(requestRoot, source); ok {
				_ = jsonutil.Set(response, target, value)
			}
		}

		copyField("instructions", "instructions")
		copyField("max_output_tokens", "max_output_tokens")
		if allowMaxTokensFallback && !jsonutil.Exists(requestRoot, "max_output_tokens") {
			copyField("max_tokens", "max_output_tokens")
		}
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
			_ = jsonutil.Set(response, "model", model)
		}
	}

	if !jsonutil.Exists(response, "model") && fallbackModelRoot != nil {
		if model, ok := jsonutil.String(fallbackModelRoot, "model"); ok && model != "" {
			_ = jsonutil.Set(response, "model", model)
		}
	}
}

func buildCompletedResponseOutput(st *oaiToResponsesState) []any {
	outputs := make([]any, 0, len(st.Reasonings)+len(st.MsgItemAdded)+len(st.FuncArgsBuf))

	for _, reasoning := range st.Reasonings {
		outputs = append(outputs, reasoningOutputItemCompleted(reasoning.ReasoningID, reasoning.ReasoningData))
	}

	for _, index := range sortedTrueIndices(st.MsgItemAdded) {
		text := ""
		if builder := st.MsgTextBuf[index]; builder != nil {
			text = builder.String()
		}
		outputs = append(outputs, assistantMessageOutputItemCompleted(responseMessageItemID(st.ResponseID, index), text))
	}

	for _, index := range sortedStringIndices(st.FuncCallIDs) {
		callID := st.FuncCallIDs[index]
		if callID == "" {
			continue
		}
		args := ""
		if builder := st.FuncArgsBuf[index]; builder != nil {
			args = builder.String()
		}
		outputs = append(outputs, functionCallOutputItem(callID, st.FuncNames[index], args, "completed"))
	}

	return outputs
}

func appendCompletedUsage(response map[string]any, st *oaiToResponsesState) {
	if !st.UsageSeen {
		return
	}
	usage := map[string]any{
		"input_tokens": st.PromptTokens,
		"input_tokens_details": map[string]any{
			"cached_tokens": st.CachedTokens,
		},
		"output_tokens": st.CompletionTokens,
		"total_tokens":  st.TotalTokens,
	}
	if st.ReasoningTokens > 0 {
		usage["output_tokens_details"] = map[string]any{
			"reasoning_tokens": st.ReasoningTokens,
		}
	}
	if st.TotalTokens == 0 {
		usage["total_tokens"] = st.PromptTokens + st.CompletionTokens
	}
	response["usage"] = usage
}

func messageChoiceText(choice map[string]any) string {
	if delta, ok := jsonutil.Object(choice, "delta"); ok {
		if content, ok := jsonutil.String(delta, "content"); ok {
			return content
		}
	}
	if message, ok := jsonutil.Object(choice, "message"); ok {
		if content, ok := jsonutil.String(message, "content"); ok {
			return content
		}
	}
	return ""
}

func choiceIndex(choice map[string]any) int {
	index, ok := jsonutil.Int64(choice, "index")
	if !ok {
		return 0
	}
	return int(index)
}

func normalizeOpenAIResponsesInput(rawJSON []byte) map[string]any {
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return nil
	}
	return root
}

func jsonStringField(root map[string]any, path string) string {
	value, ok := jsonutil.String(root, path)
	if !ok {
		return ""
	}
	return value
}

// ConvertOpenAIChatCompletionsResponseToOpenAIResponses converts OpenAI Chat Completions streaming chunks
// to OpenAI Responses SSE events (response.*).
func ConvertOpenAIChatCompletionsResponseToOpenAIResponses(_ context.Context, _ string, _ []byte, requestRawJSON []byte, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = newStreamingResponsesState()
	}
	st := (*param).(*oaiToResponsesState)

	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}

	rawJSON = bytes.TrimSpace(rawJSON)
	if len(rawJSON) == 0 || bytes.Equal(rawJSON, []byte("[DONE]")) {
		return []string{}
	}

	root := normalizeOpenAIResponsesInput(rawJSON)
	if root == nil {
		return []string{}
	}
	if objectType, ok := jsonutil.String(root, "object"); ok && objectType != "" && objectType != "chat.completion.chunk" {
		return []string{}
	}

	choices, ok := jsonutil.Array(root, "choices")
	if !ok {
		return []string{}
	}

	if !st.Started {
		resetStreamingResponsesState(
			st,
			jsonStringField(root, "id"),
			func() int64 {
				created, _ := jsonutil.Int64(root, "created")
				return created
			}(),
		)
	}
	updateResponsesUsageState(root, st)

	requestRoot := normalizeOpenAIResponsesInput(requestRawJSON)
	nextSeq := func() int {
		st.Seq++
		return st.Seq
	}

	out := make([]string, 0)
	if st.Seq == 0 {
		createdPayload := map[string]any{
			"type":            "response.created",
			"sequence_number": nextSeq(),
			"response": map[string]any{
				"id":         st.ResponseID,
				"object":     "response",
				"created_at": st.Created,
				"status":     "in_progress",
				"background": false,
				"error":      nil,
				"output":     []any{},
			},
		}
		out = append(out, emitRespJSONEvent("response.created", createdPayload))

		inProgressPayload := map[string]any{
			"type":            "response.in_progress",
			"sequence_number": nextSeq(),
			"response": map[string]any{
				"id":         st.ResponseID,
				"object":     "response",
				"created_at": st.Created,
				"status":     "in_progress",
			},
		}
		out = append(out, emitRespJSONEvent("response.in_progress", inProgressPayload))
	}

	stopReasoning := func(text string) {
		out = append(out, emitRespJSONEvent("response.reasoning_summary_text.done", map[string]any{
			"type":            "response.reasoning_summary_text.done",
			"sequence_number": nextSeq(),
			"item_id":         st.ReasoningID,
			"output_index":    st.ReasoningIndex,
			"summary_index":   0,
			"text":            text,
		}))
		out = append(out, emitRespJSONEvent("response.reasoning_summary_part.done", map[string]any{
			"type":            "response.reasoning_summary_part.done",
			"sequence_number": nextSeq(),
			"item_id":         st.ReasoningID,
			"output_index":    st.ReasoningIndex,
			"summary_index":   0,
			"part":            summaryTextPart(text),
		}))
		out = append(out, emitRespJSONEvent("response.output_item.done", map[string]any{
			"type":            "response.output_item.done",
			"sequence_number": nextSeq(),
			"output_index":    st.ReasoningIndex,
			"item":            reasoningOutputItemCompleted(st.ReasoningID, text),
		}))
		st.Reasonings = append(st.Reasonings, oaiToResponsesStateReasoning{
			ReasoningID:   st.ReasoningID,
			ReasoningData: text,
		})
		st.ReasoningID = ""
	}

	closeMessage := func(index int) {
		if !st.MsgItemAdded[index] || st.MsgItemDone[index] {
			return
		}
		text := ""
		if builder := st.MsgTextBuf[index]; builder != nil {
			text = builder.String()
		}
		itemID := responseMessageItemID(st.ResponseID, index)
		out = append(out, emitRespJSONEvent("response.output_text.done", map[string]any{
			"type":            "response.output_text.done",
			"sequence_number": nextSeq(),
			"item_id":         itemID,
			"output_index":    index,
			"content_index":   0,
			"text":            text,
			"logprobs":        []any{},
		}))
		out = append(out, emitRespJSONEvent("response.content_part.done", map[string]any{
			"type":            "response.content_part.done",
			"sequence_number": nextSeq(),
			"item_id":         itemID,
			"output_index":    index,
			"content_index":   0,
			"part":            outputTextPart(text),
		}))
		out = append(out, emitRespJSONEvent("response.output_item.done", map[string]any{
			"type":            "response.output_item.done",
			"sequence_number": nextSeq(),
			"output_index":    index,
			"item":            assistantMessageOutputItemCompleted(itemID, text),
		}))
		st.MsgItemDone[index] = true
	}

	for _, choiceValue := range choices {
		choice, ok := choiceValue.(map[string]any)
		if !ok {
			continue
		}
		index := choiceIndex(choice)
		delta, _ := jsonutil.Object(choice, "delta")

		if content, ok := jsonutil.String(delta, "content"); ok && content != "" {
			if st.ReasoningID != "" {
				stopReasoning(st.ReasoningBuf.String())
				st.ReasoningBuf.Reset()
			}

			messageItemID := responseMessageItemID(st.ResponseID, index)
			if !st.MsgItemAdded[index] {
				out = append(out, emitRespJSONEvent("response.output_item.added", map[string]any{
					"type":            "response.output_item.added",
					"sequence_number": nextSeq(),
					"output_index":    index,
					"item":            assistantMessageOutputItemInProgress(messageItemID),
				}))
				st.MsgItemAdded[index] = true
			}
			if !st.MsgContentAdded[index] {
				out = append(out, emitRespJSONEvent("response.content_part.added", map[string]any{
					"type":            "response.content_part.added",
					"sequence_number": nextSeq(),
					"item_id":         messageItemID,
					"output_index":    index,
					"content_index":   0,
					"part":            outputTextPart(""),
				}))
				st.MsgContentAdded[index] = true
			}
			out = append(out, emitRespJSONEvent("response.output_text.delta", map[string]any{
				"type":            "response.output_text.delta",
				"sequence_number": nextSeq(),
				"item_id":         messageItemID,
				"output_index":    index,
				"content_index":   0,
				"delta":           content,
				"logprobs":        []any{},
			}))
			if st.MsgTextBuf[index] == nil {
				st.MsgTextBuf[index] = &strings.Builder{}
			}
			st.MsgTextBuf[index].WriteString(content)
		}

		if reasoningDelta, ok := jsonutil.String(delta, "reasoning_content"); ok && reasoningDelta != "" {
			if st.ReasoningID == "" {
				st.ReasoningID = responseReasoningItemID(st.ResponseID, index)
				st.ReasoningIndex = index
				out = append(out, emitRespJSONEvent("response.output_item.added", map[string]any{
					"type":            "response.output_item.added",
					"sequence_number": nextSeq(),
					"output_index":    index,
					"item":            reasoningOutputItemInProgress(st.ReasoningID),
				}))
				out = append(out, emitRespJSONEvent("response.reasoning_summary_part.added", map[string]any{
					"type":            "response.reasoning_summary_part.added",
					"sequence_number": nextSeq(),
					"item_id":         st.ReasoningID,
					"output_index":    st.ReasoningIndex,
					"summary_index":   0,
					"part":            summaryTextPart(""),
				}))
			}
			st.ReasoningBuf.WriteString(reasoningDelta)
			out = append(out, emitRespJSONEvent("response.reasoning_summary_text.delta", map[string]any{
				"type":            "response.reasoning_summary_text.delta",
				"sequence_number": nextSeq(),
				"item_id":         st.ReasoningID,
				"output_index":    st.ReasoningIndex,
				"summary_index":   0,
				"delta":           reasoningDelta,
			}))
		}

		if toolCalls, ok := jsonutil.Array(delta, "tool_calls"); ok && len(toolCalls) > 0 {
			if st.ReasoningID != "" {
				stopReasoning(st.ReasoningBuf.String())
				st.ReasoningBuf.Reset()
			}
			closeMessage(index)

			toolCall, ok := toolCalls[0].(map[string]any)
			if ok {
				newCallID := jsonStringField(toolCall, "id")
				nameChunk := jsonStringField(toolCall, "function.name")
				if nameChunk != "" {
					st.FuncNames[index] = nameChunk
				}

				effectiveCallID := st.FuncCallIDs[index]
				shouldEmitItem := false
				if effectiveCallID == "" && newCallID != "" {
					effectiveCallID = newCallID
					st.FuncCallIDs[index] = newCallID
					shouldEmitItem = true
				}
				if shouldEmitItem && effectiveCallID != "" {
					out = append(out, emitRespJSONEvent("response.output_item.added", map[string]any{
						"type":            "response.output_item.added",
						"sequence_number": nextSeq(),
						"output_index":    index,
						"item":            functionCallOutputItem(effectiveCallID, st.FuncNames[index], "", "in_progress"),
					}))
				}

				if st.FuncArgsBuf[index] == nil {
					st.FuncArgsBuf[index] = &strings.Builder{}
				}

				if argsChunk := jsonStringField(toolCall, "function.arguments"); argsChunk != "" {
					referenceCallID := st.FuncCallIDs[index]
					if referenceCallID == "" {
						referenceCallID = newCallID
					}
					if referenceCallID != "" {
						out = append(out, emitRespJSONEvent("response.function_call_arguments.delta", map[string]any{
							"type":            "response.function_call_arguments.delta",
							"sequence_number": nextSeq(),
							"item_id":         responseFunctionItemID(referenceCallID),
							"output_index":    index,
							"delta":           argsChunk,
						}))
					}
					st.FuncArgsBuf[index].WriteString(argsChunk)
				}
			}
		}

		if finishReason, ok := jsonutil.String(choice, "finish_reason"); ok && finishReason != "" {
			for _, messageIndex := range sortedTrueIndices(st.MsgItemAdded) {
				closeMessage(messageIndex)
			}

			if st.ReasoningID != "" {
				stopReasoning(st.ReasoningBuf.String())
				st.ReasoningBuf.Reset()
			}

			for _, functionIndex := range sortedStringIndices(st.FuncCallIDs) {
				callID := st.FuncCallIDs[functionIndex]
				if callID == "" || st.FuncItemDone[functionIndex] {
					continue
				}
				args := "{}"
				if builder := st.FuncArgsBuf[functionIndex]; builder != nil && builder.Len() > 0 {
					args = builder.String()
				}
				out = append(out, emitRespJSONEvent("response.function_call_arguments.done", map[string]any{
					"type":            "response.function_call_arguments.done",
					"sequence_number": nextSeq(),
					"item_id":         responseFunctionItemID(callID),
					"output_index":    functionIndex,
					"arguments":       args,
				}))
				out = append(out, emitRespJSONEvent("response.output_item.done", map[string]any{
					"type":            "response.output_item.done",
					"sequence_number": nextSeq(),
					"output_index":    functionIndex,
					"item":            functionCallOutputItem(callID, st.FuncNames[functionIndex], args, "completed"),
				}))
				st.FuncItemDone[functionIndex] = true
				st.FuncArgsDone[functionIndex] = true
			}

			response := map[string]any{
				"id":         st.ResponseID,
				"object":     "response",
				"created_at": st.Created,
				"status":     "completed",
				"background": false,
				"error":      nil,
			}
			applyResponsesRequestFields(response, requestRoot, nil, false)
			if outputs := buildCompletedResponseOutput(st); len(outputs) > 0 {
				response["output"] = outputs
			}
			appendCompletedUsage(response, st)

			out = append(out, emitRespJSONEvent("response.completed", map[string]any{
				"type":            "response.completed",
				"sequence_number": nextSeq(),
				"response":        response,
			}))
		}
	}

	return out
}

// ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream builds a single Responses JSON
// from a non-streaming OpenAI Chat Completions response.
func ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(_ context.Context, _ string, _ []byte, requestRawJSON []byte, rawJSON []byte, _ *any) string {
	root := normalizeOpenAIResponsesInput(rawJSON)
	if root == nil {
		return ""
	}

	responseID := jsonStringField(root, "id")
	if responseID == "" {
		responseID = fmt.Sprintf("resp_%x_%d", time.Now().UnixNano(), atomic.AddUint64(&responseIDCounter, 1))
	}
	created, _ := jsonutil.Int64(root, "created")
	if created == 0 {
		created = time.Now().Unix()
	}

	response := map[string]any{
		"id":                 responseID,
		"object":             "response",
		"created_at":         created,
		"status":             "completed",
		"background":         false,
		"error":              nil,
		"incomplete_details": nil,
	}

	requestRoot := normalizeOpenAIResponsesInput(requestRawJSON)
	applyResponsesRequestFields(response, requestRoot, root, true)

	outputs := make([]any, 0)
	reasoningText := jsonStringField(root, "choices.0.message.reasoning_content")
	includeReasoning := reasoningText != ""
	if !includeReasoning && requestRoot != nil {
		includeReasoning = jsonutil.Exists(requestRoot, "reasoning")
	}
	if includeReasoning {
		reasoningSuffix := responseID
		if strings.HasPrefix(reasoningSuffix, "resp_") {
			reasoningSuffix = strings.TrimPrefix(reasoningSuffix, "resp_")
		}
		reasoningID := fmt.Sprintf("rs_%s", reasoningSuffix)
		outputs = append(outputs, reasoningOutputItemNonStream(reasoningID, reasoningText))
	}

	if choices, ok := jsonutil.Array(root, "choices"); ok {
		for _, choiceValue := range choices {
			choice, ok := choiceValue.(map[string]any)
			if !ok {
				continue
			}
			index := choiceIndex(choice)
			message, ok := jsonutil.Object(choice, "message")
			if !ok {
				continue
			}
			if content := jsonStringField(message, "content"); content != "" {
				outputs = append(outputs, assistantMessageOutputItemCompleted(responseMessageItemID(responseID, index), content))
			}
			if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok {
				for _, toolCallValue := range toolCalls {
					toolCall, ok := toolCallValue.(map[string]any)
					if !ok {
						continue
					}
					callID := jsonStringField(toolCall, "id")
					if callID == "" {
						continue
					}
					outputs = append(outputs, functionCallOutputItem(
						callID,
						jsonStringField(toolCall, "function.name"),
						jsonStringField(toolCall, "function.arguments"),
						"completed",
					))
				}
			}
		}
	}
	if len(outputs) > 0 {
		response["output"] = outputs
	}

	if usage, ok := jsonutil.Object(root, "usage"); ok {
		if jsonutil.Exists(usage, "prompt_tokens") || jsonutil.Exists(usage, "completion_tokens") || jsonutil.Exists(usage, "total_tokens") {
			usagePayload := map[string]any{
				"input_tokens": jsonInt64OrZero(usage, "prompt_tokens"),
				"input_tokens_details": map[string]any{
					"cached_tokens": jsonInt64OrZero(usage, "prompt_tokens_details.cached_tokens"),
				},
				"output_tokens": jsonInt64OrZero(usage, "completion_tokens"),
				"total_tokens":  jsonInt64OrZero(usage, "total_tokens"),
			}
			if reasoningTokens, ok := jsonutil.Int64(usage, "output_tokens_details.reasoning_tokens"); ok {
				usagePayload["output_tokens_details"] = map[string]any{
					"reasoning_tokens": reasoningTokens,
				}
			}
			response["usage"] = usagePayload
		} else {
			response["usage"] = usage
		}
	}

	return string(jsonutil.MarshalOrOriginal(nil, response))
}

func jsonInt64OrZero(root map[string]any, path string) int64 {
	value, ok := jsonutil.Int64(root, path)
	if !ok {
		return 0
	}
	return value
}
