package responses

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

type claudeToResponsesState struct {
	Seq          int
	ResponseID   string
	CreatedAt    int64
	CurrentMsgID string
	CurrentFCID  string
	InTextBlock  bool
	InFuncBlock  bool
	FuncArgsBuf  map[int]*strings.Builder // index -> args
	// function call bookkeeping for output aggregation
	FuncNames   map[int]string // index -> function name
	FuncCallIDs map[int]string // index -> call id
	// message text aggregation
	TextBuf strings.Builder
	// reasoning state
	ReasoningActive    bool
	ReasoningItemID    string
	ReasoningBuf       strings.Builder
	ReasoningPartAdded bool
	ReasoningIndex     int
	// usage aggregation
	InputTokens  int64
	OutputTokens int64
	UsageSeen    bool
}

var dataTag = []byte("data:")

func pickRequestJSON(originalRequestRawJSON, requestRawJSON []byte) []byte {
	if len(originalRequestRawJSON) > 0 && json.Valid(originalRequestRawJSON) {
		return originalRequestRawJSON
	}
	if len(requestRawJSON) > 0 && json.Valid(requestRawJSON) {
		return requestRawJSON
	}
	return nil
}

func emitEvent(event string, payload string) string {
	return fmt.Sprintf("event: %s\ndata: %s", event, payload)
}

func emitJSONEvent(event string, payload any) string {
	return emitEvent(event, string(jsonutil.MarshalOrOriginal(nil, payload)))
}

func parseClaudeEvent(rawJSON []byte) map[string]any {
	if !bytes.HasPrefix(rawJSON, dataTag) {
		return nil
	}
	root, errParse := jsonutil.ParseObjectBytes(bytes.TrimSpace(rawJSON[len(dataTag):]))
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

func jsonInt64Field(root map[string]any, path string) int64 {
	value, ok := jsonutil.Int64(root, path)
	if !ok {
		return 0
	}
	return value
}

func applyClaudeResponsesRequestFields(response map[string]any, requestRoot map[string]any) {
	if requestRoot == nil {
		return
	}

	copyField := func(source string, target string) {
		if value, ok := jsonutil.Get(requestRoot, source); ok {
			_ = jsonutil.Set(response, target, value)
		}
	}

	copyField("instructions", "instructions")
	copyField("max_output_tokens", "max_output_tokens")
	copyField("max_tool_calls", "max_tool_calls")
	copyField("model", "model")
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
}

func sortedFunctionIndices(funcArgs map[int]*strings.Builder) []int {
	indices := make([]int, 0, len(funcArgs))
	for index := range funcArgs {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	return indices
}

func buildClaudeCompletedOutput(st *claudeToResponsesState) []any {
	output := make([]any, 0, 2+len(st.FuncArgsBuf))

	if st.ReasoningBuf.Len() > 0 || st.ReasoningPartAdded {
		item := map[string]any{
			"id":   st.ReasoningItemID,
			"type": "reasoning",
			"summary": []any{
				map[string]any{
					"type": "summary_text",
					"text": st.ReasoningBuf.String(),
				},
			},
		}
		output = append(output, item)
	}

	if st.TextBuf.Len() > 0 || st.InTextBlock || st.CurrentMsgID != "" {
		item := map[string]any{
			"id":     st.CurrentMsgID,
			"type":   "message",
			"status": "completed",
			"content": []any{
				map[string]any{
					"type":        "output_text",
					"annotations": []any{},
					"logprobs":    []any{},
					"text":        st.TextBuf.String(),
				},
			},
			"role": "assistant",
		}
		output = append(output, item)
	}

	for _, index := range sortedFunctionIndices(st.FuncArgsBuf) {
		args := ""
		if buffer := st.FuncArgsBuf[index]; buffer != nil {
			args = buffer.String()
		}
		callID := st.FuncCallIDs[index]
		name := st.FuncNames[index]
		if callID == "" && st.CurrentFCID != "" {
			callID = st.CurrentFCID
		}

		output = append(output, map[string]any{
			"id":        fmt.Sprintf("fc_%s", callID),
			"type":      "function_call",
			"status":    "completed",
			"arguments": args,
			"call_id":   callID,
			"name":      name,
		})
	}

	return output
}

// ConvertClaudeResponseToOpenAIResponses converts Claude SSE to OpenAI Responses SSE events.
func ConvertClaudeResponseToOpenAIResponses(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = ctx
	_ = modelName

	if *param == nil {
		*param = &claudeToResponsesState{
			FuncArgsBuf: make(map[int]*strings.Builder),
			FuncNames:   make(map[int]string),
			FuncCallIDs: make(map[int]string),
		}
	}
	st := (*param).(*claudeToResponsesState)

	root := parseClaudeEvent(rawJSON)
	if root == nil {
		return []string{}
	}

	eventType := jsonStringField(root, "type")
	out := make([]string, 0)
	nextSeq := func() int {
		st.Seq++
		return st.Seq
	}

	switch eventType {
	case "message_start":
		message, ok := jsonutil.Object(root, "message")
		if !ok {
			return out
		}

		st.ResponseID = jsonStringField(message, "id")
		st.CreatedAt = time.Now().Unix()
		st.TextBuf.Reset()
		st.ReasoningBuf.Reset()
		st.ReasoningActive = false
		st.InTextBlock = false
		st.InFuncBlock = false
		st.CurrentMsgID = ""
		st.CurrentFCID = ""
		st.ReasoningItemID = ""
		st.ReasoningIndex = 0
		st.ReasoningPartAdded = false
		st.FuncArgsBuf = make(map[int]*strings.Builder)
		st.FuncNames = make(map[int]string)
		st.FuncCallIDs = make(map[int]string)
		st.InputTokens = 0
		st.OutputTokens = 0
		st.UsageSeen = false

		if usage, ok := jsonutil.Object(message, "usage"); ok {
			if value, ok := jsonutil.Int64(usage, "input_tokens"); ok {
				st.InputTokens = value
				st.UsageSeen = true
			}
			if value, ok := jsonutil.Int64(usage, "output_tokens"); ok {
				st.OutputTokens = value
				st.UsageSeen = true
			}
		}

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

	case "content_block_start":
		contentBlock, ok := jsonutil.Object(root, "content_block")
		if !ok {
			return out
		}

		index := int(jsonInt64Field(root, "index"))
		blockType := jsonStringField(contentBlock, "type")

		switch blockType {
		case "text":
			st.InTextBlock = true
			st.CurrentMsgID = fmt.Sprintf("msg_%s_0", st.ResponseID)

			out = append(out, emitJSONEvent("response.output_item.added", map[string]any{
				"type":            "response.output_item.added",
				"sequence_number": nextSeq(),
				"output_index":    0,
				"item": map[string]any{
					"id":      st.CurrentMsgID,
					"type":    "message",
					"status":  "in_progress",
					"content": []any{},
					"role":    "assistant",
				},
			}))

			out = append(out, emitJSONEvent("response.content_part.added", map[string]any{
				"type":            "response.content_part.added",
				"sequence_number": nextSeq(),
				"item_id":         st.CurrentMsgID,
				"output_index":    0,
				"content_index":   0,
				"part": map[string]any{
					"type":        "output_text",
					"annotations": []any{},
					"logprobs":    []any{},
					"text":        "",
				},
			}))

		case "tool_use":
			st.InFuncBlock = true
			st.CurrentFCID = jsonStringField(contentBlock, "id")
			name := jsonStringField(contentBlock, "name")

			out = append(out, emitJSONEvent("response.output_item.added", map[string]any{
				"type":            "response.output_item.added",
				"sequence_number": nextSeq(),
				"output_index":    index,
				"item": map[string]any{
					"id":        fmt.Sprintf("fc_%s", st.CurrentFCID),
					"type":      "function_call",
					"status":    "in_progress",
					"arguments": "",
					"call_id":   st.CurrentFCID,
					"name":      name,
				},
			}))

			if st.FuncArgsBuf[index] == nil {
				st.FuncArgsBuf[index] = &strings.Builder{}
			}
			st.FuncCallIDs[index] = st.CurrentFCID
			st.FuncNames[index] = name

		case "thinking":
			st.ReasoningActive = true
			st.ReasoningIndex = index
			st.ReasoningBuf.Reset()
			st.ReasoningItemID = fmt.Sprintf("rs_%s_%d", st.ResponseID, index)

			out = append(out, emitJSONEvent("response.output_item.added", map[string]any{
				"type":            "response.output_item.added",
				"sequence_number": nextSeq(),
				"output_index":    index,
				"item": map[string]any{
					"id":      st.ReasoningItemID,
					"type":    "reasoning",
					"status":  "in_progress",
					"summary": []any{},
				},
			}))

			out = append(out, emitJSONEvent("response.reasoning_summary_part.added", map[string]any{
				"type":            "response.reasoning_summary_part.added",
				"sequence_number": nextSeq(),
				"item_id":         st.ReasoningItemID,
				"output_index":    index,
				"summary_index":   0,
				"part": map[string]any{
					"type": "summary_text",
					"text": "",
				},
			}))

			st.ReasoningPartAdded = true
		}

	case "content_block_delta":
		delta, ok := jsonutil.Object(root, "delta")
		if !ok {
			return out
		}

		deltaType := jsonStringField(delta, "type")
		switch deltaType {
		case "text_delta":
			if text, ok := jsonutil.String(delta, "text"); ok {
				st.TextBuf.WriteString(text)
				out = append(out, emitJSONEvent("response.output_text.delta", map[string]any{
					"type":            "response.output_text.delta",
					"sequence_number": nextSeq(),
					"item_id":         st.CurrentMsgID,
					"output_index":    0,
					"content_index":   0,
					"delta":           text,
					"logprobs":        []any{},
				}))
			}

		case "input_json_delta":
			index := int(jsonInt64Field(root, "index"))
			if partialJSON, ok := jsonutil.String(delta, "partial_json"); ok {
				if st.FuncArgsBuf[index] == nil {
					st.FuncArgsBuf[index] = &strings.Builder{}
				}
				st.FuncArgsBuf[index].WriteString(partialJSON)

				out = append(out, emitJSONEvent("response.function_call_arguments.delta", map[string]any{
					"type":            "response.function_call_arguments.delta",
					"sequence_number": nextSeq(),
					"item_id":         fmt.Sprintf("fc_%s", st.CurrentFCID),
					"output_index":    index,
					"delta":           partialJSON,
				}))
			}

		case "thinking_delta":
			if st.ReasoningActive {
				if thinkingText, ok := jsonutil.String(delta, "thinking"); ok {
					st.ReasoningBuf.WriteString(thinkingText)
					out = append(out, emitJSONEvent("response.reasoning_summary_text.delta", map[string]any{
						"type":            "response.reasoning_summary_text.delta",
						"sequence_number": nextSeq(),
						"item_id":         st.ReasoningItemID,
						"output_index":    st.ReasoningIndex,
						"summary_index":   0,
						"delta":           thinkingText,
					}))
				}
			}
		}

	case "content_block_stop":
		index := int(jsonInt64Field(root, "index"))

		if st.InTextBlock {
			out = append(out, emitJSONEvent("response.output_text.done", map[string]any{
				"type":            "response.output_text.done",
				"sequence_number": nextSeq(),
				"item_id":         st.CurrentMsgID,
				"output_index":    0,
				"content_index":   0,
				"text":            "",
				"logprobs":        []any{},
			}))

			out = append(out, emitJSONEvent("response.content_part.done", map[string]any{
				"type":            "response.content_part.done",
				"sequence_number": nextSeq(),
				"item_id":         st.CurrentMsgID,
				"output_index":    0,
				"content_index":   0,
				"part": map[string]any{
					"type":        "output_text",
					"annotations": []any{},
					"logprobs":    []any{},
					"text":        "",
				},
			}))

			out = append(out, emitJSONEvent("response.output_item.done", map[string]any{
				"type":            "response.output_item.done",
				"sequence_number": nextSeq(),
				"output_index":    0,
				"item": map[string]any{
					"id":     st.CurrentMsgID,
					"type":   "message",
					"status": "completed",
					"content": []any{
						map[string]any{
							"type": "output_text",
							"text": "",
						},
					},
					"role": "assistant",
				},
			}))

			st.InTextBlock = false
		} else if st.InFuncBlock {
			args := "{}"
			if buffer := st.FuncArgsBuf[index]; buffer != nil && buffer.Len() > 0 {
				args = buffer.String()
			}

			out = append(out, emitJSONEvent("response.function_call_arguments.done", map[string]any{
				"type":            "response.function_call_arguments.done",
				"sequence_number": nextSeq(),
				"item_id":         fmt.Sprintf("fc_%s", st.CurrentFCID),
				"output_index":    index,
				"arguments":       args,
			}))

			out = append(out, emitJSONEvent("response.output_item.done", map[string]any{
				"type":            "response.output_item.done",
				"sequence_number": nextSeq(),
				"output_index":    index,
				"item": map[string]any{
					"id":        fmt.Sprintf("fc_%s", st.CurrentFCID),
					"type":      "function_call",
					"status":    "completed",
					"arguments": args,
					"call_id":   st.CurrentFCID,
					"name":      st.FuncNames[index],
				},
			}))

			st.InFuncBlock = false
		} else if st.ReasoningActive {
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
				"part": map[string]any{
					"type": "summary_text",
					"text": full,
				},
			}))

			st.ReasoningActive = false
			st.ReasoningPartAdded = false
		}

	case "message_delta":
		if usage, ok := jsonutil.Object(root, "usage"); ok {
			if value, ok := jsonutil.Int64(usage, "output_tokens"); ok {
				st.OutputTokens = value
				st.UsageSeen = true
			}
			if value, ok := jsonutil.Int64(usage, "input_tokens"); ok {
				st.InputTokens = value
				st.UsageSeen = true
			}
		}

	case "message_stop":
		response := map[string]any{
			"id":         st.ResponseID,
			"object":     "response",
			"created_at": st.CreatedAt,
			"status":     "completed",
			"background": false,
			"error":      nil,
		}

		requestRoot, _ := jsonutil.ParseObjectBytes(pickRequestJSON(originalRequestRawJSON, requestRawJSON))
		applyClaudeResponsesRequestFields(response, requestRoot)

		if output := buildClaudeCompletedOutput(st); len(output) > 0 {
			response["output"] = output
		}

		reasoningTokens := int64(0)
		if st.ReasoningBuf.Len() > 0 {
			reasoningTokens = int64(st.ReasoningBuf.Len() / 4)
		}
		if st.UsageSeen || reasoningTokens > 0 {
			usage := map[string]any{
				"input_tokens": st.InputTokens,
				"input_tokens_details": map[string]any{
					"cached_tokens": 0,
				},
				"output_tokens": st.OutputTokens,
			}
			if reasoningTokens > 0 {
				usage["output_tokens_details"] = map[string]any{
					"reasoning_tokens": reasoningTokens,
				}
			}
			total := st.InputTokens + st.OutputTokens
			if total > 0 || st.UsageSeen {
				usage["total_tokens"] = total
			}
			response["usage"] = usage
		}

		out = append(out, emitJSONEvent("response.completed", map[string]any{
			"type":            "response.completed",
			"sequence_number": nextSeq(),
			"response":        response,
		}))
	}

	return out
}

// ConvertClaudeResponseToOpenAIResponsesNonStream aggregates Claude SSE into a single OpenAI Responses JSON.
func ConvertClaudeResponseToOpenAIResponsesNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	scanner := bufio.NewScanner(bytes.NewReader(rawJSON))
	buffer := make([]byte, 52_428_800)
	scanner.Buffer(buffer, 52_428_800)

	response := map[string]any{
		"id":                 "",
		"object":             "response",
		"created_at":         int64(0),
		"status":             "completed",
		"background":         false,
		"error":              nil,
		"incomplete_details": nil,
		"output":             []any{},
		"usage": map[string]any{
			"input_tokens": 0,
			"input_tokens_details": map[string]any{
				"cached_tokens": 0,
			},
			"output_tokens":         0,
			"output_tokens_details": map[string]any{},
			"total_tokens":          0,
		},
	}

	var (
		responseID      string
		createdAt       int64
		currentMsgID    string
		currentFCID     string
		textBuf         strings.Builder
		reasoningBuf    strings.Builder
		reasoningActive bool
		reasoningItemID string
		inputTokens     int64
		outputTokens    int64
	)

	type toolState struct {
		ID   string
		Name string
		Args strings.Builder
	}
	toolCalls := make(map[int]*toolState)

	for scanner.Scan() {
		root := parseClaudeEvent(scanner.Bytes())
		if root == nil {
			continue
		}

		switch jsonStringField(root, "type") {
		case "message_start":
			message, ok := jsonutil.Object(root, "message")
			if !ok {
				continue
			}
			responseID = jsonStringField(message, "id")
			createdAt = time.Now().Unix()
			if usage, ok := jsonutil.Object(message, "usage"); ok {
				inputTokens = jsonInt64Field(usage, "input_tokens")
			}

		case "content_block_start":
			contentBlock, ok := jsonutil.Object(root, "content_block")
			if !ok {
				continue
			}
			index := int(jsonInt64Field(root, "index"))

			switch jsonStringField(contentBlock, "type") {
			case "text":
				currentMsgID = "msg_" + responseID + "_0"
			case "tool_use":
				currentFCID = jsonStringField(contentBlock, "id")
				name := jsonStringField(contentBlock, "name")
				if toolCalls[index] == nil {
					toolCalls[index] = &toolState{ID: currentFCID, Name: name}
				} else {
					toolCalls[index].ID = currentFCID
					toolCalls[index].Name = name
				}
			case "thinking":
				reasoningActive = true
				reasoningItemID = fmt.Sprintf("rs_%s_%d", responseID, index)
			}

		case "content_block_delta":
			delta, ok := jsonutil.Object(root, "delta")
			if !ok {
				continue
			}

			switch jsonStringField(delta, "type") {
			case "text_delta":
				if text, ok := jsonutil.String(delta, "text"); ok {
					textBuf.WriteString(text)
				}
			case "input_json_delta":
				index := int(jsonInt64Field(root, "index"))
				if partialJSON, ok := jsonutil.String(delta, "partial_json"); ok {
					if toolCalls[index] == nil {
						toolCalls[index] = &toolState{}
					}
					toolCalls[index].Args.WriteString(partialJSON)
				}
			case "thinking_delta":
				if reasoningActive {
					if thinkingText, ok := jsonutil.String(delta, "thinking"); ok {
						reasoningBuf.WriteString(thinkingText)
					}
				}
			}

		case "message_delta":
			if usage, ok := jsonutil.Object(root, "usage"); ok {
				outputTokens = jsonInt64Field(usage, "output_tokens")
			}
		}
	}

	response["id"] = responseID
	response["created_at"] = createdAt

	requestRoot, _ := jsonutil.ParseObjectBytes(pickRequestJSON(originalRequestRawJSON, requestRawJSON))
	applyClaudeResponsesRequestFields(response, requestRoot)

	output := make([]any, 0, 2+len(toolCalls))
	if reasoningBuf.Len() > 0 {
		output = append(output, map[string]any{
			"id":   reasoningItemID,
			"type": "reasoning",
			"summary": []any{
				map[string]any{
					"type": "summary_text",
					"text": reasoningBuf.String(),
				},
			},
		})
	}
	if currentMsgID != "" || textBuf.Len() > 0 {
		output = append(output, map[string]any{
			"id":     currentMsgID,
			"type":   "message",
			"status": "completed",
			"content": []any{
				map[string]any{
					"type":        "output_text",
					"annotations": []any{},
					"logprobs":    []any{},
					"text":        textBuf.String(),
				},
			},
			"role": "assistant",
		})
	}
	if len(toolCalls) > 0 {
		indices := make([]int, 0, len(toolCalls))
		for index := range toolCalls {
			indices = append(indices, index)
		}
		sort.Ints(indices)
		for _, index := range indices {
			call := toolCalls[index]
			args := call.Args.String()
			if args == "" {
				args = "{}"
			}
			output = append(output, map[string]any{
				"id":        fmt.Sprintf("fc_%s", call.ID),
				"type":      "function_call",
				"status":    "completed",
				"arguments": args,
				"call_id":   call.ID,
				"name":      call.Name,
			})
		}
	}
	if len(output) > 0 {
		response["output"] = output
	}

	totalTokens := inputTokens + outputTokens
	usage := map[string]any{
		"input_tokens": inputTokens,
		"input_tokens_details": map[string]any{
			"cached_tokens": 0,
		},
		"output_tokens": outputTokens,
		"total_tokens":  totalTokens,
	}
	if reasoningBuf.Len() > 0 {
		reasoningTokens := int64(len(reasoningBuf.String()) / 4)
		if reasoningTokens > 0 {
			usage["output_tokens_details"] = map[string]any{
				"reasoning_tokens": reasoningTokens,
			}
		}
	}
	response["usage"] = usage

	return string(jsonutil.MarshalOrOriginal(rawJSON, response))
}
