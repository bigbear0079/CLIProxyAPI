// Package gemini provides request translation functionality for Codex to Gemini API compatibility.
// It handles parsing and transforming Codex API requests into Gemini API format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package performs JSON data transformation to ensure compatibility
// between Codex API format and Gemini API's expected format.
package gemini

import (
	"crypto/rand"
	"encoding/json"
	"math/big"
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// ConvertGeminiRequestToCodex parses and transforms a Gemini API request into Codex API format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the Codex API.
// The function performs comprehensive transformation including:
// 1. Model name mapping and generation configuration extraction
// 2. System instruction conversion to Codex format
// 3. Message content conversion with proper role mapping
// 4. Tool call and tool result handling with FIFO queue for ID matching
// 5. Tool declaration and tool choice configuration mapping
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the Gemini API
//   - stream: A boolean indicating if the request is for a streaming response (unused in current implementation)
//
// Returns:
//   - []byte: The transformed request data in Codex API format
func ConvertGeminiRequestToCodex(modelName string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)
	outRoot := map[string]any{
		"model":        modelName,
		"instructions": "",
		"input":        []any{},
	}

	shortMap := codexGeminiBuildShortMap(root)

	// helper for generating paired call IDs in the form: call_<alphanum>
	// Gemini uses sequential pairing across possibly multiple in-flight
	// functionCalls, so we keep a FIFO queue of generated call IDs and
	// consume them in order when functionResponses arrive.
	var pendingCallIDs []string

	// genCallID creates a random call id like: call_<8chars>
	genCallID := func() string {
		const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		var b strings.Builder
		// 8 chars random suffix
		for i := 0; i < 24; i++ {
			n, _ := rand.Int(rand.Reader, big.NewInt(int64(len(letters))))
			b.WriteByte(letters[n.Int64()])
		}
		return "call_" + b.String()
	}

	inputItems := make([]any, 0)

	// System instruction -> as a user message with input_text parts
	if systemInstruction, ok := jsonutil.Object(root, "system_instruction"); ok {
		if message, ok := codexGeminiSystemMessage(systemInstruction); ok {
			inputItems = append(inputItems, message)
		}
	} else if systemInstruction, ok := jsonutil.Object(root, "systemInstruction"); ok {
		if message, ok := codexGeminiSystemMessage(systemInstruction); ok {
			inputItems = append(inputItems, message)
		}
	}

	// Contents -> messages and function calls/results
	if contents, ok := jsonutil.Array(root, "contents"); ok {
		for _, itemValue := range contents {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}

			role, _ := jsonutil.String(item, "role")
			if role == "model" {
				role = "assistant"
			}

			parts, ok := jsonutil.Array(item, "parts")
			if !ok {
				continue
			}
			for _, partValue := range parts {
				part, ok := partValue.(map[string]any)
				if !ok {
					continue
				}
				// text part
				if text, ok := jsonutil.String(part, "text"); ok {
					partType := "input_text"
					if role == "assistant" {
						partType = "output_text"
					}
					inputItems = append(inputItems, map[string]any{
						"type": "message",
						"role": role,
						"content": []any{
							map[string]any{
								"type": partType,
								"text": text,
							},
						},
					})
					continue
				}

				// function call from model
				if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
					functionCallItem := map[string]any{
						"type": "function_call",
					}
					if name, ok := jsonutil.String(functionCall, "name"); ok {
						n := name
						if short, ok := shortMap[n]; ok {
							n = short
						} else {
							n = shortenNameIfNeeded(n)
						}
						functionCallItem["name"] = n
					}
					if args, ok := jsonutil.Get(functionCall, "args"); ok {
						functionCallItem["arguments"] = codexGeminiJSONText(args)
					}
					// generate a paired random call_id and enqueue it so the
					// corresponding functionResponse can pop the earliest id
					// to preserve ordering when multiple calls are present.
					id := genCallID()
					functionCallItem["call_id"] = id
					pendingCallIDs = append(pendingCallIDs, id)
					inputItems = append(inputItems, functionCallItem)
					continue
				}

				// function response from user
				if functionResponse, ok := jsonutil.Object(part, "functionResponse"); ok {
					functionOutputItem := map[string]any{
						"type": "function_call_output",
					}
					// Prefer a string result if present; otherwise embed the raw response as a string
					if result, ok := jsonutil.Get(functionResponse, "response.result"); ok {
						functionOutputItem["output"] = codexGeminiContentString(result)
					} else if response, ok := jsonutil.Get(functionResponse, "response"); ok {
						functionOutputItem["output"] = codexGeminiContentString(response)
					}
					// attach the oldest queued call_id to pair the response
					// with its call. If the queue is empty, generate a new id.
					var id string
					if len(pendingCallIDs) > 0 {
						id = pendingCallIDs[0]
						// pop the first element
						pendingCallIDs = pendingCallIDs[1:]
					} else {
						id = genCallID()
					}
					functionOutputItem["call_id"] = id
					inputItems = append(inputItems, functionOutputItem)
					continue
				}
			}
		}
	}
	outRoot["input"] = inputItems

	// Tools mapping: Gemini functionDeclarations -> Codex tools
	if tools, ok := jsonutil.Array(root, "tools"); ok {
		codexTools := make([]any, 0)
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			functionDeclarations, ok := jsonutil.Array(tool, "functionDeclarations")
			if !ok {
				continue
			}
			for _, declarationValue := range functionDeclarations {
				declaration, ok := declarationValue.(map[string]any)
				if !ok {
					continue
				}

				codexTool := map[string]any{
					"type":   "function",
					"strict": false,
				}
				if name, ok := jsonutil.String(declaration, "name"); ok {
					name := name
					if short, ok := shortMap[name]; ok {
						name = short
					} else {
						name = shortenNameIfNeeded(name)
					}
					codexTool["name"] = name
				}
				if description, ok := jsonutil.String(declaration, "description"); ok {
					codexTool["description"] = description
				}
				if parameters, ok := jsonutil.Get(declaration, "parameters"); ok {
					if cleaned, ok := codexGeminiCleanSchema(parameters); ok {
						codexTool["parameters"] = cleaned
					}
				} else if parameters, ok := jsonutil.Get(declaration, "parametersJsonSchema"); ok {
					if cleaned, ok := codexGeminiCleanSchema(parameters); ok {
						codexTool["parameters"] = cleaned
					}
				}
				codexTools = append(codexTools, codexTool)
			}
		}
		outRoot["tools"] = codexTools
		outRoot["tool_choice"] = "auto"
	}

	// Fixed flags aligning with Codex expectations
	outRoot["parallel_tool_calls"] = true

	// Convert Gemini thinkingConfig to Codex reasoning.effort.
	// Note: Google official Python SDK sends snake_case fields (thinking_level/thinking_budget).
	reasoningEffort := "medium"
	if generationConfig, ok := jsonutil.Object(root, "generationConfig"); ok {
		reasoningEffort = codexGeminiReasoningEffort(generationConfig, reasoningEffort)
	} else if generationConfig, ok := jsonutil.Object(root, "generation_config"); ok {
		reasoningEffort = codexGeminiReasoningEffort(generationConfig, reasoningEffort)
	}
	outRoot["reasoning"] = map[string]any{
		"effort":  reasoningEffort,
		"summary": "auto",
	}
	outRoot["stream"] = true
	outRoot["store"] = false
	outRoot["include"] = []string{"reasoning.encrypted_content"}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func codexGeminiBuildShortMap(root map[string]any) map[string]string {
	names := make([]string, 0)
	if tools, ok := jsonutil.Array(root, "tools"); ok {
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			functionDeclarations, ok := jsonutil.Array(tool, "functionDeclarations")
			if !ok {
				continue
			}
			for _, declarationValue := range functionDeclarations {
				declaration, ok := declarationValue.(map[string]any)
				if !ok {
					continue
				}
				if name, ok := jsonutil.String(declaration, "name"); ok && name != "" {
					names = append(names, name)
				}
			}
		}
	}
	if len(names) == 0 {
		return map[string]string{}
	}
	return buildShortNameMap(names)
}

func codexGeminiSystemMessage(systemInstruction map[string]any) (map[string]any, bool) {
	parts, ok := jsonutil.Array(systemInstruction, "parts")
	if !ok {
		return nil, false
	}

	contentParts := make([]any, 0)
	for _, partValue := range parts {
		part, ok := partValue.(map[string]any)
		if !ok {
			continue
		}
		if text, ok := jsonutil.String(part, "text"); ok {
			contentParts = append(contentParts, map[string]any{
				"type": "input_text",
				"text": text,
			})
		}
	}
	if len(contentParts) == 0 {
		return nil, false
	}
	return map[string]any{
		"type":    "message",
		"role":    "developer",
		"content": contentParts,
	}, true
}

func codexGeminiReasoningEffort(generationConfig map[string]any, defaultEffort string) string {
	thinkingConfig, ok := jsonutil.Object(generationConfig, "thinkingConfig")
	if !ok {
		thinkingConfig, ok = jsonutil.Object(generationConfig, "thinking_config")
		if !ok {
			return defaultEffort
		}
	}

	if thinkingLevel, ok := jsonutil.String(thinkingConfig, "thinkingLevel"); ok {
		effort := strings.ToLower(strings.TrimSpace(thinkingLevel))
		if effort != "" {
			return effort
		}
	}
	if thinkingLevel, ok := jsonutil.String(thinkingConfig, "thinking_level"); ok {
		effort := strings.ToLower(strings.TrimSpace(thinkingLevel))
		if effort != "" {
			return effort
		}
	}
	if thinkingBudget, ok := jsonutil.Int64(thinkingConfig, "thinkingBudget"); ok {
		if effort, ok := thinking.ConvertBudgetToLevel(int(thinkingBudget)); ok {
			return effort
		}
	}
	if thinkingBudget, ok := jsonutil.Int64(thinkingConfig, "thinking_budget"); ok {
		if effort, ok := thinking.ConvertBudgetToLevel(int(thinkingBudget)); ok {
			return effort
		}
	}
	return defaultEffort
}

func codexGeminiJSONText(value any) string {
	return string(jsonutil.MarshalOrOriginal(nil, value))
}

func codexGeminiContentString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
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
		return string(jsonutil.MarshalOrOriginal(nil, typed))
	}
}

func codexGeminiCleanSchema(value any) (map[string]any, bool) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, false
	}
	delete(object, "$schema")
	object["additionalProperties"] = false
	codexGeminiLowercaseTypeFields(object)
	return object, true
}

func codexGeminiLowercaseTypeFields(value any) {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if key == "type" {
				if typeString, ok := child.(string); ok {
					typed[key] = strings.ToLower(typeString)
				}
			}
			codexGeminiLowercaseTypeFields(child)
		}
	case []any:
		for _, child := range typed {
			codexGeminiLowercaseTypeFields(child)
		}
	}
}

// shortenNameIfNeeded applies the simple shortening rule for a single name.
func shortenNameIfNeeded(name string) string {
	const limit = 64
	if len(name) <= limit {
		return name
	}
	if strings.HasPrefix(name, "mcp__") {
		idx := strings.LastIndex(name, "__")
		if idx > 0 {
			cand := "mcp__" + name[idx+2:]
			if len(cand) > limit {
				return cand[:limit]
			}
			return cand
		}
	}
	return name[:limit]
}

// buildShortNameMap ensures uniqueness of shortened names within a request.
func buildShortNameMap(names []string) map[string]string {
	const limit = 64
	used := map[string]struct{}{}
	m := map[string]string{}

	baseCandidate := func(n string) string {
		if len(n) <= limit {
			return n
		}
		if strings.HasPrefix(n, "mcp__") {
			idx := strings.LastIndex(n, "__")
			if idx > 0 {
				cand := "mcp__" + n[idx+2:]
				if len(cand) > limit {
					cand = cand[:limit]
				}
				return cand
			}
		}
		return n[:limit]
	}

	makeUnique := func(cand string) string {
		if _, ok := used[cand]; !ok {
			return cand
		}
		base := cand
		for i := 1; ; i++ {
			suffix := "_" + strconv.Itoa(i)
			allowed := limit - len(suffix)
			if allowed < 0 {
				allowed = 0
			}
			tmp := base
			if len(tmp) > allowed {
				tmp = tmp[:allowed]
			}
			tmp = tmp + suffix
			if _, ok := used[tmp]; !ok {
				return tmp
			}
		}
	}

	for _, n := range names {
		cand := baseCandidate(n)
		uniq := makeUnique(cand)
		used[uniq] = struct{}{}
		m[n] = uniq
	}
	return m
}
