package responses

import "github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"

func ConvertOpenAIResponsesRequestToCodex(modelName string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	if inputText, ok := jsonutil.String(root, "input"); ok {
		root["input"] = []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": inputText,
					},
				},
			},
		}
	}

	_ = jsonutil.Set(root, "stream", true)
	_ = jsonutil.Set(root, "store", false)
	_ = jsonutil.Set(root, "parallel_tool_calls", true)
	_ = jsonutil.Set(root, "include", []string{"reasoning.encrypted_content"})
	// Codex Responses rejects token limit fields, so strip them out before forwarding.
	_ = jsonutil.Delete(root, "max_output_tokens")
	_ = jsonutil.Delete(root, "max_completion_tokens")
	_ = jsonutil.Delete(root, "temperature")
	_ = jsonutil.Delete(root, "top_p")
	if serviceTier, ok := jsonutil.String(root, "service_tier"); ok && serviceTier != "priority" {
		_ = jsonutil.Delete(root, "service_tier")
	}

	_ = jsonutil.Delete(root, "truncation")
	applyResponsesCompactionCompatibility(root)

	// Delete the user field as it is not supported by the Codex upstream.
	_ = jsonutil.Delete(root, "user")

	// Convert role "system" to "developer" in input array to comply with Codex API requirements.
	convertSystemRoleToDeveloper(root)

	return jsonutil.MarshalOrOriginal(inputRawJSON, root)
}

// applyResponsesCompactionCompatibility handles OpenAI Responses context_management.compaction
// for Codex upstream compatibility.
//
// Codex /responses currently rejects context_management with:
// {"detail":"Unsupported parameter: context_management"}.
//
// Compatibility strategy:
// 1) Remove context_management before forwarding to Codex upstream.
func applyResponsesCompactionCompatibility(root map[string]any) {
	if !jsonutil.Exists(root, "context_management") {
		return
	}

	_ = jsonutil.Delete(root, "context_management")
}

// convertSystemRoleToDeveloper traverses the input array and converts any message items
// with role "system" to role "developer". This is necessary because Codex API does not
// accept "system" role in the input array.
func convertSystemRoleToDeveloper(root map[string]any) {
	inputResult, ok := jsonutil.Array(root, "input")
	if !ok {
		return
	}

	// Directly modify role values for items with "system" role
	for i := 0; i < len(inputResult); i++ {
		item, ok := inputResult[i].(map[string]any)
		if !ok {
			continue
		}
		if role, ok := item["role"].(string); ok && role == "system" {
			item["role"] = "developer"
		}
	}
}
