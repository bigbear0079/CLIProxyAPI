// Package thinking provides unified thinking configuration processing.
package thinking

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// StripThinkingConfig removes thinking configuration fields from request body.
//
// This function is used when a model doesn't support thinking but the request
// contains thinking configuration. The configuration is silently removed to
// prevent upstream API errors.
//
// Parameters:
//   - body: Original request body JSON
//   - provider: Provider name (determines which fields to strip)
//
// Returns:
//   - Modified request body JSON with thinking configuration removed
//   - Original body is returned unchanged if:
//   - body is empty or invalid JSON
//   - provider is unknown
//   - no thinking configuration found
func StripThinkingConfig(body []byte, provider string) []byte {
	if len(body) == 0 {
		return body
	}
	root, errParse := jsonutil.ParseObjectBytes(body)
	if errParse != nil {
		return body
	}

	var paths []string
	switch provider {
	case "claude":
		paths = []string{"thinking", "output_config.effort"}
	case "gemini":
		paths = []string{"generationConfig.thinkingConfig"}
	case "gemini-cli", "antigravity":
		paths = []string{"request.generationConfig.thinkingConfig"}
	case "openai":
		paths = []string{"reasoning_effort"}
	case "kimi":
		paths = []string{
			"reasoning_effort",
			"thinking",
		}
	case "codex":
		paths = []string{"reasoning.effort"}
	case "iflow":
		paths = []string{
			"chat_template_kwargs.enable_thinking",
			"chat_template_kwargs.clear_thinking",
			"reasoning_split",
			"reasoning_effort",
		}
	default:
		return body
	}

	for _, path := range paths {
		_ = jsonutil.Delete(root, path)
	}

	// Avoid leaving an empty output_config object for Claude when effort was the only field.
	if provider == "claude" {
		outputConfig, ok := jsonutil.Get(root, "output_config")
		if ok {
			if object, okObject := outputConfig.(map[string]any); okObject && len(object) == 0 {
				_ = jsonutil.Delete(root, "output_config")
			}
		}
	}
	return jsonutil.MarshalOrOriginal(body, root)
}
