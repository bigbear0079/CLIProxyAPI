// Package geminicli implements thinking configuration for Gemini CLI API format.
//
// Gemini CLI uses request.generationConfig.thinkingConfig.* path instead of
// generationConfig.thinkingConfig.* used by standard Gemini API.
package geminicli

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// Applier applies thinking configuration for Gemini CLI API format.
type Applier struct{}

var _ thinking.ProviderApplier = (*Applier)(nil)

// NewApplier creates a new Gemini CLI thinking applier.
func NewApplier() *Applier {
	return &Applier{}
}

func init() {
	thinking.RegisterProvider("gemini-cli", NewApplier())
}

// Apply applies thinking configuration to Gemini CLI request body.
func (a *Applier) Apply(body []byte, config thinking.ThinkingConfig, modelInfo *registry.ModelInfo) ([]byte, error) {
	if thinking.IsUserDefinedModel(modelInfo) {
		return a.applyCompatible(body, config)
	}
	if modelInfo.Thinking == nil {
		return body, nil
	}

	if config.Mode != thinking.ModeBudget && config.Mode != thinking.ModeLevel && config.Mode != thinking.ModeNone && config.Mode != thinking.ModeAuto {
		return body, nil
	}

	// ModeAuto: Always use Budget format with thinkingBudget=-1
	if config.Mode == thinking.ModeAuto {
		return a.applyBudgetFormat(body, config)
	}
	if config.Mode == thinking.ModeBudget {
		return a.applyBudgetFormat(body, config)
	}

	// For non-auto modes, choose format based on model capabilities
	support := modelInfo.Thinking
	if len(support.Levels) > 0 {
		return a.applyLevelFormat(body, config)
	}
	return a.applyBudgetFormat(body, config)
}

func (a *Applier) applyCompatible(body []byte, config thinking.ThinkingConfig) ([]byte, error) {
	if config.Mode != thinking.ModeBudget && config.Mode != thinking.ModeLevel && config.Mode != thinking.ModeNone && config.Mode != thinking.ModeAuto {
		return body, nil
	}

	if config.Mode == thinking.ModeAuto {
		return a.applyBudgetFormat(body, config)
	}

	if config.Mode == thinking.ModeLevel || (config.Mode == thinking.ModeNone && config.Level != "") {
		return a.applyLevelFormat(body, config)
	}

	return a.applyBudgetFormat(body, config)
}

func (a *Applier) applyLevelFormat(body []byte, config thinking.ThinkingConfig) ([]byte, error) {
	root := jsonutil.ParseObjectBytesOrEmpty(body)
	includeThoughts, includeThoughtsSet := readIncludeThoughts(root, "request.generationConfig.thinkingConfig")

	// Remove conflicting fields to avoid both thinkingLevel and thinkingBudget in output
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinkingBudget")
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinking_budget")
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinking_level")
	// Normalize includeThoughts field name to avoid oneof conflicts in upstream JSON parsing.
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.include_thoughts")

	if config.Mode == thinking.ModeNone {
		_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.includeThoughts", false)
		if config.Level != "" {
			_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.thinkingLevel", string(config.Level))
		}
		return jsonutil.MarshalOrOriginal(body, root), nil
	}

	// Only handle ModeLevel - budget conversion should be done by upper layer
	if config.Mode != thinking.ModeLevel {
		return body, nil
	}

	level := string(config.Level)
	_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.thinkingLevel", level)

	// Respect user's explicit includeThoughts setting from original body; default to true if not set
	// Support both camelCase and snake_case variants
	if !includeThoughtsSet {
		includeThoughts = true
	}
	_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.includeThoughts", includeThoughts)
	return jsonutil.MarshalOrOriginal(body, root), nil
}

func (a *Applier) applyBudgetFormat(body []byte, config thinking.ThinkingConfig) ([]byte, error) {
	root := jsonutil.ParseObjectBytesOrEmpty(body)
	includeThoughts, userSetIncludeThoughts := readIncludeThoughts(root, "request.generationConfig.thinkingConfig")

	// Remove conflicting fields to avoid both thinkingLevel and thinkingBudget in output
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinkingLevel")
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinking_level")
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinking_budget")
	// Normalize includeThoughts field name to avoid oneof conflicts in upstream JSON parsing.
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.include_thoughts")

	budget := config.Budget

	// For ModeNone, always set includeThoughts to false regardless of user setting.
	// This ensures that when user requests budget=0 (disable thinking output),
	// the includeThoughts is correctly set to false even if budget is clamped to min.
	if config.Mode == thinking.ModeNone {
		_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.thinkingBudget", budget)
		_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.includeThoughts", false)
		return jsonutil.MarshalOrOriginal(body, root), nil
	}

	// Determine includeThoughts: respect user's explicit setting from original body if provided
	// Support both camelCase and snake_case variants
	if !userSetIncludeThoughts {
		// No explicit setting, use default logic based on mode
		switch config.Mode {
		case thinking.ModeAuto:
			includeThoughts = true
		default:
			includeThoughts = budget > 0
		}
	}

	_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.thinkingBudget", budget)
	_ = jsonutil.Set(root, "request.generationConfig.thinkingConfig.includeThoughts", includeThoughts)
	return jsonutil.MarshalOrOriginal(body, root), nil
}

func readIncludeThoughts(root map[string]any, basePath string) (bool, bool) {
	if includeThoughts, ok := jsonutil.Bool(root, basePath+".includeThoughts"); ok {
		return includeThoughts, true
	}
	if includeThoughts, ok := jsonutil.Bool(root, basePath+".include_thoughts"); ok {
		return includeThoughts, true
	}
	return false, false
}
