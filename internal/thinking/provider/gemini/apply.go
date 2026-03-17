// Package gemini implements thinking configuration for Gemini models.
//
// Gemini models have two formats:
//   - Gemini 2.5: Uses thinkingBudget (numeric)
//   - Gemini 3.x: Uses thinkingLevel (string: minimal/low/medium/high)
//     or thinkingBudget=-1 for auto/dynamic mode
//
// Output format is determined by ThinkingConfig.Mode and ThinkingSupport.Levels:
//   - ModeAuto: Always uses thinkingBudget=-1 (both Gemini 2.5 and 3.x)
//   - len(Levels) > 0: Uses thinkingLevel (Gemini 3.x discrete levels)
//   - len(Levels) == 0: Uses thinkingBudget (Gemini 2.5)
package gemini

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// Applier applies thinking configuration for Gemini models.
//
// Gemini-specific behavior:
//   - Gemini 2.5: thinkingBudget format, flash series supports ZeroAllowed
//   - Gemini 3.x: thinkingLevel format, cannot be disabled
//   - Use ThinkingSupport.Levels to decide output format
type Applier struct{}

// NewApplier creates a new Gemini thinking applier.
func NewApplier() *Applier {
	return &Applier{}
}

func init() {
	thinking.RegisterProvider("gemini", NewApplier())
}

// Apply applies thinking configuration to Gemini request body.
//
// Expected output format (Gemini 2.5):
//
//	{
//	  "generationConfig": {
//	    "thinkingConfig": {
//	      "thinkingBudget": 8192,
//	      "includeThoughts": true
//	    }
//	  }
//	}
//
// Expected output format (Gemini 3.x):
//
//	{
//	  "generationConfig": {
//	    "thinkingConfig": {
//	      "thinkingLevel": "high",
//	      "includeThoughts": true
//	    }
//	  }
//	}
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

	// Choose format based on config.Mode and model capabilities:
	// - ModeLevel: use Level format (validation will reject unsupported levels)
	// - ModeNone: use Level format if model has Levels, else Budget format
	// - ModeBudget/ModeAuto: use Budget format
	switch config.Mode {
	case thinking.ModeLevel:
		return a.applyLevelFormat(body, config)
	case thinking.ModeNone:
		// ModeNone: route based on model capability (has Levels or not)
		if len(modelInfo.Thinking.Levels) > 0 {
			return a.applyLevelFormat(body, config)
		}
		return a.applyBudgetFormat(body, config)
	default:
		return a.applyBudgetFormat(body, config)
	}
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
	includeThoughts, includeThoughtsSet := readThinkingIncludeThoughts(root, "generationConfig.thinkingConfig")

	// ModeNone semantics:
	//   - ModeNone + Budget=0: completely disable thinking (not possible for Level-only models)
	//   - ModeNone + Budget>0: forced to think but hide output (includeThoughts=false)
	// ValidateConfig sets config.Level to the lowest level when ModeNone + Budget > 0.

	// Remove conflicting fields to avoid both thinkingLevel and thinkingBudget in output
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.thinkingBudget")
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.thinking_budget")
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.thinking_level")
	// Normalize includeThoughts field name to avoid oneof conflicts in upstream JSON parsing.
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.include_thoughts")

	if config.Mode == thinking.ModeNone {
		_ = jsonutil.Set(root, "generationConfig.thinkingConfig.includeThoughts", false)
		if config.Level != "" {
			_ = jsonutil.Set(root, "generationConfig.thinkingConfig.thinkingLevel", string(config.Level))
		}
		return jsonutil.MarshalOrOriginal(body, root), nil
	}

	// Only handle ModeLevel - budget conversion should be done by upper layer
	if config.Mode != thinking.ModeLevel {
		return body, nil
	}

	level := string(config.Level)
	_ = jsonutil.Set(root, "generationConfig.thinkingConfig.thinkingLevel", level)

	// Respect user's explicit includeThoughts setting from original body; default to true if not set
	// Support both camelCase and snake_case variants
	if !includeThoughtsSet {
		includeThoughts = true
	}
	_ = jsonutil.Set(root, "generationConfig.thinkingConfig.includeThoughts", includeThoughts)
	return jsonutil.MarshalOrOriginal(body, root), nil
}

func (a *Applier) applyBudgetFormat(body []byte, config thinking.ThinkingConfig) ([]byte, error) {
	root := jsonutil.ParseObjectBytesOrEmpty(body)
	includeThoughts, userSetIncludeThoughts := readThinkingIncludeThoughts(root, "generationConfig.thinkingConfig")

	// Remove conflicting fields to avoid both thinkingLevel and thinkingBudget in output
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.thinkingLevel")
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.thinking_level")
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.thinking_budget")
	// Normalize includeThoughts field name to avoid oneof conflicts in upstream JSON parsing.
	_ = jsonutil.Delete(root, "generationConfig.thinkingConfig.include_thoughts")

	budget := config.Budget

	// For ModeNone, always set includeThoughts to false regardless of user setting.
	// This ensures that when user requests budget=0 (disable thinking output),
	// the includeThoughts is correctly set to false even if budget is clamped to min.
	if config.Mode == thinking.ModeNone {
		_ = jsonutil.Set(root, "generationConfig.thinkingConfig.thinkingBudget", budget)
		_ = jsonutil.Set(root, "generationConfig.thinkingConfig.includeThoughts", false)
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

	_ = jsonutil.Set(root, "generationConfig.thinkingConfig.thinkingBudget", budget)
	_ = jsonutil.Set(root, "generationConfig.thinkingConfig.includeThoughts", includeThoughts)
	return jsonutil.MarshalOrOriginal(body, root), nil
}

func readThinkingIncludeThoughts(root map[string]any, basePath string) (bool, bool) {
	if includeThoughts, ok := jsonutil.Bool(root, basePath+".includeThoughts"); ok {
		return includeThoughts, true
	}
	if includeThoughts, ok := jsonutil.Bool(root, basePath+".include_thoughts"); ok {
		return includeThoughts, true
	}
	return false, false
}
