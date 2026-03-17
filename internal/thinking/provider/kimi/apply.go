// Package kimi implements thinking configuration for Kimi (Moonshot AI) models.
//
// Kimi models use the OpenAI-compatible reasoning_effort format for enabled thinking
// levels, but use thinking.type=disabled when thinking is explicitly turned off.
package kimi

import (
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// Applier implements thinking.ProviderApplier for Kimi models.
//
// Kimi-specific behavior:
//   - Enabled thinking: reasoning_effort (string levels)
//   - Disabled thinking: thinking.type="disabled"
//   - Supports budget-to-level conversion
type Applier struct{}

var _ thinking.ProviderApplier = (*Applier)(nil)

// NewApplier creates a new Kimi thinking applier.
func NewApplier() *Applier {
	return &Applier{}
}

func init() {
	thinking.RegisterProvider("kimi", NewApplier())
}

// Apply applies thinking configuration to Kimi request body.
//
// Expected output format (enabled):
//
//	{
//	  "reasoning_effort": "high"
//	}
//
// Expected output format (disabled):
//
//	{
//	  "thinking": {
//	    "type": "disabled"
//	  }
//	}
func (a *Applier) Apply(body []byte, config thinking.ThinkingConfig, modelInfo *registry.ModelInfo) ([]byte, error) {
	if thinking.IsUserDefinedModel(modelInfo) {
		return applyCompatibleKimi(body, config)
	}
	if modelInfo.Thinking == nil {
		return body, nil
	}

	var effort string
	switch config.Mode {
	case thinking.ModeLevel:
		if config.Level == "" {
			return body, nil
		}
		effort = string(config.Level)
	case thinking.ModeNone:
		// Respect clamped fallback level for models that cannot disable thinking.
		if config.Level != "" && config.Level != thinking.LevelNone {
			effort = string(config.Level)
			break
		}
		// Kimi requires explicit disabled thinking object.
		return applyDisabledThinking(body)
	case thinking.ModeBudget:
		// Convert budget to level using threshold mapping
		level, ok := thinking.ConvertBudgetToLevel(config.Budget)
		if !ok {
			return body, nil
		}
		effort = level
	case thinking.ModeAuto:
		// Auto mode maps to "auto" effort
		effort = string(thinking.LevelAuto)
	default:
		return body, nil
	}

	if effort == "" {
		return body, nil
	}
	return applyReasoningEffort(body, effort)
}

// applyCompatibleKimi applies thinking config for user-defined Kimi models.
func applyCompatibleKimi(body []byte, config thinking.ThinkingConfig) ([]byte, error) {
	var effort string
	switch config.Mode {
	case thinking.ModeLevel:
		if config.Level == "" {
			return body, nil
		}
		effort = string(config.Level)
	case thinking.ModeNone:
		if config.Level == "" || config.Level == thinking.LevelNone {
			return applyDisabledThinking(body)
		}
		if config.Level != "" {
			effort = string(config.Level)
		}
	case thinking.ModeAuto:
		effort = string(thinking.LevelAuto)
	case thinking.ModeBudget:
		// Convert budget to level
		level, ok := thinking.ConvertBudgetToLevel(config.Budget)
		if !ok {
			return body, nil
		}
		effort = level
	default:
		return body, nil
	}

	return applyReasoningEffort(body, effort)
}

func applyReasoningEffort(body []byte, effort string) ([]byte, error) {
	root := jsonutil.ParseObjectBytesOrEmpty(body)

	if errDeleteThinking := jsonutil.Delete(root, "thinking"); errDeleteThinking != nil {
		return body, fmt.Errorf("kimi thinking: failed to clear thinking object: %w", errDeleteThinking)
	}
	if errSetEffort := jsonutil.Set(root, "reasoning_effort", effort); errSetEffort != nil {
		return body, fmt.Errorf("kimi thinking: failed to set reasoning_effort: %w", errSetEffort)
	}
	return jsonutil.MarshalOrOriginal(body, root), nil
}

func applyDisabledThinking(body []byte) ([]byte, error) {
	root := jsonutil.ParseObjectBytesOrEmpty(body)

	if errDeleteThinking := jsonutil.Delete(root, "thinking"); errDeleteThinking != nil {
		return body, fmt.Errorf("kimi thinking: failed to clear thinking object: %w", errDeleteThinking)
	}
	if errDeleteEffort := jsonutil.Delete(root, "reasoning_effort"); errDeleteEffort != nil {
		return body, fmt.Errorf("kimi thinking: failed to clear reasoning_effort: %w", errDeleteEffort)
	}
	if errSetType := jsonutil.Set(root, "thinking.type", "disabled"); errSetType != nil {
		return body, fmt.Errorf("kimi thinking: failed to set thinking.type: %w", errSetType)
	}
	return jsonutil.MarshalOrOriginal(body, root), nil
}
