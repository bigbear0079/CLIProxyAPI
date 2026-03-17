// Package antigravity implements thinking configuration for Antigravity API format.
//
// Antigravity uses request.generationConfig.thinkingConfig.* path (same as gemini-cli)
// but requires additional normalization for Claude models:
//   - Ensure thinking budget < max_tokens
//   - Remove thinkingConfig if budget < minimum allowed
package antigravity

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// Applier applies thinking configuration for Antigravity API format.
type Applier struct{}

var _ thinking.ProviderApplier = (*Applier)(nil)

// NewApplier creates a new Antigravity thinking applier.
func NewApplier() *Applier {
	return &Applier{}
}

func init() {
	thinking.RegisterProvider("antigravity", NewApplier())
}

// Apply applies thinking configuration to Antigravity request body.
//
// For Claude models, additional constraints are applied:
//   - Ensure thinking budget < max_tokens
//   - Remove thinkingConfig if budget < minimum allowed
func (a *Applier) Apply(body []byte, config thinking.ThinkingConfig, modelInfo *registry.ModelInfo) ([]byte, error) {
	if thinking.IsUserDefinedModel(modelInfo) {
		return a.applyCompatible(body, config, modelInfo)
	}
	if modelInfo.Thinking == nil {
		return body, nil
	}

	if config.Mode != thinking.ModeBudget && config.Mode != thinking.ModeLevel && config.Mode != thinking.ModeNone && config.Mode != thinking.ModeAuto {
		return body, nil
	}

	isClaude := strings.Contains(strings.ToLower(modelInfo.ID), "claude")

	// ModeAuto: Always use Budget format with thinkingBudget=-1
	if config.Mode == thinking.ModeAuto {
		return a.applyBudgetFormat(body, config, modelInfo, isClaude)
	}
	if config.Mode == thinking.ModeBudget {
		return a.applyBudgetFormat(body, config, modelInfo, isClaude)
	}

	// For non-auto modes, choose format based on model capabilities
	support := modelInfo.Thinking
	if len(support.Levels) > 0 {
		return a.applyLevelFormat(body, config)
	}
	return a.applyBudgetFormat(body, config, modelInfo, isClaude)
}

func (a *Applier) applyCompatible(body []byte, config thinking.ThinkingConfig, modelInfo *registry.ModelInfo) ([]byte, error) {
	if config.Mode != thinking.ModeBudget && config.Mode != thinking.ModeLevel && config.Mode != thinking.ModeNone && config.Mode != thinking.ModeAuto {
		return body, nil
	}

	isClaude := false
	if modelInfo != nil {
		isClaude = strings.Contains(strings.ToLower(modelInfo.ID), "claude")
	}

	if config.Mode == thinking.ModeAuto {
		return a.applyBudgetFormat(body, config, modelInfo, isClaude)
	}

	if config.Mode == thinking.ModeLevel || (config.Mode == thinking.ModeNone && config.Level != "") {
		return a.applyLevelFormat(body, config)
	}

	return a.applyBudgetFormat(body, config, modelInfo, isClaude)
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

func (a *Applier) applyBudgetFormat(body []byte, config thinking.ThinkingConfig, modelInfo *registry.ModelInfo, isClaude bool) ([]byte, error) {
	root := jsonutil.ParseObjectBytesOrEmpty(body)
	includeThoughts, userSetIncludeThoughts := readIncludeThoughts(root, "request.generationConfig.thinkingConfig")

	// Remove conflicting fields to avoid both thinkingLevel and thinkingBudget in output
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinkingLevel")
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinking_level")
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.thinking_budget")
	// Normalize includeThoughts field name to avoid oneof conflicts in upstream JSON parsing.
	_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig.include_thoughts")

	budget := config.Budget

	// Apply Claude-specific constraints first to get the final budget value
	if isClaude && modelInfo != nil {
		removed := false
		budget, removed = a.normalizeClaudeBudget(root, budget, modelInfo)
		// Check if budget was removed entirely
		if removed {
			return jsonutil.MarshalOrOriginal(body, root), nil
		}
	}

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

// normalizeClaudeBudget applies Claude-specific constraints to thinking budget.
//
// It handles:
//   - Ensuring thinking budget < max_tokens
//   - Removing thinkingConfig if budget < minimum allowed
//
// Returns the normalized budget and whether thinkingConfig was removed entirely.
func (a *Applier) normalizeClaudeBudget(root map[string]any, budget int, modelInfo *registry.ModelInfo) (int, bool) {
	if modelInfo == nil {
		return budget, false
	}

	// Get effective max tokens
	effectiveMax, setDefaultMax := a.effectiveMaxTokens(root, modelInfo)
	if effectiveMax > 0 && budget >= effectiveMax {
		budget = effectiveMax - 1
	}

	// Check minimum budget
	minBudget := 0
	if modelInfo.Thinking != nil {
		minBudget = modelInfo.Thinking.Min
	}
	if minBudget > 0 && budget >= 0 && budget < minBudget {
		// Budget is below minimum, remove thinking config entirely
		_ = jsonutil.Delete(root, "request.generationConfig.thinkingConfig")
		return 0, true
	}

	// Set default max tokens if needed
	if setDefaultMax && effectiveMax > 0 {
		_ = jsonutil.Set(root, "request.generationConfig.maxOutputTokens", effectiveMax)
	}

	return budget, false
}

// effectiveMaxTokens returns the max tokens to cap thinking:
// prefer request-provided maxOutputTokens; otherwise fall back to model default.
// The boolean indicates whether the value came from the model default (and thus should be written back).
func (a *Applier) effectiveMaxTokens(root map[string]any, modelInfo *registry.ModelInfo) (max int, fromModel bool) {
	if maxTok, ok := jsonutil.Int64(root, "request.generationConfig.maxOutputTokens"); ok && maxTok > 0 {
		return int(maxTok), false
	}
	if modelInfo != nil && modelInfo.MaxCompletionTokens > 0 {
		return modelInfo.MaxCompletionTokens, true
	}
	return 0, false
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
