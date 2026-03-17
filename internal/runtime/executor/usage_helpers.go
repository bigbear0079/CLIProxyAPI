package executor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/usage"
)

type usageReporter struct {
	provider    string
	model       string
	authID      string
	authIndex   string
	apiKey      string
	source      string
	requestedAt time.Time
	once        sync.Once
}

func newUsageReporter(ctx context.Context, provider, model string, auth *cliproxyauth.Auth) *usageReporter {
	apiKey := apiKeyFromContext(ctx)
	reporter := &usageReporter{
		provider:    provider,
		model:       model,
		requestedAt: time.Now(),
		apiKey:      apiKey,
		source:      resolveUsageSource(auth, apiKey),
	}
	if auth != nil {
		reporter.authID = auth.ID
		reporter.authIndex = auth.EnsureIndex()
	}
	return reporter
}

func (r *usageReporter) publish(ctx context.Context, detail usage.Detail) {
	r.publishWithOutcome(ctx, detail, false)
}

func (r *usageReporter) publishFailure(ctx context.Context) {
	r.publishWithOutcome(ctx, usage.Detail{}, true)
}

func (r *usageReporter) trackFailure(ctx context.Context, errPtr *error) {
	if r == nil || errPtr == nil {
		return
	}
	if *errPtr != nil {
		r.publishFailure(ctx)
	}
}

func (r *usageReporter) publishWithOutcome(ctx context.Context, detail usage.Detail, failed bool) {
	if r == nil {
		return
	}
	if detail.TotalTokens == 0 {
		total := detail.InputTokens + detail.OutputTokens + detail.ReasoningTokens
		if total > 0 {
			detail.TotalTokens = total
		}
	}
	if detail.InputTokens == 0 && detail.OutputTokens == 0 && detail.ReasoningTokens == 0 && detail.CachedTokens == 0 && detail.TotalTokens == 0 && !failed {
		return
	}
	r.once.Do(func() {
		usage.PublishRecord(ctx, usage.Record{
			Provider:    r.provider,
			Model:       r.model,
			Source:      r.source,
			APIKey:      r.apiKey,
			AuthID:      r.authID,
			AuthIndex:   r.authIndex,
			RequestedAt: r.requestedAt,
			Failed:      failed,
			Detail:      detail,
		})
	})
}

// ensurePublished guarantees that a usage record is emitted exactly once.
// It is safe to call multiple times; only the first call wins due to once.Do.
// This is used to ensure request counting even when upstream responses do not
// include any usage fields (tokens), especially for streaming paths.
func (r *usageReporter) ensurePublished(ctx context.Context) {
	if r == nil {
		return
	}
	r.once.Do(func() {
		usage.PublishRecord(ctx, usage.Record{
			Provider:    r.provider,
			Model:       r.model,
			Source:      r.source,
			APIKey:      r.apiKey,
			AuthID:      r.authID,
			AuthIndex:   r.authIndex,
			RequestedAt: r.requestedAt,
			Failed:      false,
			Detail:      usage.Detail{},
		})
	})
}

func apiKeyFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	ginCtx, ok := ctx.Value("gin").(*gin.Context)
	if !ok || ginCtx == nil {
		return ""
	}
	if v, exists := ginCtx.Get("apiKey"); exists {
		switch value := v.(type) {
		case string:
			return value
		case fmt.Stringer:
			return value.String()
		default:
			return fmt.Sprintf("%v", value)
		}
	}
	return ""
}

func resolveUsageSource(auth *cliproxyauth.Auth, ctxAPIKey string) string {
	if auth != nil {
		provider := strings.TrimSpace(auth.Provider)
		if strings.EqualFold(provider, "gemini-cli") {
			if id := strings.TrimSpace(auth.ID); id != "" {
				return id
			}
		}
		if strings.EqualFold(provider, "vertex") {
			if auth.Metadata != nil {
				if projectID, ok := auth.Metadata["project_id"].(string); ok {
					if trimmed := strings.TrimSpace(projectID); trimmed != "" {
						return trimmed
					}
				}
				if project, ok := auth.Metadata["project"].(string); ok {
					if trimmed := strings.TrimSpace(project); trimmed != "" {
						return trimmed
					}
				}
			}
		}
		if _, value := auth.AccountInfo(); value != "" {
			return strings.TrimSpace(value)
		}
		if auth.Metadata != nil {
			if email, ok := auth.Metadata["email"].(string); ok {
				if trimmed := strings.TrimSpace(email); trimmed != "" {
					return trimmed
				}
			}
		}
		if auth.Attributes != nil {
			if key := strings.TrimSpace(auth.Attributes["api_key"]); key != "" {
				return key
			}
		}
	}
	if trimmed := strings.TrimSpace(ctxAPIKey); trimmed != "" {
		return trimmed
	}
	return ""
}

func jsonObjectAtPaths(root map[string]any, paths ...string) map[string]any {
	for _, path := range paths {
		value, ok := jsonutil.Get(root, path)
		if !ok {
			continue
		}
		object, ok := value.(map[string]any)
		if ok {
			return object
		}
	}
	return nil
}

func jsonStringAtPaths(root map[string]any, paths ...string) string {
	for _, path := range paths {
		value, ok := jsonutil.Get(root, path)
		if !ok || value == nil {
			continue
		}
		switch typed := value.(type) {
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
			out, errMarshal := json.Marshal(typed)
			if errMarshal == nil {
				return string(out)
			}
		}
	}
	return ""
}

func jsonInt64AtPaths(root map[string]any, paths ...string) int64 {
	for _, path := range paths {
		value, ok := jsonutil.Get(root, path)
		if !ok || value == nil {
			continue
		}
		switch typed := value.(type) {
		case json.Number:
			intValue, errInt := typed.Int64()
			if errInt == nil {
				return intValue
			}
		case int64:
			return typed
		case int:
			return int64(typed)
		case float64:
			return int64(typed)
		}
	}
	return 0
}

func parseCodexUsage(data []byte) (usage.Detail, bool) {
	root, errParse := jsonutil.ParseObjectBytes(data)
	if errParse != nil {
		return usage.Detail{}, false
	}
	usageNode := jsonObjectAtPaths(root, "response.usage")
	if usageNode == nil {
		return usage.Detail{}, false
	}
	detail := usage.Detail{
		InputTokens:  jsonInt64AtPaths(usageNode, "input_tokens"),
		OutputTokens: jsonInt64AtPaths(usageNode, "output_tokens"),
		TotalTokens:  jsonInt64AtPaths(usageNode, "total_tokens"),
	}
	detail.CachedTokens = jsonInt64AtPaths(usageNode, "input_tokens_details.cached_tokens")
	detail.ReasoningTokens = jsonInt64AtPaths(usageNode, "output_tokens_details.reasoning_tokens")
	return detail, true
}

func parseOpenAIUsage(data []byte) usage.Detail {
	root, errParse := jsonutil.ParseObjectBytes(data)
	if errParse != nil {
		return usage.Detail{}
	}
	usageNode := jsonObjectAtPaths(root, "usage")
	if usageNode == nil {
		return usage.Detail{}
	}
	detail := usage.Detail{
		InputTokens:  jsonInt64AtPaths(usageNode, "prompt_tokens", "input_tokens"),
		OutputTokens: jsonInt64AtPaths(usageNode, "completion_tokens", "output_tokens"),
		TotalTokens:  jsonInt64AtPaths(usageNode, "total_tokens"),
	}
	detail.CachedTokens = jsonInt64AtPaths(usageNode, "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens")
	detail.ReasoningTokens = jsonInt64AtPaths(usageNode, "completion_tokens_details.reasoning_tokens", "output_tokens_details.reasoning_tokens")
	return detail
}

func parseOpenAIStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 {
		return usage.Detail{}, false
	}
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return usage.Detail{}, false
	}
	usageNode := jsonObjectAtPaths(root, "usage")
	if usageNode == nil {
		return usage.Detail{}, false
	}
	detail := usage.Detail{
		InputTokens:  jsonInt64AtPaths(usageNode, "prompt_tokens"),
		OutputTokens: jsonInt64AtPaths(usageNode, "completion_tokens"),
		TotalTokens:  jsonInt64AtPaths(usageNode, "total_tokens"),
	}
	detail.CachedTokens = jsonInt64AtPaths(usageNode, "prompt_tokens_details.cached_tokens")
	detail.ReasoningTokens = jsonInt64AtPaths(usageNode, "completion_tokens_details.reasoning_tokens")
	return detail, true
}

func parseClaudeUsage(data []byte) usage.Detail {
	root, errParse := jsonutil.ParseObjectBytes(data)
	if errParse != nil {
		return usage.Detail{}
	}
	usageNode := jsonObjectAtPaths(root, "usage")
	if usageNode == nil {
		return usage.Detail{}
	}
	detail := usage.Detail{
		InputTokens:  jsonInt64AtPaths(usageNode, "input_tokens"),
		OutputTokens: jsonInt64AtPaths(usageNode, "output_tokens"),
		CachedTokens: jsonInt64AtPaths(usageNode, "cache_read_input_tokens"),
	}
	if detail.CachedTokens == 0 {
		// fall back to creation tokens when read tokens are absent
		detail.CachedTokens = jsonInt64AtPaths(usageNode, "cache_creation_input_tokens")
	}
	detail.TotalTokens = detail.InputTokens + detail.OutputTokens
	return detail
}

func parseClaudeStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 {
		return usage.Detail{}, false
	}
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return usage.Detail{}, false
	}
	usageNode := jsonObjectAtPaths(root, "usage")
	if usageNode == nil {
		return usage.Detail{}, false
	}
	detail := usage.Detail{
		InputTokens:  jsonInt64AtPaths(usageNode, "input_tokens"),
		OutputTokens: jsonInt64AtPaths(usageNode, "output_tokens"),
		CachedTokens: jsonInt64AtPaths(usageNode, "cache_read_input_tokens"),
	}
	if detail.CachedTokens == 0 {
		detail.CachedTokens = jsonInt64AtPaths(usageNode, "cache_creation_input_tokens")
	}
	detail.TotalTokens = detail.InputTokens + detail.OutputTokens
	return detail, true
}

func parseGeminiFamilyUsageDetail(node map[string]any) usage.Detail {
	detail := usage.Detail{
		InputTokens:     jsonInt64AtPaths(node, "promptTokenCount"),
		OutputTokens:    jsonInt64AtPaths(node, "candidatesTokenCount"),
		ReasoningTokens: jsonInt64AtPaths(node, "thoughtsTokenCount"),
		TotalTokens:     jsonInt64AtPaths(node, "totalTokenCount"),
		CachedTokens:    jsonInt64AtPaths(node, "cachedContentTokenCount"),
	}
	if detail.TotalTokens == 0 {
		detail.TotalTokens = detail.InputTokens + detail.OutputTokens + detail.ReasoningTokens
	}
	return detail
}

func parseGeminiCLIUsage(data []byte) usage.Detail {
	root, errParse := jsonutil.ParseObjectBytes(data)
	if errParse != nil {
		return usage.Detail{}
	}
	node := jsonObjectAtPaths(root, "response.usageMetadata", "response.usage_metadata")
	if node == nil {
		return usage.Detail{}
	}
	return parseGeminiFamilyUsageDetail(node)
}

func parseGeminiUsage(data []byte) usage.Detail {
	root, errParse := jsonutil.ParseObjectBytes(data)
	if errParse != nil {
		return usage.Detail{}
	}
	node := jsonObjectAtPaths(root, "usageMetadata", "usage_metadata")
	if node == nil {
		return usage.Detail{}
	}
	return parseGeminiFamilyUsageDetail(node)
}

func parseGeminiStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 {
		return usage.Detail{}, false
	}
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return usage.Detail{}, false
	}
	node := jsonObjectAtPaths(root, "usageMetadata", "usage_metadata")
	if node == nil {
		return usage.Detail{}, false
	}
	return parseGeminiFamilyUsageDetail(node), true
}

func parseGeminiCLIStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 {
		return usage.Detail{}, false
	}
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return usage.Detail{}, false
	}
	node := jsonObjectAtPaths(root, "response.usageMetadata", "usage_metadata")
	if node == nil {
		return usage.Detail{}, false
	}
	return parseGeminiFamilyUsageDetail(node), true
}

func parseAntigravityUsage(data []byte) usage.Detail {
	root, errParse := jsonutil.ParseObjectBytes(data)
	if errParse != nil {
		return usage.Detail{}
	}
	node := jsonObjectAtPaths(root, "response.usageMetadata", "usageMetadata", "usage_metadata")
	if node == nil {
		return usage.Detail{}
	}
	return parseGeminiFamilyUsageDetail(node)
}

func parseAntigravityStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 {
		return usage.Detail{}, false
	}
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return usage.Detail{}, false
	}
	node := jsonObjectAtPaths(root, "response.usageMetadata", "usageMetadata", "usage_metadata")
	if node == nil {
		return usage.Detail{}, false
	}
	return parseGeminiFamilyUsageDetail(node), true
}

var stopChunkWithoutUsage sync.Map

func rememberStopWithoutUsage(traceID string) {
	stopChunkWithoutUsage.Store(traceID, struct{}{})
	time.AfterFunc(10*time.Minute, func() { stopChunkWithoutUsage.Delete(traceID) })
}

// FilterSSEUsageMetadata removes usageMetadata from SSE events that are not
// terminal (finishReason != "stop"). Stop chunks are left untouched. This
// function is shared between aistudio and antigravity executors.
func FilterSSEUsageMetadata(payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}

	lines := bytes.Split(payload, []byte("\n"))
	modified := false
	foundData := false
	for idx, line := range lines {
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) == 0 || !bytes.HasPrefix(trimmed, []byte("data:")) {
			continue
		}
		foundData = true
		dataIdx := bytes.Index(line, []byte("data:"))
		if dataIdx < 0 {
			continue
		}
		rawJSON := bytes.TrimSpace(line[dataIdx+5:])
		traceID := ""
		if root, errParse := jsonutil.ParseObjectBytes(rawJSON); errParse == nil {
			traceID = jsonStringAtPaths(root, "traceId")
		}
		if isStopChunkWithoutUsage(rawJSON) && traceID != "" {
			rememberStopWithoutUsage(traceID)
			continue
		}
		if traceID != "" {
			if _, ok := stopChunkWithoutUsage.Load(traceID); ok && hasUsageMetadata(rawJSON) {
				stopChunkWithoutUsage.Delete(traceID)
				continue
			}
		}

		cleaned, changed := StripUsageMetadataFromJSON(rawJSON)
		if !changed {
			continue
		}
		var rebuilt []byte
		rebuilt = append(rebuilt, line[:dataIdx]...)
		rebuilt = append(rebuilt, []byte("data:")...)
		if len(cleaned) > 0 {
			rebuilt = append(rebuilt, ' ')
			rebuilt = append(rebuilt, cleaned...)
		}
		lines[idx] = rebuilt
		modified = true
	}
	if !modified {
		if !foundData {
			// Handle payloads that are raw JSON without SSE data: prefix.
			trimmed := bytes.TrimSpace(payload)
			cleaned, changed := StripUsageMetadataFromJSON(trimmed)
			if !changed {
				return payload
			}
			return cleaned
		}
		return payload
	}
	return bytes.Join(lines, []byte("\n"))
}

// StripUsageMetadataFromJSON drops usageMetadata unless finishReason is present (terminal).
// It handles both formats:
// - Aistudio: candidates.0.finishReason
// - Antigravity: response.candidates.0.finishReason
func StripUsageMetadataFromJSON(rawJSON []byte) ([]byte, bool) {
	jsonBytes := bytes.TrimSpace(rawJSON)
	if len(jsonBytes) == 0 {
		return rawJSON, false
	}
	root, errParse := jsonutil.ParseObjectBytes(jsonBytes)
	if errParse != nil {
		return rawJSON, false
	}

	// Check for finishReason in both aistudio and antigravity formats
	terminalReason := strings.TrimSpace(jsonStringAtPaths(root, "candidates.0.finishReason", "response.candidates.0.finishReason")) != ""
	usageMetadataValue, hasUsageMetadata := jsonutil.Get(root, "usageMetadata")
	if !hasUsageMetadata {
		usageMetadataValue, hasUsageMetadata = jsonutil.Get(root, "response.usageMetadata")
	}

	// Terminal chunk: keep as-is.
	if terminalReason {
		return rawJSON, false
	}

	// Nothing to strip
	if !hasUsageMetadata {
		return rawJSON, false
	}
	var changed bool

	if _, ok := jsonutil.Get(root, "usageMetadata"); ok {
		// Rename usageMetadata to cpaUsageMetadata in the message_start event of Claude
		if errSet := jsonutil.Set(root, "cpaUsageMetadata", usageMetadataValue); errSet == nil {
			_ = jsonutil.Delete(root, "usageMetadata")
			changed = true
		}
	}

	if usageMetadataValue, ok := jsonutil.Get(root, "response.usageMetadata"); ok {
		// Rename usageMetadata to cpaUsageMetadata in the message_start event of Claude
		if errSet := jsonutil.Set(root, "response.cpaUsageMetadata", usageMetadataValue); errSet == nil {
			_ = jsonutil.Delete(root, "response.usageMetadata")
			changed = true
		}
	}

	if !changed {
		return rawJSON, false
	}
	return jsonutil.MarshalOrOriginal(rawJSON, root), true
}

func hasUsageMetadata(jsonBytes []byte) bool {
	root, errParse := jsonutil.ParseObjectBytes(jsonBytes)
	if errParse != nil {
		return false
	}
	return jsonutil.Exists(root, "usageMetadata") || jsonutil.Exists(root, "response.usageMetadata")
}

func isStopChunkWithoutUsage(jsonBytes []byte) bool {
	root, errParse := jsonutil.ParseObjectBytes(jsonBytes)
	if errParse != nil {
		return false
	}
	trimmed := strings.TrimSpace(jsonStringAtPaths(root, "candidates.0.finishReason", "response.candidates.0.finishReason"))
	if trimmed == "" {
		return false
	}
	return !hasUsageMetadata(jsonBytes)
}

func jsonPayload(line []byte) []byte {
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 {
		return nil
	}
	if bytes.Equal(trimmed, []byte("[DONE]")) {
		return nil
	}
	if bytes.HasPrefix(trimmed, []byte("event:")) {
		return nil
	}
	if bytes.HasPrefix(trimmed, []byte("data:")) {
		trimmed = bytes.TrimSpace(trimmed[len("data:"):])
	}
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return nil
	}
	return trimmed
}
