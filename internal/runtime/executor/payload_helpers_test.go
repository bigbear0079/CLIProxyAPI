package executor

import (
	"encoding/json"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
)

func TestApplyPayloadConfigWithRootAppliesDefaultOverrideAndFilter(t *testing.T) {
	cfg := &config.Config{
		Payload: config.PayloadConfig{
			Default: []config.PayloadRule{
				{
					Models: []config.PayloadModelRule{{Name: "gpt-*", Protocol: "openai"}},
					Params: map[string]any{"stream": true},
				},
			},
			Override: []config.PayloadRule{
				{
					Models: []config.PayloadModelRule{{Name: "gpt-*", Protocol: "openai"}},
					Params: map[string]any{"temperature": 0.2},
				},
			},
			Filter: []config.PayloadFilterRule{
				{
					Models: []config.PayloadModelRule{{Name: "gpt-*", Protocol: "openai"}},
					Params: []string{"model"},
				},
			},
		},
	}

	payload := []byte(`{"request":{"model":"ignored","metadata":{"user_id":"existing"}}}`)
	out := applyPayloadConfigWithRoot(cfg, "gpt-5", "openai", "request", payload, nil, "")

	var root map[string]any
	if err := json.Unmarshal(out, &root); err != nil {
		t.Fatalf("json.Unmarshal returned error: %v", err)
	}

	request, ok := root["request"].(map[string]any)
	if !ok {
		t.Fatalf("request = %#v, want map[string]any", root["request"])
	}
	if request["stream"] != true {
		t.Fatalf("request.stream = %#v, want true", request["stream"])
	}
	if request["temperature"] != 0.2 {
		t.Fatalf("request.temperature = %#v, want 0.2", request["temperature"])
	}
	if _, ok := request["model"]; ok {
		t.Fatalf("request.model should have been removed, got %#v", request["model"])
	}
}

func TestApplyPayloadConfigWithRootHandlesRawAndPreservesOriginalDefaults(t *testing.T) {
	cfg := &config.Config{
		Payload: config.PayloadConfig{
			Default: []config.PayloadRule{
				{
					Models: []config.PayloadModelRule{{Name: "gemini-*", Protocol: "gemini"}},
					Params: map[string]any{"generationConfig.topK": 40},
				},
			},
			DefaultRaw: []config.PayloadRule{
				{
					Models: []config.PayloadModelRule{{Name: "gemini-*", Protocol: "gemini"}},
					Params: map[string]any{"generationConfig.responseModalities": `["TEXT"]`},
				},
			},
			OverrideRaw: []config.PayloadRule{
				{
					Models: []config.PayloadModelRule{{Name: "gemini-*", Protocol: "gemini"}},
					Params: map[string]any{"systemInstruction": `{"parts":[{"text":"hello"}]}`},
				},
			},
		},
	}

	payload := []byte(`{"request":{"generationConfig":{},"contents":[]}}`)
	original := []byte(`{"request":{"generationConfig":{"topK":1},"contents":[]}}`)

	out := applyPayloadConfigWithRoot(cfg, "gemini-2.5-pro", "gemini", "request", payload, original, "")

	var root map[string]any
	if err := json.Unmarshal(out, &root); err != nil {
		t.Fatalf("json.Unmarshal returned error: %v", err)
	}

	request, ok := root["request"].(map[string]any)
	if !ok {
		t.Fatalf("request = %#v, want map[string]any", root["request"])
	}
	generationConfig, ok := request["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig = %#v, want map[string]any", request["generationConfig"])
	}
	if _, ok := generationConfig["topK"]; ok {
		t.Fatalf("generationConfig.topK should not be defaulted when present in original payload")
	}
	responseModalities, ok := generationConfig["responseModalities"].([]any)
	if !ok {
		t.Fatalf("generationConfig.responseModalities = %#v, want []any", generationConfig["responseModalities"])
	}
	if len(responseModalities) != 1 || responseModalities[0] != "TEXT" {
		t.Fatalf("generationConfig.responseModalities = %#v, want [TEXT]", responseModalities)
	}
	systemInstruction, ok := request["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("systemInstruction = %#v, want map[string]any", request["systemInstruction"])
	}
	parts, ok := systemInstruction["parts"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("systemInstruction.parts = %#v, want single-item []any", systemInstruction["parts"])
	}
	part, ok := parts[0].(map[string]any)
	if !ok {
		t.Fatalf("systemInstruction.parts[0] = %#v, want map[string]any", parts[0])
	}
	if part["text"] != "hello" {
		t.Fatalf("systemInstruction.parts[0].text = %#v, want %q", part["text"], "hello")
	}
}
