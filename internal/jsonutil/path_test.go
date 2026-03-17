package jsonutil

import "testing"

func TestSetAndGetNestedPath(t *testing.T) {
	root := map[string]any{}

	if err := Set(root, "request.generationConfig.temperature", 0.5); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	got, ok := Get(root, "request.generationConfig.temperature")
	if !ok {
		t.Fatal("Get did not find request.generationConfig.temperature")
	}

	gotFloat, ok := got.(float64)
	if !ok || gotFloat != 0.5 {
		t.Fatalf("Get returned %#v, want 0.5", got)
	}
}

func TestSetPathAppendsWithMinusOne(t *testing.T) {
	root := map[string]any{"items": []any{}}

	if err := Set(root, "items.-1.name", "first"); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}
	if err := Set(root, "items.-1.name", "second"); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	itemsValue, ok := root["items"]
	if !ok {
		t.Fatal("items was not created")
	}
	items, ok := itemsValue.([]any)
	if !ok {
		t.Fatalf("items = %#v, want []any", itemsValue)
	}
	if len(items) != 2 {
		t.Fatalf("len(items) = %d, want 2", len(items))
	}

	first, ok := items[0].(map[string]any)
	if !ok {
		t.Fatalf("items[0] = %#v, want map[string]any", items[0])
	}
	second, ok := items[1].(map[string]any)
	if !ok {
		t.Fatalf("items[1] = %#v, want map[string]any", items[1])
	}

	if first["name"] != "first" {
		t.Fatalf("items[0].name = %#v, want %q", first["name"], "first")
	}
	if second["name"] != "second" {
		t.Fatalf("items[1].name = %#v, want %q", second["name"], "second")
	}
}

func TestDeleteRemovesObjectFieldAndArrayItem(t *testing.T) {
	root := map[string]any{
		"tools": []any{
			map[string]any{"name": "alpha"},
			map[string]any{"name": "bravo"},
		},
		"stream": true,
	}

	if err := Delete(root, "stream"); err != nil {
		t.Fatalf("Delete(stream) returned error: %v", err)
	}
	if _, ok := root["stream"]; ok {
		t.Fatal("stream should have been deleted")
	}

	if err := Delete(root, "tools.0"); err != nil {
		t.Fatalf("Delete(tools.0) returned error: %v", err)
	}

	toolsValue, ok := root["tools"]
	if !ok {
		t.Fatal("tools should remain present")
	}
	tools, ok := toolsValue.([]any)
	if !ok {
		t.Fatalf("tools = %#v, want []any", toolsValue)
	}
	if len(tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1", len(tools))
	}

	tool, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("tools[0] = %#v, want map[string]any", tools[0])
	}
	if tool["name"] != "bravo" {
		t.Fatalf("tools[0].name = %#v, want %q", tool["name"], "bravo")
	}
}

func TestSetAndGetEscapedDotPath(t *testing.T) {
	root := map[string]any{}

	if err := Set(root, `schema.properties.foo\.bar.type`, "string"); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	got, ok := Get(root, `schema.properties.foo\.bar.type`)
	if !ok {
		t.Fatal("Get did not find escaped dot path")
	}
	if got != "string" {
		t.Fatalf("Get returned %#v, want %q", got, "string")
	}

	properties, ok := Get(root, "schema.properties")
	if !ok {
		t.Fatal("schema.properties not found")
	}
	propertiesMap, ok := properties.(map[string]any)
	if !ok {
		t.Fatalf("schema.properties = %#v, want map[string]any", properties)
	}
	if _, ok := propertiesMap["foo.bar"]; !ok {
		t.Fatal(`properties["foo.bar"] not found`)
	}
}

func TestNumericObjectKeyPreservedInObjectContext(t *testing.T) {
	root := map[string]any{
		"schema": map[string]any{
			"properties": map[string]any{},
		},
	}

	if err := Set(root, "schema.properties.123.type", "string"); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	got, ok := Get(root, "schema.properties.123.type")
	if !ok {
		t.Fatal("Get did not find numeric object key path")
	}
	if got != "string" {
		t.Fatalf("Get returned %#v, want %q", got, "string")
	}

	properties, ok := Get(root, "schema.properties")
	if !ok {
		t.Fatal("schema.properties not found")
	}
	propertiesMap, ok := properties.(map[string]any)
	if !ok {
		t.Fatalf("schema.properties = %#v, want map[string]any", properties)
	}
	if _, ok := propertiesMap["123"]; !ok {
		t.Fatal(`properties["123"] not found`)
	}

	if err := Delete(root, "schema.properties.123"); err != nil {
		t.Fatalf("Delete returned error: %v", err)
	}
	if _, ok := Get(root, "schema.properties.123.type"); ok {
		t.Fatal("Delete should remove numeric object key path")
	}
	if _, ok := propertiesMap["123"]; ok {
		t.Fatal(`properties["123"] should have been deleted`)
	}
}

func TestMinusOneKeyPreservedInObjectContext(t *testing.T) {
	root := map[string]any{
		"params": map[string]any{},
	}

	if err := Set(root, "params.-1.name", "sentinel"); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	got, ok := Get(root, "params.-1.name")
	if !ok {
		t.Fatal("Get did not find -1 object key path")
	}
	if got != "sentinel" {
		t.Fatalf("Get returned %#v, want %q", got, "sentinel")
	}

	if err := Delete(root, "params.-1"); err != nil {
		t.Fatalf("Delete returned error: %v", err)
	}
	if _, ok := Get(root, "params.-1.name"); ok {
		t.Fatal("Delete should remove -1 object key path")
	}

	paramsValue, ok := root["params"]
	if !ok {
		t.Fatal("params not found")
	}
	paramsMap, ok := paramsValue.(map[string]any)
	if !ok {
		t.Fatalf("params = %#v, want map[string]any", paramsValue)
	}
	if _, ok := paramsMap["-1"]; ok {
		t.Fatal(`params["-1"] should have been deleted`)
	}
}
