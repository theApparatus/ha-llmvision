# OpenRouter Integration

OpenRouter support in LLM Vision allows you to use various AI vision models through a single API, including models from OpenAI, Anthropic, Google, and others.

## Setup

1. Go to Home Assistant Settings > Devices & Services
2. Click "Add Integration" and select LLM Vision
3. Choose "OpenRouter" as the provider
4. Configure:
   - API Key: Your OpenRouter API key
   - Default Model: Choose from available vision models

## Available Models

OpenRouter supports these vision-capable models:
- `openai/gpt-4-vision-preview`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `google/gemini-pro-vision`
- `qwen/qwen-vl-plus`

## Configuration

You can add multiple OpenRouter configurations with different default models. Each configuration will appear in your integrations list with its model name, for example:
- "OpenRouter (openai/gpt-4-vision-preview)"
- "OpenRouter (anthropic/claude-3-opus)"

This allows you to:
1. Use different models for different purposes
2. Compare responses between models
3. Have fallback options if one model is unavailable

## Usage

### Basic Service Call
```yaml
service: llmvision.image_analyzer
data:
  provider: "OpenRouter"
  image_entity: camera.front_door
  message: "What do you see in this image?"
```

### Specify Model in Service Call
```yaml
service: llmvision.image_analyzer
data:
  provider: "OpenRouter"
  model: "anthropic/claude-3-opus"
  image_entity: camera.front_door
  message: "What do you see in this image?"
```

### With Event Recording
```yaml
service: llmvision.image_analyzer
data:
  provider: "OpenRouter"
  image_entity: camera.front_door
  message: "What do you see in this image?"
  remember: true
```

## Example Automations

### Multi-Model Analysis
```yaml
automation:
  trigger:
    platform: state
    entity_id: binary_sensor.front_door_motion
  action:
    - service: llmvision.image_analyzer
      data:
        provider: "OpenRouter"
        model: "openai/gpt-4-vision-preview"
        image_entity: camera.front_door
        message: "What do you see in this image?"
        remember: true
    
    - service: llmvision.image_analyzer
      data:
        provider: "OpenRouter"
        model: "anthropic/claude-3-opus"
        image_entity: camera.front_door
        message: "What do you see in this image?"
        remember: true
```

### Model-Specific Use Cases
```yaml
# Use GPT-4V for general scene description
automation:
  trigger:
    platform: state
    entity_id: binary_sensor.driveway_motion
  action:
    service: llmvision.image_analyzer
    data:
      provider: "OpenRouter"
      model: "openai/gpt-4-vision-preview"
      image_entity: camera.driveway
      message: "Describe everything you see in detail."
      remember: true

# Use Claude-3 for specific analysis
automation:
  trigger:
    platform: state
    entity_id: binary_sensor.security_camera_motion
  action:
    service: llmvision.image_analyzer
    data:
      provider: "OpenRouter"
      model: "anthropic/claude-3-opus"
      image_entity: camera.security
      message: "Are there any security concerns in this image?"
      remember: true
```

## Tips

1. Different models have different strengths:
   - GPT-4V is great for general scene description
   - Claude-3 excels at detailed analysis
   - Gemini Pro is good for object recognition
   - Qwen VL+ offers good performance at lower cost

2. Use the Event Calendar feature to compare responses between different models

3. Consider cost efficiency:
   - Models have different pricing
   - Choose models based on your specific needs
   - Use lower-cost models for frequent checks
   - Save premium models for important analysis

4. API Usage:
   - OpenRouter adds required headers automatically
   - Responses are standardized across models
   - Error handling is consistent with other providers

5. Best Practices:
   - Set appropriate default models for each configuration
   - Use specific prompts for better results
   - Enable event recording for important detections
   - Test different models to find the best fit for your use case
