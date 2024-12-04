# Event Calendar Feature

The Event Calendar feature in LLM Vision allows you to maintain a searchable history of all AI vision detections in your Home Assistant instance.

## Setup

1. Go to Home Assistant Settings > Devices & Services
2. Click "Add Integration" and select LLM Vision
3. Choose "Event Calendar" as the provider
4. Configure the retention time:
   - Set to 0 to keep events indefinitely
   - Set to N (any positive number) to auto-delete events after N days

## How It Works

The Event Calendar:
- Creates a calendar entity in Home Assistant
- Automatically records AI detections when enabled
- Categorizes detected objects into predefined labels
- Maintains a searchable history of events
- Supports automatic cleanup of old events

### Event Categories

Events are automatically categorized with labels such as:
- Person
- Delivery/Package
- Car/Vehicle
- Bike/Bicycle
- Bus
- Truck
- Motorcycle
- Dog
- Cat

### Event Information

Each event includes:
- Summary (what was detected)
- Description (full AI response)
- Location (which camera detected it)
- Timestamp
- Duration

## Usage

### Enable Event Recording

Add `remember: true` to any LLM Vision service call:

```yaml
service: llmvision.image_analyzer
data:
  image_entity: camera.front_door
  message: "What do you see in this image?"
  remember: true  # This enables event recording
```

### View Events

Events can be viewed in multiple ways:
1. Home Assistant Calendar dashboard
2. Any calendar card in your dashboards
3. The Events tab in the LLM Vision Events calendar entity

### Storage

Events are stored locally in:
`custom_components/llmvision/events.json`

## Example Automations

### Record Package Deliveries
```yaml
automation:
  trigger:
    platform: state
    entity_id: binary_sensor.front_door_motion
  action:
    service: llmvision.image_analyzer
    data:
      image_entity: camera.front_door
      message: "Is there a package or delivery person?"
      remember: true
```

### Monitor Vehicle Activity
```yaml
automation:
  trigger:
    platform: state
    entity_id: binary_sensor.driveway_motion
  action:
    service: llmvision.image_analyzer
    data:
      image_entity: camera.driveway
      message: "What vehicles do you see?"
      remember: true
```

## Tips

1. Use specific questions in your service calls to get more accurate event categorization
2. Regular cleanup happens automatically based on your retention settings
3. Events can be manually deleted through the calendar interface
4. You can create multiple automations using different cameras and queries
5. Events can be used as triggers for other automations
