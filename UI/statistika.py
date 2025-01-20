import paho.mqtt.client as mqtt
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# MQTT Configuration
BROKER_ADDRESS = "192.168.56.1"
TOPIC_SCREENSHOT = "threejs/screenshot"
TOPIC_DETECTIONS = "threejs/detections"
MATCH_LIMIT = 100  # Number of matched messages to collect before graphing

# Data storage
screenshot_timestamps = []
detection_data = defaultdict(list)

# Results for graphing
detection_counts = {"red": 0, "green": 0}

# Callback when connecting to the MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe([(TOPIC_SCREENSHOT, 0), (TOPIC_DETECTIONS, 0)])
    else:
        print(f"Failed to connect, return code {rc}")

# Callback when a message is received
def on_message(client, userdata, msg):
    global detection_counts

    try:
        data = json.loads(msg.payload.decode("utf-8"))
        if msg.topic == TOPIC_SCREENSHOT:
            timestamp = data.get("timestamp")
            if timestamp:
                screenshot_timestamps.append(timestamp)
        elif msg.topic == TOPIC_DETECTIONS:
            timestamp = data.get("timestamp")
            sphere_state = data.get("sphere_state")
            if timestamp and sphere_state:
                detection_data[timestamp].append(sphere_state)

        # Check for matches
        matched_count = 0
        for timestamp in screenshot_timestamps:
            if timestamp in detection_data:
                matched_count += 1
                state_list = detection_data[timestamp]
                for state in state_list:
                    if state == "green":
                        detection_counts["green"] += 1
                    elif state == "red":
                        detection_counts["red"] += 1
                screenshot_timestamps.remove(timestamp)  # Remove matched screenshot timestamp
                del detection_data[timestamp]  # Remove matched detection data

        # Graph if we reach the match limit
        if matched_count >= MATCH_LIMIT:
            plot_results()

    except Exception as e:
        print(f"Error processing message: {e}")

# Plot results
def plot_results():
    labels = ["Detections (green)", "No Detections (red)"]
    counts = [detection_counts["green"], detection_counts["red"]]

    plt.bar(labels, counts, color=["green", "red"])
    plt.title("Detections vs No Detections")
    plt.ylabel("Count")
    plt.xlabel("Detection State")
    plt.show()

# Main MQTT Client Setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to MQTT broker and start listening
try:
    client.connect(BROKER_ADDRESS, 1883, 60)
    print("Starting MQTT loop...")
    client.loop_forever()
except KeyboardInterrupt:
    print("Exiting...")
    client.disconnect()
except Exception as e:
    print(f"Error: {e}")
