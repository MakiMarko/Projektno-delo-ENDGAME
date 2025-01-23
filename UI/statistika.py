import paho.mqtt.client as mqtt
import json
import matplotlib.pyplot as plt

# MQTT Configuration
BROKER_ADDRESS = "192.168.0.106"
TOPIC_DETECTIONS = "threejs/detections"
SAMPLE_LIMIT = 50  # Number of samples to collect before graphing

# Results for graphing
detection_counts = {"red": 0, "green": 0}

# Callback when connecting to the MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(TOPIC_DETECTIONS)
    else:
        print(f"Failed to connect, return code {rc}")

# Callback when a message is received
def on_message(client, userdata, msg):
    global detection_counts

    try:
        # Parse the incoming message
        data = json.loads(msg.payload.decode("utf-8"))
        #print(f"Received message: {data}")
        timestamp = data.get("timestamp")
        sphere_state = data.get("sphereState")

        if timestamp and sphere_state:
            print(f"Received detection for timestamp {timestamp} with state: {sphere_state}")

            # Count the detections based on sphere state
            if sphere_state == "green":
                detection_counts["green"] += 1
            elif sphere_state == "red":
                detection_counts["red"] += 1

            # Check if we have reached the SAMPLE_LIMIT
            if detection_counts["green"] + detection_counts["red"] >= SAMPLE_LIMIT:
                client.disconnect()  # Disconnect from the broker
                plot_results()  # Once the limit is reached, plot the results
                client.unsubscribe(TOPIC_DETECTIONS)  # Optionally unsubscribe after collecting the samples

    except Exception as e:
        print(f"Error processing message: {e}")

# Plot results
def plot_results():
    labels = ["Green Detections", "Red Detections"]
    counts = [detection_counts["green"], detection_counts["red"]]

    plt.bar(labels, counts, color=["green", "red"])
    plt.title("Detections Count")
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
