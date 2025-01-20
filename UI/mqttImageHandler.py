import paho.mqtt.client as mqtt
import base64
from PIL import Image
from io import BytesIO
import YOLOdetection
import cv2
import numpy as np
import json

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, reason_code, properties):
    # This callback is triggered upon successful connection.
    print(f"Connected with result code {reason_code}")
    # Subscribing to the screenshot topic
    client.subscribe("threejs/screenshot")

def send_detection_results(detections, timestamp):
    if len(detections) > 0:
        sphere_state = 'green'  # If detections are found, change sphere to green
    else:
        sphere_state = 'red'  # No detections, change sphere to red
    
    # Send updated sphere state with timestamp
    results = {
        "timestamp": timestamp,
        "sphereState": sphere_state
    }
    detections_json = json.dumps(results)
    mqttc.publish("threejs/detections", detections_json)
    print(f"Sent updated sphere state: {results}")

def send_to_yolo(image, timestamp):
    try:
        # Convert PIL image to OpenCV format for YOLO compatibility
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run YOLO detection (assuming YOLOdetection.detect_objects handles detection)
        processed_image, detections = YOLOdetection.process_frame(image_cv)

        send_detection_results(detections, timestamp)
        # Log or process detections as needed
        if(len(detections) > 0):
            print("Detected objects:", detections)

        return detections
    except Exception as e:
        print(f"Error running YOLO detection: {e}")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    # This callback is triggered when a message is received on a subscribed topic.
    #print(f"Received message on {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        #print(f"Received payload: {payload}") 
        # Decode the image from base64
        base64_image = payload.get("screenshot")
        timestamp = payload.get("timestamp")
        print(f"Received image with timestamp: {timestamp}")

        # Check if the base64 string starts with a data URL prefix (like 'data:image/png;base64,')
        if base64_image.startswith('data:image/png;base64,'):
            base64_image = base64_image.split('data:image/png;base64,')[1]
        
        # Manually pad the base64 string to handle incorrect padding
        padding = len(base64_image) % 4
        if padding != 0:
            base64_image += '=' * (4 - padding)  # Add the necessary padding
        
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_image)
        
        # Load the image from the byte data
        image = Image.open(BytesIO(image_data))
        
        # Optionally, save or display the image
        image.save('received_image.png')  # Save the image as 'received_image.png'
        #image.show()  # Show the image in the default image viewer

        # send the image to YOLO for object detection
        send_to_yolo(image, timestamp)
    
        #print("Image received and displayed successfully.")

    except Exception as e:
        print(f"Error processing image: {e}")

# Create the MQTT client with the latest callback version.
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Assign the callback functions to the client.
mqttc.on_connect = on_connect
mqttc.on_message = on_message

# Connect to the MQTT broker.
mqttc.connect("192.168.0.106", 1883, 60)

# Block and start the MQTT loop to process network traffic and callbacks.
mqttc.loop_forever()
