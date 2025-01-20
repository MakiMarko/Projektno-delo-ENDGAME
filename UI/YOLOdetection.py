import cv2
from ultralytics import YOLO
import pygame
from utils import combine_boxes, intersection_over_union
import image_compression
import os

model = YOLO("yolo11n.pt")

def process_frame(image):
    global crossed, no_detection_count

    # Ensure the image is valid
    if image is not None:
        # Resize image if necessary
        height, width = image.shape[0], image.shape[1]
        image_resized = cv2.resize(image, (400, 280))  # Adjust the size if needed

        # Run YOLO detection on the resized image
        results = model.predict(image_resized, classes=[1, 3])  # Class 0 = person, 1 = bicycle, 3 = bike
        #person_boxes = []
        bike_boxes = []
        bicycle_boxes = []

        # Process detection results
        for result in results:
            for det in result.boxes:
                bbox = det.xyxy[0].cpu().numpy()
                if int(det.cls) == 1:
                    bicycle_boxes.append(bbox)
                elif int(det.cls) == 3:
                    bike_boxes.append(bbox)


        # Draw the detected bounding boxes on the image and collect coordinates
        detection_coords = []  # List to store coordinates of detected objects
        for box in bicycle_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green contour
            detection_coords.append((x1, y1, x2, y2))  # Save coordinates of the detection

        for box in bike_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            detection_coords.append((x1, y1, x2, y2))

        # Return the updated frame with bounding boxes and detection coordinates
        return image, detection_coords
    return None, []


if __name__ == '__main__':
    image = cv2.imread('screenshot_26.png')  # Load your image (replace with actual path)
    processed_image, detection_coords = process_frame(image)

    if processed_image is not None:
        cv2.imshow('Processed Image', processed_image)  # Show the image with detections
        print('Detection Coordinates:', detection_coords)  # Print the coordinates of detected objects
    else:
        print("No objects detected.")

