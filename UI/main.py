from tkinter import Tk, Button, Label, filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO  # Ensure correct import for your YOLO model usage

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Initialize the GUI
root = Tk()
root.title("Video Loader")
root.geometry('1200x500')

# Label to display the video frames
label_video1 = Label(root)
label_video1.grid(row=0, column=0)
label_video2 = Label(root)
label_video2.grid(row=0, column=1)
label_video3 = Label(root)
label_video3.grid(row=0, column=2)

caps = [None, None, None]


def combine_boxes(boxA, boxB):
    x1 = min(boxA[0], boxB[0])
    y1 = min(boxA[1], boxB[1])
    x2 = max(boxA[2], boxB[2])
    y2 = max(boxA[3], boxB[3])
    return x1, y1, x2, y2


def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def process_frame(cap, index):
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (400, 280))
            height = frame.shape[0]
            frame = frame[:int(height * 0.9)]  # Keeping only the top 90% of the frame
            height = frame.shape[0]
            overlay = frame.copy()
            line_color = (0, 0, 0)
            line_thickness = 2
            alpha = 0.3

            if index == 1:  # Middle video
                line_position_1 = int(frame.shape[1] * 0.33)
                line_position_2 = int(frame.shape[1] * 0.66)
                cv2.line(overlay, (line_position_1, 0), (line_position_1, frame.shape[0]), line_color, line_thickness)
                cv2.line(overlay, (line_position_2, 0), (line_position_2, frame.shape[0]), line_color, line_thickness)
            else:  # Other videos
                line_position = int(height * 0.75)
                cv2.line(overlay, (0, line_position), (frame.shape[1], line_position), line_color, line_thickness)

            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Detect objects
            results = model.predict(frame, classes=[0, 1, 3])
            person_boxes = []
            ride_boxes = []

            for result in results:
                for det in result.boxes:
                    bbox = det.xyxy[0].cpu().numpy()
                    if int(det.cls) == 0:
                        person_boxes.append(bbox)
                    elif int(det.cls) in [1, 3]:
                        ride_boxes.append(bbox)

            merged_boxes = []

            for person in person_boxes:
                for ride in ride_boxes:
                    if intersection_over_union(person, ride) > 0.1:
                        combined_box = combine_boxes(person, ride)
                        merged_boxes.append(combined_box)

            # Draw the merged boxes
            for box in merged_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return frame
    return None


def update_frames():
    frames = [process_frame(cap, i) for i, cap in enumerate(caps)]
    labels = [label_video1, label_video2, label_video3]
    for frame, label in zip(frames, labels):
        if frame is not None:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            label.config(image=photo)
            label.image = photo

    # Schedule the next frame update
    root.after(25, update_frames)


def load_video(index):
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if video_path:
        if caps[index] is not None:
            caps[index].release()  # Release previous capture if exists
        caps[index] = cv2.VideoCapture(video_path)


# Create buttons to load videos
button_load1 = Button(root, text="Load Video 1", command=lambda: load_video(0))
button_load1.grid(row=1, column=0, pady=20)
button_load2 = Button(root, text="Load Video 2", command=lambda: load_video(1))
button_load2.grid(row=1, column=1, pady=20)
button_load3 = Button(root, text="Load Video 3", command=lambda: load_video(2))
button_load3.grid(row=1, column=2, pady=20)

# Create a button to start videos
button_start = Button(root, text="Start Videos", command=update_frames)
button_start.grid(row=2, column=1, pady=20)

root.mainloop()
