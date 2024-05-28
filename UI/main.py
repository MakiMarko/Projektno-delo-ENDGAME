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


def process_frame(cap):
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
            line_position = int(height * 0.75)
            cv2.line(overlay, (0, line_position), (frame.shape[1], line_position), line_color, line_thickness)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Detect objects
            results = model.predict(frame, classes=[0, 1, 3])
            if results:
                frame = results[0].plot()
            return frame
    return None


def update_frames():
    frames = [process_frame(cap) for cap in caps]
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
