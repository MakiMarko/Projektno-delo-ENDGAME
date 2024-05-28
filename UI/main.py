from tkinter import Tk, Button, filedialog
import cv2
from ultralytics import YOLO  # Ensure correct import for your YOLO model usage

# Initialize the YOLO model
model = YOLO("yolov8n.pt")
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


# Function to process and update the video frame
def update_frame(label_video):
    for i, cap in enumerate(caps):
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 9)  # Skip frames
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (400, 280))

            # Frame processing
            height = frame.shape[0]
            frame = frame[:int(height * 0.8)]  # Keeping only the top 80% of the frame
            height = frame.shape[0]
            overlay = frame.copy()
            line_color = (0, 0, 0)
            line_thickness = 2
            alpha = 0.3
            # Adjust line positions for a taller frame
            line_positions = [
                int(height * 0.9), int(height * 0.8), int(height * 0.7),
                int(height * 0.6), int(height * 0.5), int(height * 0.4),
                int(height * 0.3), int(height * 0.2), int(height * 0.1)
            ]
            for pos in line_positions:
                cv2.line(overlay, (0, pos), (frame.shape[1], pos), line_color, line_thickness)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Detect objects
            results = model.predict(frame, classes=[0, 1, 3])
            if results:
                frame = results[0].plot()

            # Display the frame
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            label_video.config(image=photo)
            label_video.image = photo
            # label_video.after(25, update_frame, cap, label_video)

    # Schedule the next frame update
    root.after(25, update_frame)


def load_video(index):
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if video_path:
        if caps[index] is not None:
            caps[index].release()  # Release previous capture if exists
        caps[index] = cv2.VideoCapture(video_path)
        update_frame([label_video1, label_video2, label_video3][index])


# Create a button to load video
button_load1 = Button(root, text="Load Video 1", command=lambda: load_video(0))
button_load1.grid(row=1, column=0, pady=20)
button_load2 = Button(root, text="Load Video 2", command=lambda: load_video(1))
button_load2.grid(row=1, column=1, pady=20)
button_load3 = Button(root, text="Load Video 3", command=lambda: load_video(2))
button_load3.grid(row=1, column=2, pady=20)

root.mainloop()

# Function to load and process video
def load_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)

    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            # Resize frame to make it smaller
            frame = cv2.resize(frame, (640, 360))  # Resize to width=640, height=360

            # Cut the bottom 25% of the video frame
            height = frame.shape[0]
            frame = frame[:int(height * 0.80)]  # Keep only the top 75% of the frame
            height = frame.shape[0]  # Update the height after cropping

            # Add semi-transparent horizontal lines lower on the frame
            line_color = (0, 0, 0)  # Green color for the lines
            line_thickness = 2
            alpha = 0.3  # Transparency factor
            overlay = frame.copy()
            line_positions = [int(height * 0.9), int(height * 0.8),
                              int(height * 0.7)]  # Positions at 60%, 70%, and 80% height

            for pos in line_positions:
                cv2.line(overlay, (0, pos), (frame.shape[1], pos), line_color, line_thickness)

            # Blend the overlay with the original frame
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Detect objects of specific classes (0 - person, 1 - bicycle, 3 - motorcycle)
            results = model.predict(frame, classes=[0, 1, 3])

            if results:
                frame_ = results[
                    0].plot()  # Assuming 'results' can be handled like this; adjust according to your actual object

                # Visualize
                cv2.imshow('Video Display', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


# Initialize the GUI
root = Tk()
root.title("Video Loader")
root.geometry('800x600')  # Set the size of the window to 800x600 pixels

# Create a button to load video
btn_load = Button(root, text="Load Video", command=load_video)
btn_load.pack(pady=20)

root.mainloop()
