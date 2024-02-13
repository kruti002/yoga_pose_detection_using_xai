import cv2
import os
import matplotlib.pyplot as plt
import mediapipe as mp

def is_useful_frame(frame, mp_pose):
    """
    Check if a frame contains a specific yoga pose.
    
    Parameters:
    - frame (numpy.ndarray): Image frame.
    - mp_pose (mediapipe.solutions.pose): Pose detection model.
    
    Returns:
    - bool: True if the frame is useful, False otherwise.
    """
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame using the pose detection model
    results = mp_pose.process(frame_rgb)

    # Example: Check if the pose of interest is present
    if results.pose_landmarks is not None:
        # You can add more sophisticated conditions based on pose landmarks
        return True
    else:
        return False

def extract_and_display_and_save_useful_frames_realtime(output_folder, frame_interval=20):
    """
    Extract frames from real-time camera feed, save useful frames as images, and display them using subplots.
    
    Parameters:
    - output_folder (str): Directory to save the extracted images.
    - frame_interval (int): Interval to capture frames.
    """
    # Create the output folder in the current directory
    output_folder_path = os.path.join(os.getcwd(), output_folder)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Open the camera
    cap = cv2.VideoCapture(0)

    frame_count = 0
    extracted_count = 0
    frames = []

    # Initialize mediapipe pose detection
    mp_pose = mp.solutions.pose.Pose()

    while True:
        ret, frame = cap.read()

        # If frame read is unsuccessful, break the loop
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if is_useful_frame(frame, mp_pose):
                img_name = f"useful_frame_{extracted_count}.png"
                img_path = os.path.join(output_folder_path, img_name)
                cv2.imwrite(img_path, frame)
                extracted_count += 1
                print(f"Saved useful frame {extracted_count} as {img_name}")

        frame_count += 1

        # Display frames using subplots
        if extracted_count > 0 and extracted_count % 10 == 0:
            rows = len(frames) // 2 + len(frames) % 2
            fig, axs = plt.subplots(rows, 2, figsize=(10, 10))

            for i, ax in enumerate(axs.ravel()):
                if i < len(frames):
                    ax.imshow(frames[i])
                    ax.set_title(f"Frame {i * frame_interval}")
                    ax.axis('off')
                else:
                    ax.axis('off')

            plt.tight_layout()
            plt.show()
            frames = []

    cap.release()
    cv2.destroyAllWindows()
    mp_pose.close()

# Example usage:
output_dir = "extracted_useful_frames_realtime"
extract_and_display_and_save_useful_frames_realtime(output_dir, frame_interval=20)
