import cv2
import os
import matplotlib.pyplot as plt

def extract_and_display_frames_realtime(output_folder, frame_interval=20):
    """
    Extract frames from real-time camera feed, save them as images, and display them using subplots.
    
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

    while True:
        ret, frame = cap.read()

        # If frame read is unsuccessful, break the loop
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_name = f"frame_{frame_count}.png"
            img_path = os.path.join(output_folder_path, img_name)
            
            # Convert frame from BGR to RGB for displaying with matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(img_path, frame)
            frames.append(frame_rgb)
            extracted_count += 1
            print(f"Extracted frame {frame_count} as {img_name}")

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

# Example usage:
output_dir = "extracted_frames_realtime"
extract_and_display_frames_realtime(output_dir, frame_interval=20)
