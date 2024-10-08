import os
import cv2
import numpy as np
import tifffile as tiff

def process_videos(directory, background_path, output_dir):
    # Load the background image
    background = tiff.imread(background_path)
    if background is None:
        raise FileNotFoundError(f"Background image not found at {background_path}")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}...")

            # Load the video
            frames = tiff.imread(file_path)

            # Check if frames is a list of frames or a single multi-dimensional array
            if frames.ndim == 3:  # Assuming frames have shape (num_frames, height, width)
                num_frames = frames.shape[0]
            else:
                num_frames = 1  # This handles a single frame TIFF

            # Prepare a list to hold the processed frames
            processed_frames = []

            # Subtract background and save the first 200 frames
            for i in range(num_frames):
                if num_frames > 1:
                    frame = frames[i]
                else:
                    frame = frames  # Single frame case

                if frame is not None:
                    # Subtract background directly since the frames are already arrays
                    processed_frame = cv2.subtract(frame, background)
                    processed_frames.append(processed_frame)

            # Save the processed frames as a new TIFF file
            output_filename = os.path.splitext(filename)[0]  # Removes the extension
            output_file_path = os.path.join(output_dir, f"{output_filename}_8bit.tif")
            tiff.imwrite(output_file_path, np.array(processed_frames, dtype=np.uint8))
            print(f"Saved processed video to {output_file_path}")

if __name__ == "__main__":
    dir_path = './raw_data'
    background_img_path = './AVG_camera_noise.tif'
    output_directory = ('./test_data')
    process_videos(dir_path, background_img_path, output_directory)
