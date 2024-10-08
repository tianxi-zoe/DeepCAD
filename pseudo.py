import os
import numpy as np
import tifffile as tiff


def normalize_and_convert_to_8bit(image):
    # Normalize the image to the range 0-255
    normalized_image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    return normalized_image


def sum_every_ten_frames(tiff_path, output_path, frames_per_sum=20):
    # Load the TIFF file
    with tiff.TiffFile(tiff_path) as tif:
        # Get the number of frames in the TIFF file
        num_frames = len(tif.pages)

        # Calculate the number of summed frames to be created
        num_summed_frames = num_frames // frames_per_sum

        # List to hold the summed frames
        summed_frames = []

        # Iterate over each group of ten frames
        for i in range(num_summed_frames):
            start_index = i * frames_per_sum
            end_index = start_index + frames_per_sum

            # Initialize an array to hold the sum. Start with zeros in 16-bit to handle overflow.
            summed_image = np.zeros(tif.pages[0].shape, dtype=np.uint16)

            # Sum the frames in the current group
            for j in range(start_index, end_index):
                frame = tif.pages[j].asarray()
                summed_image += frame.astype(np.uint16)

            # Normalize and convert the summed image to 8-bit
            summed_image_8bit = normalize_and_convert_to_8bit(summed_image)

            # Append the summed image to the list of summed frames
            summed_frames.append(summed_image_8bit)

        # Save the summed frames as a new multi-frame TIFF file
        tiff.imwrite(output_path, np.stack(summed_frames), photometric='minisblack')

    print(f"Summed frames saved to {output_path}")


if __name__ == "__main__":
    input_folder = './test_data'  # Path to the folder containing the input TIFF files
    output_folder = './pseudo_data'  # Path to the folder where the output TIFF files will be saved

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each TIFF file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            tiff_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".tif", "_summed.tif"))

            print(f"Processing {filename}...")
            sum_every_ten_frames(tiff_path, output_path)
            print(f"Finished processing {filename}.")
