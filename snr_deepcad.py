import os
import json
import numpy as np
import cv2
import tifffile as tiff

def convert_to_8bit(frames):
    min_val = np.min(frames)
    max_val = np.max(frames)
    frames_8bit = ((frames - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return frames_8bit

def rectangles_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

def remove_overlapping_flickers(flickers):
    flickers_to_remove = set()
    flicker_ids = list(flickers.keys())
    for i in range(len(flicker_ids)):
        for j in range(i + 1, len(flicker_ids)):
            flicker1 = flickers[flicker_ids[i]]
            flicker2 = flickers[flicker_ids[j]]
            if rectangles_overlap(flicker1['rect'], flicker2['rect']):
                if flicker1['end_frame'] - flicker1['start_frame'] >= flicker2['end_frame'] - flicker2['start_frame']:
                    flickers_to_remove.add(flicker_ids[j])
                else:
                    flickers_to_remove.add(flicker_ids[i])
    for fid in flickers_to_remove:
        del flickers[fid]
    return flickers
def adjust_threshold_and_reprocess(frames, background, initial_threshold, min_size, max_size, min_duration,
                                   max_duration):
    threshold = initial_threshold
    total_flickers = 0
    validated_flickers = []

    # Keep lowering the threshold until the number of flickers is >= 50 or the threshold hits a limit
    while total_flickers < 3 and threshold > 0:
        tracked_flickers, binary_masks, annotated_frames = track_and_validate_flickers(
            frames, background, threshold, min_size, max_size, min_duration, max_duration
        )
        total_flickers = len(tracked_flickers)

        if total_flickers < 3:
            threshold -= 2  # Decrease the threshold to detect more flickers

    # Return the valid flicker results, binary masks, and annotated frames after adjustment
    return tracked_flickers, binary_masks, annotated_frames
def track_and_validate_flickers(frames, background, threshold, min_size, max_size, min_duration, max_duration):
    flicker_id_counter = 1
    validated_flickers = []
    binary_masks = []
    flicker_frames = np.stack([cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB) for frame in frames])
    all_frame_flickers = {}

    for frame_index, frame in enumerate(frames):
        fg_mask = cv2.absdiff(frame.astype(np.uint8), background.astype(np.uint8))
        _, binarized = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
        binary_masks.append(binarized)
        contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size <= area <= max_size:
                x, y, w, h = cv2.boundingRect(contour)
                mask = cv2.drawContours(np.zeros_like(frame), [contour], -1, 255, thickness=cv2.FILLED)
                mean_intensity = cv2.mean(frame, mask=mask)[0]

                match_found = False
                for fid, flicker in list(all_frame_flickers.items()):
                    if frame_index - flicker['end_frame'] <= 5:
                        if np.hypot(flicker['location'][0] - (x + w // 2), flicker['location'][1] - (y + h // 2)) < 10:
                            flicker['intensities'].append(mean_intensity)
                            flicker['end_frame'] = frame_index
                            flicker['count'] += 1
                            match_found = True
                            all_frame_flickers[fid] = flicker

                if not match_found:
                    all_frame_flickers[flicker_id_counter] = {
                        'start_frame': frame_index,
                        'end_frame': frame_index,
                        'location': (x + w // 2, y + h // 2),
                        'intensities': [mean_intensity],
                        'area': area,
                        'count': 1,
                        'flicker_id': flicker_id_counter,
                        'mask': mask,
                        'rect': (x, y, w, h)
                    }
                    flicker_id_counter += 1

    all_frame_flickers = remove_overlapping_flickers(all_frame_flickers)

    for fid, flicker in all_frame_flickers.items():
        if (min_duration <= flicker['end_frame'] - flicker['start_frame'] + 1 <= max_duration)\
                and validate_intensity_pattern(flicker):
            validated_flickers.append(flicker)
            for i in range(flicker['start_frame'], flicker['end_frame'] + 1):
                x, y, w, h = flicker['rect']
                cv2.rectangle(flicker_frames[i], (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color in BGR
    return validated_flickers, binary_masks, flicker_frames

def validate_intensity_pattern(flicker):
    intensities = flicker['intensities']
    return len(intensities) > 1 and intensities[0] < max(intensities) or intensities[-1] < max(intensities)

def calculate_snr(frames, flickers):
    snr_results = []
    for flicker in flickers:
        flicker_frames = frames[flicker['start_frame']:flicker['end_frame']+1]
        flicker_intensities = [frame[flicker['mask'] == 255] for frame in flicker_frames]
        max_intensity = np.max([np.max(intensities) for intensities in flicker_intensities])
        std_dev = np.std([intensity for intensities in flicker_intensities for intensity in intensities])
        snr = max_intensity / std_dev if std_dev != 0 else float('inf')
        snr_results.append(snr)
    average_snr = np.mean(snr_results) if snr_results else 0
    return average_snr, snr_results

def process_tiff_video(tiff_path, output_dir):
    frames = tiff.imread(tiff_path)
    frames_8bit = convert_to_8bit(frames)
    background = np.mean(frames_8bit, axis=0)
    tracked_flickers, binary_masks, annotated_frames = track_and_validate_flickers(
        frames_8bit, background, 50, 20, 1000, 1, 400
    )
    # Save the binary masks and annotated frames
    binary_file_path = os.path.join(output_dir, 'binary_mask.tif')
    annotated_file_path = os.path.join(output_dir, 'annotated.tif')
    json_file_path = os.path.join(output_dir, 'flicker_data.json')
    tiff.imwrite(binary_file_path, np.stack(binary_masks, axis=0).astype(np.uint8))
    tiff.imwrite(annotated_file_path, annotated_frames.astype(np.uint8))

    # Save flicker data to JSON
    flicker_data = [
        {
            "total_number_of_flickers": len(tracked_flickers),
            "Flicker ID": flicker['flicker_id'],
            "Duration": flicker['end_frame'] - flicker['start_frame'] + 1,
            "Location": flicker['location'],
            "Area": flicker['area'],
            "Average Intensity": np.mean(flicker['intensities'])
        }
        for flicker in tracked_flickers
    ]
    with open(json_file_path, 'w') as f:
        json.dump(flicker_data, f, indent=4)
    average_snr, individual_snr = calculate_snr(frames, tracked_flickers)
    snr_data = {'individual_snr': individual_snr}
    snr_file_path = os.path.join(output_dir, 'snr_data.json')
    with open(snr_file_path, 'w') as f:
        json.dump(snr_data, f, indent=4)
    return average_snr

def main(input_dir):
    average_snrs = {}
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            file_path = os.path.join(input_dir, file_name)
            output_dir = os.path.join(input_dir, file_name.split('.')[0] + '_new')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            average_snr = process_tiff_video(file_path, output_dir)
    if average_snr == 0:
        print('Skipping {}'.format(file_path))
    else:
        average_snrs[file_name] = average_snr
    avg_snr_file_path = os.path.join(input_dir, 'average_snr_per_video.json')
    with open(avg_snr_file_path, 'w') as f:
        json.dump(average_snrs, f, indent=4)

if __name__ == "__main__":
    input_dir= 'deepcad'
    main(input_dir)
