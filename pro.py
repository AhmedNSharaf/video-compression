import numpy as np
import cv2
from scipy.fft import dct, idct
import huffman
import os
from glob import glob
import re

# Step 1: Video Input Handling - With Tracing
def load_video_frames(input_path, num_frames=60):
    print(f"[TRACE] Step 1: Loading frames from {input_path}...")
    frames = []
    if input_path.endswith(('.mp4', '.avi', '.mov')):
        if not os.path.exists(input_path):
            raise ValueError(f"Video file not found at: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        frame_count = 0
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (480, 270))
            frames.append(frame)
            frame_count += 1
        cap.release()
        if len(frames) < num_frames:
            raise ValueError(f"Video has only {len(frames)} frames, need {num_frames}!")
        print(f"[TRACE] Loaded {len(frames)} frames from video.")
    else:
        if not os.path.isdir(input_path):
            raise ValueError(f"Directory not found: {input_path}")
        
        image_files = sorted(glob(os.path.join(input_path, "*.jpg")), 
                           key=lambda x: int(re.search(r'(\d+)\.jpg$', x).group(1)))[:num_frames]
        
        if not image_files:
            all_files = glob(os.path.join(input_path, "*"))
            if not all_files:
                raise ValueError(f"No files found in directory: {input_path}")
            else:
                raise ValueError(f"No files matching '*.jpg' found in {input_path}. Found files: {all_files}")
        
        if len(image_files) < num_frames:
            raise ValueError(f"Found only {len(image_files)} frames, need {num_frames}!")
        
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                raise ValueError(f"Could not read {img_path}!")
            frame = cv2.resize(frame, (480, 270))
            frames.append(frame)
        print(f"[TRACE] Loaded {len(frames)} frames from directory: {image_files[:5]}...")

    if len(frames) != num_frames:
        raise ValueError(f"Loaded {len(frames)} frames, but expected {num_frames}!")
    return frames

# Step 2: Convert to YUV and Choose I/P Frames - With Tracing
def rgb_to_yuv(frame):
    print(f"[TRACE] Step 2: Converting frame to YUV...")
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    return yuv

def choose_ip_frames(frames):
    print("[TRACE] Step 2: Assigning I/P frame types...")
    frame_types = []
    for i in range(len(frames)):
        frame_types.append('I' if i % 10 == 0 else 'P')
    print(f"[TRACE] Frame types assigned: {frame_types[:5]}... (Total: {len(frame_types)})")
    return frame_types

# Step 3: Intra-frame Compression (I-frame) - With Tracing
def apply_dct_quantize(frame_yuv, block_size=8, quant_factor=10):
    print("[TRACE] Step 3: Applying DCT and quantization for I-frame...")
    height, width, _ = frame_yuv.shape
    compressed = np.zeros_like(frame_yuv, dtype=np.float32)

    for channel in range(3):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = frame_yuv[i:i+block_size, j:j+block_size, channel].astype(np.float32)
                if block.shape != (block_size, block_size):
                    continue
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                dct_block = np.round(dct_block / quant_factor) * quant_factor
                compressed[i:i+block_size, j:j+block_size, channel] = dct_block
    print("[TRACE] DCT and quantization completed for I-frame.")
    return compressed

def inverse_dct_quantize(compressed, block_size=8, quant_factor=10):
    print("[TRACE] Step 3: Inverse DCT and dequantization for I-frame...")
    height, width, channels = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)

    for channel in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = compressed[i:i+block_size, j:j+block_size, channel]
                if block.shape != (block_size, block_size):
                    continue
                block = block / quant_factor
                block = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                decompressed[i:i+block_size, j:j+block_size, channel] = block

    print("[TRACE] Inverse DCT and dequantization completed.")
    return np.clip(decompressed, 0, 255).astype(np.uint8)

# Step 4: Inter-frame Compression (P-frame) - With Tracing
def motion_estimation(prev_frame, curr_frame, block_size=16):
    print("[TRACE] Step 4: Performing motion estimation for P-frame...")
    height, width, channels = curr_frame.shape
    motion_vectors = []
    residuals = np.zeros_like(curr_frame, dtype=np.float32)

    y_curr = curr_frame[:, :, 0]
    y_prev = prev_frame[:, :, 0]

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            curr_block = y_curr[i:i+block_size, j:j+block_size]
            if curr_block.shape != (block_size, block_size):
                continue
            best_match = (0, 0)
            min_diff = float('inf')
            search_range = 8
            for di in range(-search_range, search_range + 1):
                for dj in range(-search_range, search_range + 1):
                    pi, pj = i + di, j + dj
                    if pi < 0 or pi + block_size > height or pj < 0 or pj + block_size > width:
                        continue
                    prev_block = y_prev[pi:pi+block_size, pj:pj+block_size]
                    diff = np.sum((curr_block - prev_block) ** 2)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = (di, dj)
            motion_vectors.append(best_match)
            pi, pj = i + best_match[0], j + best_match[1]
            for channel in range(channels):
                prev_block = prev_frame[pi:pi+block_size, pj:pj+block_size, channel]
                curr_block = curr_frame[i:i+block_size, j:j+block_size, channel]
                residuals[i:i+block_size, j:j+block_size, channel] = curr_block - prev_block

    print(f"[TRACE] Motion estimation completed. Found {len(motion_vectors)} motion vectors.")
    return motion_vectors, residuals

# Step 5: Entropy Coding - With Tracing
def entropy_coding(data):
    print("[TRACE] Step 5: Applying Huffman entropy coding...")
    flat_data = data.flatten().astype(np.int32)
    freq = {}
    for val in flat_data:
        freq[val] = freq.get(val, 0) + 1
    huff_tree = huffman.codebook(freq.items())
    encoded = ''.join(huff_tree[val] for val in flat_data)
    print(f"[TRACE] Huffman coding completed. Encoded length: {len(encoded)} bits.")
    return encoded, huff_tree

# Step 6: Bitstream Formation - Fixed with Separate Index for P-frames
def form_bitstream(frame_types, i_frames_data, p_frames_data):
    print("[TRACE] Step 6: Forming bitstream...")
    print(f"[TRACE] i_frames_data length: {len(i_frames_data)}, p_frames_data length: {len(p_frames_data)}")
    bitstream = []
    i_frame_idx = 0  # Index for I-frames
    p_frame_idx = 0  # Index for P-frames

    for i, frame_type in enumerate(frame_types):
        bitstream.append(f"Frame {i}: {frame_type}")
        if frame_type == 'I':
            if i_frame_idx >= len(i_frames_data):
                raise ValueError(f"[ERROR] Not enough I-frame data at index {i_frame_idx}, total I-frames: {len(i_frames_data)}")
            bitstream.append(i_frames_data[i_frame_idx][0])
            i_frame_idx += 1
        else:
            if p_frame_idx >= len(p_frames_data):
                raise ValueError(f"[ERROR] Not enough P-frame data at index {p_frame_idx}, total P-frames: {len(p_frames_data)}")
            bitstream.append(p_frames_data[p_frame_idx][0])
            p_frame_idx += 1

    bitstream_str = '\n'.join(bitstream)
    print(f"[TRACE] Bitstream formed. Total length: {len(bitstream_str.encode('utf-8'))} bytes.")
    return bitstream_str

# Step 7: Decompress and Save Video - With Tracing
def decompress_frame(frame_type, compressed_data, prev_frame=None, block_size=8, quant_factor=10):
    print(f"[TRACE] Step 7: Decompressing {frame_type}-frame...")
    if frame_type == 'I':
        decompressed = inverse_dct_quantize(compressed_data, block_size, quant_factor)
    else:
        motion_vectors, residuals = compressed_data
        height, width, channels = residuals.shape
        decompressed = np.zeros_like(residuals, dtype=np.uint8)
        mv_idx = 0
        for i in range(0, height, 16):
            for j in range(0, width, 16):
                if mv_idx >= len(motion_vectors):
                    break
                di, dj = motion_vectors[mv_idx]
                pi, pj = i + di, j + dj
                if pi < 0 or pi + 16 > height or pj < 0 or pj + 16 > width:
                    continue
                for channel in range(channels):
                    prev_block = prev_frame[pi:pi+16, pj:pj+16, channel]
                    decompressed[i:i+16, j:j+16, channel] = prev_block + residuals[i:i+16, j:j+16, channel]
                mv_idx += 1
        decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)
    print(f"[TRACE] {frame_type}-frame decompression completed.")
    return decompressed

def save_video(frames, output_path, fps=30):
    print(f"[TRACE] Step 7: Saving video to {output_path}...")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"[TRACE] Video saved successfully.")

# Main Function - With Tracing
def compress_and_save_video(input_path, output_path):
    print("[TRACE] Starting video compression process...")
    frames = load_video_frames(input_path)
    frame_types = choose_ip_frames(frames)

    print("[TRACE] Converting frames to YUV...")
    yuv_frames = [rgb_to_yuv(frame) for frame in frames]
    print(f"[TRACE] Converted {len(yuv_frames)} frames to YUV.")

    i_frames_data = []
    p_frames_data = []
    prev_frame = None
    compressed_frames = []

    print("[TRACE] Starting frame compression...")
    for i, (frame, frame_type) in enumerate(zip(yuv_frames, frame_types)):
        print(f"[TRACE] Processing frame {i+1}/{len(yuv_frames)}: {frame_type}-frame")
        if frame_type == 'I':
            compressed = apply_dct_quantize(frame)
            encoded, huff_tree = entropy_coding(compressed)
            i_frames_data.append((encoded, huff_tree))
            compressed_frames.append((frame_type, compressed))
            prev_frame = inverse_dct_quantize(compressed)
            print(f"[TRACE] Added I-frame to i_frames_data. Total I-frames: {len(i_frames_data)}")
        else:
            motion_vectors, residuals = motion_estimation(prev_frame, frame)
            encoded_mv, huff_tree_mv = entropy_coding(np.array(motion_vectors))
            encoded_res, huff_tree_res = entropy_coding(residuals)
            p_frames_data.append((encoded_mv + encoded_res, (huff_tree_mv, huff_tree_res)))
            compressed_frames.append((frame_type, (motion_vectors, residuals)))
            prev_frame = frame
            print(f"[TRACE] Added P-frame to p_frames_data. Total P-frames: {len(p_frames_data)}")

    bitstream = form_bitstream(frame_types, i_frames_data, p_frames_data)

    print("[TRACE] Decompressing frames to create output video...")
    decompressed_frames = []
    prev_frame = None
    for frame_type, compressed_data in compressed_frames:
        decompressed_yuv = decompress_frame(frame_type, compressed_data, prev_frame)
        decompressed_rgb = cv2.cvtColor(decompressed_yuv, cv2.COLOR_YUV2BGR)
        decompressed_frames.append(decompressed_rgb)
        prev_frame = decompressed_yuv

    save_video(decompressed_frames, output_path)

    print("[TRACE] Calculating compression ratio...")
    original_size = sum(frame.nbytes for frame in frames)
    compressed_size = len(bitstream.encode('utf-8'))
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    print(f"[TRACE] Compression Ratio: {compression_ratio:.2f}")
    print("[TRACE] Video compression process completed!")

if __name__ == "__main__":
    input_path = "frames/"
    output_path = "compressed_video.mp4"
    compress_and_save_video(input_path, output_path)