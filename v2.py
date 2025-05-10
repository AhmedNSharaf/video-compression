import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
import os
from bitstring import BitArray, BitStream, ReadError
import threading
from PIL import Image, ImageTk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Simple Huffman coding implementation
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def build_huffman_tree(freq_dict):
    nodes = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]
    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        nodes.append(parent)
    return nodes[0] if nodes else None

def build_huffman_codes(root, current_code="", codes=None):
    if codes is None:
        codes = {}
    if root:
        if root.char is not None:
            codes[root.char] = current_code if current_code else "0"
        build_huffman_codes(root.left, current_code + "0", codes)
        build_huffman_codes(root.right, current_code + "1", codes)
    return codes

def huffman_codebook(freqs):
    tree = build_huffman_tree(freqs)
    codes = build_huffman_codes(tree)
    # Ensure EOB (0) is included
    if 0 not in codes:
        codes[0] = '0'  # Default code if not present
    return codes

class VideoCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Compression App")
        self.video_path = None
        self.compression_thread = None
        self.playback_active = False
        self.completion_event = threading.Event()

        # GUI Components
        tk.Label(root, text="Video Compression").pack(pady=10)
        tk.Button(root, text="Select Video", command=self.select_video).pack(pady=5)
        tk.Button(root, text="Compress Video", command=self.start_compression).pack(pady=5)
        self.progress = Progressbar(root, length=200, mode="determinate")
        self.progress.pack(pady=5)
        self.status_label = tk.Label(root, text="Status: Idle")
        self.status_label.pack(pady=10)

        # Video display frame
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(pady=10)
        self.original_label = tk.Label(self.video_frame, text="Original Video")
        self.original_label.grid(row=0, column=0, padx=5)
        self.compressed_label = tk.Label(self.video_frame, text="Compressed Video")
        self.compressed_label.grid(row=0, column=1, padx=5)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.status_label.config(text=f"Status: Video selected - {os.path.basename(self.video_path)}")

    def start_compression(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first!")
            return
        if self.compression_thread and self.compression_thread.is_alive():
            messagebox.showwarning("Warning", "Compression is already in progress!")
            return

        self.status_label.config(text="Status: Starting compression...")
        self.progress["value"] = 0
        self.root.update()
        self.completion_event.clear()
        self.compression_thread = threading.Thread(target=self.compress_video_thread)
        self.compression_thread.start()
        self.check_completion()

    def check_completion(self):
        if not self.completion_event.is_set() and self.compression_thread.is_alive():
            self.root.after(100, self.check_completion)
        else:
            self.handle_post_compression()

    def compress_video_thread(self):
        compressed_file = "compressed_video.bin"
        reconstructed_file = "reconstructed_video.avi"
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        def update_progress(value):
            self.progress["value"] = value
            self.root.update_idletasks()

        try:
            self.compress(self.video_path, compressed_file, update_progress, frame_count)
            if os.path.exists(compressed_file):
                logging.info(f"Compressed bitstream size: {os.path.getsize(compressed_file)} bytes")
            else:
                logging.error("Compressed bitstream file not created")
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to create compressed bitstream."))
                return

            self.decompress(compressed_file, reconstructed_file)

            if os.path.exists(reconstructed_file) and os.path.getsize(reconstructed_file) > 0:
                original_size = os.path.getsize(self.video_path)
                compressed_size = os.path.getsize(compressed_file)
                compression_ratio = original_size / compressed_size
                psnr = self.calculate_psnr(self.video_path, reconstructed_file)
                logging.info(f"Compression complete. Ratio: {compression_ratio:.2f}, PSNR: {psnr:.2f}")
            else:
                logging.error("Compressed video file not created or empty")
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to create compressed video."))
                return
        except Exception as e:
            logging.error(f"Compression error: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Compression failed: {e}"))
        finally:
            self.completion_event.set()

    def handle_post_compression(self):
        reconstructed_file = "reconstructed_video.avi"
        if os.path.exists(reconstructed_file) and os.path.getsize(reconstructed_file) > 0:
            self.play_videos(self.video_path, reconstructed_file)
        else:
            self.status_label.config(text="Status: Compression failed - No output video.")
            messagebox.showerror("Error", "Compressed video not found or empty.")

    def play_videos(self, original_path, compressed_path):
        if self.playback_active:
            return
        self.playback_active = True

        cap_original = cv2.VideoCapture(original_path)
        cap_compressed = cv2.VideoCapture(compressed_path)
        if not cap_original.isOpened() or not cap_compressed.isOpened():
            self.status_label.config(text="Status: Error opening videos.")
            messagebox.showerror("Error", "Could not open one or both videos.")
            self.playback_active = False
            return

        fps = cap_original.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)
        width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_factor = 300 / max(width, height)
        display_width = int(width * scale_factor)
        display_height = int(height * scale_factor)

        def update_frame():
            if not self.playback_active:
                cap_original.release()
                cap_compressed.release()
                return

            ret_orig, frame_orig = cap_original.read()
            ret_comp, frame_comp = cap_compressed.read()

            if not ret_orig or not ret_comp:
                cap_original.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_compressed.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(delay, update_frame)
                return

            frame_orig = cv2.resize(frame_orig, (display_width, display_height), interpolation=cv2.INTER_AREA)
            frame_comp = cv2.resize(frame_comp, (display_width, display_height), interpolation=cv2.INTER_AREA)
            frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
            frame_comp = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)

            image_orig = Image.fromarray(frame_orig)
            image_comp = Image.fromarray(frame_comp)
            photo_orig = ImageTk.PhotoImage(image_orig)
            photo_comp = ImageTk.PhotoImage(image_comp)

            self.original_label.configure(image=photo_orig)
            self.compressed_label.configure(image=photo_comp)
            self.original_label.image = photo_orig
            self.compressed_label.image = photo_comp

            self.root.after(delay, update_frame)

        self.status_label.config(text="Status: Playing videos.")
        update_frame()

    def stop_playback(self):
        self.playback_active = False

    def compress(self, input_path, output_path, update_progress, total_frames):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        adjusted_width = ((width + 15) // 16) * 16
        adjusted_height = ((height + 15) // 16) * 16

        bitstream = BitStream()
        bitstream.append(BitArray(uint=width, length=16))
        bitstream.append(BitArray(uint=height, length=16))
        bitstream.append(BitArray(uint=int(fps), length=16))
        bitstream.append(BitArray(uint=frame_count, length=32))

        frame_idx = 0
        prev_reconstructed = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            u_sub = cv2.resize(u, (adjusted_width//2, adjusted_height//2), interpolation=cv2.INTER_AREA)
            v_sub = cv2.resize(v, (adjusted_width//2, adjusted_height//2), interpolation=cv2.INTER_AREA)

            is_iframe = (frame_idx % 10 == 0)
            bitstream.append(BitArray(bool=is_iframe, length=1))

            if is_iframe:
                encoded_y, recon_y = self.encode_iframe(y)
                encoded_u, recon_u = self.encode_iframe(u_sub)
                encoded_v, recon_v = self.encode_iframe(v_sub)
                bitstream.append(encoded_y)
                bitstream.append(encoded_u)
                bitstream.append(encoded_v)
                prev_reconstructed = (recon_y, cv2.resize(recon_u, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA), 
                                   cv2.resize(recon_v, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA))
            else:
                encoded_y, recon_y, mv_y = self.encode_pframe(y, prev_reconstructed[0])
                encoded_u, recon_u, mv_u = self.encode_pframe(u_sub, prev_reconstructed[1][::2, ::2])
                encoded_v, recon_v, mv_v = self.encode_pframe(v_sub, prev_reconstructed[2][::2, ::2])
                bitstream.append(mv_y + encoded_y)
                bitstream.append(mv_u + encoded_u)
                bitstream.append(mv_v + encoded_v)
                prev_reconstructed = (recon_y, cv2.resize(recon_u, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA), 
                                   cv2.resize(recon_v, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA))

            frame_idx += 1
            update_progress((frame_idx / total_frames) * 100)
            logging.info(f"After encoding frame {frame_idx}, bitstream length: {len(bitstream)} bits")

        cap.release()
        with open(output_path, 'wb') as f:
            bitstream.tofile(f)

    def encode_iframe(self, channel):
        blocks = self.to_blocks(channel, 8)
        all_symbols = []
        encoded_blocks = []
        recon_blocks = []

        for block in blocks:
            dct_block = dctn(block, norm='ortho')
            quantized = np.round(dct_block / 10).astype(int)
            zigzagged = self.zigzag(quantized)
            last_nonzero = np.max(np.where(zigzagged != 0)[0] + 1) if np.any(zigzagged) else 0
            for val in zigzagged[:last_nonzero]:
                all_symbols.append(abs(val) + 1)
            all_symbols.append(0)  # EOB

        freqs = np.bincount(all_symbols)
        huff_tree = huffman_codebook({i: freq for i, freq in enumerate(freqs) if freq > 0})

        bit_data = BitArray()
        bit_data.append(BitArray(uint=len(huff_tree), length=16))
        for sym, code in huff_tree.items():
            bit_data.append(BitArray(uint=sym, length=16))
            bit_data.append(BitArray(uint=len(code), length=8))
            bit_data.append(BitArray(bin=code))

        for block in blocks:
            dct_block = dctn(block, norm='ortho')
            quantized = np.round(dct_block / 10).astype(int)
            zigzagged = self.zigzag(quantized)
            last_nonzero = np.max(np.where(zigzagged != 0)[0] + 1) if np.any(zigzagged) else 0
            for val in zigzagged[:last_nonzero]:
                symbol = abs(val) + 1
                bit_data.append(BitArray(bin=huff_tree[symbol]))
                if val != 0:
                    bit_data.append(BitArray(bool=(val < 0), length=1))
            bit_data.append(BitArray(bin=huff_tree[0]))  # EOB

            coeffs = []
            for val in zigzagged[:last_nonzero]:
                coeffs.append(val)
            coeffs.extend([0] * (64 - len(coeffs)))
            dequantized = self.inverse_zigzag(np.array(coeffs), 8, 8) * 10
            recon_block = idctn(dequantized, norm='ortho')
            recon_blocks.append(recon_block)

        recon_channel = self.from_blocks(recon_blocks, channel.shape, 8)
        return bit_data, recon_channel

    def encode_pframe(self, channel, prev_channel):
        macroblock_size = 16
        blocks = self.to_blocks(channel, macroblock_size)
        prev_blocks = self.to_blocks(prev_channel, macroblock_size)
        motion_vectors = []
        residuals = []

        for i, block in enumerate(blocks):
            mv, pred_block = self.block_matching(block, prev_blocks[i], macroblock_size)
            residual = block - pred_block
            residuals.append(residual)
            motion_vectors.append(mv)

        encoded_residuals = BitArray()
        recon_residuals = []
        all_symbols = []

        for residual in residuals:
            dct_block = dctn(residual, norm='ortho')
            quantized = np.round(dct_block / 10).astype(int)
            zigzagged = self.zigzag(quantized)
            last_nonzero = np.max(np.where(zigzagged != 0)[0] + 1) if np.any(zigzagged) else 0
            for val in zigzagged[:last_nonzero]:
                all_symbols.append(abs(val) + 1)
            all_symbols.append(0)  # EOB

        freqs = np.bincount(all_symbols)
        huff_tree = huffman_codebook({i: freq for i, freq in enumerate(freqs) if freq > 0})

        encoded_residuals.append(BitArray(uint=len(huff_tree), length=16))
        for sym, code in huff_tree.items():
            encoded_residuals.append(BitArray(uint=sym, length=16))
            encoded_residuals.append(BitArray(uint=len(code), length=8))
            encoded_residuals.append(BitArray(bin=code))

        for residual in residuals:
            dct_block = dctn(residual, norm='ortho')
            quantized = np.round(dct_block / 10).astype(int)
            zigzagged = self.zigzag(quantized)
            last_nonzero = np.max(np.where(zigzagged != 0)[0] + 1) if np.any(zigzagged) else 0
            for val in zigzagged[:last_nonzero]:
                symbol = abs(val) + 1
                encoded_residuals.append(BitArray(bin=huff_tree[symbol]))
                if val != 0:
                    encoded_residuals.append(BitArray(bool=(val < 0), length=1))
            encoded_residuals.append(BitArray(bin=huff_tree[0]))  # EOB

            coeffs = []
            for val in zigzagged[:last_nonzero]:
                coeffs.append(val)
            coeffs.extend([0] * (macroblock_size * macroblock_size - len(coeffs)))
            dequantized = self.inverse_zigzag(np.array(coeffs), macroblock_size, macroblock_size) * 10
            recon_residual = idctn(dequantized, norm='ortho')
            recon_residuals.append(recon_residual)

        mv_data = BitArray()
        for mv_x, mv_y in motion_vectors:
            mv_x = max(-16, min(15, mv_x)) + 16
            mv_y = max(-16, min(15, mv_y)) + 16
            mv_data.append(BitArray(uint=mv_x, length=5))
            mv_data.append(BitArray(uint=mv_y, length=5))

        recon_channel = self.from_blocks([pred + res for pred, res in zip([self.apply_motion(prev_blocks[i], mv, macroblock_size) for i, mv in enumerate(motion_vectors)], recon_residuals)], channel.shape, macroblock_size)
        return encoded_residuals, recon_channel, mv_data

    def decompress(self, input_path, output_path):
        bitstream = BitStream()
        try:
            with open(input_path, 'rb') as f:
                bitstream = BitStream(bytes=f.read())
            logging.info(f"Bitstream loaded, size: {os.path.getsize(input_path)} bytes, total bits: {bitstream.len}")
        except Exception as e:
            logging.error(f"Failed to read bitstream: {e}")
            raise

        try:
            width = bitstream.read('uint:16')
            height = bitstream.read('uint:16')
            fps = bitstream.read('uint:16')
            frame_count = bitstream.read('uint:32')
            logging.info(f"Decoded header: width={width}, height={height}, fps={fps}, frame_count={frame_count}")
        except ReadError as e:
            logging.error(f"Bitstream too short to read header: {e}, bits remaining: {bitstream.len - bitstream.pos}")
            raise
        except Exception as e:
            logging.error(f"Failed to read bitstream header: {e}")
            raise

        adjusted_width = ((width + 15) // 16) * 16
        adjusted_height = ((height + 15) // 16) * 16

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logging.error("Video writer failed to initialize")
            raise RuntimeError("Failed to initialize video writer")

        prev_reconstructed = None
        frames_written = 0
        for frame_idx in range(frame_count):
            try:
                if bitstream.len - bitstream.pos < 1:
                    logging.error(f"Bitstream exhausted at frame {frame_idx}, bits remaining: {bitstream.len - bitstream.pos}")
                    break
                is_iframe = bitstream.read('bool')
                logging.debug(f"Processing frame {frame_idx}, is_iframe={is_iframe}, bits remaining: {bitstream.len - bitstream.pos}")
            except ReadError as e:
                logging.error(f"Bitstream read error at frame {frame_idx}: {e}, bits remaining: {bitstream.len - bitstream.pos}")
                break
            except Exception as e:
                logging.error(f"Unexpected error reading frame type at frame {frame_idx}: {e}")
                break

            if is_iframe:
                try:
                    y, huff_tree_y = self.decode_iframe(bitstream, (adjusted_height, adjusted_width))
                    u, huff_tree_u = self.decode_iframe(bitstream, (adjusted_height//2, adjusted_width//2))
                    v, huff_tree_v = self.decode_iframe(bitstream, (adjusted_height//2, adjusted_width//2))
                    u = cv2.resize(u, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
                    v = cv2.resize(v, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
                    y = np.clip(y, 0, 255).astype(np.uint8)
                    u = np.clip(u, 0, 255).astype(np.uint8)
                    v = np.clip(v, 0, 255).astype(np.uint8)
                    recon_frame = cv2.merge((y, u, v))
                    prev_reconstructed = (y, u, v)
                except ReadError as e:
                    logging.error(f"I-frame decode error at frame {frame_idx}: {e}, bits remaining: {bitstream.len - bitstream.pos}")
                    break
                except Exception as e:
                    logging.error(f"Unexpected I-frame decode error at frame {frame_idx}: {e}")
                    break
            else:
                if prev_reconstructed is None:
                    logging.error(f"No previous frame for P-frame at frame {frame_idx}")
                    break
                try:
                    mv_y, y = self.decode_pframe(bitstream, prev_reconstructed[0], (adjusted_height, adjusted_width))
                    mv_u, u = self.decode_pframe(bitstream, prev_reconstructed[1][::2, ::2], (adjusted_height//2, adjusted_width//2))
                    mv_v, v = self.decode_pframe(bitstream, prev_reconstructed[2][::2, ::2], (adjusted_height//2, adjusted_width//2))
                    u = cv2.resize(u, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
                    v = cv2.resize(v, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
                    y = np.clip(y, 0, 255).astype(np.uint8)
                    u = np.clip(u, 0, 255).astype(np.uint8)
                    v = np.clip(v, 0, 255).astype(np.uint8)
                    recon_frame = cv2.merge((y, u, v))
                    prev_reconstructed = (y, u, v)
                except ReadError as e:
                    logging.error(f"P-frame decode error at frame {frame_idx}: {e}, bits remaining: {bitstream.len - bitstream.pos}")
                    break
                except Exception as e:
                    logging.error(f"Unexpected P-frame decode error at frame {frame_idx}: {e}")
                    break

            recon_frame = cv2.resize(recon_frame, (width, height), interpolation=cv2.INTER_AREA)
            recon_frame = cv2.cvtColor(recon_frame, cv2.COLOR_YUV2BGR)
            logging.debug(f"Reconstructed frame {frame_idx} shape: {recon_frame.shape}, dtype: {recon_frame.dtype}")
            out.write(recon_frame)
            frames_written += 1
            logging.info(f"After decoding frame {frame_idx}, bits read: {bitstream.pos}")

        out.release()
        logging.info(f"Total frames written: {frames_written}")
        if bitstream.pos < bitstream.len:
            logging.warning(f"Remaining bits after decoding: {bitstream.len - bitstream.pos}")

    def decode_iframe(self, bitstream, shape):
        try:
            if bitstream.len - bitstream.pos < 16:
                raise ReadError(f"Not enough bits to read huff_size, bits remaining: {bitstream.len - bitstream.pos}")
            huff_size = bitstream.read('uint:16')
            huff_tree = {}
            for _ in range(huff_size):
                if bitstream.len - bitstream.pos < 24:  # 16 bits for sym + 8 bits for code_len
                    raise ReadError(f"Not enough bits to read Huffman entry, bits remaining: {bitstream.len - bitstream.pos}")
                sym = bitstream.read('uint:16')
                code_len = bitstream.read('uint:8')
                if bitstream.len - bitstream.pos < code_len:
                    raise ReadError(f"Not enough bits to read Huffman code, bits remaining: {bitstream.len - bitstream.pos}")
                code = bitstream.read(f'bin:{code_len}')
                huff_tree[code] = sym
            logging.debug(f"Decoded Huffman tree with {huff_size} entries")
        except ReadError as e:
            logging.error(f"Failed to decode Huffman tree: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error decoding Huffman tree: {e}")
            raise

        blocks = []
        h, w = shape
        num_blocks = (h // 8) * (w // 8)
        temp_bits = BitArray()

        for block_idx in range(num_blocks):
            coeffs = []
            while True:
                try:
                    if bitstream.len - bitstream.pos < 1:
                        raise ReadError(f"Not enough bits to read next bit in block {block_idx}, bits remaining: {bitstream.len - bitstream.pos}")
                    temp_bits.append(bitstream.read('bits:1'))
                    code = temp_bits.bin
                    if code in huff_tree:
                        symbol = huff_tree[code]
                        if symbol == 0:  # EOB
                            break
                        elif symbol == 1:  # Coefficient 0
                            val = 0
                        else:  # Non-zero coefficient
                            magnitude = symbol - 1
                            if bitstream.len - bitstream.pos < 1:
                                raise ReadError(f"Not enough bits to read sign bit in block {block_idx}, bits remaining: {bitstream.len - bitstream.pos}")
                            sign = bitstream.read('bool') if symbol > 1 else False
                            val = -magnitude if sign else magnitude
                        coeffs.append(val)
                        temp_bits = BitArray()
                except ReadError as e:
                    logging.error(f"Bitstream read error in block {block_idx}: {e}")
                    raise
                except Exception as e:
                    logging.error(f"Unexpected error in block {block_idx}: {e}")
                    raise
            coeffs.extend([0] * (64 - len(coeffs)))
            dequantized = self.inverse_zigzag(np.array(coeffs), 8, 8) * 10
            block = idctn(dequantized, norm='ortho')
            blocks.append(block)

        return self.from_blocks(blocks, shape, 8), huff_tree

    def decode_pframe(self, bitstream, prev_channel, shape):
        macroblock_size = 16
        num_blocks = (shape[0] // macroblock_size) * (shape[1] // macroblock_size)
        motion_vectors = []
        for _ in range(num_blocks):
            if bitstream.len - bitstream.pos < 10:  # 5 bits for mv_x + 5 bits for mv_y
                raise ReadError(f"Not enough bits to read motion vector, bits remaining: {bitstream.len - bitstream.pos}")
            mv_x = bitstream.read('uint:5') - 16
            mv_y = bitstream.read('uint:5') - 16
            motion_vectors.append((mv_x, mv_y))

        try:
            if bitstream.len - bitstream.pos < 16:
                raise ReadError(f"Not enough bits to read huff_size, bits remaining: {bitstream.len - bitstream.pos}")
            huff_size = bitstream.read('uint:16')
            huff_tree = {}
            for _ in range(huff_size):
                if bitstream.len - bitstream.pos < 24:
                    raise ReadError(f"Not enough bits to read Huffman entry, bits remaining: {bitstream.len - bitstream.pos}")
                sym = bitstream.read('uint:16')
                code_len = bitstream.read('uint:8')
                if bitstream.len - bitstream.pos < code_len:
                    raise ReadError(f"Not enough bits to read Huffman code, bits remaining: {bitstream.len - bitstream.pos}")
                code = bitstream.read(f'bin:{code_len}')
                huff_tree[code] = sym
            logging.debug(f"Decoded Huffman tree with {huff_size} entries")
        except ReadError as e:
            logging.error(f"Failed to decode Huffman tree: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error decoding Huffman tree: {e}")
            raise

        blocks = []
        temp_bits = BitArray()
        prev_blocks = self.to_blocks(prev_channel, macroblock_size)

        for i in range(num_blocks):
            coeffs = []
            while True:
                try:
                    if bitstream.len - bitstream.pos < 1:
                        raise ReadError(f"Not enough bits to read next bit in block {i}, bits remaining: {bitstream.len - bitstream.pos}")
                    temp_bits.append(bitstream.read('bits:1'))
                    code = temp_bits.bin
                    if code in huff_tree:
                        symbol = huff_tree[code]
                        if symbol == 0:  # EOB
                            break
                        elif symbol == 1:  # Coefficient 0
                            val = 0
                        else:  # Non-zero coefficient
                            magnitude = symbol - 1
                            if bitstream.len - bitstream.pos < 1:
                                raise ReadError(f"Not enough bits to read sign bit in block {i}, bits remaining: {bitstream.len - bitstream.pos}")
                            sign = bitstream.read('bool') if symbol > 1 else False
                            val = -magnitude if sign else magnitude
                        coeffs.append(val)
                        temp_bits = BitArray()
                except ReadError as e:
                    logging.error(f"Bitstream read error in block {i}: {e}")
                    raise
                except Exception as e:
                    logging.error(f"Unexpected error in block {i}: {e}")
                    raise
            coeffs.extend([0] * (macroblock_size * macroblock_size - len(coeffs)))
            dequantized = self.inverse_zigzag(np.array(coeffs), macroblock_size, macroblock_size) * 10
            residual = idctn(dequantized, norm='ortho')
            pred_block = self.apply_motion(prev_blocks[i], motion_vectors[i], macroblock_size)
            blocks.append(pred_block + residual)

        return BitArray(), self.from_blocks(blocks, shape, macroblock_size)

    def to_blocks(self, frame, block_size):
        h, w = frame.shape
        blocks = []
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = frame[i:i+block_size, j:j+block_size]
                if block.shape != (block_size, block_size):
                    padded_block = np.zeros((block_size, block_size), dtype=frame.dtype)
                    bh, bw = block.shape
                    padded_block[:bh, :bw] = block
                    block = padded_block
                blocks.append(block)
        return np.array(blocks)

    def from_blocks(self, blocks, shape, block_size):
        h, w = shape
        frame = np.zeros((h, w), dtype=np.float32)
        idx = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if idx < len(blocks):
                    block_h = min(block_size, h - i)
                    block_w = min(block_size, w - j)
                    frame[i:i+block_h, j:j+block_w] = blocks[idx][:block_h, :block_w]
                    idx += 1
        return frame

    def zigzag(self, block):
        zigzag_order = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
        ]
        return block.flatten()[zigzag_order]

    def inverse_zigzag(self, coeffs, rows, cols):
        zigzag_order = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
        ]
        block = np.zeros(rows * cols)
        for i, idx in enumerate(zigzag_order):
            if i < len(coeffs):
                block[idx] = coeffs[i]
        return block.reshape((rows, cols))

    def block_matching(self, block, ref_block, block_size):
        search_range = 4
        best_mv = (0, 0)
        min_sad = float('inf')
        h, w = block.shape
        ref_h, ref_w = ref_block.shape

        for dy in range(-search_range, search_range + 1, 2):
            for dx in range(-search_range, search_range + 1, 2):
                y, x = dy, dx
                if (0 <= y < ref_h - block_size) and (0 <= x < ref_w - block_size):
                    ref_patch = ref_block[y:y+block_size, x:x+block_size]
                    sad = np.sum(np.abs(block - ref_patch))
                    if sad < min_sad:
                        min_sad = sad
                        best_mv = (dx, dy)

        pred_block = self.apply_motion(ref_block, best_mv, block_size)
        return best_mv, pred_block

    def apply_motion(self, ref_block, mv, block_size):
        dx, dy = mv
        h, w = ref_block.shape
        y, x = dy, dx
        if (0 <= y < h - block_size) and (0 <= x < w - block_size):
            return ref_block[y:y+block_size, x:x+block_size]
        return ref_block[:block_size, :block_size]

    def calculate_psnr(self, original_path, recon_path):
        cap_orig = cv2.VideoCapture(original_path)
        cap_recon = cv2.VideoCapture(recon_path)
        mse_sum = 0
        frame_count = 0

        while cap_orig.isOpened() and cap_recon.isOpened():
            ret_orig, frame_orig = cap_orig.read()
            ret_recon, frame_recon = cap_recon.read()
            if not (ret_orig and ret_recon):
                break
            mse = np.mean((frame_orig - frame_recon) ** 2)
            mse_sum += mse
            frame_count += 1

        cap_orig.release()
        cap_recon.release()
        if frame_count == 0:
            return 0
        mse_avg = mse_sum / frame_count
        if mse_avg == 0:
            return float('inf')
        return 20 * np.log10(255 / np.sqrt(mse_avg))

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCompressorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_playback(), root.destroy()))
    root.mainloop()
