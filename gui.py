import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import subprocess
import os
from pro import compress_and_save_video, load_video_frames  # تأكد من تعديل هذا حسب ملفك

class VideoCompressorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Simple Video Compressor")
        self.root.geometry("700x500")
        self.root.configure(bg="#f0f0f0")
        self.frames = []
        self.input_path = ""
        self.output_path = "compressed_video30fps.mp4"

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="Video Compressor Tool", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=10)

        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=5)

        self.load_btn = tk.Button(btn_frame, text="📁 Load Frames", command=self.load_frames, width=20, bg="#4CAF50", fg="white")
        self.load_btn.grid(row=0, column=0, padx=10)

        self.compress_btn = tk.Button(btn_frame, text="🗜️ Compress Video", command=self.compress_video, width=20, state=tk.DISABLED, bg="#2196F3", fg="white")
        self.compress_btn.grid(row=0, column=1, padx=10)

        self.open_btn = tk.Button(btn_frame, text="▶️ Open Compressed Video", command=self.open_video, width=25, state=tk.DISABLED, bg="#FF9800", fg="white")
        self.open_btn.grid(row=0, column=2, padx=10)

        self.progress = ttk.Progressbar(self.root, length=600, mode='indeterminate')
        self.progress.pack(pady=10)

        self.output_text = scrolledtext.ScrolledText(self.root, width=85, height=15, state='disabled', bg="#ffffff", fg="#000")
        self.output_text.pack(pady=10)

    def log(self, message):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')

    def load_frames(self):
        path = filedialog.askdirectory(title="Select Video Frames Directory")
        if not path:
            return
        self.input_path = path
        self.log(f"[UI] Loading frames from: {self.input_path}")
        self.progress.start()

        def task():
            try:
                self.frames = load_video_frames(self.input_path)
                self.log(f"[UI] Loaded {len(self.frames)} frames successfully.")
                self.compress_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f"[ERROR] {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self.progress.stop()

        threading.Thread(target=task).start()

    def compress_video(self):
        if not self.frames:
            self.log("[UI] No frames loaded. Please load frames first.")
            return
        self.log("[UI] Starting compression...")
        self.progress.start()

        def task():
            try:
                compress_and_save_video(self.input_path, self.output_path)
                self.log("[UI] Compression completed!")
                self.open_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f"[ERROR] {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self.progress.stop()

        threading.Thread(target=task).start()

    def open_video(self):
        if os.path.exists(self.output_path):
            self.log(f"[UI] Opening {self.output_path}...")
            subprocess.run(["start", self.output_path], shell=True)  # Windows
        else:
            self.log("[ERROR] Compressed video file not found!")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCompressorGUI(root)
    root.mainloop()
