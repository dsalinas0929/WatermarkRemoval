# scripts/extract_samples.py
import os, cv2
from pathlib import Path

INPUT_FOLDER = 'input_videos'
SAMPLES_FOLDER = 'samples'   # saved frames here
FPS_SAMPLE = 1               # frames per second to extract

os.makedirs(SAMPLES_FOLDER, exist_ok=True)

# Extract frames at a specified rate (per_second) and save as PNG
def sample_frames(video_path, out_dir, per_second=1):
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    # Calculate interval in frames to achieve the desired sampling rate
    interval = max(1, int(round(fps / per_second)))
    idx = 0
    saved = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if idx % interval == 0:
            fname = f"{Path(video_path).stem}_f{idx:06d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            saved += 1
            print(f"Saved frame {idx} as {fname}")
        idx += 1
    vid.release()
    return saved

if __name__ == "__main__":
    total = 0
    for f in os.listdir(INPUT_FOLDER):
        if not f.lower().endswith(('.mp4','.mov','.mkv','.avi')): continue
        p = os.path.join(INPUT_FOLDER, f)
        print("Extracting from", f)
        total += sample_frames(p, SAMPLES_FOLDER, per_second=FPS_SAMPLE)
    print("Saved sample frames:", total)
