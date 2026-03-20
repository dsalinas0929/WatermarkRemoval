# scripts/process_video_prototype.py
import os, cv2, numpy as np, subprocess
from ultralytics import YOLO
import sys
from pathlib import Path

# Add the parent folder of simple-lama-inpainting to Python path BEFORE importing it
sys.path.append(str(Path(__file__).resolve().parent.parent / "simple-lama-inpainting"))

from simple_lama_inpainting import SimpleLama
from PIL import Image

simple_lama = SimpleLama()

# Initialize the inpainting model
# 1. Choose model (LaMa for general-purpose inpainting)
model_name = "lama"


# Initialize the inpainting model
MODEL_PATH = 'runs/segment/train9/weights/best.pt'  # replace with your trained model when ready
INPUT_FOLDER = 'input_videos'
OUTPUT_FOLDER = 'output_videos'
FRAME_FOLDER = 'work_frames'
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Try load model (if missing, set model=None and we can fallback)
try:
    model = YOLO(MODEL_PATH)
    print("Loaded segmentation model:", MODEL_PATH)
except Exception as e:
    print("No model loaded, will use fallback:", e)
    model = None

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        p = os.path.join(FRAME_FOLDER, f'frame_{i:06d}.png')
        cv2.imwrite(p, frame)
        i += 1
    cap.release()
    return i

# Detect watermark mask using YOLO segmentation model
def detect_mask_yolo(frame):
    # returns binary mask same size as frame
    results = model.predict(frame, imgsz=640, device='cuda', conf=0.25)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for r in results:
        if r.masks is not None:
            m = r.masks.data.cpu().numpy().sum(axis=0)
            m = (m > 0.5).astype('uint8') * 255

            # Resize mask if needed
            if m.shape != mask.shape:
                m = cv2.resize(m, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- Expand mask area (key part) ---
            kernel_size = 45  # try 10–30 depending on your watermark size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            m = cv2.dilate(m, kernel, iterations=1)

            mask = cv2.bitwise_or(mask, m)
    return mask

# Fallback detection if no model: simple thresholding to find "Sora" text (adjust as needed)
def fallback_detect(frame):
    # VERY simple fallback: detect "Sora" white text via threshold near top areas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # focus on top half to reduce noise:
    h,w = th.shape
    roi = th[:h//2, w//3:w]  # right-top area heuristic
    cnts,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        if ww*hh > 100: # filter tiny noise
            cv2.rectangle(mask, (w//3 + x, y), (w//3 + x + ww, y + hh), 255, -1)
    mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=2)
    return mask

# Inpaint the frame using SimpleLama with the detected mask
def inpaint_frame(frame, mask, idx):
    # Ensure mask is binary uint8 (0/255)
    import numpy as np
    # handle empty or missing mask
    if mask is None or mask.sum() == 0:
        return frame

    # normalize mask to uint8 0/255
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255

    # Convert OpenCV BGR numpy image to PIL RGB
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        # if conversion fails, assume image already RGB-like
        img_rgb = frame
    pil_img = Image.fromarray(img_rgb)

    # Ensure mask size matches image and convert to PIL 'L'
    h, w = pil_img.size[1], pil_img.size[0]
    if mask.shape[:2] != (h, w):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask
    pil_mask = Image.fromarray(mask_resized).convert('L')

    # Call SimpleLama (expects PIL image + mask)
    try:
        result = simple_lama(pil_img, pil_mask)
    except Exception as e:
        print("SimpleLama inpainting failed:", e)
        return frame
    
    img_path = os.path.join("frames", f"{idx}.png")
    mask_path = os.path.join("frames", f"{idx}_mask.png")
    res_path = os.path.join("frames", f"{idx}_res.png")

    # ✅ Save both images
    pil_img.save(img_path)
    pil_mask.save(mask_path)
    result.save(res_path)

    # Convert result (PIL) back to OpenCV BGR numpy image
    if isinstance(result, np.ndarray):
        out = result
    else:
        out_rgb = np.array(result)
        try:
            out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            out = out_rgb

    return out

# Reassemble video from frames and attach original audio using ffmpeg
def reassemble_and_attach_audio(frame_folder, original_video, output_video, fps):
    temp_video = 'temp_out.mp4'
    # write video from frames
    cmd = [
        'ffmpeg', '-y', '-r', str(fps), '-i',
        os.path.join(frame_folder, 'frame_%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', temp_video
    ]
    subprocess.call(cmd)
    # merge audio
    cmd2 = ['ffmpeg','-y','-i', temp_video, '-i', original_video, '-c:v','copy','-c:a','aac','-map','0:v:0','-map','1:a:0', output_video]
    subprocess.call(cmd2)
    os.remove(temp_video)

# Main processing function for each video file
def process_file(video_file):
    print("Processing:", video_file)
    # clear frame folder
    for f in os.listdir(FRAME_FOLDER):
        os.remove(os.path.join(FRAME_FOLDER, f))
    input_path = os.path.join(INPUT_FOLDER, video_file)
    cnt = extract_frames(input_path)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    for i in range(cnt):
        p = os.path.join(FRAME_FOLDER, f'frame_{i:06d}.png')
        frame = cv2.imread(p)
        if frame is None: continue
        if model is not None:
            mask = detect_mask_yolo(frame)
        else:
            mask = fallback_detect(frame)
        out = inpaint_frame(frame, mask, i)
        cv2.imwrite(p, out)
    outp = os.path.join(OUTPUT_FOLDER, video_file)
    reassemble_and_attach_audio(FRAME_FOLDER, input_path, outp, fps)
    print("Saved:", outp)

if __name__ == "__main__":
    for v in os.listdir(INPUT_FOLDER):
        if v.lower().endswith(('.mp4','.mov','.mkv','.avi')):
            process_file(v)
