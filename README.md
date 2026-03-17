# Watermark Removal from Videos

This repository implements a pipeline for removing watermarks from videos. The pipeline covers dataset preparation (frame extraction), watermark detection/segmentation (YOLOv8 segmentation), and content reconstruction using a LaMa-style inpainting wrapper (packaged under `simple-lama-inpainting`). The code is intended as a proof-of-concept and a starting point for production work.

Key features
- Extract sample frames from videos for dataset creation (frame sampling).
- Segment/detect watermark regions using a YOLO segmentation model (Ultralytics YOLOv8).
- Fallback detection for simple/high-contrast watermarks when no segmentation model is available.
- Expand/dilate masks to fully cover watermark artifacts before inpainting.
- Inpaint masked regions with the included LaMa-based inpainting wrapper.
- Reassemble processed frames back into a video and restore audio using ffmpeg.

## Table of contents
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Usage details](#usage-details)
- [How it works (pipeline)](#how-it-works-pipeline)
- [Configuration and tuning](#configuration-and-tuning)
- [Training a segmentation model (recommended)](#training-a-segmentation-model-recommended)
- [Troubleshooting & notes](#troubleshooting--notes)
- [Assumptions and next steps](#assumptions-and-next-steps)
- [License](#license)

## Repository layout

Top-level folders you will interact with often:

- `scripts/` - main helper scripts and prototype pipeline (frame extraction, processing).
	- `extract_samples.py` - extract frames from `input_videos` to `samples/`.
	- `process_video_prototype.py` - main prototype pipeline: extract frames, detect, inpaint, reassemble.
	- `gen_image.py` - small image/mask generator used for experimentation.
	- `test.py` - misc testing/visualization code.
- `simple-lama-inpainting/` - local wrapper for the LaMa-style inpainting model used by the pipeline.
- `input_videos/` - place source videos here.
- `output_videos/` - processed videos are saved here.
- `frames/` - intermediate frame, mask and result images are saved here during processing.
- `samples/` - extracted frames (samples) used for dataset preparation / quick inspection.
- `dataset/` - expected dataset layout for training (contains `data.yaml`, `train/`, `valid/`, `test/`).
- `runs/segment/` - training outputs for segmentation (Ultralytics default project layout). The pipeline references a trained model at `runs/segment/train9/weights/best.pt` by default.

## Requirements

The Python dependencies are listed in `requirements.txt`.

System requirements
- Python 3.8+ (virtualenv/venv recommended)
- A system `ffmpeg` binary available on PATH (used to reassemble videos and merge audio)
- A CUDA-capable GPU is recommended for reasonable performance when running YOLO and the inpainting model; CPU will work but be slow.

Install (example)

```bash
# create and activate venv (macOS / zsh)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ensure ffmpeg is installed (macOS, Homebrew)
brew install ffmpeg
```

Note: `requirements.txt` currently lists the main Python packages used: `ultralytics`, `opencv-python`, `numpy`, `ffmpeg-python`, and `torch`.

## Quick start

1. Put the source video(s) into `input_videos/`.
2. Adjust `MODEL_PATH` in `scripts/process_video_prototype.py` if you have a trained YOLO segmentation model. If not, the script will use a simple fallback detector.
3. Run the prototype pipeline:

```bash
python scripts/process_video_prototype.py
```

The script will:
- extract frames into `work_frames/`,
- detect watermark masks using YOLO (or fallback),
- dilate the mask and inpaint each frame using the local LaMa wrapper,
- reassemble frames into an output video in `output_videos/` and merge audio.

Small utilities
- To extract sample frames only (useful for dataset creation):

```bash
python scripts/extract_samples.py
```

This saves frames into `samples/` at 1 FPS by default (see `FPS_SAMPLE` in `scripts/extract_samples.py`).

## Usage details

Important configuration variables are near the top of `scripts/process_video_prototype.py`:
- `MODEL_PATH` - path to a trained YOLO segmentation model (Ultralytics). Default used in the repo: `runs/segment/train9/weights/best.pt`.
- `INPUT_FOLDER`, `OUTPUT_FOLDER`, `FRAME_FOLDER` - folders for I/O and temporary frames.

The main processing flow in `process_video_prototype.py`:
1. `extract_frames()` - write frames to `work_frames/frame_######.png`.
2. For each frame, run either `detect_mask_yolo()` (if YOLO model loaded) or `fallback_detect()`.
	 - YOLO-based detection collects `r.masks` from predictions and composes a binary mask.
	 - The mask is dilated (morphological dilation) to ensure the watermark area is fully covered. The dilation kernel size is configurable.
3. `inpaint_frame()` converts the frame and mask to PIL, calls the `SimpleLama` wrapper, saves intermediate images to `frames/`, then returns the inpainted frame.
4. `reassemble_and_attach_audio()` uses `ffmpeg` to create a video from processed frames and merges the original audio stream.

## How it works (pipeline)

- Prepare dataset: extract frames (samples) from videos and label watermarks if you plan to train a segmentation model. Dataset layout compatible with Ultralytics/YOLO is already scaffolded under `dataset/`.
- Train segmentation model: use Ultralytics YOLOv8 segmentation (task=segment) to detect watermark shapes. Save best weights under a `runs/segment/...` path.
- Detection: for each frame, run the segmentation model to produce a binary mask describing watermark pixels. If the model is not present, a simple heuristic fallback (`fallback_detect`) attempts to threshold and find high-contrast regions.
- Mask processing: dilate the mask so the inpainting algorithm receives a margin around the watermark.
- Inpainting: call `simple-lama-inpainting` wrapper (LaMa-style inpainting) with a PIL image + mask, producing a reconstructed image.
- Reassemble: write processed frames to a temporary video and merge audio from the original file.

## Configuration and tuning

- Mask dilation kernel size: in `detect_mask_yolo()` the kernel size (`kernel_size = 45`) is an important hyperparameter; smaller watermarks may require smaller kernels (10–30), large logos may require larger.
- YOLO detection confidence: when loading YOLO, the detection uses `conf=0.25` in predict; adjust for your dataset.
- Input image size for YOLO: currently `imgsz=640`. Increase for small watermarks at the cost of speed.
- Batch processing and speed: the prototype is single-threaded and handles frames one-by-one. For production, consider batching masks/inpainting and using GPU for the inpainting model.

## Training a segmentation model (recommended)

If you want better masks, label watermark regions and train a YOLOv8 segmentation model. Example ultralytics training command:

```bash
# example (adjust to your environment and model choice)
yolo task=segment mode=train model=yolov8s-seg.pt data=dataset/data.yaml epochs=100 imgsz=640 project=runs/segment name=train9
```

After training, point `MODEL_PATH` in `scripts/process_video_prototype.py` to the model weights (e.g. `runs/segment/train9/weights/best.pt`).

## Troubleshooting & notes

- If `process_video_prototype.py` fails to load a model, it falls back to a simple threshold-based detector. This fallback is intentionally basic and only suitable for very high-contrast watermarks.
- Ensure `ffmpeg` is installed and on PATH; if `reassemble_and_attach_audio` silently fails, run the ffmpeg commands printed by the script manually to inspect errors.
- GPU: running YOLO and the inpainting model on CPU will be slow. If you have CUDA, ensure the `torch` version installed supports your CUDA toolkit.
- If mask alignment issues occur, check that masks and frames share the same resolution; the code attempts to resize masks to image size before inpainting.

## Assumptions and next steps

Assumptions made while preparing this README:

- The included `simple-lama-inpainting` wrapper provides a callable `SimpleLama` object that accepts a PIL image and a binary PIL mask, returning a PIL inpainted image. (The repository contains a small wrapper directory for this purpose.)
- A trained YOLO segmentation model may not be present by default. If not present, the script uses the `fallback_detect()` heuristic.

Suggested next improvements (low-risk):

- Add a small training script or instructions that convert `samples/` into a labelled segmentation dataset automatically.
- Add a config file or CLI arguments to `process_video_prototype.py` so parameters (model path, kernel size, folders) are easier to change.
- Add multiprocessing or GPU-batched inpainting to speed up large videos.

## Contact & License

This repository includes a `LICENSE` file—refer to it for license details. For questions or improvements, open issues or contact the project owner directly.

---

If you'd like, I can:
- add CLI flags to `process_video_prototype.py` for configuration,
- create a minimal training script that uses `samples/` and `dataset/` to produce a YOLO dataset, or
- implement batching/multiprocessing for faster processing.

Tell me which follow-up you'd prefer and I'll implement it next.

