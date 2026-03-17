from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('runs/segment/train9/weights/best.pt')
# frame = cv2.imread('samples/temp_video_f000000.png')
frame = cv2.imread('frames/0.png')

results = model.predict(frame, imgsz=640, device='cuda', conf=0.25)
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
# Visualize
result_frame = results[0].plot()
mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
result_frame = cv2.hconcat([frame, mask_3ch, result_frame])
cv2.imshow('result', result_frame)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.imwrite('frame_result.png', result_frame)
