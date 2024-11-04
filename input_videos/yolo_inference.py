#Kavin Ilanchezhian
#Testing YOLO
import os
from ultralytics import YOLO


model = YOLO('yolov8x')



image_path = '/Users/kilan43/Documents/VScode/Personal/Tennis-Analyzer/input_videos/image.png'
image_path2 = '/Users/kilan43/Documents/VScode/Personal/Tennis-Analyzer/input_videos/alcaraz_sinner.png'

video_path = '/Users/kilan43/Documents/VScode/Personal/Tennis-Analyzer/input_videos/input_video.mp4'
video_path2 = '/Users/kilan43/Documents/VScode/Personal/Tennis-Analyzer/input_videos/alcaraz2.mp4'
#print("File exists:", os.path.exists(image_path))


result = model.predict(video_path2, save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)

