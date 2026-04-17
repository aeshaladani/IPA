import cv2
import os

folder = "data/SrBenchmark/Set5/LR_bicubic/X4"

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    img = cv2.imread(path)

    #  FORCE VERY SMALL SIZE
    img = cv2.resize(img, (64, 64))

    cv2.imwrite(path, img)

print("All images resized to 64x64")