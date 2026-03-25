import cv2

# read current HR (wrong one)
img = cv2.imread("data/SrBenchmark/Set5/HR/image1.png")

# resize to correct size (128x128)
img = cv2.resize(img, (128, 128))

cv2.imwrite("data/SrBenchmark/Set5/HR/image1.png", img)

print("HR fixed to 128x128 ✅")