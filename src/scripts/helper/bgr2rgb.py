import cv2
import glob
import os

root_dir = "."

for file in glob.glob(os.path.join(root_dir, "*")):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file, img)
