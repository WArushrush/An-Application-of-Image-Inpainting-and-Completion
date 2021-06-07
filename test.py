import cv2
import numpy as np
from PIL import Image
# A = np.array(Image.open("IMAGE/ref.png"))
A = cv2.imread("cup_a.jpg", 1)
A = cv2.rectangle(A, (140, 120), (150, 130), (0, 0, 0), 100)
cv2.imwrite("test_b.jpg", A)
cv2.imshow("image", A)
cv2.waitKey(0)
