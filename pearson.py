import cv2
import numpy as np
from scipy.stats import pearsonr

# Load images

before = cv2.imread('1.jpg')
after = cv2.imread('2.jpg')
# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute Pearson correlation coefficient between the two images
correlation_coefficient, _ = pearsonr(before_gray.flatten(), after_gray.flatten())
print("Pearson Correlation Coefficient: {:.4f}".format(correlation_coefficient))

# Calculate absolute difference image
diff = cv2.absdiff(before_gray, after_gray)

# Threshold the difference image
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

# Draw rectangles and contours on images
for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
        cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('mask', mask)
cv2.imshow('filled after', filled_after)
cv2.waitKey()