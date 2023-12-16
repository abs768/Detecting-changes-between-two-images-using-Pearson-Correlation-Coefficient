from skimage.metrics import structural_similarity
import cv2
import numpy as np

# Load images
before_image = cv2.imread('leftimage.jpg')
after_image = cv2.imread('rightimage.jpg')

# Convert images to grayscale
before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

# Compute Structural Similarity Index (SSIM) between the two images
(similarity_score, difference_image) = structural_similarity(before_gray, after_gray, full=True)
similarity_score_percent = similarity_score * 100
print("Image Similarity: {:.4f}%".format(similarity_score_percent))

# Convert the difference image to the 8-bit unsigned integer representation
difference_image_uint8 = (difference_image * 255).astype("uint8")
difference_image_box = cv2.merge([difference_image_uint8, difference_image_uint8, difference_image_uint8])

# Threshold the difference image to obtain the regions where the images differ
_, threshold_image = cv2.threshold(difference_image_uint8, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Find contours of the differing regions
contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to highlight the differing regions
mask = np.zeros(before_image.shape, dtype='uint8')

# Create a copy of the after image with the differing regions filled
filled_after_image = after_image.copy()

# Iterate over the contours and process the differing regions
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 40:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(before_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(after_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(difference_image_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
        cv2.drawContours(filled_after_image, [contour], 0, (0, 255, 0), -1)

# Display the images
cv2.imshow('Before Image', before_image)
cv2.imshow('After Image', after_image)
cv2.imshow('Difference Image', difference_image_uint8)
cv2.imshow('Difference Image with Box', difference_image_box)
cv2.imshow('Mask', mask)
cv2.imshow('Filled After Image', filled_after_image)
cv2.waitKey()