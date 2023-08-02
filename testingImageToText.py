import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('image.jpg')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 



########## Transforming image in different ways
# image = cv2.imread('./IMG_6974.jpg')

# cv2.imshow('img', gray)
# cv2.waitKey(0)

# cv2.imshow('img', thresh)
# cv2.waitKey(0)

# cv2.imshow('img', opening)
# cv2.waitKey(0)

# cv2.imshow('img', canny)
# cv2.waitKey(0)


############### Putting boxes around characters
# import cv2
# import pytesseract

# img = cv2.imread('IMG_6974.jpg')

# h, w, c = img.shape
# boxes = pytesseract.image_to_boxes(img) 
# for b in boxes.splitlines():
#     b = b.split(' ')
#     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)




############### Putting boxes around words
import cv2
import pytesseract
from pytesseract import Output

from scipy.spatial.distance import hamming


with open('ground_truth.txt') as f:
    ground_truth = f.readlines()

ground_truth = " ".join(ground_truth).replace('\n', '')
print(ground_truth)

img = cv2.imread('IMG_6974.jpg')

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#cv2.imshow('img', img)
#cv2.waitKey(0)

# NOTE: SHOULD REPLACE THE SUBSTRING "THE WINDOW ON THE WEST 879" at every instance

original_image_string = pytesseract.image_to_string(img).replace('\n', ' ')

print(original_image_string)

# still need to research if this is the best distance metric to use
from Levenshtein import distance as levenshtein_distance


# similarity = calculate_levenshtein_distance(list(ground_truth), list("hello"))  # 2
#hamming_distance('book', 'tooth')      # 3

#hamming_distance = hamming(list(ground_truth), list(string))
#print(hamming_distance)

# print(similarity)
# print(baseline_similarity)

gray = get_grayscale(img)
thresh = thresholding(gray)

threshold_image_string = pytesseract.image_to_string(thresh).replace('\n', ' ')

opening = opening(gray)

opening_image_string = pytesseract.image_to_string(opening).replace('\n', ' ')

canny = canny(gray)

canny_image_string = pytesseract.image_to_string(canny).replace('\n', ' ')

print(threshold_image_string)
print(opening_image_string)
print(canny_image_string)



baseline_similarity = levenshtein_distance(list(ground_truth), list("")) # just something to compare against
similarity = levenshtein_distance(list(ground_truth), list(original_image_string)) 
threshold_similarity = levenshtein_distance(list(ground_truth), list(threshold_image_string))
opening_similarity = levenshtein_distance(list(ground_truth), list(opening_image_string))
canny_similarity = levenshtein_distance(list(ground_truth), list(canny_image_string))

print("Original image similarity: ", similarity)
print("Threshold image similarity: ", threshold_similarity)
print("Opening image similarity: ", opening_similarity)
print("Canny image similarity: ", canny_similarity)
print("Baseline similarity: ", baseline_similarity)
