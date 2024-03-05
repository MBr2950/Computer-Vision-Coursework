import numpy as np
import cv2

# loading and resizing the image 
lighthouse = cv2.imread('./IconDataset/png/001-lighthouse.png')
image = cv2.imread('./Task3Dataset/images/test_image_10.png')

lighthouse = cv2.cvtColor(lighthouse, cv2.COLOR_RGB2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

sift = cv2.SIFT_create()

# each descriptor is an array, kinda like a hash of the point 
keypoints1, descriptors1 = sift.detectAndCompute(lighthouse, None)
keypoints2, descriptors2 = sift.detectAndCompute(image, None)
# print(cv2.KeyPoint_convert(keypoints1[:1]))

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key= lambda x: x.distance)
print(matches[0].trainIdx)
print(matches[0].queryIdx)
print(matches[0].imgIdx)
print(matches[0].distance)

# Array matches is ordered by strength of match, i.e. first matches are strongest
matchesImage = cv2.drawMatches(lighthouse, keypoints1, image, keypoints2, matches[:10], image, flags=2)

# showing the output
cv2.imshow('image', cv2.resize(matchesImage, (800, 600)))

k = cv2.waitKey(0) & 0xff

if k == 27:
    cv2.destroyAllWindows()