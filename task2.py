import cv2
import os
from scipy import ndimage

# needs implementing more fully
def build_gaussian_pyramid(image, num_scales):
        pyramid = [image]
        for i in range(num_scales - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

# loads image given relative dir filepath
def load_images(dir):
    files = [dir + file for file in os.listdir(dir) if file.endswith('.png')]
    images = []
    for file in files:
        image = cv2.imread(file)
        images.append(image)
    return images

def main():
    # Load all icons
    iconDir = "IconDataset/png/"
    icons = load_images(iconDir)
    # Convert all icons to grayscale
    icons = [cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY) for icon in icons]
    # Convert background to 0
    icons = [cv2.threshold(icon, 255, 0, cv2.THRESH_TOZERO_INV)[1] for icon in icons]

    # Load all test images
    testDir = "Task2Dataset/images/"
    testImages = load_images(testDir)
    # Convert all test images to grayscale
    testImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in testImages]
    # Convert background to 0
    testImages = [cv2.threshold(image, 255, 0, cv2.THRESH_TOZERO_INV)[1] for image in testImages]



    # Build Gaussian pyramid for all icons
    # Determine the appropriate number of scales and rotations per class, considering factors like Gaussian kernel specs, initial/final scales, number of scale levels (octaves), and rotations.
    # Justify parameter choices based on runtime vs. accuracy trade-off.
    num_scales = 4
    icon_pyramids = []
    for icon in icons:
        pyramid = build_gaussian_pyramid(icon, num_scales)
        icon_pyramids.append(pyramid)

    # rotate each level of pyramid, and add to pyramid
    num_rotations = 4
    for i, pyramid in enumerate(icon_pyramids):
        rotated_pyramid = []
        for level in pyramid:
            rotations = [ndimage.rotate(level, angle, reshape=False) for angle in range(0, 360, 360//num_rotations)]
            rotated_pyramid += rotations
        icon_pyramids[i] = rotated_pyramid

    
    # show a pyramid
    for level in icon_pyramids[0]:
        cv2.imshow('level', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    # Slide each template over the test image and measure similarities
    for testImage in [testImages[-1]]:
        for pyramid in icon_pyramids:
            for template in pyramid:
                result = cv2.matchTemplate(testImage, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > 0.9:
                    top_left = max_loc
                    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
                    cv2.rectangle(testImage, top_left, bottom_right, (0, 0, 0), 2)
                    similarity_score = max_val
                    cv2.imshow('Matched Image', testImage)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    

    
    



if __name__ == '__main__':
    main()