import cv2
import os
from scipy import ndimage
import numpy as np

# needs implementing more fully
def build_gaussian_pyramid(image, num_scales):
        pyramid = [image]
        for i in range(num_scales - 1):
            image = cv2.pyrDown(image)
            pyramid.insert(0, image)
        return pyramid

# loads image given relative dir filepath
def load_images(dir):
    files = [dir + file for file in os.listdir(dir) if file.endswith('.png')]
    images = []
    for file in files:
        image = cv2.imread(file)
        images.append((file.split('/')[-1], image))
    return images

def main():

    print("Loading icons...")
    iconDir = "IconDataset/png/"
    icons = load_images(iconDir)
    iconNames = [icon[0] for icon in icons]
    icons = [icon[1] for icon in icons]
    # Convert all icons to grayscale
    icons = [cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY) for icon in icons]
    # Convert background to 0
    icons = [cv2.threshold(icon, 255, 0, cv2.THRESH_TOZERO_INV)[1] for icon in icons]
    icons = [cv2.pyrDown(ic) for ic in icons]

    # icons = [[cv2.pyrDown(cv2.pyrDown(ic)), cv2.pyrDown(ic), ic] for ic in icons]
    # iconNames = [[ic, ic, ic] for ic in iconNames]
    
    # i = []
    # for icon in icons:
    #     i += icon
    # icons = i

    # i = []
    # for icon in iconNames:
    #     i += icon
    # iconNames = i


    print("Icons loaded.")


    print("Loading test images...")
    testDir = "Task2Dataset/images/"
    testImages = load_images(testDir)
    testImages = [image[1] for image in testImages]

    # Convert all test images to grayscale
    testImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in testImages]
    # Convert background to 0
    testImages = [cv2.threshold(image, 255, 0, cv2.THRESH_TOZERO_INV)[1] for image in testImages]
    print("Test images loaded.")


    # Build Gaussian pyramid for all icons
    # Determine the appropriate number of scales and rotations per class, considering factors like Gaussian kernel specs, initial/final scales, number of scale levels (octaves), and rotations.
    # Justify parameter choices based on runtime vs. accuracy trade-off.
    
    print("Building icon pyramids...")
    
    NUM_SCALES = 4
    icon_pyramids = []
    for icon in icons:
        pyramid = build_gaussian_pyramid(icon, NUM_SCALES)
        icon_pyramids.append(pyramid)

    print("Pyramids built.")

    # # rotate each level of pyramid, and add to pyramid
    # num_rotations = 1
    # for i, pyramid in enumerate(icon_pyramids):
    #     rotated_pyramid = []
    #     for level in pyramid:
    #         rotations = [ndimage.rotate(level, angle, reshape=False) for angle in range(0, 360, 360//num_rotations)]
    #         rotated_pyramid += rotations
    #     icon_pyramids[i] = rotated_pyramid



    matches = []
    SIM_LIMIT = 50

    # Slide each template over the test image and measure similarities
    for testImage in [testImages[6]]:
        image_scales = build_gaussian_pyramid(testImage, NUM_SCALES)

        #search for each icon
        for i, pyramid in enumerate(icon_pyramids):
            name = iconNames[i]

            min_icon_match = np.inf
            min_icon_match_location = None

            #search up gasussian pyramid
            for (template, test) in zip(pyramid, image_scales):
            
                top_left, score = match_template_rss(test, template)
                if top_left is None:
                    break
                if score > SIM_LIMIT:
                    break
                print(score)

                min_icon_match = score
                min_icon_match_location = top_left

            if (min_icon_match_location != None):
                matches.append((name, min_icon_match_location, template, testImage))
            print("processed ", i)
    
    sorted_matches = sorted(matches, key=lambda tup: tup[1])

    for match in sorted_matches[:5]:

        top_left = match[1]
        template = match[2]
        testImage = match[3]
        name = match[0]

        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        cv2.rectangle(testImage, top_left, bottom_right, (0, 0, 0), 2)
        cv2.putText(testImage, name, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('Matched Image', testImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def match_template_rss(image, template):

    match = None
    min_rss = np.inf

    for y in range(image.shape[0] - template.shape[0]):
        for x in range(image.shape[1] - template.shape[1]):
            section = image[y:y+template.shape[0], x:x+template.shape[1]]
            rss = np.sum((section - template) ** 2) / (template.shape[0] * template.shape[1])
            if rss < min_rss:
                min_rss = rss
                match = (x, y)

    return (match, min_rss)


    



if __name__ == '__main__':
    main()