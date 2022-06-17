import cv2
import os
import numpy as np

folder = "../ringscan/diorfoldclose"
numImages = 24

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """ Helper function to resize an OpenCV image. """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127

def normalize(data, alpha, beta):
    return cv2.normalize(data, None, alpha, beta, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    

def preload(data):
    global start
    
    # Turn to grayscale
    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    # Normalization
    data = normalize(data,0,255)
    data = data.astype(np.uint8)
    data = np.array(data)

    return data


def getLights(images):
    lights = []
    blobimgs = []
    for image in images:
        image[image< np.percentile(image, 95)] = 0
        image[image!=0] = 255
        light = np.copy(image)
        light = highpass(light, 15)

        # erosion followed by dilation
        kernel = np.ones((15,15),np.uint8)
        light = cv2.morphologyEx(light, cv2.MORPH_OPEN, kernel)
        retval, threshold = cv2.threshold(light, np.percentile(light, 90), 255, cv2.THRESH_BINARY_INV)
        blobimgs.append(threshold)
        
    for image in blobimgs:
        tempImage = np.copy(image)
        for im1 in blobimgs:
            if not np.array_equal(im1, image):
                tempImage[tempImage == im1] = 255
                
        tempImage = cv2.morphologyEx(tempImage, cv2.MORPH_OPEN, kernel)
        # Blob detection (for detecting specular highlights from light)
        detector = cv2.SimpleBlobDetector()
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.01
        params.filterByColor = False
        params.filterByInertia = False
        params.filterByArea = False
        params.filterByConvexity = False
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        lights.append(detector.detect(tempImage))
    return lights

def loadImages(folder, number):
    images = []
    for i in range(number):
        img = cv2.imread(os.path.join(folder, "o" + str(i) + ".bmp"))
        #img = cv2.imread(os.path.join(folder, str(i+1) + ".bmp"))
        if img is not None:
            image = preload(img)
            images.append(np.array(image))
    print("read", str(len(images)), "images")
    return images


def removeBlobs(image, lightpoints):
    # Get center point
    cx = int(image.shape[0]/2)
    cy = int(image.shape[1]/2)
    
    # Add filled circles on specular highlights so that we can ignore those spots
    for lp in lightpoints:
        # Get coordinates of highlight center
        x,y = lp.pt
        x, y = int(x), int(y)
        
        # Lenght of line to be drawn across the screen, the length has to be long enough for the line to cut the image
        length = image.shape[0] + image.shape[1]
        
        
        # Calculate the coordinates of line going through the center point that is perpendicular to the line going from the center point to the highlight point
        lx = cx - x
        ly = cy - y
        
        mag = np.sqrt(lx*lx + ly*ly)
        lx = lx/mag
        ly = ly/mag
        temp = lx
        lx = -ly
        ly = temp
        
        r1x = int(cx + lx * length)
        r1y = int(cy + ly * length)
        r2x = int(cx + lx * -length)
        r2y = int(cy + ly * -length)
        
        # Create coordinates for a rectangle that will be drawn (To fill the area cut by the cut line)
        r1xo, r1yo, r2xo, r2yo = r1x, r1y, r2x, r2y
        if((r1y+r2y)/2 < y):
            r1y += 2*cy
            r2y += 2*cy
        else:
            r1y -= 2*cy
            r2y -= 2*cy
        if((r1x+r2x)/2< x):
            r1x += 2*cx
            r2x += 2*cx
        else:
            r1x -= 2*cx
            r2x -= 2*cx
        
        # Turn points into contours, draw the contours
        points = [[r1xo,r1yo],[r2xo,r2yo], [r2x,r2y], [r1x,r1y]]
        points = np.array(points).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(image,[points],0,0,-1)


images = loadImages(folder, numImages)
lightpoints = getLights(np.copy(images))



for image in images:
        image[image< np.percentile(images, 99)] = 0
        image[image!=0] = 255
        


finalImg = np.zeros(images[0].shape)
bg = np.zeros(images[0].shape)
bg[images[0] == images[1]] = 255


# Get difference between images
for image, lightpoint in zip(images, lightpoints):
    bg[image != bg] = 0
    removeBlobs(image, lightpoint)
    finalImg = finalImg + image
    

finalImg[finalImg == bg] = 0
    

cv2.imshow('Final image before filtering', resize(finalImg, 1000,1000))
cv2.waitKey(0)
cv2.imwrite(os.path.join(folder, "beforeFilter.bmp"), finalImg)

kernel = np.ones((5,5),np.uint8)
contourImg  = cv2.dilate(finalImg,kernel,iterations = 3)

cv2.imshow('Image after erosion followed by dilation', resize(contourImg, 1000,1000))
cv2.waitKey(0)
cv2.imwrite(os.path.join(folder, "erosionDilation.bmp"), contourImg)

# Draw contours
kernel = np.ones((5,5), np.uint8)
contourImg = cv2.GaussianBlur(np.uint8(contourImg), (13,13), 5)
dilatedIm = cv2.dilate(contourImg, kernel, iterations=2)
ret, thresh = cv2.threshold(np.uint8(dilatedIm), 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

finalimg = cv2.cvtColor(np.float32(finalImg), cv2.COLOR_GRAY2BGR)
finalimg = cv2.GaussianBlur(np.uint8(finalimg), (13,13), 1)
cv2.drawContours(finalimg, contours, -1, (0,0,255), 1, cv2.LINE_8)

cv2.imwrite(os.path.join(folder, "contours.bmp"), finalimg)
cv2.imshow('Final image with contours', resize(finalimg, 1000,1000))
cv2.waitKey(0)
