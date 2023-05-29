import cv2
import numpy as np

#initialization of variables used within the trackbars
alpha = 0
alpha_max = 500
beta = 0
beta_max = 350
gamma = 1.0
gamma_max = 200

def basicLinearTransform():
    #Used instead of for loops
    res = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)
    img_corrected = cv2.hconcat([resized, res])
    cv2.imshow("Brightness and contrast adjustments", img_corrected)

def gammaCorrection():
    ## [changing-contrast-brightness-gamma-correction]
    #lookUpTable used to improve performance(256 valued need to be calculated once)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(resized, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

    img_gamma_corrected = cv2.hconcat([resized, res])
    cv2.imshow("Gamma correction", img_gamma_corrected)

def on_linear_transform_alpha_trackbar(val):
    global alpha
    alpha = val / 100
    basicLinearTransform()

def on_linear_transform_beta_trackbar(val):
    global beta
    beta = val - 100
    basicLinearTransform()

def on_gamma_correction_trackbar(val):
    global gamma
    gamma = val / 100
    gammaCorrection()

def readImagesAndTimes():
    times = np.array([1 / 30.0, 0.25, 2.5, 15.0], dtype=np.float32)

    filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images, times


if __name__ == '__main__':
    # Read images and exposure times
    print("Reading images")

    images, times = readImagesAndTimes()

    # Align input images
    print("Aligning images")
    # MTB = median threshold bitmaps
    # 1 to pixels brighter than median luminance, 0 otherwise
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    # Obtain Camera Response Function (CRF)
    # Filtering "bad pixels"
    print("Calculating Camera Response Function (CRF)")
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)

    # Merge images(exposures) into an HDR linear image
    print("Merging images into one HDR image")
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
    # Save HDR image
    cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
    print("Saved hdrDebevec.hdr ")

    #tonemap = converting HDR image to 8 bits/channel image
    # # Tonemap using Drago's method to obtain 24-bit color image
    print("Tonemapping using Drago's method")
    #Values obtained by trial and error
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    #Pleasant results multiplying by 3
    ldrDrago = 3 * ldrDrago
    cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
    print("Saved ldr-Drago.jpg")

    # # Tonemap using Reinhard's method to obtain 24-bit color image
    print("Tonemapping using Reinhard's method")
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
    print("Saved ldr-Reinhard.jpg")

    # # Tonemap using Mantiuk's method to obtain 24-bit color image
    print("Tonemapping using Mantiuk's method")
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
    print("Saved ldr-Mantiuk.jpg")


    #img_original = cv2.imread("objects_24bits.bmp")
    img_original = cv2.imread("ldr-Drago.jpg")

    #resize of original generated image
    scale_percent = 20
    width = int(img_original.shape[1] * scale_percent / 100)
    height = int(img_original.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img_original, dim, interpolation=cv2.INTER_AREA)

    img_corrected = np.empty((resized.shape[0], resized.shape[1] * 2, resized.shape[2]),
                             resized.dtype)
    img_gamma_corrected = np.empty((resized.shape[0], resized.shape[1] * 2, resized.shape[2]),
                                   resized.dtype)

    img_corrected = cv2.hconcat([resized, resized])
    img_gamma_corrected = cv2.hconcat([resized, resized])

    cv2.namedWindow('Brightness and contrast adjustments')
    cv2.namedWindow('Gamma correction')

    alpha_init = int(alpha + 100)
    cv2.createTrackbar('Alpha gain (contrast)', 'Brightness and contrast adjustments', alpha_init, alpha_max,
                      on_linear_transform_alpha_trackbar)
    beta_init = beta + 100
    cv2.createTrackbar('Beta bias (brightness)', 'Brightness and contrast adjustments', beta_init, beta_max,
                      on_linear_transform_beta_trackbar)
    gamma_init = int(gamma * 100)
    cv2.createTrackbar('Gamma correction', 'Gamma correction', gamma_init, gamma_max, on_gamma_correction_trackbar)

    on_linear_transform_alpha_trackbar(alpha_init)
    on_gamma_correction_trackbar(gamma_init)

    img_original = cv2.imread("ldr-Drago.jpg")
    reinhard = cv2.imread("ldr-Reinhard.jpg")
    mantiuk = cv2.imread("ldr-Mantiuk.jpg")

    # resize
    widthR = int(reinhard.shape[1] * scale_percent / 100)
    heightR = int(reinhard.shape[0] * scale_percent / 100)
    dimR = (widthR, heightR)
    resizedR = cv2.resize(reinhard, dimR, interpolation=cv2.INTER_AREA)
    pathReinhard = r'/home/dz/Desktop/ProiectPI/ldr-Reinhard.jpg'
    imageR = cv2.imread(pathReinhard)

    # resize
    widthM = int(mantiuk.shape[1] * scale_percent / 100)
    heightM = int(mantiuk.shape[0] * scale_percent / 100)
    dimM = (widthM, heightM)
    resizedM = cv2.resize(mantiuk, dimM, interpolation=cv2.INTER_AREA)
    pathMantiuk = r'/home/dz/Desktop/ProiectPI/ldr-Mantiuk.jpg'
    imageM = cv2.imread(pathMantiuk)

    window_nameD = "imageD"
    window_nameR = "imageR"
    window_nameM = "imageM"

    cv2.imshow(window_nameD, resized)
    cv2.imshow(window_nameR, resizedR)
    cv2.imshow(window_nameM, resizedM)

    cv2.waitKey()

