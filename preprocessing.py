import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def gammaCorrection(inputImgName:str, outputImgName:str, sigma =3, plot_hist=False):
    # image = cv2.imread(inputImgName)
    # clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    # b = clahe.apply(image[:, :, 0])
    # g = clahe.apply(image[:, :, 1])
    # r = clahe.apply(image[:, :, 2])
    # equalized = np.dstack((b, g, r))
    # cv2.imwrite(outputImgName, equalized)

    # img = cv2.imread("1-1.jpg", cv2.IMREAD_COLOR)
    #
    # # normalize float versions
    # norm_img2 = cv2.normalize(img, None, alpha=0, beta=0.8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #
    # # scale to uint8
    # norm_img2 = np.clip(norm_img2, 0, 1)
    # norm_img2 = (255 * norm_img2).astype(np.uint8)
    #
    # cv2.imwrite("new.jpg", norm_img2)

    img = cv2.imread(inputImgName)
    original = img.copy()
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    cv2.imwrite(outputImgName, img)

    # img = cv2.imread(inputImgName)
    # stretched = np.zeros(img.shape)
    # for i in range(img.shape[2]):  # looping through the bands
    #     band = img[:, :, i]  # copiying each band into the variable `band`
    #     if np.min(band) < 0:  # if the min is less that zero, first we add min to all pixels so min becomes 0
    #         band = band + np.abs(np.min(band))
    #     band = band / np.max(band)
    #     band = band * 255  # convertaning values to 0-255 range
    #     if plot_hist:
    #         plt.hist(band.ravel(), bins=256)  # calculating histogram
    #         plt.show()
    #     # plt.imshow(band)
    #     # plt.show()
    #     std = np.std(band)
    #     mean = np.mean(band)
    #     max = mean + (sigma * std)
    #     min = mean - (sigma * std)
    #     band = (band - min) / (max - min)
    #     band = band * 255
    #     # this streching cuases the values less than `mean-simga*std` to become negative
    #     # and values greater than `mean+simga*std` to become more than 255
    #     # so we clip the values ls 0 and gt 255
    #     band[band > 255] = 255
    #     band[band < 0] = 0
    #     print('band', i, np.min(band), np.mean(band), np.std(band), np.max(band))
    #     if plot_hist:
    #         plt.hist(band.ravel(), bins=256)  # calculating histogram
    #         plt.show()
    #     stretched[:, :, i] = band
    # stretched = stretched.astype('int')
    # cv2.imwrite(outputImgName, stretched)

mainDir = 'animal'
outputDir = 'preprocessedAnimals'

if not os.path.exists(outputDir):
    os.makedirs(outputDir)


for dir in os.listdir(mainDir):
    currentDir = os.path.join(mainDir, dir)
    resultDir = outputDir + '/' + dir
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    for image in os.listdir(currentDir):
        currentImg = os.path.join(currentDir, image)
        resulImg = resultDir + '/' + image
        gammaCorrection(currentImg, resulImg)


 
