import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def ACE(image):
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    b = clahe.apply(image[:, :, 0])
    g = clahe.apply(image[:, :, 1])
    r = clahe.apply(image[:, :, 2])
    equalized = np.dstack((b, g, r))
    return equalized


def non_linear_histogram_equalization(img, gamma_value=1.0):
    # Преобразование изображения в YCbCr цветовое пространство
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Извлечение яркостной составляющей изображения
    img_y = img_ycbcr[:,:,0].astype("float")

    # Нормализация яркости до [0, 1]
    img_y_norm = img_y / 255.0

    # Вычисление гистограммы изображения
    hist, bins = np.histogram(img_y_norm, 256, [0,1])

    # Создание функции преобразования
    cdf = np.cumsum(hist) / np.sum(hist)
    t_func = np.power(cdf, gamma_value)

    # Применение функции преобразования
    img_y_norm_eq = np.interp(img_y_norm, bins[:-1], t_func)

    # Восстановление оригинальной шкалы яркости
    img_y_eq = np.round(img_y_norm_eq * 255.0).astype("uint8")

    # Объединение яркостной составляющей с другими компонентами в YCbCr
    img_ycbcr_eq = img_ycbcr.copy()
    img_ycbcr_eq[:,:,0] = img_y_eq

    # Конвертация в BGR цветовое пространство
    img_eq = cv2.cvtColor(img_ycbcr_eq, cv2.COLOR_YCrCb2BGR)

    return img_eq
from scipy.signal import convolve2d


def high_boost(image):
    gauss = cv2.GaussianBlur(image, (7, 7), 0)
    unsharp_image = cv2.addWeighted(image, 2, gauss, -1, 0)
    return unsharp_image
def nlhe(image, kernel_size=3, clip_limit=0.01):
    image = np.float32(image)/255.0

    # Создание ядра верхнего пути
    half_kernel_size = kernel_size // 2
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    r = np.sqrt(x*x+y*y)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[r <= 1] = 1

    # Разделение каналов изображения
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    # Обработка каждого канала
    for channel in [red, green, blue]:
        # Вычисление гистограммы
        hist, bins = np.histogram(channel, bins=np.arange(257))
        hist = np.cumsum(hist) / hist.sum()

        # Определение порогового значения
        threshold = clip_limit / (2 * half_kernel_size * half_kernel_size)

        # Локальное выравнивание гистограммы
        app_ref = convolve2d(channel, kernel, mode='same')
        ref_mean = convolve2d(channel, kernel, mode='same') / kernel.sum()
        ref_var = convolve2d(channel**2, kernel, mode='same') / kernel.sum() - ref_mean**2
        clip_val = np.max(ref_var) * threshold
        ref_var.clip(min=clip_val, out=ref_var)
        ref_scale = np.sqrt(ref_var) / 128
        ref_channel = (channel - ref_mean) / ref_scale
        ref_hist, bins = np.histogram(ref_channel, bins=np.arange(-128, 128))
        ref_hist = np.cumsum(ref_hist) / ref_hist.sum()
        mapping = np.interp(channel.flatten(), bins[:-1], ref_hist)
        channel[:] = np.interp(mapping, ref_hist, bins[:-1]).reshape(channel.shape)

    # Конвертация изображения обратно в формат uint8 в диапазоне [0, 255]
    image = np.uint8(np.clip(image * 255.0, 0, 255))

    return image

def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def imageEnhancement(image):
    #res = ACE(image)
    #res = non_linear_histogram_equalization(image)
    res = high_boost(image)
    res = nlhe(res)
    #res = adjust_gamma(res)
    return res

 
class Preprocessing:
    def run(self, mainDir:str, outputDir:str):
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
                inputImage = cv2.imread(currentImg)
                prepImage = imageEnhancement(inputImage)
                cv2.imwrite(resulImg, prepImage)
        return None
