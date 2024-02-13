import numpy as np 
import pandas as pd 
from PIL import Image

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# open the image and return 3 matrices, each corresponding to one channel (R, G and B channels)
def openImage(imagePath):
    imOrig = Image.open(imagePath)
    im = np.array(imOrig)

    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]
     
    return [aRed, aGreen, aBlue, imOrig]

def compressSingleChannel(channelDataMatrix, singularValuesLimit):
    uChannel, sChannel, vhChannel = np.linalg.svd(channelDataMatrix)
    aChannelCompressed = np.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = np.matmul(uChannel[:, 0:k], np.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = np.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed

def calculate_psnr(original, compressed):
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    
    mse = np.mean((original - compressed) ** 2)
    max_pixel_value = np.amax(original)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr
image1 = openImage('/kaggle/input/finalmathematics-for-machine-learning-assignment-1/part2-image1.jpg')
image2 = openImage('/kaggle/input/finalmathematics-for-machine-learning-assignment-1/part2-image2.jpeg')

images = (image1,image2)

image_dimensions = ((900,1280),(1920,1080))
singularValuesLimit = (330,440)
for i in (0,1):
    image_data = images[i]
    CompressedRed = compressSingleChannel(image_data[0], singularValuesLimit[i])
    CompressedGreen = compressSingleChannel(image_data[1], singularValuesLimit[i])
    CompressedBlue = compressSingleChannel(image_data[2], singularValuesLimit[i])
    
    originalImage = image_data[3]
    
    imr = Image.fromarray(CompressedRed, mode=None)
    img = Image.fromarray(CompressedGreen, mode=None)
    imb = Image.fromarray(CompressedBlue, mode=None)
    
    psnr_red = calculate_psnr(image_data[0], CompressedRed)
    psnr_green = calculate_psnr(image_data[1], CompressedGreen)
    psnr_blue = calculate_psnr(image_data[2], CompressedBlue)
    
    
    
    newImage = Image.merge("RGB", (imr, img, imb))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(originalImage)
    axs[0].set_title('Original Image')

    axs[1].imshow(newImage)
    axs[1].set_title('Compressed Image')
    
    plt.show()
    
    mr,mc = image_dimensions[i]
    originalSize = mr * mc * 3
    compressedSize = singularValuesLimit[i] * (1 + mr + mc) * 3
    
    print('Original size:')
    print(originalSize)

    print('Compressed size:')
    print(compressedSize)

    print('Ratio compressed size / original size:')
    ratio = compressedSize * 1.0 / originalSize
    print(ratio)
    
    print('Compressed image size is ' + str(round(ratio * 100, 2)) + '% of the original image ')
    
    print("PSNR (Red channel):", psnr_red)
    print("PSNR (Green channel):", psnr_green)
    print("PSNR (Blue channel):", psnr_blue)