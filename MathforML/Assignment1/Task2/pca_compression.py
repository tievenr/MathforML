import numpy as np 
import pandas as pd 

from PIL import Image
import matplotlib.pyplot as plt


def openImage(image_path):
    image = Image.open(image_path)
    
    img_data = np.array(image)
    
    red_channel = img_data[:,:,0]
    green_channel = img_data[:,:,1]
    blue_channel = img_data[:,:,2]
    
    return red_channel,green_channel,blue_channel,image

def center_data(data):
    mean = np.mean(data)
    centered_data = data - mean
    return centered_data,mean

def PCA(centered_data,k):
    centered_data = centered_data.astype(np.float64)
    covariance_matrix = np.cov(centered_data,rowvar=False)
    eigenvalues,eigenvectors = np.linalg.eig(covariance_matrix)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:,sorted_indices[:k]]
    return top_eigenvectors

def compress_channel(channel,top_eigenvectors,mean):
    reduced_data = np.dot(channel,top_eigenvectors)
    
    compressed_data = np.dot(reduced_data,top_eigenvectors.T) + mean
    return compressed_data

def show_imageandmetrics(original_image, compressed_image):
    original_size = original_image.size[0] * original_image.size[1] * 3  # Width * Height * Number of channels (3 for RGB)
    compressed_size = compressed_image.size * compressed_image.itemsize  # Size in bytes
    compression_ratio = original_size / compressed_size
    
    mse = np.mean((np.array(original_image) - np.array(compressed_image)) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))

    print("Compression Factor Ratio:", compression_ratio)
    print("PSNR:", psnr)
    
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)  # Clip to valid range and cast to uint8

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image)
    plt.title('Compressed')

    plt.show()
image1 = openImage('/kaggle/input/finalmathematics-for-machine-learning-assignment-1/part2-image1.jpg')
image2 = openImage('/kaggle/input/finalmathematics-for-machine-learning-assignment-1/part2-image2.jpeg')

images = (image1,image2)

k=1000

for image in images:
    image_data = []
    for i in range(0,3):
        centered_data,mean = center_data(image[i])
        eigenvectors = PCA(centered_data,k)
        compressed_data = compress_channel(image[i],eigenvectors,mean)
        image_data.append(compressed_data)
    compressed_image = np.stack(image_data,axis=-1)
    original_image = image[3]
    show_imageandmetrics(original_image,compressed_image)
    