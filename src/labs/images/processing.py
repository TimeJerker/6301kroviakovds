import numpy as np
import cv2
import time
import os
from PIL import Image

PATH = "paintings"
IMAGE_NAME = "1972_278_8_O.jpg"

img_pil = Image.open(os.path.join(PATH,IMAGE_NAME))
img_np = np.array(img_pil)

def greyscale():
    start_time = time.time()
    colors = np.array([0.299, 0.587, 0.114])
    my_grey = np.sum(colors * img_np, axis=(2))
    #my_grey = (0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]).astype(np.uint8)
    my_time = time.time() - start_time

    start_time = time.time()
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv_time = time.time() - start_time

    print("My grey:", my_time)
    print("OpenCV grey: ", cv_time)

    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_grey_my.jpg"), my_grey)
    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_grey_cv.jpg"), cv_grey)

def my_filter2D(image, kernel, return_float: bool = False):
        k_h, k_w = kernel.shape[:2]
        pad_h, pad_w = k_h // 2, k_w // 2

        is_color = len(image.shape) == 3

        if is_color:
            i_h, i_w, channels = image.shape
            pad_width = ((pad_h,pad_h),(pad_w,pad_w), (0,0))
            output = np.zeros((i_h,i_w,channels), dtype=np.float32)
        else:
            i_h, i_w = image.shape
            pad_width = ((pad_h,pad_h),(pad_w,pad_w))
            output = np.zeros((i_h,i_w), dtype=np.float32)
        
        padded_img = np.pad(image, pad_width, mode='edge')

        for y in range(i_h):
            for x in range(i_w):
                region = padded_img[y : y + k_h, x : x + k_w]
                a = np.transpose(region,(2,0,1))
                output[y, x] = np.sum(a * kernel, axis=(1,2))
        if return_float or not is_color:
            return output
        else:
            return np.clip(output, 0, 255).astype(np.uint8)

def filter():

    # if is_not_color:
    #     img_pil = Image.open(os.path.join(PATH,"1972_278_8_O_grey_my.jpg"))
    #     img_np = np.array(img_pil)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    start_time = time.time()
    my_filtered = my_filter2D(img_np, kernel)
    my_time = time.time() - start_time

    start_time = time.time()
    cv_filtered = cv2.filter2D(img_np, -1, kernel)
    cv_time = time.time() - start_time

    print("My filter2D:", my_time)
    print("OpenCV filter2D: ", cv_time)

    my_filtered_BGR = cv2.cvtColor(my_filtered, cv2.COLOR_RGB2BGR)
    cv_filtered_BGR = cv2.cvtColor(cv_filtered, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_filter2D_my.jpg"), my_filtered_BGR)
    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_filter2D_cv.jpg"), cv_filtered_BGR)

def gaussian_blur():
    def my_gaussian_blur(image, kernel_size: int = 5, sigma: float = 1.0):
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return my_filter2D(image, kernel)
    
    start_time = time.time()
    my_blurred = my_gaussian_blur(img_np)
    my_time = time.time() - start_time

    start_time = time.time()
    cv_blurred = cv2.GaussianBlur(img_np,(5,5), 1.0)
    cv_time = time.time() - start_time

    print("My blurred:", my_time)
    print("OpenCV blurred: ", cv_time)

    my_blurred_BGR = cv2.cvtColor(my_blurred, cv2.COLOR_RGB2BGR)
    cv_blurred_BGR = cv2.cvtColor(cv_blurred, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_gaussian_blur_my.jpg"), my_blurred_BGR)
    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_gaussian_blur_cv.jpg"), cv_blurred_BGR)

def sobel():
    def my_sobel(image):
        sobel_x = np.array([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]])
        sobel_y = np.array([[1,2,1],
                            [0,0,0],
                            [-1,-2,-1]])
        
        Gx = my_filter2D(image, sobel_x, True)
        Gy = my_filter2D(image, sobel_y, True)

        magnitude = np.sqrt(Gx.astype(np.float32)**2 + Gy.astype(np.float32)**2)
        magnitude = np.clip(magnitude,0,255).astype(np.uint8)
        return magnitude

    start_time = time.time()
    my_edges = my_sobel(img_np)
    my_time = time.time() - start_time

    start_time = time.time()
    sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
    cv_magnitude = cv2.magnitude(sobel_x, sobel_y)
    cv_edges = np.clip(cv_magnitude,0,255).astype(np.uint8)
    cv_time = time.time() - start_time

    print("My sobel:", my_time)
    print("OpenCV sobel: ", cv_time)

    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_sobel_my.jpg"), my_edges)
    cv2.imwrite(os.path.join(PATH, "1972_278_8_O_sobel_cv.jpg"), cv_edges)
