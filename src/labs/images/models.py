import numpy as np
import cv2
import time
import os
from PIL import Image
import csv
import requests
import os
import json
from dataclasses import dataclass

PATH = "paintings"
IMAGE_NAME = "1972_278_8_O.jpg"

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Time {func.__name__}: {elapsed} sec.")
        return result
    return wrapper

@dataclass(slots=True)
class Artwork:
    _image: np.array
    _metadata: dict[str]

    @property
    def image(self):
        return self._image.copy()
    
    @property
    def metadata(self) -> dict[str]:
        return self._metadata
    
    def __str__(self) -> str:
        title = self._metadata.get("title", "Unknown")
        object_id = self._metadata.get("objectID", "Unknown")
        artist = self._metadata.get("artistDisplayName", "Unknown")
        shape = self._image.shape
        return f"Artwork(title={title}, objectID={object_id}, artist={artist}, shape={shape})"

    def __add__(self, other: "Artwork") -> "Artwork":
        if not isinstance(other, Artwork):
            raise TypeError("only Artwork types")
        
        a = self.image

        if a.shape != other.image.shape:
            raise ValueError("size must be equal")
        
        a[0,0,0]=255

        result = np.clip(
            a.astype(np.float32) + other.image.astype(np.float32),0,255).astype(np.uint8)

        

        merged_metadata = {
            "title": f"{self.metadata.get('title', 'Unknown')} + {other.metadata.get('title', 'Unknown')}",
            "source_ids": [
                self.metadata.get("objectID"),
                other.metadata.get("objectID")
            ]
        }
        return Artwork(result, merged_metadata)

    def greyscale(self)-> np.ndarray:
        colors = np.array([0.299, 0.587, 0.114])
        grey = np.sum(colors * self._image, axis=2)
        return np.clip(grey, 0, 255).astype(np.uint8)

        
    @staticmethod
    def filter2d(image: np.ndarray, kernel: np.ndarray, return_float: bool = False) -> np.ndarray:
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
                    if is_color:
                        a = np.transpose(region,(2,0,1))
                        output[y, x] = np.sum(a * kernel, axis=(1,2))
                    else:
                        output[y, x] = np.sum(region * kernel)
            if return_float or not is_color:
                return output
            else:
                return np.clip(output, 0, 255).astype(np.uint8)

    def sharpen(self) -> np.ndarray:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return self.filter2d(self._image, kernel)

    def gaussian_blur(self, kernel_size: int = 5, sigma: float = 1.0)  -> np.ndarray:
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return self.filter2d(self._image, kernel)

    def sobel(self)  -> np.ndarray:
        #grey = self.greyscale()

        sobel_x = np.array([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]])
        sobel_y = np.array([[1,2,1],
                            [0,0,0],
                            [-1,-2,-1]])
            

        gx = self.filter2d(self.image, sobel_x, return_float=True)
        gy = self.filter2d(self.image, sobel_y, return_float=True)

        magnitude = np.sqrt(gx**2 + gy**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

def counter(func):
    def wrapper(*args):
        wrapper.calls += 1
        print(f"Call {wrapper.calls} from {func.__name__}")
        return func(*args)
    wrapper.calls = 0
    return wrapper

class ImageProcessor:
    __slots__ = ("_path",)

    def __init__(self, path: str = PATH) -> None:
        self._path = path
        os.makedirs(self._path, exist_ok=True)
    
    @counter
    def get_painting(self, csv_file: str, index: int):
        with open(csv_file, newline='', encoding='utf-8-sig') as museum_csv:
            museum_info = csv.DictReader(museum_csv)
            painting = [row for row in museum_info if row['Classification'] == 'Paintings'][index]
            return painting
    
    def download_artwork(self, painting: dict[str], image_name: str = IMAGE_NAME):
        response = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{painting['Object ID']}")
        object_data = response.json()

        image = object_data['primaryImage']
        image_response = requests.get(image)

        image_path = os.path.join(self._path, image_name)
        with open(image_path, "wb") as f:
            f.write(image_response.content)

        json_path = os.path.join(self._path, image_name.replace(".jpg", ".json"))
        with open(json_path, "w") as f:
            json.dump(object_data, f, indent=2)
        
        img_pil = Image.open(os.path.join(PATH,IMAGE_NAME))
        img_np = np.array(img_pil)

        return Artwork(img_np, object_data)
    
    @counter
    def save_image(self, image: np.ndarray, filename: str) -> str:
        save_path = os.path.join(self._path, filename)

        if len(image.shape) == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = image

        cv2.imwrite(save_path, image_to_save)
        return save_path

    def process_artwork(self, artwork: Artwork, base_name: str = IMAGE_NAME) -> None:

        grey = artwork.greyscale()
        self.save_image(grey, base_name.replace(".jpg", "_grey.jpg"))

        sharpened = artwork.sharpen()
        self.save_image(sharpened, base_name.replace(".jpg","_sharpen.jpg"))

        blurred = artwork.gaussian_blur()
        self.save_image(blurred, base_name.replace(".jpg","_gaussian.jpg"))

        edges = artwork.sobel()
        self.save_image(edges, base_name.replace(".jpg","_sobel.jpg"))



if __name__ == "__main__":
    processor = ImageProcessor()
    processor2 = ImageProcessor()
    index = 1234

    printing = processor.get_painting("/Users/krovich/Desktop/6301krovyakovds/MetObjects.csv", index)
    printing2 = processor.get_painting("/Users/krovich/Desktop/6301krovyakovds/MetObjects.csv", index)

    my_art = processor.download_artwork(printing)
    my_art2 = processor.download_artwork(printing2)
    processor.save_image(my_art.image, "new_picture.jpg")

    my_art2 = Artwork(my_art.sobel(), my_art.metadata)

    print(my_art.image.shape)
    print(my_art2.image.shape)
    #print(my_art) 
    
    #processor.process_artwork(my_art, IMAGE_NAME)

    painting = my_art + my_art2

    processor.save_image(my_art2.image, "new_picture2.jpg")
    printing3 = processor.get_painting("/Users/krovich/Desktop/6301krovyakovds/MetObjects.csv", index)

