import numpy as np
import cv2
import time
import os
from PIL import Image
import csv
import requests
import os
import json
import asyncio
import aiofiles
import aiohttp
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass

PATH = "data_lab4"

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Time {func.__name__}: {elapsed} sec.")
        return result
    return wrapper

def async_time_decorator(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\nTotal execution time for '{func.__name__}': {elapsed:.2f} seconds.")
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

class ImageProcessor:
    __slots__ = ("_path", "_executor")

    def __init__(self, path: str = PATH) -> None:
        self._path = path
        os.makedirs(self._path, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=8)

    @staticmethod
    def _run_processing_worker(image: np.ndarray, index: int, obj_id: str):
        print(f"Processing image {index} (ID: {obj_id}) started")

        art = Artwork(image, {"objectID": obj_id})
        
        results = {
            "index": index,
            "object_id": obj_id,
            "grey": art.greyscale(),
            "sharpen": art.sharpen(),
            "blur": art.gaussian_blur(),
            "sobel": art.sobel(),
            "original": image
        }

        print(f"Processing image {index} (ID: {obj_id}) finished")

        return results
    
    async def _download_single(self, session, index, obj_id):
        print(f"Downloading {index} started")

        async with session.get(
            f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
        ) as response:
            object_data = await response.json()
            image_url = object_data['primaryImage']

        async with session.get(image_url) as img_res:
            img = await img_res.read()

        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(f"Downloading {index} finished")

        return index, obj_id, img_rgb
    
    async def get_painting(self, csv_file: str, count: int):
        with open(csv_file, newline='', encoding='utf-8-sig') as museum_csv:
            museum_info = csv.DictReader(museum_csv)
            painting_count = 0
            yielded_count = 0
            
            for row in museum_info:
                if row['Classification'] == 'Paintings':
                    painting_count += 1
                    if painting_count < 1235:
                        continue

                    yield painting_count, row
                    yielded_count += 1
                    
                    if yielded_count >= count:
                        break
    
    async def download_artwork(self, paintings):
        semaphore = asyncio.Semaphore(5)

        async with aiohttp.ClientSession() as session:

            async def limited_download(index, painting):
                async with semaphore:
                    return await self._download_single(
                        session, index, painting['Object ID']
                    )

            tasks = []
            async for index, painting in paintings:
                #task = asyncio.create_task(
                #    limited_download(index, painting)
                #)
                task = limited_download(index, painting)
                print(f"type(task) = {type(task)}")
                tasks.append(task)

            for task in asyncio.as_completed(tasks):
                print(f"AAAAAAAAAA {type(task)}")
                yield await task

    async def parall_processing(self, source):
        loop = asyncio.get_event_loop()
        tasks = []
        async for index, obj_id, img in source:

            processed = loop.run_in_executor(
                self._executor, self._run_processing_worker, img, index, obj_id
            )
            print(f"type(processed) = {type(processed)}")
            tasks.append(processed)
        for completed_task in asyncio.as_completed(tasks):
            yield await completed_task

    
    async def save_image(self, processed):
        async for item in processed:
            idx = item['index']
            obj_id = item['object_id']
            print(f"Saving results for image {idx} (ID: {obj_id}) started")
            for key in ['original', 'grey', 'sharpen', 'blur', 'sobel']:
                img = item[key]
                filename = f"{idx}_{obj_id}_{key}.png"
                save_path = os.path.join(self._path, filename)
                
                img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
                
                _, buf = cv2.imencode('.png', img_to_save)
                async with aiofiles.open(save_path, 'wb') as f:
                    await f.write(buf.tobytes())
            print(f"Image {idx} (ID: {obj_id}) saved successfully")

    @async_time_decorator
    async def run(self, csv_file, count):
        source = self.get_painting(csv_file, count)
        processed = self.download_artwork(source)
        process_gen = self.parall_processing(processed)
        await self.save_image(process_gen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    processor = ImageProcessor()
    asyncio.run(processor.run("/Users/krovich/Desktop/6301kroviakovds/data/MetObjects.csv", args.count))
