from images.processing import filter, greyscale, gaussian_blur, sobel

if __name__ == "__main__":
    tasks_to_run = [
        greyscale, 
        filter,
        gaussian_blur, 
        sobel
    ]
    
    for task in tasks_to_run:
        task()
        print()
        