import cv2
import numpy as np
import os

def show_resized_image(window_name, img, width=600):
    h, w = img.shape[:2]
    height = int(h * (width / w))
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, img)

def process_coins(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("找不到圖片！")
        return

    # 處理流程
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_erode = np.ones((7, 7), np.uint8)
    kernel_dialation = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(thresh, kernel_erode, iterations=2)
    dialation = cv2.dilate(erosion, kernel_dialation, iterations=5)

    show_resized_image('Original', img, width=500)
    show_resized_image('Threshold', thresh, width=500)
    show_resized_image('eroded', erosion, width=500)
    show_resized_image('Result', dialation, width=500)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    target_image = '/home/user/ws/src/opencv_demo/coin.jpg'
    process_coins(target_image)

if __name__ == "__main__":
    main()