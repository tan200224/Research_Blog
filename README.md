# Image Segmentation

## Threshold, Binarization, and Adaptive Thresholding

Threshold is a type of image segmentation that divides the image based on its pixel value. 
There are different thresholding methods, such as: BINARY, BINARY_INV, IRUNC, TOZERO, TOZERO_INV.
    
    
<img width="340" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/6ac0ca24-f155-4303-9b1a-8bfb7c62f551">

As the image shows, The different threshold methods create different effects on the image. 

**BINARY**: It separates the colors of the image into white and black. If the pixel value of the image is less than the threshold value, it will turn black, while if it is greater than the threshold value, then the color will become white.

**TRUNC**: When the pixel values are greater than the threshold value, it will set all the pixel values into the threshold value. However, if the pixel values are less than the threshold value, then I will turn them into black.

**TOZERO** When the pixel values are greater than the threshold value, the pixel value will remain unchanged. Otherwise, it will turn black.

    cv2.threshold(image, threshold value, 255, cv2.THRESHOLD_METHOD (THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV))

    # It will return two values.
    ret, threshold = cv2.threshold(image, 200, 255, Cv2.TRESHOLD_BINARY

## Adaptive Thresholding
1. ADAPTIVE_THRESH_MEAN_C
2. THRESH_OTSU

    cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst
* src – Source 8-bit single-channel image.
* dst – Destination image of the same size and the same type as src .
* maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
* adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
* thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
* blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
C – Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.


To Compare each of their performance:
### Orginal Image
![img](https://github.com/tan200224/Blog/assets/68765056/5d436d5c-f576-47e8-82f6-05d2bbb3adc6)
### cv2.threshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.TRESH_BINARY, 3, 5)
![image](https://github.com/tan200224/Blog/assets/68765056/5ad71aca-dca5-40f3-85d2-76e5c431e7e5)
### cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTS)
![image](https://github.com/tan200224/Blog/assets/68765056/3f2f0adc-12b1-4eb2-88c4-60b8df9f9a9a)
### Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
![image](https://github.com/tan200224/Blog/assets/68765056/15a0e6ec-8d50-46d1-b702-7e7b937803e2)


## Skimage Threshold Local
threshold_local(image, block_size, offset=10)
It calculates the threshold in the region with a characteristic size block_size surrounding each pixel.

    from skimage.filters import threshold_local

    # We get the Value component from the HSV color space
    # then we apply adaptive thresholding
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 25, offset=15, method="gaussian")

    # Apply the threshold operation
    thresh = (V > T).astype("uint8") * 255
    imshow("threshold_local", thresh)
![image](https://github.com/tan200224/Blog/assets/68765056/43025fc0-e0fc-4b07-939f-03db15aee88e)

    
