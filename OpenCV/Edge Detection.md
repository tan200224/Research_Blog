# Dilation, Erosion and edg Detection
1. Dilation: Adds pixels to the boundaries of object in an image
2. Erosion: Remove pixels from the boundaries of object in an image
3. opening: Erosion first, then dilation
4. closing: Dilation first, then erosion

### Orginal image
![image](https://github.com/tan200224/Blog/assets/68765056/2c255309-8ea0-4417-9bcc-12b1c17cfdb9)
### Erosion
![image](https://github.com/tan200224/Blog/assets/68765056/a7e9b2b8-5d60-4098-80a2-64a05caaa550)
### Dilation
![image](https://github.com/tan200224/Blog/assets/68765056/6f013153-7c07-4b56-94bc-46628826afde)
### Opening
![image](https://github.com/tan200224/Blog/assets/68765056/90feac4b-7a38-4549-a76c-23d557eaa93e)
### Closing
![image](https://github.com/tan200224/Blog/assets/68765056/9d93cb02-23b7-4252-8811-f07fc69d02cf)


## Canny Edge Detection
The edge detection is based on the change of colors. 
![image](https://github.com/tan200224/Blog/assets/68765056/628dbbff-58ce-4ce8-8f18-4681ea6bedbb)
As the image shown, the changes in colors are able to be defined as a function. Whenever there is a change in color, it will produce changes in pixel value. Then we can use the derivative to identify
if there are edges there. 

The function take three input (image, minValue, maxValue, aperture_size (default by 3)

**NOTE**: If the pixel value is less than the minValue, it will not be considered as an edge. The value in between minValue and maxValue might be considered to be an edge depending on how their intensities are connected. However, anything that is greater than the maxValue will be consider as an edge.

    canny = cv2.Canny(image, 50, 120)
    imshow('Canny 1', canny)
![image](https://github.com/tan200224/Blog/assets/68765056/14a8655e-469c-4763-b0c0-1a9fda0dc269)

    canny = cv2.Canny(image, 200, 240)
    imshow('Canny Narrow', canny)
 ![image](https://github.com/tan200224/Blog/assets/68765056/acb8fd3d-4d10-4e44-b668-8db192b0c205)


## Auto Canny
    def autoCanny(image):
    # Finds optimal thresholds based on median image pixel intensity
    blurred_img = cv2.blur(image, ksize=(5,5))
    med_val = np.median(image)
    lower = int(max(0, 0.66 * med_val))
    upper = int(min(255, 1.33 * med_val))
    edges = cv2.Canny(image=image, threshold1=lower, threshold2=upper)
    return edges

    auto_canny = autoCanny(image)
    imshow("auto canny", auto_canny)
![image](https://github.com/tan200224/Blog/assets/68765056/1eeee277-06ee-4451-930e-9692cb2182aa)




