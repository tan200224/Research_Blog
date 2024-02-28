# Contours - Drawing, Hierarchy and Modes

Contours are continous lines or curves that bound or cover the full boundary of an object.

## Applying cv2.findContours()
**NOTE**: for findCountours to work, the background has to be balck and foreground (i.e. the text or objects). Otherwise you'll need to invert the image by using cv2.bitwise_not(input image)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use a copy of your image, e.g. edged,copy() since findContours change the image
    contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), thickness = 2)
![image](https://github.com/tan200224/Blog/assets/68765056/d54dd63e-1c27-4938-ba8f-64ef642be5c6)
![image](https://github.com/tan200224/Blog/assets/68765056/f3769e18-5ede-412e-bf25-6c321b925490)


## We can use Canny edges instead of thresholding
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), thickness=2)
![image](https://github.com/tan200224/Blog/assets/68765056/7d1e978f-e16a-4dc7-ac75-7265eb2bb331)
![image](https://github.com/tan200224/Blog/assets/68765056/45592ce9-7bf8-4cf7-96a9-34d2bbfee9ad)


## Remember these Countouring steps
1. Grayscale
2. Threshold or Canny edge detection to binarized image


## Retrieval Modes

### RETR_LIST
Retrieves all the contours, but doesn't create any parent-child relationship. Parents and kids are equal under this rule, and they are just contours. ie they all belong to the same hierarchy level.


### RETR_EXTERNAL
Return only the extreme outer flag. All child contours are left behind. Unlike the RETR_LIST, this one will only return the outer line of the letters. It will not draw the contours inside the letters.

![image](https://github.com/tan200224/Blog/assets/68765056/af7ce38e-190f-4335-8010-6b8474d5b5ce)


### RETR_CCOMP
Retrieves all the contours and arranges them to a 2-level hierarchy. ie external contours of the object are placed in hierarchy_1. The contours of holes inside the object are placed in hierarchy_2. If any object is inside it, its contour is placed again in hierarchy-1 only. And its hole in hierachy_2 and so on.
      
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), thickness=2)
![image](https://github.com/tan200224/Blog/assets/68765056/654ed710-066d-478e-987c-a8d899fa8ecc)


### RETR_TREE
It retrieves all the contours and creates a full family hierarchy list

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), thickness=2)
![image](https://github.com/tan200224/Blog/assets/68765056/d95b8363-2b50-45d9-91f3-838e00dff8e7)


## Contouring modes

### CHAIN_APPROX_NONE
Only store the endpoint of the line, which can save some space.





