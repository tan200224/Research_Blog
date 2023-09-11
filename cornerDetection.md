# Finding Corners
1. To use Harris corners to find corncers
2. Use good features to track

## What is a corner?
A corner is a point whose local neighborhood stands in two dominant and different edge directions. In other words, a corner can be interpreted as the junction of two edges, where an edge is a sudden change in image brightness. Corners are the important features in the image, and they are generally termed as interest points which are invariant to translation, rotation, and illumination.
![image](https://github.com/tan200224/Blog/assets/68765056/9b09ba76-b587-4978-aee9-2d6124a55081)

### Harris Corner Detection
cv2.cornerHarris(input image, block size, ksize, k)
Input image - should be grayscale and float32 type.
blockSize - the size of neighborhood considered for corner detection
ksize - aperture parameter of Sobel derivative used.
k - harris detector free parameter in the equation
Output – array of corner locations (x,y)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The cornerHarris function requires the array datatype to be float32
    gray = np.float32(gray)
    
    harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)
    
    #We use dilation of the corner points to enlarge them\
    kernel = np.ones((7,7),np.uint8)
    harris_corners = cv2.dilate(harris_corners, kernel, iterations = 2)
    
    # Threshold for an optimal value, it may vary depending on the image.
    image[harris_corners > 0.025 * harris_corners.max() ] = [255, 127, 127]

![image](https://github.com/tan200224/Blog/assets/68765056/329e4816-7573-4707-aaba-e1aecd97b5db)


### Good Feature Track
cv2.goodFeaturesToTrack(input image, maxCorners, qualityLevel, minDistance)

Input Image - 8-bit or floating-point 32-bit, single-channel image.
maxCorners – Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
qualityLevel – Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure (smallest eigenvalue). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality - - measure less than 15 are rejected.
minDistance – Minimum possible Euclidean distance between the returned corners.

    corners = cv2.goodFeaturesToTrack(gray, 150, 0.0005, 10)
    
    for corner in corners:
        x, y = corner[0]
        x = int(x)
        y = int(y)
        cv2.rectangle(img,(x-10,y-10),(x+10,y+10),(0,255,0), 2)
        
    imshow("Corners Found", img)

![image](https://github.com/tan200224/Blog/assets/68765056/f08453b6-d042-41e1-b3b9-cdc2786d0083)



