# Line, Circle, and Blob Detection
1. Houghlines
2. Probalilistic hough lines
3. Hough Circles
4. Blob Detection


## Line Detection Using Hough Lines
The Hough transform takes a binary edge map as input and attempts to locate edges placed as straight lines. The idea of hough transform is that every edge point in the edge map is transformed to all possible lines that could pass through that points.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 170, apertureSize = 3)
    
    # Run houghline using a rho accuracy of 1 pixel
    # theta accuracy of np.pi/180 whihc is 1 degree
    # our line threshold is set to 240 (number of points on line)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 240)
    
    # We iterate through each line and convert it to the format
    # Required by cv2.lines
    for line in lines:
      rho, theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a * rho
      y0 = b * rho
      x1 = int(x0 + 1000 * (-b))
      y1 = int(y0 + 1000 * (a))
      x2 = int(x0 - 1000 * (-b))
      y2 = int(y0 - 1000 * (a))
      cv2.line(image, (x1,y1), (x2,y2), (255,0,0), 2)
![image](https://github.com/tan200224/Blog/assets/68765056/2db36c5d-d840-475d-af54-439671c2daeb)
![image](https://github.com/tan200224/Blog/assets/68765056/93b16f5f-a4d7-4110-8058-ab1456284fcd)

## Probabilistic Hough Lines
It uses random sampling of the edge points. These algorithms can be divided based on how they map image space to parameter space.

cv2.HoughLinesP(binarized image, accuracy, accuracy, threshold, minimum line length, max line gap)

This would save more space compared to hough lines.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 170, apertureSize = 3)
    
    # Run hough line using a rho accuracy of 1 pixel
    # theta accuracy of np.pi/180, which is 1 degree
    # Our line threshold is set to 240 (the number of points on a line)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 3, 25)
    print(lines.shape)
    
    for x in range(0, len(lines)):
      for x1,y1,x2,y2 in lines[x]:
        cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)
    
    imshow('Probabilistic Hough Lines', image)
![image](https://github.com/tan200224/Blog/assets/68765056/3721e35d-f915-47d3-a07c-1b7e709c0705)
![image](https://github.com/tan200224/Blog/assets/68765056/5bb5e67c-28ec-4da2-a7df-6261c4520bf0)


## Circle Detection
cv2.HoughCircles(image, method, dp, MinDist, param1, param2, minRadius, MaxRadius)

method - Currently only cv2.HOUGH_GRADIENT available
dp - Inverse ratio of accumulator resolution
MinDist - The minimum distance between the center of detected circles
param1 - Gradient value used in the edge detection
param2 - Accumulator threshold for the HOUGH_GRADIENT method (Lower, allows more circles to be detected)
minRadius - Limits the smallest circle to this size
maxRadius - similarly set the limit for the largest circles

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 25)
    circles = np.uint16(np.around(circles))
    
    for i in circles[0,:]:
      cv2.circle(image, (i[0], i[1]), i[2], (0,0,255), 5)
      cv2.circle(image, (i[0], i[1]), 2, (0,0,255), 8)
![image](https://github.com/tan200224/Blog/assets/68765056/dd40be97-114d-4a35-8b5f-1752ebe254ff)
![image](https://github.com/tan200224/Blog/assets/68765056/53c84ef6-e069-4d44-b79a-6ce577a78905)


## Blob Detection
Blob detection is looking for some interesting or unique form of stuff that consists throughout the image.

    cv2.drawKeypoints(input image, keypoints, blank_output_array, color, flags)
    
    detector = cv2.SimpleBlobDetector_create()
    
    # Detect blobs
    keypoints = detector.detect(image)
    
    blank = np.zeros((1,1))
    blob = cv2.drawKeypoints(image, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
![image](https://github.com/tan200224/Blog/assets/68765056/a9b12ed2-f56a-44b0-83d9-8c4a94b1c6b0)
![image](https://github.com/tan200224/Blog/assets/68765056/e23a8a3c-98db-4ef8-9f70-cdbc1d937bf5)





    
