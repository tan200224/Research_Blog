# Prespective Transforms
1. Use OpenCV's getPerspective Transform
2. Use findContours to get corners and automate perspective transform

Perspective Transform translates an image's scale into other scales or different aspect ratios. 
    
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imshow('After thresholding', th2)
    
    # Use a copy of your image e.g. edged.copy(), since findContours alters the image
    contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours, note this overwrites the input image (inplace operation)
    # Use '-1' as the 3rd parameter to draw all
    cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
    imshow('Contours overlaid on original image', image)

![image](https://github.com/tan200224/Blog/assets/68765056/1ebd51a8-7a24-42a9-a5ba-d0f774ab687c)
![image](https://github.com/tan200224/Blog/assets/68765056/11aa4dbc-6c46-404c-be17-0d2c5141082c)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # loop over the contours
    for cnt in sorted_contours:
    	# approximate the contour
    	perimeter = cv2.arcLength(cnt, True)
    	approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
    
    	if len(approx) == 4:
    		break
    
    # Our x, y cordinates of the four corners
    print("Our 4 corner points are:")
    print(approx)

Our 4 corner points are:
[[[326  15]]

 [[ 83 617]]

 [[531 779]]

 [[697 211]]]

    # Transform the data type into float first because that is what the transform function takes in
    inputPts = np.float32(approx)
    
    # Order obtained here is top left, bottom left, bottom right, top right
    # The following points are responding to the array position of approx. (the top left is [326 15] etc)
    outputPts = np.float32([[0,0], [0,800], [500,800], [500,0]])
    
    # Get our Transform Matrix, M
    M = cv2.getPerspectiveTransform(inputPts, outputPts)
    
    # Apply the transform Matrix M using warp persepctive
    # The input here for the function is an image, the transform Matrix, and the final size that we want in our image
    dst = cv2.warpPerspective(image, M, (500, 800))
    imshow("Perspective", dst)

![image](https://github.com/tan200224/Blog/assets/68765056/e670529c-861e-4481-ba48-49a538b44313)

