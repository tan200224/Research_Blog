# Moments, Sorting, Approximating and Matching Contours

## Sorting by Area using cv2.ContourArea and cv2.Moments

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 200)
    
    # We use RETR_EXTERNAL, which means we don't want any hierarchy
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
![image](https://github.com/tan200224/Blog/assets/68765056/b9288fed-a234-4e82-9895-2e4aa6c95902)
![image](https://github.com/tan200224/Blog/assets/68765056/9cf91617-4210-4ec6-b19f-e475ccc01212)
![image](https://github.com/tan200224/Blog/assets/68765056/796493f7-839c-4660-9b8e-571da1ef73c7)


 
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Use moments to calculate the center of each shape, and put text on it
    for (i, c) in enumerate(sorted_contours):
      # moments is the center point of the contours
      M = cv2.moments(c)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      # Those are all in place function
      cv2.putText(image, str(i+1), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
      cv2.drawContours(image, [c], -1, (255,0,0), 3)
![image](https://github.com/tan200224/Blog/assets/68765056/3e5dcfc4-cbca-4408-b5ec-5732757eee5e)
![image](https://github.com/tan200224/Blog/assets/68765056/456390fa-fab1-4f9d-b51b-cb46bb547a6c)


## Approximating Contours using ApproxPolyDP
It approximates a contour shape to another shape with less number of vertices depending upon the precision we specify
cv2.approxPolyDP(contour, Approximation Accuracy, Closed)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
      x,y,w,h = cv2.boundingRect(c)
      cv2.rectangle(orig_image, (x,y), (x+w, y+h), (0,0,255), 2)
      cv2.drawContours(image, [c], 0, (0,255,0), 2)
    
    for c in contours:
      # Calculate accuracy as a percent of the contour perimeter
      accuracy = 0.03 * cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, accuracy, True)
      cv2.drawContours(copy, [approx], 0, (0, 255, 0), 2)
![image](https://github.com/tan200224/Blog/assets/68765056/47648c75-0f9d-4a96-8cc5-2938b02507fe)
![image](https://github.com/tan200224/Blog/assets/68765056/cb1bccd2-2839-44d7-bf30-080811a168ec)
![image](https://github.com/tan200224/Blog/assets/68765056/02e5d1a8-dfcb-4c10-9c59-473593272608)
![image](https://github.com/tan200224/Blog/assets/68765056/cf1b252a-25f0-4687-8ce3-1c54718223a4)


## Convex Hull
Convex Hull will look similar to contour approximation, but it is not (Both may provide the same results in some cases).

The cv2.convexHull() Function checks and corrects a curve for convexity defects. Generally speaking, convex curves are always bulged out, or at least flat. If it is bulged inside, it is called a convexity defect

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    ret, thresh = cv2.threshold(gray, 176, 255, 0)
    
    # Find Contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, [c], -1, (0,255,0), 2)
    
    # Sort contours by Area and Then remove the largest frame contour
    n = len(contours) -1
    contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]
    
    # Iterate through contours and draw the convex hull
    for c in contours:
      hull = cv2.convexHull(c)
      cv2.drawContours(orig_image, [hull], 0, (0,255,0), 2)
![image](https://github.com/tan200224/Blog/assets/68765056/b5bf16d9-3025-4766-b48c-d5fcd9cdf643)
![image](https://github.com/tan200224/Blog/assets/68765056/700b6c3a-1afb-43d4-a3ce-b96de5ddd8c0)


## Matching Contours
cv2.matchShape(contour template, contour, method, method parameter)
output match value(the lower the value, the closer we get)

**Contour template** - Reference the contour that we're trying to find in the new image

**Contour** -  The individual contour we are checking against

**Method** - Type of contour matching(1,2,3)

**Method Parameter** - Leave alone as 0.0 (not fully utilized in Python OpenCV)

    template = cv2.imread('images/4star.jpg', 0)
    imshow('Template', template)
    
    # Load the target image with the shape that we are trying to match
    target = cv2.imread('images/shapestomatch.jpg')
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the template
    ret, thresh1 = cv2.threshold(template, 127, 255, 0)
    ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)
    
    # Find the contours in the template
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    # We need to sort the contours by area so that we can remove the largest
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # We extract the second largest contour, which will be  our template contour
    template_contour = contours[1]
    
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    for c in contours:
      # Iterate through each contour in the target image and
      # Use cv2.matchShape to compare contour shapes
      match = cv2.matchShapes(template_contour, c, 3, 0.0)
      print(match)
    
      if match < 0.15:
        closest_contour = c
      else:
        closest_contour = []
    
    cv2.drawContours(target, [closest_contour], -1, (0,255,0), 3)
    imshow('Output', target)
![image](https://github.com/tan200224/Blog/assets/68765056/eca5ea86-24aa-4e00-ba8b-0b253ad2669a)
![image](https://github.com/tan200224/Blog/assets/68765056/b6ab091f-8aad-4398-868e-581e89ab4d8a)





