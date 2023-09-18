# Filtering Colors
1. How to use the HSV Color Space to Filter by Color

![image](https://github.com/tan200224/Blog/assets/68765056/e1f40770-dc4d-4010-9fcc-f2648a23e3d8)
Hue: 0 - 179
Saturation: 0 - 255
Value (Intensity): 0 - 255


    # define range of BLUE color in HSV
    lower = np.array([90,0,0])
    upper = np.array([135,255,255])
    
    # Convert image from RBG/BGR to HSV so we easily filter
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Use inRange to capture only the values between lower & upper 
    mask = cv2.inRange(hsv_img, lower, upper)
    
    # Perform Bitwise AND on mask and our original frame
    res = cv2.bitwise_and(image, image, mask=mask)
    
    imshow('Original', image)  
    imshow('mask', mask)
    imshow('Filtered Color Only', res)
    
![image](https://github.com/tan200224/Blog/assets/68765056/bab02889-a88d-4368-8043-046dc17060a9)
![image](https://github.com/tan200224/Blog/assets/68765056/90ea4a9d-8e68-45b7-95e2-2ed777a4c2b2)
![image](https://github.com/tan200224/Blog/assets/68765056/7de1f16c-ab82-4421-a03d-0f7a7962df4a)


## Background and Foreground Subtraction
1. Background subtraction with Gaussian Mixture-based Background/Foreground Segmentation Algorithm
2. Improved adaptive Gausian mixture model for background subtraction

![image](https://github.com/tan200224/Blog/assets/68765056/d1e0b8ac-be29-4e71-b882-487ee70f5d71)


## What is Background Subtraction?
Background subtraction is a computer vision technique where we seek to isolate the background from the 'moving' foreground. Consider vehicles traversing a roadway, or persons walking along a sidewalk.

It sounds simple in theory (i.e. just keep the station pixels and remove the ones that were changing). However, things like lighting condition changes, shadows etc. can make things get more complicated.

Several algorithms were introduced for this purpose. In the following, we will have a look at two algorithms from the bgsegm module.

    # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
    out = cv2.VideoWriter('walking_output_GM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
    
    # Initlaize background subtractor
    foreground_background = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    # Loop once video is successfully loaded
    while True:
        
        ret, frame = cap.read()
    
        if ret: 
          # Apply background subtractor to get our foreground mask
          foreground_mask = foreground_background.apply(frame)
          out.write(foreground_mask)
          imshow("Foreground Mask", foreground_mask)
        else:
          break
    
    cap.release()
    out.release()

https://github.com/rajeevratan84/ModernComputerVision/raw/main/walking_short_clip.mp4


