# Comparing Images MSE and Structual Similarity


## Mean Squared Error (MSE)
The MSE between the two images is the sum of the squared difference between the two images. This can easily be implemented with numpy.
The lower the MSE the more similar the images are.


    def mse(image1, image2):
    	# Images must be of the same dimension
    	error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    	error /= float(image1.shape[0] * image1.shape[1])
    
    	return error

### 3 Images of fireworks
1. Fireworks 1
2. Fireworks 1 with brighness enhanced
3. Fireworks 2


        fireworks1 = cv2.imread('images/fireworks.jpeg')
        fireworks2 = cv2.imread('images/fireworks2.jpeg')
        
        M = np.ones(fireworks1.shape, dtype = "uint8") * 100 
        fireworks1b = cv2.add(fireworks1, M)
        
        imshow("fireworks 1", fireworks1)
        imshow("Increasing Brightness", fireworks1b)
        imshow("fireworks 2", fireworks2)

![image](https://github.com/tan200224/Blog/assets/68765056/535db95d-2198-421e-a8c0-464b7a825563)
![image](https://github.com/tan200224/Blog/assets/68765056/993ebdd4-f100-4ebd-b87b-e083c6099253)
![image](https://github.com/tan200224/Blog/assets/68765056/0f850516-34ef-4f65-b3f8-522ecc02b660)


    from skimage.metrics import structural_similarity
    def compare(image1, image2):
      image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
      image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
      print('MSE = {:.2f}'.format(mse(image1, image2)))
      print('SS = {:.2f}'.format(structural_similarity(image1, image2)))


    compare(fireworks1, fireworks1)
MSE = 0.00
SS = 1.00

    compare(fireworks1, fireworks2)
MSE = 2125.41
SS = 0.48

    compare(fireworks1, fireworks1b)
MSE = 8809.38
SS = 0.52

    compare(fireworks2, fireworks1b)
MSE = 13418.54
SS = 0.19
