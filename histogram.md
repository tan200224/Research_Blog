# Histograms and K-Means Clustering Dominant Colors

    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()
    
    # Viewing Separate Color Channels
    color = ('b', 'g', 'r')
    
    # We now separate the colors and plot each in the Histogram
    for i, col in enumerate(color):
        histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram2, color = col)
        plt.xlim([0,256])
    
    plt.show()
![image](https://github.com/tan200224/Blog/assets/68765056/0497fc16-6d96-4c5e-a36e-a4ff21c69aac)
![image](https://github.com/tan200224/Blog/assets/68765056/e3ad7cf4-7264-40b7-b5e1-870dc13b5775)

This graph shows the brightness of the image. It shows the value of the left side of the image is greater than the right side, which means the left side of the image is brighter than the right side of the image.


![image](https://github.com/tan200224/Blog/assets/68765056/fbf6cf3c-79eb-49ae-bdd5-bad686e28fc6)

This image shows that the image is warm becuase it contains more red. 

### cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".

channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.

mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)

histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].

ranges : this is our RANGE. Normally, it is [0,256].

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # We plot a histogram, ravel() flatens our image array
    plt.hist(image.ravel(), 256, [0, 256]); plt.show()
    
    # Viewing Separate Color Channels
    color = ('b', 'g', 'r')
    
    # We now separate the colors and plot each in the Histogram
    for i, col in enumerate(color):
        histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram2, color = col)
        plt.xlim([0,256])
![image](https://github.com/tan200224/Blog/assets/68765056/af75559b-c9f3-42f6-b138-cbab565dfb5d)
![image](https://github.com/tan200224/Blog/assets/68765056/1cad1b9b-124a-4857-95ec-3b6984f1bb48)
![image](https://github.com/tan200224/Blog/assets/68765056/fcf78d90-d9c3-4462-9fd8-e94944027122)


## K-Means Clustering to obtain the dominant colors in an image
    def centroidHistogram(clt):
        # Create a histrogram for the clusters based on the pixels in each cluster
        # Get the labels for each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    
        # Create our histogram
        (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    
        # normalize the histogram, so that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
    
        return hist
    
    def plotColors(hist, centroids):
        # Create our blank barchart
        bar = np.zeros((100, 500, 3), dtype = "uint8")
    
        x_start = 0
        # iterate over the percentage and dominant color of each cluster
        for (percent, color) in zip(hist, centroids):
          # plot the relative percentage of each cluster
          end = x_start + (percent * 500)
          cv2.rectangle(bar, (int(x_start), 0), (int(end), 100),
            color.astype("uint8").tolist(), -1)
          x_start = end
        return bar

    from sklearn.cluster import KMeans
    
    image = cv2.imread('images/tobago.jpg')
    imshow("Input", image)
    
    # We reshape our image into a list of RGB pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    print(image.shape)
    
    number_of_clusters = 5
    clt = KMeans(number_of_clusters)
    clt.fit(image)
    
    hist = centroidHistogram(clt)
    bar = plotColors(hist, clt.cluster_centers_)
    
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

![image](https://github.com/tan200224/Blog/assets/68765056/77abafa5-9dee-47b1-af57-26e1a5b45f8e)
![image](https://github.com/tan200224/Blog/assets/68765056/8d5742f1-df13-4794-98e2-4384226acdde)
