# Confusion Maxtrix

A confusion matrix is a tble t hat is often used to describe the performance of a classification model on a set of test data for which the true values are known.

<img width="773" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/03bb73e0-9ec0-4da7-9bf1-344fccf6f0f5">


### Binary Classification Problems

<img width="742" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/67ef40e4-f155-4a6d-ae79-f1b437578586">

Recall or  true positive rate or sensitivity: True positives/True Positive Labels = 90/100 = 0.9. When it is a Yes, how often do we predict Yes.

False Positive Rate = False positives/True negative Labels = 5/45 = 0.11. When it is actually no, how often does it predict Yes. 

True Negative Rate = True Nagtive/True Negative labels. When it is actually no, how often does it predict No. 

Precision - True Positive / Predicted Yes = 90/95. When it is Yes, how often it is right?

F1 Score - is the harmonic mean of both precision and recall. It is more informative than accuracy alone as it. = (precision x Recall) / (precision + recall)


### Load the model
    !wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/mnist_cnn_net.pth

    
    # Create an instance of the model
    net = Net()
    net.to(device)
    
    # Load weights from the specified path
    net.load_state_dict(torch.load('mnist_cnn_net.pth'))



### Calculate the accuracy

    correct = 0 
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Move our data to GPU
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.3}%')



### Display the misclassified images
1. We can spot what types of images are challenging for our model
2. We can spot any incorrectly labeled images
3. If sometimes we can't correctly identify the class, seeing your CNN struggle hurts less :)


# Set model to evaluation or inference mode
net.eval()

    # We don't need gradients for validation, so wrap in 
    # no_grad to save memory
    with torch.no_grad():
        for data in testloader:
            images, labels = data
    
            # Move our data to GPU
            images = images.to(device)
            labels = labels.to(device)
    
            # Get our outputs
            outputs = net(images)
    
            # use torch.argmax() to get the predictions, argmax is used for long_tensors
            predictions = torch.argmax(outputs, dim=1)
    
            # For test data in each batch we identify when predictions did not match the labe
            # then we print out the actual ground truth 
            for i in range(data[0].shape[0]):
                pred = predictions[i].item()
                label = labels[i]
                if(label != pred):
                    print(f'Actual Label: {label}, Predicted Label: {pred}')       
                    img = np.reshape(images[i].cpu().numpy(),[28,28])
                    imgshow("", np.uint8(img), size = 1)



### Creating Confusion Matrix 

    from sklearn.metrics import confusion_matrix
    
    
    # Initialize blank tensors to store our predictions and labels lists(tensors)
    pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
    label_list = torch.zeros(0, dtype=torch.long, device='cpu')
    
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(testloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
    
            # Append batch prediction results
            pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
            label_list = torch.cat([label_list, classes.view(-1).cpu()])
    
    # Confusion matrix
    conf_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())
    print(conf_mat)


[[ 973    0    2    0    0    1    1    1    2    0]
 [   0 1128    1    1    0    0    2    1    2    0]
 [   2    2 1018    1    1    0    0    4    4    0]
 [   0    0    0 1001    0    5    0    0    4    0]
 [   0    0    2    0  971    0    0    0    0    9]
 [   2    0    0    3    0  884    1    0    0    2]
 [   6    2    1    0    2    5  939    0    3    0]
 [   0    2    7    1    0    0    0 1014    2    2]
 [   5    0    3    0    0    1    1    2  959    3]
 [   2    2    0    3    5    3    0    5    3  986]]



![image](https://github.com/tan200224/Blog/assets/68765056/574e0436-ad8f-4cbe-b0b5-bb104ee582a4)


### Create a more presentable plot

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


![image](https://github.com/tan200224/Blog/assets/68765056/0b044fec-be1f-445c-9dde-fbea142d5c98)


### Classification Report

    from sklearn.metrics import classification_report
    
    print(classification_report(label_list.numpy(), pred_list.numpy()))

              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.99      0.99      1032
           3       0.99      0.99      0.99      1010
           4       0.99      0.99      0.99       982
           5       0.98      0.99      0.99       892
           6       0.99      0.98      0.99       958
           7       0.99      0.99      0.99      1028
           8       0.98      0.98      0.98       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000





