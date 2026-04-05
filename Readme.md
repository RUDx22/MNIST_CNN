\#MNIST DIGIT RECOGNIZATION (for learning purpose):



DATASET=from MNIST Digit dataset

**date:4/4/26**

**Stage\_1:**

&#x09;->data loading

&#x09;->normalizing data , converting image to tensors.

&#x09;->made two different dataset one for training and testing

&#x09;->made loaddataset with each batch = 64

&#x09;#class,functions

&#x09;->class named digit,contains 2 functions 1)init 2)forward

&#x09;->contains 2convolution layer for finding features,one pooling for get same output image size as input image and shrink image.

&#x09;->used dropout for regularization to avoid overfitting

&#x09;->used relu for ignoring negative values.

&#x09;->fc1 and fc2 are flatten to convert 2d feature maps to 1d vector for decision making

&#x09;#loss and optimization

&#x09;->crossentropyloss is used as it is classification it calculate confidence of outputs

&#x09;->adam optimizer used to optimize as it also remembers where it was last time in gradient functions

&#x09;

&#x09;#training 

&#x09;we only run it 1 time as dataset with train and test data is  enough to get above 90% accuracy

&#x09;#evaluate

&#x09;tested model is accurate by 1st unseen image from testdata.





**Date:4/4/26**

**Stage\_2:**

realized model runs on cpu only even though it has decent gpu.

checked for gpu availability on python library

&#x09;->attached to(device) in model,labels and inputs in training phase





GOT ERROR IN TESTING PHASE :

&#x20;  no tensor on same device founded tensor on both cpu and cuda 

&#x20;  modified testing phase:

&#x09;		made new variable called testimage: attached .to(device) so labels and images are on same device

&#x20;                       made output variable for output=model(inputimg)









**IMPROVEMENT:**

&#x20; model train and testing time reduced compared to stage 1.

