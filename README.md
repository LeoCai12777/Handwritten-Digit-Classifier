# Handwritten-Digit-Classifier
I programmed this classifier using Pytorch to identify handwritten digits from the MNIST dataset.

# Training Loss/Testing Results
```
test set         accuracy
---------------  ----------
training         97.41%
vanilla test     98.33%
vertical flip    98.93%
horizontal flip  98.63%
weak noise       98.46%
medium noise     99.00%
strong noise     89.44%
```
![Bar graph displaying test results.](https://github.com/LeoCai12777/Handwritten-Digit-Classifier/blob/main/testresults.pdf?raw=true)

# Design of the Neural Net
This convolutional neural network is written to follow the VGG11 model (model A in table 1 in [this paper](https://arxiv.org/pdf/1409.1556.pdf)).

The VGG11 model is as follow:
- Conv(001, 064, 3, 1, 1) - BatchNorm(064) - ReLU - MaxPool(2, 2)
- Conv(064, 128, 3, 1, 1) - BatchNorm(128) - ReLU - MaxPool(2, 2)
- Conv(128, 256, 3, 1, 1) - BatchNorm(256) - ReLU
- Conv(256, 256, 3, 1, 1) - BatchNorm(256) - ReLU - MaxPool(2, 2)
- Conv(256, 512, 3, 1, 1) - BatchNorm(512) - ReLU
- Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
- Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU
- Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
- FC(0512, 4096) - ReLU - Dropout(0.5)
- FC(4096, 4096) - ReLU - Dropout(0.5)
- FC(4096, 10)

The 8 convolutions give the classifer the ability to recognize small, medium, and large scale features in the image.
After each convolution, I use batch normalization to standardize the outputs and reduce the effect of exploding gradients.
The ReLU activation function used in each layer adds non-linearity to the model.
The max pooling functions I use increase generalization by ignoring less signification inputs and reduces dimension, thereby accelerating training.
In the fully connected layers, I use a dropout probability of 0.5 to prevent overfitting on the training set.
