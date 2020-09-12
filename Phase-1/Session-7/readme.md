# Assignment 7

In this assignment, we had to create a model that will be trained on the cifar 10 dataset. The constraints in this assignment were:

1. Atleast one depthwise convolution should be used.
2. Atleast one dilated convolution should be used.
3. Parameter count should be less than one million.
4. Number of epochs can be anything.

## Package used
The athena package was developed and used in this assignment. [Link to the package documentation](https://firekind.github.io/athena).

## Model architecture

The model follows a group convolution architecture, with one branch containing depthwise convolutions and the other containing dilated convolutions. Three such blocks are present in the architecture, separated by a MaxPool and a 1x1 conv. The output block contains a GAP and 2 1x1 convolutions. Dropout layers with a 0.25 dropout value and batch normalization were used in the blocks. [Link to the source code](https://github.com/firekind/athena/blob/master/athena/models/cifar10_v1.py).

The architecture is as follows:

![architecture](res/model_arch.png)

## Receptive field calculations
Here are some of the receptive field calculations:

![depthwise_rf](res/depthwise_rf.png)

![dilation_rf](res/dilation_rf.png)

## Training
The model was trained for 50 epochs, with an SGD optimizer with learning rate 0.008 and momentum 0.95. The highest accuracy was achieved at epoch 48, and it was around 83%. A snippet of the training output is shown below:

```
Epoch: 45 / 50
391/391 [==============================] - 38s 97ms/step - loss: 0.4089 - accuracy: 90.2380
Test set: Average loss: 0.5928, Accuracy: 8171/10000 (81.71%)

Epoch: 46 / 50
391/391 [==============================] - 38s 97ms/step - loss: 0.2184 - accuracy: 90.2660
Test set: Average loss: 0.5807, Accuracy: 8168/10000 (81.68%)

Epoch: 47 / 50
391/391 [==============================] - 38s 97ms/step - loss: 0.2928 - accuracy: 90.4080
Test set: Average loss: 0.6067, Accuracy: 8072/10000 (80.72%)

Epoch: 48 / 50
391/391 [==============================] - 38s 97ms/step - loss: 0.4077 - accuracy: 90.4120
Test set: Average loss: 0.5109, Accuracy: 8363/10000 (83.63%)

Epoch: 49 / 50
391/391 [==============================] - 38s 97ms/step - loss: 0.2714 - accuracy: 90.7680
Test set: Average loss: 0.6770, Accuracy: 7949/10000 (79.49%)

Epoch: 50 / 50
391/391 [==============================] - 38s 98ms/step - loss: 0.2823 - accuracy: 91.0240
Test set: Average loss: 0.6403, Accuracy: 8039/10000 (80.39%)
```

## Lessons learned:
1. Group convolutions are useful. In one way, they let the model learn and choose / combine between different branches that act on the data differently.
2. Model overfits too much. Need to look into ways to reduce the overfitting.
3. Writing your own package helps with experiment with different types of models quickly. Its a nice thing to have in your toolkit.
4. Depthwise convolution helps in reducing the parameter count by a huge margin.
5. Incorrect use of Dilated convolution can negatively impact accuracy of the model.