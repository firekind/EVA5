# Assignment 4

## Architecture

- The model follows a squeeze and expand architecture, where the channels follow a 8-16-32-8-16-32 pattern. The final convolutional layer has 10 channels.
- The architecture consists of 7 convolutional layers.
- Max pool is applied after conv layer 2 and conv layer 5. 
- A 1x1 convolutional layer is used to reduce the channel size from 32 to 8. 
- After each layer, batch norm followed by a dropout of 5% is applied. 
- The activation function used is ReLU.

## Parameters
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
         Dropout2d-3            [-1, 8, 26, 26]               0
            Conv2d-4           [-1, 16, 24, 24]           1,168
       BatchNorm2d-5           [-1, 16, 24, 24]              32
         Dropout2d-6           [-1, 16, 24, 24]               0
         MaxPool2d-7           [-1, 16, 12, 12]               0
       BatchNorm2d-8           [-1, 16, 12, 12]              32
         Dropout2d-9           [-1, 16, 12, 12]               0
           Conv2d-10           [-1, 32, 10, 10]           4,640
      BatchNorm2d-11           [-1, 32, 10, 10]              64
        Dropout2d-12           [-1, 32, 10, 10]               0
           Conv2d-13            [-1, 8, 10, 10]             264
      BatchNorm2d-14            [-1, 8, 10, 10]              16
        Dropout2d-15            [-1, 8, 10, 10]               0
           Conv2d-16             [-1, 16, 8, 8]           1,168
      BatchNorm2d-17             [-1, 16, 8, 8]              32
        Dropout2d-18             [-1, 16, 8, 8]               0
        MaxPool2d-19             [-1, 16, 4, 4]               0
      BatchNorm2d-20             [-1, 16, 4, 4]              32
        Dropout2d-21             [-1, 16, 4, 4]               0
           Conv2d-22             [-1, 32, 2, 2]           4,640
           Conv2d-23             [-1, 10, 1, 1]           1,290
================================================================
Total params: 13,474
Trainable params: 13,474
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.51
Params size (MB): 0.05
Estimated Total Size (MB): 0.56
----------------------------------------------------------------
```