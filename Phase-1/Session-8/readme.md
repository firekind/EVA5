# Assignment 8

In this assignment we had to train a ResNet-18 model upto at least 85% accuracy. No contraints were given, just that we had to use the package developed in the previous session to train the model.

## Package used
The athena package was improved on (a lot) and used in this assignment. [Link to the package documentation](https://firekind.github.io/athena).

## Training
For training,
1. Cross Entropy loss function was used
2. SGD optimizer with initial learning rate of `0.1`, momentum `0.9` and weight decay `5e-4` was used
3. `StepLR` as a learning rate scheduler, with step size as `100` and gamma as `0.1`
4. number of epochs trained for was `115` (found through trial and error)
5. No data augmentation, (since the next session is about data augmentation 😛), but the images were normalized with mean (0.4914, 0.4822, 0.4465) and std (0.2023, 0.1994, 0.2010)

Here's a snippet of the training logs (taken from [assignment.ipynb](./assignment.ipynb)). The max accuracy that was achieved was 88.53%:

```
Epoch: 56 / 115
391/391 [==============================] - 63s 161ms/step - loss: 0.1707 - accuracy: 94.2440
Test set: Average loss: 0.7512, Accuracy: 7741/10000 (77.41%)

Epoch: 57 / 115
391/391 [==============================] - 63s 161ms/step - loss: 0.1559 - accuracy: 94.7080
Test set: Average loss: 0.6444, Accuracy: 8190/10000 (81.90%)

Epoch: 58 / 115
391/391 [==============================] - 63s 161ms/step - loss: 0.1666 - accuracy: 94.3280
Test set: Average loss: 0.7000, Accuracy: 7934/10000 (79.34%)

Epoch: 59 / 115
391/391 [==============================] - 63s 161ms/step - loss: 0.1601 - accuracy: 94.5340
Test set: Average loss: 0.6810, Accuracy: 8014/10000 (80.14%)

Epoch: 60 / 115
391/391 [==============================] - 63s 161ms/step - loss: 0.1543 - accuracy: 94.7360

.
.
.

Epoch: 109 / 115
391/391 [==============================] - 63s 160ms/step - loss: 0.0027 - accuracy: 100.0000
Test set: Average loss: 0.3890, Accuracy: 8853/10000 (88.53%)
```

and the jupyter notebook: [assignment.ipynb](./assignment.ipynb)

## Loss and accuracy curves

Here are the loss and accuracy curves:

![loss and accuracy curves](./logs/ResNet-18/ResNet-18%20with%20Cross%20Entropy%20Loss/images/loss_acc_plot.png)


## Misclassified images

And here are some images that the model misclassified

![misclassified](./logs/ResNet-18/ResNet-18%20with%20Cross%20Entropy%20Loss/images/misclassified_plot.png)

## Lessons learned

1. **HAVE A CHECKPOINT MANAGEMENT SYSTEM**. This is vital when you have long training times and when you don't want to go bald.

2. Accuracy suddenly rises and loss suddenly drops at epoch 100. This is most likely due to the change in learning rate (due to `StepLR`). 

3. The step size of `StepLR` could have been much lesser (around epoch 20 maybe?)
if not step size, the first drop in learning rate could have happened much earlier.

4. weight decay in SGD helps in controlling overfitting to some degree

5. overfitting is turning out to be a problem. I guess image augmentation to the rescue!

6. Don't delete old train logs that failed. Those can prove to be useful for future reference.

## Appendix

Tensorboard was used to log the losses, accuracies and the model itself. The tensorboard event file is present in [./logs/ResNet-18/ResNet-18 with Cross Entropy Loss](./logs/ResNet-18/ResNet-18%20with%20Cross%20Entropy%20Loss). So an interactive plot and model of this assignment can be viewed using tensorboard.
