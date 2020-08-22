# Session 5 notes

## Setup
- set up transforms
- set up dataloaders
- look at the data
    - shapes, min, max, mean, std, var
    - for deployment, calculate mean and std on the entire dataset. Academia calculates it only for train dataset
- set up train and test loops
- write some model
- train it and see

## Skeleton
- clean up the model arch, get basic architecture done (like expand and squeeze arch)
    - example, can add transition blocks, reduce final channel size
    - stop when size is 7x7 (for example)
    - dont add all the changes in this step
        - otherwise you won't know which change made the most difference
- train and test the model again

## Reduce paramter count
- reduce the number of channels in the convolutional layers depending on the dataset
    - eg: from 32-64-128 -> 10-10-20
- train and test the model again
    - any improvement confirms that reducing the number of channels is the reason for improvement
    - improvement is there if the difference between train and test accuracy is less. The lesser, the better
- train and test the model again

## Add batch norm
- batch norm adds parameters. number of trainable parameters = 2*number channels in batch norm
- train and test the model again


## Add dropout
- Add dropout for regularization
- train and test the model again
> We can use the same dropout object multiple times in `forward()`
- dropout reduces train and test accuracy, but it reduces the difference between the two.

## Add Global Average Pooling (GAP)
- Add GAP since final image size is 7x7, to reduce it to 1x1 GAP is used.
- GAP does not decrease accuracy. Accuracy drops because the parameter count drops.

## Increase Capacity 
- if GAP is used, add an extra layer
    - for example if 7x7 is final image size, reduce it to 5x5
        - not a good practice but ok for MNIST
    - add extra layer after GAP
        - can add before since it works for MNIST

## Correct Max pooling location
- correct max pooling location
- can also correct dropout location

## Add image augmentation
- Can add image augmentation such as rotation to train data

## Play with learning rate
- Add lr scheduling

    
