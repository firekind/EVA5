# What happens during the training of a DNN

The primary task of a DNN is to map an input *x* to an output *y*. So essentially, a DNN can be thought of as a function *f* where,

<p align="center"><img src="Assets/equation1.png" height=30/></p>

At the heart of any DNN, lies the neuron, which computes the weighted sum of its input and adds a bias term to it, to produce the final output.

<p align="center"><img src="Assets/neuron.png" width=300/></p>

Mathematically,
<p align="center"><img src="Assets/equation2.png" height=70/></p>

where *x<sub>i</sub>* are the inputs, *w<sub>i</sub>* is the weight (the 'importance' of an input, so to speak.) of each input, and *b* is the bias term.

Now, this equation can be rewritten as:
<p align="center"><img src="Assets/equation3.png" height=30/></p>

where, *w* and *x* are vectors containing the weights of the inputs and the inputs respectively.

The values of the weights *w* and bias *b* should be such that the mapping *f* should hold true for input *x*. Finding these values is what training a DNN is about.
