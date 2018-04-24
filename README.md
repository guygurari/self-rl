# Self RL

The goal is to identify the controlled 'character' in an RL setting in an
unsupervised way. The first step is to create a network that tries to predict
the next frame. Then, based on that output find that features that maximize the
mutual information with the actions.

* `my-cart-tf.py`: A simple convnet that tries to predict the next step of a
cart moving on a line.
* `my-cart-keras.py`: Simpler implementation using Keras
