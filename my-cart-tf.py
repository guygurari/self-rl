#!/usr/bin/env python3

import argparse
import collections
import numpy as np
import tensorflow as tf
from carts import *

def cart_prediction_model_fn(features, labels, mode, params):
    """Create a model that predicts the next cart state, given its current
state and an action. If labels not specified, only the outputs and
predictions are returned."""
    # We use sigmoid activation because each 'pixel' is either 1 (for cart) or
    # 0 (for empty).
    outputs = tf.layers.conv1d(
        inputs=features['x'],
        filters=1,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.sigmoid,
        use_bias=True,
        name='conv1d')

    predictions = tf.round(outputs, name='predictions')
    pred_dict = {'outputs': outputs, 'predictions': predictions}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_dict)

    # loss = tf.losses.mean_squared_error(labels, outputs)
    loss = tf.reduce_mean(tf.square(labels - outputs), name='loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # equal_pixels has the same shape as predictions and labels.
    # It is 1 for each pixel that agrees, 0 for those that don't.
    equal_pixels = tf.cast(tf.equal(predictions, labels), tf.float32)

    # Take the minimum in each sample. If one pixel disagrees the sample
    # will be 0. If all pixels agree it will be 1. So accuracy is based
    # on exactly predicting the next frame.
    equal_samples = tf.reduce_min(equal_pixels, axis=1)

    metrics = {'soft_accuracy': tf.metrics.mean(equal_pixels,
                                                name='soft_accuracy_metric'),
               'hard_accuracy': tf.metrics.mean(equal_samples,
                                                name='hard_accuracy_metric')}

    pred_dict['soft_accuracy'] = tf.reduce_mean(
        equal_pixels, name='soft_accuracy')
    pred_dict['hard_accuracy'] = tf.reduce_mean(
        equal_samples, name='hard_accuracy')

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

def create_cart_prediction_input_fn(x, y, epochs, batch_size=128):
    """Cart state predictor input pipeline."""
    def input_fn():
        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(len(x))
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)

        # One-shot means it goes once through the data, but
        # the dataset will repeat itself 'epochs' times
        iterator = dataset.make_one_shot_iterator()

        # get_next() returns Tensors that get filled with batches
        features, labels = iterator.get_next()

        feature_cols = {'x': features}
        return feature_cols, labels

    return input_fn

# def create_cart_mask_model(state, predicted_state=None, margin=None):
#     """Create a model that finds a mask for which the prediction model works
#     best. inputs = states. predictions = predicted state."""
#     soft_mask = tf.layers.conv1d(
#         inputs=inputs,
#         filters=1,
#         kernel_size=3,
#         strides=1,
#         padding='same',
#         activation=tf.sigmoid,
#         use_bias=True, name='conv1d')

#     mask = tf.round(soft_mask)

#     if predicted_state is None:
#         return soft_mask, mask
    
#     assert margin is not None
    
#     # loss = mask * relu(|s-s'| - margin) + (1-mask) * relu(margin - |s-s'|)
#     mar = tf.constant(margin, dtype=tf.float32, name='margin')
#     s = state
#     sp = predicted_state
#     relu = tf.nn.relu
#     loss_terms = soft_mask * relu(tf.abs(s-sp) - mar)
#     loss_terms += (1. - soft_mask) * relu(mar - tf.abs(s-sp))
#     loss = tf.reduce_mean(loss_terms)
#     return soft_mask, mask, loss

def predict_next_state(cart, model):
    x = np.reshape(cart.state(), (1, cart.L, 1)).astype(np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x}, shuffle=False)
    y = model.predict(input_fn=predict_input_fn)
    pred = list(y)[0]['predictions']
    return pred.reshape((cart.L))

def train_cart_predictor(cart, n_train, train_epochs):
    tf.logging.info('')
    tf.logging.info('=========== Building Models =========')
    models = {}

    params = {'learning_rate': 1e-4}
    train_batch_size = 64
    n_val = 50000

    for a in action_space:
        models[a] = tf.estimator.Estimator(
            model_fn=cart_prediction_model_fn,
            params=params,
            model_dir='models/cart-predictor/%s' % action_names[a])

    if train_epochs > 0:
        tf.logging.info('')
        tf.logging.info('=========== Training =========')

        (train_x, train_y, train_actions) = sample_cart_states(cart, n_train)

        for a in action_space:
            tf.logging.info('')
            tf.logging.info('Action: %s' % action_names[a])

            # Get the training samples for action a
            (x, y) = get_states_for_action(train_x, train_y, train_actions, a)
            tf.logging.info('Have %d examples' % len(x))

            input_fn = create_cart_prediction_input_fn(
                x, y, train_epochs, train_batch_size)

            logging_hook = tf.train.LoggingTensorHook(
                tensors={'loss': 'loss',
                         'soft_acc': 'soft_accuracy',
                         'hard_acc': 'hard_accuracy'},
                every_n_secs=1)

            models[a].train(input_fn=input_fn, steps=None,
                            hooks=[logging_hook])

    tf.logging.info('\n=========== Validation =========')
    (val_x, val_y, val_actions) = sample_cart_states(cart, n_val)

    for a in action_space:
        (x, y) = get_states_for_action(val_x, val_y, val_actions, a)
        val_input_fn = create_cart_prediction_input_fn(x, y, epochs=1)
        ev = models[a].evaluate(input_fn=val_input_fn)
        tf.logging.info('Evaluation: %s' % ev)

    tf.logging.info('')
    tf.logging.info('=========== Prediction =========')
    tf.logging.info('Current state:\t\t%s' % str(cart))

    for a in action_space:
        pred = predict_next_state(cart, models[a])
        tf.logging.info('Prediction after %s:\t%s'
                        % (action_names[a], state_string(pred)))

def main():
    parser = argparse.ArgumentParser(
        description='Controlled Cart Simulator [TM]')
    parser.add_argument('--L', type=int, default=10,
                        help='Length')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs')
    parser.add_argument('--N', type=int, default=50000,
                        help='Training samples')
    # parser.add_argument('-f', '--foo', action='store_true', help='Do foo')
    # parser.add_argument('rest', nargs='*')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    cart = ControlledCart(L=args.L)
    train_cart_predictor(cart, args.N, args.epochs)

if __name__ == '__main__':
    main()
