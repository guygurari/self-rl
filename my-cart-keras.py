#!/usr/bin/env python3

import numpy as np
import keras as k

action_names = {
    -1 : 'LEFT',
    0 : 'NONE',
    1 : 'RIGHT',
}
action_space = list(action_names.keys())

def make_cart_state(L, pos):
    state = np.full(L, fill_value=0)
    state[pos] = 1
    return state

def move_cart(L, pos, action):
    """Return the new cart position. action=-1,0,1 for left/nothing/right."""
    pos += action

    # Boundary conditions
    if pos < 0:
        pos = 0
    if pos >= L:
        pos = L - 1

    return pos

def sample_random_action():
    return np.random.choice(action_space)

def state_string(state):
    """A string representation of the cart's state."""
    return '[%s]' % ''.join([('.' if x == 0 else '@') for x in state])

class Cart:
    def __init__(self, L):
        self.L = L
        self.pos = np.random.randint(L)

    def __str__(self):
        return state_string(self.state())

    def state(self):
        return make_cart_state(self.L, self.pos)

class ConstantCart(Cart):
    """A cart with constant position."""
    def __init__(self, L):
        super().__init__(L)

    def next_state(self, action):
        return self.state()

class ControlledCart(Cart):
    """A cart that is controlled by a left/right action."""
    def __init__(self, L):
        super().__init__(L)

    def next_state(self, action):
        self.pos = move_cart(self.L, self.pos, action)
        return self.state()
        
class RandomCart(Cart):
    """A cart that moves randomly regardless of action."""
    def __init__(self, L):
        super().__init__(L)

    def next_state(self, action):
        random_action = sample_random_action()
        self.pos = move_cart(self.L, self.pos, random_action)
        return self.state()

def sample_cart_states(cart, n):
    actions = np.array([sample_random_action() for _ in range(n)])

    # Shape is (samples, width, channels)
    states = np.ndarray((n+1, cart.L, 1))

    # reshape() adds the 'channels' dimension
    states[0, :] = cart.state().reshape((-1, 1))

    for i in range(1, n+1):
        states[i, :] = cart.next_state(actions[i-1]).reshape((-1, 1))

    x = states[0:-1, :]
    y = states[1:, :]
    return (x, y, actions)

def get_states_for_action(all_x, all_y, all_actions, a):
    """Get just the (x,y) whose action is a."""
    actions_mask = (all_actions == a)
    num_actions = np.count_nonzero(actions_mask)
    assert num_actions > 0
    x = all_x[actions_mask]
    y = all_y[actions_mask]
    return (x, y)

# def cart_state_to_model_input(cart):
#     return np.reshape(cart.state(), (1, cart.L, 1))

def predict_next_state(cart, model):
    x = np.reshape(cart.state(), (1, cart.L, 1))
    y = model.predict(x).round()
    return y.reshape((cart.L))

def predict_next_state_and_print(cart, model):
    next_state = predict_next_state(cart, model)
    print(state_string(next_state))

cart = ControlledCart(L=10)

# for _ in range(20):
#     print(cart)
#     cart.next_state(0)

n_train = 50000
n_val = 10000
train_epochs = 100
train_batch_size = 64

(train_x, train_y, train_actions) = sample_cart_states(cart, n_train)
(val_x, val_y, val_actions) = sample_cart_states(cart, n_val)

# Indexed by actions
models = {}

print('=========== Training =========')
for a in action_space:
    print('\nAction: %s' % action_names[a])

    # Create the model. We use sigmoid because each 'pixel' is either 1 (for
    # cart) or 0 (for empty).
    model = k.models.Sequential()
    model.add(k.layers.Conv1D(input_shape=(cart.L, 1),
                              filters=1,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Get the training samples for action a
    (x, y) = get_states_for_action(train_x, train_y, train_actions, a)

    model.fit(x, y, epochs=train_epochs, batch_size=train_batch_size)
    models[a] = model

print('=========== Validation =========')
for a in action_space:
    print('\nAction: %s' % action_names[a])
    (x, y) = get_states_for_action(val_x, val_y, val_actions, a)
    loss_and_metrics = models[a].evaluate(x, y, batch_size=128)
    print('Accuracy: %f' % loss_and_metrics[1])

print('=========== Prediction =========')
print('Current state:\t\t', cart)
print('Prediction after LEFT:\t', state_string(predict_next_state(cart, models[-1])))
print('Prediction after NONE:\t', state_string(predict_next_state(cart, models[0])))
print('Prediction after RIGHT:\t', state_string(predict_next_state(cart, models[1])))
