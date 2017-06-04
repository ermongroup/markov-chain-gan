import tensorflow as tf

from abc import abstractmethod


class SamplerBase(object):
    @abstractmethod
    def __call__(self, batch_size):
        return None

    @property
    def xlim(self):
        return None

    @property
    def ylim(self):
        return None


class DiscriminatorBase(object):
    def __init__(self, input_size, output_size, name):
        self.input_size_ = input_size
        self.output_size_ = output_size
        self.name_ = name

    @property
    def input_size(self):
        return self.input_size_

    @property
    def output_size(self):
        return self.output_size_

    @property
    def name(self):
        return self.name_

    @abstractmethod
    def __call__(self, input, reuse=True):
        return None

    @abstractmethod
    def loss(self, prediction, target):
        return None


class TransitionBase(object):
    def __init__(self, input_size, name):
        self.input_size_ = input_size
        self.state_size_ = input_size
        self.name_ = name

    @property
    def input_size(self):
        return self.input_size_

    @property
    def state_size(self):
        return self.state_size_

    @property
    def name(self):
        return self.name_

    @abstractmethod
    def __call__(self, states):
        return None


class TransitionCell(tf.contrib.rnn.RNNCell):
    def __init__(self, transition_fn):
        self.input_size = transition_fn.input_size
        self.name = transition_fn.name
        self.transition = transition_fn

    @property
    def state_size(self):
        return self.input_size

    @property
    def output_size(self):
        return self.state_size

    def __call__(self, inputs, states, scope=None):
        with tf.variable_scope(self.name) as vs:
            state = self.transition(states)
            return state, state