from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework.ops import convert_to_tensor


class ESN(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, wr2_scale=0.7, connectivity=0.1, leaky=1.0,
                 activation=math_ops.tanh,
                 win_init=init_ops.random_normal_initializer(),
                 wr_init=init_ops.random_normal_initializer(),
                 bias_init=init_ops.random_normal_initializer()
                 ):
        self._num_units = num_units
        self._leaky = leaky  # 漏れ率
        self._activation = activation

        def _wr_initializer(shape, dtype, partition_info=None):
            wr = wr_init(shape, dtype=dtype)

            connectivity_mask = math_ops.cast(
                math_ops.less_equal(
                    random_ops.random_uniform(shape),
                    connectivity),
                dtype
            )

            #  Echo state propertyを満たすために、リザーバー層の重みを調整する。
            #  ここでは類似計算を行う。
            #  wr2_scaleは希望のスペクトル半径になる。
            wr = math_ops.multiply(wr, connectivity_mask)
            wr_norm2 = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(wr)))
            is_norm_0 = math_ops.cast(math_ops.equal(wr_norm2, 0), dtype)
            wr = wr * wr2_scale / (wr_norm2 + 1 * is_norm_0)

            return wr

        self._win_initializer = win_init  # input weight
        self._bias_initializer = bias_init
        self._wr_initializer = _wr_initializer  # リザーバー層の重み

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        Params:
            inputs: 2-D Tensor, [batch_size x input_size]
            state: リザーバー層内の各ノードの状態を持つ。2-D Tensor, [batch_size x self.state_size]
            scope: VariableScope; defaults to calss ESN
        Returns:
            tuple (output, new_state)
            output = new_stat = (1-leaky)*state + leaky * activation(W_in*input + W_r * state + B)
        """
        inputs = convert_to_tensor(inputs)
        input_size = inputs.get_shape().as_list()[1]  # 系列長？
        dtype = inputs.dtype

        with vs.variable_scope(scope or type(self).__name__):
            win = vs.get_variable('InputMatrix', [input_size, self._num_units], dtype=dtype,
                                  trainable=False, initializer=self._win_initializer)

            wr = vs.get_variable('ReservoirMatrix', [self._num_units, self._num_units], dtype=dtype,
                                 trainable=False, initializer=self._wr_initializer)

            b = vs.get_variable('Bias', [self._num_units], dtype=dtype,
                                trainable=False, initializer=self._bias_initializer)

            in_mat = array_ops.concat([inputs, state], axis=1)
            weights_mat = array_ops.concat([win, wr], axis=0)

            output = (1 - self._leaky) * state + self._leaky * self._activation(math_ops.matmul(in_mat, weights_mat) + b)

            return output, output
