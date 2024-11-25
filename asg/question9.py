from _context import vugrad

import numpy as np

from experiments.train_mlp import MLP
import vugrad as vg

from vugrad import Op


class ReLU(Op):
    """
    Op for element-wise application of ReLU function.
    """

    @staticmethod
    def forward(context, input):
        relu_x = np.maximum(0, input)
        context['input'] = input  # Store input for backward computation
        return relu_x

    @staticmethod
    def backward(context, go):
        input = context['input']
        grad = (input > 0).astype(float)
        return go * grad


class MLP_ReLU(MLP):

    def forward(self, input):
        assert len(input.size()) == 2

        # first layer
        hidden = self.layer1(input)

        # non-linearity
        hidden = ReLU.do_forward(hidden)
        # -- We've called a utility function here, to mimin how this is usually done in pytorch. We could also do:
        #    hidden = Sigmoid.do_forward(hidden)

        # second layer
        output = self.layer2(hidden)

        # softmax activation
        output = vg.logsoftmax(output)
        # -- the logsoftmax computes the _logarithm_ of the probabilities produced by softmax. This makes the computation
        #    of the CE loss more stable when the probabilities get close to 0 (remember that the CE loss is the logarithm
        #    of these probabilities). It needs to be implemented in a specific way. See the source for details.

        return output
