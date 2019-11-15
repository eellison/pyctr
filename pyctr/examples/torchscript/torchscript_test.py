import numpy as np
import torch
from absl.testing import absltest as test
from absl.testing import parameterized

from pyctr.examples.torchscript.torchscript import specialize


def random_inputs_numpy(batch_size, max_seq_len, input_size, hidden_size):
    inputs = np.random.normal(size=(batch_size, 2 * max_seq_len, input_size))
    seq_len = np.arange(max_seq_len)  # Not random for equal computation.
    w = np.random.normal(size=(input_size + hidden_size, hidden_size))
    b = np.random.normal(size=(hidden_size,))
    init_state = np.zeros((batch_size, hidden_size))
    return inputs, seq_len, w, b, init_state


def random_inputs_torch(batch_size, max_seq_len, input_size, hidden_size):
    np_inputs = random_inputs_numpy(batch_size, max_seq_len, input_size, hidden_size)
    return tuple(torch.tensor(a) for a in np_inputs)


class TorchscriptTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 3, 1]),
            torch.tensor([1]),
            torch.tensor([10]),
            torch.tensor([10, 11, 19]),
            True,
        ),
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 3, 1]),
            torch.tensor([[1]]),
            torch.tensor([[10]]),
            torch.tensor([10, 11, 19]),
            False,
        ),
    )
    def test_if_basic(self, w_1, w_2, b_1, b_2, x, cond: bool):
        def pytorch_test_fn(w_1, w_2, b_1, b_2, x, cond: bool):
            return w_1 @ x + b_1 if cond else w_2 @ x + b_2

        def if_basic_fn(w_1, w_2, b_1, b_2, x, cond: bool):
            if cond:
                return (w_1 @ x) + b_1
            else:
                return (w_2 @ x) + b_2

        fn, ast = specialize(if_basic_fn, w_1, w_2, b_1, b_2, x, cond)

        fn_result = fn(w_1, w_2, b_1, b_2, x, cond)
        pytorch_result = pytorch_test_fn(w_1, w_2, b_1, b_2, x, cond)

        self.assertEqual(fn_result, pytorch_result)

    @parameterized.parameters((torch.tensor(5), torch.tensor(5)))
    def test_statements(self, w, x):
        def statements_basic(w, x):
            t = torch.mm(w, x)
            ret = torch.tanh(t)
            return ret

        fn, ast = specialize(statements_basic, w, x)

    @parameterized.parameters(random_inputs_torch(10, 10, 50, 256))
    def test_rnn(self, ins, seq_len, w, b, init_state):
        def rnn_basic(ins, seq_len, w, b, init_state):
            inputs_time_major = torch.transpose(ins, 1, 0)
            state = init_state
            for i in range(int(torch.max(seq_len))):
                x = inputs_time_major[i]
                h = torch.cat((x, state), 1)
                state = torch.tanh(torch.add(torch.mm(h, w), b))  # Basic RNN cell
            return state

        fn, ast = specialize(rnn_basic, ins, seq_len, w, b, init_state)

        fn_result = fn(ins, seq_len, w, b, init_state)
        pytorch_result = rnn_basic(ins, seq_len, w, b, init_state)
        self.assertTrue(bool((fn_result == pytorch_result).all()))


if __name__ == "__main__":
    test.main()
