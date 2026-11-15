import jax
import equinox as eqx


class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, data_dim, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(data_dim, 64, key=key1),
            eqx.nn.Linear(64, 32, key=key2),
            eqx.nn.Linear(32, data_dim, key=key3),
        ]

    def __call__(self, x):
        x = jax.nn.silu(self.layers[0](x))  # Swish = SiLU
        x = jax.nn.silu(self.layers[1](x))
        x = self.layers[2](x)
        return x
