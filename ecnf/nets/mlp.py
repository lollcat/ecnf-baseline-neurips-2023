from typing import Sequence, Callable

from flax import linen as nn
import jax

class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = jax.nn.silu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        if self.activate_final:
            x = self.activation(x)
        return x


if __name__ == '__main__':
    import jax.numpy as jnp
    model = MLP([12, 8, 4])
    batch = jnp.ones((32, 10))
    variables = model.init(jax.random.PRNGKey(0), batch)
    output = model.apply(variables, batch)
    print(output)