from typing import Sequence, Callable, Optional

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


class NonLinearLayerWithResidualAndLayerNorm(nn.Module):
    output_size: int
    activation_fn: Callable = jax.nn.silu

    @nn.compact
    def __call__(self, x):
        out = self.activation_fn(nn.Dense(self.output_size)(nn.LayerNorm()(x)))
        return out + x


class StableMLP(nn.Module):
    """MLP with layer norm and residual connections."""
    mlp_units: Sequence[int]
    activate_final: bool = False
    zero_init_output: bool = False
    output_variance_scaling: Optional[float] = False
    stable_layer: bool = True
    activation: Callable = jax.nn.silu
    name: Optional[str] = None

    def setup(self) -> None:
        if not self.activate_final:
            assert len(self.mlp_units) > 1, "MLP is single linear layer with no non-linearity"
            n_output_params = self.mlp_units[-1]
            mlp_units = self.mlp_units[:-1]
        for i in range(len(self.mlp_units) - 1):  # Make sure mlp_units have constant width.
            assert self.mlp_units[i] == self.mlp_units[i+1]
        if self.stable_layer:
            layers = [nn.Dense(self.mlp_units[0]), self.activation]
            layers.extend([NonLinearLayerWithResidualAndLayerNorm(layer_width, activation_fn=self.activation)
                           for layer_width in self.mlp_units[1:]])
            self.mlp_function = nn.Sequential(layers)
        else:
            self.mlp_function = MLP(self.mlp_units, activate_final=True, activation=self.activation)

        if self.zero_init_output or self.output_variance_scaling:
            assert self.activate_final is False
        if not self.activate_final:
            self.final_layer = \
                nn.Dense(self.mlp_units[-1], kernel_init=nn.initializers.zeros_init()) if self.zero_init_output else \
                nn.Dense(self.mlp_units[-1],
                         kernel_init=nn.initializers.variance_scaling(
                             self.output_variance_scaling, "fan_avg", "uniform")
                          if self.output_variance_scaling else nn.linear.default_kernel_init)

    def __call__(self, params):
        out = self.mlp_function(params)
        if not self.activate_final:
            out = self.final_layer(out)
        return out



if __name__ == '__main__':
    import jax.numpy as jnp
    model = MLP([12, 8, 4])
    batch = jnp.ones((32, 10))
    variables = model.init(jax.random.PRNGKey(0), batch)
    output = model.apply(variables, batch)
    print(output)

    model = StableMLP([12, 12])
    batch = jnp.ones((32, 10))
    variables = model.init(jax.random.PRNGKey(0), batch)
    output = model.apply(variables, batch)
    print(output)
