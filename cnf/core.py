from typing import Protocol, NamedTuple, Optional, Callable, Tuple

import chex

Time = chex.Array

class VectorFieldApply(Protocol):
    def __call__(self, params: chex.Array, x: chex.Array, t: chex.Array,
                 features: Optional[chex.Array] = None) -> chex.Array:
        """
        Args:
            params: Neural network parameters
            x: Event.
            t: Time.
            features: Features to condition on (e.g. graph features).

        Returns:
            vector_field: Vector field output by the model
        """


class GetConditionalVectorField(Protocol):
    def __call__(self, x0: chex.Array, x1: chex.Array, t: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            x0: Event from base distribution.
            x1: Event from target distribution
            t: Time.

        Returns:
            x_t: Sample x_t = \phi_t(x_0 | x_1)
            u(x_t | x_1): Conditional vector field.
        """

def optimal_transport_conditional_vf(x0: chex.Array, x1: chex.Array, t: chex.Array, sigma_min: float) -> \
        Tuple[chex.Array, chex.Array]:
    x_t = (1 - (1-sigma_min)*t) * x0 + t * x1
    u_t = x1 - (1 - sigma_min)*x0
    return x_t, u_t


class FlowMatchingCNF(NamedTuple):
    """Define all the callables needed for a flow matching CNF."""
    init: Callable[[chex.PRNGKey, chex.Array, Optional[chex.Array]], chex.ArrayTree]
    apply: VectorFieldApply
    sample_base: Callable[[chex.PRNGKey, int], chex.Array]
    get_x_t_and_conditional_u_t: GetConditionalVectorField
