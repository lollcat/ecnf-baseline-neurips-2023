from typing import Callable, Sequence, Tuple, Optional


import jax.numpy as jnp
import jax
import e3nn_jax as e3nn
import chex
from flax import linen as nn

from ecnf.utils.graph import get_senders_and_receivers_fully_connected
from ecnf.utils.numerical import safe_norm
from ecnf.nets.mlp import StableMLP, MLP


class EGCL(nn.Module):
    """A version of EGCL coded only with haiku (not e3nn) so works for arbitary dimension of inputs.

    Follows notation of https://arxiv.org/abs/2105.09016.
        Attributes:
            name (str)
            mlp_units (Sequence[int]): sizes of hidden layers for all MLPs
            residual_h (bool): whether to use a residual connectio probability density for scalars
            residual_x (bool): whether to use a residual connectio probability density for vectors.
            normalization_constant (float): Value to normalize the output of MLP multiplying message vectors.
                C in the en normalizing flows paper (https://arxiv.org/abs/2105.09016).
            variance_scaling_init (float): Value to scale the output variance of MLP multiplying message vectors
    """
    name: str
    mlp_units: Sequence[int]
    n_invariant_feat_hidden: int
    activation_fn: Callable
    residual_h: bool
    residual_x: bool
    stable_mlp: bool
    normalization_constant: float
    variance_scaling_init: float

    def setup(self) -> None:
        mlp_units = self.mlp_units
        activation_fn = self.activation_fn
        mlp_net = StableMLP if self.stable_mlp else MLP
        self._mlp_net = mlp_net
        self.phi_e = mlp_net(mlp_units, activation=activation_fn, activate_final=True)

        self.phi_x_torso = mlp_net(mlp_units, activate_final=True, activation=activation_fn)
        self.phi_h = mlp_net((*mlp_units, self.n_invariant_feat_hidden), activate_final=False,
                                 activation=activation_fn)

    @nn.compact
    def __call__(self, node_positions: chex.Array, node_features: chex.Array, senders: chex.Array,
                 receivers: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """E(N)GNN layer implementation.

        Args:
            node_positions [n_nodes, 3]-ndarray: augmented set of euclidean coordinates for each node
            node_features [n_nodes, self.n_invariant_feat_hidden]-ndarray: scalar features at each node
            senders: [n_edges]-ndarray: sender nodes for each edge
            receivers: [n_edges]-ndarray: receiver nodes for each edge

        Returns:
            vectors_out [n_nodes, 3]-ndarray: augmented set of euclidean coordinates for each node
            features_out [n_nodes, self.n_invariant_feat_hidden]-ndarray: scalar features at each node
        """
        chex.assert_rank(node_positions, 2)
        chex.assert_rank(node_features, 2)
        chex.assert_rank(senders, 1)
        chex.assert_equal_shape([senders, receivers])
        n_nodes, dim = node_positions.shape
        avg_num_neighbours = n_nodes - 1
        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))

        # Prepare the edge attributes.
        vectors = node_positions[receivers] - node_positions[senders]
        lengths = safe_norm(vectors, axis=-1, keepdims=True)

        edge_feat_in = jnp.concatenate([node_features[senders], node_features[receivers], lengths], axis=-1)

        # build messages
        m_ij = self.phi_e(edge_feat_in)

        # Get positional output
        phi_x_out = self.phi_x_torso(m_ij)
        phi_x_out = nn.Dense(
            1, kernel_init=nn.initializers.variance_scaling(self.variance_scaling_init, "fan_avg", "uniform")
        )(phi_x_out)

        shifts_ij = (
            phi_x_out
            * vectors
            / (self.normalization_constant + lengths)
        )  # scale vectors by messages and
        shifts_i = e3nn.scatter_sum(
            data=shifts_ij, dst=receivers, output_size=n_nodes
        )
        vectors_out = shifts_i / avg_num_neighbours
        chex.assert_equal_shape((vectors_out, node_positions))

        # Get feature output
        e = nn.Dense(1)(m_ij)

        e = jax.nn.sigmoid(e)
        m_i = e3nn.scatter_sum(
            data=m_ij*e, dst=receivers, output_size=n_nodes
        ) / jnp.sqrt(avg_num_neighbours)
        phi_h_in = jnp.concatenate([m_i, node_features], axis=-1)
        features_out = self.phi_h(phi_h_in)
        chex.assert_equal_shape((features_out, node_features))

        # Final processing and conversion into plain arrays.
        if self.residual_h:
            features_out = features_out + node_features
        if self.residual_x:
            vectors_out = node_positions + vectors_out
        return vectors_out, features_out


class EGNN(nn.Module):
    """Configuration of EGNN."""
    n_blocks: int  # number of layers
    mlp_units: Sequence[int]
    n_invariant_feat_hidden: int
    name: Optional[str] = None
    activation_fn: Callable = jax.nn.silu
    stable_mlp: bool = True
    residual_h: bool = True
    residual_x: bool = True
    normalization_constant: float = 1.0
    variance_scaling_init: float = 0.001

    @nn.compact
    def __call__(self,
        positions: chex.Array,
        node_features: chex.Array,
        global_features: chex.Array,  # Time embedding.
    ) -> chex.Array:
        assert positions.ndim in (2, 3)
        vmap = positions.ndim == 3
        if vmap:
            return jax.vmap(self.call_single)(positions, node_features, global_features)
        else:
            return self.call_single(positions, node_features, global_features)


    def call_single(self,
        positions: chex.Array,
        node_features: chex.Array,
        global_features: chex.Array,  # Time embedding.
    ) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(positions, 2)
        chex.assert_rank(node_features, 2)
        chex.assert_rank(global_features, 1)
        n_nodes = positions.shape[0]
        chex.assert_axis_dimension(node_features, 0, n_nodes)

        senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)

        n_nodes, dim = positions.shape

        # Setup torso input.
        vectors = positions - positions.mean(axis=0, keepdims=True)
        initial_vectors = vectors
        h = node_features

        # Loop through torso layers.
        for i in range(self.n_blocks):
            h = jnp.concatenate([h, jnp.repeat(global_features[None], n_nodes, axis=0)], axis=1)
            h = nn.Dense(self.n_invariant_feat_hidden)(h)
            vectors, h = EGCL(
                name=str(i),
                mlp_units=self.mlp_units,
                n_invariant_feat_hidden=self.n_invariant_feat_hidden,
                activation_fn=self.activation_fn,
                residual_h=self.residual_h,
                residual_x=self.residual_x,
                normalization_constant=self.normalization_constant,
                variance_scaling_init=self.variance_scaling_init,
                stable_mlp=self.stable_mlp,
                              )(vectors, h, senders, receivers)

        chex.assert_shape(vectors, (n_nodes, dim))
        chex.assert_shape(h, (n_nodes, self.n_invariant_feat_hidden))

        if self.residual_x:
            vectors = vectors - initial_vectors

        vectors = vectors - positions.mean(axis=0, keepdims=True)  # Zero-CoM.

        vectors = vectors * nn.Dense(1, kernel_init=nn.initializers.zeros_init())(h)

        return vectors
