from typing import Callable

import chex
import jax.numpy as jnp
import distrax
import jax


def get_rotation_matrix_from_angle_2d(angle):
    rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                          [jnp.sin(angle), jnp.cos(angle)]])
    return rotation


def get_rotation_matrix_from_z_a1_a2(z: chex.Array, a1: chex.Array, a2: chex.Array):
    """
    Works as following:
        (1) We rotate about this x-axis by `a2`.
        (2) We rotate about the y-axis such that the rotated x-axis has a z coordinate of `z`.
        (3) We rotate by `a1` about the Z-axis.
    See https://www.mathworks.com/help/phased/ref/roty.html for visualisation of rotation directions.
    """
    a0 = jnp.arctan2(-z, jnp.sqrt(1 - z ** 2))  # Minus because rotation happens from z->x.

    # Rotation about X-axis.
    R1 = jnp.array([
        [1., 0., 0.],
        [0., jnp.cos(a2), -jnp.sin(a2)],
        [0., jnp.sin(a2), jnp.cos(a2)]
    ])
    # Rotation about Y-Axis
    R2 = jnp.array([
        [jnp.cos(a0), 0., jnp.sin(a0)],
        [0., 1., 0],
        [-jnp.sin(a0), 0., jnp.cos(a0)]
    ])
    # Rotate about Z-axis.
    R3 = jnp.array([
        [jnp.cos(a1), -jnp.sin(a1), 0],
        [jnp.sin(a1), jnp.cos(a1), 0],
        [0., 0., 1]
    ])
    return R3 @ R2 @ R1


def random_rotation_matrix(key: chex.PRNGKey, dim: int) -> chex.Array:
    if dim == 3:
        key1, key2, key3 = jax.random.split(key, 3)
        z1, log_p_z1 = distrax.Uniform(low=-1., high=1.).sample_and_log_prob(seed=key1)
        a1, log_p_a1 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)
        a2, log_p_a2 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key3)
        rotation_matrix = get_rotation_matrix_from_z_a1_a2(z1, a1, a2)
    else:
        assert dim == 2
        angles = jax.random.uniform(key) * jnp.pi * 2 - jnp.pi
        rotation_matrix = get_rotation_matrix_from_angle_2d(angles)
    return rotation_matrix


def assert_function_is_equivariant(
        equivariant_fn: Callable[[chex.Array], chex.Array],
        n_nodes: int,
        dim: int = 3,
        key: chex.PRNGKey = jax.random.PRNGKey(0)
) -> None:
    key1, key2 = jax.random.split(key)
    input = jax.random.normal(key1, (n_nodes, dim))
    rotation = random_rotation_matrix(key2, dim)

    input_g = (rotation @ input.T).T

    out = equivariant_fn(input)
    out_then_g = (rotation @ out.T).T
    g_then_out = equivariant_fn(input_g)

    chex.assert_trees_all_close(out_then_g, g_then_out, atol=1e-6, rtol=1e-6)

