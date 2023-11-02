from typing import Optional

import jax.numpy as jnp
import chex
import jax

def safe_norm(x: chex.Array, axis: int = None, keepdims: bool = False) -> jnp.ndarray:
    """nan-safe norm. Copied from mace-jax"""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0, 1, x2) ** 0.5

def vector_rejection(a: chex.Array, b: chex.Array) -> chex.Array:
    chex.assert_rank(a, 1)
    chex.assert_equal_shape((a, b))
    vector_proj = b * jnp.dot(a, b) / jnp.dot(b, b)
    return a - vector_proj

def rotate_3d(x: chex.Array, theta: chex.Array, phi: chex.Array) -> chex.Array:
    chex.assert_shape(theta, ())
    chex.assert_shape(x, (3,))
    rotation_matrix_1 = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta), 0],
         [jnp.sin(theta), jnp.cos(theta), 0],
         [0,              0,              1]]
    )
    rotation_matrix_2 = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(phi), -jnp.sin(phi)],
        [0, jnp.sin(phi), jnp.cos(phi)],
         ])
    x = jnp.matmul(rotation_matrix_1, x)
    x = jnp.matmul(rotation_matrix_2, x)
    return x

def get_leading_axis_tree(tree: chex.ArrayTree, n_dims: int = 1) -> chex.Shape:
    flat_tree, _ = jax.tree_util.tree_flatten(tree)
    leading_shape = flat_tree[0].shape[:n_dims]
    chex.assert_tree_shape_prefix(tree, leading_shape)
    return leading_shape



def maybe_masked_mean(array: chex.Array, mask: Optional[chex.Array]):
    chex.assert_rank(array, 1)
    if mask is None:
        return jnp.mean(array)
    else:
        chex.assert_equal_shape([array, mask])
        array = jnp.where(mask, array, jnp.zeros_like(array))
        divisor = jnp.sum(mask)
        divisor = jnp.where(divisor == 0, jnp.ones_like(divisor), divisor)  # Prevent nan when fully masked.
        return jnp.sum(array) / divisor
