# Adapted from https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing#scrollTo=zOsoqPdXHuL5.

from typing import Protocol, NamedTuple, Optional, Callable, Tuple

import chex
import distrax
import jax.numpy as jnp

from ecnf.cnf.core import CNF, VectorFieldApply, SampleAndLogProbBase, SampleBase, LogProbBase

Time = chex.Array

class ScoreFnApply(Protocol):
    def __call__(self, params: chex.Array, x: chex.Array, t: chex.Array,
                 features: Optional[chex.Array] = None) -> chex.Array:
        """
        Args:
            params: Neural network parameters
            x: Event.
            t: Time.
            features: Features to condition on (e.g. graph features).

        Returns:
            score: Score output by the model.
        """


def marginal_prob_std(t: Time, sigma: float) -> chex.Array:
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    return jnp.sqrt((sigma ** (2 * t) - 1.) / 2. / jnp.log(sigma))


def diffusion_coeff(t: Time, sigma: float) -> chex.Array:
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return sigma ** t


class DiffusionCNF(NamedTuple):
    """CNF from diffusion model's reverse SDE marginal distribution."""
    apply: VectorFieldApply
    sample_base: SampleBase
    log_prob_base: LogProbBase
    sample_and_log_prob_base: SampleAndLogProbBase


class DiffusionModel(NamedTuple):
    """Define all the callables needed for a flow matching CNF."""
    init: Callable[[chex.PRNGKey, chex.Array, Optional[chex.Array]], chex.ArrayTree]
    apply: ScoreFnApply
    sigma: float
    zero_mean: bool
    dim: int
    eps: float = 1e-5


    def marginal_prob_std_fn(self, t: Time) -> chex.Array:
        return marginal_prob_std(t, self.sigma)

    def diffusion_coeff(self, t: Time) -> chex.Array:
        return diffusion_coeff(t, self.sigma)

    def reverse_ode_vector_field(self, params: chex.Array, x: chex.Array, t: chex.Array,
                 features: Optional[chex.Array] = None) -> chex.Array:
        chex.assert_rank(x, 2)
        chex.assert_rank(t, 1)
        t = 1 - t  # For CNF by convention we set t=0 to be the base distribution.
        score = self.apply(params, x, t, features)
        g = self.diffusion_coeff(t)
        vector_field = -0.5 * g[:, None]**2 * score
        return vector_field

    def sample_base(self, key: chex.PRNGKey, n: int) -> chex.Array:
        t_base = jnp.array(1.)
        sigma_max = self.marginal_prob_std_fn(t_base)
        if self.zero_mean:
            raise NotImplementedError
        else:
            base = distrax.MultivariateNormalDiag(
                loc=jnp.zeros(self.dim),
                scale_diag=jnp.ones(self.dim)*sigma_max)
        return base._sample_n(key, n)

    def log_prob_base(self, value: chex.Array) -> chex.Array:
        t_base = jnp.array(1.)
        sigma_max = self.marginal_prob_std_fn(t_base)
        if self.zero_mean:
            raise NotImplementedError
        else:
            base = distrax.MultivariateNormalDiag(
                loc=jnp.zeros(self.dim),
                scale_diag=jnp.ones(self.dim)*sigma_max)
        return base.log_prob(value)


    def sample_and_log_prob_base(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> \
            Tuple[chex.Array, chex.Array]:
        t_base = jnp.array(1.)
        sigma_max = self.marginal_prob_std_fn(t_base)
        if self.zero_mean:
            raise NotImplementedError
        else:
            base = distrax.MultivariateNormalDiag(
                loc=jnp.zeros(self.dim),
                scale_diag=jnp.ones(self.dim)*sigma_max)
        return base.sample_and_log_prob(seed=seed, sample_shape=sample_shape)


    def to_cnf(self) -> CNF:
        cnf = DiffusionCNF(
            sample_base=self.sample_base,
            apply=self.reverse_ode_vector_field,
            log_prob_base=self.log_prob_base,
            sample_and_log_prob_base=self.sample_and_log_prob_base
        )
        return cnf
