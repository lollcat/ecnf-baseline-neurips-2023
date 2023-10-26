from typing import Optional, Sequence, Tuple

from functools import partial
import os
import itertools

import distrax
import jax.numpy as jnp
import jax
import optax
import chex
import matplotlib.pyplot as plt
from flax import linen as nn

from ecnf.cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf
from ecnf.cnf.sample_and_log_prob import sample_cnf, get_log_prob
from ecnf.cnf.gradient_step import TrainingState, flow_matching_update_fn
from ecnf.utils.loop import TrainConfig, run_training
from ecnf.utils.loggers import ListLogger

def setup_target_data(n_train: int = int(1e4), n_test: int = 256):
    key = jax.random.PRNGKey(0)

    n_mixes = 8
    dim = 2
    log_var_scaling = 0.1
    loc_scaling = 10.
    logits = jnp.ones(n_mixes)
    mean = jax.random.uniform(shape=(n_mixes, dim), key=key, minval=-1.0, maxval=1.0) * loc_scaling
    log_var = jnp.ones(shape=(n_mixes, dim)) * log_var_scaling

    mixture_dist = distrax.Categorical(logits=logits)
    var = jax.nn.softplus(log_var)
    components_dist = distrax.Independent(
        distrax.Normal(loc=mean, scale=var), reinterpreted_batch_ndims=1)
    distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
            )

    key1, key2 = jax.random.split(key)
    samples_train = distribution.sample(seed=key1, sample_shape=(n_train,))
    samples_test = distribution.sample(seed=key2, sample_shape=(n_test,))
    return samples_train, samples_test, distribution


def get_timestep_embedding(timesteps: chex.Array, embedding_dim: int):
    """Build sinusoidal embeddings (from Fairseq)."""
    # https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb#scrollTo=O5rq6xovwhgP

    assert timesteps.ndim == 1
    timesteps = timesteps * 1000

    half_dim = embedding_dim // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    # if embedding_dim % 2 == 1:  # zero pad
    #     emb = jax.lax.pad(emb, 0, ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class VectorNet(nn.Module):
    features: Sequence[int] = (512, 512, 512)
    embedding_dim: int = 32

    @nn.compact
    def __call__(self, x: chex.Array, t: chex.Array,
             features: Optional[chex.Array] = None) -> chex.Array:
         chex.assert_rank(x, 2)
         chex.assert_rank(t, 1)
         event_dim = x.shape[-1]

         t_embed = get_timestep_embedding(t, self.embedding_dim)

         for feature in self.features:
            nn_in = jnp.concatenate([x, t_embed], axis=-1)
            x = nn.Dense(feature)(nn_in)
            x = nn.activation.gelu(x)
         out = nn.Dense(event_dim)(x)
         return out



def setup_training():
    lr = 1e-4
    dim = 2
    batch_size = 64
    n_iteration = int(1e2)
    logger = ListLogger()
    seed = 0
    n_eval = 5


    train_data, test_data, target_distribution = setup_target_data()
    optimizer = optax.adamw(lr)

    sigma_min = 1e-4
    base_scale = 5.
    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(dim), scale_diag=jnp.ones(dim)*base_scale)

    get_cond_vector_field = partial(optimal_transport_conditional_vf, sigma_min=sigma_min)
    net = VectorNet()

    cnf = FlowMatchingCNF(init=net.init, apply=net.apply, get_x_t_and_conditional_u_t=get_cond_vector_field,
                          sample_base=base._sample_n, sample_and_log_prob_base=base.sample_and_log_prob,
                          log_prob_base=base.log_prob)


    def init_state(key: chex.PRNGKey) -> TrainingState:
        params = cnf.init(key, train_data[:2], jnp.zeros(2,))
        opt_state = optimizer.init(params=params)
        state = TrainingState(params=params, opt_state=opt_state, key=key)
        return state

    ds_size = train_data.shape[0]

    def run_epoch(state: TrainingState) -> Tuple[TrainingState, dict]:
        key, subkey = jax.random.split(state.key)
        ds_indices = jax.random.permutation(subkey, jnp.arange(ds_size))
        state = state._replace(key=key)
        infos = []

        for j in range(ds_size // batch_size):
            batch = train_data[ds_indices[j*batch_size:(j+1)*batch_size]]
            state, info = flow_matching_update_fn(
                cnf=cnf,
                opt_update=optimizer.update,
                state=state,
                x_data=batch,
                features=None
            )
            infos.append(info)

        info = jax.tree_map(lambda *xs: jnp.stack(xs), *infos)
        return state, info

    def eval_and_plot(state: TrainingState, key: chex.PRNGKey,
                 iteration_n: int, save: bool, plots_dir: str) -> dict:
        key1, key2 = jax.random.split(key)
        key_batch = jax.random.split(key1, test_data.shape[0])
        features = None
        log_prob, log_prob_base, delta_log_likelihood = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, None))(cnf, state.params, test_data, key_batch,
                                                                            features)
        target_log_prob = target_distribution.log_prob(test_data)
        chex.assert_equal_shape((log_prob, target_log_prob))
        info = {}
        info.update(
            test_log_lik=jnp.mean(log_prob),
            test_kl=jnp.mean(target_log_prob - log_prob))


        log_prob_approx, log_prob_base, delta_log_likelihood = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, None, None))(
            cnf, state.params, test_data, key_batch, features, True)
        chex.assert_equal_shape((log_prob_approx, target_log_prob))
        info.update(test_approx_log_lik=jnp.mean(log_prob_approx))


        key, subkey = jax.random.split(key)
        features = None

        # Plot samples.
        n_samples_plotting = 512
        key_batch = jax.random.split(key, n_samples_plotting)
        flow_samples = jax.vmap(sample_cnf, in_axes=(None, None, 0, None))(
            cnf, state.params, key_batch, features)
        fig1, axs = plt.subplots(1)
        axs.plot(flow_samples[:, 0], flow_samples[:, 1], "o", label="flow samples", alpha=0.4)
        axs.plot(train_data[:n_samples_plotting, 0], train_data[:n_samples_plotting, 1],
                 "o", label="target samples", alpha=0.4)
        axs.legend()

        fig2, axs = plt.subplots(1, 2, figsize=(10, 5))
        bound = 8
        n_points = 10
        x_points_dim1 = jnp.linspace(-bound, bound, n_points)
        x_points_dim2 = jnp.linspace(-bound, bound, n_points)
        x_points = jnp.array(list(itertools.product(x_points_dim1, x_points_dim2)))
        vectors_t05 = cnf.apply(state.params, x_points, t=jnp.ones(n_points**2)*0.5, features=None)
        axs[0].quiver(x_points[:, 0], x_points[:, 1], vectors_t05[:, 0], vectors_t05[:, 1])
        axs[0].set_title(f"model score at t={0.5}")
        axs[0].plot(train_data[:n_samples_plotting, 0], train_data[:n_samples_plotting, 1],
                 "o", label="target samples", alpha=0.2)
        vectors_t001 = cnf.apply(state.params, x_points, t=jnp.ones(n_points ** 2) * 0.01, features=None)
        axs[1].quiver(x_points[:, 0], x_points[:, 1], vectors_t001[:, 0], vectors_t001[:, 1])
        axs[1].set_title(f"model score at t={0.01}")
        axs[1].plot(train_data[:n_samples_plotting, 0], train_data[:n_samples_plotting, 1],
                 "o", label="target samples", alpha=0.2)

        figs = [fig1, fig2]
        for j, figure in enumerate(figs):
            if save:
                figure.savefig(
                    os.path.join(plots_dir, "plot_%03i_iter_%08i.png" % (j, iteration_n))
                )
            else:
                plt.show()
            plt.close(figure)

        return info


    train_config = TrainConfig(
        n_iteration=n_iteration,
        logger=logger,
        seed=seed,
        n_checkpoints=0,
        n_eval=n_eval,
        init_state=init_state,
        update_state=run_epoch,
        eval_and_plot_fn=eval_and_plot,
        save=False,
        save_dir="/tmp"
    )

    return train_config






if __name__ == '__main__':

    config = setup_training()
    run_training(config)
