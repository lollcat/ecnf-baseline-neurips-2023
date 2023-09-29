from typing import Optional, Sequence, Tuple

from functools import partial
import os

import distrax
import jax.numpy as jnp
import jax
import optax
import chex
from flax import linen as nn

from cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf, GetConditionalVectorField
from cnf.gradient_step import TrainingState, flow_matching_update_fn
from ecnf.nets.mlp import MLP
from ecnf.utils.loop import TrainConfig, run_training
from ecnf.utils.loggers import ListLogger

def setup_target_data(n: int = 128):
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
    components_dist = distrax.Independent(distrax.Normal(loc=mean, scale=var),
                                          reinterpreted_batch_ndims=1)
    distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
            )

    samples = distribution.sample(seed=key, sample_shape=(n,))
    return samples

class VectorNet(nn.Module):
    features: Sequence[int] = (32, 32)
    @nn.compact
    def __call__(self, x: chex.Array, t: chex.Array,
             features: Optional[chex.Array] = None):
         chex.assert_rank(x, 2)
         chex.assert_rank(t, 1)
         nn_in = jnp.concatenate([x, t[:, None]], axis=-1)
         mlp = MLP(features=(*self.features, x.shape[-1]), activate_final=False)
         return mlp(nn_in)



def setup_training():
    lr = 4e-4
    dim = 2
    batch_size = 32
    n_iteration = 100
    logger = ListLogger()
    seed = 0
    n_eval = 5


    target_data = setup_target_data()
    optimizer = optax.adam(lr)

    sigma_min = 0.01
    base_scale = 10.
    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(dim), scale_diag=jnp.ones(dim)*base_scale)

    get_cond_vector_field = partial(optimal_transport_conditional_vf, sigma_min=sigma_min)
    net = VectorNet()

    cnf = FlowMatchingCNF(init=net.init, apply=net.apply, get_x_t_and_conditional_u_t=get_cond_vector_field,
                          sample_base=base._sample_n)


    def init_state(key: chex.PRNGKey) -> TrainingState:
        params = cnf.init(key, target_data[:2], jnp.zeros(2,))
        opt_state = optimizer.init(params=params)
        state = TrainingState(params=params, opt_state=opt_state, key=key)
        return state

    ds_size = target_data.shape[0]

    def run_epoch(state: TrainingState) -> Tuple[TrainingState, dict]:
        key, subkey = jax.random.split(state.key)
        ds_indices = jax.random.permutation(subkey, jnp.arange(ds_size))
        state = state._replace(key=key)
        infos = []

        for j in range(ds_size // batch_size):
            batch = target_data[ds_indices[j*batch_size:(j+1)*batch_size]]
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
        key, subkey = jax.random.split(key)
        figs = []
        for j, figure in enumerate(figs):
            if save:
                figure.savefig(
                    os.path.join(plots_dir, "plot_%03i_iter_%08i.png" % (j, iteration_n))
                )
            else:
                plt.show()
            plt.close(figure)
        info = {}
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
    import matplotlib.pyplot as plt

    samples = setup_target_data(n=1000)
    plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.4)
    plt.show()

    config = setup_training()
    run_training(config)
