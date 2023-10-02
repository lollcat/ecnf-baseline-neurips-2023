from typing import Tuple

import os
import jax.numpy as jnp
import jax
import optax
import chex
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree

from ecnf.cnf.build_cnf import build_cnf
from ecnf.cnf.sample_and_log_prob import sample_cnf, get_log_prob
from ecnf.cnf.gradient_step import TrainingState, flow_matching_update_fn
from ecnf.utils.loop import TrainConfig, run_training
from ecnf.utils.loggers import ListLogger
from ecnf.targets.data import load_lj13



def setup_training():
    lr = 1e-4
    batch_size = 8
    n_iteration = int(1e4)
    logger = ListLogger()
    seed = 0
    n_eval = 5


    train_data_, valid_data_, test_data_ = load_lj13(1000)
    optimizer = optax.adamw(lr)

    _, unravel_pytree = ravel_pytree(train_data_[0])
    ravel_pytree_batched = jax.vmap(lambda x: ravel_pytree(x)[0])
    train_pos_flat = ravel_pytree_batched(train_data_.positions)
    train_features_flat = ravel_pytree_batched(train_data_.features)
    test_pos_flat = ravel_pytree_batched(test_data_.positions)
    test_features_flat = ravel_pytree_batched(test_data_.features)
    valid_pos_flat = ravel_pytree_batched(valid_data_.positions)
    valid_features_flat = ravel_pytree_batched(valid_data_.features)

    n_nodes, dim = train_data_.positions.shape[1:]

    cnf = build_cnf(dim=dim, n_frames=n_nodes)


    def init_state(key: chex.PRNGKey) -> TrainingState:
        params = cnf.init(key, train_pos_flat[:2], jnp.zeros(2), train_features_flat[:2])
        opt_state = optimizer.init(params=params)
        state = TrainingState(params=params, opt_state=opt_state, key=key)
        return state

    ds_size = train_pos_flat.shape[0]

    def run_epoch(state: TrainingState) -> Tuple[TrainingState, dict]:
        key, subkey = jax.random.split(state.key)
        ds_indices = jax.random.permutation(subkey, jnp.arange(ds_size))
        state = state._replace(key=key)
        infos = []

        for j in range(ds_size // batch_size):
            batch_pos = train_pos_flat[ds_indices[j*batch_size:(j+1)*batch_size]]
            batch_feat = train_features_flat[ds_indices[j*batch_size:(j+1)*batch_size]]
            state, info = flow_matching_update_fn(
                cnf=cnf,
                opt_update=optimizer.update,
                state=state,
                x_data=batch_pos,
                features=batch_feat
            )
            infos.append(info)

        info = jax.tree_map(lambda *xs: jnp.stack(xs), *infos)
        return state, info

    def eval_and_plot(state: TrainingState, key: chex.PRNGKey,
                 iteration_n: int, save: bool, plots_dir: str) -> dict:
        return {}
        key1, key2 = jax.random.split(key)
        key_batch = jax.random.split(key1, test_pos_flat.shape[0])
        log_prob = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, 0))(cnf, state.params, test_pos_flat, key_batch,
                                                                            test_features_flat)
        info = {}
        info.update(
            test_log_lik=jnp.mean(log_prob)
        )


        log_prob_approx = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, 0, None))(
            cnf, state.params, test_pos_flat, key_batch, test_features_flat, True)
        info.update(test_approx_log_lik=jnp.mean(log_prob_approx))


        key, subkey = jax.random.split(key)
        features = None

        # Plot samples.
        # n_samples_plotting = 512
        # key_batch = jax.random.split(key, n_samples_plotting)
        # flow_samples = jax.vmap(sample_cnf, in_axes=(None, None, 0, None))(
        #     cnf, state.params, key_batch, features)
        # fig1, axs = plt.subplots(1)
        # axs.plot(flow_samples[:, 0], flow_samples[:, 1], "o", label="flow samples", alpha=0.4)
        # axs.plot(train_data[:n_samples_plotting, 0], train_data[:n_samples_plotting, 1],
        #          "o", label="target samples", alpha=0.4)
        # axs.legend()
        #
        # figs = [fig1,]
        figs = []
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
