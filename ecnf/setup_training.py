from typing import Tuple, Callable, Optional, Sequence

import os
import pathlib
from functools import partial

import jax.numpy as jnp
import jax
import optax
import wandb
import chex
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
from omegaconf import DictConfig

from ecnf.utils.numerical import maybe_masked_mean
from ecnf.targets.data import FullGraphSample
from ecnf.cnf.core import FlowMatchingCNF
from ecnf.cnf.build_cnf import build_cnf
from ecnf.cnf.sample_and_log_prob import sample_cnf, get_log_prob, sample_and_log_prob_cnf
from ecnf.cnf.gradient_step import TrainingState, flow_matching_update_fn
from ecnf.utils.loop import TrainConfig
from ecnf.utils.setup_train_objects import setup_logger
from ecnf.utils.loggers import WandbLogger
from ecnf.utils.plotting import bin_samples_by_dist, get_pairwise_distances_for_plotting, get_counts
from ecnf.utils.evaluation import eval_fn, calculate_forward_ess


Plotter = Callable[[TrainingState, FullGraphSample, chex.PRNGKey], Sequence[plt.figure]]


def setup_default_plotter(
        cnf: FlowMatchingCNF,
        n_nodes: int,
        dim: int,
        n_samples_plotting: int) -> Plotter:

    def default_plotter(
            state: TrainingState,
            train_data_: FullGraphSample,
            key: chex.PRNGKey,
    ) -> Sequence[plt.figure]:
        features_flat = train_data_.features[0].flatten()

        key, subkey = jax.random.split(key)
        key_batch = jax.random.split(key, n_samples_plotting)
        flow_samples_flat = jax.vmap(sample_cnf, in_axes=(None, None, 0, 0))(
            cnf, state.params, key_batch, jnp.repeat(features_flat[None], n_samples_plotting, axis=0))
        flow_samples = jnp.reshape(flow_samples_flat, (n_samples_plotting, n_nodes, dim))

        # Plot samples.
        bins_x, count_list = bin_samples_by_dist([train_data_.positions[:n_samples_plotting]], max_distance=10.)
        plotting_n_nodes = train_data_.positions.shape[1]
        pairwise_distances_flow = get_pairwise_distances_for_plotting(flow_samples, plotting_n_nodes, max_distance=10.)
        counts_flow = get_counts(pairwise_distances_flow, bins_x)

        fig1, ax = plt.subplots(1, figsize=(5, 5))
        ax.stairs(count_list[0], bins_x, label="train samples", alpha=0.4, fill=True)
        ax.stairs(counts_flow, bins_x, label="flow samples", alpha=0.4, fill=True)
        ax.legend()

        figs = [fig1, ]
        return figs

    return default_plotter


def setup_training(
        cfg: DictConfig,
        load_dataset: Callable[[int, int], Tuple[FullGraphSample, FullGraphSample]],
        target_log_prob_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
        plotter: Optional[Plotter] = None
) -> TrainConfig:

    batch_size = cfg.training.batch_size
    n_samples_plotting = cfg.training.plot_batch_size

    logger = setup_logger(cfg)

    if isinstance(logger, WandbLogger) and cfg.training.save_in_wandb_dir:
        save_path = os.path.join(wandb.run.dir, cfg.training.save_dir)
    else:
        save_path = cfg.training.save_dir

    pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)


    train_data_, test_data_ = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)

    # Ensure Zero-Com.
    train_data_ = train_data_._replace(
        positions=train_data_.positions - jnp.mean(train_data_.positions, axis=1, keepdims=True))
    test_data_ = test_data_._replace(
        positions=test_data_.positions - jnp.mean(test_data_.positions, axis=1, keepdims=True))

    optimizer_config = cfg.training.optimizer
    if optimizer_config.use_schedule:
        n_batches_per_epoch = train_data_.positions.shape[0] // batch_size
        n_iter_total = cfg.training.n_training_iter * n_batches_per_epoch
        lr = optax.warmup_cosine_decay_schedule(
                init_value=float(optimizer_config.init_lr),
                peak_value=float(optimizer_config.peak_lr),
                end_value=float(optimizer_config.end_lr),
                warmup_steps=optimizer_config.n_iter_warmup,
                decay_steps=n_iter_total
                )
    else:
        lr = optimizer_config.init_lr
    optimizer = optax.adam(lr)

    _, unravel_pytree = ravel_pytree(train_data_[0])
    ravel_pytree_batched = jax.vmap(lambda x: ravel_pytree(x)[0])
    train_pos_flat = ravel_pytree_batched(train_data_.positions)
    train_features_flat = ravel_pytree_batched(train_data_.features)
    test_pos_flat = ravel_pytree_batched(test_data_.positions)
    test_features_flat = ravel_pytree_batched(test_data_.features)

    n_nodes, dim = train_data_.positions.shape[1:]


    cnf = build_cnf(dim=dim,
                    n_frames=n_nodes,
                    sigma_min=cfg.flow.sigma_min,
                    base_scale=cfg.flow.base_scale,
                    n_blocks_egnn=cfg.flow.network.n_blocks_egnn,
                    mlp_units=cfg.flow.network.mlp_units,
                    n_invariant_feat_hidden=cfg.flow.network.n_invariant_feat_hidden,
                    time_embedding_dim=cfg.flow.network.time_embedding_dim,
                    n_features=int(train_features_flat.max() + 1),
                    )


    def init_state(key: chex.PRNGKey) -> TrainingState:
        params = cnf.init(key, train_pos_flat[:2], jnp.zeros(2), train_features_flat[:2])
        opt_state = optimizer.init(params=params)

        ema_params = params if cfg.training.use_ema else jnp.array(None)
        state = TrainingState(params=params, opt_state=opt_state, key=key,
                              ema_params=ema_params)
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


    if target_log_prob_fn and cfg.training.eval_n_model_samples is not None:
        def eval_batch_free_fn(key: chex.PRNGKey, state: TrainingState) -> dict:
            def forward(carry: None, xs: chex.PRNGKey):
                key = xs
                samples, log_q = sample_and_log_prob_cnf(cnf, state.params, key, features=train_features_flat[0],
                                                        approx=cfg.training.eval_exact_log_prob,
                                                        use_fixed_step_size=cfg.training.use_fixed_step_size
                                                         )
                samples = jnp.reshape(samples, (-1, n_nodes, dim))
                log_p = target_log_prob_fn(samples)
                log_w = log_p - log_q
                return None, log_w

            n_batches = cfg.training.eval_n_model_samples
            _, log_w = jax.lax.scan(forward, init=None, xs=jax.random.split(key, n_batches))
            log_w = log_w.flatten()
            rv_ess = 1 / jnp.sum(jax.nn.softmax(log_w) ** 2) / log_w.shape[0]
            info = {}
            info.update(rv_ess=rv_ess)
            return info
    else:
        eval_batch_free_fn = None


    def eval_on_data_batch_fn(data: chex.ArrayTree, key: chex.PRNGKey, mask: chex.Array, state: TrainingState) -> \
            Tuple[chex.Array, dict]:
        key1, key2 = jax.random.split(key)
        test_pos_flat, test_features_flat = data
        key_batch = jax.random.split(key1, test_pos_flat.shape[0])

        if cfg.training.eval_exact_log_prob:
            log_q, log_prob_base, delta_log_lik = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, 0, None))(
                cnf, state.params, test_pos_flat, key_batch, test_features_flat,
                cfg.training.use_fixed_step_size
            )
        else:
            log_q, log_prob_base, delta_log_lik = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, 0, None, None))(
                cnf, state.params, test_pos_flat, key_batch, test_features_flat, True, cfg.training.use_fixed_step_size)

        info = {}
        info.update(
            test_log_lik=maybe_masked_mean(log_q, mask),
            test_log_prob_base=maybe_masked_mean(log_prob_base, mask),
            test_delta_log_lik=maybe_masked_mean(delta_log_lik, mask),
        )

        if target_log_prob_fn is not None:
            test_pos = jnp.reshape(test_pos_flat, (-1, n_nodes, dim))
            log_p = target_log_prob_fn(test_pos)
            log_w = log_p - log_q
        else:
            log_w = None
        return log_w, info


    if plotter is None:
        plotter = setup_default_plotter(cnf=cnf, n_nodes=n_nodes, dim=dim, n_samples_plotting=n_samples_plotting)


    def eval_and_plot(
            state: TrainingState, key: chex.PRNGKey,
            iteration_n: int, save: bool, plots_dir: str) -> dict:

        if cfg.training.use_ema and (cfg.training.n_training_iter - 1) == iteration_n:
            state = state._replace(params=state.ema_params)

        info, log_w_fwd, flat_mask = eval_fn(
            x=(test_pos_flat, test_features_flat),
            key=key,
            eval_on_test_batch_fn=partial(eval_on_data_batch_fn, state=state),
            eval_batch_free_fn=partial(eval_batch_free_fn, state=state) if eval_batch_free_fn is not None else None,
            batch_size=cfg.training.eval_batch_size,
        )

        if target_log_prob_fn is not None:
            further_info = calculate_forward_ess(log_w_fwd, mask=flat_mask)
            info.update(further_info)


        figs = plotter(state, train_data_, key)

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
        n_iteration=cfg.training.n_training_iter,
        logger=logger,
        seed=cfg.training.seed,
        n_checkpoints=cfg.training.n_checkpoints,
        n_eval=cfg.training.n_eval,
        init_state=init_state,
        update_state=run_epoch,
        eval_and_plot_fn=eval_and_plot,
        save=cfg.training.save,
        save_dir=save_path)

    return train_config
