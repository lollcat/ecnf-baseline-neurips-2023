from time import time
import os
import re

from omegaconf import DictConfig
import yaml
from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp
import wandb
import pickle

api = wandb.Api()


from ecnf.cnf.build_cnf import build_cnf

from examples.qm9 import load_dataset
from ecnf.setup_training import setup_eval
from load_checkpoint_measure_sampling_time import get_wandb_run



def download_checkpoint(tags, seed, max_iter, base_path):
    assert os.path.exists(base_path)
    run = get_wandb_run(tags, seed)
    for file in run.files():
        if re.search(fr'.*{max_iter-1}.pkl', str(file)):
            file.download(exist_ok=True)
            path = re.search(r"([^\s]*model_checkpoints[^\s]*)", str(file)).group()
            new_path = f"{base_path}/seed{seed}.pkl"
            os.replace(path, new_path)
            print("saved" + path)

    state = pickle.load(open(new_path, "rb"))
    return state


def download_wandb_checkpoint(auto_download: bool = False, file_path: str = './seed0.pkl'):
    # Returns params from checkpoint at the end of training.
    if auto_download:
        state = download_checkpoint(tags=["qm9", "flow_matching"], seed=0, max_iter=16000, base_path=".")
    else:
        state = pickle.load(open(file_path, "rb"))
    return state


if __name__ == '__main__':
    cfg = DictConfig(yaml.safe_load(open('examples/config/qm9.yaml', 'r')))

    train_data_, test_data_ = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size,
                                           True)
    test_data_ = test_data_[:2]


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
                    n_features=1,
                    )

    state = download_wandb_checkpoint()

    key = jax.random.PRNGKey(0)

    evaluate = setup_eval(cfg, cnf, n_nodes, dim, target_log_prob_fn=None,
               test_pos_flat=test_pos_flat, test_features_flat=test_features_flat)

    info = evaluate(state=state, key=key, iteration_n=-10)
    print(info)
