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
from ecnf.cnf.sample_and_log_prob import sample_cnf

from examples.qm9 import load_dataset


def get_wandb_run(tags, seed, donwload_latest_completed: bool = True):
    filter_list = [{"tags": tag} for tag in tags]
    filter_list.extend([
         {"config.training": {"$regex": f"'seed': {seed},"}}
    ])
    filters = {"$and": filter_list}
    if 'fab' not in tags:
        filters.update({"$not": {"tags": 'fab'}})
    runs = api.runs(path='flow-ais-bootstrap/fab',
                    filters=filters)
    if len(runs) > 1:
        if not donwload_latest_completed:
            print(f"found {len(runs)} multiple runs, getting first")
            return runs[-1]
        print(f"Found {len(runs)} runs for tags {tags}, seed {seed}. "
              f"Taking the most recent.")
    elif len(runs) == 0:
        raise Exception(f"No runs for for tags {tags}, seed {seed} found!")
    j = 0
    while not "finished" in str(runs[j]):
        j += 1
    run = runs[j]  # Get latest finished run.
    return run


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
    params = state.params
    return params


def download_wandb_checkpoint():
    # Returns params from checkpoint at the end of training.
    params = download_checkpoint(tags=["qm9", "flow_matching"], seed=0, max_iter=16000, base_path=".")
    return params



if __name__ == '__main__':
    cfg = DictConfig(yaml.safe_load(open('examples/config/qm9.yaml', 'r')))

    train_data_, test_data_ = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size,
                                           True)

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
                    time_embedding_dim=cfg.flow.network.time_embedding_dim
                    )


    n_samples = 10
    key = jax.random.PRNGKey(0)
    params = download_wandb_checkpoint()
    times = []
    samples = []

    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        start = time()
        flow_samples_flat = jax.jit(sample_cnf, static_argnums=0)(cnf, params, subkey, train_features_flat[0])
        run_time = time() - start
        if i == 0:
            print(f"compile time of {run_time}")
        else:
            times.append(run_time)
        samples.append(flow_samples_flat)
        assert flow_samples_flat.shape == (n_nodes*dim,)

    print(times)
