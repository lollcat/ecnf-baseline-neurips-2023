from typing import Tuple

from functools import partial

import hydra
from omegaconf import DictConfig

from ecnf.utils.loop import run_training
from ecnf.targets.data import load_aldp, FullGraphSample
from ecnf.setup_training import setup_training


def load_dataset(train_set_size, valid_set_size, final_run: bool,
                 train_path: str, test_path: str, valid_path: str) -> Tuple[FullGraphSample, FullGraphSample]:
    train_data, valid_data, test_data = load_aldp(
        train_path=train_path, test_path=test_path, val_path=valid_path,
        train_n_points=train_set_size
    )
    if not final_run:
        return train_data, valid_data[:valid_set_size]
    else:
        return train_data, test_data[:valid_set_size]


@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    local = False
    if local:
        cfg.logger = DictConfig({"list_logger": None})
        cfg.training.save = False
        cfg.training.batch_size = 2
        cfg.training.eval_batch_size = 2
        cfg.training.n_training_iter = 10
        cfg.training.train_set_size = 8
        cfg.training.test_set_size = 8
        cfg.training.plot_batch_size = 16
        cfg.flow.network.mlp_units = (4,)
        cfg.flow.network.n_blocks_egnn = 1
        cfg.flow.network.n_invariant_feat_hidden = 8
        cfg.flow.network.time_embedding_dim = 6

    train_config = setup_training(
        cfg,
        load_dataset=partial(load_dataset,
                             train_path=cfg.target.train_path,
                             test_path=cfg.target.test_path,
                             valid_path=cfg.target.valid_path,
                             final_run=cfg.training.final_run),
    )
    run_training(train_config)


if __name__ == '__main__':
    run()
