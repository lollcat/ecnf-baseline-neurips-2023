from typing import Tuple, Optional

from functools import partial

import hydra
from omegaconf import DictConfig

from ecnf.utils.loop import run_training
from ecnf.targets.data import load_dw4, FullGraphSample
from ecnf.targets.target_energy.double_well import log_prob_fn
from examples.setup_training import setup_training



def load_dataset(train_set_size: int, valid_set_size: Optional[int], final_run: bool) -> Tuple[FullGraphSample, FullGraphSample]:
    train, valid, test = load_dw4(train_set_size)
    if not final_run:
        return train, valid[:valid_set_size]
    else:
        return train, test[:valid_set_size]

@hydra.main(config_path="./config", config_name="dw4.yaml")
def run(cfg: DictConfig):
    local = True
    if local:
        cfg.logger = DictConfig({"list_logger": None})
        cfg.training.save = False
        cfg.flow.base_scale = 10.
        cfg.training.batch_size = 8
        cfg.training.eval_batch_size = 9
        cfg.training.n_training_iter = 10
        cfg.training.train_set_size = 80
        cfg.training.test_set_size = 80
        cfg.training.plot_batch_size = 16
        cfg.flow.network.mlp_units = (16,)
        cfg.flow.network.n_blocks_egnn = 2
        cfg.flow.network.n_invariant_feat_hidden = 8
        cfg.flow.network.time_embedding_dim = 6

    train_config = setup_training(
        cfg,
        load_dataset=partial(load_dataset, final_run=cfg.training.final_run),
        target_log_prob_fn=log_prob_fn
    )
    run_training(train_config)


if __name__ == '__main__':
    run()
