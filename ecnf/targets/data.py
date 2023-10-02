from typing import Tuple, Optional, Union, NamedTuple

from pathlib import Path
import jax.numpy as jnp
import numpy as np
import chex

GraphFeatures = chex.Array  # Non-positional information.
Positions = chex.Array


class FullGraphSample(NamedTuple):
    positions: Positions
    features: GraphFeatures

    def __getitem__(self, i):
        return FullGraphSample(self.positions[i], self.features[i])


def positional_dataset_only_to_full_graph(positions: chex.Array) -> FullGraphSample:
    """Convert positional dataset into full graph by using zeros for features. Assumes data is only for x, and not
    augmented coordinates."""
    chex.assert_rank(positions, 3)  # [n_data_points, n_nodes, dim]
    features = jnp.zeros((*positions.shape[:-1], 1), dtype=int)
    return FullGraphSample(positions=positions, features=features)


def load_lj13(
    train_set_size: int = 1000, path: Optional[Union[Path, str]] = None
) -> Tuple[FullGraphSample, FullGraphSample, FullGraphSample]:
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    # Train data
    if path is None:
        here = Path(__file__).parent
        path = here / "data"
    path = Path(path)
    fpath_train = path / "holdout_data_LJ13.npy"
    fpath_idx = path / "idx_LJ13.npy"
    fpath_val_test = path / "all_data_LJ13.npy"

    train_data = jnp.asarray(np.load(fpath_train, allow_pickle=True), dtype=float)
    idxs = jnp.asarray(np.load(fpath_idx, allow_pickle=True), dtype=int)
    val_test_data = jnp.asarray(np.load(fpath_val_test, allow_pickle=True), dtype=float)

    val_data = val_test_data[1000:2000]
    test_data = val_test_data[:1000]

    assert train_set_size <= len(idxs)
    train_data = train_data[idxs[:train_set_size]]

    val_data = jnp.reshape(val_data, (-1, 13, 3))
    test_data = jnp.reshape(test_data, (-1, 13, 3))
    train_data = jnp.reshape(train_data, (-1, 13, 3))

    return (
        positional_dataset_only_to_full_graph(train_data),
        positional_dataset_only_to_full_graph(val_data),
        positional_dataset_only_to_full_graph(test_data),
    )