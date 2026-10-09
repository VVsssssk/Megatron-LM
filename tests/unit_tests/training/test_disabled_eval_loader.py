# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset

from megatron.core.datasets.utils import Split
from megatron.training.datasets import data_samplers


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Sampler regressions use only the synthetic in-memory dataset below."""


class SplitDataset(Dataset):
    def __init__(self, split, size, split_attribute):
        setattr(self, split_attribute, split)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


@pytest.fixture
def loader_args(monkeypatch):
    args = SimpleNamespace(
        dataloader_type='single',
        eval_iters=0,
        full_validation=False,
        micro_batch_size=1,
        num_workers=0,
        use_varlen_dataset=True,
        varlen_sbhd_validation=False,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
    )
    monkeypatch.setattr(data_samplers, 'get_args', lambda: args)
    monkeypatch.setattr(data_samplers.mpu, 'get_data_parallel_rank', lambda: 0)
    monkeypatch.setattr(data_samplers.mpu, 'get_data_parallel_world_size', lambda: 1)
    return args


@pytest.mark.parametrize('split', [Split.valid, Split.test])
@pytest.mark.parametrize('attribute', ['split', 'index_split'])
def test_disabled_evaluation_does_not_build_empty_sampler(loader_args, split, attribute):
    dataset = SplitDataset(split, 0, attribute)
    assert data_samplers.build_pretraining_data_loader(dataset, 0) is None


@pytest.mark.parametrize('split', [Split.train, Split.valid, Split.test])
def test_enabled_splits_yield_samples(loader_args, split):
    loader_args.eval_iters = 1
    dataset = SplitDataset(split, 2, 'index_split')
    loader = data_samplers.build_pretraining_data_loader(dataset, 0)
    assert list(loader) == [[0], [1]]


def test_training_still_requires_nonempty_dataset(loader_args):
    with pytest.raises(AssertionError, match='no sample to consume'):
        data_samplers.build_pretraining_data_loader(SplitDataset(Split.train, 0, 'split'), 0)


def test_full_validation_remains_enabled(loader_args):
    loader_args.full_validation = True
    dataset = SplitDataset(Split.valid, 2, 'split')
    loader = data_samplers.build_pretraining_data_loader(dataset, 0)
    assert isinstance(loader.batch_sampler, data_samplers.MegatronFullValidationSampler)
    assert list(loader) == [[0], [1]]
