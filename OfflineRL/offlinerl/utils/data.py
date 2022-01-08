import os
import torch
import pprint
import numpy as np
from typing import *

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import dataset
from torch.utils.data import dataloader

from loguru import logger


def to_array_as(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return x.detach().cpu().numpy().astype(y.dtype)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tensor):
        return torch.as_tensor(x).to(y)
    else:
        return x


class BufferDataset(dataset.Dataset):
    def __init__(self, buffer, batch_size=256):
        self.buffer = buffer
        self.batch_size = batch_size
        self.length = len(self.buffer)

    def __getitem__(self, index):
        indices = np.random.randint(0, self.length, self.batch_size)
        data = self.buffer[indices]

        return data

    def __len__(self):
        return self.length


class BufferDataloader(dataloader.DataLoader):
    def sample(self, batch_size=None):
        if (
            not hasattr(self, "buffer_loader")
            or batch_size != self.buffer_loader._dataset.batch_size
        ):
            if not hasattr(self, "buffer_loader"):
                self.buffer_loader = self.__iter__()
            elif batch_size is None:
                pass
            else:
                self.dataset.batch_size = batch_size
                self.buffer_loader = self.__iter__()
        try:
            return self.buffer_loader.__next__()
        except:
            self.buffer_loader = self.__iter__()
            return self.buffer_loader.__next__()


class Batch:
    """A batch of named data."""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(dict(*args, **kwargs))

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        """Return self.key."""
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) -> bool:
        """Return key in self."""
        return key in self.__dict__

    def __getstate__(self) -> Dict[str, Any]:
        """Pickling interface.

        Only the actual data are serialized for both efficiency and simplicity.
        """
        state = {}
        for k, v in self.items():
            if isinstance(v, Batch):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickling interface.

        At this point, self is an empty Batch instance that has not been
        initialized, so it can safely be initialized by the pickle state.
        """
        self.__init__(**state)  # type: ignore

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __getitem__(
        self, index: Union[str, Union[slice, int, np.ndarray, List[int]]]
    ) -> "Batch":
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch = Batch()
        for k, v in self.items():
            batch[k] = v[index]
        return batch

    def __setitem__(
        self,
        index: Union[str, Union[slice, int, np.ndarray, List[int]]],
        value: Any,
    ) -> None:
        """Assign value to self[index]."""
        if isinstance(index, str):
            self.__dict__[index] = value
        else:
            assert isinstance(value, Batch)
            for k, v in value.items():
                self[k][index] = v

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + "(\n"
        flag = False
        for k, v in self.items():
            rpl = "\n" + " " * (6 + len(k))
            obj = pprint.pformat(v).replace("\n", rpl)
            s += f"    {k}: {obj},\n"
            flag = True
        if flag:
            s += ")"
        else:
            s = self.__class__.__name__ + "()"
        return s

    def to_numpy(self) -> "Batch":
        """Change all torch.Tensor to numpy.ndarray in-place."""
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.detach().cpu().numpy()
        return self

    def to_torch(
        self, dtype: torch.dtype = torch.float32, device: str = "cpu"
    ) -> "Batch":
        """Change all numpy.ndarray to torch.Tensor in-place."""
        for k, v in self.items():
            self[k] = torch.as_tensor(v, dtype=dtype, device=device)
        return self

    @staticmethod
    def cat(batches: List["Batch"], axis: int = 0) -> "Batch":
        """Concatenate a list of Batch object into a single new batch."""
        if isinstance(list(batches[0].values())[0], np.ndarray):
            cat_func = np.concatenate
        else:
            cat_func = torch.cat
        batch = Batch()
        for k in batches[0].keys():
            batch[k] = cat_func([b[k] for b in batches], axis=axis)
        return batch

    @staticmethod
    def stack(batches: List["Batch"], axis: int = 0) -> "Batch":
        """Stack a list of Batch object into a single new batch."""
        if isinstance(list(batches[0].values())[0], np.ndarray):
            stack_func = np.stack
        else:
            stack_func = torch.stack
        batch = Batch()
        for k in batches[0].keys():
            batch[k] = stack_func([b[k] for b in batches], axis=axis)
        return batch

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def shape(self) -> List[int]:
        data_shape = []
        for v in self.__dict__.values():
            try:
                data_shape.append(list(v.shape))
            except AttributeError:
                data_shape.append([])
        return (
            list(map(min, zip(*data_shape)))
            if len(data_shape) > 1
            else data_shape[0]
        )

    def split(self, size: Union[int, List[int]], shuffle: bool = True):
        if type(size) == list:
            return self._split_with_sizes(size, shuffle)
        else:
            return self._split_with_chunk(size, shuffle)

    def _split_with_chunk(
        self, size: int, shuffle: bool = True, merge_last: bool = False
    ) -> Iterator["Batch"]:
        length = len(self)
        assert 1 <= size  # size can be greater than length, return whole batch
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield self[indices[idx:]]
                break
            yield self[indices[idx : idx + size]]

    def _split_with_sizes(
        self, sizes: List[int], shuffle: bool = True
    ) -> List["Batch"]:
        length = len(self)
        assert sum(sizes) == length, "Wrong sizes"
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)

        res = []
        start_size = 0
        for size in sizes:
            res.append(self[indices[start_size : start_size + size]])
            start_size += size
        return res

    def sample(self, batch_size):
        length = len(self)
        assert 1 <= batch_size

        indices = np.random.randint(0, length, batch_size)

        return self[indices]


class SampleBatch(Batch):
    def sample(self, batch_size):
        length = len(self)
        assert 1 <= batch_size

        indices = np.random.randint(0, length, batch_size)

        return self[indices]


def sample(batch: Batch, batch_size: int):
    length = len(batch)
    assert 1 <= batch_size

    indices = np.random.randint(0, length, batch_size)

    return batch[indices]


def get_scaler(data):
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(data)

    return scaler


class ModelBuffer:
    def __init__(self, buffer_size):
        self.data = None
        self.buffer_size = int(buffer_size)

    def put(self, batch_data):
        batch_data.to_torch(device="cpu")

        if self.data is None:
            self.data = batch_data
        else:
            self.data = Batch.cat([self.data, batch_data], axis=0)

        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size :]

    def __len__(self):
        if self.data is None:
            return 0
        return self.data.shape[0]

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self), size=(batch_size))
        return self.data[indexes]


class LoggedReplayBuffer:
    def __init__(self, buffer_size, log_path=None):
        self.data = None
        self.add_count = 0
        self.log_count = 0
        self.buffer_size = int(buffer_size)
        self.log_path = log_path

    def put(self, batch_data: Batch):
        batch_data.to_torch(device="cpu")

        if self.data is None:
            self.data = batch_data
        else:
            self.data = Batch.cat([self.data, batch_data], axis=0)

        self.add_count += 1
        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size :]

        if self.add_count % self.buffer_size == 0:
            self._log_data()

    def _log_data(self):
        # logging data to file
        if self.log_path is None:
            return
        else:
            if not os.path.exists(self.log_path):
                logger.info(f"{self.log_path} not exists, creating!!!")
                os.makedirs(self.log_path)

        full_log_path = f"{self.log_path}/part-{self.log_count}.npz"
        if full_log_path is not None and os.path.exists(full_log_path):
            logger.info(f"{full_log_path} exists !!!")
        else:
            np.savez_compressed(
                full_log_path,
                observations=self.data["obs"].numpy(),
                actions=self.data["act"].numpy(),
                rewards=self.data["rew"].numpy(),
                terminals=self.data["done"].numpy(),
                timeouts=self.data["done"].numpy(),
                next_observations=self.data["obs_next"].numpy(),
            )
            logger.info(f"Logging buffer to {full_log_path}.")
            self.log_count += 1

    def __len__(self):
        if self.data is None:
            return 0
        return self.data.shape[0]

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self), size=(batch_size))
        return self.data[indexes]
