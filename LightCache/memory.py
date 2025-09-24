import math
import tqdm

round_func = math.ceil

# Adapted from https://github.com/wuyushuwys/FMEDiffusion
# Original Author: @wuyushuwys

class MemoryEfficientMixin:
    _num_chunk = None
    _spatial_slicing = True
    _num_spatial_chunk = None
    _pbar_kwargs = dict(
        disable=True,
    )

    @property
    def num_chunk(self):
        return self._num_chunk

    @num_chunk.setter
    def num_chunk(self, value):
        assert isinstance(value, int)
        self._num_chunk = value

    @property
    def spatial_slicing(self):
        return self._spatial_slicing

    @spatial_slicing.setter
    def spatial_slicing(self, value):
        assert isinstance(value, bool)
        self._spatial_slicing = value

    @property
    def num_spatial_chunk(self):
        return self._num_spatial_chunk if self._num_spatial_chunk is not None else self._num_chunk

    @num_spatial_chunk.setter
    def num_spatial_chunk(self, value):
        assert isinstance(value, int)
        self._num_spatial_chunk = value

    def enable_progressive_bar(self):
        self._pbar_kwargs['disable'] = False

    def disable_progressive_bar(self):
        self._pbar_kwargs['disable'] = True

    def progress_bar(self, *args, **kwargs):
        kw = self._pbar_kwargs.copy()
        kw.update(kwargs)
        return tqdm.tqdm(range(*args), **kwargs)