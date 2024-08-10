# Copied from PyTorch Lightning at their suggestion,
# since they are deprecating this class in a future
# version of the package.

from typing import Any, Optional, Union

import torch
from torch.nn import Module
from typing_extensions import Self

from typing import Any, List, Optional, Union

import torch
from torch.nn import Module
from typing_extensions import Self


class DeviceDtypeModuleMixin(Module):
    __jit_unused_properties__: List[str] = ["device", "dtype"]

    def __init__(self) -> None:
        super().__init__()
        self._dtype: Union[str, torch.dtype] = torch.get_default_dtype()
        self._device = torch.device("cpu")

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]) -> None:
        # necessary to avoid infinite recursion
        raise RuntimeError(
            "Cannot set the dtype explicitly. Please use module.to(new_dtype)."
        )

    @property
    def device(self) -> torch.device:
        device = self._device

        # make this more explicit to always include the index
        if device.type == "cuda" and device.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        return device

    def to(self, *args: Any, **kwargs: Any) -> Self:  # type: ignore[valid-type]
        """See :meth:`torch.nn.Module.to`."""
        # this converts `str` device to `torch.device`
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        self.__update_properties(device=device, dtype=dtype)
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:  # type: ignore[valid-type]
        """Moves all model parameters and buffers to the GPU. This also makes associated parameters and buffers
        different objects. So it should be called before constructing optimizer if the module will live on GPU
        while being optimized.
        Arguments:
            device: If specified, all parameters will be copied to that device. If `None`, the current CUDA device
                index will be used.
        Returns:
            Module: self
        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        self.__update_properties(device=device)
        return super().cuda(device=device)

    def cpu(self) -> Self:  # type: ignore[valid-type]
        """See :meth:`torch.nn.Module.cpu`."""
        self.__update_properties(device=torch.device("cpu"))
        return super().cpu()

    def type(self, dst_type: Union[str, torch.dtype]) -> Self:  # type: ignore[valid-type]
        """See :meth:`torch.nn.Module.type`."""
        self.__update_properties(dtype=dst_type)
        return super().type(dst_type=dst_type)

    def float(self) -> Self:  # type: ignore[valid-type]
        """See :meth:`torch.nn.Module.float`."""
        self.__update_properties(dtype=torch.float)
        return super().float()

    def double(self) -> Self:  # type: ignore[valid-type]
        """See :meth:`torch.nn.Module.double`."""
        self.__update_properties(dtype=torch.double)
        return super().double()

    def half(self) -> Self:  # type: ignore[valid-type]
        """See :meth:`torch.nn.Module.half`."""
        self.__update_properties(dtype=torch.half)
        return super().half()

    def __update_properties(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        def apply_fn(module: Union[DeviceDtypeModuleMixin, Module]) -> None:
            if not isinstance(module, DeviceDtypeModuleMixin):
                return
            if device is not None:
                module._device = device
            if dtype is not None:
                module._dtype = dtype

        self.apply(apply_fn)
