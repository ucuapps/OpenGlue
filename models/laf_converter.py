"""Module that implements different strategies for converting Local Affine Frame (LAF)
to side information used in positional encoding by SuperGlue
"""
from typing import Iterable, Optional

import kornia.feature as KF
import torch
from abc import ABC, abstractmethod


class BaseLAFConversionFunction(ABC):
    @abstractmethod
    def __call__(self, lafs: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def side_info_dim(self):
        pass


class LAF2LogScale(BaseLAFConversionFunction):
    def __call__(self, lafs: torch.Tensor) -> torch.Tensor:
        """
        Extract log-scale from LAFs.
        Args:
            lafs: tensor of shape (B, N, 2, 3)

        Returns:
            tensor of shape (B, N, 1)
        """
        return torch.log(KF.laf.get_laf_scale(lafs)).squeeze(-1)

    @property
    def side_info_dim(self):
        return 1


class LAF2SinCosOrientation(BaseLAFConversionFunction):
    def __call__(self, lafs: torch.Tensor) -> torch.Tensor:
        """
        Extract orientation of LAFs anr return sine and cosine of orientation angle.
        Args:
            lafs: tensor of shape (B, N, 2, 3)

        Returns:
            tensor of shape (B, N, 2)
        """
        scale = KF.laf.get_laf_scale(lafs).squeeze(-1)
        return torch.flip(lafs[..., 0, :-1], dims=(-1,)) / scale

    @property
    def side_info_dim(self):
        return 2


class LAF2AffineGeom(BaseLAFConversionFunction):
    def __call__(self, lafs: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized affine geometry from LAFs
        Args:
            lafs: tensor of shape (B, N, 2, 3)

        Returns:
            tensor of shape (B, N, 4)
        """
        scale = KF.laf.get_laf_scale(lafs).squeeze(-1)
        return torch.flatten(lafs[..., :-1], start_dim=2) / scale

    @property
    def side_info_dim(self):
        return 4


class LAFConverter:
    """Class for converting LAFs to geometric side info
     in the format appropriate for SuperGlue"""

    def __init__(self, cvt_funcs: Optional[Iterable[BaseLAFConversionFunction]] = None):
        """
        Initialize LAFConverter object
        Args:
            cvt_funcs: container of functions used independently to transform LAFs to side information
        """
        self.cvt_funcs = cvt_funcs

    def __call__(self, lafs: torch.Tensor) -> torch.Tensor:
        """
        Transform LAFs to side infor with each function independently and concatenate the result.
        Args:
            lafs: tensor of shape (B, N, 2, 3)
        Returns:
            tensor of shape (B, N, *), where last dimension is sum of shapes returned by individual functions
        """
        if self.cvt_funcs is None:
            B, N, _, _ = lafs.size()
            return lafs.new_empty(B, N, 0)
        return torch.cat([f(lafs) for f in self.cvt_funcs], dim=-1)

    @property
    def side_info_dim(self):
        if self.cvt_funcs is None:
            return 0
        else:
            return sum(f.side_info_dim for f in self.cvt_funcs)


def get_laf_to_sideinfo_converter(method_name: str = 'none') -> LAFConverter:
    """
    Get LAF converter with appropriate transformations given method name.
    Args:
        method_name: name of one of the methods provided by thi module

    Returns:
        callable object for converting lafs to side information
    """
    if method_name.lower() == 'none':
        return LAFConverter()
    elif method_name.lower() == 'rotation':
        return LAFConverter([LAF2SinCosOrientation()])
    elif method_name.lower() == 'scale_rotation':
        return LAFConverter([LAF2LogScale(), LAF2SinCosOrientation()])
    elif method_name.lower() == 'scale':
        return LAFConverter([LAF2LogScale()])
    elif method_name.lower() == 'affine':
        return LAFConverter([LAF2LogScale(), LAF2AffineGeom()])
    else:
        raise NameError('Unexpected name for the method: {}'.format(method_name))
