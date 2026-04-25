import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

AVGPOOL3D_CONFIGS = [
    # 3x3x3 kernel, stride 2, padding 1
    ((4, 3, 16, 32, 32), 3, 2, 1, False, True, None),
    # Test count_include_pad=False
    ((4, 3, 16, 32, 32), 3, 2, 1, False, False, None),
    # Non-cubic kernel and stride
    ((8, 16, 14, 28, 28), (3, 4, 5), (1, 1, 2), 1, False, True, None),
    # Test ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, True, True, None),
    # Test divisor_override
    ((1, 1, 7, 7, 7), 2, 1, 0, False, True, 1),
    # Larger case from a typical CNN
    ((1, 64, 28, 56, 56), 3, 2, 1, False, True, None),
    # No padding, count_include_pad=False
    ((2, 8, 8, 16, 16), 2, 2, 0, False, False, None),
    # Non-square padding
    ((2, 8, 16, 16, 20), 2, 2, (1, 0, 1), False, True, None),
]


@pytest.mark.avg_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_avg_pool3d_forward(
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dtype,
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    res_out = flag_gems.avg_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_avg_pool3d_backward(
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dtype,
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    out_grad = torch.randn_like(ref_out, dtype=inp.dtype, device=flag_gems.device)
    ref_out_grad = utils.to_reference(out_grad, True)
    ref_inp_grad = torch.ops.aten.avg_pool3d_backward(
        ref_out_grad,
        ref_inp,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )

    with flag_gems.use_gems():
        res_inp_grad = torch.ops.aten.avg_pool3d_backward(
            out_grad,
            inp,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    utils.gems_assert_close(res_inp_grad, ref_inp_grad, dtype)