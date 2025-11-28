from functools import wraps
from pandas import DataFrame, Series
from typing import Callable, Dict, Union, Any, List, Tuple

DeriveMap = Dict[str, List[str]]
DeriveFunc = Callable[[DataFrame], Any]


def derive_wrapper(name: str, map: DeriveMap) -> Callable[[DeriveFunc], DeriveFunc]:
    def decorator(fun: DeriveFunc) -> DeriveFunc:
        @wraps(fun)
        def wrapper(df: DataFrame) -> None:
            missing = [col for col in map.keys() if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            df[name] = fun(df)

        setattr(wrapper, 'required_columns', tuple(
            set(tuple(v) for v in map.values()) |
            set(map.keys())
        ))
        setattr(wrapper, 'name', name)
        setattr(wrapper, 'map', map)
        return wrapper
    return decorator


# WARN ONLY BF16
@derive_wrapper("Tensor Flops", {
    # "SQ_INSTS_VALU_MFMA_MOPS_I8": ['sum'],
    # "SQ_INSTS_VALU_MFMA_MOPS_F64": ['sum'],
    # "SQ_INSTS_VALU_MFMA_MOPS_F32": ['sum'],
    "SQ_INSTS_VALU_MFMA_MOPS_BF16": ['sum'],
    # "SQ_INSTS_VALU_MFMA_MOPS_F16": ['sum'],
})
def derive_tensor_flops_rocm(df: DataFrame) -> Series[int]:
    return 512 * (
        # df["SQ_INSTS_VALU_MFMA_MOPS_I8"] +
        # df["SQ_INSTS_VALU_MFMA_MOPS_F64"] +
        # df["SQ_INSTS_VALU_MFMA_MOPS_F32"] +
        df["SQ_INSTS_VALU_MFMA_MOPS_BF16"]
        # + df["SQ_INSTS_VALU_MFMA_MOPS_F16"]
    )


@derive_wrapper("Tensor Util", {
    "SQ_VALU_MFMA_BUSY_CYCLES": ['sum'],
    "GRBM_GUI_ACTIVE": ['sum'],
})
def derive_tensor_util_rocm(df: DataFrame) -> Series[int]:
    n_xcd = 8
    n_cu = 304
    return (
        100 * df["SQ_VALU_MFMA_BUSY_CYCLES"] /
        (n_cu * df["GRBM_GUI_ACTIVE"] / n_xcd * 4)
    )


@derive_wrapper("Cycle Duration", {
    "GRBM_GUI_ACTIVE": ['sum'],
})
def derive_cycle_duration_rocm(df: DataFrame) -> Series[float]:
    n_xcd = 8
    freq = 2100
    return df["GRBM_GUI_ACTIVE"] / n_xcd / freq


@derive_wrapper("Wave Occupancy", {
    "MeanOccupancyPerCU": ["GRBM_GUI_ACTIVE"],
})
def derive_wave_occupancy_rocm(df: DataFrame) -> Series[float]:
    max_waves_per_cu = 32
    return df["MeanOccupancyPerCU"] / max_waves_per_cu * 100


@derive_wrapper("LDS Bytes", {
    "SQ_LDS_IDX_ACTIVE": ['sum'],
    "SQ_LDS_BANK_CONFLICT": ['sum'],
})
def derive_lds_bytes_rocm(df: DataFrame) -> Series[int]:
    lds_banks_per_cu = 32
    return (((
            df["SQ_LDS_IDX_ACTIVE"] -
            df["SQ_LDS_BANK_CONFLICT"]
            ) * 4) * lds_banks_per_cu)


@derive_wrapper("LDS % of Peak BW", {
    "SQ_LDS_IDX_ACTIVE": ['sum'],
    "SQ_LDS_BANK_CONFLICT": ['sum'],
    "dur": ['sum'],
})
def derive_lds_pop_bw_rocm(df: DataFrame) -> Series[float]:
    lds_banks_per_cu = 32
    cu_per_gpu = 304
    max_sclk = 2100e6

    b = (
        df["SQ_LDS_IDX_ACTIVE"] -
        df["SQ_LDS_BANK_CONFLICT"]
    ) * 4 * lds_banks_per_cu
    bw = b / (df["dur"]*1e-9)
    peak_bw = (max_sclk * cu_per_gpu) * 4 * lds_banks_per_cu
    return bw / peak_bw * 100


@derive_wrapper("L1 Bytes", {
    "TCP_TOTAL_CACHE_ACCESSES_sum": ['sum'],
})
def derive_l1_bytes_rocm(df: DataFrame) -> Series[int]:
    return df["TCP_TOTAL_CACHE_ACCESSES_sum"] * 128


@derive_wrapper("L1 % of Peak BW", {
    "TCP_TOTAL_CACHE_ACCESSES_sum": ['sum'],
    "dur": ['sum'],
})
def derive_l1_pop_bw_rocm(df: DataFrame) -> Series[float]:
    cu_per_gpu = 304
    max_sclk = 2100e6

    b = df["TCP_TOTAL_CACHE_ACCESSES_sum"].astype(float) * 128
    bw = b / (df["dur"]*1e-9)
    peak_bw = (max_sclk * 128) * cu_per_gpu
    return bw / peak_bw * 100


@derive_wrapper("L2 Bytes", {
    "TCC_REQ_sum": ['sum'],
})
def derive_l2_bytes_rocm(df: DataFrame) -> Series[int]:
    return df["TCC_REQ_sum"] * 128


@derive_wrapper("L2 % of Peak BW", {
    "TCC_REQ_sum": ['sum'],
    "dur": ['sum'],
})
def derive_l2_pop_bw_rocm(df: DataFrame) -> Series[float]:
    n_xcd = 8
    l2_banks = 16
    total_l2_chan = n_xcd * l2_banks
    max_sclk = 2100e6

    b = df["TCC_REQ_sum"] * 128
    bw = b / (df["dur"]*1e-9)
    peak_bw = max_sclk * 128 * total_l2_chan

    return bw / peak_bw * 100


@derive_wrapper("HBM Bytes", {
    "TCC_BUBBLE_sum": ['sum'],
    "TCC_EA0_RDREQ_32B_sum": ['sum'],
    "TCC_EA0_RDREQ_sum": ['sum'],
    "TCC_EA0_WRREQ_64B_sum": ['sum'],
    "TCC_EA0_WRREQ_sum": ['sum'],
})
def derive_hbm_bytes_rocm(df: DataFrame) -> Series[int]:
    read_bytes = (
        128 * df["TCC_BUBBLE_sum"] +
        64 * (
            df["TCC_EA0_RDREQ_sum"] -
            df["TCC_BUBBLE_sum"] -
            df["TCC_EA0_RDREQ_32B_sum"]
        ) +
        32 * df["TCC_EA0_RDREQ_32B_sum"])
    write_bytes = (
        (df["TCC_EA0_WRREQ_64B_sum"] * 64) +
        ((df["TCC_EA0_WRREQ_sum"] - df["TCC_EA0_WRREQ_64B_sum"]) * 32)
    )
    return read_bytes + write_bytes


@derive_wrapper("HBM % of Peak BW", {
    "TCC_BUBBLE_sum": ['sum'],
    "TCC_EA0_RDREQ_32B_sum": ['sum'],
    "TCC_EA0_RDREQ_sum": ['sum'],
    "TCC_EA0_WRREQ_64B_sum": ['sum'],
    "TCC_EA0_WRREQ_sum": ['sum'],
    "dur": ['sum'],
})
def derive_hbm_pop_bw_rocm(df: DataFrame) -> Series[float]:
    hbm_channels = 128
    max_mclk = 1300*1e6
    hbm_bw = max_mclk * 32 * hbm_channels

    rb = (
        128 * df["TCC_BUBBLE_sum"] +
        64 * (
            df["TCC_EA0_RDREQ_sum"] -
            df["TCC_BUBBLE_sum"] -
            df["TCC_EA0_RDREQ_32B_sum"]
        ) +
        32 * df["TCC_EA0_RDREQ_32B_sum"])
    wb = (
        (df["TCC_EA0_WRREQ_64B_sum"] * 64) +
        ((df["TCC_EA0_WRREQ_sum"] - df["TCC_EA0_WRREQ_64B_sum"]) * 32)
    )
    b = rb + wb
    bw = b / (df["dur"]*1e-9)

    return bw / hbm_bw * 100


@derive_wrapper("L1 Hitrate", {
    "TCP_TCC_READ_REQ_sum": ['sum'],
    "TCP_TCC_WRITE_REQ_sum": ['sum'],
    "TCP_TCC_ATOMIC_WITH_RET_REQ_sum": ['sum'],
    "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum": ['sum'],
    "TCP_TOTAL_CACHE_ACCESSES_sum": ['sum'],
})
def derive_l1_hitrate_rocm(df: DataFrame) -> Series[float]:
    return 100 - ((100 * (df["TCP_TCC_READ_REQ_sum"] + df["TCP_TCC_WRITE_REQ_sum"]
                          + df["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"]
                          + df["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"]))
                  / df["TCP_TOTAL_CACHE_ACCESSES_sum"])


@derive_wrapper("L2 Hitrate", {
    "TCC_HIT_sum": ['sum'],
    "TCC_MISS_sum": ['sum'],
})
def derive_l2_hitrate_rocm(df: DataFrame) -> Series[float]:
    hits = df["TCC_HIT_sum"]
    misses = df["TCC_MISS_sum"]
    return 100 * hits / (hits + misses)
