#!/usr/bin/env python3

import pandas as pd
import numpy as np
from math import ceil


def derive_duration(df: pd.DataFrame) -> pd.DataFrame:
    assert "dur" in df.columns
    df['Duration'] = df['dur'].astype(float)


def derive_kern_duration(df: pd.DataFrame) -> pd.DataFrame:
    assert "Kernel_Duration" in df.columns
    df['Kernel Duration'] = df['Kernel_Duration'].astype(float)


def derive_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    assert "dur" in df.columns and "timestamp_first" in df.columns and "timestamp_last" in df.columns
    elapsed = (df.get("timestamp_last").astype(float) +
               df.get("duration_last").astype(float) - df.get('timestamp_first').astype(float))
    print(elapsed.index)
    df['Elapsed'] = elapsed


def derive_my_mfma_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_VALU_MFMA_BUSY_CYCLES" in df.columns and "compute_unit_busy" in df.columns
    df["my_mfma_util"] = (
        100 * df.get("SQ_VALU_MFMA_BUSY_CYCLES") /
        (df.get("compute_unit_busy") * 4)
    )


def derive_mfma_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_VALU_MFMA_BUSY_CYCLES" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    n_xcd = 8
    n_cu = 304
    df["MFMA Util"] = (
        100 * df.get("SQ_VALU_MFMA_BUSY_CYCLES") /
        (n_cu * df.get("GRBM_GUI_ACTIVE") / n_xcd * 4)
    )


def derive_lds_stalled_perc(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_LDS_UNALIGNED_STALL' in df.columns and 'SQ_LDS_ADDR_CONFLICT' in df.columns and 'SQ_LDS_BANK_CONFLICT' in df.columns and 'SQ_LDS_IDX_ACTIVE' in df.columns
    value = (
        (df.get("SQ_LDS_BANK_CONFLICT") * 3.125) /
        (df.get("SQ_LDS_IDX_ACTIVE") - df.get("SQ_LDS_BANK_CONFLICT"))
    )
    df["LDS Stalled"] = value

    # stall_cycles = (
    #     df.get('SQ_LDS_UNALIGNED_STALL') + df.get('SQ_LDS_ADDR_CONFLICT') +
    #     df.get('SQ_LDS_BANK_CONFLICT')
    # )
    # total_cycles = stall_cycles + df.get('SQ_LDS_IDX_ACTIVE')
    # df["LDS Stalled"] = 100 * (
    #     stall_cycles /
    #     total_cycles
    # )


def derive_lds_bank_conflict_perc(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_LDS_BANK_CONFLICT' in df.columns and 'SQ_LDS_IDX_ACTIVE' in df.columns
    value = 100 * (
        (df.get('SQ_LDS_IDX_ACTIVE') - df.get('SQ_LDS_BANK_CONFLICT')) /
        df.get('SQ_LDS_IDX_ACTIVE')
    )
    # value = (df.get("SQ_LDS_BANK_CONFLICT") * 3.125) / \
    #     (df.get("SQ_LDS_IDX_ACTIVE") - df.get("SQ_LDS_BANK_CONFLICT"))
    df["LDS Bank Conflicts/Access"] = value


def derive_lds_bw_pop(df: pd.DataFrame) -> pd.DataFrame:
    lds_banks_per_cu = 32
    cu_per_gpu = 304
    max_sclk = 2100
    if "SQ_LDS_IDX_ACTIVE" in df.columns and "SQ_LDS_BANK_CONFLICT" in df.columns and "dur" in df.columns:
        value = 100 * ((
            (df.get("SQ_LDS_IDX_ACTIVE") - df.get("SQ_LDS_BANK_CONFLICT")
             ) * 4) * lds_banks_per_cu) / (df.get("dur").astype(float)*1e3)
        peak = (max_sclk * cu_per_gpu) * 0.128
        df["LDS Bandwidth Percent of Peak"] = value/peak


def derive_lds_bw(df: pd.DataFrame) -> pd.DataFrame:
    lds_banks_per_cu = 32
    if "SQ_LDS_IDX_ACTIVE" in df.columns and "SQ_LDS_BANK_CONFLICT" in df.columns and "dur" in df.columns:
        value = ((
            (df.get("SQ_LDS_IDX_ACTIVE") - df.get("SQ_LDS_BANK_CONFLICT")
             ) * 4) * lds_banks_per_cu) / (df.get("dur").astype(float)*1e-6)
        df["LDS Bandwidth"] = value


def derive_lds_bytes(df: pd.DataFrame) -> pd.DataFrame:
    lds_banks_per_cu = 32
    if "SQ_LDS_IDX_ACTIVE" in df.columns and "SQ_LDS_BANK_CONFLICT" in df.columns and "dur" in df.columns:
        value = ((
            (df.get("SQ_LDS_IDX_ACTIVE") - df.get("SQ_LDS_BANK_CONFLICT")
             ) * 4) * lds_banks_per_cu)
        df["LDS Bytes"] = value


def derive_active_cu(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    max_waves_per_cu = 32
    assert 'SQ_BUSY_CU_CYCLES' in df.columns and 'GRBM_GUI_ACTIVE' in df.columns
    sequencer_cycles = np.floor(
        (4 * df.get("SQ_BUSY_CU_CYCLES")) /
        (df.get("GRBM_GUI_ACTIVE")/n_xcd)
    )
    df['Active CUs'] = np.minimum(
        (sequencer_cycles / max_waves_per_cu * 8) +
        np.minimum(sequencer_cycles % max_waves_per_cu, 8),
        cu_per_gpu).astype(int)


def derive_reported_duration(df: pd.DataFrame) -> pd.DataFrame:
    assert 'GRBM_GUI_ACTIVE' in df.columns
    n_xcd = 8
    freq = 2100
    df['Reported Duration'] = df.get("GRBM_GUI_ACTIVE") / n_xcd / freq


def derive_reported_duration_diff(df: pd.DataFrame) -> pd.DataFrame:
    assert 'GRBM_GUI_ACTIVE' in df.columns
    n_xcd = 8
    freq = 2100
    reported_duration = df.get("GRBM_GUI_ACTIVE") / n_xcd / freq
    actual_duration = df.get("dur").astype(float)
    df['Reported Duration Difference'] = (
        100 * (actual_duration-reported_duration)/actual_duration
    )


def derive_wavefront_occupancy_pop(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    max_waves_per_cu = 32
    assert 'SQ_LEVEL_WAVES' in df.columns and 'GRBM_GUI_ACTIVE' in df.columns
    # sequencer_cycles = np.floor(
    #     (4 * df.get("SQ_BUSY_CU_CYCLES")) /
    #     (df.get("GRBM_GUI_ACTIVE")/n_xcd)
    # )
    # active_cus = np.minimum(
    #     (sequencer_cycles / max_waves_per_cu * 8) +
    #     np.minimum(sequencer_cycles % max_waves_per_cu, 8),
    #     cu_per_gpu).astype(int)
    cu_per_gpu = 304
    wavefronts = (
        df['SQ_LEVEL_WAVES'] / df['GRBM_GUI_ACTIVE'] / n_xcd
    )
    value = 100*wavefronts / (max_waves_per_cu * cu_per_gpu)
    df["Wavefront Occupancy"] = value


# def derive_LDS_bank_conflicts_to_access_pop(df: pd.DataFrame) -> pd.DataFrame:
#     if "SQ_LDS_IDX_ACTIVE" in df.columns and "SQ_LDS_BANK_CONFLICT" in df.columns:
#         diff = (df.get("SQ_LDS_IDX_ACTIVE") - df.get("SQ_LDS_BANK_CONFLICT"))
#         value = df.get("SQ_LDS_BANK_CONFLICT") / diff
#         peak = 32
#         df["LDS_bank_conflicts_to_access_pop"] = 100 * value / peak
#         df.drop(
#             columns=[
#                 "SQ_LDS_IDX_ACTIVE",
#                 "SQ_LDS_BANK_CONFLICT",
#             ],
#             inplace=True,
#         )

#     return df


def derive_lds_util(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "SQ_LDS_IDX_ACTIVE" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    df["LDS Util"] = (
        100 * df.get("SQ_LDS_IDX_ACTIVE") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))


def derive_lds_access_rate(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "SQ_ACTIVE_INST_LDS" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    # TODO check the 200 it doesn't quite make sense
    df["LDS Access Rate"] = (
        200 * df.get("SQ_ACTIVE_INST_LDS") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))


def derive_lds_latency(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_LDS' in df.columns and 'SQ_INST_LEVEL_LDS' in df.columns
    value = df.get("SQ_INST_LEVEL_LDS") / df.get("SQ_INSTS_LDS")
    df["LDS Latency"] = value


def derive_vmem_latency(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VMEM' in df.columns and 'SQ_INST_LEVEL_VMEM' in df.columns
    value = df.get("SQ_INST_LEVEL_VMEM") / df.get("SQ_INSTS_VMEM")
    df["VMEM Latency"] = value


def derive_l1d_addr_hitrate(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_UTCL1_TRANSLATION_HIT_sum" in df.columns and "TCP_UTCL1_REQUEST_sum" in df.columns
    value = (100*df.get("TCP_UTCL1_TRANSLATION_HIT_sum") /
             df.get("TCP_UTCL1_REQUEST_sum"))
    df["L1D Address Translation Hit Rate"] = value


def derive_vl1d_hit_rate(df: pd.DataFrame) -> pd.DataFrame:
    # test1 = df.get("TCP_TCC_READ_REQ_sum")
    # test2 = df.get("TCP_TCC_WRITE_REQ_sum")
    # test3 = df.get("TCP_TCC_ATOMIC_WITH_RET_REQ_sum")
    # test4 = df.get("TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum")
    # test5 = df.get("TCP_TOTAL_CACHE_ACCESSES_sum")
    # res = (test1+test2+test3+test4)
    # assert np.all(res <= test5)

    assert "TCP_TCC_READ_REQ_sum" in df.columns and "TCP_TCC_WRITE_REQ_sum" in df.columns and "TCP_TCC_ATOMIC_WITH_RET_REQ_sum" in df.columns and "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum" in df.columns and "TCP_TOTAL_CACHE_ACCESSES_sum" in df.columns
    value = ((100 - ((100 * (((df.get("TCP_TCC_READ_REQ_sum") + df.get("TCP_TCC_WRITE_REQ_sum"))
              + df.get("TCP_TCC_ATOMIC_WITH_RET_REQ_sum")) + df.get("TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum")))
             / df.get("TCP_TOTAL_CACHE_ACCESSES_sum"))))
    # mask = value < 0
    # print(value[mask])
    df["vL1D Hit Rate"] = value


def derive_vl1d_bytes(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_TOTAL_CACHE_ACCESSES_sum" in df.columns and "dur" in df.columns
    value = df.get("TCP_TOTAL_CACHE_ACCESSES_sum").astype(float) * 128
    df["vL1D Bytes"] = value


def derive_vl1d_bandwidth(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_TOTAL_CACHE_ACCESSES_sum" in df.columns and "dur" in df.columns
    value = (df.get("TCP_TOTAL_CACHE_ACCESSES_sum").astype(float) *
             128 / (df.get("dur").astype(float) * 1e-6))
    df["vL1D Bandwidth"] = value


def derive_vl1d_bandwidth_pop(df: pd.DataFrame) -> pd.DataFrame:
    cu_per_gpu = 304
    max_sclk = 2100
    assert "TCP_TOTAL_CACHE_ACCESSES_sum" in df.columns and "dur" in df.columns
    value = (df.get("TCP_TOTAL_CACHE_ACCESSES_sum").astype(float) *
             128 / (df.get("dur").astype(float) * 1e3))
    peak = ((max_sclk / 1000) * 128) * cu_per_gpu
    df["vL1D Bandwidth Percent of Peak"] = value/peak*100


def derive_vl1d_requests_to_transactions(df: pd.DataFrame) -> pd.DataFrame:
    assert 'TCP_TCC_READ_REQ_sum' in df.columns


def derive_vl1d_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_GATE_EN2_sum" in df.columns and "TCP_GATE_EN1_sum" in df.columns
    value = df.get("TCP_GATE_EN2_sum") * 100 / df.get("TCP_GATE_EN1_sum")
    df["vL1D Util"] = value


def derive_l2_data_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_PENDING_STALL_CYCLES_sum" in df.columns and "TCP_GATE_EN1_sum" in df.columns
    value = (100 * df.get("TCP_PENDING_STALL_CYCLES_sum") /
             df.get("TCP_GATE_EN1_sum"))
    df["L2->vL1D Data Stall"] = value


def derive_l2_req_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_TCR_TCP_STALL_CYCLES_sum" in df.columns and "TCP_GATE_EN1_sum" in df.columns
    value = (100 * df.get("TCP_TCR_TCP_STALL_CYCLES_sum") /
             df.get("TCP_GATE_EN1_sum"))
    df["L2->vL1D Req Stall"] = value


def derive_l2_tag_ram_read_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_READ_TAGCONFLICT_STALL_CYCLES_sum" in df.columns and "TCP_GATE_EN1_sum" in df.columns
    value = (100 * df.get("TCP_READ_TAGCONFLICT_STALL_CYCLES_sum") /
             df.get("TCP_GATE_EN1_sum"))
    df["L2->vL1D Tag RAM Read Stall"] = value


def derive_l2_tag_ram_write_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_WRITE_TAGCONFLICT_STALL_CYCLES_sum" in df.columns and "TCP_GATE_EN1_sum" in df.columns
    value = (100 * df.get("TCP_WRITE_TAGCONFLICT_STALL_CYCLES_sum") /
             df.get("TCP_GATE_EN1_sum"))
    df["L2->vL1D Tag RAM Write Stall"] = value


def derive_l2_tag_ram_atomic_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_ATOMIC_TAGCONFLICT_STALL_CYCLES_sum" in df.columns and "TCP_GATE_EN1_sum" in df.columns
    value = (100 * df.get("TCP_ATOMIC_TAGCONFLICT_STALL_CYCLES_sum") /
             df.get("TCP_GATE_EN1_sum"))
    df["L2->vL1D Tag RAM Atomic Stall"] = value


def derive_ta_busy(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "TA_TA_BUSY_sum" in df.columns
    value = (100 * df.get("TA_TA_BUSY_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))
    df["TA Busy"] = value


def derive_ta_address_stall(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "TA_ADDR_STALLED_BY_TC_CYCLES_sum" in df.columns
    value = (100 * df.get("TA_ADDR_STALLED_BY_TC_CYCLES_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))
    df["TA->vL1D address stall"] = value


def derive_ta_data_stall(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "TA_DATA_STALLED_BY_TC_CYCLES_sum" in df.columns
    value = (100 * df.get("TA_DATA_STALLED_BY_TC_CYCLES_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))
    df["TA->vL1D data stall"] = value


def derive_ta_dp_stall(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "TA_ADDR_STALLED_BY_TD_CYCLES_sum" in df.columns
    value = (100 * df.get("TA_ADDR_STALLED_BY_TD_CYCLES_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))
    df["TA->TD stall"] = value


def derive_data_return_busy(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "TD_TD_BUSY_sum" in df.columns
    value = (100 * df.get("TD_TD_BUSY_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))
    df["TD busy"] = value


def derive_vl1d_data_return_stall(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "TD_TC_STALL_sum" in df.columns
    value = (100 * df.get("TD_TC_STALL_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))
    df["vL1D->TD stall"] = value


def derive_workgroup_manager_stall(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "TD_SPI_STALL_sum" in df.columns
    value = (100 * df.get("TD_SPI_STALL_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu))
    df["Workgroup Manager->TD stall"] = value


def derive_l2_hit_rate(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_HIT_sum" in df.columns and "TCC_MISS_sum" in df.columns
    hits = df.get("TCC_HIT_sum")
    misses = df.get("TCC_MISS_sum")
    value = 100 * hits / (hits + misses)
    df["L2 Hit Rate"] = value


def derive_l2_cache_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_BUSY_sum" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    n_xcd = 8
    l2_banks = 16
    total_l2_chan = n_xcd * l2_banks
    value = (100 * df.get("TCC_BUSY_sum") /
             (df.get("GRBM_GUI_ACTIVE") / n_xcd * total_l2_chan))
    df["L2 Util"] = value


def derive_l2_cache_my_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_BUSY_sum" in df.columns and "TCC_CYCLE_sum" in df.columns
    value = (100 * df.get("TCC_BUSY_sum") /
             (df.get("TCC_CYCLE_sum")))
    df["My L2 Util"] = value


def derive_l2_cache_bytes(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_REQ_sum" in df.columns and "dur" in df.columns
    value = df.get("TCC_REQ_sum") * 128
    df["L2 Bytes"] = value


def derive_l2_cache_bw(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_REQ_sum" in df.columns and "dur" in df.columns
    value = (
        df.get("TCC_REQ_sum") * 128 /
        (df.get("dur").astype(float) * 1e-6)
    )
    df["L2 Bandwidth"] = value


def derive_l2_cache_bw_pop(df: pd.DataFrame) -> pd.DataFrame:
    max_sclk = 2100
    n_xcd = 8
    l2_banks = 16
    total_l2_chan = n_xcd * l2_banks

    assert "TCC_REQ_sum" in df.columns and "dur" in df.columns
    value = (
        df.get("TCC_REQ_sum") * 128 /
        (df.get("dur").astype(float) * 1e3)
    )
    peak = (max_sclk / 1000) * 128 * total_l2_chan

    df["L2 Bandwidth Percent of Peak"] = 100*value/peak


def derive_l2_fabric_read_bw(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_BUBBLE_sum" in df.columns and "TCC_EA0_RDREQ_32B_sum" in df.columns and "TCC_EA0_RDREQ_sum" in df.columns and "dur" in df.columns
    read_bytes = (128 * df.get("TCC_BUBBLE_sum") +
                  64 * (df.get("TCC_EA0_RDREQ_sum") - df.get("TCC_BUBBLE_sum") - df.get("TCC_EA0_RDREQ_32B_sum")) +
                  32 * df.get("TCC_EA0_RDREQ_32B_sum"))
    value = read_bytes / (df.get("dur").astype(float) * 1e3)
    df["L2 Fabric Read Bandwidth"] = value


def derive_l2_fabric_write_bw(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_EA0_WRREQ_64B_sum" in df.columns and "TCC_EA0_WRREQ_sum" in df.columns and "dur" in df.columns
    write_bytes = (
        (df.get("TCC_EA0_WRREQ_64B_sum") * 64) +
        ((df.get("TCC_EA0_WRREQ_sum") - df.get("TCC_EA0_WRREQ_64B_sum")) * 32)
    )
    value = write_bytes / (df.get("dur").astype(float)*1e3)
    df["L2 Fabric Write&Atomic Bandwidth"] = value


def derive_l2_fabric_read_bw_pop(df: pd.DataFrame) -> pd.DataFrame:
    hbm_channels = 128
    max_mclk = 1300
    hbm_bw = max_mclk / 1000 * 32 * hbm_channels
    assert "TCC_BUBBLE_sum" in df.columns and "TCC_EA0_RDREQ_32B_sum" in df.columns and "TCC_EA0_RDREQ_sum" in df.columns and "dur" in df.columns
    read_bytes = (128 * df.get("TCC_BUBBLE_sum") +
                  64 * (df.get("TCC_EA0_RDREQ_sum") - df.get("TCC_BUBBLE_sum") - df.get("TCC_EA0_RDREQ_32B_sum")) +
                  32 * df.get("TCC_EA0_RDREQ_32B_sum"))
    value = read_bytes / (df.get("dur").astype(float) * 1e3)
    df["L2 Fabric Read Bandwidth Percent of Peak"] = value / hbm_bw * 100


def derive_l2_fabric_write_bw_pop(df: pd.DataFrame) -> pd.DataFrame:
    hbm_channels = 128
    max_mclk = 1300
    hbm_bw = max_mclk / 1000 * 32 * hbm_channels
    assert "TCC_EA0_WRREQ_64B_sum" in df.columns and "TCC_EA0_WRREQ_sum" in df.columns and "dur" in df.columns
    write_bytes = (
        (df.get("TCC_EA0_WRREQ_64B_sum") * 64) +
        ((df.get("TCC_EA0_WRREQ_sum") - df.get("TCC_EA0_WRREQ_64B_sum")) * 32)
    )
    value = write_bytes / (df.get("dur").astype(float)*1e3)
    df["L2 Fabric Write&Atomic Bandwidth Percent of Peak"] = value / hbm_bw * 100


def l2_fabric_read_latency(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_EA0_RDREQ_LEVEL_sum" in df.columns and "TCC_EA0_RDREQ_sum" in df.columns
    df["l2_fabric_read_latency"] = (
        df.get("TCC_EA0_RDREQ_LEVEL_sum") /
        df.get("TCC_EA0_RDREQ_sum")
    )


def l2_fabric_write_latency(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_EA0_WRREQ_LEVEL_sum" in df.columns and "TCC_EA0_WRREQ_sum" in df.columns
    df["l2_fabric_write_latency"] = (
        df.get("TCC_EA0_WRREQ_LEVEL_sum") /
        df.get("TCC_EA0_WRREQ_sum")
    )


def derive_sl1d_cache_hit_rate(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQC_DCACHE_HITS" in df.columns and "SQC_DCACHE_MISSES" in df.columns and "SQC_DCACHE_MISSES_DUPLICATE" in df.columns
    hits = df.get("SQC_DCACHE_HITS")
    misses = (df.get("SQC_DCACHE_MISSES") +
              df.get("SQC_DCACHE_MISSES_DUPLICATE"))
    value = 100 * hits / (hits + misses)
    df["sL1D Hit Rate"] = value


def derive_sl1d_cache_bw_pop(df: pd.DataFrame) -> pd.DataFrame:
    max_sclk = 2100
    se_per_gpu = 32
    n_cu = 304
    cu_per_se = float(n_cu) / float(se_per_gpu)
    sq_per_se = cu_per_se / 2
    sq_per_se = ceil(sq_per_se)
    sqc_per_gpu = int(sq_per_se) * se_per_gpu

    assert "dur" in df.columns and "SQC_DCACHE_REQ" in df.columns
    value = (df.get("SQC_DCACHE_REQ") * 100000 / (max_sclk *
             sqc_per_gpu * (df.get("dur").astype(float) * 1e3)))
    df["sL1D Cache Bandwidth Percent of Peak"] = value


def derive_sl1d_l2_bw_pop(df: pd.DataFrame) -> pd.DataFrame:
    max_sclk = 2100
    se_per_gpu = 32
    n_cu = 304
    cu_per_se = float(n_cu) / float(se_per_gpu)
    sq_per_se = cu_per_se / 2
    sq_per_se = ceil(sq_per_se)
    sqc_per_gpu = int(sq_per_se) * se_per_gpu

    assert "SQC_TC_DATA_READ_REQ" in df.columns and "SQC_TC_DATA_WRITE_REQ" in df.columns and "SQC_TC_DATA_ATOMIC_REQ" in df.columns
    value = ((
        df.get("SQC_TC_DATA_READ_REQ") +
        df.get("SQC_TC_DATA_WRITE_REQ") +
        df.get("SQC_TC_DATA_ATOMIC_REQ")) * 100000 /
        (2 * max_sclk * sqc_per_gpu * (df.get("SQC_DCACHE_REQ").astype(float) * 1e3)))
    df["sL1D->L2 Bandwidth Percent of Peak"] = value


def derive_l1i_cache_hit_rate(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQC_ICACHE_MISSES" in df.columns and "SQC_ICACHE_MISSES_DUPLICATE" in df.columns and "SQC_ICACHE_HITS" in df.columns
    hits = df.get("SQC_ICACHE_HITS")
    misses = (df.get("SQC_ICACHE_MISSES") +
              df.get("SQC_ICACHE_MISSES_DUPLICATE"))
    value = 100 * hits / (hits + misses)
    df["L1I Hit Rate"] = value


def derive_l1i_l2_bw_pop(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQC_TC_INST_REQ" in df.columns and "dur" in df.columns
    max_sclk = 2100
    se_per_gpu = 32
    n_cu = 304
    cu_per_se = float(n_cu) / float(se_per_gpu)
    sq_per_se = cu_per_se / 2
    sq_per_se = ceil(sq_per_se)
    sqc_per_gpu = int(sq_per_se) * se_per_gpu

    value = (
        (df.get("SQC_TC_INST_REQ") * 100000) /
        (2 * max_sclk * sqc_per_gpu * df.get("dur").astype(float)*1e3)
    )
    df["L1I->L2 Bandwidth Percent of Peak"] = value


def derive_inffab_bytes(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_BUBBLE_sum" in df.columns and "TCC_EA0_RDREQ_32B_sum" in df.columns and "TCC_EA0_RDREQ_sum" in df.columns and "TCC_EA0_WRREQ_64B_sum" in df.columns and "TCC_EA0_WRREQ_sum" in df.columns
    read_bytes = (128 * df.get("TCC_BUBBLE_sum") +
                  64 * (df.get("TCC_EA0_RDREQ_sum") - df.get("TCC_BUBBLE_sum") - df.get("TCC_EA0_RDREQ_32B_sum")) +
                  32 * df.get("TCC_EA0_RDREQ_32B_sum"))
    write_bytes = (
        (df.get("TCC_EA0_WRREQ_64B_sum") * 64) +
        ((df.get("TCC_EA0_WRREQ_sum") - df.get("TCC_EA0_WRREQ_64B_sum")) * 32)
    )
    b = read_bytes + write_bytes
    df["InfFab Bytes"] = b


def derive_inffab_32b_req(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_EA0_RDREQ_32B_sum" in df.columns and "TCC_EA0_WRREQ_64B_sum" in df.columns and "TCC_EA0_WRREQ_sum" in df.columns
    read_req = df.get("TCC_EA0_RDREQ_32B_sum")
    write_req = df.get("TCC_EA0_WRREQ_sum") - df.get("TCC_EA0_WRREQ_64B_sum")
    df["InfFab 32B Requests"] = read_req + write_req


def derive_inffab_64b_req(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_BUBBLE_sum" in df.columns and "TCC_EA0_RDREQ_32B_sum" in df.columns and "TCC_EA0_RDREQ_sum" in df.columns and "TCC_EA0_WRREQ_64B_sum" in df.columns and "TCC_EA0_WRREQ_sum" in df.columns
    read_req = (df.get("TCC_EA0_RDREQ_sum") -
                df.get("TCC_BUBBLE_sum") - df.get("TCC_EA0_RDREQ_32B_sum"))
    write_req = df.get("TCC_EA0_WRREQ_64B_sum")
    df["InfFab 64B Requests"] = read_req + write_req


def derive_inffab_128b_req(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_BUBBLE_sum" in df.columns
    b = df.get("TCC_BUBBLE_sum")
    df["InfFab 128B Requests"] = b


def derive_l2_streaming_req(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_STREAMING_REQ_sum" in df.columns
    df["L2 Streaming Req"] = df.get("TCC_STREAMING_REQ_sum")


def derive_inffab_bw(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_BUBBLE_sum" in df.columns and "TCC_EA0_RDREQ_32B_sum" in df.columns and "TCC_EA0_RDREQ_sum" in df.columns and "TCC_EA0_WRREQ_64B_sum" in df.columns and "TCC_EA0_WRREQ_sum" in df.columns
    read_bytes = (128 * df.get("TCC_BUBBLE_sum") +
                  64 * (df.get("TCC_EA0_RDREQ_sum") - df.get("TCC_BUBBLE_sum") - df.get("TCC_EA0_RDREQ_32B_sum")) +
                  32 * df.get("TCC_EA0_RDREQ_32B_sum"))
    write_bytes = (
        (df.get("TCC_EA0_WRREQ_64B_sum") * 64) +
        ((df.get("TCC_EA0_WRREQ_sum") - df.get("TCC_EA0_WRREQ_64B_sum")) * 32)
    )
    b = read_bytes + write_bytes
    df["InfFab Bandwidth"] = b / (df.get("dur").astype(float)*1e-6)


def derive_ea_uncached_bytes(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_EA0_WR_UNCACHED_32B_sum" in df.columns and "TCC_EA0_RD_UNCACHED_32B_sum" in df.columns
    b = (df.get("TCC_EA0_WR_UNCACHED_32B_sum") +
         df.get("TCC_EA0_RD_UNCACHED_32B_sum")) * 32
    df["EA Uncached Bytes"] = b


def derive_salu_util(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "SQ_ACTIVE_INST_SCA" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    value = ((100 * df.get("SQ_ACTIVE_INST_SCA")) /
             (df.get("GRBM_GUI_ACTIVE")/n_xcd * cu_per_gpu))
    df["SALU Util"] = value


def derive_valu_f32_flops_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F32" in df.columns and "SQ_INSTS_VALU_MUL_F32" in df.columns and "SQ_INSTS_VALU_FMA_F32" in df.columns and "SQ_INSTS_VALU_TRANS_F32" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    flops = 64 * (
        df["SQ_INSTS_VALU_ADD_F32"]
        + df["SQ_INSTS_VALU_MUL_F32"]
        + (2 * df["SQ_INSTS_VALU_FMA_F32"])
        + df["SQ_INSTS_VALU_TRANS_F32"]
    )
    n_xcd = 8
    cu_per_gpu = 304
    cycles = (df["GRBM_GUI_ACTIVE"]/n_xcd * cu_per_gpu)
    # slide 5 col 1
    # https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf?ref=ghost.twave.zone
    thr_flops = 256 * cycles
    df["VALU F32 FLOPs Util"] = 100 * flops / thr_flops


def derive_valu_cycles(df: pd.DataFrame) -> pd.DataFrame:
    cu_per_gpu = 304
    assert "SQ_ACTIVE_INST_VALU" in df.columns
    df["VALU Cycles"] = df.get("SQ_ACTIVE_INST_VALU")/cu_per_gpu


def derive_valu_util(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "SQ_ACTIVE_INST_VALU" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    value = ((100 * df.get("SQ_ACTIVE_INST_VALU")) /
             (df.get("GRBM_GUI_ACTIVE")/n_xcd * cu_per_gpu))

    df["VALU Util"] = value


def derive_vmem_util(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "SQ_ACTIVE_INST_FLAT" in df.columns and "SQ_ACTIVE_INST_VMEM" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    value = ((100 * (df.get("SQ_ACTIVE_INST_FLAT") + df.get("SQ_ACTIVE_INST_VMEM"))) /
             (df.get("GRBM_GUI_ACTIVE")/n_xcd * cu_per_gpu))

    df["VMEM Util"] = value


def derive_branch_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_ACTIVE_INST_MISC" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    n_xcd = 8
    cu_per_gpu = 304
    value = ((100 * df.get("SQ_ACTIVE_INST_MISC")) /
             (df.get("GRBM_GUI_ACTIVE")/n_xcd * cu_per_gpu))

    df["Branch Util"] = value


def derive_salu_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_SALU" in df.columns
    value = df.get("SQ_INSTS_SALU")
    df["SALU Issued"] = value


def derive_smem_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_SMEM" in df.columns
    value = df.get("SQ_INSTS_SMEM")
    df["SMEM Issued"] = value


def derive_flat_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_FLAT" in df.columns
    value = df.get("SQ_INSTS_FLAT")
    df["Flat Issued"] = value


def derive_vmem_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VMEM" in df.columns and "SQ_INSTS_LDS" in df.columns
    value = df.get("SQ_INSTS_VMEM") + df.get("SQ_INSTS_LDS")
    df["VMEM Issued"] = value


def derive_branch_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_BRANCH" in df.columns
    value = df.get("SQ_INSTS_BRANCH")
    df["Branch Issued"] = value


def derive_sendmsg_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_SENDMSG" in df.columns
    value = df.get("SQ_INSTS_SENDMSG")
    df["SENDMSG Issued"] = value


# def derive_valu_issued(df: pd.DataFrame) -> pd.DataFrame:
#     n_xcd = 8
#     cu_per_gpu = 304
#     assert "SQ_ACTIVE_INST_VALU" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
#     value = ((100 * df.get("SQ_ACTIVE_INST_VALU")) /
#              (df.get("GRBM_GUI_ACTIVE")/n_xcd * cu_per_gpu))

#     df["VALU Active Util"] = value


def derive_mfma_i8_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_MFMA_I8" in df.columns
    value = df.get("SQ_INSTS_VALU_MFMA_I8")
    df["MFMA I8 Issued"] = value


def derive_mfma_f16_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_MFMA_F16" in df.columns
    value = df.get("SQ_INSTS_VALU_MFMA_F16")
    df["MFMA F16 Issued"] = value


def derive_mfma_f32_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_MFMA_F32" in df.columns
    value = df.get("SQ_INSTS_VALU_MFMA_F32")
    df["MFMA F32 Issued"] = value


def derive_mfma_f64_issued(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_MFMA_F64" in df.columns
    value = df.get("SQ_INSTS_VALU_MFMA_F64")
    df["MFMA F64 Issued"] = value


def derive_mfma_flops_f16(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F16' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_F16"] * 512
    df["MFMA F16 Flops"] = value


def derive_mfma_flops_bf16(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_BF16' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * 512
    df["MFMA BF16 Flops"] = value


def derive_mfma_flops_f32(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F32' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_F32"] * 512
    df["MFMA F32 Flops"] = value


def derive_mfma_flops_f64(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F64' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_F64"] * 512
    df["MFMA F64 Flops"] = value


def derive_mfma_iops_i8(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_I8"] * 512
    df["MFMA I8 IOPs"] = value


def derive_mfma_flops_sec_f16(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F16' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_F16"] * \
        512 / (df["dur"].astype(float)*1e-6)
    df["MFMA F16 FLOPS"] = value


def derive_mfma_flops_sec_bf16(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_BF16' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * \
        512 / (df["dur"].astype(float)*1e-6)
    df["MFMA BF16 FLOPS"] = value


def derive_mfma_flops_bf16_util(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_BF16' in df.columns
    flops = df["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * \
        512 / (df["dur"].astype(float)*1e-6)
    theoretical_flops = 1307.4432e12
    value = 100 * flops / theoretical_flops
    df["MFMA BF16 Util"] = value


def derive_mfma_flops_sec_f32(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F32' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_F32"] * \
        512 / (df["dur"].astype(float)*1e-6)
    df["MFMA F32 FLOPS"] = value


def derive_mfma_flops_sec_f64(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F64' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_F64"] * \
        512 / (df["dur"].astype(float)*1e-6)
    df["MFMA F64 FLOPS"] = value


def derive_mfma_iops_sec_i8(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns
    value = df["SQ_INSTS_VALU_MFMA_MOPS_I8"] * \
        512 / (df["dur"].astype(float)*1e-6)
    df["MFMA I8 IOPS"] = value


def derive_mfma_flops_sec(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F64' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_F32' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_BF16' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_F16' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns
    value = 512 * (
        df["SQ_INSTS_VALU_MFMA_MOPS_I8"] +
        df["SQ_INSTS_VALU_MFMA_MOPS_F64"] +
        df["SQ_INSTS_VALU_MFMA_MOPS_F32"] +
        df["SQ_INSTS_VALU_MFMA_MOPS_BF16"] +
        df["SQ_INSTS_VALU_MFMA_MOPS_F16"]
    ) / (df["dur"].astype(float) * 1e-6)
    df["MFMA FLOPS"] = value


def derive_mfma_flops(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_F64' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_F32' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_BF16' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_F16' in df.columns and 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns
    value = 512 * (
        # df["SQ_INSTS_VALU_MFMA_MOPS_I8"] +
        # df["SQ_INSTS_VALU_MFMA_MOPS_F64"] +
        # df["SQ_INSTS_VALU_MFMA_MOPS_F32"] +
        df["SQ_INSTS_VALU_MFMA_MOPS_BF16"]
        # df["SQ_INSTS_VALU_MFMA_MOPS_F16"]
    )
    df["MFMA Flops"] = value


def derive_valu_flops_f16(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F16" in df.columns and "SQ_INSTS_VALU_MUL_F16" in df.columns and "SQ_INSTS_VALU_FMA_F16" in df.columns and "SQ_INSTS_VALU_TRANS_F16" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F16"]
        + df["SQ_INSTS_VALU_MUL_F16"]
        + (2 * df["SQ_INSTS_VALU_FMA_F16"])
        + df["SQ_INSTS_VALU_TRANS_F16"]
    )
    df["VALU F16 Flops"] = value


def derive_valu_flops_f32(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F32" in df.columns and "SQ_INSTS_VALU_MUL_F32" in df.columns and "SQ_INSTS_VALU_FMA_F32" in df.columns and "SQ_INSTS_VALU_TRANS_F32" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F32"]
        + df["SQ_INSTS_VALU_MUL_F32"]
        + (2 * df["SQ_INSTS_VALU_FMA_F32"])
        + df["SQ_INSTS_VALU_TRANS_F32"]
    )
    df["VALU F32 Flops"] = value


def derive_valu_flops_f64(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F64" in df.columns and "SQ_INSTS_VALU_MUL_F64" in df.columns and "SQ_INSTS_VALU_FMA_F64" in df.columns and "SQ_INSTS_VALU_TRANS_F64" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F64"]
        + df["SQ_INSTS_VALU_MUL_F64"]
        + (2 * df["SQ_INSTS_VALU_FMA_F64"])
        + df["SQ_INSTS_VALU_TRANS_F64"]
    )
    df["VALU F64 Flops"] = value


def derive_valu_flops_sec_f16(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F16" in df.columns and "SQ_INSTS_VALU_MUL_F16" in df.columns and "SQ_INSTS_VALU_FMA_F16" in df.columns and "SQ_INSTS_VALU_TRANS_F16" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F16"]
        + df["SQ_INSTS_VALU_MUL_F16"]
        + (2 * df["SQ_INSTS_VALU_FMA_F16"])
        + df["SQ_INSTS_VALU_TRANS_F16"]
    ) / (df["dur"].astype(float) * 1e-6)
    df["VALU F16 FLOPS"] = value


def derive_valu_flops_sec_f32(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F32" in df.columns and "SQ_INSTS_VALU_MUL_F32" in df.columns and "SQ_INSTS_VALU_FMA_F32" in df.columns and "SQ_INSTS_VALU_TRANS_F32" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F32"]
        + df["SQ_INSTS_VALU_MUL_F32"]
        + (2 * df["SQ_INSTS_VALU_FMA_F32"])
        + df["SQ_INSTS_VALU_TRANS_F32"]
    ) / (df["dur"].astype(float) * 1e-6)
    df["VALU F32 FLOPS"] = value


def derive_valu_flops_sec_f64(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F64" in df.columns and "SQ_INSTS_VALU_MUL_F64" in df.columns and "SQ_INSTS_VALU_FMA_F64" in df.columns and "SQ_INSTS_VALU_TRANS_F64" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F64"]
        + df["SQ_INSTS_VALU_MUL_F64"]
        + (2 * df["SQ_INSTS_VALU_FMA_F64"])
        + df["SQ_INSTS_VALU_TRANS_F64"]
    ) / (df["dur"].astype(float) * 1e-6)
    df["VALU F64 FLOPS"] = value


def derive_valu_flops_sec(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F16" in df.columns and "SQ_INSTS_VALU_MUL_F16" in df.columns and "SQ_INSTS_VALU_FMA_F16" in df.columns and "SQ_INSTS_VALU_TRANS_F16" in df.columns and "SQ_INSTS_VALU_ADD_F32" in df.columns and "SQ_INSTS_VALU_MUL_F32" in df.columns and "SQ_INSTS_VALU_FMA_F32" in df.columns and "SQ_INSTS_VALU_TRANS_F32" in df.columns and "SQ_INSTS_VALU_ADD_F64" in df.columns and "SQ_INSTS_VALU_MUL_F64" in df.columns and "SQ_INSTS_VALU_FMA_F64" in df.columns and "SQ_INSTS_VALU_TRANS_F64" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F64"]
        + df["SQ_INSTS_VALU_MUL_F64"]
        + (2 * df["SQ_INSTS_VALU_FMA_F64"])
        + df["SQ_INSTS_VALU_TRANS_F64"]
        + df["SQ_INSTS_VALU_ADD_F32"]
        + df["SQ_INSTS_VALU_MUL_F32"]
        + (2 * df["SQ_INSTS_VALU_FMA_F32"])
        + df["SQ_INSTS_VALU_TRANS_F32"]
        + df["SQ_INSTS_VALU_ADD_F16"]
        + df["SQ_INSTS_VALU_MUL_F16"]
        + (2 * df["SQ_INSTS_VALU_FMA_F16"])
        + df["SQ_INSTS_VALU_TRANS_F16"]
    ) / (df["dur"].astype(float)*1e-6)
    df["VALU FLOPS"] = value


def derive_valu_flops(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_INSTS_VALU_ADD_F16" in df.columns and "SQ_INSTS_VALU_MUL_F16" in df.columns and "SQ_INSTS_VALU_FMA_F16" in df.columns and "SQ_INSTS_VALU_TRANS_F16" in df.columns and "SQ_INSTS_VALU_ADD_F32" in df.columns and "SQ_INSTS_VALU_MUL_F32" in df.columns and "SQ_INSTS_VALU_FMA_F32" in df.columns and "SQ_INSTS_VALU_TRANS_F32" in df.columns and "SQ_INSTS_VALU_ADD_F64" in df.columns and "SQ_INSTS_VALU_MUL_F64" in df.columns and "SQ_INSTS_VALU_FMA_F64" in df.columns and "SQ_INSTS_VALU_TRANS_F64" in df.columns
    value = 64 * (
        df["SQ_INSTS_VALU_ADD_F64"]
        + df["SQ_INSTS_VALU_MUL_F64"]
        + (2 * df["SQ_INSTS_VALU_FMA_F64"])
        + df["SQ_INSTS_VALU_TRANS_F64"]
        + df["SQ_INSTS_VALU_ADD_F32"]
        + df["SQ_INSTS_VALU_MUL_F32"]
        + (2 * df["SQ_INSTS_VALU_FMA_F32"])
        + df["SQ_INSTS_VALU_TRANS_F32"]
        + df["SQ_INSTS_VALU_ADD_F16"]
        + df["SQ_INSTS_VALU_MUL_F16"]
        + (2 * df["SQ_INSTS_VALU_FMA_F16"])
        + df["SQ_INSTS_VALU_TRANS_F16"]
    )
    df["VALU Flops"] = value


def derive_vmem_active_util(df: pd.DataFrame) -> pd.DataFrame:
    n_xcd = 8
    cu_per_gpu = 304
    assert "SQ_ACTIVE_INST_FLAT" in df.columns and "SQ_ACTIVE_INST_VMEM" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    value = ((100 * (df.get("SQ_ACTIVE_INST_FLAT") + df.get("SQ_ACTIVE_INST_VMEM"))) /
             (df.get("GRBM_GUI_ACTIVE")/n_xcd * cu_per_gpu))

    df["VMEM Active Util"] = value


def derive_branch_active_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "SQ_ACTIVE_INST_MISC" in df.columns and "GRBM_GUI_ACTIVE" in df.columns
    n_xcd = 8
    cu_per_gpu = 304
    value = ((100 * df.get("SQ_ACTIVE_INST_MISC")) /
             (df.get("GRBM_GUI_ACTIVE")/n_xcd * cu_per_gpu))

    df["Branch Active Util"] = value


def derive_workgroup_manager_ns(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_REQ_NO_ALLOC_CSN" in df.columns and "GRBM_SPI_BUSY"
    n_xcd = 8
    se_per_gpu = 32
    value = 100*(
        df.get("SPI_RA_REQ_NO_ALLOC_CSN") /
        (df.get("GRBM_SPI_BUSY") / n_xcd * se_per_gpu)
    )
    df["Workgroup Manager Not Scheduled"] = value


def derive_scheduler_pipe_ns(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_REQ_NO_ALLOC" in df.columns and "GRBM_SPI_BUSY"
    n_xcd = 8
    se_per_gpu = 32
    value = 100*(
        df.get("SPI_RA_REQ_NO_ALLOC") /
        (df.get("GRBM_SPI_BUSY") / n_xcd * se_per_gpu)
    )
    df["Scheduler Pipe Not Scheduled"] = value


def derive_scheduler_pipe_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_RES_STALL_CSN" in df.columns and "GRBM_SPI_BUSY"
    n_xcd = 8
    se_per_gpu = 32
    value = 100*(
        df.get("SPI_RA_RES_STALL_CSN") /
        (df.get("GRBM_SPI_BUSY") / n_xcd * se_per_gpu)
    )
    df["Scheduler Pipe Stall"] = value


def derive_scratch_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_TMP_STALL_CSN" in df.columns and "GRBM_SPI_BUSY"
    n_xcd = 8
    se_per_gpu = 32
    value = 100*(
        df.get("SPI_RA_TMP_STALL_CSN") /
        (df.get("GRBM_SPI_BUSY") / n_xcd * se_per_gpu)
    )
    df["Scratch Stall"] = value


def derive_simd_waveslot_insf(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_WAVE_SIMD_FULL_CSN" in df.columns and "GRBM_GUI_ACTIVE"
    n_xcd = 8
    se_per_gpu = 32
    value = 100*(
        df.get("SPI_RA_WAVE_SIMD_FULL_CSN") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * se_per_gpu)
    )
    df["Insufficient SIMD Waveslots"] = value


def derive_simd_vgpr_insf(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_VGPR_SIMD_FULL_CSN" in df.columns and "GRBM_GUI_ACTIVE"
    n_xcd = 8
    cu_per_gpu = 304
    value = 100*(
        df.get("SPI_RA_VGPR_SIMD_FULL_CSN") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * cu_per_gpu)
    )
    df["Insufficient SIMD VGPRs"] = value


def derive_vgpr_writes(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SPI_VWC_CSC_WR' in df.columns and 'SPI_CSN_WAVE' in df.columns
    df["VGPR Writes"] = 4 * df.get('SPI_VWC_CSC_WR') / df.get('SPI_CSN_WAVE')


def derive_mfma_flop_inst(df: pd.DataFrame) -> pd.DataFrame:
    assert 'SQ_INSTS_VALU_MFMA_MOPS_BF16' in df.columns
    n_xcd = 8
    cu_per_gpu = 304
    mfma_cycles = (
        df.get('SQ_INSTS_VALU_MFMA_MOPS_BF16') * 512 /
        2 / (16*16*16) * 16 / (4 * cu_per_gpu)
    )
    df["MFMA Flop Cycles"] = 100 * mfma_cycles / \
        (df.get("GRBM_GUI_ACTIVE") / n_xcd)


def derive_simd_sgpr_insf(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_SGPR_SIMD_FULL_CSN" in df.columns and "GRBM_GUI_ACTIVE"
    n_xcd = 8
    se_per_gpu = 32
    value = 100*(
        df.get("SPI_RA_SGPR_SIMD_FULL_CSN") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * se_per_gpu)
    )
    df["Insufficient SIMD SGPRs"] = value


def derive_cu_lds_insf(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_LDS_CU_FULL_CSN" in df.columns and "GRBM_GUI_ACTIVE"
    n_xcd = 8
    se_per_gpu = 32
    value = 400*(
        df.get("SPI_RA_LDS_CU_FULL_CSN") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * se_per_gpu)
    )
    df["Insufficient CU LDS"] = value


def derive_cu_barriers_insf(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_BAR_CU_FULL_CSN" in df.columns and "GRBM_GUI_ACTIVE"
    n_xcd = 8
    se_per_gpu = 32
    value = 400*(
        df.get("SPI_RA_BAR_CU_FULL_CSN") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * se_per_gpu)
    )
    df["Insufficient CU Barriers"] = value


def derive_cu_workgroup_limit(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_TGLIM_CU_FULL_CSN" in df.columns and "GRBM_GUI_ACTIVE"
    n_xcd = 8
    se_per_gpu = 32
    value = 400*(
        df.get("SPI_RA_TGLIM_CU_FULL_CSN") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * se_per_gpu)
    )
    df["Insufficient CU Workgroup Limit"] = value


def derive_cu_wavefront_limit(df: pd.DataFrame) -> pd.DataFrame:
    assert "SPI_RA_WVLIM_STALL_CSN" in df.columns and "GRBM_GUI_ACTIVE"
    n_xcd = 8
    se_per_gpu = 32
    value = 400*(
        df.get("SPI_RA_WVLIM_STALL_CSN") /
        (df.get("GRBM_GUI_ACTIVE") / n_xcd * se_per_gpu)
    )
    df["Insufficient CU Wavefront Limit"] = value


def derive_l2_uncached(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCC_UC_REQ_sum" in df.columns
    value = df.get("TCC_UC_REQ_sum")
    df["l2_uncached"] = value


def derive_vl1_uncached(df: pd.DataFrame) -> pd.DataFrame:
    assert "TCP_TCC_UC_READ_REQ_sum" in df.columns and "TCP_TCC_UC_WRITE_REQ_sum" in df.columns and "TCP_TCC_UC_ATOMIC_REQ_sum" in df.columns
    value = (
        df.get("TCP_TCC_UC_READ_REQ_sum") +
        df.get("TCP_TCC_UC_WRITE_REQ_sum") +
        df.get("TCP_TCC_UC_ATOMIC_REQ_sum")
    )
    df["vl1_uncached"] = value


def derive_cpf_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPF_CPF_STAT_BUSY" in df.columns and "CPF_CPF_STAT_IDLE" in df.columns
    value = (
        100 * df.get("CPF_CPF_STAT_BUSY") /
        (df.get("CPF_CPF_STAT_IDLE") + df.get("CPF_CPF_STAT_BUSY"))
    )
    df["Command Processor Fetcher Util"] = value


def derive_cpf_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPF_CPF_STAT_STALL" in df.columns and "CPF_CPF_STAT_BUSY" in df.columns
    value = 100 * df.get("CPF_CPF_STAT_STALL") / (df.get("CPF_CPF_STAT_BUSY"))
    df["Command Processor Fetcher Stall"] = value


def derive_cpf_l2_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPF_CPF_TCIU_BUSY" in df.columns and "CPF_CPF_TCIU_IDLE" in df.columns
    value = (
        100 * df.get("CPF_CPF_TCIU_BUSY") /
        (df.get("CPF_CPF_TCIU_BUSY") + df.get("CPF_CPF_TCIU_IDLE"))
    )
    df["Command Processor Fetcher->L2 Util"] = value


def derive_cpf_l2_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPF_CPF_TCIU_STALL" in df.columns and "CPF_CPF_TCIU_BUSY" in df.columns
    value = (
        100 * df.get("CPF_CPF_TCIU_STALL") / (df.get("CPF_CPF_TCIU_BUSY"))
    )
    df["Command Processor Fetcher->L2 Stall"] = value


def derive_cpf_utcl1_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPF_CMP_UTCL1_STALL_ON_TRANSLATION" in df.columns and "CPF_CPF_STAT_BUSY" in df.columns
    value = (
        100 * df.get("CPF_CMP_UTCL1_STALL_ON_TRANSLATION") /
        (df.get("CPF_CPF_STAT_BUSY"))
    )
    df["Command Processor Fetcher->UTCL1 Stall"] = value


def derive_cpc_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPC_CPC_STAT_BUSY" in df.columns and "CPC_CPC_STAT_BUSY" in df.columns and "CPC_CPC_STAT_IDLE" in df.columns
    value = (
        100 * df.get("CPC_CPC_STAT_BUSY") /
        (df.get("CPC_CPC_STAT_IDLE") + df.get("CPC_CPC_STAT_BUSY"))
    )
    df["Packet Processor Util"] = value


def derive_cpc_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPC_CPC_STAT_STALL" in df.columns and "CPC_CPC_STAT_BUSY" in df.columns
    value = (
        100 * df.get("CPC_CPC_STAT_STALL") / df.get("CPC_CPC_STAT_BUSY")
    )
    df["Packet Processor Stall"] = value


def derive_cpc_decoding_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPC_ME1_BUSY_FOR_PACKET_DECODE" in df.columns and "CPC_CPC_STAT_BUSY" in df.columns
    value = (
        100 * df.get("CPC_ME1_BUSY_FOR_PACKET_DECODE") /
        df.get("CPC_CPC_STAT_BUSY")
    )
    df["Packet Processor Decoding Util"] = value


def derive_cpc_workgroup_manager_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPC_ME1_DC0_SPI_BUSY" in df.columns and "CPC_CPC_STAT_BUSY" in df.columns
    value = (
        100 * df.get("CPC_ME1_DC0_SPI_BUSY") /
        df.get("CPC_CPC_STAT_BUSY")
    )
    df["Packet Processor Workgroup Manager Util"] = value


def derive_cpc_l2_util(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPC_CPC_TCIU_BUSY" in df.columns and "CPC_CPC_TCIU_IDLE" in df.columns
    value = (
        100 * df.get("CPC_CPC_TCIU_BUSY") /
        (df.get("CPC_CPC_TCIU_BUSY") + df.get("CPC_CPC_TCIU_IDLE"))
    )
    df["Packet Processor L2 Util"] = value


def derive_cpc_utcl1_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPC_UTCL1_STALL_ON_TRANSLATION" in df.columns and "CPC_CPC_STAT_BUSY" in df.columns
    value = (
        100 * df.get("CPC_UTCL1_STALL_ON_TRANSLATION") /
        df.get("CPC_CPC_STAT_BUSY")
    )
    df["Packet Processor UTCL1 Stall"] = value


def derive_cpc_utcl2_stall(df: pd.DataFrame) -> pd.DataFrame:
    assert "CPC_CPC_UTCL2IU_BUSY" in df.columns and "CPC_CPC_UTCL2IU_IDLE" in df.columns
    value = (
        100 * df.get("CPC_CPC_UTCL2IU_BUSY") /
        (df.get("CPC_CPC_UTCL2IU_BUSY") + df.get("CPC_CPC_UTCL2IU_IDLE"))
    )
    df["Packet Processor UTCL2 Stall"] = value


def derive_vgpr_reg(df: pd.DataFrame) -> pd.DataFrame:
    assert "VGPR_Count" in df.columns
    df["VGPR Count"] = df.get("VGPR_Count")


def derive_lds_alloc(df: pd.DataFrame) -> pd.DataFrame:
    assert "LDS_Block_Size" in df.columns
    df["LDS Alloc"] = df.get("LDS_Block_Size")


def derive_lds_occ(df: pd.DataFrame) -> pd.DataFrame:
    assert "LDS_Block_Size" in df.columns
    df["LDS Occupancy"] = 65536 // df.get("LDS_Block_Size")


def derive_scratch_alloc(df: pd.DataFrame) -> pd.DataFrame:
    assert "Scratch_Size" in df.columns
    df["Scratch Alloc"] = df.get("Scratch_Size")


def derive_wpwg(df: pd.DataFrame) -> pd.DataFrame:
    assert "Workgroup_Size" in df.columns
    wave_per_workgroup = df.get("Workgroup_Size") / 256
    df["VGPR Count"] = df.get("VGPR_Count")


def derive_vl1d_coalesce(df: pd.DataFrame) -> pd.DataFrame:
    assert "TA_TOTAL_WAVEFRONTS_sum" in df.columns and "TCP_TOTAL_ACCESSES_sum" in df.columns
    df["vL1D Coalesce"] = ((df.get("TA_TOTAL_WAVEFRONTS_sum") * 64)
                           * 100) / (df.get("TCP_TOTAL_ACCESSES_sum") * 4)
