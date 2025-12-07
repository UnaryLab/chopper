from itertools import chain
from enum import Enum, auto
from matplotlib.figure import Figure

import chopper
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt

from chopper.common.load import get_df
from chopper.common.rocm_metrics import (
    derive_valu_cycles,
    derive_inffab_128b_req,
    derive_inffab_64b_req,
    derive_inffab_32b_req,
    derive_vl1d_coalesce,
    derive_l2_streaming_req,
    derive_ea_uncached_bytes,
    derive_inffab_bytes,
    derive_inffab_bw,
    derive_lds_alloc,
    derive_scratch_alloc,
    derive_lds_occ,
    derive_cpf_util,
    derive_cpf_stall,
    derive_cpf_l2_util,
    derive_cpf_l2_stall,
    derive_cpf_utcl1_stall,
    derive_cpc_util,
    derive_cpc_stall,
    derive_cpc_decoding_util,
    derive_cpc_workgroup_manager_util,
    derive_cpc_l2_util,
    derive_cpc_utcl1_stall,
    derive_cpc_utcl2_stall,
    derive_mfma_flop_inst,
    derive_reported_duration,
    derive_reported_duration_diff,
    derive_mfma_util,
    derive_vgpr_reg,
    derive_lds_util,
    derive_lds_bw_pop,
    derive_lds_bw,
    derive_lds_bytes,
    derive_lds_stalled_perc,
    derive_lds_bank_conflict_perc,
    derive_lds_access_rate,
    derive_l1d_addr_hitrate,
    derive_l2_hit_rate,
    derive_l2_cache_util,
    derive_l2_cache_my_util,
    derive_l2_cache_bytes,
    derive_l2_cache_bw,
    derive_wavefront_occupancy_pop,
    derive_duration,
    derive_kern_duration,
    derive_elapsed,
    derive_vl1d_hit_rate,
    derive_vl1d_util,
    derive_vl1d_bytes,
    derive_vl1d_bandwidth,
    derive_vl1d_bandwidth_pop,
    derive_l2_data_stall,
    derive_l2_req_stall,
    derive_sl1d_cache_hit_rate,
    derive_l2_fabric_read_bw,
    derive_l2_fabric_read_bw_pop,
    derive_l2_fabric_write_bw,
    derive_l2_fabric_write_bw_pop,
    derive_l2_tag_ram_write_stall,
    derive_l2_tag_ram_read_stall,
    derive_l2_tag_ram_atomic_stall,
    derive_l2_cache_bw_pop,
    derive_ta_busy,
    derive_ta_address_stall,
    derive_ta_data_stall,
    derive_ta_dp_stall,
    derive_data_return_busy,
    derive_vl1d_data_return_stall,
    derive_workgroup_manager_stall,
    derive_sl1d_cache_bw_pop,
    derive_sl1d_l2_bw_pop,
    derive_l1i_l2_bw_pop,
    derive_l1i_cache_hit_rate,
    derive_salu_util,
    derive_valu_util,
    derive_vmem_util,
    derive_branch_util,
    derive_salu_issued,
    derive_smem_issued,
    derive_flat_issued,
    derive_vmem_issued,
    derive_branch_issued,
    derive_sendmsg_issued,
    derive_mfma_i8_issued,
    derive_mfma_f16_issued,
    derive_mfma_f32_issued,
    derive_mfma_f64_issued,
    derive_mfma_iops_i8,
    derive_mfma_iops_sec_i8,
    derive_mfma_flops_f16,
    derive_mfma_flops_bf16,
    derive_mfma_flops_sec_f16,
    derive_mfma_flops_sec_bf16,
    derive_mfma_flops_f32,
    derive_mfma_flops_sec_f32,
    derive_mfma_flops_f64,
    derive_mfma_flops_sec_f64,
    derive_mfma_flops,
    derive_mfma_flops_sec,
    derive_valu_flops_f16,
    derive_valu_flops_f32,
    derive_valu_flops_f64,
    derive_valu_flops_sec_f16,
    derive_valu_flops_sec_f32,
    derive_valu_flops_sec_f64,
    derive_valu_flops,
    derive_valu_flops_sec,
    derive_workgroup_manager_ns,
    derive_scheduler_pipe_ns,
    derive_scheduler_pipe_stall,
    derive_scratch_stall,
    derive_vgpr_writes,
    derive_simd_waveslot_insf,
    derive_simd_vgpr_insf,
    derive_simd_sgpr_insf,
    derive_cu_lds_insf,
    derive_cu_barriers_insf,
    derive_cu_workgroup_limit,
    derive_cu_wavefront_limit,
    derive_l2_uncached,
    derive_vl1_uncached,
    derive_active_cu,
    derive_lds_latency,
    derive_vmem_latency,
    derive_mfma_flops_bf16_util,
    derive_valu_f32_flops_util,
)


derives = {
    'VALU F32 FLOPs Util':
    (
        '%',
        (
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "GRBM_GUI_ACTIVE",
        ),
        derive_valu_f32_flops_util
    ),
    'L2 Streaming Req':
    (
        'count',
        (
            "TCC_STREAMING_REQ_sum",
        ),
        derive_l2_streaming_req,
    ),
    'InfFab 32B Requests':
    (
        'count',
        (
            "TCC_EA0_RDREQ_32B_sum",
            "TCC_EA0_WRREQ_64B_sum",
            "TCC_EA0_WRREQ_sum",
        ),
        derive_inffab_32b_req,
    ),
    'InfFab 64B Requests':
    (
        'count',
        (
            "TCC_BUBBLE_sum",
            "TCC_EA0_RDREQ_32B_sum",
            "TCC_EA0_RDREQ_sum",
            "TCC_EA0_WRREQ_64B_sum",
            "TCC_EA0_WRREQ_sum",
        ),
        derive_inffab_64b_req,
    ),
    'InfFab 128B Requests':
    (
        'count',
        (
            "TCC_BUBBLE_sum",
        ),
        derive_inffab_128b_req,
    ),
    'VALU Cycles':
    (
        'count',
        (
            "SQ_ACTIVE_INST_VALU",
        ),
        derive_valu_cycles,
    ),
    'vL1D Coalesce':
    (
        '%',
        (
            "TA_TOTAL_WAVEFRONTS_sum",
            "TCP_TOTAL_ACCESSES_sum"
        ),
        derive_vl1d_coalesce
    ),
    "EA Uncached Bytes":
    (
        'count',
        (
            "TCC_EA0_WR_UNCACHED_32B_sum",
            "TCC_EA0_RD_UNCACHED_32B_sum",
        ),
        derive_ea_uncached_bytes,
    ),
    'InfFab Bytes':
        (
            'count',
            (
                "TCC_BUBBLE_sum",
                "TCC_EA0_RDREQ_32B_sum",
                "TCC_EA0_RDREQ_sum",
                "TCC_EA0_WRREQ_64B_sum",
                "TCC_EA0_WRREQ_sum",
            ),
            derive_inffab_bytes,
    ),
    'InfFab Bandwidth':
        (
            'bytes/s',
            (
                "TCC_BUBBLE_sum",
                "TCC_EA0_RDREQ_32B_sum",
                "TCC_EA0_RDREQ_sum",
                "TCC_EA0_WRREQ_64B_sum",
                "TCC_EA0_WRREQ_sum",
            ),
            derive_inffab_bw,
    ),
    'VGPR Count':
        (
            'count',
            ('VGPR_Count',),
            derive_vgpr_reg,
    ),
    'LDS Alloc':
        (
            'count',
            ('LDS_Block_Size',),
            derive_lds_alloc,
    ),
    'Scratch Alloc':
        (
            'count',
            ('Scratch_Size',),
            derive_scratch_alloc,
    ),
    'LDS Occupancy':
        (
            'count',
            ('LDS_Block_Size',),
            derive_lds_occ,
    ),
    'MFMA Util':
        (
            '%',
            ('SQ_VALU_MFMA_BUSY_CYCLES', 'GRBM_GUI_ACTIVE'),
            derive_mfma_util,
    ),
    # 'my_mfma_util':
    #     (
    #         '%',
    #         ('SQ_VALU_MFMA_BUSY_CYCLES', 'compute_unit_busy'),
    #         derive_my_mfma_util,
    #     ),
    'LDS Util':
        (
            '%',
            ('SQ_LDS_IDX_ACTIVE', 'GRBM_GUI_ACTIVE'),
            derive_lds_util,
    ),

    'LDS Bandwidth Percent of Peak':
        (
            '%',
            (
                "SQ_LDS_IDX_ACTIVE",
                "SQ_LDS_BANK_CONFLICT",
            ),
            derive_lds_bw_pop,
    ),
    'LDS Bandwidth':
        (
            '%',
            (
                "SQ_LDS_IDX_ACTIVE",
                "SQ_LDS_BANK_CONFLICT",
            ),
            derive_lds_bw,
    ),
    'LDS Bytes':
        (
            'count',
            (
                "SQ_LDS_IDX_ACTIVE",
                "SQ_LDS_BANK_CONFLICT",
            ),
            derive_lds_bytes,
    ),
    'LDS Access Rate':
        (
            '%',
            (
                'SQ_ACTIVE_INST_LDS',
                'GRBM_GUI_ACTIVE',
            ),
            derive_lds_access_rate,
    ),
    'LDS Stalled':
        (
            '%',
            (
                'SQ_LDS_UNALIGNED_STALL',
                'SQ_LDS_ADDR_CONFLICT',
                'SQ_LDS_BANK_CONFLICT',
                'SQ_LDS_IDX_ACTIVE',
            ),
            derive_lds_stalled_perc,
    ),
    "LDS Bank Conflicts/Access":
        (
            '%',
            (
                'SQ_LDS_BANK_CONFLICT',
                'SQ_LDS_IDX_ACTIVE',
            ),
            derive_lds_bank_conflict_perc,
    ),
    "LDS Latency":
        (
            'cycles',
            (
                "SQ_INSTS_LDS",
                "SQ_INST_LEVEL_LDS",
            ),
            derive_lds_latency,
    ),
    "VMEM Latency":
        (
            'cycles',
            (
                "SQ_INSTS_VMEM",
                "SQ_INST_LEVEL_VMEM",
            ),
            derive_vmem_latency,
    ),
    'L1D Address Translation Hit Rate':
        (
            '%',
            ('TCP_UTCL1_TRANSLATION_HIT_sum', 'TCP_UTCL1_REQUEST_sum'),
            derive_l1d_addr_hitrate,
    ),
    'L2 Hit Rate':
        (
            '%',
            ('TCC_HIT_sum', 'TCC_MISS_sum'),
            derive_l2_hit_rate,
    ),
    'L2 Util':
        (
            '%',
            ('TCC_BUSY_sum', 'GRBM_GUI_ACTIVE'),
            derive_l2_cache_util,
    ),
    'My L2 Util':
        (
            '%',
            ('TCC_BUSY_sum', 'TCC_CYCLE_sum'),
            derive_l2_cache_my_util,
    ),
    'L2 Bandwidth Percent of Peak':
        (
            '%',
            (
                "TCC_REQ_sum",
            ),
            derive_l2_cache_bw_pop,
    ),
    'L2 Bytes':
        (
            '%',
            (
                "TCC_REQ_sum",
            ),
            derive_l2_cache_bytes,
    ),
    'L2 Bandwidth':
        (
            '%',
            (
                "TCC_REQ_sum",
            ),
            derive_l2_cache_bw,
    ),
    'L2 Bandwidth Percent of Peak':
        (
            '%',
            (
                "TCC_REQ_sum",
            ),
            derive_l2_cache_bw_pop,
    ),
    'Wavefront Occupancy':
        (
            '%',
            (
                'SQ_LEVEL_WAVES',
                'GRBM_GUI_ACTIVE'
            ),
            derive_wavefront_occupancy_pop,
    ),
    'Kernel Duration':
        (
            'us',
            ('Kernel_Duration',),
            derive_kern_duration,
    ),
    'Duration':
        (
            'us',
            (),
            derive_duration,
    ),
    'Elapsed':
        (
            'us',
            (),
            derive_elapsed,
    ),
    'vL1D Hit Rate':
        (
            '%',
            (
                "TCP_TCC_READ_REQ_sum",
                "TCP_TCC_WRITE_REQ_sum",
                "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
                "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum",
                "TCP_TOTAL_CACHE_ACCESSES_sum",
            ),
            derive_vl1d_hit_rate,
    ),
    'vL1D Bandwidth Percent of Peak':
        (
            '%',
            (
                "TCP_TOTAL_CACHE_ACCESSES_sum",
            ),
            derive_vl1d_bandwidth_pop,
    ),
    'vL1D Bandwidth':
        (
            '%',
            (
                "TCP_TOTAL_CACHE_ACCESSES_sum",
            ),
            derive_vl1d_bandwidth,
    ),
    'vL1D Bytes':
        (
            'count',
            (
                "TCP_TOTAL_CACHE_ACCESSES_sum",
            ),
            derive_vl1d_bytes,
    ),
    'vL1D Util':
        (
            '%',
            (
                "TCP_GATE_EN2_sum",
                "TCP_GATE_EN1_sum",
            ),
            derive_vl1d_util,
    ),
    'L2->vL1D Data Stall':
        (
            '%',
            (
                "TCP_PENDING_STALL_CYCLES_sum",
                "TCP_GATE_EN1_sum",
            ),
            derive_l2_data_stall,
    ),
    'L2->vL1D Req Stall':
        (
            '%',
            (
                "TCP_TCR_TCP_STALL_CYCLES_sum",
                "TCP_GATE_EN1_sum",
            ),
            derive_l2_req_stall,
    ),
    'L2->vL1D Tag RAM Read Stall':
        (
            '%',
            (
                "TCP_READ_TAGCONFLICT_STALL_CYCLES_sum",
                "TCP_GATE_EN1_sum",
            ),
            derive_l2_tag_ram_read_stall,
    ),
    'L2->vL1D Tag RAM Write Stall':
        (
            '%',
            (
                "TCP_WRITE_TAGCONFLICT_STALL_CYCLES_sum",
                "TCP_GATE_EN1_sum",
            ),
            derive_l2_tag_ram_write_stall,
    ),
    'L2->vL1D Tag RAM Atomic Stall':
        (
            '%',
            (
                "TCP_ATOMIC_TAGCONFLICT_STALL_CYCLES_sum",
                "TCP_GATE_EN1_sum",
            ),
            derive_l2_tag_ram_atomic_stall,
    ),
    'TA Busy':
        (
            '%',
            (
                "TA_TA_BUSY_sum",
                "GRBM_GUI_ACTIVE",
            ),
            derive_ta_busy,
    ),
    'TA->vL1D address stall':
        (
            '%',
            (
                "TA_ADDR_STALLED_BY_TC_CYCLES_sum",
                "GRBM_GUI_ACTIVE",
            ),
            derive_ta_address_stall,
    ),
    'TA->vL1D data stall':
        (
            '%',
            (
                "TA_DATA_STALLED_BY_TC_CYCLES_sum",
                "GRBM_GUI_ACTIVE",
            ),
            derive_ta_data_stall,
    ),
    'TA->TD stall':
        (
            '%',
            (
                "TA_ADDR_STALLED_BY_TD_CYCLES_sum",
                "GRBM_GUI_ACTIVE",
            ),
            derive_ta_dp_stall,
    ),
    'TD busy':
        (
            '%',
            (
                "TD_TD_BUSY_sum",
                "GRBM_GUI_ACTIVE",
            ),
            derive_data_return_busy,
    ),
    'vL1D->TD stall':
        (
            '%',
            (
                "TD_TC_STALL_sum",
                "GRBM_GUI_ACTIVE",
            ),
            derive_vl1d_data_return_stall,
    ),
    'Workgroup Manager->TD stall':
        (
            '%',
            (
                "TD_SPI_STALL_sum",
                "GRBM_GUI_ACTIVE",
            ),
            derive_workgroup_manager_stall,
    ),
    'sL1D Hit Rate':
        (
            '%',
            (
                "SQC_DCACHE_HITS",
                "SQC_DCACHE_MISSES",
                "SQC_DCACHE_MISSES_DUPLICATE",
            ),
            derive_sl1d_cache_hit_rate,
    ),
    'sL1D Cache Bandwidth Percent of Peak':
        (
            '%',
            (
                "SQC_DCACHE_REQ",
                "Kernel_Duration",
            ),
            derive_sl1d_cache_bw_pop,
    ),
    'sL1D->L2 Bandwidth Percent of Peak':
        (
            '%',
            (
                "SQC_TC_DATA_READ_REQ",
                "SQC_TC_DATA_WRITE_REQ",
                "SQC_TC_DATA_ATOMIC_REQ",
            ),
            derive_sl1d_l2_bw_pop,
    ),
    'L1I Hit Rate':
        (
            '%',
            (
                "SQC_ICACHE_MISSES",
                "SQC_ICACHE_MISSES_DUPLICATE",
                "SQC_ICACHE_HITS",
            ),
            derive_l1i_cache_hit_rate,
    ),
    'L1I->L2 Bandwidth Percent of Peak':
        (
            '%',
            (
                "SQC_TC_INST_REQ",
                "Kernel_Duration",
            ),
            derive_l1i_l2_bw_pop,
    ),
    'L2 Fabric Read Bandwidth':
        (
            'GB/s',
            (
                "TCC_BUBBLE_sum",
                "TCC_EA0_RDREQ_32B_sum",
                "TCC_EA0_RDREQ_sum",
                "TCC_BUBBLE_sum",
            ),
            derive_l2_fabric_read_bw,
    ),
    'L2 Fabric Write&Atomic Bandwidth':
        (
            'GB/s',
            (
                "TCC_EA0_WRREQ_64B_sum",
                "TCC_EA0_WRREQ_sum",
                "TCP_TOTAL_CACHE_ACCESSES_sum",
            ),
            derive_l2_fabric_write_bw,
    ),
    'L2 Fabric Read Bandwidth Percent of Peak':
        (
            '%',
            (
                "TCC_BUBBLE_sum",
                "TCC_EA0_RDREQ_32B_sum",
                "TCC_EA0_RDREQ_sum",
                "TCC_BUBBLE_sum",
            ),
            derive_l2_fabric_read_bw_pop,
    ),
    'L2 Fabric Write&Atomic Bandwidth Percent of Peak':
        (
            '%',
            (
                "TCC_EA0_WRREQ_64B_sum",
                "TCC_EA0_WRREQ_sum",
                "TCP_TOTAL_CACHE_ACCESSES_sum",
            ),
            derive_l2_fabric_write_bw_pop,
    ),
    'SALU Util':
        (
            '%',
            (
                "SQ_ACTIVE_INST_SCA",
                "GRBM_GUI_ACTIVE",
            ),
            derive_salu_util,
    ),
    'VALU Util':
        (
            '%',
            (
                "SQ_ACTIVE_INST_VALU",
                "GRBM_GUI_ACTIVE",
            ),
            derive_valu_util,
    ),
    'VMEM Util':
        (
            '%',
            (
                "SQ_ACTIVE_INST_FLAT",
                "SQ_ACTIVE_INST_VMEM",
                "GRBM_GUI_ACTIVE",
            ),
            derive_vmem_util,
    ),
    'Branch Util':
        (
            '%',
            (
                "SQ_ACTIVE_INST_MISC",
                "GRBM_GUI_ACTIVE",
            ),
            derive_branch_util,
    ),
    'SALU Issued':
        (
            'count',
            (
                "SQ_INSTS_SALU",
            ),
            derive_salu_issued,
    ),
    'SMEM Issued':
        (
            'count',
            (
                "SQ_INSTS_SMEM",
            ),
            derive_smem_issued,
    ),
    'Flat Issued':
        (
            'count',
            (
                "SQ_INSTS_FLAT",
            ),
            derive_flat_issued,
    ),
    'VMEM Issued':
        (
            'count',
            (
                "SQ_INSTS_VMEM",
                "SQ_INSTS_LDS",
            ),
            derive_vmem_issued,
    ),
    'Branch Issued':
        (
            'count',
            (
                "SQ_INSTS_BRANCH",
            ),
            derive_branch_issued,
    ),
    'SENDMSG Issued':
        (
            'count',
            (
                "SQ_INSTS_SENDMSG",
            ),
            derive_sendmsg_issued,
    ),
    'MFMA I8 Issued':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_I8",
            ),
            derive_mfma_i8_issued,
    ),
    'MFMA F16 Issued':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_F16",
            ),
            derive_mfma_f16_issued,
    ),
    'MFMA F32 Issued':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_F32",
            ),
            derive_mfma_f32_issued,
    ),
    'MFMA F64 Issued':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_F64",
            ),
            derive_mfma_f64_issued,
    ),
    'MFMA I8 IOPs':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_I8",
            ),
            derive_mfma_iops_i8,
    ),
    'MFMA F16 Flops':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_F16",
            ),
            derive_mfma_flops_f16,
    ),
    'MFMA BF16 Flops':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_BF16",
            ),
            derive_mfma_flops_bf16,
    ),
    'MFMA F32 Flops':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_F32",
            ),
            derive_mfma_flops_f32,
    ),
    'MFMA F64 Flops':
        (
            'count',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_F64",
            ),
            derive_mfma_flops_f64,
    ),
    'MFMA I8 IOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_I8",
            ),
            derive_mfma_iops_sec_i8,
    ),
    'MFMA F16 FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_F16",
            ),
            derive_mfma_flops_sec_f16,
    ),
    'MFMA BF16 FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_BF16",
            ),
            derive_mfma_flops_sec_bf16,
    ),
    'MFMA F32 FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_F32",
            ),
            derive_mfma_flops_sec_f32,
    ),
    'MFMA F64 FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_F64",
            ),
            derive_mfma_flops_sec_f64,
    ),
    'MFMA Flops':
        (
            'Flops',
            (
                'SQ_INSTS_VALU_MFMA_MOPS_F64',
                'SQ_INSTS_VALU_MFMA_MOPS_F32',
                'SQ_INSTS_VALU_MFMA_MOPS_BF16',
                'SQ_INSTS_VALU_MFMA_MOPS_F16',
                'SQ_INSTS_VALU_MFMA_MOPS_I8',
            ),
            derive_mfma_flops,
    ),
    'MFMA FLOPS':
        (
            'Flops/sec',
            (
                'SQ_INSTS_VALU_MFMA_MOPS_F64',
                'SQ_INSTS_VALU_MFMA_MOPS_F32',
                'SQ_INSTS_VALU_MFMA_MOPS_BF16',
                'SQ_INSTS_VALU_MFMA_MOPS_F16',
                'SQ_INSTS_VALU_MFMA_MOPS_I8',
            ),
            derive_mfma_flops_sec,
    ),

    'MFMA Flop Cycles':
        (
            '%',
            (
                "SQ_INSTS_VALU_MFMA_MOPS_BF16",
                "GRBM_GUI_ACTIVE",
            ),
            derive_mfma_flop_inst,
    ),
    'VALU F16 Flops':
        (
            'Flops',
            (
                "SQ_INSTS_VALU_ADD_F16",
                "SQ_INSTS_VALU_MUL_F16",
                "SQ_INSTS_VALU_FMA_F16",
                "SQ_INSTS_VALU_TRANS_F16",
            ),
            derive_valu_flops_f16,
    ),
    'VALU F32 Flops':
        (
            'Flops',
            (
                "SQ_INSTS_VALU_ADD_F32",
                "SQ_INSTS_VALU_MUL_F32",
                "SQ_INSTS_VALU_FMA_F32",
                "SQ_INSTS_VALU_TRANS_F32",
            ),
            derive_valu_flops_f32,
    ),
    'VALU F64 Flops':
        (
            'Flops',
            (
                "SQ_INSTS_VALU_ADD_F64",
                "SQ_INSTS_VALU_MUL_F64",
                "SQ_INSTS_VALU_FMA_F64",
                "SQ_INSTS_VALU_TRANS_F64",
            ),
            derive_valu_flops_f64,
    ),
    'VALU F16 FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_ADD_F16",
                "SQ_INSTS_VALU_MUL_F16",
                "SQ_INSTS_VALU_FMA_F16",
                "SQ_INSTS_VALU_TRANS_F16",
            ),
            derive_valu_flops_sec_f16,
    ),
    'VALU F32 FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_ADD_F32",
                "SQ_INSTS_VALU_MUL_F32",
                "SQ_INSTS_VALU_FMA_F32",
                "SQ_INSTS_VALU_TRANS_F32",
            ),
            derive_valu_flops_sec_f32,
    ),
    'VALU F64 FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_ADD_F64",
                "SQ_INSTS_VALU_MUL_F64",
                "SQ_INSTS_VALU_FMA_F64",
                "SQ_INSTS_VALU_TRANS_F64",
            ),
            derive_valu_flops_sec_f64,
    ),
    'VALU Flops':
        (
            'Flops',
            (
                "SQ_INSTS_VALU_ADD_F16",
                "SQ_INSTS_VALU_MUL_F16",
                "SQ_INSTS_VALU_FMA_F16",
                "SQ_INSTS_VALU_TRANS_F16",
                "SQ_INSTS_VALU_ADD_F32",
                "SQ_INSTS_VALU_MUL_F32",
                "SQ_INSTS_VALU_FMA_F32",
                "SQ_INSTS_VALU_TRANS_F32",
                "SQ_INSTS_VALU_ADD_F64",
                "SQ_INSTS_VALU_MUL_F64",
                "SQ_INSTS_VALU_FMA_F64",
                "SQ_INSTS_VALU_TRANS_F64",
            ),
            derive_valu_flops,
    ),
    'VALU FLOPS':
        (
            'Flops/sec',
            (
                "SQ_INSTS_VALU_ADD_F16",
                "SQ_INSTS_VALU_MUL_F16",
                "SQ_INSTS_VALU_FMA_F16",
                "SQ_INSTS_VALU_TRANS_F16",
                "SQ_INSTS_VALU_ADD_F32",
                "SQ_INSTS_VALU_MUL_F32",
                "SQ_INSTS_VALU_FMA_F32",
                "SQ_INSTS_VALU_TRANS_F32",
                "SQ_INSTS_VALU_ADD_F64",
                "SQ_INSTS_VALU_MUL_F64",
                "SQ_INSTS_VALU_FMA_F64",
                "SQ_INSTS_VALU_TRANS_F64",
            ),
            derive_valu_flops_sec,
    ),
    'Workgroup Manager Not Scheduled':
        (
            '%',
            (
                "SPI_RA_REQ_NO_ALLOC_CSN",
                "GRBM_SPI_BUSY",
            ),
            derive_workgroup_manager_ns,
    ),
    'Scheduler Pipe Not Scheduled':
        (
            '%',
            (
                "SPI_RA_REQ_NO_ALLOC",
                "GRBM_SPI_BUSY",
            ),
            derive_scheduler_pipe_ns,
    ),
    'Scheduler Pipe Stall':
        (
            '%',
            (
                "SPI_RA_RES_STALL_CSN",
                "GRBM_SPI_BUSY",
            ),
            derive_scheduler_pipe_stall,
    ),
    'Scratch Stall':
        (
            '%',
            (
                "SPI_RA_TMP_STALL_CSN",
                "GRBM_SPI_BUSY",
            ),
            derive_scratch_stall,
    ),
    'VGPR Writes':
        (
            'count',
            (
                "SPI_VWC_CSC_WR",
                "SPI_CSN_WAVE",
            ),
            derive_vgpr_writes,
    ),
    'Insufficient SIMD Waveslots':
        (
            '%',
            (
                "SPI_RA_WAVE_SIMD_FULL_CSN",
                "GRBM_GUI_ACTIVE",
            ),
            derive_simd_waveslot_insf,
    ),
    'Insufficient SIMD VGPRs':
        (
            '%',
            (
                "SPI_RA_VGPR_SIMD_FULL_CSN",
                "GRBM_GUI_ACTIVE",
            ),
            derive_simd_vgpr_insf,
    ),
    'Insufficient SIMD SGPRs':
        (
            '%',
            (
                "SPI_RA_SGPR_SIMD_FULL_CSN",
                "GRBM_GUI_ACTIVE",
            ),
            derive_simd_sgpr_insf,
    ),
    'Insufficient CU LDS':
        (
            '%',
            (
                "SPI_RA_LDS_CU_FULL_CSN",
                "GRBM_GUI_ACTIVE",
            ),
            derive_cu_lds_insf,
    ),
    'Insufficient CU Barriers':
        (
            '%',
            (
                "SPI_RA_BAR_CU_FULL_CSN",
                "GRBM_GUI_ACTIVE",
            ),
            derive_cu_barriers_insf,
    ),
    'Insufficient CU Workgroup Limit':
        (
            '%',
            (
                "SPI_RA_TGLIM_CU_FULL_CSN",
                "GRBM_GUI_ACTIVE",
            ),
            derive_cu_workgroup_limit,
    ),
    'Insufficient CU Wavefront Limit':
        (
            '%',
            (
                "SPI_RA_WVLIM_STALL_CSN",
                "GRBM_GUI_ACTIVE",
            ),
            derive_cu_wavefront_limit,
    ),
    'l2_uncached':
        (
            'count',
            (
                "TCC_UC_REQ_sum",
            ),
            derive_l2_uncached,
    ),
    'vl1_uncached':
        (
            'count',
            (
                "TCP_TCC_UC_READ_REQ_sum",
                "TCP_TCC_UC_WRITE_REQ_sum",
                "TCP_TCC_UC_ATOMIC_REQ_sum",
            ),
            derive_vl1_uncached,
    ),
    'Active CUs':
        (
            'count',
            (
                "SQ_BUSY_CU_CYCLES",
                "GRBM_GUI_ACTIVE",
            ),
            derive_active_cu,
    ),
    'Command Processor Fetcher Util':
        (
            '%',
            (
                "CPF_CPF_STAT_BUSY",
                "CPF_CPF_STAT_IDLE",
            ),
            derive_cpf_util,
    ),
    'Command Processor Fetcher Stall':
        (
            '%',
            (
                "CPF_CPF_STAT_STALL",
                "CPF_CPF_STAT_BUSY",
            ),
            derive_cpf_stall,
    ),
    'Command Processor Fetcher->L2 Util':
        (
            '%',
            (
                "CPF_CPF_TCIU_BUSY",
                "CPF_CPF_TCIU_IDLE",
            ),
            derive_cpf_l2_util,
    ),
    'Command Processor Fetcher->L2 Stall':
        (
            '%',
            (
                "CPF_CPF_TCIU_STALL",
                "CPF_CPF_TCIU_BUSY",
            ),
            derive_cpf_l2_stall,
    ),
    'Command Processor Fetcher->UTCL1 Stall':
        (
            '%',
            (
                "CPF_CMP_UTCL1_STALL_ON_TRANSLATION",
                "CPF_CPF_STAT_BUSY",
            ),
            derive_cpf_utcl1_stall,
    ),
    'Packet Processor Util':
        (
            '%',
            (
                "CPC_CPC_STAT_BUSY",
                "CPC_CPC_STAT_BUSY",
                "CPC_CPC_STAT_IDLE",
            ),
            derive_cpc_util,
    ),
    'Packet Processor Stall':
        (
            '%',
            (
                "CPC_CPC_STAT_STALL",
                "CPC_CPC_STAT_BUSY",
            ),
            derive_cpc_stall,
    ),
    'Packet Processor Decoding Util':
        (
            '%',
            (
                "CPC_ME1_BUSY_FOR_PACKET_DECODE",
                "CPC_CPC_STAT_BUSY",
            ),
            derive_cpc_decoding_util,
    ),
    'Packet Processor Workgroup Manager Util':
        (
            '%',
            (
                "CPC_ME1_DC0_SPI_BUSY",
                "CPC_CPC_STAT_BUSY",
            ),
            derive_cpc_workgroup_manager_util,
    ),
    'Packet Processor L2 Util':
        (
            '%',
            (
                "CPC_CPC_TCIU_BUSY",
                "CPC_CPC_TCIU_IDLE",
            ),
            derive_cpc_l2_util,
    ),
    'Packet Processor UTCL1 Stall':
        (
            '%',
            (
                "CPC_UTCL1_STALL_ON_TRANSLATION",
                "CPC_CPC_STAT_BUSY",
            ),
            derive_cpc_utcl1_stall,
    ),
    'Packet Processor UTCL2 Stall':
        (
            '%',
            (
                "CPC_CPC_UTCL2IU_BUSY",
                "CPC_CPC_UTCL2IU_IDLE",
            ),
            derive_cpc_utcl2_stall,
    ),
    'Reported Duration':
        (
            'us',
            (
                'GRBM_GUI_ACTIVE',
            ),
            derive_reported_duration,
    ),
    'Reported Duration Difference':
        (
            '%',
            (
                'GRBM_GUI_ACTIVE',
            ),
            derive_reported_duration_diff,
    ),
    'MFMA BF16 Util':
        (
            '%',
            (
                'SQ_INSTS_VALU_MFMA_MOPS_BF16',
            ),
            derive_mfma_flops_bf16_util,
    ),
    # 'LDS_bank_conflicts_to_access_pop':
    #     (
    #         ('SQ_LDS_IDX_ACTIVE', 'SQ_LDS_BANK_CONFLICT'),
    #         derive_LDS_bank_conflicts_to_access_pop,
    #     ),
}


class Norm(Enum):
    BYTES = auto()
    PERC = auto()
    INSTS = auto()
    NONE = auto()


def norm_string(x, norm_type, duration, bw):
    assert norm_type == Norm.BYTES or norm_type == Norm.PERC or norm_type == Norm.INSTS or norm_type == Norm.NONE
    if norm_type == Norm.NONE:
        return x
    if norm_type == Norm.PERC:
        return f"{x:.2f}"

    if bw:
        x = x / duration
    if x > 10**12:
        return f"{x / 10**12:.2f} {'TB' if norm_type == Norm.BYTES else 'TInst'}{'/s' if bw else ''}"
    elif x > 10**9:
        return f"{x / 10**9:.2f} {'GB' if norm_type == Norm.BYTES else 'GInst'}{'/s' if bw else ''}"
    elif x > 10**6:
        return f"{x / 10**6:.2f} {'MB' if norm_type == Norm.BYTES else 'MInst'}{'/s' if bw else ''}"
    elif x > 10**3:
        return f"{x / 10**3:.2f} {'KB' if norm_type == Norm.BYTES else 'KInst'}{'/s' if bw else ''}"
    else:
        return f"{x:.0f} {'B' if norm_type == Norm.BYTES else 'Inst'}{'/s' if bw else ''}"


def get_data(
    path: str = '~/data/ispass_v2/k002-003_MODEL_NAMEllama_USE_FSDP20_ITERS20_WAIT9_ACTIVE10_POWER_MAN0_ADJUST_STEPS3_WAIT_STEPS50_INITIAL_POWER_CAP750_REALLOC_POWER0_MAX_ADJ15_USE_SUM1_USE_LAST0_USE_MAX0_USE_GLOBAL1_POWER_BUDGET0_FP80_GRAD_ACC99999_PROFILE_TYPE2/b2s8/counters.pkl',
):

    selected_metrics = (
        "L2 Hit Rate",
        "L2 Util",
        "vL1D Hit Rate",
        "vL1D Util",
        "LDS Util",
        "sL1D Hit Rate",
        "L1I Hit Rate",
        "Active CUs",
        "SALU Util",
        "VALU Util",
        "VMEM Util",
        "MFMA Util",
        "Branch Util",
        "L2 Fabric Read Bandwidth Percent of Peak",
        "L2 Fabric Write&Atomic Bandwidth Percent of Peak",
    )
    sum_cols = tuple(chain.from_iterable(derives[cur_metric][1]
                     for cur_metric in selected_metrics))

    # get non-derivation metrics
    sum_cols += (
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
        "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
        "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum",
        "TCC_READ_sum",
        "TCC_WRITE_sum",
        "TCC_ATOMIC_sum",
        "SQC_DCACHE_REQ",
        "SQC_TC_DATA_READ_REQ",
        "SQC_TC_DATA_WRITE_REQ",
        "SQC_TC_DATA_ATOMIC_REQ",
        "SQC_TC_INST_REQ",
        "SQC_ICACHE_REQ",
        "TCC_EA0_RDREQ_32B_sum",
        "TCC_EA0_RDREQ_sum",
        "TCC_EA0_RDREQ_32B_sum",
        "TCC_EA0_WRREQ_64B_sum",
        "TCC_EA0_WRREQ_sum",
        "TCC_EA0_WRREQ_64B_sum",
        "TCC_EA0_RDREQ_sum",
        "TCC_BUBBLE_sum",
        "TCC_EA0_WRREQ_sum",
        "TCC_EA0_WRREQ_DRAM_sum",
        "TCC_EA0_RDREQ_DRAM_sum",
        "TCC_EA0_ATOMIC_sum",
        "TCP_TOTAL_READ_sum",
        "TCP_TOTAL_WRITE_sum",
        "TCP_TOTAL_ATOMIC_WITH_RET_sum",
        "TCP_TOTAL_ATOMIC_WITHOUT_RET_sum",
        "SQ_INSTS_LDS",
        "Scratch_Size",
        "LDS_Block_Size",
        "SGPR_Count",
        "VGPR_Count",
        "SPI_CSN_NUM_THREADGROUPS",
        "SPI_CSN_WAVE",
        "SQ_LDS_IDX_ACTIVE",
        "SQ_LDS_BANK_CONFLICT",
        "dur",
    )
    sum_cols = tuple(set(sum_cols))

    group_map = {c: ['sum'] for c in sum_cols}

    return get_df(
        path,
        fix_names=True,
        group_arr=['gpu', 'iteration', 'layer', 'operator-name'],
        group_map=group_map,
    ), selected_metrics


def draw(
    fig: Figure,
    input_data,
    agg_meth: str = 'median',
    operation: str = 'b_mlp_gp',
):
    if agg_meth == "median":
        comp_func = np.median
    elif agg_meth == "min":
        comp_func = np.min
    elif agg_meth == "max":
        comp_func = np.max
    else:
        raise ValueError(f"Invalid sortby for memory analysis: {agg_meth}")

    def agg_func(x, norm_type, duration, bw):
        return norm_string(comp_func(x), norm_type, comp_func(duration), bw)

    df, selected_metrics = input_data
    df = df[df['operator-name'] == operation].copy()
    # df = df[df['operator-name'] == 'b_attn_fa']

    derive_cols = tuple(derives[cur_metric][2]
                        for cur_metric in selected_metrics)
    for dc in derive_cols:
        dc(df)

    duration = df.get("dur").astype(float)*1e-6
    bw = False

    fig.clear()
    memory_analysis_ax = fig.add_subplot()

    memory_analysis_ax.set_title("Memory Chart")

    fig.patch.set_facecolor('gray')
    memory_analysis_ax.set_xlim(0, 16)
    memory_analysis_ax.set_ylim(0, 6)
    memory_analysis_ax.set_aspect('equal')
    memory_analysis_ax.axis('off')

    ibuff_x = (0.25, 1.25)
    ibuff_y = (2.5, 5.5)
    ibuff_width = ibuff_x[1]-ibuff_x[0]
    ibuff_height = ibuff_y[1]-ibuff_y[0]

    n_ibuff = 4
    instr_dispatch = [
        'Branch Util',
        'LDS Util',
        'VMEM Util',
        'MFMA Util',
        'VALU Util',
        'SALU Util',
    ]
    n_insts = len(instr_dispatch)
    inst_height = ibuff_height / n_insts
    inst_len = .2
    dec_len = inst_len
    wire_len = dec_len*6

    overlay_mult = 0.075
    ibuff_diff = overlay_mult*ibuff_width

    fontsize = 10

    for i in range(n_ibuff):
        ibuff = mpatches.Rectangle(
            (ibuff_x[0], ibuff_y[0]), ibuff_width, ibuff_height, edgecolor='orange', facecolor='black', lw=2)
        memory_analysis_ax.add_patch(ibuff)

        for j in range(n_insts):
            inst_y = ibuff_y[0]+inst_height*(j+.5)

            memory_analysis_ax.annotate("", xytext=(ibuff_x[1], inst_y), xy=(ibuff_x[1]+inst_len+ibuff_diff*i, inst_y),
                                        arrowprops=dict(arrowstyle="-", lw=2, color='orange'))
            if i == 0:
                trap_width = 0.125
                trap_height = n_ibuff*ibuff_diff
                trapezoid = mpatches.Polygon(
                    [
                        [inst_len+ibuff_x[1]+trap_width, inst_y +
                            ibuff_diff/2 - trap_height*3/4],
                        [inst_len+ibuff_x[1], inst_y +
                            ibuff_diff/2 - trap_height],
                        [inst_len+ibuff_x[1], inst_y + ibuff_diff/2],
                        [inst_len+ibuff_x[1]+trap_width, inst_y +
                            ibuff_diff/2 - trap_height/4],
                    ],
                    closed=True, edgecolor='orange', facecolor='white', lw=2
                )
                memory_analysis_ax.add_patch(trapezoid)
                memory_analysis_ax.annotate("",
                                            xytext=(
                                                inst_len +
                                                ibuff_x[1]+trap_width,
                                                inst_y+ibuff_diff/2-trap_height/2
                                            ),
                                            xy=(
                                                inst_len +
                                                ibuff_x[1] +
                                                trap_width+dec_len,
                                                inst_y+ibuff_diff/2-trap_height/2
                                            ),
                                            arrowprops=dict(arrowstyle="-", lw=2, color='orange'))

                mux_width = trap_width*2
                memory_analysis_ax.annotate("",
                                            xytext=(
                                                inst_len+ibuff_x[1] +
                                                trap_width+dec_len+mux_width,
                                                inst_y+ibuff_diff/2-trap_height/2
                                            ),
                                            xy=(
                                                inst_len+ibuff_x[1]+trap_width +
                                                dec_len+mux_width+wire_len,
                                                inst_y+ibuff_diff/2-trap_height/2
                                            ),
                                            arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
                memory_analysis_ax.text(inst_len+ibuff_x[1]+trap_width+dec_len+mux_width+ibuff_diff,
                                        inst_y+ibuff_diff*3/2-trap_height/2,
                                        f"{instr_dispatch[j]}: {agg_func(df[instr_dispatch[j]], Norm.PERC, duration, bw)}%", color="white", fontsize=fontsize, ha='left')

                if j == 0:
                    mux_y = ibuff_y[1] - (n_ibuff-1)*ibuff_diff/2
                    memory_analysis_ax.text(inst_len+ibuff_x[1]+trap_width, mux_y + ibuff_diff,
                                            "Instr Dispatch", color="white", fontsize=fontsize, ha='left')
                    trapezoid = mpatches.Polygon(
                        [
                            [inst_len+ibuff_x[1]+trap_width+mux_width+dec_len, mux_y
                                - ibuff_height + ibuff_height/30],
                            [inst_len+ibuff_x[1]+trap_width+dec_len, mux_y
                                - ibuff_height],
                            [inst_len+ibuff_x[1]+trap_width+dec_len, mux_y],
                            [inst_len+ibuff_x[1]+trap_width+mux_width +
                                dec_len, mux_y - ibuff_height/30],
                        ],
                        closed=True, edgecolor='orange', facecolor='white', lw=2
                    )
                    memory_analysis_ax.add_patch(trapezoid)

        if i == 0:
            exec_buff_width = 1.5
            exec_buff_x = (inst_len+ibuff_x[1]+trap_width +
                           dec_len+mux_width+wire_len+ibuff_diff,
                           inst_len+ibuff_x[1]+trap_width +
                           dec_len+mux_width+wire_len+exec_buff_width+ibuff_diff)
            exec_buff_y = (
                ibuff_y[0]-(n_ibuff-1)*ibuff_diff/2,
                ibuff_y[1]-(n_ibuff-1)*ibuff_diff/2)
            exec_buff_height = exec_buff_y[1]-exec_buff_y[0]
            exec_buff = mpatches.Rectangle(
                (exec_buff_x[0], exec_buff_y[0]), exec_buff_width, exec_buff_height, edgecolor='orange', facecolor='black', lw=2)
            memory_analysis_ax.add_patch(exec_buff)

            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1] +
                                    ibuff_diff, "Exec", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-4*ibuff_diff,
                                    "Active CUs", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-6*ibuff_diff,
                                    f"{agg_func(df['Active CUs'], Norm.NONE, duration, bw):.0f}/{304}", color="Yellow", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-10*ibuff_diff,
                                    f"VGPRS: {agg_func(df['VGPR_Count'], Norm.NONE, duration, bw):.0f}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-12*ibuff_diff,
                                    f"SGPRS: {agg_func(df['SGPR_Count'], Norm.NONE, duration, bw):.0f}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-14*ibuff_diff,
                                    f"LDS Alloc: {agg_func(df['LDS_Block_Size'], Norm.NONE, duration, bw):.0f}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-16*ibuff_diff,
                                    f"Scratch Alloc: {agg_func(df['Scratch_Size'], Norm.NONE, duration, bw):.0f}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-18*ibuff_diff,
                                    f"Wavefronts: {agg_func(df['SPI_CSN_WAVE'], Norm.NONE, duration, bw):.0f}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[0]+ibuff_diff, exec_buff_y[1]-20*ibuff_diff,
                                    f"Workgroups: {agg_func(df['SPI_CSN_NUM_THREADGROUPS'], Norm.NONE, duration, bw):.0f}", color="white", fontsize=fontsize, ha='left')

            lds_buff_width = 1
            lds_buff_height = exec_buff_height/5*.7
            lds_buff_x = (
                exec_buff_x[0]+exec_buff_width+wire_len,
                exec_buff_x[0]+exec_buff_width+wire_len+lds_buff_width,
            )
            lds_buff_y = exec_buff_y[1] - lds_buff_height/.7
            lds_buff = mpatches.Rectangle(
                (lds_buff_x[0], lds_buff_y), lds_buff_width, lds_buff_height, edgecolor='orange', facecolor='white', lw=2)
            memory_analysis_ax.annotate("",
                                        xytext=(
                                            lds_buff_x[0],
                                            lds_buff_y + lds_buff_height/2
                                        ),
                                        xy=(
                                            exec_buff_x[1],
                                            lds_buff_y + lds_buff_height/2
                                        ),
                                        arrowprops=dict(arrowstyle="<->", lw=2, color='orange'))
            memory_analysis_ax.text(lds_buff_x[0]+ibuff_diff, lds_buff_y+lds_buff_height+ibuff_diff,
                                    "LDS", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(lds_buff_x[0]+ibuff_diff, lds_buff_y+lds_buff_height/2,
                                    f"Util: {agg_func(df['LDS Util'], Norm.PERC, duration, bw)}%", color="black", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(exec_buff_x[1]+ibuff_diff, lds_buff_y+lds_buff_height/4,
                                    f"Req: {agg_func(df['SQ_INSTS_LDS'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')
            lds_bytes = (
                (df.get("SQ_LDS_IDX_ACTIVE") -
                 df.get("SQ_LDS_BANK_CONFLICT"))
                * 4
                * 32
            )
            memory_analysis_ax.text(exec_buff_x[1]+ibuff_diff, lds_buff_y+lds_buff_height*3/4,
                                    f"Total: {agg_func(lds_bytes, Norm.BYTES, duration, bw)}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.add_patch(lds_buff)

            vl1_cache_buff_width = lds_buff_width
            vl1_cache_buff_height = exec_buff_height*2/5*.7
            vl1_cache_buff_x = (
                exec_buff_x[0]+exec_buff_width+wire_len,
                exec_buff_x[0]+exec_buff_width +
                wire_len+vl1_cache_buff_width,
            )
            vl1_cache_buff_y = lds_buff_y - vl1_cache_buff_height/.7
            vl1_cache_buff = mpatches.Rectangle(
                (vl1_cache_buff_x[0], vl1_cache_buff_y), vl1_cache_buff_width, vl1_cache_buff_height, edgecolor='orange', facecolor='white', lw=2)
            memory_analysis_ax.text(vl1_cache_buff_x[0]+ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height+ibuff_diff,
                                    f"Vector L1 Cache", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(vl1_cache_buff_x[0]+ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height*3/5,
                                    f"Hit: {agg_func(df['vL1D Hit Rate'], Norm.PERC, duration, bw)}%", color="black", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(vl1_cache_buff_x[0]+ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height*4/5,
                                    f"Util: {agg_func(df['vL1D Util'], Norm.PERC, duration, bw)}%", color="black", fontsize=fontsize, ha='left')
            bytes_to_l1 = df['TCP_TOTAL_WRITE_sum'] * 64
            bytes_from_l1 = df['TCP_TOTAL_READ_sum'] * 64

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            vl1_cache_buff_x[0],
                                            vl1_cache_buff_y + vl1_cache_buff_height*4/5
                                        ),
                                        xy=(
                                            exec_buff_x[1],
                                            vl1_cache_buff_y + vl1_cache_buff_height*4/5
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(exec_buff_x[1]+ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height*4/5+ibuff_diff,
                                    f"Bytes: {agg_func(bytes_from_l1, Norm.BYTES, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            exec_buff_x[1],
                                            vl1_cache_buff_y + vl1_cache_buff_height/2
                                        ),
                                        xy=(
                                            vl1_cache_buff_x[0],
                                            vl1_cache_buff_y + vl1_cache_buff_height/2
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(exec_buff_x[1]+ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height/2+ibuff_diff,
                                    f"Bytes: {agg_func(bytes_to_l1, Norm.BYTES, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            exec_buff_x[1],
                                            vl1_cache_buff_y + vl1_cache_buff_height/5
                                        ),
                                        xy=(
                                            vl1_cache_buff_x[0],
                                            vl1_cache_buff_y + vl1_cache_buff_height/5
                                        ),
                                        arrowprops=dict(arrowstyle="<->", lw=2, color='orange'))
            memory_analysis_ax.text(exec_buff_x[1]+ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height/5+ibuff_diff,
                                    f"Atomic: {agg_func(df['TCP_TOTAL_ATOMIC_WITH_RET_sum'] + df['TCP_TOTAL_ATOMIC_WITHOUT_RET_sum'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.add_patch(vl1_cache_buff)

            sl1d_cache_buff_width = lds_buff_width
            sl1d_cache_buff_height = exec_buff_height*2/5*.7
            sl1d_cache_buff_x = (
                exec_buff_x[0]+exec_buff_width+wire_len,
                exec_buff_x[0]+exec_buff_width +
                wire_len+sl1d_cache_buff_width,
            )
            sl1d_cache_buff_y = vl1_cache_buff_y - sl1d_cache_buff_height/.7
            sl1d_cache_buff = mpatches.Rectangle(
                (sl1d_cache_buff_x[0], sl1d_cache_buff_y), sl1d_cache_buff_width, sl1d_cache_buff_height, edgecolor='orange', facecolor='white', lw=2)
            memory_analysis_ax.text(sl1d_cache_buff_x[0]+ibuff_diff, sl1d_cache_buff_y+sl1d_cache_buff_height+ibuff_diff,
                                    "Scalar L1D Cache", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(sl1d_cache_buff_x[0]+ibuff_diff, sl1d_cache_buff_y+sl1d_cache_buff_height*3/5,
                                    f"Hit: {agg_func(df['sL1D Hit Rate'], Norm.PERC, duration, bw)}%", color="black", fontsize=fontsize, ha='left')
            memory_analysis_ax.annotate("",
                                        xytext=(
                                            sl1d_cache_buff_x[0],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height/2
                                        ),
                                        xy=(
                                            exec_buff_x[1],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height/2
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(exec_buff_x[1]+ibuff_diff, sl1d_cache_buff_y+sl1d_cache_buff_height/2+ibuff_diff,
                                    f"Rd: {agg_func(df['SQC_DCACHE_REQ'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.add_patch(sl1d_cache_buff)

            l1i_cache_buff_width = lds_buff_width
            l1i_cache_buff_height = exec_buff_height/5*.7
            l1i_cache_buff_x = (
                exec_buff_x[0]+exec_buff_width+wire_len,
                exec_buff_x[0]+exec_buff_width +
                wire_len+l1i_cache_buff_width,
            )
            l1i_cache_buff_y = sl1d_cache_buff_y - l1i_cache_buff_height/.7
            l1i_cache_buff = mpatches.Rectangle(
                (l1i_cache_buff_x[0], l1i_cache_buff_y), l1i_cache_buff_width, l1i_cache_buff_height, edgecolor='orange', facecolor='white', lw=2)
            memory_analysis_ax.text(l1i_cache_buff_x[0]+ibuff_diff, l1i_cache_buff_y+l1i_cache_buff_height+ibuff_diff,
                                    "Intr L1 Cache", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(l1i_cache_buff_x[0]+ibuff_diff, l1i_cache_buff_y+l1i_cache_buff_height/2,
                                    f"Hit: {agg_func(df['L1I Hit Rate'], Norm.PERC, duration, bw)}%", color="black", fontsize=fontsize, ha='left')
            memory_analysis_ax.annotate("",
                                        xytext=(
                                            l1i_cache_buff_x[0],
                                            l1i_cache_buff_y + l1i_cache_buff_height/2
                                        ),
                                        xy=(
                                            ibuff_x[0] +
                                            ibuff_diff*n_ibuff,
                                            l1i_cache_buff_y + l1i_cache_buff_height/2
                                        ),
                                        arrowprops=dict(arrowstyle="-", lw=2, color='orange'))
            memory_analysis_ax.annotate("",
                                        xytext=(
                                            ibuff_x[0] +
                                            ibuff_diff*n_ibuff,
                                            l1i_cache_buff_y + l1i_cache_buff_height/2
                                        ),
                                        xy=(
                                            ibuff_x[0] +
                                            ibuff_diff*n_ibuff,
                                            ibuff_y[0] -
                                            ibuff_diff*n_ibuff,
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(exec_buff_x[1]-ibuff_diff, l1i_cache_buff_y+l1i_cache_buff_height/2+ibuff_diff,
                                    f"Fetch: {agg_func(df['SQC_ICACHE_REQ'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.add_patch(l1i_cache_buff)

            l2_cache_buff_width = exec_buff_width
            l2_cache_buff_height = lds_buff_height/.7 + vl1_cache_buff_height/.7 + \
                sl1d_cache_buff_height/.7 + l1i_cache_buff_height/.7
            l2_cache_buff_x = (
                exec_buff_x[1]+wire_len*2+lds_buff_width,
                exec_buff_x[1]+wire_len*2 +
                lds_buff_width + l2_cache_buff_width,
            )
            l2_cache_buff_y = exec_buff_y[0] - l1i_cache_buff_height/.7
            l2_cache_buff = mpatches.Rectangle(
                (l2_cache_buff_x[0], l2_cache_buff_y), l2_cache_buff_width, l2_cache_buff_height, edgecolor='orange', facecolor='black', lw=2)
            memory_analysis_ax.text(l2_cache_buff_x[0]+ibuff_diff, l2_cache_buff_y+l2_cache_buff_height+ibuff_diff,
                                    "L2 Cache", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.add_patch(l2_cache_buff)
            bytes_from_l2 = (
                df["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"] * 64
                + df["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"] * 64
                + df["TCP_TCC_READ_REQ_sum"] * 64
            )
            bytes_to_l2 = (
                df["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"] * 64
                + df["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"] * 64
                + df["TCP_TCC_WRITE_REQ_sum"] * 64
            )

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            l2_cache_buff_x[0],
                                            vl1_cache_buff_y + vl1_cache_buff_height*2/5
                                        ),
                                        xy=(
                                            vl1_cache_buff_x[1],
                                            vl1_cache_buff_y + vl1_cache_buff_height*2/5
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(vl1_cache_buff_x[1] + ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height*2/5+ibuff_diff,
                                    f"Bytes: {agg_func(bytes_from_l2, Norm.BYTES, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            vl1_cache_buff_x[1],
                                            vl1_cache_buff_y + vl1_cache_buff_height*4/5
                                        ),
                                        xy=(
                                            l2_cache_buff_x[0],
                                            vl1_cache_buff_y + vl1_cache_buff_height*4/5
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(vl1_cache_buff_x[1]+ibuff_diff, vl1_cache_buff_y+vl1_cache_buff_height*4/5+ibuff_diff,
                                    f"Bytes: {agg_func(bytes_to_l2, Norm.BYTES, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            l2_cache_buff_x[0],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height*3/4
                                        ),
                                        xy=(
                                            vl1_cache_buff_x[1],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height*3/4
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(vl1_cache_buff_x[1]+ibuff_diff, sl1d_cache_buff_y+sl1d_cache_buff_height*3/4+ibuff_diff,
                                    f"Rd: {agg_func(df['SQC_TC_DATA_READ_REQ'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.annotate("",
                                        xytext=(
                                            vl1_cache_buff_x[1],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height/2
                                        ),
                                        xy=(
                                            l2_cache_buff_x[0],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height/2
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(vl1_cache_buff_x[1]+ibuff_diff, sl1d_cache_buff_y+sl1d_cache_buff_height/2+ibuff_diff,
                                    f"Wr: {agg_func(df['SQC_TC_DATA_WRITE_REQ'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.annotate("",
                                        xytext=(
                                            l2_cache_buff_x[0],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height/4
                                        ),
                                        xy=(
                                            vl1_cache_buff_x[1],
                                            sl1d_cache_buff_y + sl1d_cache_buff_height/4
                                        ),
                                        arrowprops=dict(arrowstyle="<->", lw=2, color='orange'))
            memory_analysis_ax.text(vl1_cache_buff_x[1]+ibuff_diff, sl1d_cache_buff_y+sl1d_cache_buff_height/4+ibuff_diff,
                                    f"Atomic: {agg_func(df['SQC_TC_DATA_ATOMIC_REQ'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            l2_cache_buff_x[0],
                                            l1i_cache_buff_y + l1i_cache_buff_height/2
                                        ),
                                        xy=(
                                            vl1_cache_buff_x[1],
                                            l1i_cache_buff_y + l1i_cache_buff_height/2
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(vl1_cache_buff_x[1]+ibuff_diff, l1i_cache_buff_y+l1i_cache_buff_height/2+ibuff_diff,
                                    f"Fetch: {agg_func(df['SQC_TC_INST_REQ'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.text(l2_cache_buff_x[0]+ibuff_diff, l2_cache_buff_y+l2_cache_buff_height*3/10,
                                    f"Rd: {agg_func(df['TCC_READ_sum'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(l2_cache_buff_x[0]+ibuff_diff, l2_cache_buff_y+l2_cache_buff_height*4/10,
                                    f"Wr: {agg_func(df['TCC_WRITE_sum'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(l2_cache_buff_x[0]+ibuff_diff, l2_cache_buff_y+l2_cache_buff_height*5/10,
                                    f"Atomic: {agg_func(df['TCC_ATOMIC_sum'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(l2_cache_buff_x[0]+ibuff_diff, l2_cache_buff_y+l2_cache_buff_height*6/10,
                                    f"Hit: {agg_func(df['L2 Hit Rate'], Norm.PERC, duration, bw)}%", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.text(l2_cache_buff_x[0]+ibuff_diff, l2_cache_buff_y+l2_cache_buff_height*7/10,
                                    f"Util: {agg_func(df['L2 Util'], Norm.PERC, duration, bw)}%", color="white", fontsize=fontsize, ha='left')

            # to fabric arrows
            fabric_width = exec_buff_width
            fabric_height = l2_cache_buff_height / 2
            fabric_x = (
                l2_cache_buff_x[1]+wire_len,
                l2_cache_buff_x[1]+wire_len+fabric_width,
            )
            fabric_y = l2_cache_buff_y + fabric_height / 2
            fabric = mpatches.Rectangle(
                (fabric_x[0], fabric_y), fabric_width, fabric_height, edgecolor='orange', facecolor='black', lw=2)
            memory_analysis_ax.text(fabric_x[0]+ibuff_diff, fabric_y+fabric_height+ibuff_diff,
                                    "Fabric", color="white", fontsize=fontsize, ha='left')
            fabric_util = (df.get("L2 Fabric Read Bandwidth Percent of Peak") +
                           df.get("L2 Fabric Write&Atomic Bandwidth Percent of Peak"))
            memory_analysis_ax.text(fabric_x[0]+ibuff_diff, fabric_y+fabric_height/2+ibuff_diff,
                                    f"Util: {agg_func(fabric_util, Norm.PERC, duration, bw)}%", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.add_patch(fabric)
            bytes_to_fabric = (
                (df["TCC_EA0_WRREQ_64B_sum"] * 64)
                + ((df["TCC_EA0_WRREQ_sum"] -
                   df["TCC_EA0_WRREQ_64B_sum"]) * 32)
            )
            bytes_from_fabric = (128 * df.get("TCC_BUBBLE_sum") +
                                 64 * (df.get("TCC_EA0_RDREQ_sum") - df.get("TCC_BUBBLE_sum") - df.get("TCC_EA0_RDREQ_32B_sum")) +
                                 32 * df.get("TCC_EA0_RDREQ_32B_sum"))
            memory_analysis_ax.annotate("",
                                        xytext=(
                                            fabric_x[0],
                                            fabric_y + fabric_height*3/4
                                        ),
                                        xy=(
                                            l2_cache_buff_x[1],
                                            fabric_y + fabric_height*3/4
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(l2_cache_buff_x[1]+ibuff_diff, fabric_y + fabric_height*3/4+ibuff_diff,
                                    f"Bytes: {agg_func(bytes_from_fabric, Norm.BYTES, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            l2_cache_buff_x[1],
                                            fabric_y + fabric_height/2
                                        ),
                                        xy=(
                                            fabric_x[0],
                                            fabric_y + fabric_height/2
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(l2_cache_buff_x[1]+ibuff_diff, fabric_y + fabric_height/2+ibuff_diff,
                                    f"Bytes: {agg_func(bytes_to_fabric, Norm.BYTES, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            l2_cache_buff_x[1],
                                            fabric_y + fabric_height/4
                                        ),
                                        xy=(
                                            fabric_x[0],
                                            fabric_y + fabric_height/4
                                        ),
                                        arrowprops=dict(arrowstyle="<->", lw=2, color='orange'))
            memory_analysis_ax.text(l2_cache_buff_x[1]+ibuff_diff, fabric_y + fabric_height/4+ibuff_diff,
                                    f"Atomic: {agg_func(df['TCC_EA0_ATOMIC_sum'], Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            hbm_width = exec_buff_width
            hbm_height = fabric_height / 2
            hbm_x = (
                fabric_x[1]+wire_len,
                fabric_x[1]+wire_len+hbm_width,
            )
            hbm_y = fabric_y + hbm_height / 2
            hbm = mpatches.Rectangle(
                (hbm_x[0], hbm_y), hbm_width, hbm_height, edgecolor='orange', facecolor='black', lw=2)
            memory_analysis_ax.text(hbm_x[0]+ibuff_diff, hbm_y+hbm_height+ibuff_diff,
                                    "HBM", color="white", fontsize=fontsize, ha='left')
            memory_analysis_ax.add_patch(hbm)

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            hbm_x[0],
                                            hbm_y + hbm_height*2/3
                                        ),
                                        xy=(
                                            fabric_x[1],
                                            hbm_y + hbm_height*2/3
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))

            reqs_to_hbm = df.get("TCC_EA0_WRREQ_DRAM_sum")
            reqs_from_hbm = df.get("TCC_EA0_RDREQ_DRAM_sum")

            memory_analysis_ax.text(fabric_x[1]+ibuff_diff, hbm_y + hbm_height*2/3+ibuff_diff,
                                    f"Req: {agg_func(reqs_from_hbm, Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

            memory_analysis_ax.annotate("",
                                        xytext=(
                                            fabric_x[1],
                                            hbm_y + hbm_height/3
                                        ),
                                        xy=(
                                            hbm_x[0],
                                            hbm_y + hbm_height/3
                                        ),
                                        arrowprops=dict(arrowstyle="->", lw=2, color='orange'))
            memory_analysis_ax.text(fabric_x[1]+ibuff_diff, hbm_y + hbm_height/3+ibuff_diff,
                                    f"Req: {agg_func(reqs_to_hbm, Norm.INSTS, duration, bw)}", color="white", fontsize=fontsize, ha='left')

        if i == n_ibuff-1:
            memory_analysis_ax.text(ibuff_x[0]+ibuff_width/8, ibuff_y[0]+ibuff_height*.75,
                                    "Instr Buff", color='white', fontsize=fontsize, ha='left')
        else:
            ibuff_x = (ibuff_x[0]-ibuff_diff, ibuff_x[1]-ibuff_diff)
            ibuff_y = (ibuff_y[0]-ibuff_diff, ibuff_y[1]-ibuff_diff)
    plt.show()
