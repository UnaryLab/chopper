import enum
from dataclasses import dataclass
from pandas import DataFrame, Series


@dataclass
class PaperMode:
    """Paper mode settings for publication-quality figures.

    Attributes:
        enabled: Whether paper mode is enabled
        left: Left margin for subplot adjustment
        right: Right margin for subplot adjustment
        bottom: Bottom margin for subplot adjustment
        top: Top margin for subplot adjustment
        wspace: Width spacing between subplots
        hspace: Height spacing between subplots
        ncol: Number of paper columns (width multiplier)
        figsize_ratio: Height/width ratio for figure sizing
    """
    enabled: bool = False
    left: float = 0.1
    right: float = 0.9
    bottom: float = 0.1
    top: float = 0.9
    wspace: float = 0.2
    hspace: float = 0.3
    ncol: int = 1
    figsize_ratio: float = 1.0
    legend_bbox: tuple[float, float] | None = None


class Framework(enum.IntEnum):
    """Training framework enumeration for trace processing.
    
    Attributes:
        FSDPv1: PyTorch FSDP version 1
        FSDPv2: PyTorch FSDP version 2
    """
    FSDPv1 = 1
    FSDPv2 = 2


def no_overlap_mask(df: DataFrame, framework: Framework = Framework.FSDPv1) -> Series:
    """Create a boolean mask for non-overlapping computation kernels.
    
    Identifies kernels that do not overlap with communication operations,
    based on framework-specific operator patterns and NCCL kernel names.
    
    Args:
        df: DataFrame containing trace data with 'name' and 'operator-name' columns
        framework: Framework enum specifying which framework patterns to use
        
    Returns:
        Boolean Series where True indicates non-overlapping kernels
    """
    nccl_mask = df["name"].str.startswith("ncclDevKernel")
    match framework:
        case Framework.FSDPv1:
            return ~(df["operator-name"].isin((
                'b_FullyShardedDataParallel._pre_forward',
                'f_FullyShardedDataParallel._post_backward_hook',
                'f_FullyShardedDataParallel._pre_backward_prefetch',
            )) | nccl_mask)
        case Framework.FSDPv2:
            pattern = '|'.join((
                'FSDP::post_backward_reduce',
                'FSDP::pre_forward',
                'FSDP::all_gather_copy_out',
                'FSDP::all_gather',
            ))
            return ~(
                df["operator-name"].str.contains(pattern, na=False, regex=True)
                | nccl_mask
            )


def assign_chunks(df: DataFrame) -> DataFrame:
    """Assign training phase chunks (forward, backward, optimizer) to trace events.
    
    Categorizes operators into forward pass, backward pass, or optimizer step
    based on operator name patterns.
    
    Args:
        df: DataFrame containing trace data with 'operator-name' column
        
    Returns:
        DataFrame with added 'chunk' column containing 'fwd', 'bwd', or 'opt'
    """
    opt_grad_mask = (df['operator-name'].str.startswith(
        'f_Optimizer', na=False) |
        df['operator-name'].str.startswith(
        'f_b_ga', na=False) |
        df['operator-name'].str.startswith(
        'f_b_ar', na=False)
    )

    bwd_mask = (
        (df['operator-name'].str.startswith('b_', na=False) & ((df['operator-name'] != 'b_FullyShardedDataParallel._pre_forward')) |
         (
             (df['operator-name'] == 'b_FullyShardedDataParallel._pre_forward') &
            (~df['name'].str.startswith("ncclDevKernel", na=False) | df['name_cpu_op'].str.endswith('allreduce', na=False)) &
             (df['name_cpu_op'] != 'aten::copy_')
        )) |
        (df['operator-name'] == 'f_FullyShardedDataParallel._post_backward_hook') |
        (df['operator-name'] == 'f_FullyShardedDataParallel._pre_backward_prefetch') |
        df['operator-name'].str.startswith('f_b_', na=False)
    ) & ~opt_grad_mask

    fwd_mask = ~opt_grad_mask & ~bwd_mask & ~df['operator-name'].isna()

    df.loc[fwd_mask, 'chunk'] = 'fwd'
    df.loc[bwd_mask, 'chunk'] = 'bwd'
    df.loc[opt_grad_mask, 'chunk'] = 'opt'
    return df


def fix_names(df: DataFrame) -> DataFrame:
    """Normalize operator names for consistent analysis.
    
    Applies name transformations to standardize operator naming conventions,
    such as removing redundant prefixes and normalizing layer names.
    
    Args:
        df: DataFrame containing trace data with 'operator-name' column
        
    Returns:
        DataFrame with normalized operator names
    """
    fix_mask = df['operator-name'].str.startswith('f_b_', na=False)
    df.loc[fix_mask, 'operator-name'] = df.loc[
        fix_mask,
        'operator-name'].str.replace('f_b_', 'b_')

    fix_mask = df['operator-name'].str.endswith(
        'Optimizer.step#AdamW.step', na=False)
    df.loc[fix_mask, 'operator-name'] = 'opt_step'

    fix_mask = df['operator-name'].str.contains('_fc_', na=False)
    df.loc[fix_mask, 'operator-name'] = df.loc[
        fix_mask,
        'operator-name'].str.replace('_fc_', '_mlp_')
    return df


def assign_operator_type(df: DataFrame) -> DataFrame:
    """Categorize operators by computational type.
    
    Classifies operators into GEMM (matrix multiply), FlashAttention, or
    vectorized operations based on operator name patterns.
    
    Args:
        df: DataFrame containing trace data with 'operator-name' column
        
    Returns:
        DataFrame with added 'operator-type' column containing 'GEMM', 'FA', or 'Vec'
    """
    gemm_mask = df['operator-name'].str.endswith('p', na=False) & ~(
        df['operator-name'].str.endswith('Optimizer.step#AdamW.step', na=False))
    fa_mask = df['operator-name'].str.endswith('attn_fa', na=False)
    vec_mask = ~(gemm_mask | fa_mask)

    df.loc[gemm_mask, 'operator-type'] = 'GEMM'
    df.loc[fa_mask, 'operator-type'] = 'FA'
    df.loc[vec_mask, 'operator-type'] = 'Vec'
    return df
