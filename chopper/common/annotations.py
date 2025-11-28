import enum
from pandas import DataFrame, Series


class Framework(enum.Enum):
    FSDPv1 = enum.auto()
    FSDPv2 = enum.auto()


def no_overlap_mask(df: DataFrame, framework: Framework = Framework.FSDPv1) -> Series:
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
            return ~df["operator-name"].str.contains(pattern, na=False, regex=True)


# TODO clean up manual patches
def assign_chunks(df: DataFrame) -> DataFrame:
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
            (~df['name'].str.startswith("ncclDevKernel") | df['name_cpu_op'].str.endswith('allreduce')) &
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


# TODO clean up manual patches
def fix_names(df: DataFrame) -> DataFrame:
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


# TODO clean up manual patches
def assign_operator_type(df: DataFrame) -> DataFrame:
    gemm_mask = df['operator-name'].str.endswith('p', na=False) & ~(
        df['operator-name'].str.endswith('Optimizer.step#AdamW.step', na=False))
    fa_mask = df['operator-name'].str.endswith('attn_fa', na=False)
    vec_mask = ~(gemm_mask | fa_mask)

    df.loc[gemm_mask, 'operator-type'] = 'GEMM'
    df.loc[fa_mask, 'operator-type'] = 'FA'
    df.loc[vec_mask, 'operator-type'] = 'Vec'
    return df
