from chopper.common.load import get_df
from chopper.common.colors import okabe_ito
from chopper.common.cache import load_pickle
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from chopper.common.annotations import Framework, no_overlap_mask

def get_data(
    ts_files: list[str] = ("./ts.pkl",),
    variants: list[str] = ("default",),
    frameworks: list[Framework] = (Framework.FSDPv2,),
):
    dfs = [
        get_df(ts_file, framework=fw)
        for ts_file, fw in zip(ts_files, frameworks)
    ]
    return dfs, variants, frameworks

def draw(
    fig: Figure,
    input_data,
    y_shrink: float = 1.0,
    y_max: float = float("inf"),
    y_min: float = float("-inf"),
    alpha: float = 1.0,
    idx_start: int = 0,
    idx_end: int = -1,
    comm_kern_filter: list[str] = [],
):
    dfs, variants, frameworks = input_data

    assert len(dfs) == 1 and len(variants) == 1, "Only visualizing one dataframe"

    framework = frameworks[0]
    comm_kerns = None
    gpus = None
    for df in dfs:
        ovr_mask = no_overlap_mask(df, framework=framework)
        if comm_kerns is None:
            assert gpus is None
            if len(comm_kern_filter):
                comm_kerns = sorted(
                    kn
                    for kn in df[~ovr_mask]["name_cpu_op"].unique()
                    if any(kf in kn for kf in comm_kern_filter)
                )
            else:
                comm_kerns = sorted(df[~ovr_mask]["name_cpu_op"].unique())
            gpus = sorted(df["gpu"].unique())
        else:
            if len(comm_kern_filter):
                assert comm_kerns == sorted(
                    kn
                    for kn in df[~ovr_mask]["name_cpu_op"].unique()
                    if any(kf in kn for kf in comm_kern_filter)
                )
            else:
                assert comm_kerns == sorted(df[~ovr_mask]["name_cpu_op"].unique())
            assert gpus == sorted(df["gpu"].unique())

    n_cols = len(gpus)
    n_rows = len(comm_kerns)

    df = dfs[0]

    iters = sorted(df["iteration"].unique())
    df = df[df["iteration"].isin(iters[idx_start:idx_end])]
    ovr_mask = no_overlap_mask(df, framework=framework)

    fig.clear()
    axs = tuple(tuple(fig.add_subplot(n_rows, n_cols, i*n_cols+j+1)
                for j in range(n_cols)) for i in range(n_rows))
    gy_min, gy_max = float("inf"), float("-inf")
    gx_min, gx_max = float("inf"), float("-inf")
    for gpu in gpus:
        for i, comm_kern in enumerate(comm_kerns):
            ax = axs[i][gpu]
            tmp_df = df[~ovr_mask][df.loc[~ovr_mask, "name_cpu_op"] == comm_kern]
            gpu_mask = tmp_df["gpu"] == gpu
            ax.set_title(comm_kern)
            ax.scatter(
                tmp_df.loc[gpu_mask, "ts"],
                tmp_df.loc[gpu_mask, "dur"],
                # color=list(okabe_ito)[i],
                color='black',
                s=0.5,
                alpha=alpha,
            )
            ymin_ax, ymax_ax = ax.get_ylim()
            xmin_ax, xmax_ax = ax.get_xlim()
            gx_min = min(gx_min, xmin_ax)
            gx_max = max(gx_max, xmax_ax)
            gy_min = min(gy_min, ymin_ax)
            gy_max = max(gy_max, ymax_ax)
    for gpu in gpus:
        for i in range(len(comm_kerns)):
            if y_min != float('-inf') and y_max != float('inf'):
                axs[i][gpu].set_ylim(y_min*y_shrink, y_max*y_shrink)
            else:
                axs[i][gpu].set_ylim(gy_min*y_shrink, gy_max*y_shrink)
            axs[i][gpu].set_xlim(gx_min, gx_max)
