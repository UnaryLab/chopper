"""Device-level hardware counter collector using rocprofiler-sdk.

Samples counters in the background (no kernel serialization) alongside
kernel dispatch tracing, all in the same clock domain. Loaded via
LD_PRELOAD, configured via environment variables.
"""

import os
import pathlib
import subprocess
from math import ceil
from pathlib import Path


LIB_NAME = "libchopper_device_counters.so"
MAX_COUNTERS_PER_GROUP = 4


def _get_lib_path() -> str:
    """Locate the device counters shared library.

    With editable installs, __file__ points to the source tree but the
    .so lives in site-packages. Check both.
    """
    env_path = os.environ.get("CHOPPER_DEVICE_LIB")
    if env_path and os.path.isfile(env_path):
        return env_path

    pkg_dir = pathlib.Path(__file__).resolve().parent
    source_lib = pkg_dir / "lib" / LIB_NAME
    if source_lib.is_file():
        return str(source_lib)

    # Editable install: .so is in site-packages, not source tree
    import importlib.util
    spec = importlib.util.find_spec("chopper.profile.telemetry")
    assert spec is not None and spec.origin is not None, (
        f"Cannot find chopper.profile.telemetry package"
    )
    installed_lib = pathlib.Path(spec.origin).parent / "lib" / LIB_NAME
    assert installed_lib.is_file(), (
        f"{LIB_NAME} not found at {source_lib} or {installed_lib}. "
        f"Build with: pip install -e . (or set CHOPPER_DEVICE_LIB)"
    )
    return str(installed_lib)


def _build_groups(counter_names: list | None) -> list[list[str]]:
    """Build counter groups from CLI input.

    counter_names comes from argparse with action='append', nargs='+':
      - None: no --counters flag
      - [['A', 'B', 'C', 'D', 'E']]: single --counters, auto-group by 4
      - [['A', 'B'], ['C', 'D']]: multiple --counters, explicit groups

    Returns a list of groups, where each group is a list of counter names.
    """
    if counter_names is None or len(counter_names) == 0:
        return []

    assert isinstance(counter_names[0], list), (
        f"Expected list of lists from argparse, got {type(counter_names[0])}"
    )

    if len(counter_names) == 1:
        # Single --counters flag: auto-group by MAX_COUNTERS_PER_GROUP
        flat = counter_names[0]
        n_groups = ceil(len(flat) / MAX_COUNTERS_PER_GROUP)
        groups = []
        for gi in range(n_groups):
            group = flat[gi * MAX_COUNTERS_PER_GROUP:(gi + 1) * MAX_COUNTERS_PER_GROUP]
            groups.append(group)
        return groups

    # Multiple --counters flags: explicit groups, assert each <= 4
    for gi, group in enumerate(counter_names):
        assert len(group) <= MAX_COUNTERS_PER_GROUP, (
            f"Group {gi} has {len(group)} counters ({group}), "
            f"max is {MAX_COUNTERS_PER_GROUP}"
        )
        assert len(group) > 0, f"Group {gi} is empty"
    return counter_names


def main(
    stop: bool,
    program,
    counter_names,
    outdir,
    container,
    nvidia,
    sample_ms=1,
):
    """Collect device-level hardware counters.

    This is the "stopper" process -- it runs the user's workload and
    triggers shutdown of all other collectors when it finishes.

    Groups counters into hardware-compatible batches and runs the
    workload once per batch with the device counter tool loaded.
    """
    assert nvidia is False, "NVIDIA device counters are not supported currently"

    lib_path = _get_lib_path()

    container_args: list[str] = []
    if container is not None and container.endswith(".sif"):
        container_args.extend(("apptainer", "exec", "--rocm", container))

    if outdir is None:
        outdir = Path.cwd()
    else:
        outdir = Path(outdir)

    groups = _build_groups(counter_names)
    if len(groups) > 0:
        leading_zeros = max(1, len(str(len(groups) - 1)))

        for gi, group in enumerate(groups):
            dir_num = str(gi).zfill(leading_zeros)
            counter_dir = outdir / f"chopper_device_counters{dir_num}"
            counter_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["CHOPPER_COUNTERS"] = ",".join(group)
            env["CHOPPER_SAMPLE_MS"] = str(sample_ms)
            env["CHOPPER_TRACE_OUTPUT"] = str(counter_dir / "kernel_traces.csv")
            env["CHOPPER_COUNTER_OUTPUT"] = str(counter_dir / "counter_samples.csv")

            if "LD_PRELOAD" in env:
                env["LD_PRELOAD"] = lib_path + ":" + env["LD_PRELOAD"]
            else:
                env["LD_PRELOAD"] = lib_path

            proc_args = (*container_args, *program)
            subprocess.run(proc_args, env=env, check=True)
    else:
        # No counters specified, just run the program
        subprocess.run((*container_args, *program), check=True)
