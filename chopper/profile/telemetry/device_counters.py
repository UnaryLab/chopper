"""Device-level hardware counter collector using rocprofiler-sdk.

Samples counters in the background (no kernel serialization) alongside
kernel dispatch tracing, all in the same clock domain. Loaded via
LD_PRELOAD, configured via environment variables.
"""

import os
import pathlib
import site
import subprocess
from math import ceil
from pathlib import Path


LIB_NAME = "libchopper_device_counters.so"
MAX_COUNTERS_PER_GROUP = 4


def _get_lib_path() -> str:
    """Locate the device counters shared library.

    Search order:
      1. CHOPPER_DEVICE_LIB env var (explicit override)
      2. Relative to this file: lib/
      3. Installed site-packages
    """
    env_path = os.environ.get("CHOPPER_DEVICE_LIB")
    if env_path and os.path.isfile(env_path):
        return env_path

    pkg_dir = pathlib.Path(__file__).resolve().parent

    candidates = [
        pkg_dir / "lib" / LIB_NAME,
        pkg_dir / LIB_NAME,
    ]

    sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    user_sp = getattr(site, "getusersitepackages", lambda: None)()
    if user_sp:
        sp_dirs.append(user_sp)

    for sp in sp_dirs:
        candidates.append(
            pathlib.Path(sp) / "chopper" / "profile" / "telemetry" / "lib" / LIB_NAME
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched = [str(c) for c in candidates]
    raise FileNotFoundError(
        f"Could not locate {LIB_NAME}. "
        f"Build it with: make -C chopper/profile/telemetry/src\n"
        f"Or set CHOPPER_DEVICE_LIB to the library location.\n"
        f"Searched: {searched}"
    )


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

    if counter_names is not None and len(counter_names) > 0:
        n_counters = len(counter_names)
        leading_zeros = len(str(n_counters))
        n_groups = ceil(n_counters / MAX_COUNTERS_PER_GROUP)

        for gi in range(n_groups):
            group = counter_names[gi * MAX_COUNTERS_PER_GROUP:(gi + 1) * MAX_COUNTERS_PER_GROUP]
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
