from pathlib import Path
import subprocess
from math import ceil
from chopper.common.printing import info


def main(
    stop: bool,
    program,
    counters,
    outdir,
    nvidia,
):
    assert nvidia is False, "NVIDIA counters are not supported currently"
    prof_exe = 'rocprofv3'  # TODO change for nvidia

    if counters is not None:
        n_counters = len(counters)
        assert n_counters > 0
        leading_zeros = len(str(n_counters))

        if outdir is None:
            outdir = Path.cwd()
        else:
            outdir = Path(outdir)

        # run up to three counters at a time
        # TODO make custom
        citers = ceil(n_counters / 3)
        for citer in range(citers):
            iter_counters = counters[citer*3:(citer+1)*3]
            dir_num = str(citer).zfill(leading_zeros)
            counter_dir = outdir / f'chopper_counters{dir_num}'
            proc_args = (
                prof_exe,
                '--pmc',
                ','.join(iter_counters),
                '-d',
                str(counter_dir.resolve()),
                '--',
                *program
            )
            subprocess.run(
                proc_args,
                check=True,
            )
    else:
        subprocess.run(
            program,
            check=True,
        )
