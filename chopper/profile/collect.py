from argparse import ArgumentParser
import subprocess
from math import ceil
from pathlib import Path


def main(counters, nvidia, outdir, program):
    assert nvidia is False, "NVIDIA is not supported currently"
    prof_exe = 'rocprofv3'
    if outdir is None:
        outdir = Path.cwd()
    else:
        outdir = Path(outdir)

    n_counters = len(counters)
    leading_zeros = len(str(n_counters))

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


if __name__ == "__main__":
    parser = ArgumentParser(
        usage='pass program to run and performance counters to collect')
    parser.add_argument(
        '--counters',
        nargs='+',
        required=True,
        help='Name of hardware counters to collect'
    )
    parser.add_argument(
        '--nvidia',
        action='store_true',
        required=False,
        help='Not supported currently'
    )
    parser.add_argument(
        '--out-dir',
        required=False,
        help='directory to put counters'
    )
    parser.add_argument(
        'program',
        nargs='*',
        help='program to run'
    )
    args = parser.parse_args()
    main(args.counters, args.nvidia, args.out_dir, args.program)
