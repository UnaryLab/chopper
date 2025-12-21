from argparse import ArgumentParser
from chopper.profile import telemetry
from chopper.profile.runner import Runner


def main(counters, nvidia, cpu_telemetry, gpu_telemetry, outdir, program):
    runner = Runner()

    if cpu_telemetry:
        runner.add(
            telemetry.cpu.main,
            False,
            outdir=outdir,
        )
    if gpu_telemetry:
        runner.add(
            telemetry.gpu.main,
            False,
            nvidia=nvidia,
            outdir=outdir,
        )
    runner.add(
        telemetry.counters.main,
        True,
        program,
        counters,
        outdir,
        nvidia,
    )
    runner.start()
    runner.join()


if __name__ == "__main__":
    parser = ArgumentParser(
        usage='pass program to run and what to collect (i.e., hardware counters, CPU and GPU telemetry)',
    )
    parser.add_argument(
        '--counters',
        nargs='+',
        required=False,
        help='Name of hardware counters to collect'
    )
    parser.add_argument(
        '--nvidia',
        action='store_true',
        required=False,
        help='Not supported currently'
    )
    parser.add_argument(
        '--cpu-telemetry',
        action='store_true',
        required=False,
        help='collect CPU telemetry'
    )
    parser.add_argument(
        '--gpu-telemetry',
        action='store_true',
        required=False,
        help='collect GPU telemetry'
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
    main(
        args.counters,
        args.nvidia,
        args.cpu_telemetry,
        args.cpu_telemetry,
        args.out_dir,
        args.program,
    )
