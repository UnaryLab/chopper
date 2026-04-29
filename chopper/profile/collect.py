from argparse import ArgumentParser
from chopper.profile.telemetry import cpu, gpu, counters, device_counters
from chopper.profile.runner import Runner


def main(program,
         counter_names,
         outdir,
         container,
         nvidia,
         cpu_telemetry,
         gpu_telemetry,
         device,
         telemetry_on=0.0,
         telemetry_off=0.1,
         sample_ms=1):
    if len(program) == 0:
        print("Please pass a program to run")
        return -1

    runner = Runner()

    if cpu_telemetry:
        runner.add(
            cpu.main,
            False,
            outdir=outdir,
            on=telemetry_on,
            off=telemetry_off,
        )
    if gpu_telemetry:
        runner.add(
            gpu.main,
            False,
            nvidia=nvidia,
            outdir=outdir,
            on=telemetry_on,
            off=telemetry_off,
        )

    if device:
        runner.add(
            device_counters.main,
            True,
            program,
            counter_names,
            outdir,
            container,
            nvidia,
            sample_ms,
        )
    else:
        runner.add(
            counters.main,
            True,
            program,
            counter_names,
            outdir,
            container,
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
        '--output-dir',
        required=False,
        default=".",
        help='directory to put counters'
    )
    parser.add_argument(
        '--container',
        required=False,
        help="Container image to use"
    )
    parser.add_argument(
        '--device',
        action='store_true',
        required=False,
        help='Use device-level counter sampling (no kernel serialization) instead of rocprofv3 dispatch profiling'
    )
    parser.add_argument(
        '--sample-ms',
        type=int,
        default=1,
        help='Device counter sampling interval in ms (default: 1, only used with --device)'
    )
    parser.add_argument(
        '--telemetry-on',
        type=float,
        default=0.0,
        help='duration (seconds) to sample continuously before pausing (default: 0.0)',
    )
    parser.add_argument(
        '--telemetry-off',
        type=float,
        default=0.1,
        help='sleep duration (seconds) between samples (default: 0.1 = 10 Hz)',
    )
    parser.add_argument(
        'program',
        nargs='*',
        help='program to run',
    )
    args = parser.parse_args()
    exit(main(
        args.program,
        args.counters,
        args.output_dir,
        args.container,
        args.nvidia,
        args.cpu_telemetry,
        args.gpu_telemetry,
        args.device,
        args.telemetry_on,
        args.telemetry_off,
        args.sample_ms,
    ))
