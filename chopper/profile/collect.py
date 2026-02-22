"""Main entry point for collecting profiling data.

Orchestrates collection of hardware counters, CPU telemetry, and GPU telemetry
during distributed training runs. Uses the Runner class to manage concurrent
data collection threads.
"""
from argparse import ArgumentParser
from chopper.profile.telemetry import cpu, gpu, counters
from chopper.profile.runner import Runner


def main(program,
    """Collect profiling data during program execution.
    
    Starts telemetry collection threads and runs the target program
    with performance counter instrumentation.
    
    Args:
        program: Command and arguments to execute
        counter_names: List of hardware counter names to collect
        outdir: Output directory for collected data
        container: Container runtime to use (docker/singularity)
        nvidia: If True, collect NVIDIA GPU telemetry
        cpu_telemetry: If True, collect CPU metrics
        gpu_telemetry: If True, collect GPU metrics
        
    Returns:
        Exit code (0 for success, -1 for errors)
    """
         counter_names,
         outdir,
         container,
         nvidia,
         cpu_telemetry,
         gpu_telemetry):
    if len(program) == 0:
        print("Please pass a program to run")
        return -1

    runner = Runner()

    if cpu_telemetry:
        runner.add(
            cpu.main,
            False,
            outdir=outdir,
        )
    if gpu_telemetry:
        runner.add(
            gpu.main,
            False,
            nvidia=nvidia,
            outdir=outdir,
        )
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
    ))
