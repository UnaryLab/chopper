# Chopper - GPU Characterization Tool

A GPU characterization tool that takes [chrome traces](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU) and optionally [hardware counters](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-6.3.0/how-to/using-rocprofv3.html) for multiple GPUs as input, and generates compute, memory, and architecture visualizations.

## Setup

To install chopper run:
```
pip install .
```

## Profiling

The chopper profiler is invoked using its python module.
In this example, GPU and CPU telemetry are enabled.
```
python -m chopper.profile.collect --gpu-telemetry --cpu-telemetry -- <workload args>
```
It is up to the workload to enable the PyTorch profiler.

To only execute the workload args and/or hardware profiler on a container while collecting telemetry outside of it, pass `--container` with the apptainer `.sif` file path.
Otherwise, chopper can be installed and ran inside a container.

## Visualization

To select and view plots run:

```
python -m chopper.window
```

## Tests

To type check with `mypy` run:
```
python -m tests
```

## Related projects
- [HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis)
- [rocm-systems](https://github.com/ROCm/rocm-systems/tree/0e04fdd57165029fcf6d9226d22cf5d3e5370b74)
