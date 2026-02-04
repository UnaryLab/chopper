from time import monotonic_ns
from time import sleep
import psutil
import pandas as pd


def main(
    stop: bool,
    filename: str = 'cpu.pkl',
    outdir: str = '.',
    on: float = 0.0,
    off: float = 0.1,
    **kwargs,
):
    assert hasattr(psutil.Process, "cpu_num"), "not supported"

    num_cpus = psutil.cpu_count()

    results = []
    pause_ts = monotonic_ns() + int(on * 1e9)

    while not stop.value:
        ts = monotonic_ns()
        cpu_entries = {}
        cpus_percent = psutil.cpu_percent(percpu=True)

        for cpu_num in range(num_cpus):
            cpu_entries[cpu_num] = {'percent': cpus_percent[cpu_num]}

        for p in psutil.process_iter(['name', 'cmdline', 'cpu_num']):
            (
                cpu_entries[p.info['cpu_num']]
                .setdefault('name', [])
                .append(p.info['name'])
            )
            (
                cpu_entries[p.info['cpu_num']]
                .setdefault('cmdline', [])
                .append(p.info['cmdline'])
            )

        for cpu, entry in cpu_entries.items():
            results.append({'cpu': cpu, 'ts': ts, **entry})

        if ts >= pause_ts:
            sleep(off)
            pause_ts = monotonic_ns() + int(on * 1e9)

    pd.DataFrame(results).to_pickle(f"{outdir}/{filename}")
