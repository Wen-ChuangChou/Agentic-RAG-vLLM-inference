"""Copy a virtualenv to job-local storage; invoke its Python directly afterwards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Stage environments inside a Slurm allocation.')
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination == source or source in destination.parents:
        raise SystemExit('Destination must be outside the source environment.')
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(f'Staging Python environment: {source} -> {destination}', flush=True)
    # Split site-packages by package so metadata reads can overlap. Preserve
    # symlinks, including the module-provided interpreter and lib64 -> lib.
    tasks = []
    for entry in source.iterdir():
        if entry.name != 'lib' or entry.is_symlink():
            tasks.append((entry, destination / entry.name))
            continue
        for version in entry.iterdir():
            target = destination / 'lib' / version.name
            target.mkdir(parents=True)
            for child in version.iterdir():
                if child.name == 'site-packages':
                    (target / child.name).mkdir()
                    tasks.extend((item, target / child.name / item.name) for item in child.iterdir())
                else:
                    tasks.append((child, target / child.name))

    def copy(pair):
        src, dst = pair
        if src.is_symlink():
            dst.symlink_to(os.readlink(src))
        elif src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy, tasks))
    print(f'Environment staged in {time.monotonic() - started:.1f}s', flush=True)
    print(f'Use {destination / "bin/python"} (activation scripts may retain source paths).', flush=True)


if __name__ == '__main__':
    main()
