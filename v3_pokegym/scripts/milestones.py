import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

MAP_DATA = Path(__file__).parent.parent / "pokegym" / "map_data.json"
MAPS = {int(r["id"]): r["name"] for r in json.loads(MAP_DATA.read_text())["regions"]}
WIDTH = 30


def label(name):
    kind, _, num = name.partition("_")
    return f"map {MAPS.get(int(num), num)}" if kind == "map" else name


def load(run):
    reached = defaultdict(list)
    with open(Path(run) / "milestones.csv") as f:
        for row in csv.DictReader(f):
            reached[row["milestone"]].append((int(row["global_step"]), float(row["wall_s"])))
    return reached


if __name__ == "__main__":
    runs = {Path(p).name: load(p) for p in sys.argv[1:]}
    order = sorted(
        {m for r in runs.values() for m in r},
        key=lambda m: min(min(r[m])[0] for r in runs.values() if m in r),
    )
    print("milestone".ljust(34) + "".join(f"{name[:WIDTH - 2]:>{WIDTH}}" for name in runs))
    print(" " * 34 + "".join(f"{'envs  first_step  first_h':>{WIDTH}}" for _ in runs))
    for m in order:
        line = label(m).ljust(34)
        for r in runs.values():
            if m in r:
                step, wall = min(r[m])
                line += f"{len(r[m]):>{WIDTH - 22}}{step:>12,}{wall / 3600:>10.2f}"
            else:
                line += f"{'-':>{WIDTH}}"
        print(line)
