from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayoutStats:
    rows: int
    cols: int
    shelves: int
    goals: int


def read_layout(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty layout: {path}")
    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError(f"Layout must be rectangular: {path}")
    for line in lines:
        for char in line:
            if char.lower() not in ".xg":
                raise ValueError(f"Unsupported layout character {char!r} in {path}")
    return lines


def layout_stats(lines: list[str]) -> LayoutStats:
    return LayoutStats(
        rows=len(lines),
        cols=len(lines[0]),
        shelves=sum(line.count("x") + line.count("X") for line in lines),
        goals=sum(line.count("g") + line.count("G") for line in lines),
    )


def build_similar_training_layout(reference_lines: list[str], variant: int = 0) -> list[str]:
    """Build a training layout with the same warehouse family but different cells.

    The course layout is kept as the held-out target map.  This generator keeps
    the target map's global properties: grid size, three storage zones, five
    two-column shelf stacks per zone, stack height, and the same number of
    bottom workstations.  It then shifts shelf stacks and storage bands by small
    deterministic offsets so the policy trains on a related but non-identical
    RWARE instance.
    """
    stats = layout_stats(reference_lines)
    shelf_bands = _shelf_bands(reference_lines)
    stack_starts = _stack_starts(reference_lines, shelf_bands[0])
    if len(shelf_bands) != 3 or len(stack_starts) != 5:
        raise ValueError(
            "The training layout generator expects the course-style map: "
            "3 shelf bands and 5 two-column stacks per band."
        )

    output = [["." for _ in range(stats.cols)] for _ in range(stats.rows)]
    for y, line in enumerate(reference_lines):
        for x, char in enumerate(line):
            if char.lower() == "g":
                output[y][x] = "g"

    # Keep each training map close to the target distribution.  Only one shelf
    # band is shifted per variant; using large shifts on every band made the
    # local follower learn a layout-specific, overly conservative WAIT bias.
    variants = [
        {"band": 0, "dx": -1, "dy": 0},
        {"band": 1, "dx": 1, "dy": 0},
        {"band": 2, "dx": -1, "dy": 0},
        {"band": 0, "dx": 0, "dy": 1},
        {"band": 1, "dx": 0, "dy": -1},
        {"band": 2, "dx": 0, "dy": -1},
        {"band": 0, "dx": 1, "dy": 0},
        {"band": 2, "dx": 1, "dy": 0},
    ]
    selected = variants[variant % len(variants)]
    for band_idx, (start_y, end_y) in enumerate(shelf_bands):
        height = end_y - start_y + 1
        shifted_y = start_y + (selected["dy"] if band_idx == selected["band"] else 0)
        if shifted_y < 0 or shifted_y + height > stats.rows - 1:
            raise ValueError("Generated shelf band would leave the map.")
        for stack_idx, start_x in enumerate(stack_starts):
            dx = selected["dx"] if band_idx == selected["band"] else 0
            shifted_x = min(max(1, start_x + dx), stats.cols - 3)
            for y in range(shifted_y, shifted_y + height):
                for x in range(shifted_x, shifted_x + 2):
                    if output[y][x] == "g":
                        raise ValueError("Generated shelf overlaps a goal.")
                    output[y][x] = "x"

    generated = ["".join(row) for row in output]
    generated_stats = layout_stats(generated)
    if generated_stats.shelves != stats.shelves or generated_stats.goals != stats.goals:
        raise ValueError(
            "Generated training layout changed shelf/goal counts: "
            f"{generated_stats} vs reference {stats}"
        )
    if generated == reference_lines:
        raise ValueError("Generated training layout unexpectedly equals target layout.")
    return generated


def ensure_similar_training_layouts(
    reference_path: Path,
    output_dir: Path,
    count: int,
    force: bool = False,
) -> list[Path]:
    if count < 1:
        raise ValueError("count must be >= 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_lines = read_layout(Path(reference_path))
    paths: list[Path] = []
    seen: set[tuple[str, ...]] = set()
    variant = 0
    max_attempts = 64
    while len(paths) < count and variant < max_attempts:
        generated = build_similar_training_layout(reference_lines, variant=variant)
        variant += 1
        key = tuple(generated)
        if key in seen:
            continue
        seen.add(key)
        path = output_dir / f"train_map_similar_{len(paths):02d}.txt"
        if force or not path.exists():
            path.write_text("\n".join(generated) + "\n", encoding="utf-8")
        paths.append(path)
    if len(paths) < count:
        raise ValueError(f"Only generated {len(paths)} unique training layouts; requested {count}.")
    return paths


def _shelf_bands(lines: list[str]) -> list[tuple[int, int]]:
    rows_with_shelves = [
        y for y, line in enumerate(lines) if any(char.lower() == "x" for char in line)
    ]
    bands: list[tuple[int, int]] = []
    start = prev = rows_with_shelves[0]
    for row in rows_with_shelves[1:]:
        if row == prev + 1:
            prev = row
            continue
        bands.append((start, prev))
        start = prev = row
    bands.append((start, prev))
    return bands


def _stack_starts(lines: list[str], band: tuple[int, int]) -> list[int]:
    start_y, end_y = band
    cols = len(lines[0])
    shelf_cols = []
    for x in range(cols):
        if all(lines[y][x].lower() == "x" for y in range(start_y, end_y + 1)):
            shelf_cols.append(x)
    starts: list[int] = []
    idx = 0
    while idx < len(shelf_cols):
        start = shelf_cols[idx]
        end = start
        idx += 1
        while idx < len(shelf_cols) and shelf_cols[idx] == end + 1:
            end = shelf_cols[idx]
            idx += 1
        if end - start + 1 != 2:
            raise ValueError("Expected two-column shelf stacks in the reference layout.")
        starts.append(start)
    return starts
