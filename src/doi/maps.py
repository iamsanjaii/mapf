"""MovingAI .map loader (Stage 1 benchmark maps)."""
from src.environment.grid import CellType, Grid

FREE_CHARS = frozenset(".G")
OBSTACLE_CHARS = frozenset("@OTW")


def load_movingai(path: str) -> Grid:
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    height = width = None
    i = 0
    while i < len(lines) and lines[i].strip() != "map":
        key, _, value = lines[i].partition(" ")
        if key == "height":
            height = int(value)
        elif key == "width":
            width = int(value)
        i += 1
    if height is None or width is None or i >= len(lines):
        raise ValueError(f"{path}: not a MovingAI map")
    rows = lines[i + 1:i + 1 + height]
    if len(rows) != height or any(len(r) < width for r in rows):
        raise ValueError(f"{path}: expected {height} rows of {width} cells")
    grid = Grid(width, height)
    for r, row in enumerate(rows):
        for c in range(width):
            ch = row[c]
            if ch in OBSTACLE_CHARS:
                grid.set(r, c, CellType.OBSTACLE)
            elif ch not in FREE_CHARS:
                raise ValueError(f"{path}: unknown cell character {ch!r} at ({r}, {c})")
    return grid


def write_synthetic_warehouse(path: str, aisles: int = 5, shelf_rows: int = 4, width: int = 31) -> str:
    """Write a small MovingAI-format warehouse (shelf blocks with one-cell gaps) for pilots and tests."""
    height = aisles * (shelf_rows + 1) + 1
    rows = []
    for r in range(height):
        if r % (shelf_rows + 1) == 0:
            rows.append("." * width)
        else:
            row = ["@"] * width
            for c in range(0, width, 5):
                row[c] = "."
            row[0] = row[-1] = "."
            rows.append("".join(row))
    with open(path, "w") as f:
        f.write(f"type octile\nheight {height}\nwidth {width}\nmap\n" + "\n".join(rows) + "\n")
    return path
