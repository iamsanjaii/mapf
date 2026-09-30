"""
tests/test_astar.py
~~~~~~~~~~~~~~~~~~~~
Unit tests for the A* planner and heuristics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.environment.grid import Grid, CellType
from src.planning.astar import AStarPlanner
from src.planning.heuristics import manhattan, euclidean, chebyshev, get_heuristic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """5×5 empty grid."""
    g = Grid(5, 5)
    return g


@pytest.fixture
def blocked_grid():
    """5×5 grid with a wall across the middle — goal still reachable via detour."""
    g = Grid(5, 5)
    for c in range(4):
        g.set(2, c, CellType.OBSTACLE)
    return g


@pytest.fixture
def isolated_grid():
    """5×5 grid where goal is completely surrounded by obstacles."""
    g = Grid(5, 5)
    g.set(0, 4, CellType.OBSTACLE)
    g.set(1, 4, CellType.OBSTACLE)
    g.set(1, 3, CellType.OBSTACLE)
    return g


# ---------------------------------------------------------------------------
# Heuristic tests
# ---------------------------------------------------------------------------

class TestHeuristics:
    def test_manhattan_same_cell(self):
        assert manhattan((3, 3), (3, 3)) == 0

    def test_manhattan_horizontal(self):
        assert manhattan((0, 0), (0, 5)) == 5

    def test_manhattan_diagonal(self):
        assert manhattan((0, 0), (3, 4)) == 7

    def test_euclidean_same_cell(self):
        assert euclidean((2, 2), (2, 2)) == pytest.approx(0.0)

    def test_euclidean_horizontal(self):
        assert euclidean((0, 0), (0, 3)) == pytest.approx(3.0)

    def test_chebyshev_diagonal(self):
        # Chebyshev(0,0 → 3,4) = max(3,4) = 4
        assert chebyshev((0, 0), (3, 4)) == 4

    def test_get_heuristic_returns_callable(self):
        h = get_heuristic("manhattan")
        assert callable(h)

    def test_get_heuristic_unknown_raises(self):
        with pytest.raises(ValueError):
            get_heuristic("unknown_heuristic")

    def test_admissibility_manhattan_4dir(self, small_grid):
        """Manhattan never overestimates on a 4-directional grid."""
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), manhattan)
        assert result.success
        h_estimate = manhattan((0, 0), (4, 4))
        assert h_estimate <= result.cost + 1e-9, "Manhattan overestimated!"


# ---------------------------------------------------------------------------
# A* planner tests
# ---------------------------------------------------------------------------

class TestAStarPlanner:
    def test_finds_path_empty_grid(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), manhattan)
        assert result.success
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (4, 4)

    def test_path_is_connected(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), manhattan)
        path = result.path
        for i in range(len(path) - 1):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            dist = abs(r0 - r1) + abs(c0 - c1)
            assert dist == 1, f"Non-adjacent steps at index {i}: {path[i]} → {path[i+1]}"

    def test_start_equals_goal(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (2, 2), (2, 2), manhattan)
        assert result.success
        assert result.path == [(2, 2)]
        assert result.cost == 0.0

    def test_unreachable_goal_returns_none(self):
        """Goal surrounded by obstacles is unreachable."""
        g = Grid(5, 5)
        # Surround (4,4)
        g.set(3, 4, CellType.OBSTACLE)
        g.set(4, 3, CellType.OBSTACLE)
        planner = AStarPlanner()
        result = planner.plan(g, (0, 0), (4, 4), manhattan)
        assert not result.success
        assert result.path is None

    def test_detour_around_wall(self, blocked_grid):
        planner = AStarPlanner()
        result = planner.plan(blocked_grid, (0, 0), (4, 4), manhattan)
        assert result.success
        # Path must go through column 4 (gap in wall)
        assert (2, 4) in result.path

    def test_cost_equals_path_length_minus_one(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), manhattan)
        assert result.cost == pytest.approx(len(result.path) - 1)

    def test_nodes_expanded_positive(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), manhattan)
        assert result.nodes_expanded > 0

    def test_runtime_positive(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), manhattan)
        assert result.runtime_ms >= 0.0

    def test_forbidden_cells_respected(self, small_grid):
        """Planner must not enter forbidden cells."""
        planner = AStarPlanner()
        # Direct path (0,0) → (0,4) goes through columns; forbid all of row 0
        forbidden = {(0, c) for c in range(1, 5)}
        result = planner.plan(small_grid, (0, 0), (0, 4), manhattan, forbidden=forbidden)
        if result.path:
            for cell in result.path[1:]:  # skip start
                assert cell not in forbidden, f"Entered forbidden cell {cell}"

    def test_euclidean_heuristic_finds_path(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), euclidean)
        assert result.success

    def test_chebyshev_heuristic_finds_path(self, small_grid):
        planner = AStarPlanner()
        result = planner.plan(small_grid, (0, 0), (4, 4), chebyshev)
        assert result.success

    def test_out_of_bounds_start(self):
        g = Grid(5, 5)
        planner = AStarPlanner()
        result = planner.plan(g, (-1, 0), (4, 4), manhattan)
        assert not result.success

    def test_obstacle_not_entered(self):
        g = Grid(5, 5)
        g.set(0, 1, CellType.OBSTACLE)
        g.set(1, 0, CellType.OBSTACLE)
        planner = AStarPlanner()
        # (0,0) → (2,2): only route via (0,2) and (1,2)
        result = planner.plan(g, (0, 0), (2, 2), manhattan)
        if result.path:
            for cell in result.path:
                assert g.get(*cell) != CellType.OBSTACLE, \
                    f"Path entered obstacle at {cell}"
