"""
tests/test_mapf.py
~~~~~~~~~~~~~~~~~~~
Unit tests for multi-robot planning (prioritised MAPF and coordination).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.environment.grid import Grid, CellType
from src.environment.generator import EnvironmentGenerator
from src.mapf.conflict import ConflictDetector
from src.mapf.prioritized import PrioritizedPlanner
from src.mapf.reservation import ReservationTable
from src.robots.manager import RobotManager
from src.robots.robot import Robot, RobotStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid_5():
    return Grid(5, 5)


@pytest.fixture
def two_robot_manager():
    starts = [(0, 0), (4, 4)]
    goals  = [(4, 4), (0, 0)]
    return RobotManager(starts, goals)


# ---------------------------------------------------------------------------
# ReservationTable tests
# ---------------------------------------------------------------------------

class TestReservationTable:
    def test_reserve_and_check(self):
        rt = ReservationTable()
        rt.reserve_path(robot_id=0, path=[(0, 0), (0, 1), (0, 2)])
        assert rt.is_reserved((0, 1), t=1, exclude_robot=1)
        assert not rt.is_reserved((0, 1), t=1, exclude_robot=0)

    def test_goal_lock_persists(self):
        rt = ReservationTable()
        rt.reserve_path(robot_id=0, path=[(0, 0), (0, 1)])
        # Goal is (0,1), locked from t=1 onward
        assert rt.is_reserved((0, 1), t=5, exclude_robot=1)

    def test_clear_robot(self):
        rt = ReservationTable()
        rt.reserve_path(robot_id=0, path=[(0, 0), (0, 1), (0, 2)])
        rt.clear_robot(0)
        assert not rt.is_reserved((0, 1), t=1, exclude_robot=1)

    def test_forbidden_cells_at(self):
        rt = ReservationTable()
        rt.reserve_path(robot_id=0, path=[(1, 0), (1, 1), (1, 2)])
        forbidden = rt.forbidden_cells_at(t=1, exclude_robot=1)
        assert (1, 1) in forbidden

    def test_no_self_conflict(self):
        rt = ReservationTable()
        rt.reserve_path(robot_id=0, path=[(0, 0), (0, 1)])
        assert not rt.is_reserved((0, 0), t=0, exclude_robot=0)


# ---------------------------------------------------------------------------
# RobotManager tests
# ---------------------------------------------------------------------------

class TestRobotManager:
    def test_robots_created(self, two_robot_manager):
        assert len(two_robot_manager.robots) == 2

    def test_initial_positions(self, two_robot_manager):
        assert two_robot_manager.robots[0].current_pos == (0, 0)
        assert two_robot_manager.robots[1].current_pos == (4, 4)

    def test_priority_assignment(self, two_robot_manager):
        two_robot_manager.assign_priorities_by_distance()
        by_priority = two_robot_manager.robots_by_priority()
        assert by_priority[0].priority <= by_priority[1].priority

    def test_all_done_initially_false(self, two_robot_manager):
        assert not two_robot_manager.all_done()

    def test_mismatched_starts_goals_raises(self):
        with pytest.raises(ValueError):
            RobotManager([(0, 0), (1, 1)], [(4, 4)])


# ---------------------------------------------------------------------------
# Prioritised planner tests
# ---------------------------------------------------------------------------

class TestPrioritizedPlanner:
    def test_finds_paths_for_all_robots(self, small_grid_5):
        starts = [(0, 0), (0, 4)]
        goals  = [(4, 4), (4, 0)]
        rm = RobotManager(starts, goals)
        rm.assign_priorities_by_distance()

        pp = PrioritizedPlanner()
        result = pp.plan(rm.robots, small_grid_5)

        # Both robots should find paths on an empty 5×5 grid
        assert result.success_count == 2
        assert result.fail_count == 0

    def test_paths_are_valid(self, small_grid_5):
        starts = [(0, 0), (0, 4)]
        goals  = [(4, 4), (4, 0)]
        rm = RobotManager(starts, goals)
        rm.assign_priorities_by_distance()

        pp = PrioritizedPlanner()
        result = pp.plan(rm.robots, small_grid_5)

        for rid, path in result.paths.items():
            assert path is not None
            # Check start and end
            robot = rm.get(rid)
            assert path[0] == robot.start
            assert path[-1] == robot.goal

    def test_robot_doesnt_enter_obstacle(self):
        g = Grid(5, 5)
        g.set(2, 2, CellType.OBSTACLE)

        starts = [(0, 0), (0, 4)]
        goals  = [(4, 4), (4, 0)]
        rm = RobotManager(starts, goals)

        pp = PrioritizedPlanner()
        result = pp.plan(rm.robots, g)

        for path in result.paths.values():
            if path:
                for cell in path:
                    assert g.get(*cell) != CellType.OBSTACLE, \
                        f"Path entered obstacle at {cell}"

    def test_makespan_positive(self, small_grid_5):
        starts = [(0, 0), (0, 4)]
        goals  = [(4, 4), (4, 0)]
        rm = RobotManager(starts, goals)
        pp = PrioritizedPlanner()
        result = pp.plan(rm.robots, small_grid_5)
        assert result.makespan > 0

    def test_total_cost_positive(self, small_grid_5):
        starts = [(0, 0), (0, 4)]
        goals  = [(4, 4), (4, 0)]
        rm = RobotManager(starts, goals)
        pp = PrioritizedPlanner()
        result = pp.plan(rm.robots, small_grid_5)
        assert result.total_cost > 0

    def test_runtime_ms_positive(self, small_grid_5):
        starts = [(0, 0), (0, 4)]
        goals  = [(4, 4), (4, 0)]
        rm = RobotManager(starts, goals)
        pp = PrioritizedPlanner()
        result = pp.plan(rm.robots, small_grid_5)
        assert result.runtime_ms >= 0


# ---------------------------------------------------------------------------
# Environment generator tests
# ---------------------------------------------------------------------------

class TestEnvironmentGenerator:
    def test_reproducible_with_same_seed(self):
        gen1 = EnvironmentGenerator(20, 20, seed=42)
        gen2 = EnvironmentGenerator(20, 20, seed=42)
        grid1, starts1, goals1, _ = gen1.generate(num_robots=5)
        grid2, starts2, goals2, _ = gen2.generate(num_robots=5)
        import numpy as np
        assert np.array_equal(grid1.array, grid2.array)
        assert starts1 == starts2
        assert goals1 == goals2

    def test_different_seeds_differ(self):
        gen1 = EnvironmentGenerator(20, 20, seed=42)
        gen2 = EnvironmentGenerator(20, 20, seed=99)
        grid1, _, _, _ = gen1.generate(num_robots=5)
        grid2, _, _, _ = gen2.generate(num_robots=5)
        import numpy as np
        assert not np.array_equal(grid1.array, grid2.array)

    def test_starts_and_goals_are_free(self):
        gen = EnvironmentGenerator(20, 20, obstacle_density=0.20, seed=42)
        grid, starts, goals, _ = gen.generate(num_robots=5)
        for pos in starts + goals:
            assert grid.get(*pos) == CellType.FREE, \
                f"Start/goal {pos} is not FREE"

    def test_starts_do_not_overlap_goals(self):
        gen = EnvironmentGenerator(20, 20, seed=42)
        _, starts, goals, _ = gen.generate(num_robots=5)
        assert set(starts).isdisjoint(set(goals))

    def test_obstacle_count_approximate(self):
        import numpy as np
        density = 0.20
        gen = EnvironmentGenerator(20, 20, obstacle_density=density, seed=42)
        grid, _, _, _ = gen.generate(num_robots=3)
        n_obs = int((grid.array == CellType.OBSTACLE).sum())
        expected = int(20 * 20 * density)
        # Allow ±5 cells tolerance
        assert abs(n_obs - expected) <= 5
