"""
tests/test_conflicts.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for conflict detection.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.mapf.conflict import ConflictDetector, ConflictType


class TestConflictDetector:

    def test_no_conflicts_different_paths(self):
        paths = {
            0: [(0, 0), (0, 1), (0, 2)],
            1: [(2, 0), (2, 1), (2, 2)],
        }
        conflicts = ConflictDetector.detect_all(paths)
        vertex = [c for c in conflicts if c.conflict_type == ConflictType.VERTEX]
        edge   = [c for c in conflicts if c.conflict_type == ConflictType.EDGE]
        assert len(vertex) == 0
        assert len(edge) == 0

    def test_vertex_conflict_detected(self):
        """Both robots at (1,1) at t=1."""
        paths = {
            0: [(0, 0), (1, 1), (2, 2)],
            1: [(2, 0), (1, 1), (0, 2)],
        }
        conflicts = ConflictDetector.detect_all(paths)
        vertex = [c for c in conflicts if c.conflict_type == ConflictType.VERTEX]
        assert len(vertex) >= 1
        assert vertex[0].timestep == 1
        assert vertex[0].location == (1, 1)

    def test_edge_conflict_detected(self):
        """Robot 0 goes A→B and Robot 1 goes B→A between t=0 and t=1."""
        paths = {
            0: [(0, 0), (0, 1)],
            1: [(0, 1), (0, 0)],
        }
        conflicts = ConflictDetector.detect_all(paths)
        edge = [c for c in conflicts if c.conflict_type == ConflictType.EDGE]
        assert len(edge) >= 1
        c = edge[0]
        assert c.timestep == 0
        assert (c.location, c.location_b) in [((0, 0), (0, 1)), ((0, 1), (0, 0))]

    def test_goal_conflict_same_goal(self):
        """Two robots have the same goal."""
        paths = {
            0: [(0, 0), (0, 1), (0, 2)],
            1: [(2, 0), (1, 0), (0, 2)],
        }
        conflicts = ConflictDetector.detect_all(paths)
        goal = [c for c in conflicts if c.conflict_type == ConflictType.GOAL]
        assert len(goal) >= 1

    def test_no_false_positive_different_times(self):
        """Robots cross the same cell at different timesteps — no conflict."""
        paths = {
            0: [(0, 0), (1, 0), (2, 0)],
            1: [(2, 0), (2, 1), (2, 2)],
        }
        # Robot 0 is at (2,0) at t=2; Robot 1 is at (2,0) at t=0 → different times
        conflicts = ConflictDetector.detect_all(paths)
        vertex = [c for c in conflicts if c.conflict_type == ConflictType.VERTEX]
        # They're both at (2,0) but at different timesteps
        same_time_vertex = [c for c in vertex if c.timestep in (0, 1)]
        # At t=0: R0 at (0,0), R1 at (2,0) — no conflict
        # At t=1: R0 at (1,0), R1 at (2,1) — no conflict
        # At t=2: R0 at (2,0), R1 at (2,2) — no conflict
        assert len(same_time_vertex) == 0

    def test_count_by_type(self):
        paths = {
            0: [(0, 0), (1, 1)],
            1: [(2, 0), (1, 1)],
        }
        conflicts = ConflictDetector.detect_all(paths)
        counts = ConflictDetector.count_by_type(conflicts)
        assert counts["VERTEX"] >= 1

    def test_severity_score_ordering(self):
        from src.mapf.conflict import Conflict, ConflictType
        goal_c = Conflict(ConflictType.GOAL, (0, 1), -1, (0, 0))
        vertex_c = Conflict(ConflictType.VERTEX, (0, 1), 1, (1, 1))
        edge_c = Conflict(ConflictType.EDGE, (0, 1), 0, (0, 0), (0, 1))
        assert (ConflictDetector.severity_score(goal_c) >
                ConflictDetector.severity_score(vertex_c) >
                ConflictDetector.severity_score(edge_c))

    def test_single_robot_no_conflicts(self):
        paths = {0: [(0, 0), (1, 0), (2, 0), (3, 0)]}
        conflicts = ConflictDetector.detect_all(paths)
        assert len(conflicts) == 0

    def test_empty_paths_no_conflicts(self):
        conflicts = ConflictDetector.detect_all({})
        assert len(conflicts) == 0
