"""Animation of one or more arms on the same scenario: map, rent meter, plain-English caption and cost counter.

One Scene draws every view: the interactive window (play/pause, restart, speed, scrub), the GIF and the HTML player."""
import math
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.doi.metrics import carried_kind, slots_full_at
from src.doi.narrate import ARM_NOTES
from src.doi.kinds import KINDS
from src.doi.story import caption_at, cell_name, detour_paid, kind_label, obstacle_prices, plain_events, rent_meters
from src.environment.grid import CellType

Pos = Tuple[int, int]
COLORS = {"free": (0.96, 0.96, 0.93), "wall": (0.18, 0.19, 0.22), "pushed": (0.30, 0.72, 0.42),
          "over": (0.82, 0.22, 0.20), "rent": (0.96, 0.68, 0.18), "rack": (0.52, 0.40, 0.28),
          "dump": (0.88, 0.83, 0.70)}
TRAIL = 8
CAPTION_WIDTH = 50


def cumulative_cost(result, cfg=None) -> np.ndarray:
    """Cost paid up to each tick (the per-tick costs recorded by the world; ends at result.J)."""
    return np.concatenate([[0.0], np.cumsum(result.tick_cost)])


def goal_timelines(result, scenario) -> Dict[int, List[Optional[Pos]]]:
    """The goal each robot is heading to at each tick, derived from where it actually stops."""
    out = {}
    for i, traj in result.trajectory.items():
        goals = scenario.tasks[i]
        k, line = 0, []
        for pos in traj:
            if k < len(goals) and pos == goals[k]:
                k += 1
            line.append(goals[k] if k < len(goals) else None)
        out[i] = line
    return out


def pushing_robots(result, t: int) -> set:
    """Robots in the middle of a push run at tick t."""
    return {p["robot"] for p in result.pushes if p["start_tick"] <= t - 1 <= p["end_tick"]}


def _background(scenario, result) -> np.ndarray:
    g = scenario.grid
    img = np.zeros((g.height, g.width, 3))
    for r in range(g.height):
        for c in range(g.width):
            img[r, c] = COLORS["wall"] if g.get(r, c) == CellType.OBSTACLE else COLORS["free"]
    for (r, c), slot in scenario.slots.items():
        img[r, c] = COLORS[slot]
    return img


def _obstacles(result, t: int) -> Dict[Pos, str]:
    trace = result.obstacle_trace
    return trace[min(t, len(trace) - 1)]


IDLE = {"never": "Never pushes anything: every blocked trip takes the long way round",
        "free": "Benchmark: every obstacle is gone from the start"}
HEADLINE = ("Rent or push: a robot whose route runs through an obstacle can go round (paying rent) or push it aside. "
            "When the rent the fleet has paid reaches the price of pushing, a robot pushes.")
def legend_for(scenario, results) -> str:
    """Legend listing only what is on this map."""
    present = {k for r in results.values() for trace in (r.obstacle_trace or [scenario.obstacles])
               for k in trace.values()}
    kinds = ", ".join(f"{KINDS[k].glyph} {kind_label(k)}" for k in KINDS if k in present)
    slots = "   brown = rack, sand = dump zone (a small square on it = a stored obstacle)" if scenario.slots else ""
    return (f"dark = permanent wall   coloured squares = removable obstacles ({kinds or 'none'})   "
            f"green bar = cleared{slots}\ncircle = robot (black ring = pushing)   hollow square = where it started   "
            f"star = the cell it is heading to (same colour)")


class Scene:
    """Figure plus draw(t) for a set of arms; frames are driven by the window, a GIF writer or the HTML writer."""

    def __init__(self, scenario, results: Dict[str, object], cfg, title: str = "", controls: bool = False,
                 dpi: int = 100):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.scenario, self.results, self.cfg = scenario, results, cfg
        self.names = list(results)
        self.g = scenario.grid
        self.horizon = max(max(len(t) for t in r.trajectory.values()) for r in results.values()) - 1
        self.costs = {n: cumulative_cost(r) for n, r in results.items()}
        self.goals = {n: goal_timelines(r, scenario) for n, r in results.items()}
        self.events = {n: plain_events(r, scenario) for n, r in results.items()}
        prices = obstacle_prices(results)
        self.meters = {n: rent_meters(r, prices) for n, r in results.items()}
        top = max((m.cumulative[-1] for ms in self.meters.values() for m in ms), default=0.0)
        top_price = max((m.price for ms in self.meters.values() for m in ms), default=1.0)
        self.meter_max = max(top, top_price * 1.3) * 1.08
        rows = max((len(ms) for ms in self.meters.values()), default=0)
        self.cmap = plt.get_cmap("tab10")
        self.backgrounds = {n: _background(scenario, r) for n, r in results.items()}

        panel_w = 5.0
        panel_h = panel_w * self.g.height / self.g.width
        meter_h = 0.5 + 0.28 * max(rows, 1)
        extra = 0.65 if controls else 0.0
        width = max(panel_w * len(self.names), 9.0)
        legend = "\n".join(textwrap.fill(line, int(width * 12)) for line in legend_for(scenario, results).split("\n"))
        legend_h = 0.17 * (legend.count("\n") + 1)
        self.fig = plt.figure(figsize=(width, panel_h + meter_h + 1.25 + legend_h + extra), dpi=dpi)
        gs = self.fig.add_gridspec(3, len(self.names), height_ratios=[panel_h, meter_h, 1.0], hspace=0.3,
                                   top=1 - (0.85 + extra) / self.fig.get_figheight(), bottom=(0.2 + legend_h) / self.fig.get_figheight())
        self.map_axes = [self.fig.add_subplot(gs[0, k]) for k in range(len(self.names))]
        self.meter_axes = [self.fig.add_subplot(gs[1, k]) for k in range(len(self.names))]
        self.text_axes = [self.fig.add_subplot(gs[2, k]) for k in range(len(self.names))]
        self.fig.suptitle(title or f"{scenario.name}: {len(scenario.starts)} robot(s), same tasks for every arm",
                          fontsize=12, weight="bold", y=0.985)
        self.fig.text(0.5, 1 - 0.55 / self.fig.get_figheight(), textwrap.fill(HEADLINE, 100), ha="center", va="center",
                      fontsize=8.5,
                      color="#444444")
        self.fig.text(0.5, 0.1 / self.fig.get_figheight(), legend, ha="center", va="bottom", fontsize=8,
                      linespacing=1.5)

    def draw(self, t: int) -> None:
        for k, name in enumerate(self.names):
            self._draw_map(k, name, t)
            self._draw_meters(k, name, t)
            self._draw_caption(k, name, t)

    def _draw_map(self, k: int, name: str, t: int) -> None:
        res, ax, g = self.results[name], self.map_axes[k], self.g
        ax.clear()
        ax.imshow(self.backgrounds[name], interpolation="nearest")
        for (r, c), kind in sorted(_obstacles(res, t).items()):
            spec = KINDS.get(kind)
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=spec.color if spec else (0.5, 0.5, 0.5),
                                   edgecolor="k", linewidth=0.8, zorder=3))
            ax.text(c, r, spec.glyph if spec else "?", ha="center", va="center", fontsize=8, color="white",
                    weight="bold", zorder=4)
        stored = {e["target"]: e["kind"] for e in res.carry_log if e["mode"] == "carry" and e["tick"] < t}
        for (r, c), kind in sorted(stored.items()):                    # an obstacle parked on a rack or dump cell
            ax.add_patch(Rectangle((c - 0.32, r - 0.32), 0.64, 0.64, facecolor=KINDS[kind].color, edgecolor="k",
                                   linewidth=0.8, zorder=3))
            ax.text(c, r, KINDS[kind].glyph, ha="center", va="center", fontsize=6, color="white", weight="bold",
                    zorder=4)
        pushing = pushing_robots(res, t)
        for i, traj in sorted(res.trajectory.items()):
            color = self.cmap(i % 10)
            line = self.goals[name][i]
            goal = line[min(t, len(line) - 1)] if t < len(traj) else None
            if goal is not None:
                ax.plot(goal[1], goal[0], marker="*", color=color, markersize=9, markeredgecolor="k",
                        markeredgewidth=0.4, zorder=5)
            ax.add_patch(Rectangle((traj[0][1] - 0.22, traj[0][0] - 0.22), 0.44, 0.44, fill=False, edgecolor=color,
                                   linewidth=1.6, zorder=5))
            if t >= len(traj):
                continue
            trail = traj[max(0, t - TRAIL):t + 1]
            if len(trail) > 1:
                ax.plot([p[1] for p in trail], [p[0] for p in trail], color=color, alpha=0.45, lw=2, zorder=5)
            r, c = traj[t]
            ax.add_patch(Circle((c, r), 0.38, facecolor=color, edgecolor="k" if i in pushing else "white",
                                linewidth=2.6 if i in pushing else 0.8, zorder=6))
            ax.text(c, r, str(i), ha="center", va="center", fontsize=7, color="white", zorder=7, weight="bold")
            load = carried_kind(res, i, t)
            if load is not None:
                ax.add_patch(Rectangle((c + 0.05, r - 0.5), 0.45, 0.45, facecolor=KINDS[load].color,
                                       edgecolor="white", linewidth=1.2, zorder=8))
        ax.set_xlim(-0.5, g.width - 0.5)
        ax.set_ylim(g.height - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        so_far = self.costs[name][min(t, len(self.costs[name]) - 1)]
        label = "FINAL cost" if t >= self.horizon else "cost so far"
        ax.set_title(f"{name}   |   tick {t}   |   {label} {so_far:.0f}", fontsize=10, weight="bold")
        ax.set_xlabel(ARM_NOTES.get(name, ""), fontsize=7.5, color="#555555")

    def _draw_meters(self, k: int, name: str, t: int) -> None:
        ax = self.meter_axes[k]
        ax.clear()
        meters = self.meters[name]
        ax.set_xlim(0, self.meter_max)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if not meters:
            ax.set_xticks([])
            ax.set_yticks([])
            why = ("no robot's route runs through an obstacle: no detour, nothing to push"
                   if detour_paid(self.results[name]) == 0 else
                   "detours were paid, but no push was cheaper than going round")
            ax.text(0.5, 0.5, why, ha="center", va="center", transform=ax.transAxes, fontsize=8.5, color="#555555")
            return
        ax.set_ylim(len(meters) - 0.4, -0.9)
        ax.set_yticks(range(len(meters)))
        ax.set_yticklabels([f"{kind_label(m.kind)} {cell_name(m.cell)}" for m in meters], fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("detour cost paid so far (rent)  vs  price of pushing it (black line)", fontsize=7.5)
        for row, m in enumerate(meters):
            paid, pushed, status = m.at(t), m.pushed_by(t), m.status(t)
            ax.barh(row, paid, height=0.4, color=COLORS["pushed"] if pushed else
                    (COLORS["over"] if paid >= m.price else COLORS["rent"]))
            ax.plot([m.price, m.price], [row - 0.3, row + 0.3], color="k", lw=2)
            ax.text(m.price, row - 0.32, f"price {m.price:.0f}", ha="center", va="bottom", fontsize=6.5)
            ax.text(min(paid, self.meter_max) + self.meter_max * 0.01, row,
                    f"{paid:.0f}" + (f"  {status}" if status else ""),
                    va="center", ha="left", fontsize=7)

    def _draw_caption(self, k: int, name: str, t: int) -> None:
        tx = self.text_axes[k]
        tx.clear()
        tx.axis("off")
        now, before = caption_at(self.events[name], t, IDLE.get(name, "Robots walk their tasks; detour rent is being counted"))
        now_lines = textwrap.wrap(now, CAPTION_WIDTH)
        tx.text(0.0, 1.0, "\n".join(now_lines), va="top", ha="left", fontsize=9.5, weight="bold",
                transform=tx.transAxes)
        if before:
            tx.text(0.0, 1.0 - 0.17 * (len(now_lines) + 0.4), "\n".join(textwrap.wrap("before: " + before,
                    CAPTION_WIDTH + 8)), va="top", ha="left", fontsize=8, color="#666666", transform=tx.transAxes)

    def show(self, fps: int = 6) -> None:
        """Interactive window: play/pause (space), restart, speed and a time scrubber."""
        from matplotlib.widgets import Button, Slider
        fig = self.fig
        n = self.horizon + 1
        state = {"t": 0, "playing": True}
        h = fig.get_figheight()
        y = 1 - 1.15 / h                       # a strip under the headline, so no window size can cut it off
        play = Button(fig.add_axes([0.03, y, 0.08, 0.34 / h]), "Pause")
        restart = Button(fig.add_axes([0.12, y, 0.08, 0.34 / h]), "Restart")
        speed = Slider(fig.add_axes([0.30, y, 0.15, 0.34 / h]), "speed ", 1, 8, valinit=1, valstep=1)
        scrub = Slider(fig.add_axes([0.58, y, 0.34, 0.34 / h]), "tick ", 0, self.horizon, valinit=0, valstep=1)

        def render(t: int) -> None:
            state["t"] = t
            self.draw(t)
            scrub.eventson = False
            scrub.set_val(t)
            scrub.eventson = True
            fig.canvas.draw_idle()

        def set_playing(on: bool) -> None:
            state["playing"] = on
            play.label.set_text("Pause" if on else ("Replay" if state["t"] >= n - 1 else "Play"))
            fig.canvas.draw_idle()

        def step() -> None:
            if not state["playing"]:
                return
            render(min(n - 1, state["t"] + int(speed.val)))
            if state["t"] >= n - 1:
                set_playing(False)

        def on_play(_event=None) -> None:
            if not state["playing"] and state["t"] >= n - 1:
                render(0)
            set_playing(not state["playing"])

        def on_restart(_event=None) -> None:
            render(0)
            set_playing(True)

        play.on_clicked(on_play)
        restart.on_clicked(on_restart)
        scrub.on_changed(lambda v: (set_playing(False), render(int(v))))
        fig.canvas.mpl_connect("key_press_event", lambda e: on_play() if e.key == " " else None)
        self._widgets = (play, restart, speed, scrub)   # keep references alive
        timer = fig.canvas.new_timer(interval=int(1000 / fps))
        timer.add_callback(step)
        timer.start()
        render(0)
        self.plt.show()


def _frame_ticks(horizon: int, max_frames: int) -> List[int]:
    step = max(1, math.ceil(horizon / max_frames))
    return list(range(0, horizon + 1, step)) + [horizon] * 8


def write_html(scene: Scene, path: str, fps: int, max_frames: int, title: str) -> str:
    """A self-contained page with a play/pause/scrub player; works in any browser, unlike a GIF in Preview."""
    import matplotlib
    from matplotlib.animation import FuncAnimation
    matplotlib.rcParams["animation.embed_limit"] = 128
    ticks = _frame_ticks(scene.horizon, max_frames)
    anim = FuncAnimation(scene.fig, lambda i: scene.draw(ticks[i]), frames=len(ticks), interval=1000 / fps)
    player = anim.to_jshtml(fps=fps, default_mode="loop")
    page = (f"<!doctype html><meta charset='utf-8'><title>{title}</title>"
            "<body style='font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px'>"
            f"<h2>{title}</h2><p>{HEADLINE}</p>{player}</body>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(page)
    return path


def animate_runs(scenario, results: Dict[str, object], cfg, path: Optional[str] = None, show: bool = False,
                 fps: int = 6, max_frames: int = 240, title: str = "", html: Optional[str] = None):
    """Show the window (show=True) and/or write a GIF (path) and/or an HTML player (html)."""
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if show and sys.platform == "darwin":
        try:
            plt.switch_backend("MacOSX")
        except Exception:
            pass
    elif not show:
        plt.switch_backend("Agg")
    saved = None
    if path or html:
        scene = Scene(scenario, results, cfg, title, dpi=80)
        if html:
            saved = write_html(scene, html, fps, min(max_frames, 150), title or scenario.name)
        if path:
            ticks = _frame_ticks(scene.horizon, max_frames)
            anim = FuncAnimation(scene.fig, lambda i: scene.draw(ticks[i]), frames=len(ticks), interval=1000 / fps,
                                 repeat=False)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            anim.save(path, writer=PillowWriter(fps=fps), dpi=72)
            saved = path
        plt.close(scene.fig)
    if show:
        Scene(scenario, results, cfg, title, controls=True).show(fps)
    return saved
