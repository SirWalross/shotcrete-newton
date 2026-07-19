# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Throughput ablations of the voxel solver's optimizations.

Measures solver steps/s on a production-sized batch with each optimization toggled:

* **bounding boxes** -- via the solver's ``use_bounding_boxes`` flag; when disabled,
  solidify/adhesion/drop-down scan the full grid every step instead of the spray
  neighbourhood.
* **CUDA graphs** -- the example captures a pair of ping-pong solver steps into a
  CUDA graph and replays it, eliminating the per-kernel launch overhead.

All four combinations are measured; results are printed and plotted as a bar chart to
``voxel_ablations.pdf``.

Run with::

    uv run -m newton.examples voxel_ablations --viewer null
"""

import gc
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.voxel import _plot_style

# production grid: padded voxel dimensions per environment (x lateral, y depth, z up)
GRID_X = 256
GRID_Y = 96
GRID_Z = 256
VOXEL_SIZE = 0.005  # m
NOZZLE_DISTANCE = 0.35  # m
# two batch sizes: small batches are launch-overhead bound (where CUDA graphs help),
# large batches are GPU bound (where the bounding boxes dominate)
ENV_COUNTS = [4, 64]

WARMUP_STEPS = 4
TIMED_STEPS = 200

CONFIGS = [
    (False, False, "baseline\n(no bboxes,\n no graph)"),
    (True, False, "only bounding\nboxes"),
    (False, True, "only CUDA graph"),
    (True, True, "both bboxes and\nCUDA graph"),
]


class Example:
    def __init__(self, viewer, num_frames=None):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_step = 0
        self.total_steps = len(ENV_COUNTS) * len(CONFIGS)
        self.viewer = viewer
        self.device = wp.get_device()
        self.reported = False
        self.metrics = None

        np.random.seed(13)  # noqa: NPY002 -- the solver draws its speed distributions from legacy np.random

        # rows of (label, steps_per_s or None, note)
        self.results = []

        self.model = self._build_model(ENV_COUNTS[0])
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)

    def _build_model(self, num_envs: int):
        nozzle = newton.ModelBuilder()
        # the solver looks up the TCP body via the `/World/envs/env_*/<name>` USD-style key
        body = nozzle.add_body(xform=wp.transform_identity(), key="/World/envs/env_0/nozzle", mass=1.0)
        nozzle.add_shape_sphere(body, radius=0.02)

        wall_j = GRID_Y - 2
        gy = wall_j - round(NOZZLE_DISTANCE / VOXEL_SIZE)
        translation = wp.vec3(0.25 * VOXEL_SIZE, (gy + 0.25) * VOXEL_SIZE, (GRID_Z // 2 + 0.25) * VOXEL_SIZE)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)  # spray along +y

        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_world(nozzle, xform=wp.transform(translation, rotation))
        model = builder.finalize()

        shape = (num_envs, GRID_X, GRID_Y, GRID_Z)
        model.voxel_wet = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        model.voxel_dry = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        model.voxel_distance = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        model.voxel_load = wp.zeros(shape, dtype=wp.int16, device=self.device)
        model.voxel_pos = wp.zeros((num_envs,), dtype=wp.vec3f, device=self.device)
        return model

    def run_config(self, num_envs: int, use_bboxes: bool, use_graph: bool) -> tuple[float | None, str]:
        model = self._build_model(num_envs)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        rewards = newton.VoxelRewards((num_envs, GRID_X - 2, GRID_Y - 2, GRID_Z - 2), 16, self.device)
        solver = newton.solvers.SolverVoxel(
            model,
            tcp_body_name="nozzle",
            h=VOXEL_SIZE,
            use_bounding_boxes=use_bboxes,
        )
        solver.reset(state_0, wp.array(np.arange(num_envs), dtype=wp.int32, device=self.device))

        for _ in range(WARMUP_STEPS):
            solver.step(state_0, state_1, control, None, rewards, self.frame_dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()

        if use_graph:
            try:
                with wp.ScopedCapture() as capture:
                    # capture a ping-pong pair so the graph is state-stationary
                    solver.step(state_0, state_1, control, None, rewards, self.frame_dt)
                    solver.step(state_1, state_0, control, None, rewards, self.frame_dt)
                graph = capture.graph
            except Exception as exc:
                return None, f"graph capture failed: {exc}"
            wp.capture_launch(graph)  # graph warm-up
            wp.synchronize()
            t0 = time.perf_counter()
            for _ in range(TIMED_STEPS // 2):
                wp.capture_launch(graph)
            wp.synchronize()
            elapsed = time.perf_counter() - t0
            return 2 * (TIMED_STEPS // 2) / elapsed, ""

        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            solver.step(state_0, state_1, control, None, rewards, self.frame_dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        return TIMED_STEPS / elapsed, ""

    def step(self):
        if self.sim_step >= self.total_steps:
            if not self.reported:
                self.report()
            return

        num_envs = ENV_COUNTS[self.sim_step // len(CONFIGS)]
        use_bboxes, use_graph, label = CONFIGS[self.sim_step % len(CONFIGS)]
        steps_per_s, note = self.run_config(num_envs, use_bboxes, use_graph)
        self.results.append((num_envs, label, steps_per_s, note))
        flat = label.replace("\n", " ")
        if steps_per_s is None:
            print(f"{num_envs} envs, {flat}: {note}")
        else:
            print(f"{num_envs} envs, {flat}: {steps_per_s:.2f} steps/s")
        gc.collect()

        self.sim_time += self.frame_dt
        self.sim_step += 1
        if self.sim_step == self.total_steps:
            self.report()

    def report(self):
        self.reported = True
        baselines = {e: s for e, label, s, _ in self.results if label.startswith("baseline")}

        print("\n=== Optimization ablations ===")
        print(
            f"grid {GRID_X}x{GRID_Y}x{GRID_Z} voxels, batches of {ENV_COUNTS} environments, "
            f"{TIMED_STEPS} timed steps after {WARMUP_STEPS} warm-up steps, GPU: {self.device.name}"
        )
        print(f"\n{'envs':>5} {'configuration':>28} {'steps/s':>10} {'vs baseline':>12}")
        for num_envs, label, steps_per_s, note in self.results:
            flat = label.replace("\n", " ")
            if steps_per_s is None:
                print(f"{num_envs:5d} {flat:>28} {'-':>10}  {note}")
            else:
                print(f"{num_envs:5d} {flat:>28} {steps_per_s:10.2f} {steps_per_s / baselines[num_envs]:11.2f}x")

        achieved = [s for _, _, s, _ in self.results if s is not None]
        self.metrics = {"achieved_configs": len(achieved), "baseline_steps_per_s": baselines[max(ENV_COUNTS)]}
        self.save_plot(baselines)

    def save_plot(self, baselines: dict):
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("matplotlib not available, skipping the figure")
            return

        _plot_style.setup(plt)
        fig, axes = plt.subplots(1, len(ENV_COUNTS), figsize=(4.0 * len(ENV_COUNTS), 3.0))
        for ax, num_envs in zip(np.atleast_1d(axes), ENV_COUNTS, strict=True):
            rows = [r for r in self.results if r[0] == num_envs]
            labels = [label for _, label, _, _ in rows]
            values = [s if s is not None else 0.0 for _, _, s, _ in rows]
            colors = [_plot_style.SERIES[0] if i == 0 else "#86b6ef" for i in range(len(values))]
            bars = ax.bar(labels, values, color=colors, width=0.62)
            for bar, (_, _, steps_per_s, _) in zip(bars, rows, strict=True):
                if steps_per_s is None:
                    ax.annotate(
                        "n/a",
                        xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        color=_plot_style.TEXT_SECONDARY,
                    )
                else:
                    ax.annotate(
                        f"{steps_per_s:.1f}\n({steps_per_s / baselines[num_envs]:.2f}$\\times$)",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7.0,
                        color=_plot_style.TEXT,
                    )
            ax.set_ylabel("solver steps/s")
            ax.set_ylim(0, 1.3 * max(values))
            ax.set_xlabel(f"{num_envs} environments")
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="x", labelsize=7.0)
        fig.tight_layout()
        fig.savefig("voxel_ablations.pdf")
        plt.close(fig)
        print("saved figure: voxel_ablations.pdf")

    def test_final(self):
        if not self.reported:
            self.report()
        assert self.metrics["achieved_configs"] >= 2, "too few configurations ran successfully"
        assert self.metrics["baseline_steps_per_s"] > 1.0, "baseline throughput implausibly low"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=len(ENV_COUNTS) * len(CONFIGS))

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_frames=args.num_frames)

    newton.examples.run(example, args)
