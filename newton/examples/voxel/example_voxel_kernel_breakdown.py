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

"""Per-kernel time breakdown of a voxel-solver step.

The solver's ``collect_timings`` flag records every ScopedTimer section (with GPU
synchronization) into a dictionary. This example steps a production-sized batch and
attributes the section timings of every step to the pipeline stages

    deposition / redistribution / dripping / solidify / adhesion / rewards,

printed as a per-step table and plotted as one stacked bar per step to
``voxel_kernel_breakdown.pdf``. The adhesion check runs every 10th step and shows up
as periodic spikes. Note that the synchronization needed for per-section timing
serializes the pipeline, so the stacked totals are slightly pessimistic compared to
free-running throughput (see the throughput example).

Run with::

    uv run -m newton.examples voxel_kernel_breakdown --viewer null --num-frames 50
"""

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
NUM_WORLDS = 64

# ScopedTimer section -> pipeline stage. Only mutually exclusive sections are listed
# (nested sub-timers of these are ignored to avoid double counting).
STAGE_OF_SECTION = {
    "alloca": "trajectory",
    "spray trajectory": "trajectory",
    "spray rebound": "trajectory",
    "spray deposit": "deposition",
    "spray backtrack deposit": "deposition",
    "update bbox": "deposition",
    "update global bbox": "deposition",
    "spray redistribution": "redistribution",
    "drip": "dripping",
    "solidify": "solidify",
    "adhesion check": "adhesion",
    "rewards": "rewards",
}
STAGES = ["trajectory", "deposition", "redistribution", "dripping", "solidify", "adhesion", "rewards"]


class Example:
    def __init__(self, viewer, num_frames=50):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_step = 0
        self.total_steps = num_frames
        self.viewer = viewer
        self.device = wp.get_device()
        self.reported = False
        self.metrics = None

        np.random.seed(12)  # noqa: NPY002 -- the solver draws its speed distributions from legacy np.random

        nozzle = newton.ModelBuilder()
        # the solver looks up the TCP body via the `/World/envs/env_*/<name>` USD-style key
        body = nozzle.add_body(xform=wp.transform_identity(), key="/World/envs/env_0/nozzle", mass=1.0)
        nozzle.add_shape_sphere(body, radius=0.02)

        wall_j = GRID_Y - 2
        gy = wall_j - round(NOZZLE_DISTANCE / VOXEL_SIZE)
        translation = wp.vec3(0.25 * VOXEL_SIZE, (gy + 0.25) * VOXEL_SIZE, (GRID_Z // 2 + 0.25) * VOXEL_SIZE)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)  # spray along +y

        builder = newton.ModelBuilder()
        for _ in range(NUM_WORLDS):
            builder.add_world(nozzle, xform=wp.transform(translation, rotation))
        self.model = builder.finalize()

        shape = (NUM_WORLDS, GRID_X, GRID_Y, GRID_Z)
        self.model.voxel_wet = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_dry = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_distance = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_load = wp.zeros(shape, dtype=wp.int16, device=self.device)
        self.model.voxel_pos = wp.zeros((NUM_WORLDS,), dtype=wp.vec3f, device=self.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.rewards = newton.VoxelRewards((NUM_WORLDS, GRID_X - 2, GRID_Y - 2, GRID_Z - 2), 16, self.device)

        self.solver = newton.solvers.SolverVoxel(
            self.model,
            tcp_body_name="nozzle",
            h=VOXEL_SIZE,
            collect_timings=True,
        )
        self.solver.reset(self.state_0, wp.array(np.arange(NUM_WORLDS), dtype=wp.int32, device=self.device))

        # per-step stage times in ms; timer lists grow per step, so track the consumed length
        self.stage_times = np.zeros((num_frames, len(STAGES)))
        self.consumed = dict.fromkeys(STAGE_OF_SECTION, 0)

        self.viewer.set_model(self.model)

    def step(self):
        if self.sim_step >= self.total_steps:
            if not self.reported:
                self.report()
            return

        self.rewards.step()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.rewards, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        # attribute this step's new timer entries to their stages
        timings = self.solver.timing_dict
        for section, stage in STAGE_OF_SECTION.items():
            values = timings.get(section, [])
            new = values[self.consumed[section] :]
            self.consumed[section] = len(values)
            self.stage_times[self.sim_step, STAGES.index(stage)] += sum(new)

        self.sim_time += self.frame_dt
        self.sim_step += 1
        if self.sim_step == self.total_steps:
            self.report()

    def report(self):
        self.reported = True
        times = self.stage_times[10: self.sim_step]
        mean = times.mean(axis=0)
        total = mean.sum()

        print("\n=== Per-kernel breakdown ===")
        print(
            f"grid {GRID_X}x{GRID_Y}x{GRID_Z} voxels, {NUM_WORLDS} environments, solver defaults, "
            f"{self.sim_step} steps, per-section GPU synchronization"
        )
        print(f"\n{'step':>4} " + " ".join(f"{s:>14}" for s in STAGES) + f" {'total':>10}")
        for i, row in enumerate(times):
            print(f"{i:4d} " + " ".join(f"{v:14.2f}" for v in row) + f" {row.sum():10.2f}")
        print("\nmean per step (ms):")
        for stage, value in zip(STAGES, mean, strict=True):
            print(f"  {stage:>14}: {value:8.2f}  ({100 * value / total:5.1f} %)")
        print(f"  {'total':>14}: {total:8.2f}")

        self.metrics = {"mean_total_ms": float(total), "dominant": STAGES[int(np.argmax(mean))]}
        self.save_plot(times)
        print(f"\ndominant stage: {self.metrics['dominant']}")

    def save_plot(self, times: np.ndarray):
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("matplotlib not available, skipping the figure")
            return

        _plot_style.setup(plt)
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        steps = np.arange(times.shape[0])
        bottom = np.zeros(times.shape[0])
        for s, stage in enumerate(STAGES):
            ax.bar(
                steps,
                times[:, s],
                bottom=bottom,
                width=1.0,
                color=_plot_style.SERIES[s],
                linewidth=0.15,
                edgecolor="white",
                label=stage,
            )
            bottom += times[:, s]
        ax.set_xlim(-0.5, times.shape[0] - 0.5)
        ax.set_xlabel("simulation step")
        ax.set_ylabel("time per step [ms]")
        ax.set_title(
            "Step time by pipeline stage"
        )
        ax.grid(axis="x", visible=False)
        ax.legend(ncols=3, loc="upper left", columnspacing=1.2)
        ax.set_ylim(0, 1.35 * bottom.max())
        fig.savefig("voxel_kernel_breakdown.pdf")
        plt.close(fig)
        print("saved figure: voxel_kernel_breakdown.pdf")

    def test_final(self):
        if not self.reported:
            self.report()
        assert self.metrics["mean_total_ms"] > 0.0, "no timings were collected"
        # the adhesion check must appear in its periodic steps
        adhesion = self.stage_times[: self.sim_step, STAGES.index("adhesion")]
        assert adhesion.max() > 0.0, "adhesion check never ran or was not timed"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=50)

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_frames=args.num_frames)

    newton.examples.run(example, args)
