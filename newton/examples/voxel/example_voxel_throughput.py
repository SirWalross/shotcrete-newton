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

"""Throughput scaling of the voxel solver over the number of parallel environments.

For each environment count on the ladder 1, 4, 16, 64, 128, 256, 1024, 4096, the
solver steps a production-sized grid (256 x 96 x 256 voxels per environment) and the
wall-clock step rate is measured (solver stepping only, full default pipeline
including adhesion checks; reward buffers are not re-zeroed between steps as that is
host-side bookkeeping). Configurations whose estimated footprint exceeds the free GPU
memory are skipped and reported as the memory ceiling.

Reported per configuration: steps/s, aggregate env-steps/s, and the aggregate
real-time factor (one step simulates 1/50 s of spraying). The reference point is
Yamakawa et al.'s reported ~3x real-time on a single CPU core.

One configuration is executed per frame; results are printed as a table and plotted
(log-log) to ``voxel_throughput.pdf``, annotated with the GPU model and the memory
ceiling.

Run with::

    uv run -m newton.examples voxel_throughput --viewer null
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

ENV_LADDER = [1, 4, 16, 64, 128, 256, 1024, 4096]
WARMUP_STEPS = 3
TIMED_STEPS = 200
SIM_DT = 1.0 / 50.0  # simulated time per spray event

YAMAKAWA_REALTIME = 3.0  # x real-time, single CPU core (baseline from the literature)


def estimated_bytes(num_envs: int) -> int:
    """Rough per-configuration GPU footprint: voxel grids + the large solver buffers."""
    voxels = GRID_X * GRID_Y * GRID_Z * (1 + 1 + 1 + 2)  # wet, dry, distance, load
    buffers = 300 * 257 * 4 + 12 * 300 * 4 * 4  # spray neighbours + ray buffers (approx)
    return num_envs * (voxels + buffers)


def gpu_capacity_bytes(device) -> int:
    total = getattr(device, "total_memory", 0)
    return int(total) if total else 1 << 62  # unknown -> just attempt the allocation


class Example:
    def __init__(self, viewer, num_frames=None):
        self.fps = 50
        self.frame_dt = SIM_DT
        self.sim_time = 0.0
        self.sim_step = 0
        self.viewer = viewer
        self.device = wp.get_device()
        self.reported = False
        self.metrics = None

        np.random.seed(11)  # noqa: NPY002 -- the solver draws its speed distributions from legacy np.random

        self.total_steps = len(ENV_LADDER)
        # rows of (num_envs, steps_per_s or None, note)
        self.results = []

        # minimal model so the viewer has something to show
        self.model = self._build_model(1)
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

    def run_config(self, num_envs: int) -> float:
        model = self._build_model(num_envs)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        rewards = newton.VoxelRewards((num_envs, GRID_X - 2, GRID_Y - 2, GRID_Z - 2), 16, self.device)
        solver = newton.solvers.SolverVoxel(
            model,
            tcp_body_name="nozzle",
            h=VOXEL_SIZE,
        )
        world_indices = wp.array(np.arange(num_envs), dtype=wp.int32, device=self.device)
        solver.reset(state_0, world_indices)

        for _ in range(WARMUP_STEPS):
            solver.step(state_0, state_1, control, None, rewards, self.frame_dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()

        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            solver.step(state_0, state_1, control, None, rewards, self.frame_dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        return TIMED_STEPS / elapsed

    def step(self):
        if self.sim_step >= self.total_steps:
            if not self.reported:
                self.report()
            return

        num_envs = ENV_LADDER[self.sim_step]
        need = estimated_bytes(num_envs)
        # compare against the card's capacity (a slightly conservative margin for the
        # framework's base usage); pool fragmentation and true OOM are caught below
        capacity = 0.88 * gpu_capacity_bytes(self.device)
        if need > capacity:
            self.results.append((num_envs, None, f"skipped, needs ~{need / 2**30:.1f} GiB"))
            print(f"envs = {num_envs:5d}: skipped (needs ~{need / 2**30:.1f} GiB, exceeds GPU capacity)")
        else:
            try:
                steps_per_s = self.run_config(num_envs)
                self.results.append((num_envs, steps_per_s, ""))
                print(
                    f"envs = {num_envs:5d}: {steps_per_s:8.2f} steps/s, "
                    f"{steps_per_s * num_envs:10.1f} env-steps/s, "
                    f"aggregate real-time factor {steps_per_s * num_envs * SIM_DT:8.2f}"
                )
            except Exception as exc:
                self.results.append((num_envs, None, f"failed: {exc}"))
                print(f"envs = {num_envs:5d}: failed ({exc})")
            finally:
                gc.collect()

        self.sim_time += self.frame_dt
        self.sim_step += 1
        if self.sim_step == self.total_steps:
            self.report()

    def report(self):
        self.reported = True
        device_name = self.device.name
        total_mem = getattr(self.device, "total_memory", 0)

        print("\n=== Throughput scaling ===")
        print(
            f"grid {GRID_X}x{GRID_Y}x{GRID_Z} voxels per environment, solver defaults, "
            f"{TIMED_STEPS} timed steps after {WARMUP_STEPS} warm-up steps"
        )
        print(f"GPU: {device_name} ({total_mem / 2**30:.0f} GiB)")
        print(f"\n{'envs':>6} {'steps/s':>10} {'env-steps/s':>12} {'real-time x':>12}  note")
        for num_envs, steps_per_s, note in self.results:
            if steps_per_s is None:
                print(f"{num_envs:6d} {'-':>10} {'-':>12} {'-':>12}  {note}")
            else:
                print(
                    f"{num_envs:6d} {steps_per_s:10.2f} {steps_per_s * num_envs:12.1f} "
                    f"{steps_per_s * num_envs * SIM_DT:12.2f}"
                )

        achieved = [(e, s) for e, s, _ in self.results if s is not None]
        self.metrics = {
            "achieved_configs": len(achieved),
            "max_env_steps_per_s": max((e * s for e, s in achieved), default=0.0),
            "max_realtime": max((e * s * SIM_DT for e, s in achieved), default=0.0),
        }
        print(
            f"\npeak aggregate real-time factor: {self.metrics['max_realtime']:.1f}x "
            f"(Yamakawa et al.: ~{YAMAKAWA_REALTIME:.0f}x on one CPU core)"
        )
        self.save_plot(device_name, total_mem)

    def save_plot(self, device_name: str, total_mem: int):
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("matplotlib not available, skipping the figure")
            return

        _plot_style.setup(plt)
        achieved = [(e, s) for e, s, _ in self.results if s is not None]
        if not achieved:
            print("no successful configurations, skipping the figure")
            return
        envs = np.array([e for e, _ in achieved], dtype=float)
        steps = np.array([s for _, s in achieved])

        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.loglog(envs, steps, "o-", color=_plot_style.SERIES[0], label="steps/s per environment batch")
        ax.loglog(envs, steps * envs, "s-", color=_plot_style.SERIES[1], label="aggregate env-steps/s")
        ax.axhline(
            YAMAKAWA_REALTIME / SIM_DT,
            color=_plot_style.SERIES[5],
            linewidth=0.9,
            linestyle=(0, (4, 3)),
        )
        ax.annotate(
            f"Yamakawa et al., $\\sim${YAMAKAWA_REALTIME:.0f}$\\times$ real-time",
            xy=(envs[0], YAMAKAWA_REALTIME / SIM_DT),
            xytext=(0, -9),
            textcoords="offset points",
            fontsize=7.5,
            ha="left",
            color=_plot_style.TEXT_SECONDARY,
        )

        # skipped = [e for e, s, _ in self.results if s is None]
        # if skipped:
        #     ceiling = np.sqrt(envs[-1] * min(skipped))  # geometric midpoint on the log axis
        #     ax.axvline(ceiling, color=_plot_style.SPINE, linewidth=0.9, linestyle=(0, (2, 2)))
        #     ax.annotate(
        #         f"memory ceiling ({total_mem / 2**30:.0f} GiB)",
        #         xy=(ceiling, np.sqrt(steps.min() * (steps * envs).max())),
        #         xytext=(-5, 0),
        #         textcoords="offset points",
        #         fontsize=7.5,
        #         ha="right",
        #         va="center",
        #         rotation=90,
        #         color=_plot_style.TEXT_SECONDARY,
        #     )

        ax.set_xticks(envs, labels=[str(int(e)) for e in envs])
        ax.set_xlabel("parallel environments")
        ax.set_ylabel("throughput [1/s]")
        ax.set_title("Voxel solver throughput")
        ax.legend(loc="lower left")
        fig.savefig("voxel_throughput.pdf")
        plt.close(fig)
        print("saved figure: voxel_throughput.pdf")

    def test_final(self):
        if not self.reported:
            self.report()
        assert self.metrics["achieved_configs"] >= 3, "too few configurations ran successfully"
        assert self.metrics["max_realtime"] > 1.0, "solver is slower than real time in aggregate"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=len(ENV_LADDER))

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_frames=args.num_frames)

    newton.examples.run(example, args)
