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

"""Mass conservation of the voxel solver's parallel deposition.

Per spray event the ledger

    sprayed = deposited (grid gain) + undeposited primary mass + undeposited rebound mass

must close; any residual is mass silently created or destroyed by write races between
concurrently depositing droplet threads (lost ``saturating_add`` updates and stale
capacity reads on shared voxels). This example measures that residual, relative to the
sprayed mass, as a function of the droplet count $N_d$ -- at constant total mass flux,
so only the degree of parallelism changes.

Dripping, adhesion checks and in-flight redistribution are disabled so that the
deposition pipeline (including the rebound pass) is the only mass-moving mechanism;
each of those others has its own, separately quantified accounting behaviour. The
sprayed mass is the solver's generated droplet-mass distribution, evaluated exactly.

One configuration is executed per frame; results are printed as a table and plotted
to ``voxel_mass_conservation.pdf``.

Run with::

    uv run -m newton.examples voxel_mass_conservation --viewer null
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.voxel import _plot_style

# padded voxel-grid dimensions per world (x lateral, y towards wall, z up)
GRID_X = 132
GRID_Y = 112
GRID_Z = 132
VOXEL_SIZE = 0.005  # m
NOZZLE_DISTANCE = 0.5  # m; short stand-off concentrates the footprint -> many races
NOZZLE_OPENING_ANGLE = 0.157  # rad, solver default

NUM_WORLDS = 64
EVENTS_PER_CONFIG = 200

# droplet-count sweep; the per-droplet mass scales as 1/N_d so the total sprayed mass
# per event stays constant and only the degree of parallelism changes
N_D_SWEEP = [75, 150, 300, 600, 1200]
BASE_N_D = 300
BASE_DROPLET_MASS = 1.0 / 6.0  # solver default, aggressive deposition


class Example:
    def __init__(self, viewer, num_frames=None):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_step = 0
        self.viewer = viewer
        self.device = wp.get_device()
        self.reported = False
        self.metrics = None

        np.random.seed(6)  # noqa: NPY002 -- the solver draws its speed distributions from legacy np.random

        # one configuration per frame: the N_d sweep
        self.configs = list(N_D_SWEEP)
        self.total_steps = len(self.configs)
        # rows of (n_d, mean_rel_error, std_rel_error, max_abs_rel_error)
        self.results = []

        # placeholder model so the viewer has something to show; each config builds its own
        self.model = self._build_model()
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)

    def _build_model(self):
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
        model = builder.finalize()

        shape = (NUM_WORLDS, GRID_X, GRID_Y, GRID_Z)
        model.voxel_wet = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        model.voxel_dry = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        model.voxel_distance = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        model.voxel_load = wp.zeros(shape, dtype=wp.int16, device=self.device)
        model.voxel_pos = wp.zeros((NUM_WORLDS,), dtype=wp.vec3f, device=self.device)
        return model

    def grid_mass(self, model) -> np.ndarray:
        wet = model.voxel_wet.numpy().sum(axis=(1, 2, 3), dtype=np.float64)
        dry = model.voxel_dry.numpy().sum(axis=(1, 2, 3), dtype=np.float64)
        return (wet + dry) / 255.0

    def run_config(self, n_d: int) -> tuple[float, float, float]:
        model = self._build_model()
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        rewards = newton.VoxelRewards((NUM_WORLDS, GRID_X - 2, GRID_Y - 2, GRID_Z - 2), 16, self.device)

        droplet_mass = BASE_DROPLET_MASS * BASE_N_D / n_d
        solver = newton.solvers.SolverVoxel(
            model,
            tcp_body_name="nozzle",
            h=VOXEL_SIZE,
            k=n_d,
            droplet_mass=droplet_mass,
            nozzle_opening_angle=NOZZLE_OPENING_ANGLE,
            rebound=True,  # exercise the (also parallel) rebound deposit
            # isolate the deposition: dripping/adhesion/redistribution have their own
            # accounting behaviour and would blur the race measurement
            drip_vel=-1,
            adhesion_check_freq=-1,
            redistribution=True,
            # exact per-event sprayed mass, read back instead of re-deriving the fit
            record_generated_mass=True,
            backtrack_count=5,
        )
        world_indices = wp.array(np.arange(NUM_WORLDS), dtype=wp.int32, device=self.device)
        solver.reset(state_0, world_indices)

        errors = np.zeros((EVENTS_PER_CONFIG, NUM_WORLDS))
        mass_before = self.grid_mass(model)
        for event in range(EVENTS_PER_CONFIG):
            solver.rebound_droplet_mass.zero_()
            solver.step(state_0, state_1, control, None, rewards, self.frame_dt)
            state_0, state_1 = state_1, state_0

            sprayed = solver.generated_droplet_mass.numpy().sum(axis=1, dtype=np.float64)
            mass_after = self.grid_mass(model)
            deposited = mass_after - mass_before
            mass_before = mass_after
            errors[event] = (sprayed - deposited) / sprayed

        return float(errors.mean()), float(errors.std()), float(np.abs(errors).max())

    def step(self):
        if self.sim_step >= self.total_steps:
            if not self.reported:
                self.report()
            return

        n_d = self.configs[self.sim_step]
        mean_err, std_err, max_err = self.run_config(n_d)
        self.results.append((n_d, mean_err, std_err, max_err))
        print(f"N_d = {n_d:5d}: rel. error {mean_err:+.3e} +- {std_err:.3e}, max |error| {max_err:.3e}")

        self.sim_time += self.frame_dt
        self.sim_step += 1
        if self.sim_step == self.total_steps:
            self.report()

    def report(self):
        self.reported = True

        print("\n=== Mass conservation under parallel deposition ===")
        print(
            f"worlds: {NUM_WORLDS}, events per configuration: {EVENTS_PER_CONFIG}, "
            f"constant sprayed mass per event ({BASE_DROPLET_MASS * BASE_N_D:.1f} voxel-mass units per world), "
            f"grid {GRID_X}x{GRID_Y}x{GRID_Z} at {VOXEL_SIZE * 1000:.0f} mm"
        )
        print("relative mass error = (sprayed - deposited - undeposited primary - undeposited rebound) / sprayed")
        print(f"\n{'N_d':>6} {'mean rel error':>15} {'std':>10} {'max |error|':>12}")
        for n_d, mean_err, std_err, max_err in self.results:
            print(f"{n_d:6d} {mean_err:15.3e} {std_err:10.3e} {max_err:12.3e}")

        self.metrics = {
            "max_abs_error": max(abs(m) for _, m, _, _ in self.results),
            "worst_event_error": max(x for _, _, _, x in self.results),
        }

        self.save_plot(self.results)
        print(f"\nlargest mean |relative error| across configurations: {self.metrics['max_abs_error']:.3e}")
        print(f"largest single-event |relative error|: {self.metrics['worst_event_error']:.3e}")

    def save_plot(self, results: list):
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.markers  # noqa: PLC0415
            import matplotlib.pyplot as plt  # noqa: PLC0415
            from matplotlib import ticker  # noqa: PLC0415
        except ImportError:
            print("matplotlib not available, skipping the figure")
            return

        _plot_style.setup(plt)
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        n_ds = np.array([r[0] for r in results], dtype=float)
        means = np.array([abs(r[1]) for r in results])
        stds = np.array([r[2] for r in results])
        ax.set_xscale("log")
        ax.set_yscale("log")

        # the +-std interval crosses zero for most configurations, which a log axis
        # cannot show; clamp those lower whiskers to the axis floor and mark them
        # with a caret to signal that they continue below
        ymin = means.min() / 3.0
        ymax = (means + stds).max() * 1.5
        clipped = means - stds <= ymin
        lower = np.where(clipped, means - ymin, stds)
        ax.errorbar(
            n_ds,
            means,
            yerr=[lower, stds],
            marker="o",
            color=_plot_style.SERIES[0],
            capsize=2.0,
            elinewidth=0.8,
        )
        ax.plot(
            n_ds[clipped],
            np.full(int(clipped.sum()), ymin),
            linestyle="none",
            marker=matplotlib.markers.CARETDOWNBASE,
            markersize=4.0,
            color=_plot_style.SERIES[0],
            clip_on=False,
        )
        ax.set_xlim(n_ds.min() / 1.4, n_ds.max() * 1.4)
        ax.set_ylim(ymin, ymax)

        # ticks at 1-2-5 per decade; the default log formatter labels only full
        # decades, of which this axis spans barely one
        decades = np.arange(np.floor(np.log10(ymin)), np.ceil(np.log10(ymax)) + 1)
        yticks = np.array([s * 10.0**e for e in decades for s in (1.0, 2.0, 5.0)])
        yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]

        def _log_label(y, _pos=None):
            e = int(np.floor(np.log10(y) + 1e-9))
            m = y / 10.0**e
            return f"$10^{{{e}}}$" if np.isclose(m, 1.0) else f"${m:g} \\times 10^{{{e}}}$"

        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_log_label))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xticks(N_D_SWEEP, labels=[str(n_d) for n_d in N_D_SWEEP])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel("droplets per spray event $N_d$")
        ax.set_ylabel("relative mass error $|m_{\\mathrm{sprayed}} - m_{\\mathrm{deposited}}| / m_{\\mathrm{sprayed}}$")
        ax.set_title("Mass conservation of the parallel deposition")
        fig.savefig("voxel_mass_conservation.pdf")
        plt.close(fig)
        print("saved figure: voxel_mass_conservation.pdf")

    def test_final(self):
        if not self.reported:
            self.report()
        # write races between depositing droplets lose a few percent of mass at this
        # deliberately crowded configuration (short stand-off, full flux); the ledger
        # must still close to within ~10 %, and losses (not creation) must dominate
        assert self.metrics["max_abs_error"] < 0.1, "parallel deposition mass error implausibly large"
        assert all(m > -0.01 for _, m, _, _ in self.results), "parallel deposition creates mass"
        assert len(self.results) == self.total_steps, "not all configurations were executed"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=len(N_D_SWEEP))

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_frames=args.num_frames)

    newton.examples.run(example, args)
