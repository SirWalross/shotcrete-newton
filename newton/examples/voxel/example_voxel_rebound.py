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

"""Rebound-rate validation of the voxel shotcrete solver against literature curves.

Each world sprays a flat vertical wall with a stationary nozzle for one episode and
records the rebound mass fraction (rebounded mass / mass leaving the nozzle). Two
sweeps share the 64 worlds:

* worlds 0..31  -- incidence angle sweep, 0..75 deg from the wall normal, at a fixed
  1.0 m nozzle distance measured along the spray axis;
* worlds 32..63 -- nozzle distance sweep, 0.5..1.8 m along the spray axis, at
  perpendicular incidence.

The solver's rebound model, ``rate = min(0.1 + 0.2*|sin(theta)| + 0.3*(1.2 - d)^2, 1)``,
was distilled from wet-mix shotcrete literature (Melbye; Armelin & Banthia 1998;
Su et al. 2022); this example checks that the *simulated* episode-level rebound (which
also feels the spray-cone width, gravity bending of the droplet paths, and the evolving
deposit geometry) actually reproduces the reported wet-mix range of roughly 5-40 % and
the published angle/distance trends. All results are printed as raw numbers.

Run with::

    uv run -m newton.examples voxel_rebound --viewer null --num-worlds 64 --num-frames 100
"""

import numpy as np
import warp as wp

import newton
import newton.examples

# grid layout: padded voxel-array dimensions (world axes: x lateral, y towards wall, z up).
# A coarser grid than the lateral-flow example (10 mm voxels) keeps the up-to-2 m-deep,
# tilted-spray domain affordable; the rebound rate depends only on physical angle and
# distance, not on the voxel resolution.
GRID_X = 240
GRID_Y = 190
GRID_Z = 66
VOXEL_SIZE = 0.010  # m, solver parameter `h`

WALL_HIT_X = 130  # voxel x of the point on the wall the nozzle aims at
NOZZLE_Z = 44  # voxel z of the nozzle (and, minus gravity drop, of the wall hit point)

# spray configuration (solver defaults unless noted; the incident-mass bookkeeping
# below restates the droplet distribution, keep in sync with SolverVoxel)
DROPLET_COUNT = 300  # solver parameter `k`
# reduced from the 1/6 default so that the coarse 10 mm voxels do not accrete an
# unrealistically fast-growing deposit that would shift the effective nozzle distance
DROPLET_MASS = 1.0 / 48.0
NOZZLE_OPENING_ANGLE = 0.157  # rad, solver default

ANGLE_SWEEP_DEG = np.linspace(0.0, 75.0, 32)
ANGLE_SWEEP_DISTANCE = 1.0  # m, fixed along-axis distance for the angle sweep
DISTANCE_SWEEP_M = np.linspace(0.5, 1.8, 32)  # capped at 1.8 m: the solver integrates
# droplet paths for at most ~2.0 m, so beyond ~1.8 m parts of the cone would never
# reach the wall

# ---------------------------------------------------------------------------
# Literature reference values (wet-mix shotcrete unless noted).
# ---------------------------------------------------------------------------
# Melbye, "Sprayed Concrete for Rock Support" (MBT, 1994-2001): theoretical rebound
# curves vs. spray angle and vs. nozzle distance, digitized from their reproduction in
# Li et al., "Study on material rebound characteristics of shotcrete: A review" (Fig. 4).
# Angles are measured from the wall normal (0 = perpendicular); the distance points
# beyond 1 m assume a linear axis between the figure's 1 m and >3 m labels.
MELBYE_ANGLE_DEG = np.array([0.0, 18.0, 36.0, 54.0, 72.0, 90.0])
MELBYE_ANGLE_REBOUND = np.array([0.11, 0.15, 0.27, 0.47, 0.73, 1.00])
MELBYE_DISTANCE_M = np.array([0.2, 0.36, 0.52, 0.68, 0.84, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
MELBYE_DISTANCE_REBOUND = np.array([0.61, 0.48, 0.35, 0.23, 0.14, 0.11, 0.15, 0.24, 0.35, 0.48, 0.62])
# Melbye/Sika practice numbers: wet-mix rebound under good practice (perpendicular,
# 1-2 m stand-off): 5-15 %.
MELBYE_WETMIX_RANGE = (0.05, 0.15)
# Overall wet-mix span reported across the literature (Melbye/Sika 5-15 % good practice;
# Su et al. 9-15 % measured, ~20-34 % off-optimum; overhead/poor practice to ~40 %).
LITERATURE_WETMIX_RANGE = (0.05, 0.40)
# Su et al. (2022), "Analysis of Rebound Rate of Wet Shotcrete Based on Experiment and
# Discrete Element Method", Shock and Vibration 2022:1840580. Incidence was fixed at
# perpendicular, so this source constrains the distance sweep only. Field tests
# (Table 3, across aggregate sizes/nozzles) and the DEM distance sweep (Fig. 11,
# 5-11 mm aggregate; U-shaped with the minimum at 0.9-1.0 m).
SU_2022_FIELD = {0.6: (0.114, 0.153), 1.0: (0.091, 0.112)}  # distance m -> rebound range
SU_2022_DEM_DISTANCE_M = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
SU_2022_DEM_REBOUND = np.array([0.216, 0.203, 0.196, 0.180, 0.202, 0.209, 0.218])
# Armelin & Banthia (1998), "Mechanics of aggregate rebound in shotcrete" (Parts I/II):
# single-aggregate impact experiments at normal incidence, dry-mix losses up to 50 %;
# their penetration/adhesion-work model implies rebound grows as the impact tilts away
# from the normal (only this trend is compared -- dry-mix magnitudes exceed wet-mix).
ARMELIN_BANTHIA_TREND = (
    "rebound minimal at perpendicular impact, increasing with inclination; dry-mix losses up to 50 %"
)


def incident_ray_masses(count: int, opening_angle: float, droplet_mass: float) -> np.ndarray:
    """Per-ray droplet masses as generated by the solver, before redistribution.

    Mirrors ``update_directions_kernel``/``mass_ratio`` of the voxel solver; keep in
    sync when the solver's droplet mass distribution changes.
    """
    i = np.arange(count)
    z = 1.0 - (1.0 - np.cos(opening_angle)) * (i + 0.5) / count
    r = np.arccos(z) / opening_angle
    ratio = (0.713 * np.exp(-(((r - 0.207) / 0.357) ** 2)) + 0.711 * np.exp(-(((r + 0.207) / 0.357) ** 2))) * 4.0
    return ratio * droplet_mass


def rebound_formula(theta_rad: float, distance_m: float) -> float:
    """Nominal per-droplet rebound rate of the solver at the given impact geometry."""
    return min(0.1 + 0.2 * abs(np.sin(theta_rad)) + 0.3 * (1.2 - distance_m) ** 2, 1.0)


def quat_from_x_to(direction: np.ndarray) -> wp.quat:
    """Quaternion rotating the +x axis onto ``direction`` (unit vector)."""
    x_axis = np.array([1.0, 0.0, 0.0])
    d = direction / np.linalg.norm(direction)
    axis = np.cross(x_axis, d)
    s = np.linalg.norm(axis)
    c = float(np.dot(x_axis, d))
    if s < 1.0e-12:
        return wp.quat_identity() if c > 0.0 else wp.quat(0.0, 0.0, 1.0, 0.0)
    axis = axis / s
    half = 0.5 * np.arctan2(s, c)
    return wp.quat(*(np.sin(half) * axis), np.cos(half))


def grid_to_world(gx: float, gy: float, gz: float) -> wp.vec3:
    """World-space nozzle position for a target voxel-grid position.

    Inverts the solver's grid mapping (``positions = trunc(world / h) + (width // 2, 0, 0)``)
    and biases each coordinate away from the truncation boundary.
    """
    n = np.array([gx - GRID_X // 2, gy, gz], dtype=np.float64)
    bias = np.where(n >= 0.0, 0.25, -0.25)
    return wp.vec3(*((n + bias) * VOXEL_SIZE))


class Example:
    def __init__(self, viewer, num_worlds=64, num_frames=100):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_step = 0
        self.total_steps = num_frames
        self.num_worlds = num_worlds
        self.viewer = viewer
        self.device = wp.get_device()
        self.reported = False

        np.random.seed(2022)  # noqa: NPY002 -- the solver draws its speed distributions from legacy np.random

        # split the worlds between the two sweeps
        n_angle = num_worlds // 2
        angles_deg = np.interp(np.arange(n_angle), np.linspace(0, n_angle - 1, len(ANGLE_SWEEP_DEG)), ANGLE_SWEEP_DEG)
        distances = np.interp(
            np.arange(num_worlds - n_angle),
            np.linspace(0, num_worlds - n_angle - 1, len(DISTANCE_SWEEP_M)),
            DISTANCE_SWEEP_M,
        )
        self.theta_deg = np.concatenate([angles_deg, np.zeros(num_worlds - n_angle)])
        self.distance_m = np.concatenate([np.full(n_angle, ANGLE_SWEEP_DISTANCE), distances])
        self.sweep = ["angle"] * n_angle + ["distance"] * (num_worlds - n_angle)

        wall_j = GRID_Y - 2
        self.wall_j = wall_j

        nozzle = newton.ModelBuilder()
        # the solver looks up the TCP body via the `/World/envs/env_*/<name>` USD-style key
        body = nozzle.add_body(xform=wp.transform_identity(), key="/World/envs/env_0/nozzle", mass=1.0)
        nozzle.add_shape_sphere(body, radius=0.02)

        builder = newton.ModelBuilder()
        for w in range(num_worlds):
            theta = np.deg2rad(self.theta_deg[w])
            d = self.distance_m[w]
            # aim at the fixed wall point; the nozzle sits `d` back along the spray axis,
            # tilted in the horizontal plane so gravity acts symmetrically across the sweep
            direction = np.array([np.sin(theta), np.cos(theta), 0.0])
            gx = WALL_HIT_X - d * np.sin(theta) / VOXEL_SIZE
            gy = wall_j - d * np.cos(theta) / VOXEL_SIZE
            assert gx >= 2 and gy >= 2, f"grid too small for world {w} (theta={theta}, d={d})"
            xform = wp.transform(grid_to_world(gx, gy, NOZZLE_Z), quat_from_x_to(direction))
            builder.add_world(nozzle, xform=xform)
        self.model = builder.finalize()

        shape = (num_worlds, GRID_X, GRID_Y, GRID_Z)
        self.model.voxel_wet = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_dry = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_distance = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_load = wp.zeros(shape, dtype=wp.int16, device=self.device)
        self.model.voxel_pos = wp.zeros((num_worlds,), dtype=wp.vec3f, device=self.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        self.rewards = newton.VoxelRewards((num_worlds, GRID_X - 2, GRID_Y - 2, GRID_Z - 2), 2, self.device)

        self.solver = newton.solvers.SolverVoxel(
            self.model,
            tcp_body_name="nozzle",
            h=VOXEL_SIZE,
            k=DROPLET_COUNT,
            droplet_mass=DROPLET_MASS,
            nozzle_opening_angle=NOZZLE_OPENING_ANGLE,
            rebound=True,
            # respreading recycles eroded material into the droplet masses, which would
            # contaminate the rebound bookkeeping (and, at grazing incidence, its
            # overlapping-ray races multiply mass); disable the in-flight mass shaping to
            # keep the per-droplet masses at their incident values until the rebound step
            redistribution=False,
        )
        self.solver.reset(self.state_0, wp.array(np.arange(num_worlds), dtype=wp.int32, device=self.device))

        self.ray_masses = incident_ray_masses(DROPLET_COUNT, NOZZLE_OPENING_ANGLE, DROPLET_MASS)
        self.injected_per_step = float(self.ray_masses.sum())
        self.rebound_mass = np.zeros(num_worlds)
        self.hit_mass = np.zeros(num_worlds)
        self.baseline_grid_mass = self.grid_mass()
        self.metrics = None

        self.viewer.set_model(self.model)

    def grid_mass(self) -> np.ndarray:
        """Total material mass in the voxel grid per world, in voxel-mass units."""
        wet = self.model.voxel_wet.numpy().sum(axis=(1, 2, 3), dtype=np.float64)
        dry = self.model.voxel_dry.numpy().sum(axis=(1, 2, 3), dtype=np.float64)
        return (wet + dry) / 255.0

    def step(self):
        if self.sim_step >= self.total_steps:
            if not self.reported:
                self.report()
            return

        self.rewards.step()
        # the rebound kernel skips droplets that leave the grid, so clear the buffer to
        # avoid re-counting stale values from the previous spray event
        self.solver.rebound_droplet_mass.zero_()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.rewards, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        self.rebound_mass += self.solver.rebound_droplet_mass.numpy().sum(axis=1, dtype=np.float64)

        # diagnostic: per-droplet (pre-redistribution) mass that reached the wall region
        hits = self.solver.ray_trajectory.numpy()[:, :, 0, :].astype(np.int64)
        near_wall = (
            (hits[..., 0] >= 1)
            & (hits[..., 0] < GRID_X - 1)
            & (hits[..., 1] >= self.wall_j - 25)
            & (hits[..., 1] < GRID_Y)
            & (hits[..., 2] >= 1)
            & (hits[..., 2] < GRID_Z - 1)
        )
        self.hit_mass += (near_wall * self.ray_masses[None, :]).sum(axis=1)

        self.sim_time += self.frame_dt
        self.sim_step += 1
        if self.sim_step == self.total_steps:
            self.report()

    def report(self):
        self.reported = True
        injected = self.injected_per_step * self.total_steps
        rebound_frac = self.rebound_mass / injected
        hit_share = self.hit_mass / injected
        deposited_frac = (self.grid_mass() - self.baseline_grid_mass) / injected
        formula = np.array(
            [rebound_formula(np.deg2rad(t), d) for t, d in zip(self.theta_deg, self.distance_m, strict=True)]
        )

        n_angle = self.sweep.count("angle")
        angle_r = rebound_frac[:n_angle]
        dist_r = rebound_frac[n_angle:]
        angle_corr = float(np.corrcoef(self.theta_deg[:n_angle], angle_r)[0, 1])
        dist_min_at = float(self.distance_m[n_angle:][np.argmin(dist_r)])

        self.metrics = {
            "rebound_min": float(rebound_frac.min()),
            "rebound_max": float(rebound_frac.max()),
            "angle_corr": angle_corr,
            "angle_rise": float(angle_r[-1] - angle_r[0]),
            "dist_min_at": dist_min_at,
            "deposited_frac_mean": float(deposited_frac.mean()),
        }

        print("\n=== Rebound rate vs. literature (Melbye; Armelin & Banthia 1998; Su et al. 2022) ===")
        print(
            f"worlds: {self.num_worlds} (episodes), spray events per episode: {self.total_steps}, "
            f"voxel size: {VOXEL_SIZE * 1000:.0f} mm, droplets per event: {DROPLET_COUNT}, "
            f"injected mass per event: {self.injected_per_step:.3f}"
        )
        print(
            "columns: sweep type, incidence angle (deg from wall normal), nozzle distance (m along axis), "
            "simulated episode rebound fraction, nominal solver formula value, "
            "share of droplet mass reaching the wall region, deposited/injected mass"
        )
        print(
            f"\n{'world':>5} {'sweep':>8} {'theta_deg':>9} {'dist_m':>6} "
            f"{'rebound_sim':>11} {'formula':>8} {'hit_share':>9} {'deposited':>9}"
        )
        for w in range(self.num_worlds):
            print(
                f"{w:5d} {self.sweep[w]:>8} {self.theta_deg[w]:9.2f} {self.distance_m[w]:6.3f} "
                f"{rebound_frac[w]:11.4f} {formula[w]:8.4f} {hit_share[w]:9.4f} {deposited_frac[w]:9.4f}"
            )

        print("\n--- Literature comparison: angle sweep (fixed d = 1.0 m) ---")
        melbye_angle = np.interp(self.theta_deg[:n_angle], MELBYE_ANGLE_DEG, MELBYE_ANGLE_REBOUND)
        print("theta_deg, rebound_sim, melbye_curve:")
        for w in range(n_angle):
            print(f"  {self.theta_deg[w]:6.2f} {angle_r[w]:8.4f} {melbye_angle[w]:8.4f}")
        rmse_melbye_angle = float(np.sqrt(np.mean((angle_r - melbye_angle) ** 2)))
        print(f"RMSE vs. Melbye angle curve (0-75 deg): {rmse_melbye_angle:.4f}")
        print(f"Armelin & Banthia (1998): {ARMELIN_BANTHIA_TREND}")
        print(
            f"simulated angle trend: rebound({self.theta_deg[0]:.0f} deg) = {angle_r[0]:.4f} -> "
            f"rebound({self.theta_deg[n_angle - 1]:.0f} deg) = {angle_r[-1]:.4f}, "
            f"correlation with angle: {angle_corr:.3f}"
        )

        print("\n--- Literature comparison: distance sweep (perpendicular) ---")
        dist_d = self.distance_m[n_angle:]
        melbye_dist = np.interp(dist_d, MELBYE_DISTANCE_M, MELBYE_DISTANCE_REBOUND)
        su_dem = np.interp(dist_d, SU_2022_DEM_DISTANCE_M, SU_2022_DEM_REBOUND)
        in_su = (dist_d >= SU_2022_DEM_DISTANCE_M[0]) & (dist_d <= SU_2022_DEM_DISTANCE_M[-1])
        print("dist_m, rebound_sim, melbye_curve, su_2022_dem (5-11 mm aggregate, 0.6-1.2 m only):")
        for i, d in enumerate(dist_d):
            su_txt = f"{su_dem[i]:8.4f}" if in_su[i] else "       -"
            print(f"  {d:6.3f} {dist_r[i]:8.4f} {melbye_dist[i]:8.4f} {su_txt}")
        rmse_melbye_dist = float(np.sqrt(np.mean((dist_r - melbye_dist) ** 2)))
        rmse_su_dem = float(np.sqrt(np.mean((dist_r[in_su] - su_dem[in_su]) ** 2)))
        print(f"RMSE vs. Melbye distance curve (0.5-1.8 m): {rmse_melbye_dist:.4f}")
        print(f"RMSE vs. Su et al. (2022) DEM distance curve (0.6-1.2 m): {rmse_su_dem:.4f}")
        for d_field, (lo, hi) in SU_2022_FIELD.items():
            sim_at = float(dist_r[np.argmin(np.abs(dist_d - d_field))])
            print(f"Su et al. (2022) field rebound at {d_field:.1f} m: {lo:.3f}-{hi:.3f}; simulated: {sim_at:.4f}")
        print(
            f"distance sweep: minimum rebound at d = {dist_min_at:.3f} m "
            f"(solver formula minimum at 1.2 m; Melbye ~1.0 m; Su et al. DEM 0.9-1.0 m)"
        )

        print("\n--- Overall range check ---")
        print(
            f"Melbye/Sika wet-mix good practice (perpendicular, 1-2 m): "
            f"{MELBYE_WETMIX_RANGE[0]:.2f}-{MELBYE_WETMIX_RANGE[1]:.2f}; "
            f"simulated at (0 deg, 1.2 m): "
            f"{float(dist_r[np.argmin(np.abs(dist_d - 1.2))]):.4f}"
        )
        print(
            f"literature wet-mix span {LITERATURE_WETMIX_RANGE[0]:.2f}-{LITERATURE_WETMIX_RANGE[1]:.2f}; "
            f"simulated sweep range: {rebound_frac.min():.4f}-{rebound_frac.max():.4f}"
        )

    def test_final(self):
        if not self.reported:
            self.report()
        assert self.metrics["rebound_min"] > 0.03, "rebound fraction below any reported wet-mix value"
        assert self.metrics["rebound_max"] < 0.6, "rebound fraction above the reported wet-mix range"
        assert self.metrics["angle_corr"] > 0.8, "rebound does not increase with incidence angle"
        assert self.metrics["angle_rise"] > 0.05, "angle sweep spans too little rebound variation"
        assert 0.9 <= self.metrics["dist_min_at"] <= 1.5, "distance-sweep minimum far from the 1.2 m optimum"
        assert self.metrics["deposited_frac_mean"] > 0.5, "most sprayed material never deposited"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=64, help="Total number of episodes (worlds).")
    parser.set_defaults(num_frames=100)

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_worlds=args.num_worlds, num_frames=args.num_frames)

    newton.examples.run(example, args)
