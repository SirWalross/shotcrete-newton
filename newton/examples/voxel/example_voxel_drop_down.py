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

"""Qualitative visualization of cohesive failure (drop-down) in the voxel solver.

A stationary nozzle sprays perpendicularly at a fixed spot on a vertical wall. At the
default delivery rate the wet material accumulates faster than it solidifies, the
deposit's weight eventually exceeds the load capacity propagated through its
adhesion/shear/compression network, and the solver's ``drop_down_kernel`` deletes the
overloaded voxels and drops them to the floor -- the cohesive-failure ("drop down")
event this example captures. With ``--rebar`` a rebar mesh sits ``--rebar-cover``
meters (default 0.10) in front of the wall; the growing deposit engulfs the bars,
which anchor the load and distance fields, and the mesh carries the deposit well past
the bare wall's support envelope -- the first drop-down happens markedly later (about
step 590 instead of 430 at the default settings) and less material is lost. The cover
must stay inside the bare deposit's critical protrusion (~0.125 m here): a mesh
further out is never reached by the deposit and cannot support anything. With
``--overhead`` the nozzle instead sprays straight up at a ceiling: the deposit hangs
by adhesion alone (no shear path to a vertical anchor), so the drop-down happens at
much thinner layers and much sooner than on the wall.

The failure is then widened into a crater seeded at the break surface: failed voxels
that snapped off still-standing material cast a radially decaying damage ball (peak
``failure_damage``, falloff ``failure_damage_decay`` per voxel, so a radius of at most
``failure_damage / failure_damage_decay``) onto the surrounding deposit, and a single
follow-up drop-down removes everything whose remaining load capacity went negative.
Chunks that fall merely because they were already disconnected seed nothing -- craters
around falling debris would cascade through the healthy deposit. The cut is
all-or-nothing: below ``failure_trigger`` break-surface voxels (fresh spray dripping
off an at-capacity face) nothing fires and the shed simply falls; at or above it the
full crater is carved and further cuts are suppressed for ``failure_cooldown`` checks,
so the aftershocks of the collapse shed as plain drop-downs -- afterwards the spray
builds the surface back up until the deposit outgrows its load envelope and the cycle
repeats.

Every step the exposed deposit surface is extracted on the GPU (a voxel is rendered
iff its total density reaches the occupancy threshold and at least one of its six
neighbours does not) and drawn as per-voxel cubes: rebar in rusty steel, shotcrete
shaded from dry beige to wet brown by its wet-mass fraction; wall and floor are drawn
as flat slabs. A fixed camera watches the spray spot from the side.

The failure moment is detected through ``rewards.adhesion_failure_amount`` (the mass
that detached this step). Rendered frames are kept in a ring buffer at one capture
per adhesion check, and once the failure fires the 5 frames before it, the failure
frame itself, and the 4 following captures are written as PNG images (10 total).

For every captured frame a diagnostic figure of the central y-z plane
(``slice_*.png``) is written alongside: the load-capacity field ``voxel_load``
(diverging map, red = overloaded, fails when negative) and the distance-to-support
field ``voxel_distance`` (sequential map), with sub-threshold voxels masked out.

Frame capture needs the OpenGL viewer (``--viewer gl``, optionally ``--headless``);
with ``--viewer null`` the episode still runs, validates the failure detection, and
writes the slice diagnostics. Run with::

    uv run -m newton.examples voxel_drop_down --viewer gl --headless
"""

import os
import sys

# pyglet must see this before it is first imported to pick its EGL backend on
# machines without an X display; harmless to leave unset when a display exists
if "--headless" in sys.argv and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLET_HEADLESS", "1")

import collections

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.voxel import _plot_style

# grid layout: padded voxel-array dimensions (world axes: x lateral, y towards wall, z up)
GRID_X = 130
GRID_Y = 110
GRID_Z = 130
VOXEL_SIZE = 0.005  # m, solver parameter `h`

DROPLET_COUNT = 300  # solver parameter `k`
# at the full default per-event mass the wet delivery rate per voxel sits at the
# per-event solidification rate `tc` already at 1.0 m stand-off (see the lateral-flow
# example); at 0.5 m the flux is ~4x more concentrated, so the stationary spot
# reliably overloads and drops
DROPLET_MASS = 1.0 / 12.0
NOZZLE_DISTANCE = 0.5  # m, perpendicular nozzle-to-wall distance

# optional rebar mesh in front of the wall (--rebar): 15 mm bars at 75 mm pitch. The
# cover must stay inside the bare deposit's critical protrusion (~0.125 m here, the
# ~20-voxel support-chain envelope): only then does the growing deposit engulf the
# mesh, get re-anchored by it and demonstrably drop down later; a mesh further out is
# never reached and cannot carry anything
REBAR_COVER = 0.05  # m, wall face to bar axis
REBAR_THICKNESS = 3  # voxels
REBAR_SPACING = 15  # voxels
REBAR_COUNT = (8, 8)  # (vertical bars, horizontal bars)
REBAR_OFFSET = 20  # first bar position (voxels), centers the 8x8 mesh on the spray spot

# material densities, mirroring the solver's encoding (values are wet==dry for the
# static materials): occupancy threshold, rebar marker
DENSITY_HALF = 128
DENSITY_REBAR = 254

COLOR_WALL = wp.vec3(0.58, 0.58, 0.60)
COLOR_FLOOR = wp.vec3(0.42, 0.42, 0.44)
COLOR_REBAR = wp.constant(wp.vec3(0.50, 0.27, 0.19))
COLOR_DRY = wp.constant(wp.vec3(0.76, 0.74, 0.70))
COLOR_WET = wp.constant(wp.vec3(0.44, 0.41, 0.37))


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
    """World-space position for a voxel-grid position (inverse of the solver mapping)."""
    n = np.array([gx - GRID_X // 2, gy, gz], dtype=np.float64)
    bias = np.where(n >= 0.0, 0.25, -0.25)
    return wp.vec3(*((n + bias) * VOXEL_SIZE))


@wp.func
def occupied(wet: wp.array4d(dtype=wp.uint8), dry: wp.array4d(dtype=wp.uint8), i: int, j: int, k: int) -> bool:
    # outside the grid and the static slabs (wall, floor, and -- in overhead mode --
    # the ceiling) count as occupied so the faces shared with them stay hidden
    if i < 0 or i >= wet.shape[1] or j < 0 or k >= wet.shape[3]:
        return True
    if j >= wet.shape[2] - 2 or k <= 0 or k >= wet.shape[3] - 2:
        return True
    return wp.int32(wet[0, i, j, k]) + wp.int32(dry[0, i, j, k]) >= DENSITY_HALF


@wp.kernel
def extract_surface_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    voxel_pos: wp.array(dtype=wp.vec3f),
    h: float,
    count: wp.array(dtype=wp.int32),
    xforms: wp.array(dtype=wp.transform),
    colors: wp.array(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    k = k + 1  # the floor plane at k = 0 is rendered as a static slab
    if k >= wet.shape[3] - 2:
        return  # the ceiling slab (overhead mode) is rendered statically too
    w = wp.int32(wet[0, i, j, k])
    d = wp.int32(dry[0, i, j, k])
    if w + d < DENSITY_HALF:
        return
    if (
        occupied(wet, dry, i - 1, j, k)
        and occupied(wet, dry, i + 1, j, k)
        and occupied(wet, dry, i, j - 1, k)
        and occupied(wet, dry, i, j + 1, k)
        and occupied(wet, dry, i, j, k - 1)
        and occupied(wet, dry, i, j, k + 1)
    ):
        return  # fully enclosed, not visible
    idx = wp.atomic_add(count, 0, 1)
    if idx >= xforms.shape[0]:
        return
    p = voxel_pos[0] + wp.vec3(
        (wp.float32(i - wet.shape[1] // 2) + 0.5) * h,
        (wp.float32(j) + 0.5) * h,
        (wp.float32(k) + 0.5) * h,
    )
    xforms[idx] = wp.transform(p, wp.quat_identity())
    if w == DENSITY_REBAR and d == DENSITY_REBAR:
        colors[idx] = COLOR_REBAR
    else:
        wet_frac = wp.float32(w) / wp.float32(wp.max(w + d, 1))
        colors[idx] = COLOR_DRY + (COLOR_WET - COLOR_DRY) * wet_frac


class Example:
    def __init__(
        self,
        viewer,
        num_frames=6000,
        headless=False,
        frame_interval=10,
        failure_threshold=5.0,
        strength_scale=1.0,
        droplet_mass_scale=1.0,
        failure_damage=1000.0,
        failure_decay=10.0,
        failure_trigger=20.0,
        failure_cooldown=10,
        nozzle_distance=NOZZLE_DISTANCE,
        rebar=False,
        rebar_cover=REBAR_COVER,
        overhead=False,
        output_dir="voxel_drop_down_frames",
    ):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_step = 0
        self.num_frames = num_frames
        self.headless = headless
        self.frame_interval = frame_interval
        self.failure_threshold = failure_threshold
        self.output_dir = output_dir
        self.viewer = viewer
        self.device = wp.get_device()

        np.random.seed(2025)  # noqa: NPY002 -- the solver draws its speed distributions from legacy np.random

        assert not (rebar and overhead), "the rebar mesh is only generated in front of the vertical wall"
        self.overhead = overhead
        wall_j = GRID_Y - 2
        ceiling_k = GRID_Z - 2
        if overhead:
            # nozzle below the ceiling center, spraying straight up: the deposit hangs
            # by adhesion alone, so it drops down at much thinner layers than the wall
            base_gz = ceiling_k - round(nozzle_distance / VOXEL_SIZE)
            assert base_gz >= 2, "grid too low for the requested nozzle distance"
            nozzle_grid = (GRID_X // 2, GRID_Y // 2, base_gz)
            nozzle_dir = np.array([0.0, 0.0, 1.0])
        else:
            base_gy = wall_j - round(nozzle_distance / VOXEL_SIZE)
            assert base_gy >= 2, "grid too shallow for the requested nozzle distance"
            nozzle_grid = (GRID_X // 2, base_gy, GRID_Z // 2)
            nozzle_dir = np.array([0.0, 1.0, 0.0])

        nozzle = newton.ModelBuilder()
        # the solver looks up the TCP body via the `/World/envs/env_*/<name>` USD-style key
        body = nozzle.add_body(xform=wp.transform_identity(), key="/World/envs/env_0/nozzle", mass=1.0)
        nozzle.add_shape_sphere(body, radius=0.02)

        builder = newton.ModelBuilder()
        xform = wp.transform(grid_to_world(*nozzle_grid), quat_from_x_to(nozzle_dir))
        builder.add_world(nozzle, xform=xform)
        self.model = builder.finalize()

        shape = (1, GRID_X, GRID_Y, GRID_Z)
        self.model.voxel_wet = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_dry = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_distance = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.model.voxel_load = wp.zeros(shape, dtype=wp.int16, device=self.device)
        self.model.voxel_pos = wp.zeros((1,), dtype=wp.vec3f, device=self.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        self.rewards = newton.VoxelRewards((1, GRID_X - 2, GRID_Y - 2, GRID_Z - 2), 1, self.device)

        self.solver = newton.solvers.SolverVoxel(
            self.model,
            tcp_body_name="nozzle",
            h=VOXEL_SIZE,
            k=DROPLET_COUNT,
            droplet_mass=DROPLET_MASS * droplet_mass_scale,
            shear_strength=3.0 * strength_scale,
            adhesion_strength=1.4 * strength_scale,
            compression_strength=80.0 * strength_scale,
            wet_strength_penalty=1.0,
            failure_damage=failure_damage,
            failure_damage_decay=failure_decay,
            failure_trigger=failure_trigger,
            generate_rebar=rebar,
            # fully spray-transparent bars: with the default interception chance the
            # thin mesh plates over within a few steps, catches the (closer, more
            # concentrated) spray in front of the deposit and sheds from there EARLIER
            # than the bare wall; transparent bars stay inert until the wall deposit
            # engulfs them and only then anchor it from inside
            alpha=0.0,
            use_bounding_boxes=False,
            sigma=0.5,
        )
        # bar axis sits rebar_cover meters in front of the wall face; with --rebar the
        # mesh anchors the deposit (rebar voxels carry LOAD_MAX at distance 0) so the
        # drop-down happens much later and above the load the bare wall could hold
        cover = round(rebar_cover / VOXEL_SIZE)
        assert 2 <= cover <= wall_j - 4, "rebar cover outside the grid"
        rebar_settings = {
            "rebar_offset_hor": wp.array([wp.vec3i(REBAR_OFFSET, wall_j - cover, 0)], dtype=wp.vec3i),
            "rebar_offset_ver": wp.array([wp.vec3i(0, wall_j - cover, REBAR_OFFSET)], dtype=wp.vec3i),
            "rebar_thickness": wp.array([REBAR_THICKNESS], dtype=wp.int32),
            "rebar_spacing": wp.array([wp.vec2i(REBAR_SPACING, REBAR_SPACING)], dtype=wp.vec2i),
            "rebar_count": REBAR_COUNT,
        }
        self.world_indices = wp.array([0], dtype=wp.int32, device=self.device)
        self.solver.reset(self.state_0, self.world_indices, rebar_settings=rebar_settings)
        if overhead:
            # anchored ceiling slab, mirroring the wall/floor slabs the solver resets
            self.model.voxel_wet[:, :, :, ceiling_k:].fill_(255)
            self.model.voxel_dry[:, :, :, ceiling_k:].fill_(255)
            self.model.voxel_distance[:, :, :, ceiling_k:].fill_(0)

        # surface-voxel instancing buffers (compacted on the GPU each frame)
        self.max_instances = 300_000
        self.surf_xforms = wp.empty(self.max_instances, dtype=wp.transform, device=self.device)
        self.surf_colors = wp.empty(self.max_instances, dtype=wp.vec3, device=self.device)
        self.surf_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.capacity_warned = False

        # static wall and floor slabs (world-space extents of the corresponding grid slabs)
        half_x = GRID_X // 2 * VOXEL_SIZE
        self.wall_slab = (
            (half_x, VOXEL_SIZE, GRID_Z / 2 * VOXEL_SIZE),
            wp.array(
                [wp.transform(wp.vec3(0.0, (GRID_Y - 1) * VOXEL_SIZE, GRID_Z / 2 * VOXEL_SIZE), wp.quat_identity())],
                dtype=wp.transform,
            ),
            wp.array([COLOR_WALL], dtype=wp.vec3),
        )
        self.floor_slab = (
            (half_x, GRID_Y / 2 * VOXEL_SIZE, VOXEL_SIZE / 2),
            wp.array(
                [wp.transform(wp.vec3(0.0, GRID_Y / 2 * VOXEL_SIZE, VOXEL_SIZE / 2), wp.quat_identity())],
                dtype=wp.transform,
            ),
            wp.array([COLOR_FLOOR], dtype=wp.vec3),
        )
        self.slabs = [("/wall", self.wall_slab), ("/floor", self.floor_slab)]
        if overhead:
            self.slabs.append(
                (
                    "/ceiling",
                    (
                        (half_x, GRID_Y / 2 * VOXEL_SIZE, VOXEL_SIZE),
                        wp.array(
                            [
                                wp.transform(
                                    wp.vec3(0.0, GRID_Y / 2 * VOXEL_SIZE, (GRID_Z - 1) * VOXEL_SIZE),
                                    wp.quat_identity(),
                                )
                            ],
                            dtype=wp.transform,
                        ),
                        wp.array([COLOR_WALL], dtype=wp.vec3),
                    ),
                )
            )

        # ring-buffer capture state; entries are (step, image or None, field slices)
        self.pre_frames = collections.deque(maxlen=10)
        self.post_frames = []
        self.post_count = 100  # failure frame + 4 after
        self.next_capture = None
        self.failure_step = None
        self.total_lost = 0.0
        self.lost_history = []  # per-step detached mass, for the rebuild check
        self.frames_saved = False
        self.saved_paths = []
        self.saved_frames = []
        self.saved_slice_paths = []
        self.closing = False
        self.can_capture = hasattr(viewer, "get_frame")
        self.slice_x = GRID_X // 2  # y-z diagnostic plane through the spray axis

        self.viewer.set_model(self.model)
        # camera: off-axis view of the spray spot so the nozzle does not occlude it,
        # framing both the deposit and the floor where failed material lands
        if overhead:
            spot_y = GRID_Y // 2 * VOXEL_SIZE
            ceiling_z = ceiling_k * VOXEL_SIZE
            target = np.array([0.0, spot_y, ceiling_z - 0.16])
            pos = np.array([0.40, spot_y - 0.45, ceiling_z - 0.52])
        else:
            wall_y = wall_j * VOXEL_SIZE
            spot_z = GRID_Z // 2 * VOXEL_SIZE
            target = np.array([0.0, wall_y, spot_z - 0.08])
            pos = np.array([0.34, wall_y - 0.52, spot_z + 0.10])
        d = target - pos
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        pitch = float(np.degrees(np.arcsin(d[2] / np.linalg.norm(d))))
        self.viewer.set_camera(wp.vec3(*pos), pitch, yaw)

    def step(self):
        if self.closing or self.sim_step >= self.num_frames:
            print("stopping")
            return
        else:
            print("step", self.sim_step, "time", self.sim_time)

        self.rewards.step()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.rewards, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        lost = float(self.rewards.adhesion_failure_amount.numpy()[0])
        self.total_lost += lost
        self.lost_history.append(lost)
        if self.failure_step is None and lost >= self.failure_threshold:
            self.failure_step = self.sim_step
            print(f"adhesion failure at step {self.sim_step}: {lost:.1f} voxel-mass units dropped")

        self.sim_time += self.frame_dt
        self.sim_step += 1

    def render(self):
        if self.closing:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for name, (scale, xform, color) in self.slabs:
            self.viewer.log_shapes(name, newton.GeoType.BOX, scale, xform, color)
        self._log_surface_voxels()
        self.viewer.end_frame()

        s = self.sim_step - 1  # index of the step this frame shows
        if not self.frames_saved and s >= 0:
            self._capture(s)
            if s + 1 >= self.num_frames and not self.frames_saved:
                print(f"no adhesion failure within {self.num_frames} steps; saving the buffered frames")
                self._save_frames()
        if self.headless and (self.frames_saved or s + 1 >= self.num_frames):
            # nothing left to capture: end the harness loop (run() closes again, safely)
            self.viewer.close()
            self.closing = True

    def _log_surface_voxels(self):
        self.surf_count.zero_()
        wp.launch(
            extract_surface_kernel,
            dim=(GRID_X, GRID_Y - 2, GRID_Z - 1),
            inputs=[
                self.model.voxel_wet,
                self.model.voxel_dry,
                self.model.voxel_pos,
                VOXEL_SIZE,
                self.surf_count,
                self.surf_xforms,
                self.surf_colors,
            ],
            device=self.device,
        )
        n = int(self.surf_count.numpy()[0])
        if n > self.max_instances:
            if not self.capacity_warned:
                print(f"surface voxel count {n} exceeds capacity {self.max_instances}; rendering a subset")
                self.capacity_warned = True
            n = self.max_instances
        if n > 0:
            half = VOXEL_SIZE / 2
            self.viewer.log_shapes(
                "/deposit", newton.GeoType.BOX, (half, half, half), self.surf_xforms[:n], self.surf_colors[:n]
            )

    def _grab(self):
        try:
            return self.viewer.get_frame().numpy()
        except Exception as e:  # CUDA-GL interop is unavailable on some setups
            print(f"frame capture unavailable ({e}); continuing without saving images")
            self.can_capture = False
            return None

    def _slice_fields(self):
        """Center y-z slices of the diagnostic fields (y rows, z columns)."""
        cx = self.slice_x
        wet = self.model.voxel_wet.numpy()[0, cx].astype(np.int32)
        dry = self.model.voxel_dry.numpy()[0, cx].astype(np.int32)
        return {
            "load": self.model.voxel_load.numpy()[0, cx].astype(np.int32),
            "distance": self.model.voxel_distance.numpy()[0, cx].astype(np.int32),
            "density": wet + dry,
        }

    def _capture(self, s: int):
        if self.failure_step is None:
            if s % self.frame_interval == 0:
                self.pre_frames.append((s, self._grab() if self.can_capture else None, self._slice_fields()))
        else:
            if self.next_capture is None:
                self.next_capture = s  # first render after the failing step: the failure frame
            if s >= self.next_capture and len(self.post_frames) < self.post_count:
                self.post_frames.append((s, self._grab() if self.can_capture else None, self._slice_fields()))
                self.next_capture = s + self.frame_interval
            if len(self.post_frames) >= self.post_count:
                self._save_frames()

    def _save_frames(self):
        self.frames_saved = True
        frames = list(self.pre_frames) + self.post_frames
        if not frames:
            print("no frames captured, nothing to save")
            return
        try:
            from matplotlib.image import imsave  # noqa: PLC0415
        except ImportError:
            from PIL import Image  # noqa: PLC0415

            def imsave(path, img):
                Image.fromarray(img).save(path)

        os.makedirs(self.output_dir, exist_ok=True)
        for idx, (step, img, slices) in enumerate(frames):
            tag = "_failure" if step == self.failure_step else ""
            if img is not None:
                path = os.path.join(self.output_dir, f"frame_{idx:02d}_step{step:04d}{tag}.png")
                imsave(path, img)
                self.saved_paths.append(path)
                self.saved_frames.append(img)
            self._save_slice_plot(idx, step, tag, slices)
        print(
            f"saved {len(self.saved_paths)} frames and {len(self.saved_slice_paths)} "
            f"slice diagnostics to {self.output_dir}/"
        )

    def _save_slice_plot(self, idx: int, step: int, tag: str, slices: dict):
        """Two-panel heatmap of the load and distance fields in the central y-z plane."""
        try:
            import matplotlib  # noqa: PLC0415

            # matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            return

        _plot_style.setup(plt)

        # rows: z (up), cols: y (wall at the right). Sub-threshold voxels are masked so
        # only load-bearing material shows -- except negative-load voxels, which stay
        # visible even when already emptied: those are exactly the ones drop_down removed
        empty = slices["density"] < 25
        load = np.where(empty & (slices["load"] >= 0), np.nan, slices["load"]).T
        dist = np.where(empty, np.nan, slices["distance"]).T

        fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), sharey=True, sharex=True)
        cmap_load = matplotlib.colormaps["RdBu"].copy()
        cmap_load.set_bad("0.94")
        im0 = axes[0].imshow(
            load,
            origin="lower",
            cmap=cmap_load,
            norm=matplotlib.colors.SymLogNorm(linthresh=32.0, vmin=-10.0, vmax=2550.0),
            interpolation="nearest",
        )
        axes[0].set_title(f"load capacity (step {step})")
        axes[0].set_xlabel("y (voxels, wall right)")
        axes[0].set_ylabel("z (voxels)")
        fig.colorbar(im0, ax=axes[0], label="load (negative fails)")

        cmap_dist = matplotlib.colormaps["viridis"].copy()
        cmap_dist.set_bad("0.94")
        cmap_dist.set_over("orangered")  # flags stale / unsupported values incl. DISTANCE_MAX
        im1 = axes[1].imshow(dist, origin="lower", cmap=cmap_dist, vmin=0.0, vmax=64.0, interpolation="nearest")
        axes[1].set_title(f"distance to support (step {step})")
        axes[1].set_xlabel("y (voxels, wall right)")
        fig.colorbar(im1, ax=axes[1], label="distance", extend="max")

        cmap_dens = matplotlib.colormaps["RdBu"].copy()
        cmap_dens.set_bad("0.94")
        im2 = axes[2].imshow(
            slices["density"],
            origin="lower",
            cmap=cmap_dens,
            norm=matplotlib.colors.SymLogNorm(linthresh=32.0, vmin=0.0, vmax=255.0),
            interpolation="nearest",
        )
        axes[2].set_title(f"load capacity (step {step})")
        axes[2].set_xlabel("y (voxels, wall right)")
        axes[2].set_ylabel("z (voxels)")
        fig.colorbar(im2, ax=axes[2], label="density")

        fig.suptitle(f"y-z slice at x = {self.slice_x}{' -- failure step' if tag else ''}")
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"slice_{idx:02d}_step{step:04d}{tag}.png")
        # plt.show()
        fig.savefig(path, dpi=200)
        self.saved_slice_paths.append(path)

    def test_final(self):
        assert self.failure_step is not None, (
            "no adhesion failure occurred; lower --strength-scale or raise --droplet-mass-scale"
        )
        assert self.total_lost >= self.failure_threshold, "detached mass below the failure threshold"
        # rebuild check: after the failure event the crater must stop chewing, i.e. a
        # window with only marginal shedding must exist so fresh spray can re-accumulate
        # on the break surface. The overfed spot keeps shedding a little at every
        # adhesion check and the exact level varies run to run (GPU atomics), so the
        # bound is relative: the quietest post-failure window must lose far less than
        # the loudest one -- a permanently cascading failure has no such contrast.
        post = self.lost_history[self.failure_step + 1 :]
        if len(post) >= 100:
            window = 50
            sums = [sum(post[start : start + window]) for start in range(len(post) - window + 1)]
            quiet = min(sums) < max(50.0, 0.2 * max(sums))
            assert quiet, (
                f"failure keeps propagating: quietest {window}-step window lost {min(sums):.0f}, "
                f"loudest {max(sums):.0f}"
            )
        if self.saved_paths:
            assert all(os.path.exists(p) for p in self.saved_paths), "saved frame files are missing"
            assert len(self.saved_paths) >= self.post_count + 1, "too few frames captured around the failure"
            assert np.any(self.saved_frames[0] != self.saved_frames[-1]), "captured frames are identical"
        if self.saved_slice_paths:
            assert all(os.path.exists(p) for p in self.saved_slice_paths), "slice diagnostic files are missing"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Steps between captured frames (default: the solver's adhesion-check period).",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=50.0,
        help="Detached mass (voxel-mass units) in one step that counts as the failure event.",
    )
    parser.add_argument(
        "--strength-scale",
        type=float,
        default=1.0,
        help="Multiplier on the adhesion and shear strengths (lower fails sooner).",
    )
    parser.add_argument(
        "--droplet-mass-scale",
        type=float,
        default=1.0,
        help="Multiplier on the per-event droplet mass (higher fails sooner).",
    )
    parser.add_argument(
        "--failure-damage",
        type=float,
        default=1200.0,
        help="Peak load damage cast around just-failed voxels (0 disables the crater cut).",
    )
    parser.add_argument(
        "--failure-decay",
        type=float,
        default=50.0,
        help="Damage lost per voxel of distance; crater radius is at most failure-damage/failure-decay voxels.",
    )
    parser.add_argument(
        "--failure-trigger",
        type=float,
        default=20.0,
        help="Minimum break-surface size (snapped-off voxels) for the crater cut to fire; smaller sheds "
        "(fresh spray dripping off the face) fall without carving anything.",
    )
    parser.add_argument(
        "--failure-cooldown",
        type=int,
        default=10,
        help="Adhesion checks after a full-strength crater during which the cut is suppressed, so the "
        "aftershocks of a collapse shed as plain drop-downs instead of cascading craters.",
    )
    parser.add_argument(
        "--nozzle-distance",
        type=float,
        default=NOZZLE_DISTANCE,
        help="Perpendicular nozzle-to-wall distance in meters.",
    )
    parser.add_argument(
        "--rebar",
        action="store_true",
        help="Embed a rebar mesh in front of the wall; once the deposit engulfs it, the anchored bars "
        "carry it and the drop-down happens markedly later than on the bare wall.",
    )
    parser.add_argument(
        "--rebar-cover",
        type=float,
        default=REBAR_COVER,
        help="Distance from the wall face to the rebar axis in meters (with --rebar). Must stay inside "
        "the bare deposit's ~0.125 m critical protrusion, or the deposit never reaches the mesh.",
    )
    parser.add_argument(
        "--overhead",
        action="store_true",
        help="Spray straight up at a ceiling instead of at the vertical wall; the deposit hangs by "
        "adhesion alone, so it drops down at much thinner layers and much sooner.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="voxel_drop_down_frames", help="Directory for the saved PNG frames."
    )
    parser.set_defaults(num_frames=6000)

    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer,
        num_frames=args.num_frames,
        headless=args.headless,
        frame_interval=args.frame_interval,
        failure_threshold=args.failure_threshold,
        strength_scale=args.strength_scale,
        droplet_mass_scale=args.droplet_mass_scale,
        failure_damage=args.failure_damage,
        failure_decay=args.failure_decay,
        failure_trigger=args.failure_trigger,
        failure_cooldown=args.failure_cooldown,
        nozzle_distance=args.nozzle_distance,
        rebar=args.rebar,
        rebar_cover=args.rebar_cover,
        overhead=args.overhead,
        output_dir=args.output_dir,
    )

    newton.examples.run(example, args)
