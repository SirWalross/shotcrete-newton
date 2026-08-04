from __future__ import annotations

import warp as wp
from warp.context import Devicelike


class VoxelRewards:
    def __init__(self, size, decimation: int, device: Devicelike = None, render_decimation: int | None = None):
        self.size = size
        self.decimation = decimation
        # Decimation used solely for the rendered height-map. Defaults to ``decimation`` so that,
        # unless an env explicitly requests a finer render grid, the renderer shares the (decimated)
        # rewards grid and no extra memory/compute is used. When it differs from ``decimation`` a
        # separate ``render_distance`` grid is allocated and populated by the solver; all reward,
        # termination and observation maths keep operating on the unchanged ``decimation`` grids.
        self.render_decimation = decimation if render_decimation is None else render_decimation
        with wp.ScopedDevice(device):
            # rigid contacts
            self.distance = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.prev_distance = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            # Depth map as seen by an occluded lidar: cells within ``occlusion_distance`` of the
            # nozzle keep their last-seen value (the spray blocks the sensor there). Maintained
            # entirely by the solver; ``prev_distance_occluded`` carries the last-seen values forward.
            self.distance_occluded = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.prev_distance_occluded = wp.zeros(
                (size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32
            )
            self.distance_without_rebar = wp.zeros(
                (size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32
            )
            self.distance_without_air_gap = wp.zeros(
                (size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32
            )
            self.prev_distance_without_air_gap = wp.zeros(
                (size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32
            )
            self.smoothness = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.air_gap = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.prev_air_gap = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.adhesion_failure_amount = wp.zeros((size[0],), dtype=wp.float32)
            self.out_of_bounds_spray = wp.zeros((size[0],), dtype=wp.float32)
            self.tcp_position = wp.zeros((size[0],), dtype=wp.vec3i)

            # render-only height map; aliases `distance` when no separate render grid is requested.
            if self.render_decimation == self.decimation:
                self.render_distance = self.distance
            else:
                self.render_distance = wp.zeros(
                    (size[0], size[1] // self.render_decimation, size[3] // self.render_decimation), dtype=wp.float32
                )

    def step(self):
        # In-place copies, NOT `wp.clone` + rebind: the captured CUDA graph holds the original
        # buffer pointers (e.g. `prev_distance_occluded` is read by spray_reward_kernel inside the
        # graph). Rebinding to a fresh allocation is a use-after-free for the graph — the recycled
        # memory ends up aliasing other per-step clones, which silently disabled the occlusion
        # carry-forward (the occluded map read last step's clean depth instead of last-seen values).
        wp.copy(self.prev_distance, self.distance)
        wp.copy(self.prev_distance_without_air_gap, self.distance_without_air_gap)
        wp.copy(self.prev_air_gap, self.air_gap)
        wp.copy(self.prev_distance_occluded, self.distance_occluded)

        self.distance.zero_()
        self.distance_occluded.zero_()
        self.distance_without_rebar.zero_()
        self.distance_without_air_gap.zero_()
        self.smoothness.zero_()
        self.air_gap.zero_()
        self.adhesion_failure_amount.zero_()
        self.out_of_bounds_spray.zero_()
        self.tcp_position.zero_()
        if self.render_distance is not self.distance:
            self.render_distance.zero_()

    def reset(self, world_indices: wp.array(dtype=int)):
        self.distance[world_indices].zero_()
        self.prev_distance[world_indices].zero_()
        self.distance_occluded[world_indices].zero_()
        self.prev_distance_occluded[world_indices].zero_()
        self.distance_without_rebar[world_indices].zero_()
        self.distance_without_air_gap[world_indices].zero_()
        self.prev_distance_without_air_gap[world_indices].zero_()
        self.smoothness[world_indices].zero_()
        self.air_gap[world_indices].zero_()
        self.prev_air_gap[world_indices].zero_()
        self.adhesion_failure_amount[world_indices].zero_()
        self.out_of_bounds_spray[world_indices].zero_()
        self.tcp_position[world_indices].zero_()
        if self.render_distance is not self.distance:
            self.render_distance[world_indices].zero_()

    @property
    def device(self):
        """
        Returns the device on which the buffers are allocated.
        """
        return self.distance.device
