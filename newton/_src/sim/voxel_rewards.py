from __future__ import annotations

import warp as wp
from warp.context import Devicelike


class VoxelRewards:
    def __init__(self, size, decimation: int, device: Devicelike = None):
        self.size = size
        self.decimation = decimation
        with wp.ScopedDevice(device):
            # rigid contacts
            self.distance = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.prev_distance = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.smoothness = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.air_gap = wp.zeros((size[0], size[1] // decimation, size[3] // decimation), dtype=wp.float32)
            self.adhesion_failure_amount = wp.zeros((size[0],), dtype=wp.float32)
            self.out_of_bounds_spray = wp.zeros((size[0],), dtype=wp.float32)

    def step(self):
        self.prev_distance = wp.clone(self.distance)
        self.distance.zero_()
        self.smoothness.zero_()
        self.air_gap.zero_()
        self.adhesion_failure_amount.zero_()
        self.out_of_bounds_spray.zero_()

    def reset(self, world_indices: wp.array(dtype=int)):
        self.distance[world_indices].zero_()
        self.prev_distance[world_indices].zero_()
        self.smoothness[world_indices].zero_()
        self.air_gap[world_indices].zero_()
        self.adhesion_failure_amount[world_indices].zero_()
        self.out_of_bounds_spray[world_indices].zero_()

    @property
    def device(self):
        """
        Returns the device on which the buffers are allocated.
        """
        return self.distance.device
