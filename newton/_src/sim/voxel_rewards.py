from __future__ import annotations

import warp as wp
from warp.context import Devicelike


class VoxelRewards:
    def __init__(self, size, device: Devicelike = None):
        with wp.ScopedDevice(device):
            # rigid contacts
            self.distance = wp.zeros((size[0], size[1] // 16, size[2] // 16), dtype=wp.float32)
            self.smoothness = wp.zeros((size[0], size[1] // 16, size[2] // 16), dtype=wp.float32)
            self.air_gap = wp.zeros((size[0], size[1] // 16, size[2] // 16), dtype=wp.float32)
            self.prev_distance = wp.zeros((size[0], size[1] // 16, size[2] // 16), dtype=wp.float32)
            self.prev_smoothness = wp.zeros((size[0], size[1] // 16, size[2] // 16), dtype=wp.float32)
            self.prev_air_gap = wp.zeros((size[0], size[1] // 16, size[2] // 16), dtype=wp.float32)
            self.adhesion_failure_amount = wp.zeros((size[0],), dtype=wp.float32)

    def step(self):
        self.prev_distance, self.distance = self.distance, self.prev_distance
        self.distance.zero_()
        self.prev_smoothness, self.smoothness = self.smoothness, self.prev_smoothness
        self.smoothness.zero_()
        self.prev_air_gap, self.air_gap = self.air_gap, self.prev_air_gap
        self.air_gap.zero_()
        self.adhesion_failure_amount.zero_()

    @property
    def device(self):
        """
        Returns the device on which the buffers are allocated.
        """
        return self.distance.device
