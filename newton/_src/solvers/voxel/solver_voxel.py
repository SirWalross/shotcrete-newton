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

import sys

import numpy as np
import warp as wp

LINEAR_SPACING = wp.float32(0.002)  # m

import re

from ...core.types import override
from ...sim import Contacts, Control, Model, State, VoxelRewards
from ..mujoco import SolverMuJoCo
from ..solver import SolverBase
from .kernels import (
    SPRAY_COUNT,
    capacity_propagation_kernel,
    drip_kernel,
    drop_down_kernel,
    failure_spread_kernel,
    initialize_load_kernel,
    randomize_directions_kernel,
    respreading_kernel,
    solidify_kernel,
    spray_backtrack_kernel,
    spray_distribution_kernel,
    spray_neighbours_kernel,
    spray_overlap_kernel,
    spray_rebound_kernel,
    spray_redistribution_kernel,
    spray_reward,
    spray_trajectory_kernel,
    sum_kernel,
    update_directions_kernel,
    update_distances_kernel,
)


def get_sphere_indices(radius: int):
    """
    Generates (N, 3) array of integer offsets for voxels inside a sphere.
    """
    # 1. Define the bounding box range (-R to +R)
    r_int = int(np.ceil(radius))
    rng = np.arange(-r_int, r_int + 1)

    # 2. Create a 3D grid of coordinates
    # indexing='ij' ensures matrix indexing (x, y, z order)
    x, y, z = np.meshgrid(rng, rng, rng, indexing="ij")

    # 3. Calculate squared distance from center (0,0,0)
    dist_sq = x**2 + y**2 + z**2

    # 4. Create a boolean mask for valid voxels
    mask = dist_sq <= radius**2

    # 5. Extract the coordinates using the mask and stack them
    # Result is shape (N, 3) where N is number of valid voxels
    indices = np.vstack((x[mask], y[mask], z[mask])).T

    return indices


class SolverVoxel(SolverBase):
    def __init__(
        self,
        model: Model,
        *,
        s: int = 9,
        backtrack_count: int = 10,
        h: float = 0.005,
        tc: float = 50.0,
        k: int = 300,
        droplet_mass: float = 1.0 / 2.0,
        spray_velocity: float = 20.0,
        sigma: float = 1.0,
        drip_vel: int = 5,
        drip_amount: float = 1.0,
        respreading_backtracking_amount: int = 25,
        rebound_opening_angle: float = 0.7,
        nozzle_opening_angle: float = 0.157,
        overlap_distance: int = 50,
        anisotropic_distance_weight: float = 2.8,
        shear_strength: float = 50.0,
        adhesion_strength: float = 20.0,
        compression_strength: float = 2000.0,
        wet_strength_penalty: float = 0.2,
        mujoco_config,
    ):
        super().__init__(model=model)
        mujoco_config.pop("solver_type")

        self.active = False
        self.synchronize = False

        self.h = h
        self.tc = tc
        self.k = k
        self.backtrack_count = backtrack_count
        self.shape = self.model.voxel_wet.shape
        self.total_droplet_mass = droplet_mass
        self.sigma = sigma
        self.drip_vel = drip_vel
        self.respreading_backtracking_amount = respreading_backtracking_amount
        self.rebound_opening_angle = rebound_opening_angle
        self.nozzle_opening_angle = nozzle_opening_angle
        self.overlap_distance = overlap_distance
        self.anisotropic_distance_weight = anisotropic_distance_weight
        self.shear_strength = shear_strength
        self.adhesion_strength = adhesion_strength
        self.compression_strength = compression_strength
        self.wet_strength_penalty = wet_strength_penalty

        self.ball_indices = wp.array(get_sphere_indices(s // 2), dtype=wp.vec3i)
        self.positions = wp.zeros((self.shape[0], self.k), dtype=wp.vec3)
        self.directions = wp.zeros((self.shape[0], self.k), dtype=wp.vec3)
        self.droplet_mass = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
        self.ray_trajectory = wp.zeros((self.shape[0], self.k, self.backtrack_count), dtype=wp.vec3i)
        self.ray_rebound_trajectory = wp.zeros((self.shape[0], self.k, 1), dtype=wp.vec3i)

        # speed distributions
        self.speed_distribution = wp.array(
            np.abs(np.random.normal(spray_velocity, spray_velocity / 20.0, self.k)), dtype=wp.float32
        )
        self.rebound_speed_distribution = wp.array(
            np.abs(np.random.normal(spray_velocity / 50.0, spray_velocity / 2000.0, self.k)), dtype=wp.float32
        )

        # find indices for the end-effector bodies in the different envs
        self.ee_body_indices = wp.array(
            [i for i, key in enumerate(self.model.body_key) if re.match("/World/envs/env_.*/Robot/ee_link", key)],
            dtype=int,
        )
        assert self.ee_body_indices.shape[0] == self.shape[0], "Number of end-effectors does not match number of envs"

        self.mujoco = SolverMuJoCo(model, **mujoco_config)
        self.i = 0

    @override
    def step(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, rewards: VoxelRewards, dt: float
    ):
        with wp.ScopedTimer("spraying", active=self.active, synchronize=self.synchronize):
            self.deposit(wp.clone(state_in.body_q[self.ee_body_indices]))
        if self.i % 10 == 0:
            with wp.ScopedTimer("adhesion check", active=self.active, synchronize=self.synchronize):
                self.adhesion_check(rewards)
        with wp.ScopedTimer("solidify", active=self.active, synchronize=self.synchronize):
            wp.launch(solidify_kernel, dim=self.shape, inputs=[self.model.voxel_wet, self.model.voxel_dry, self.tc])
        if self.i % self.drip_vel == 0:
            with wp.ScopedTimer("drip", active=self.active, synchronize=self.synchronize):
                for z in range(self.shape[3] - 2):
                    wp.launch(
                        drip_kernel,
                        dim=(self.shape[0], self.shape[1] - 2, self.shape[2] - 2),
                        inputs=[self.model.voxel_wet, self.model.voxel_dry, self.model.voxel_distance, z],
                    )
        self.update_rewards(rewards)
        self.i += 1

        return self.mujoco.step(state_in, state_out, control, contacts, dt)

    def update_rewards(self, rewards: VoxelRewards):
        with wp.ScopedTimer("rewards", active=self.active, synchronize=self.synchronize):
            wp.launch(
                spray_reward,
                dim=(self.shape[0], self.shape[1], self.shape[3]),
                inputs=[self.model.voxel_wet, self.model.voxel_dry],
                outputs=[rewards.distance, rewards.smoothness, rewards.air_gap],
            )

    def adhesion_check(self, rewards: VoxelRewards):
        self.model.voxel_load.zero_()
        with wp.ScopedTimer("initialize load", active=self.active, synchronize=self.synchronize):
            wp.launch(
                initialize_load_kernel,
                dim=self.shape,
                inputs=[self.model.voxel_wet, self.model.voxel_dry, self.model.voxel_load],
            )
        with wp.ScopedTimer("capacity propagation", active=self.active, synchronize=self.synchronize):
            for _ in range(4):
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(self.shape[0], self.shape[1] - 2, self.shape[2] - 2, 1),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_load,
                        self.model.voxel_distance,
                        0,
                        self.shape[3] - 2,
                        wp.vec3i(0, 0, 1),
                        self.wet_strength_penalty,
                        self.compression_strength,
                        self.shear_strength,
                        self.adhesion_strength,
                    ],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(self.shape[0], self.shape[1] - 2, 1, self.shape[3] - 2),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_load,
                        self.model.voxel_distance,
                        -self.shape[2] + 2,
                        self.shape[2] - 2,
                        wp.vec3i(0, -1, 0),
                        self.wet_strength_penalty,
                        self.compression_strength,
                        self.shear_strength,
                        self.adhesion_strength,
                    ],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(self.shape[0], self.shape[1] - 2, self.shape[2] - 2, 1),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_load,
                        self.model.voxel_distance,
                        -self.shape[3] + 2,
                        self.shape[3] - 2,
                        wp.vec3i(0, 0, -1),
                        self.wet_strength_penalty,
                        self.compression_strength,
                        self.shear_strength,
                        self.adhesion_strength,
                    ],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(self.shape[0], 1, self.shape[2] - 2, self.shape[3] - 2),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_load,
                        self.model.voxel_distance,
                        0,
                        self.shape[1] - 2,
                        wp.vec3i(1, 0, 0),
                        self.wet_strength_penalty,
                        self.compression_strength,
                        self.shear_strength,
                        self.adhesion_strength,
                    ],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(self.shape[0], 1, self.shape[2] - 2, self.shape[3] - 2),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_load,
                        self.model.voxel_distance,
                        -self.shape[1] + 2,
                        self.shape[1] - 2,
                        wp.vec3i(-1, 0, 0),
                        self.wet_strength_penalty,
                        self.compression_strength,
                        self.shear_strength,
                        self.adhesion_strength,
                    ],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(self.shape[0], self.shape[1] - 2, 1, self.shape[3] - 2),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_load,
                        self.model.voxel_distance,
                        0,
                        self.shape[2] - 2,
                        wp.vec3i(0, 1, 0),
                        self.wet_strength_penalty,
                        self.compression_strength,
                        self.shear_strength,
                        self.adhesion_strength,
                    ],
                )
        with wp.ScopedTimer("failure spread", active=self.active, synchronize=self.synchronize):
            wp.launch(
                failure_spread_kernel,
                dim=(self.shape[0], self.k, self.ball_indices.shape[0]),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.model.voxel_load,
                    self.ball_indices,
                    self.ray_trajectory[:, :, 0],
                ],
            )
        with wp.ScopedTimer("drop down", active=self.active, synchronize=self.synchronize):
            rewards.adhesion_failure_amount.zero_()
            wp.launch(
                drop_down_kernel,
                dim=(self.shape[0], self.shape[1], self.shape[2]),
                inputs=[self.model.voxel_wet, self.model.voxel_dry, self.model.voxel_distance, self.model.voxel_load],
                outputs=[rewards.adhesion_failure_amount],
            )

    def update_rebound_distances(self):
        with wp.ScopedTimer("update distances", active=self.active, synchronize=self.synchronize):
            for _ in range(5):
                wp.launch(
                    update_distances_kernel,
                    dim=(self.shape[0], self.k, self.ball_indices.shape[0]),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_distance,
                        self.ball_indices,
                        self.ray_rebound_trajectory[:, :, 0],
                    ],
                )

    def update_distances(self):
        with wp.ScopedTimer("update distances", active=self.active, synchronize=self.synchronize):
            for _ in range(5):
                wp.launch(
                    update_distances_kernel,
                    dim=(self.shape[0], self.k, self.ball_indices.shape[0]),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_distance,
                        self.ball_indices,
                        self.ray_trajectory[:, :, 0],
                    ],
                )

    def deposit(self, transforms: wp.array(dtype=wp.vec3f)):
        with wp.ScopedTimer("alloca", active=self.active, synchronize=self.synchronize):
            wp.launch(
                update_directions_kernel,
                dim=(self.shape[0], self.k),
                inputs=[self.nozzle_opening_angle, transforms, self.total_droplet_mass],
                outputs=[self.positions, self.directions, self.droplet_mass],
            )
        ray_indices = wp.zeros((self.shape[0], self.k), dtype=wp.int32)
        with wp.ScopedTimer("spray trajectory", active=self.active, synchronize=self.synchronize):
            wp.launch(
                spray_trajectory_kernel,
                dim=(self.shape[0], self.k, SPRAY_COUNT),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.positions,
                    self.directions,
                    self.speed_distribution,
                    LINEAR_SPACING,
                ],
                outputs=[ray_indices],
            )
            avg_ray_index = wp.zeros((self.shape[0],), dtype=wp.int32)
            wp.launch(kernel=sum_kernel, dim=(self.shape[0], self.k), inputs=[ray_indices, avg_ray_index])
            wp.launch(
                respreading_kernel,
                dim=(self.shape[0], self.k, self.respreading_backtracking_amount),
                inputs=[
                    self.model.voxel_wet,
                    self.sigma,
                    ray_indices,
                    self.positions,
                    self.directions,
                    self.speed_distribution,
                    LINEAR_SPACING,
                    avg_ray_index,
                    self.droplet_mass,
                ],
                outputs=[],
            )
            wp.launch(
                spray_backtrack_kernel,
                dim=(self.shape[0], self.k, self.backtrack_count),
                inputs=[self.positions, self.directions, self.speed_distribution, LINEAR_SPACING, ray_indices],
                outputs=[self.ray_trajectory],
            )
        with wp.ScopedTimer("spray rebound", active=self.active, synchronize=self.synchronize):
            rebound_droplet_mass = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
            ray_indices = wp.zeros((self.shape[0], self.k), dtype=wp.int32)
            rebound_directions = wp.zeros((self.shape[0], self.k), dtype=wp.vec3f)
            wp.launch(
                spray_rebound_kernel,
                dim=(self.shape[0], self.k),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.positions,
                    self.ray_trajectory[:, :, 0],
                    self.directions,
                    ray_indices,
                    self.speed_distribution,
                    self.total_droplet_mass,
                    LINEAR_SPACING,
                ],
                outputs=[rebound_droplet_mass, rebound_directions],
            )
            wp.launch(
                randomize_directions_kernel,
                dim=(self.shape[0], self.k),
                inputs=[rebound_directions, self.rebound_opening_angle, self.i],
            )
            wp.launch(
                spray_trajectory_kernel,
                dim=(self.shape[0], self.k, SPRAY_COUNT),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.ray_trajectory[:, :, 0],
                    rebound_directions,
                    self.rebound_speed_distribution,
                    LINEAR_SPACING / 5.0,
                ],
                outputs=[ray_indices],
            )
            wp.launch(
                spray_backtrack_kernel,
                dim=(self.shape[0], self.k, 1),
                inputs=[
                    self.ray_trajectory[:, :, 0],
                    rebound_directions,
                    self.rebound_speed_distribution,
                    LINEAR_SPACING / 5.0,
                    ray_indices,
                ],
                outputs=[self.ray_rebound_trajectory],
            )
        with wp.ScopedTimer("spray redistribution", active=self.active, synchronize=self.synchronize):
            spray_overlap = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
            with wp.ScopedTimer("spray overlap", active=self.active, synchronize=self.synchronize):
                wp.launch(
                    spray_overlap_kernel,
                    dim=(self.shape[0], self.k, self.k - 1),
                    inputs=[self.ray_trajectory[:, :, 0]],
                    outputs=[spray_overlap],
                )
            with wp.ScopedTimer("spray redistribution loop", active=self.active, synchronize=self.synchronize):
                for _ in range(10):
                    wp.launch(
                        spray_redistribution_kernel,
                        dim=(self.shape[0], self.k, self.k),
                        inputs=[self.ray_trajectory[:, :, 0], spray_overlap, self.droplet_mass, self.positions],
                    )
        with wp.ScopedTimer("spray deposit", active=self.active, synchronize=self.synchronize):
            for k in range(self.backtrack_count):
                spray_neighbours = wp.zeros((self.shape[0], self.k, self.ball_indices.shape[0]), dtype=wp.float32)
                density = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
                neighbour_count = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
                with wp.ScopedTimer("spray neighbours", active=self.active, synchronize=self.synchronize):
                    wp.launch(
                        spray_neighbours_kernel,
                        dim=spray_neighbours.shape,
                        inputs=[
                            self.model.voxel_wet,
                            self.model.voxel_dry,
                            self.ball_indices,
                            self.ray_trajectory[:, :, k],
                        ],
                        outputs=[spray_neighbours, density, neighbour_count],
                    )
                with wp.ScopedTimer("spray distribution", active=self.active, synchronize=self.synchronize):
                    for _ in range(20):
                        wp.launch(
                            spray_distribution_kernel,
                            dim=spray_neighbours.shape,
                            inputs=[
                                self.model.voxel_wet,
                                self.model.voxel_dry,
                                self.ball_indices,
                                self.ray_trajectory[:, :, k],
                                spray_neighbours,
                                self.droplet_mass,
                                neighbour_count,
                            ],
                        )
            self.update_distances()
        with wp.ScopedTimer("spray backtrack deposit", active=self.active, synchronize=self.synchronize):
            spray_neighbours = wp.zeros((self.shape[0], self.k, self.ball_indices.shape[0]), dtype=wp.float32)
            density = wp.zeros(
                (self.shape[0], self.k),
                dtype=wp.float32,
            )
            neighbour_count = wp.zeros(
                (self.shape[0], self.k),
                dtype=wp.float32,
            )
            wp.launch(
                spray_neighbours_kernel,
                dim=spray_neighbours.shape,
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.ball_indices,
                    self.ray_rebound_trajectory[:, :, 0],
                ],
                outputs=[spray_neighbours, density, neighbour_count],
            )
            for _ in range(20):
                wp.launch(
                    spray_distribution_kernel,
                    dim=spray_neighbours.shape,
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.ball_indices,
                        self.ray_rebound_trajectory[:, :, 0],
                        spray_neighbours,
                        rebound_droplet_mass,
                        neighbour_count,
                    ],
                )
            self.update_rebound_distances()
