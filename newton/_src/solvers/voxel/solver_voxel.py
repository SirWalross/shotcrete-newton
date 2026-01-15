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

import numpy as np
import warp as wp

LINEAR_SPACING = wp.float32(0.002)  # m

import re

import newton

from ...core.types import override
from ...sim import Contacts, Control, Model, State, VoxelRewards
from ..solver import SolverBase
from .kernels import (
    SPRAY_COUNT,
    capacity_propagation_kernel,
    drip_kernel,
    drop_down_kernel,
    expand_global_bbox_kernel,
    failure_spread_kernel,
    initialize_load_kernel,
    out_of_bounds_spray_kernel,
    randomize_directions_kernel,
    reset_bbox_kernel,
    reset_global_bbox_kernel,
    respreading_kernel,
    set_floor_kernel,
    set_wall_kernel,
    solidify_kernel,
    spray_backtrack_kernel,
    spray_distribution_kernel,
    spray_neighbours_kernel,
    spray_overlap_kernel,
    spray_rebound_kernel,
    spray_redistribution_kernel,
    spray_reward_kernel,
    spray_trajectory_kernel,
    sum_kernel,
    update_bbox_kernel,
    update_body_positions_kernel,
    update_cond_kernel,
    update_directions_kernel,
    update_distances_kernel,
    update_robot_position_kernel,
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
        tcp_body_name: str,
        s: int = 9,
        backtrack_count: int = 10,
        h: float = 0.005,
        tc: int = 5,
        k: int = 300,
        droplet_mass: float = 1.0 / 6.0,
        spray_velocity: float = 20.0,
        sigma: float = 1.0,
        drip_vel: int = 5,
        drip_amount: float = 1.0,
        respreading_backtracking_amount: int = 25,
        rebound_opening_angle: float = 0.7,
        nozzle_opening_angle: float = 0.157,
        overlap_distance: float = 50.0,
        anisotropic_distance_weight: float = 2.8,
        shear_strength: int = 50,
        adhesion_strength: int = 20,
        compression_strength: int = 1285,
        wet_strength_penalty: int = 5,
        debug_mode: bool = False,
    ):
        super().__init__(model=model)

        self.active = debug_mode
        self.synchronize = debug_mode

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
        self.positions = wp.zeros((self.shape[0], self.k), dtype=wp.vec3i)
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
            [i for i, key in enumerate(self.model.body_key) if re.match(f"/World/envs/env_.*/{tcp_body_name}", key)],
            dtype=int,
        )
        assert self.ee_body_indices.shape[0] == self.shape[0], "Number of end-effectors does not match number of envs"

        self.i = wp.zeros(1, dtype=int)
        self.drip_cond = wp.zeros(1, dtype=int)
        self.adhesion_cond = wp.zeros(1, dtype=int)
        self.ray_indices = wp.zeros((self.shape[0], self.k), dtype=wp.int32)
        self.rebound_droplet_mass = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
        self.rebound_directions = wp.zeros((self.shape[0], self.k), dtype=wp.vec3f)
        self.avg_ray_index = wp.zeros((self.shape[0],), dtype=wp.int32)
        self.spray_overlap = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
        self.spray_neighbours = wp.zeros((self.shape[0], self.k, self.ball_indices.shape[0]), dtype=wp.float32)
        self.density = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
        self.neighbour_count = wp.zeros((self.shape[0], self.k), dtype=wp.float32)
        self.spray_bbox = wp.zeros((self.shape[0], 12), dtype=wp.int32)
        self.global_bbox = wp.array(
            np.tile(np.array([[100000, 100000, 100000, 0, 0, 0]]), (self.shape[0], 1)), dtype=wp.int32
        )

    @override
    def step(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, rewards: VoxelRewards, dt: float
    ):
        with wp.ScopedTimer("step", active=self.active, synchronize=self.synchronize):
            wp.launch(
                update_cond_kernel, dim=1, inputs=[self.i, self.drip_vel], outputs=[self.drip_cond, self.adhesion_cond]
            )
            wp.launch(reset_bbox_kernel, dim=(self.shape[0],), outputs=[self.spray_bbox])
            with wp.ScopedTimer("spraying", active=self.active, synchronize=self.synchronize):
                self.deposit(wp.clone(state_in.body_q[self.ee_body_indices]), self.model.voxel_pos)
            with wp.ScopedTimer("update global bbox", active=self.active, synchronize=self.synchronize):
                wp.launch(expand_global_bbox_kernel, dim=(self.shape[0],), inputs=[self.global_bbox, self.spray_bbox])
            with wp.ScopedTimer("adhesion check", active=self.active, synchronize=self.synchronize):
                wp.capture_if(self.adhesion_cond, on_true=lambda: self.adhesion_check(rewards))
            with wp.ScopedTimer("solidify", active=self.active, synchronize=self.synchronize):
                wp.launch(
                    solidify_kernel,
                    dim=self.shape,
                    inputs=[self.model.voxel_wet, self.model.voxel_dry, self.tc, self.global_bbox],
                )
            with wp.ScopedTimer("drip", active=self.active, synchronize=self.synchronize):
                wp.capture_if(self.drip_cond, on_true=self.drip)
            self.update_rewards(rewards)
            wp.launch(
                update_body_positions_kernel,
                dim=state_in.body_q.shape,
                inputs=[state_in.body_q],
                outputs=[state_out.body_q],
            )
            # wp.launch(
            #     update_robot_position_kernel,
            #     dim=state_in.joint_q.shape,
            #     inputs=[
            #         state_in.joint_q,
            #         state_in.joint_qd,
            #         state_in.joint_q.shape[0] // self.model.num_worlds,
            #         self.model.joint_velocity_limit,
            #         control.joint_target_pos,
            #         control.joint_target_vel,
            #         dt,
            #     ],
            #     outputs=[
            #         state_out.joint_q,
            #         state_out.joint_qd,
            #     ],
            # )
            # newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)
        return state_out

    @override
    def reset(self, state_out: State, world_indices: wp.array(dtype=int)):
        self.model.voxel_wet[world_indices].fill_(0)
        self.model.voxel_dry[world_indices].fill_(0)
        self.model.voxel_distance[world_indices].fill_(255)
        self.model.voxel_load[world_indices].fill_(0)
        wp.launch(
            set_floor_kernel,
            dim=(world_indices.shape[0], self.shape[1], self.shape[2]),
            inputs=[
                self.model.voxel_wet,
                self.model.voxel_dry,
                self.model.voxel_distance,
                world_indices,
            ],
        )
        wp.launch(
            set_wall_kernel,
            dim=(world_indices.shape[0], self.shape[1], self.shape[3]),
            inputs=[
                self.model.voxel_wet,
                self.model.voxel_dry,
                self.model.voxel_distance,
                world_indices,
            ],
        )
        wp.launch(reset_global_bbox_kernel, dim=(world_indices.shape[0],), inputs=[self.global_bbox, world_indices])

    def update_rewards(self, rewards: VoxelRewards):
        with wp.ScopedTimer("rewards", active=self.active, synchronize=self.synchronize):
            with wp.ScopedTimer("spray reward calculation", active=self.active, synchronize=self.synchronize):
                wp.launch(
                    spray_reward_kernel,
                    dim=(self.shape[0], self.shape[1] - 2, self.shape[3] - 2),
                    inputs=[self.model.voxel_wet, self.model.voxel_dry, self.h, rewards.decimation, self.global_bbox],
                    outputs=[rewards.distance, rewards.smoothness, rewards.air_gap],
                )
            with wp.ScopedTimer("out of bounds spray calculation", active=self.active, synchronize=self.synchronize):
                wp.launch(
                    out_of_bounds_spray_kernel,
                    dim=(self.shape[0], self.k),
                    inputs=[self.model.voxel_wet, self.ray_trajectory],
                    outputs=[rewards.out_of_bounds_spray],
                )

    def drip(self):
        wp.launch(
            drip_kernel,
            dim=(self.shape[2] - 2, self.shape[1] - 2, self.shape[0]),
            inputs=[self.model.voxel_wet, self.model.voxel_dry, self.model.voxel_distance, self.shape[3] - 2],
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
            for _ in range(2):
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(self.shape[0], self.shape[1] - 2, self.shape[2] - 2, 1),
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.model.voxel_load,
                        self.model.voxel_distance,
                        self.spray_bbox,
                        0,
                        self.shape[3] - 3,
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
                        self.spray_bbox,
                        -self.shape[2] + 3,
                        self.shape[2] - 3,
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
                        self.spray_bbox,
                        -self.shape[3] + 3,
                        self.shape[3] - 3,
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
                        self.spray_bbox,
                        0,
                        self.shape[1] - 3,
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
                        self.spray_bbox,
                        -self.shape[1] + 3,
                        self.shape[1] - 3,
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
                        self.spray_bbox,
                        0,
                        self.shape[2] - 3,
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
                dim=(self.ball_indices.shape[0], self.k, self.shape[0]),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.model.voxel_load,
                    self.ball_indices,
                    self.ray_trajectory[:, :, 0],
                ],
            )
        with wp.ScopedTimer("drop down", active=self.active, synchronize=self.synchronize):
            wp.launch(
                drop_down_kernel,
                dim=(self.shape[0], self.shape[1], self.shape[2]),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.model.voxel_distance,
                    self.model.voxel_load,
                    self.spray_bbox,
                ],
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

    def deposit(self, ee_transforms: wp.array(dtype=wp.vec3f), voxel_pos: wp.array(dtype=wp.vec3f)):
        with wp.ScopedTimer("alloca", active=self.active, synchronize=self.synchronize):
            wp.launch(
                update_directions_kernel,
                dim=(self.shape[0], self.k),
                inputs=[
                    self.nozzle_opening_angle,
                    ee_transforms,
                    voxel_pos,
                    self.total_droplet_mass,
                    self.i,
                    self.k,
                    self.h,
                    self.shape[1],
                ],
                outputs=[self.positions, self.directions, self.droplet_mass],
            )
        with wp.ScopedTimer("spray trajectory", active=self.active, synchronize=self.synchronize):
            self.ray_indices.zero_()
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
                    self.h,
                ],
                outputs=[self.ray_indices],
            )
            self.avg_ray_index.zero_()
            wp.launch(kernel=sum_kernel, dim=(self.shape[0], self.k), inputs=[self.ray_indices, self.avg_ray_index])
            wp.launch(
                respreading_kernel,
                dim=(self.shape[0], self.k, self.respreading_backtracking_amount),
                inputs=[
                    self.model.voxel_wet,
                    self.sigma,
                    self.ray_indices,
                    self.positions,
                    self.directions,
                    self.speed_distribution,
                    LINEAR_SPACING,
                    self.avg_ray_index,
                    self.droplet_mass,
                    self.h,
                    self.k,
                ],
                outputs=[],
            )
            wp.launch(
                spray_backtrack_kernel,
                dim=(self.shape[0], self.k, self.backtrack_count),
                inputs=[
                    self.positions,
                    self.directions,
                    self.speed_distribution,
                    LINEAR_SPACING,
                    self.ray_indices,
                    self.h,
                ],
                outputs=[self.ray_trajectory],
            )
        with wp.ScopedTimer("spray rebound", active=self.active, synchronize=self.synchronize):
            wp.launch(
                spray_rebound_kernel,
                dim=(self.shape[0], self.k),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.positions,
                    self.ray_trajectory[:, :, 0],
                    self.directions,
                    self.ray_indices,
                    self.speed_distribution,
                    self.droplet_mass,
                    LINEAR_SPACING,
                    self.h,
                ],
                outputs=[self.rebound_droplet_mass, self.rebound_directions],
            )
            wp.launch(
                randomize_directions_kernel,
                dim=(self.shape[0], self.k),
                inputs=[self.rebound_directions, self.rebound_opening_angle, self.i],
            )
            self.ray_indices.zero_()
            wp.launch(
                spray_trajectory_kernel,
                dim=(self.shape[0], self.k, SPRAY_COUNT),
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.ray_trajectory[:, :, 0],
                    self.rebound_directions,
                    self.rebound_speed_distribution,
                    LINEAR_SPACING / 5.0,
                    self.h,
                ],
                outputs=[self.ray_indices],
            )
            wp.launch(
                spray_backtrack_kernel,
                dim=(self.shape[0], self.k, 1),
                inputs=[
                    self.ray_trajectory[:, :, 0],
                    self.rebound_directions,
                    self.rebound_speed_distribution,
                    LINEAR_SPACING / 5.0,
                    self.ray_indices,
                    self.h,
                ],
                outputs=[self.ray_rebound_trajectory],
            )
        with wp.ScopedTimer("spray redistribution", active=self.active, synchronize=self.synchronize):
            with wp.ScopedTimer("spray overlap", active=self.active, synchronize=self.synchronize):
                wp.launch(
                    spray_overlap_kernel,
                    dim=(self.shape[0], self.k),
                    inputs=[self.ray_trajectory[:, :, 0], self.overlap_distance, self.k],
                    outputs=[self.spray_overlap],
                )
            with wp.ScopedTimer("spray redistribution loop", active=self.active, synchronize=self.synchronize):
                for _ in range(5):
                    wp.launch(
                        spray_redistribution_kernel,
                        dim=(self.shape[0], self.k),
                        inputs=[
                            self.ray_trajectory[:, :, 0],
                            self.spray_overlap,
                            self.droplet_mass,
                            ee_transforms,
                            self.overlap_distance,
                            self.anisotropic_distance_weight,
                            self.k,
                        ],
                    )
        with wp.ScopedTimer("spray deposit", active=self.active, synchronize=self.synchronize):
            for k in range(self.backtrack_count):
                self.density.zero_()
                self.neighbour_count.zero_()
                with wp.ScopedTimer("spray neighbours", active=self.active, synchronize=self.synchronize):
                    wp.launch(
                        spray_neighbours_kernel,
                        dim=self.spray_neighbours.shape,
                        inputs=[
                            self.model.voxel_wet,
                            self.model.voxel_dry,
                            self.ball_indices,
                            self.ray_trajectory[:, :, k],
                        ],
                        outputs=[self.spray_neighbours, self.density, self.neighbour_count],
                    )
                with wp.ScopedTimer("spray distribution", active=self.active, synchronize=self.synchronize):
                    for _ in range(5):
                        wp.launch(
                            spray_distribution_kernel,
                            dim=[self.spray_neighbours.shape[2], self.spray_neighbours.shape[1], self.spray_neighbours.shape[0]],
                            inputs=[
                                self.model.voxel_wet,
                                self.model.voxel_dry,
                                self.ball_indices,
                                self.ray_trajectory[:, :, k],
                                self.spray_neighbours,
                                self.droplet_mass,
                                self.neighbour_count,
                            ],
                        )
            self.update_distances()
        with wp.ScopedTimer("spray backtrack deposit", active=self.active, synchronize=self.synchronize):
            self.density.zero_()
            self.neighbour_count.zero_()
            wp.launch(
                spray_neighbours_kernel,
                dim=self.spray_neighbours.shape,
                inputs=[
                    self.model.voxel_wet,
                    self.model.voxel_dry,
                    self.ball_indices,
                    self.ray_rebound_trajectory[:, :, 0],
                ],
                outputs=[self.spray_neighbours, self.density, self.neighbour_count],
            )
            for _ in range(5):
                wp.launch(
                    spray_distribution_kernel,
                    dim=self.spray_neighbours.shape,
                    inputs=[
                        self.model.voxel_wet,
                        self.model.voxel_dry,
                        self.ball_indices,
                        self.ray_rebound_trajectory[:, :, 0],
                        self.spray_neighbours,
                        self.rebound_droplet_mass,
                        self.neighbour_count,
                    ],
                )
            self.update_rebound_distances()
        with wp.ScopedTimer("update bbox", active=self.active, synchronize=self.synchronize):
            wp.launch(
                update_bbox_kernel,
                dim=(self.shape[0], self.k, self.backtrack_count),
                inputs=[
                    self.ray_trajectory,
                    self.ray_rebound_trajectory,
                ],
                outputs=[self.spray_bbox],
            )
