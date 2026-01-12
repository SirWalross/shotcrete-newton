import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
from constants import (
    ADHESION_CHECK,
    BACKTRACK_COUNT,
    BALL_COUNT,
    DRIP,
    DROPLET_MASS,
    LINEAR_SPACING,
    NOZZLE_OPENING_ANGLE,
    REBOUND_OPENING_ANGLE,
    REBOUND_SPEED_DISTRIBUTION,
    REDISTRIBUTION,
    RESPREADING,
    RESPREADING_BACKTRACKING_AMOUNT,
    SIGMA,
    SOLIDIFY,
    SPEED_DISTRIBUTION,
    SPRAY,
    SPRAY_COUNT,
    TC,
    K,
    L,
    S,
    X,
    Y,
    Z,
)

np.set_printoptions(threshold=sys.maxsize)

from voxel_kernels import (
    averaging_kernel,
    capacity_propagation_kernel,
    drip_kernel,
    drop_down_kernel,
    drop_mass_kernel,
    dryness_averaging_kernel,
    dryness_kernel,
    extract_particles,
    failure_distance_kernel,
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
    spray_trajectory_kernel,
    stability_kernel,
    sum_kernel,
    update_directions_kernel,
    update_distances_kernel,
)


def get_sphere_indices(radius):
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


def get_circle_indices(radius):
    """
    Generates (N, 2) array of integer offsets for pixels inside a 2D circle,
    EXCLUDING the center (0,0).
    """
    # 1. Define bounding box
    r_int = int(np.ceil(radius))
    rng = np.arange(-r_int, r_int + 1)

    # 2. Create 2D grid
    x, y = np.meshgrid(rng, rng, indexing="ij")

    # 3. Calculate squared distance
    dist_sq = x**2 + y**2

    # 4. Create mask: Inside radius AND not the center
    # dist_sq > 0 ensures (0,0) is excluded
    mask = (dist_sq <= radius**2) & (dist_sq > 0)

    # 5. Extract indices
    indices = np.vstack((x[mask], y[mask])).T

    return indices


wp.init()

print(wp.config.kernel_cache_dir)


class Voxel:
    def __init__(self, viewer, active):
        self.active = active
        self.synchronize = active

        distance = np.zeros((X + 2, Y + 2, Z + 2), dtype=np.float32) + 1e6
        dry = np.zeros((X + 2, Y + 2, Z + 2), dtype=np.float32)
        dry[:, :, :7] = 10.0
        distance[:, :, :7] = 0.0
        dry[:, Y - 5 :, :] = 10.0
        distance[:, Y - 5 :, :] = 0.0
        self.dry = wp.array(dry, dtype=wp.float32)
        self.distance = wp.array(distance, dtype=wp.float32)

        wet = np.zeros((X + 2, Y + 2, Z + 2), dtype=np.float32)
        self.wet = wp.array(wet, dtype=wp.float32)

        self.ball_indices = wp.array(get_sphere_indices(S // 2), dtype=wp.vec3i)
        if self.ball_indices.shape[0] != BALL_COUNT:
            print(self.ball_indices.shape)
            exit(1)
        self.drip_indices = get_circle_indices(1)

        self.viewer = viewer
        self.points = wp.zeros((X * Y * Z,), dtype=wp.vec3)
        self.radii = wp.zeros((self.points.shape[0],), dtype=wp.float32) + 0.01
        self.colours = wp.zeros((self.points.shape[0],), dtype=wp.vec3) + wp.vec3(0.0)
        self.i = 0
        self.last_frame = 0.0
        self.movement = np.zeros((200 * 3, 3), dtype=int)
        self.movement[:300, 0] = 70
        # self.movement[0:60:3, 0] = 70 + np.linspace(0, 200, 20)
        # self.movement[1:60:3, 0] = 70 + np.linspace(0, 200, 20)
        # self.movement[2:60:3, 0] = 70 + np.linspace(0, 200, 20)
        # self.movement[60:120, 0] = self.movement[0:60, 0][::-1]
        # self.movement[120:240, 0] = self.movement[0:120, 0][::-1]
        self.movement[300:500, 0] = 269 - np.linspace(0, 199, 200)
        self.movement[500:, 0] = 70
        # self.movement[:, 1] = 10
        # self.movement[:, 0] = X // 2
        self.movement[:200, 1] = Y - 7 - 300
        self.movement[200:300, 1] = 20 + np.linspace(0, 198, 100)
        self.movement[300:500, 1] = 220
        self.movement[500:600, 1] = 218 - np.linspace(0, 198, 100)
        self.movement[:, 2] = Z // 2
        # self.movement = wp.array(self.movement, dtype=wp.vec3i)
        self.nozzle_direction = wp.vec3f(0, 1, 0)

        self.directions = wp.zeros((K,), dtype=wp.vec3)
        self.droplet_mass = wp.zeros((K,), dtype=wp.float32)
        self.ray_trajectory = wp.zeros((K, BACKTRACK_COUNT), dtype=wp.vec3i)
        self.ray_rebound_trajectory = wp.zeros((K, 1), dtype=wp.vec3i)

    def adhesion_check(self):
        current_load = wp.zeros_like(self.wet)
        with wp.ScopedTimer("initialize load", active=self.active, synchronize=self.synchronize):
            wp.launch(initialize_load_kernel, dim=(X + 2, Y + 2, Z + 2), inputs=[self.wet, self.dry, current_load])
        with wp.ScopedTimer("capacity propagation", active=self.active, synchronize=self.synchronize):
            for _ in range(4):
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(X, Y, 1),
                    inputs=[self.wet, self.dry, current_load, self.distance, 0, Z, wp.vec3i(0, 0, 1)],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(X, 1, Z),
                    inputs=[self.wet, self.dry, current_load, self.distance, -Y, Y, wp.vec3i(0, -1, 0)],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(X, Y, 1),
                    inputs=[self.wet, self.dry, current_load, self.distance, -Z, Z, wp.vec3i(0, 0, -1)],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(1, Y, Z),
                    inputs=[self.wet, self.dry, current_load, self.distance, 0, X, wp.vec3i(1, 0, 0)],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(1, Y, Z),
                    inputs=[self.wet, self.dry, current_load, self.distance, -X, X, wp.vec3i(-1, 0, 0)],
                )
                wp.launch(
                    capacity_propagation_kernel,
                    dim=(X, 1, Z),
                    inputs=[self.wet, self.dry, current_load, self.distance, 0, Y, wp.vec3i(0, 1, 0)],
                )
        with wp.ScopedTimer("failure spread", active=self.active, synchronize=self.synchronize):
            wp.launch(
                failure_spread_kernel,
                dim=(K, BALL_COUNT),
                inputs=[self.wet, self.dry, current_load, self.ball_indices, self.ray_trajectory[:, 0]],
            )
        with wp.ScopedTimer("drop down", active=self.active, synchronize=self.synchronize):
            wp.launch(
                drop_down_kernel,
                dim=(X + 2, Y + 2),
                inputs=[self.wet, self.dry, self.distance, current_load],
            )

    def update_rebound_distances(self):
        with wp.ScopedTimer("update distances", active=self.active, synchronize=self.synchronize):
            for _ in range(5):
                wp.launch(
                    update_distances_kernel,
                    dim=(K, BALL_COUNT),
                    inputs=[self.wet, self.dry, self.distance, self.ball_indices, self.ray_rebound_trajectory[:, 0]],
                )

    def update_distances(self):
        with wp.ScopedTimer("update distances", active=self.active, synchronize=self.synchronize):
            for _ in range(5):
                wp.launch(
                    update_distances_kernel,
                    dim=(K, BALL_COUNT),
                    inputs=[self.wet, self.dry, self.distance, self.ball_indices, self.ray_trajectory[:, 0]],
                )

    def deposit(self):
        positions = wp.full((K,), wp.vec3i(*self.movement[self.i % 600]), dtype=wp.vec3i)
        with wp.ScopedTimer("alloca", active=self.active, synchronize=self.synchronize):
            wp.launch(
                update_directions_kernel,
                dim=(K,),
                inputs=[NOZZLE_OPENING_ANGLE, self.nozzle_direction, DROPLET_MASS, np.random.uniform(0, 2 * np.pi)],
                outputs=[self.directions, self.droplet_mass],
            )
        ray_indices = wp.zeros((K,), dtype=wp.int32)
        with wp.ScopedTimer("spray trajectory", active=self.active, synchronize=self.synchronize):
            wp.launch(
                spray_trajectory_kernel,
                dim=(K, SPRAY_COUNT),
                inputs=[
                    self.wet,
                    self.dry,
                    positions,
                    self.directions,
                    SPEED_DISTRIBUTION,
                    LINEAR_SPACING,
                ],
                outputs=[ray_indices],
            )
            if RESPREADING:  # and self.i == 80:
                avg_ray_index = wp.zeros(1, dtype=wp.int32)
                wp.launch(kernel=sum_kernel, dim=(K,), inputs=[ray_indices, avg_ray_index])
                # print(avg_ray_index)
                # print(ray_indices)
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(self.ray_trajectory[:, 0].numpy()[:, 0], self.ray_trajectory[:, 0].numpy()[:, 2], ray_indices.numpy())
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(self.ray_trajectory[:, 0].numpy()[:, 0], self.ray_trajectory[:, 0].numpy()[:, 2], self.droplet_mass.numpy())
                wp.launch(
                    respreading_kernel,
                    dim=(K, RESPREADING_BACKTRACKING_AMOUNT),
                    inputs=[
                        self.wet,
                        SIGMA,
                        ray_indices,
                        positions,
                        self.directions,
                        SPEED_DISTRIBUTION,
                        LINEAR_SPACING,
                        avg_ray_index,
                        self.droplet_mass,
                    ],
                    outputs=[],
                )
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(self.ray_trajectory[:, 0].numpy()[:, 0], self.ray_trajectory[:, 0].numpy()[:, 2], self.droplet_mass.numpy())
                # plt.show()
                # exit(1)
            wp.launch(
                spray_backtrack_kernel,
                dim=(K, BACKTRACK_COUNT),
                inputs=[
                    positions,
                    self.directions,
                    SPEED_DISTRIBUTION,
                    LINEAR_SPACING,
                    ray_indices,
                ],
                outputs=[self.ray_trajectory],
            )
        with wp.ScopedTimer("spray rebound", active=self.active, synchronize=self.synchronize):
            rebound_droplet_mass = wp.zeros((K,), dtype=wp.float32)
            ray_indices = wp.zeros((K,), dtype=wp.int32)
            rebound_directions = wp.zeros((K,), dtype=wp.vec3f)
            wp.launch(
                spray_rebound_kernel,
                dim=(K,),
                inputs=[
                    self.wet,
                    self.dry,
                    positions,
                    self.ray_trajectory[:, 0],
                    self.directions,
                    ray_indices,
                    SPEED_DISTRIBUTION,
                    DROPLET_MASS,
                    LINEAR_SPACING,
                ],
                outputs=[rebound_droplet_mass, rebound_directions],
            )
            wp.launch(
                randomize_directions_kernel,
                dim=(K,),
                inputs=[rebound_directions, REBOUND_OPENING_ANGLE, self.i],
            )
            wp.launch(
                spray_trajectory_kernel,
                dim=(K, SPRAY_COUNT),
                inputs=[
                    self.wet,
                    self.dry,
                    self.ray_trajectory[:, 0],
                    rebound_directions,
                    REBOUND_SPEED_DISTRIBUTION,
                    LINEAR_SPACING / 5.0,
                ],
                outputs=[ray_indices],
            )
            wp.launch(
                spray_backtrack_kernel,
                dim=(K, 1),
                inputs=[
                    self.ray_trajectory[:, 0],
                    rebound_directions,
                    REBOUND_SPEED_DISTRIBUTION,
                    LINEAR_SPACING / 5.0,
                    ray_indices,
                ],
                outputs=[self.ray_rebound_trajectory],
            )
        with wp.ScopedTimer("spray redistribution", active=self.active, synchronize=self.synchronize):
            if REDISTRIBUTION:
                spray_overlap = wp.zeros((K,), dtype=wp.float32)
                with wp.ScopedTimer("spray overlap", active=self.active, synchronize=self.synchronize):
                    wp.launch(
                        spray_overlap_kernel,
                        dim=(K, K - 1),
                        inputs=[self.ray_trajectory[:, 0]],
                        outputs=[spray_overlap],
                    )
                with wp.ScopedTimer("spray redistribution loop", active=self.active, synchronize=self.synchronize):
                    for _ in range(10):
                        wp.launch(
                            spray_redistribution_kernel,
                            dim=(K, K),
                            inputs=[self.ray_trajectory[:, 0], spray_overlap, self.droplet_mass, self.nozzle_direction],
                        )
        with wp.ScopedTimer("spray deposit", active=self.active, synchronize=self.synchronize):
            remaining_mass = wp.clone(self.droplet_mass)
            for k in range(BACKTRACK_COUNT):
                spray_neighbours = wp.zeros((K, BALL_COUNT), dtype=wp.float32)
                density = wp.zeros((K,), dtype=wp.float32)
                neighbour_count = wp.zeros((K,), dtype=wp.float32)
                with wp.ScopedTimer("spray neighbours", active=self.active, synchronize=self.synchronize):
                    wp.launch(
                        spray_neighbours_kernel,
                        dim=(K, BALL_COUNT),
                        inputs=[self.wet, self.dry, self.ball_indices, self.ray_trajectory[:, k]],
                        outputs=[spray_neighbours, density, neighbour_count],
                    )
                with wp.ScopedTimer("spray distribution", active=self.active, synchronize=self.synchronize):
                    for _ in range(100):
                        wp.launch(
                            spray_distribution_kernel,
                            dim=(K, BALL_COUNT),
                            inputs=[
                                self.wet,
                                self.dry,
                                self.ball_indices,
                                self.ray_trajectory[:, k],
                                spray_neighbours,
                                self.droplet_mass,
                                neighbour_count,
                            ],
                            outputs=[remaining_mass],
                        )
                        self.droplet_mass = wp.clone(remaining_mass)
                        if self.droplet_mass.numpy().sum() < 1e-4:
                            break
                    else:
                        # backtrack
                        continue
                    break
            else:
                print("backtracked too much")
                self.viewer._paused = True
                return
            self.update_distances()
        with wp.ScopedTimer("spray backtrack deposit", active=self.active, synchronize=self.synchronize):
            spray_neighbours = wp.zeros((K, BALL_COUNT), dtype=wp.float32)
            density = wp.zeros((K,), dtype=wp.float32)
            neighbour_count = wp.zeros((K,), dtype=wp.float32)
            wp.launch(
                spray_neighbours_kernel,
                dim=(K, BALL_COUNT),
                inputs=[self.wet, self.dry, self.ball_indices, self.ray_rebound_trajectory[:, 0]],
                outputs=[spray_neighbours, density, neighbour_count],
            )
            remaining_mass = wp.clone(rebound_droplet_mass)
            for _ in range(100):
                wp.launch(
                    spray_distribution_kernel,
                    dim=(K, BALL_COUNT),
                    inputs=[
                        self.wet,
                        self.dry,
                        self.ball_indices,
                        self.ray_rebound_trajectory[:, 0],
                        spray_neighbours,
                        rebound_droplet_mass,
                        neighbour_count,
                    ],
                    outputs=[remaining_mass],
                )
                rebound_droplet_mass = wp.clone(remaining_mass)
                if rebound_droplet_mass.numpy().sum() < 1e-4:
                    break
            self.update_rebound_distances()

    def step(self):
        if SPRAY:
            with wp.ScopedTimer("spraying", active=self.active, synchronize=self.synchronize):
                for _ in range(5):
                    self.deposit()
                    if self.viewer.is_paused():
                        break
        if self.i % 10 == 0 and ADHESION_CHECK:
            with wp.ScopedTimer("adhesion check", active=self.active, synchronize=self.synchronize):
                self.adhesion_check()
        if SOLIDIFY:
            with wp.ScopedTimer("solidify", active=self.active, synchronize=self.synchronize):
                wp.launch(solidify_kernel, dim=(X, Y, Z), inputs=[self.wet, self.dry, TC])
        if self.i % L == 0 and DRIP:
            with wp.ScopedTimer("drip", active=self.active, synchronize=self.synchronize):
                for z in range(Z + 1):
                    wp.launch(drip_kernel, dim=(X, Y), inputs=[self.wet, self.dry, self.distance, z])
        self.i += 1
        with wp.ScopedTimer("Particle Extraction", active=self.active, synchronize=self.synchronize):
            wp.launch(
                extract_particles, dim=(X, Y, Z), inputs=[self.wet, self.dry, self.points, self.radii, self.colours]
            )

    def render(self):
        self.viewer.begin_frame(1.0 / 60.0)
        self.viewer.log_points("part", self.points, self.radii, self.colours)
        self.viewer.end_frame()
        # if (time.time() - self.last_frame) < DT:
        #     time.sleep(DT - (time.time() - self.last_frame))
        # self.last_frame = time.time()


if __name__ == "__main__":
    active = False
    viewer = newton.viewer.ViewerGL()
    viewer.set_camera(wp.vec3f(1.5, -0.5, 0.5), 10.0, 90.0)
    voxel = Voxel(viewer, active)
    i = 0
    t = time.time()
    while voxel.viewer.is_running():
        if not voxel.viewer.is_paused():
            with wp.ScopedTimer("step", active=True, synchronize=True):
                voxel.step()

        if i % 10 == 0 or voxel.viewer.is_paused():
            print(f"{i}: {(time.time() - t) / 1.0 * 1000.0} ms")
            with wp.ScopedTimer("render", active=False):
                voxel.render()
            t = time.time()
            # input()

        i += 1

    voxel.viewer.close()
