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

import math
import warnings

import numpy as np
import warp as wp

from ..geometry.raycast import raycast_sensor_kernel
from ..sim import Model, State


@wp.kernel
def clamp_no_hits_kernel(depth_image: wp.array(dtype=float), max_dist: float):
    """Kernel to replace max_distance values with -1.0 to indicate no intersection."""
    tid = wp.tid()
    if depth_image[tid] >= max_dist:
        depth_image[tid] = -1.0


INT32_MAX = (1 << 31) - 1
# Upper bound on work per pixel when ray-marching particles
MAX_PARTICLE_RAY_MARCH_STEPS = 1 << 20


class RaycastSensor:
    """Raycast-based depth sensor for generating depth images.

    The RaycastSensor simulates a depth camera by casting rays from a virtual camera through each pixel
    in an image. For each pixel, it finds the closest intersection with the scene geometry and records
    the distance as a depth value.

    The sensor supports perspective cameras with configurable field of view, aspect ratio, and resolution.
    The resulting depth image has the same resolution as specified, with depth values representing the
    distance from the camera to the closest surface along each ray.

    .. rubric:: Camera Coordinate System

    The camera uses a right-handed coordinate system where:
    - The forward direction (camera_direction) is the direction the camera is looking
    - The up direction (camera_up) defines the camera's vertical orientation
    - The right direction (camera_right) is computed as the cross product of forward and up

    .. rubric:: Depth Values

    - Positive depth values: Distance to the closest surface
    - Negative depth values (-1.0): No intersection found (ray missed all geometry)

    Attributes:
        device: The device (CPU/GPU) where computations are performed
        camera_position: 3D position of the camera in world space
        camera_direction: Forward direction vector (normalized)
        camera_up: Up direction vector (normalized)
        camera_right: Right direction vector (normalized)
        fov_radians: Vertical field of view in radians
        aspect_ratio: Width/height aspect ratio
        width: Image width in pixels
        height: Image height in pixels
        depth_image: 2D depth image array (height, width)
    """

    def __init__(
        self,
        model: Model,
        camera_position: np.array,
        camera_direction: np.array,
        camera_up: tuple[float, float, float] | np.ndarray,
        camera_width: float,
        camera_height: float,
        width: int,
        height: int,
        indices: np.array,
        h: float = 0.005,
        max_distance: float = 1000.0,
    ):
        """Initialize a RaycastSensor.

        Args:
            model: The Newton model containing the geometry to raycast against
            camera_position: 3D position of the camera in voxel space
            camera_direction: Forward direction of the camera (will be normalized)
            camera_up: Up direction of the camera (will be normalized)
            camera_width: Width of camera in voxel space
            camera_height: Height of camera in voxel space
            width: Image width in pixels
            height: Image height in pixels
            max_distance: Maximum ray distance; rays beyond this return no hit
        """
        self.model = model
        self.device = model.device
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.width = width
        self.height = height
        self.max_distance = max_distance
        self.h = h

        # Set initial camera parameters
        self.camera_position = camera_position
        camera_up = np.array(camera_up, dtype=np.float32)
        self._world_indices = wp.array(indices, dtype=wp.int32, device=self.device)

        # Create depth image buffer
        self._count = self.camera_position.shape[0]
        self._depth_buffer = wp.zeros((self._count, height, width), dtype=float, device=self.device)
        self.depth_image = self._depth_buffer
        self._resolution = wp.vec2(float(width), float(height))
        self._scale = wp.vec2(camera_width, camera_height)

        # Compute camera basis vectors and warp vectors
        self._compute_camera_basis(camera_direction, camera_up)

        # Lazily constructed structure for particle queries
        self._particle_grid: wp.HashGrid | None = None
        self._particle_step_warning_emitted = False

    def _compute_camera_basis(self, direction: np.ndarray, up: np.ndarray):
        """Compute orthonormal camera basis vectors and update warp vectors.

        Args:
            direction: Camera direction vector (will be normalized)
            up: Camera up vector (will be normalized)
        """
        # Normalize direction vectors
        self.camera_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        self.camera_up = up / np.linalg.norm(up)

        # Compute right vector as cross product of forward and up
        self.camera_right = np.cross(self.camera_direction, self.camera_up)
        right_norms = np.linalg.norm(self.camera_right, axis=1, keepdims=True)

        # Slice out the directions that need fixing
        mask = (right_norms < 1e-8).squeeze()
        degen_dirs = self.camera_direction[mask]

        # Vectorized version of the "if abs(z) < 0.9" check
        # We use np.where to select between the two up-vector candidates
        # condition shape: (M, 1), broadcasted against the candidates shape (3,) -> result (M, 3)
        use_z_up = np.abs(degen_dirs[:, 2:3]) < 0.9

        up_candidates = np.where(
            use_z_up,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),  # True case
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # False case
        )

        # Recalculate cross product for the degenerate subset
        fixed_rights = np.cross(degen_dirs, up_candidates)

        # Update the original arrays with the fixed values
        self.camera_right[mask] = fixed_rights

        # Recalculate norms for the fixed vectors so normalization below works
        right_norms[mask] = np.linalg.norm(fixed_rights, axis=1, keepdims=True)

        # 4. Final normalization
        # Now safe to divide because we fixed the zero-length vectors
        self.camera_right = self.camera_right / right_norms

        # Recompute up vector to ensure orthogonality
        self.camera_up = np.cross(self.camera_right, self.camera_direction)
        self.camera_up = self.camera_up / np.linalg.norm(self.camera_up, axis=1, keepdims=True)

    def eval(
        self,
        state: State,
        march_step: float | None = None,
    ):
        """Evaluate the raycast sensor to generate a depth image.

        Casts rays from the camera through each pixel and records the distance to the closest
        intersection with the scene geometry. When ``include_particles`` is enabled (not enabled by default),
        particles stored in the simulation state are also considered.

        Args:
            state: The current state of the simulation containing body poses
            include_particles: Whether to test ray intersections against particles present in ``state``
            march_step: Optional stride for the distance in voxels for marching.
        """

        # Reset depth buffer to maximum distance
        self._depth_buffer.fill_(self.max_distance)

        wp.launch(
            kernel=raycast_sensor_kernel,
            dim=(self.width, self.height, self._count),
            inputs=[
                self.model.voxel_wet,
                self.model.voxel_dry,
                self.model.voxel_world,
                self._world_indices,
                # Camera parameters
                wp.array(self.camera_position, dtype=wp.vec3f),
                wp.array(self.camera_direction, dtype=wp.vec3f),
                wp.array(self.camera_up, dtype=wp.vec3f),
                wp.array(self.camera_right, dtype=wp.vec3f),
                self._scale,
                self._resolution,
                self.h
            ],
            outputs=[self._depth_buffer],
            device=self.device,
        )

        # Set pixels that still have max_distance to -1.0 to indicate no hit
        self._clamp_no_hits()

    def _clamp_no_hits(self):
        """Replace max_distance values with -1.0 to indicate no intersection."""
        # Flatten the depth buffer for linear indexing
        flattened_buffer = self._depth_buffer.flatten()

        wp.launch(
            kernel=clamp_no_hits_kernel,
            dim=self.height * self.width,
            inputs=[flattened_buffer, self.max_distance],
            device=self.device,
        )

    def get_depth_image(self) -> wp.array2d:
        """Get the depth image as a 2D array.

        Returns:
            2D depth image array with shape (height, width). Values are:
            - Positive: Distance to closest surface
            - -1.0: No intersection found
        """
        return self.depth_image

    def get_depth_image_numpy(self) -> np.ndarray:
        """Get the depth image as a numpy array.

        Returns:
            Numpy array with shape (height, width) containing depth values.
            Values are the same as get_depth_image() but as a numpy array.
        """
        return self.depth_image.numpy()

    def update_camera_pose(
        self,
        position: tuple[float, float, float] | np.ndarray | None = None,
        direction: tuple[float, float, float] | np.ndarray | None = None,
        up: tuple[float, float, float] | np.ndarray | None = None,
    ):
        """Update the camera pose parameters.

        Args:
            position: New camera position (if provided)
            direction: New camera direction (if provided, will be normalized)
            up: New camera up vector (if provided, will be normalized)
        """
        if position is not None:
            self.camera_position = np.array(position, dtype=np.float32)

        if direction is not None or up is not None:
            # Use current values if not provided
            camera_dir = np.array(direction, dtype=np.float32) if direction is not None else self.camera_direction
            camera_up = np.array(up, dtype=np.float32) if up is not None else self.camera_up

            # Recompute camera basis using shared method
            self._compute_camera_basis(camera_dir, camera_up)

    def update_camera_parameters(
        self,
        width: int | None = None,
        height: int | None = None,
        max_distance: float | None = None,
    ):
        """Update camera intrinsic parameters.

        Args:
            fov_radians: New vertical field of view in radians
            width: New image width in pixels
            height: New image height in pixels
            max_distance: New maximum ray distance
        """
        recreate_buffer = False

        if width is not None and width != self.width:
            self.width = width
            recreate_buffer = True

        if height is not None and height != self.height:
            self.height = height
            recreate_buffer = True

        if max_distance is not None:
            self.max_distance = max_distance

        if recreate_buffer:
            self.aspect_ratio = float(self.width) / float(self.height)
            self._resolution = wp.vec2(float(self.width), float(self.height))
            self._depth_buffer = wp.zeros((self.height, self.width), dtype=float, device=self.device)
            self.depth_image = self._depth_buffer

    def point_camera_at(
        self,
        target: tuple[float, float, float] | np.ndarray,
        position: tuple[float, float, float] | np.ndarray | None = None,
        up: tuple[float, float, float] | np.ndarray | None = None,
    ):
        """Point the camera at a specific target location.

        Args:
            target: 3D point to look at
            position: New camera position (if provided)
            up: Up vector for camera orientation (default: [0, 0, 1])
        """
        if position is not None:
            self.camera_position = np.array(position, dtype=np.float32)
        if up is None:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        target = np.array(target, dtype=np.float32)
        direction = target - self.camera_position

        self.update_camera_pose(
            position=self.camera_position,
            direction=direction,
            up=up,
        )
