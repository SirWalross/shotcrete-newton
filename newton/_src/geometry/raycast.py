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

# Some ray intersection functions are adapted from https://iquilezles.org/articles/intersectors/

import warp as wp

from .types import (
    GeoType,
)

# A small constant to avoid division by zero and other numerical issues
MINVAL = 1e-15


@wp.func
def ray_for_pixel(
    camera_position: wp.vec3,
    camera_direction: wp.vec3,
    camera_up: wp.vec3,
    camera_right: wp.vec3,
    scale: wp.vec2,
    resolution: wp.vec2,
    pixel_x: int,
    pixel_y: int,
):
    """
    Generate a ray for a given pixel in a perspective camera.

    Args:
        camera_position: Camera position in world space
        camera_direction: Camera forward direction (normalized)
        camera_up: Camera up direction (normalized)
        camera_right: Camera right direction (normalized)
        camera_fov: Vertical field of view in radians
        camera_aspect_ratio: Width/height aspect ratio
        camera_near_clip: Near clipping plane distance
        resolution: Image resolution as (width, height)
        pixel_x: Pixel x coordinate (0 to width-1)
        pixel_y: Pixel y coordinate (0 to height-1)

    Returns:
        Tuple of (ray_origin, ray_direction) in world space. With the direction normalized.
    """
    width = resolution[0]
    height = resolution[1]

    # Convert to normalized coordinates [-1, 1] with (0,0) at center
    ndc_x = (2.0 * float(pixel_x) + 1.0) / width - 1.0
    ndc_y = 1.0 - (2.0 * float(pixel_y) + 1.0) / height  # Flip Y axis

    # Scale by camera scale
    cam_x = ndc_x * scale[0]
    cam_y = ndc_y * scale[1]

    return camera_position + camera_right * cam_x + camera_up * cam_y, camera_direction


@wp.kernel
def raycast_sensor_kernel(
    # Model
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    voxel_world_idx: wp.array(dtype=wp.int32),
    world_indices: wp.array(dtype=wp.int32),
    # Camera parameters
    camera_position: wp.vec3,
    camera_direction: wp.vec3,
    camera_up: wp.vec3,
    camera_right: wp.vec3,
    scale: wp.vec2,
    resolution: wp.vec2,
    # Output (per-pixel results)
    hit_distances: wp.array2d(dtype=float),
):
    pixel_x, pixel_y, voxel_idx = wp.tid()
    idx = world_indices[voxel_idx]

    # check if the world index of the voxel and the sensor are equal
    if voxel_world_idx[voxel_idx] != idx:
        return

    # Skip if out of bounds
    if pixel_x >= resolution[0] or pixel_y >= resolution[1]:
        return

    # Generate ray for this pixel
    ray_origin, ray_direction = ray_for_pixel(
        camera_position,
        camera_direction,
        camera_up,
        camera_right,
        scale,
        resolution,
        pixel_x,
        pixel_y,
    )

    t = wp.int32(0)

    for i in range(1000):
        pos = wp.vec3i(
            wp.int32(wp.rint(ray_origin[0] + wp.float32(i) * ray_direction[0])),
            wp.int32(wp.rint(ray_origin[1] + wp.float32(i) * ray_direction[1])),
            wp.int32(wp.rint(ray_origin[2] + wp.float32(i) * ray_direction[2])),
        )
        if pos[0] < 256 and pos[0] >= 0 and pos[1] < 256 and pos[1] >= 0 and pos[2] < 256 and pos[2] >= 0:
            w = wp.float32(wet[idx, pos[0], pos[1], pos[2]])
            d = wp.float32(dry[idx, pos[0], pos[1], pos[2]])
            if (w + d) > 0.5:
                t = i
                break

    hit_distances[idx, pixel_x, pixel_y] = t
