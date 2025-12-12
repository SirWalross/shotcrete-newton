import warp as wp
from pyglet.window.key import O

from constants import (
    ADHESION_CHECK,
    ADHESION_STRENGTH,
    ANISOTRPIC_DISTANCE_WEIGHT,
    COMPRESSION_STRENGTH,
    OVERLAP_DISTANCE,
    REBOUND,
    RENDERING_THRESHOLD,
    SHEAR_STRENGTH,
    SPRAY_COUNT,
    WET_STRENGTH_PENALTY,
    H,
    K,
    S,
    X,
    Y,
    Z,
)


@wp.kernel
def solidify_kernel(wet: wp.array3d(dtype=wp.float32), dry: wp.array3d(dtype=wp.float32), tc: wp.float32):
    i, j, k = wp.tid()
    w = wet[i][j][k]
    d = dry[i][j][k]
    w = relu(wp.min(w + d, 1.0) - d)
    diff = wp.min(w, wp.float32(1.0) / tc)
    wet[i][j][k] = relu(w - diff - wp.max(0.0, w + d - 1.0))
    dry[i][j][k] = d + diff


@wp.kernel
def update_distances_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    distances: wp.array3d(dtype=wp.float32),
    indices: wp.array(dtype=wp.vec3i),
    positions: wp.array(dtype=wp.vec3i),
):
    i, j = wp.tid()
    wp.atomic_min(
        distances,
        positions[i][0] + indices[j][0],
        positions[i][1] + indices[j][1],
        positions[i][2] + indices[j][2],
        wp.where(
            (
                wet[positions[i][0] + indices[j][0], positions[i][1] + indices[j][1], positions[i][2] + indices[j][2]]
                + dry[positions[i][0] + indices[j][0], positions[i][1] + indices[j][1], positions[i][2] + indices[j][2]]
            )
            < 0.1,
            100.0,
            wp.min(
                wp.spatial_vector(
                    distances[
                        positions[i][0] + indices[j][0] + 1,
                        positions[i][1] + indices[j][1],
                        positions[i][2] + indices[j][2],
                    ]
                    + 1.0,
                    distances[
                        positions[i][0] + indices[j][0] - 1,
                        positions[i][1] + indices[j][1],
                        positions[i][2] + indices[j][2],
                    ]
                    + 1.0,
                    distances[
                        positions[i][0] + indices[j][0],
                        positions[i][1] + indices[j][1] + 1,
                        positions[i][2] + indices[j][2],
                    ]
                    + 1.0,
                    distances[
                        positions[i][0] + indices[j][0],
                        positions[i][1] + indices[j][1] - 1,
                        positions[i][2] + indices[j][2],
                    ]
                    + 1.0,
                    distances[
                        positions[i][0] + indices[j][0],
                        positions[i][1] + indices[j][1],
                        positions[i][2] + indices[j][2] + 1,
                    ]
                    + 5.0,
                    distances[
                        positions[i][0] + indices[j][0],
                        positions[i][1] + indices[j][1],
                        positions[i][2] + indices[j][2] - 1,
                    ]
                    + 0.1,
                )
            ),
        ),
    )


@wp.kernel
def initialize_load_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    current_load: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    d = wet[i, j, k] + dry[i, j, k]
    current_load[i, j, k] = wp.where(dry[i, j, k] > 5.0, 1000.0, wp.where(d > 0.5, -d, 0.0))


@wp.func
def compression_strength(wet: wp.float32, dry: wp.float32) -> wp.float32:
    return (wet * WET_STRENGTH_PENALTY + dry) * COMPRESSION_STRENGTH


@wp.func
def shear_strength(wet: wp.float32, dry: wp.float32) -> wp.float32:
    return (wet * WET_STRENGTH_PENALTY + dry) * SHEAR_STRENGTH


@wp.func
def adhesion_strength(wet: wp.float32, dry: wp.float32) -> wp.float32:
    return (wet * WET_STRENGTH_PENALTY + dry) * ADHESION_STRENGTH


@wp.func
def strength(wet: wp.float32, dry: wp.float32, direction: wp.vec3i) -> wp.float32:
    return wp.where(
        direction[2] == 1,
        compression_strength(wet, dry),
        wp.where(direction[2] == -1, adhesion_strength(wet, dry), shear_strength(wet, dry)),
    )


@wp.kernel
def drop_down_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    distance: wp.array3d(dtype=wp.float32),
    current_load: wp.array3d(dtype=wp.float32),
):
    i, j = wp.tid()
    write_pos = wp.int32(1)
    for k in range(Z + 2):
        if current_load[i, j, k] < 0:
            # drop down
            w = wp.atomic_exch(wet, i, j, k, 0.0)
            d = wp.atomic_exch(dry, i, j, k, 0.0)
            distance[i, j, k] = 1e6
            wet[i, j, write_pos] = w + d
            distance[i, j, write_pos] = distance[i, j, write_pos - 1] + 0.1
            write_pos += 1
        elif (wet[i, j, k] + dry[i, j, k]) > 0.5:
            write_pos = k + 1


@wp.kernel
def capacity_propagation_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    current_load: wp.array3d(dtype=wp.float32),
    distance: wp.array3d(dtype=wp.float32),
    offset: wp.int32,
    length: wp.int32,
    direction: wp.vec3i,
):
    i, j, k = wp.tid()
    for l in range(length):
        indices = wp.vec3i(i + 1, j + 1, k + 1) + direction * (l + offset)
        other = wp.vec3i(i + 1, j + 1, k + 1) + direction * (l + offset + 1)
        wd = wet[other[0], other[1], other[2]]
        dd = dry[other[0], other[1], other[2]]
        dist = distance[indices[0], indices[1], indices[2]]
        if distance[other[0], other[1], other[2]] > dist and (wd + dd) > 0.5:
            # pass capacity to neighbour
            w = wet[indices[0], indices[1], indices[2]]
            d = dry[indices[0], indices[1], indices[2]]
            load = current_load[indices[0], indices[1], indices[2]]
            num_neighbours = wp.float32(1.0)
            if direction[2] == 0:
                # calculate horizontal neighbours
                num_neighbours = wp.max(
                    1.0,
                    wp.float32(
                        distance[indices[0] + 1, indices[1], indices[2]] > dist
                        and (wet[indices[0] + 1, indices[1], indices[2]] + dry[indices[0] + 1, indices[1], indices[2]])
                        > 0.5
                    )
                    + wp.float32(
                        distance[indices[0] - 1, indices[1], indices[2]] > dist
                        and (wet[indices[0] - 1, indices[1], indices[2]] + dry[indices[0] - 1, indices[1], indices[2]])
                        > 0.5
                    )
                    + wp.float32(
                        distance[indices[0], indices[1] + 1, indices[2]] > dist
                        and (wet[indices[0], indices[1] + 1, indices[2]] + dry[indices[0], indices[1] + 1, indices[2]])
                        > 0.5
                    )
                    + wp.float32(
                        distance[indices[0], indices[1] - 1, indices[2]] > dist
                        and (wet[indices[0], indices[1] - 1, indices[2]] + dry[indices[0], indices[1] - 1, indices[2]])
                        > 0.5
                    ),
                )
            wp.atomic_max(
                current_load,
                other[0],
                other[1],
                other[2],
                wp.min(load / num_neighbours, strength(w, d, direction)) - wd - dd,
            )


@wp.kernel
def failure_spread_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    current_load: wp.array3d(dtype=wp.float32),
    indices: wp.array(dtype=wp.vec3i),
    positions: wp.array(dtype=wp.vec3i),
):
    i, j = wp.tid()
    if (
        wet[positions[i][0] + indices[j][0], positions[i][1] + indices[j][1], positions[i][2] + indices[j][2]]
        + dry[positions[i][0] + indices[j][0], positions[i][1] + indices[j][1], positions[i][2] + indices[j][2]]
    ) > 0.5:
        dist = wp.float32(S // 2) - wp.length(wp.vec3f(indices[j]))
        wp.atomic_sub(
            current_load,
            positions[i][0] + indices[j][0],
            positions[i][1] + indices[j][1],
            positions[i][2] + indices[j][2],
            relu(0.5 * dist),
        )


@wp.func
def relu(a: wp.float32) -> wp.float32:
    return wp.float32(a >= 0.0) * a


@wp.func
def drip(wet: wp.float32, density: wp.float32) -> wp.float32:
    return wp.min(wet + density, wp.float32(1.0)) - density


@wp.kernel
def drip_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    distance: wp.array3d(dtype=wp.float32),
    k: wp.int32,
):
    i, j = wp.tid()
    w = wet[i + 1, j + 1, k + 1]
    d = dry[i + 1, j + 1, k + 1]
    density_below = dry[i + 1, j + 1, k] + wet[i + 1, j + 1, k]
    drip_amount = wp.min(w, 1.0 - density_below)
    dist = distance[i + 1, j + 1, k + 1]
    dist_below = distance[i + 1, j + 1, k]
    if drip_amount > 0.0:
        # only drip below
        wp.atomic_add(wet, i + 1, j + 1, k, drip_amount)
        wp.atomic_add(wet, i + 1, j + 1, k + 1, -drip_amount)
        wp.atomic_min(distance, i + 1, j + 1, k, dist + 1.0)
    elif w > 0.0:
        # distribute drip to the side
        d1 = 1.0 - dry[i + 2, j + 1, k] - wet[i + 2, j + 1, k]
        d2 = 1.0 - dry[i + 1, j + 2, k] - wet[i + 1, j + 2, k]
        d3 = 1.0 - dry[i, j + 1, k] - wet[i, j + 1, k]
        d4 = 1.0 - dry[i + 1, j, k] - wet[i + 1, j, k]
        density_side = d1 + d2 + d3 + d4
        if density_side > 0.0:
            drip_amount = wp.min(w, density_side)
            wp.atomic_add(wet, i + 2, j + 1, k, drip_amount * d1 / density_side)
            wp.atomic_min(distance, i + 2, j + 1, k, dist_below + 1.0)
            wp.atomic_add(wet, i + 1, j + 2, k, drip_amount * d2 / density_side)
            wp.atomic_min(distance, i + 1, j + 2, k, dist_below + 1.0)
            wp.atomic_add(wet, i, j + 1, k, drip_amount * d3 / density_side)
            wp.atomic_min(distance, i, j + 1, k, dist_below + 1.0)
            wp.atomic_add(wet, i + 1, j, k, drip_amount * d4 / density_side)
            wp.atomic_min(distance, i + 1, j, k, dist_below + 1.0)
            wp.atomic_add(wet, i + 1, j + 1, k + 1, -drip_amount)
    if w + d <= drip_amount and w > 0.0:
        distance[i + 1, j + 1, k + 1] = 1e6


@wp.kernel
def stability_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    offset: wp.vec2i,
    stability: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    value = wp.float32(1.0)
    k = wp.int32(0)
    while k < (offset[1] + 256) and value != 0.0:
        value = wp.min(wp.max((wet[i + offset[0]][j][k] + dry[i + offset[0]][j][k] - 0.5) * 2.0, 0.0), value)
        k += 1
    stability[i][j] = value


@wp.kernel
def dryness_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    offset: wp.vec2i,
    dryness: wp.array2d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    if (i + offset[0]) >= 0 and (i + offset[0]) < X + 2 and (k + offset[1]) >= 10 and (k + offset[1]) < Z + 2:
        wp.atomic_add(
            dryness,
            i,
            j,
            dry[i + offset[0]][j][k + offset[1]]
            / (wet[i + offset[0]][j][k + offset[1]] + dry[i + offset[0]][j][k + offset[1]] + 1e-4)
            / 256.0,
        )


@wp.kernel
def averaging_kernel(
    input: wp.array2d(dtype=wp.float32),
    indices: wp.array(dtype=wp.vec2i),
    average: wp.array2d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    wp.atomic_add(average, i, j, input[i * 4 + indices[k][0], j * 4 + indices[k][1]] / 16.0)


@wp.kernel
def dryness_averaging_kernel(
    dryness: wp.array2d(dtype=wp.float32),
    indices: wp.array(dtype=wp.vec2i),
    average: wp.array(dtype=wp.float32),
):
    i, j, k = wp.tid()
    wp.atomic_add(average, i, dryness[j * 4 + indices[k][0], i * 4 + indices[k][1]] / 16.0 / 64.0)


@wp.kernel
def failure_distance_kernel(
    stability_avg: wp.array2d(dtype=wp.float32),
    dryness_avg: wp.array(dtype=wp.float32),
    distances: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    j = wp.int32(0)
    while j < Y // 4:
        if stability_avg[i][j] > 0.2 or (dryness_avg[j] + wp.float32(j) / 2e3) > 0.015:
            break
        j += 1
    distances[i] = j


@wp.func
def valid_drop(center_of_block: wp.vec3i, voxel: wp.vec3i) -> bool:
    diff = wp.vec3f(
        wp.float32(voxel[0] - center_of_block[0]) / 256.0,
        wp.float32(voxel[1] - center_of_block[1]) / 4.0,
        wp.float32(voxel[2] - center_of_block[2]) / 256.0,
    )
    return wp.length(diff) > 1.0


@wp.kernel
def drop_mass_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    distances: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.vec2i),
    offset: wp.vec2i,
    drop_mass: wp.array2d(dtype=wp.float32),
):
    i, j, k, l = wp.tid()
    index = wp.vec3i(i * 4 + indices[l][0] + offset[0], j * 4 + indices[l][1], k + offset[1])
    if index[0] >= 0 and index[0] < X + 2 and index[2] >= 10 and index[2] < Z + 2:
        if j < distances[i]:
            # drop whole mass in block
            mass = wet[index[0], index[1], index[2]] + dry[index[0], index[1], index[2]]
            wet[index[0], index[1], index[2]] = 0.0
            dry[index[0], index[1], index[2]] = 0.0
            wp.atomic_add(drop_mass, i, j, mass)
        elif j == distances[i]:
            # drop some mass in block to smooth out
            if valid_drop(wp.vec3i(128, j * 4, 128), wp.vec3i(i * 4 + indices[l][0], j * 4 + indices[l][0], k)):
                mass = wet[index[0], index[1], index[2]] + dry[index[0], index[1], index[2]]
                wet[index[0], index[1], index[2]] = 0.0
                dry[index[0], index[1], index[2]] = 0.0
                wp.atomic_add(drop_mass, i, j, mass)


@wp.kernel
def extract_particles(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=wp.float32),
    colours: wp.array(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    w = wet[i, j, k]
    d = dry[i, j, k]
    points[i * Y * Z + j * Z + k] = wp.vec3(
        wp.float32(i) * wp.float32((w + d) > RENDERING_THRESHOLD) * 0.01,
        wp.float32(j) * wp.float32((w + d) > RENDERING_THRESHOLD) * 0.01,
        wp.float32(k) * wp.float32((w + d) > RENDERING_THRESHOLD) * 0.01,
    )
    radius[i * Y * Z + j * Z + k] = 0.01
    if k < 7:
        colours[i * Y * Z + j * Z + k] = wp.vec3(1.0, 1.0, 1.0)
    else:
        colours[i * Y * Z + j * Z + k] = wp.vec3(w, 1.0, d)


@wp.kernel
def spray_trajectory_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    ray_pos: wp.array(dtype=wp.vec3i),
    ray_dir: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.float32),
    linear_spacing: wp.float32,
    ray_index: wp.array(dtype=wp.int32),
):
    i, j = wp.tid()
    t = wp.float32(j + 10) * linear_spacing / velocities[i]
    pos = ray_pos[i] + wp.vec3i(
        wp.int32(wp.rint(ray_dir[i][0] * velocities[i] * t / H)),
        wp.int32(wp.rint(ray_dir[i][1] * velocities[i] * t / H)),
        wp.int32(wp.rint((ray_dir[i][2] * velocities[i] * t - 1.0 / 2.0 * 9.81 * t * t) / H)),
    )
    if pos[0] < X and pos[0] >= 0 and pos[1] < Y and pos[1] >= 0 and pos[2] < Z and pos[2] >= 0:
        w = wp.float32(wet[pos[0], pos[1], pos[2]])
        d = wp.float32(dry[pos[0], pos[1], pos[2]])
        if (w + d) > 0.5:
            wp.atomic_max(ray_index, i, SPRAY_COUNT - j - 10)


@wp.kernel
def sum_kernel(data: wp.array(dtype=wp.int32), out_sum: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.atomic_add(out_sum, 0, data[i])


@wp.kernel
def respreading_kernel(
    wet: wp.array3d(dtype=wp.float32),
    sigma: wp.float32,
    ray_index: wp.array(dtype=wp.int32),
    ray_pos: wp.array(dtype=wp.vec3i),
    ray_dir: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.float32),
    linear_spacing: wp.float32,
    average_index: wp.array(dtype=wp.int32),
    droplet_mass: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    if wp.float32(ray_index[i] - j) > (wp.float32(average_index[0]) / wp.float32(K) + 5.0):
        t = wp.float32(SPRAY_COUNT - ray_index[i] + j) * linear_spacing / velocities[i]
        pos = ray_pos[i] + wp.vec3i(
            wp.int32(wp.rint(ray_dir[i][0] * velocities[i] * t / H)),
            wp.int32(wp.rint(ray_dir[i][1] * velocities[i] * t / H)),
            wp.int32(wp.rint((ray_dir[i][2] * velocities[i] * t - 1.0 / 2.0 * 9.81 * t * t) / H)),
        )
        w = (
            wp.atomic_exch(wet, pos[0], pos[1], pos[2], 0.0)
            + wp.atomic_exch(wet, pos[0] + 1, pos[1], pos[2], 0.0)
            + wp.atomic_exch(wet, pos[0] - 1, pos[1], pos[2], 0.0)
            + wp.atomic_exch(wet, pos[0], pos[1], pos[2] + 1, 0.0)
            + wp.atomic_exch(wet, pos[0], pos[1], pos[2] - 1, 0.0)
        )
        wp.atomic_add(droplet_mass, i, w)


@wp.kernel
def spray_rebound_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    ray_pos: wp.array(dtype=wp.vec3i),
    ray_hit_pos: wp.array(dtype=wp.vec3i),
    ray_dir: wp.array(dtype=wp.vec3),
    ray_index: wp.array(dtype=wp.int32),
    velocities: wp.array(dtype=wp.float32),
    droplet_mass: wp.float32,
    linear_spacing: wp.float32,
    rebound_amount: wp.array(dtype=wp.float32),
    directions: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    n = wp.normalize(
        wp.vec3f(
            wp.float32(
                (
                    wet[ray_hit_pos[i][0] - 1, ray_hit_pos[i][1], ray_hit_pos[i][2]]
                    + dry[ray_hit_pos[i][0] - 1, ray_hit_pos[i][1], ray_hit_pos[i][2]]
                )
                > 0.5
            )
            - wp.float32(
                (
                    wet[ray_hit_pos[i][0] + 1, ray_hit_pos[i][1], ray_hit_pos[i][2]]
                    + dry[ray_hit_pos[i][0] + 1, ray_hit_pos[i][1], ray_hit_pos[i][2]]
                )
                > 0.5
            ),
            wp.float32(
                (
                    wet[ray_hit_pos[i][0], ray_hit_pos[i][1] - 1, ray_hit_pos[i][2]]
                    + dry[ray_hit_pos[i][0], ray_hit_pos[i][1] - 1, ray_hit_pos[i][2]]
                )
                > 0.5
            )
            - wp.float32(
                (
                    wet[ray_hit_pos[i][0], ray_hit_pos[i][1] + 1, ray_hit_pos[i][2]]
                    + dry[ray_hit_pos[i][0], ray_hit_pos[i][1] + 1, ray_hit_pos[i][2]]
                )
                > 0.5
            ),
            wp.float32(
                (
                    wet[ray_hit_pos[i][0], ray_hit_pos[i][1], ray_hit_pos[i][2] - 1]
                    + dry[ray_hit_pos[i][0], ray_hit_pos[i][1], ray_hit_pos[i][2] - 1]
                )
                > 0.5
            )
            - wp.float32(
                (
                    wet[ray_hit_pos[i][0], ray_hit_pos[i][1], ray_hit_pos[i][2] + 1]
                    + dry[ray_hit_pos[i][0], ray_hit_pos[i][1], ray_hit_pos[i][2] + 1]
                )
                > 0.5
            )
            + 1e-4,
        )
    )
    t = wp.float32(SPRAY_COUNT - ray_index[i]) * linear_spacing / velocities[i]
    v = wp.normalize(ray_dir[i] * velocities[i] + wp.vec3f(0.0, 0.0, -9.81) * t)

    # calculate spraying factors that influence rebound
    angle = wp.acos(-wp.dot(v, n))
    distance = wp.length(wp.vec3f(ray_hit_pos[i]) - wp.vec3f(ray_pos[i])) * H

    # calculate rebound rate
    rate = wp.min(0.1 + 0.8 * wp.abs(wp.sin(angle)) + 0.5 * (1.2 - distance) * (1.2 - distance), 1.0)
    rebound_amount[i] = rate * droplet_mass * wp.float32(REBOUND)
    directions[i] = v - 2.0 * wp.dot(v, n) * n


@wp.kernel
def spray_backtrack_kernel(
    ray_pos: wp.array(dtype=wp.vec3i),
    ray_dir: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.float32),
    linear_spacing: wp.float32,
    ray_index: wp.array(dtype=wp.int32),
    ray_trajectory: wp.array2d(dtype=wp.vec3i),
):
    i, j = wp.tid()
    t = wp.float32(SPRAY_COUNT - ray_index[i] - j) * linear_spacing / velocities[i]
    pos = ray_pos[i] + wp.vec3i(
        wp.int32(wp.rint(ray_dir[i][0] * velocities[i] * t / H)),
        wp.int32(wp.rint(ray_dir[i][1] * velocities[i] * t / H)),
        wp.int32(wp.rint((ray_dir[i][2] * velocities[i] * t - 1.0 / 2.0 * 9.81 * t * t) / H)),
    )
    ray_trajectory[i][j] = pos


@wp.kernel
def spray_overlap_kernel(
    voxels: wp.array(dtype=wp.vec3i),
    overlap: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    value = wp.float32(-1.0)
    for j in range(K):
        value += relu(
            (wp.float32(OVERLAP_DISTANCE) - wp.length(wp.vec3f(voxels[i]) - wp.vec3f(voxels[j])))
            / wp.float32(OVERLAP_DISTANCE)
        )
    overlap[i] = (value / wp.float32(K) * 2.0) * (value / wp.float32(K) * 2.0) * (value / wp.float32(K) * 2.0)


@wp.func
def anisotropic_distance(p1: wp.vec3, p2: wp.vec3, direction: wp.vec3, parallel_weight: float):
    diff = p1 - p2
    dot_product = wp.dot(diff, direction)
    parallel_vec = direction * dot_product
    perp_vec = diff - parallel_vec
    weighted_diff = (parallel_vec * parallel_weight) + perp_vec
    return wp.length(weighted_diff)


@wp.kernel
def spray_redistribution_kernel(
    voxels: wp.array(dtype=wp.vec3i),
    overlap: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    direction: wp.vec3f,
):
    i, j, _ = wp.tid()
    dist = anisotropic_distance(wp.vec3f(voxels[i]), wp.vec3f(voxels[j]), direction, ANISOTRPIC_DISTANCE_WEIGHT)
    ij_overlap = relu((wp.float32(OVERLAP_DISTANCE // 4) - dist) / wp.float32(OVERLAP_DISTANCE // 4))
    if overlap[j] > overlap[i]:
        # move mass from partner
        m = mass[j] * (overlap[j] - overlap[i]) / overlap[j] * 0.4 * ij_overlap
        wp.atomic_sub(mass, j, m)
        wp.atomic_add(mass, i, m)


@wp.kernel
def spray_neighbours_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    ball_indices: wp.array(dtype=wp.vec3i),
    voxels: wp.array(dtype=wp.vec3i),
    spray_neighbours: wp.array2d(dtype=wp.float32),
    density: wp.array(dtype=wp.float32),
    neighbour_count: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()

    w = wp.float32(
        wet[voxels[i][0] + ball_indices[j][0], voxels[i][1] + ball_indices[j][1], voxels[i][2] + ball_indices[j][2]]
    )
    d = wp.float32(
        dry[voxels[i][0] + ball_indices[j][0], voxels[i][1] + ball_indices[j][1], voxels[i][2] + ball_indices[j][2]]
    )
    spray_neighbours[i][j] = wp.float32(
        relu(1.0 - w - d)
        * wp.float32(
            (
                (
                    dry[
                        voxels[i][0] + ball_indices[j][0] + 1,
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2],
                    ]
                    + wet[
                        voxels[i][0] + ball_indices[j][0] + 1,
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2],
                    ]
                )
                > 0.5
            )
            or (
                (
                    dry[
                        voxels[i][0] + ball_indices[j][0] - 1,
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2],
                    ]
                    + wet[
                        voxels[i][0] + ball_indices[j][0] - 1,
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2],
                    ]
                )
                > 0.5
            )
            or (
                (
                    dry[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1] + 1,
                        voxels[i][2] + ball_indices[j][2],
                    ]
                    + wet[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1] + 1,
                        voxels[i][2] + ball_indices[j][2],
                    ]
                )
                > 0.5
            )
            or (
                (
                    dry[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1] - 1,
                        voxels[i][2] + ball_indices[j][2],
                    ]
                    + wet[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1] - 1,
                        voxels[i][2] + ball_indices[j][2],
                    ]
                )
                > 0.5
            )
            or (
                (
                    dry[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2] + 1,
                    ]
                    + wet[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2] + 1,
                    ]
                )
                > 0.5
            )
            or (
                (
                    dry[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2] - 1,
                    ]
                    + wet[
                        voxels[i][0] + ball_indices[j][0],
                        voxels[i][1] + ball_indices[j][1],
                        voxels[i][2] + ball_indices[j][2] - 1,
                    ]
                )
                > 0.5
            )
            or (w + d) > 0.5
        )
    )
    wp.atomic_add(density, i, spray_neighbours[i][j])
    wp.atomic_add(neighbour_count, i, wp.float32(spray_neighbours[i][j] != 0.0))


@wp.kernel
def spray_distribution_kernel(
    wet: wp.array3d(dtype=wp.float32),
    dry: wp.array3d(dtype=wp.float32),
    ball_indices: wp.array(dtype=wp.vec3i),
    voxels: wp.array(dtype=wp.vec3i),
    spray_neighbours: wp.array2d(dtype=wp.float32),
    remaining_mass: wp.array(dtype=wp.float32),
    neighbour_count: wp.array(dtype=wp.float32),
    remaining_mass_out: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()

    w = wp.float32(
        wet[voxels[i][0] + ball_indices[j][0], voxels[i][1] + ball_indices[j][1], voxels[i][2] + ball_indices[j][2]]
    )
    d = wp.float32(
        dry[voxels[i][0] + ball_indices[j][0], voxels[i][1] + ball_indices[j][1], voxels[i][2] + ball_indices[j][2]]
    )
    diff = wp.min(
        (relu(remaining_mass[i]) / (neighbour_count[i] + 1.0)) * wp.float32(spray_neighbours[i][j] != 0.0),
        relu(1.0 - w - d),
    )
    wp.atomic_add(
        wet,
        voxels[i][0] + ball_indices[j][0],
        voxels[i][1] + ball_indices[j][1],
        voxels[i][2] + ball_indices[j][2],
        diff,
    )
    wp.atomic_sub(remaining_mass_out, i, diff)


@wp.kernel
def randomize_directions_kernel(
    ray_dir: wp.array(dtype=wp.vec3),
    opening_angle: wp.float32,
    seed: wp.int32,
):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    z = wp.cos(opening_angle) + wp.randf(state) * (1.0 - wp.cos(opening_angle))
    phi = wp.randf(state) * wp.pi * 2.0
    ray_dir[i] = vector_in_cone(z, phi, ray_dir[i])


@wp.func
def vector_in_cone(z: wp.float32, phi: wp.float32, direction: wp.vec3f) -> wp.vec3f:
    r = wp.sqrt(1.0 - z * z)
    x = r * wp.cos(phi)
    y = r * wp.sin(phi)
    t = wp.normalize(
        wp.cross(
            wp.vec3(
                wp.where(wp.abs(direction[0]) < 0.9, 1.0, 0.0), wp.where(wp.abs(direction[0]) < 0.9, 0.0, 1.0), 0.0
            ),
            direction,
        )
    )
    b = wp.cross(direction, t)
    return t * x + b * y + direction * z


@wp.kernel
def update_directions_kernel(
    nozzle_angle: wp.float32,
    nozzle_direction: wp.vec3f,
    droplet_mass: wp.float32,
    phi_offset: wp.float32,
    directions: wp.array(dtype=wp.vec3f),
    mass: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    z = 1.0 - (1.0 - wp.cos(nozzle_angle)) * (wp.float32(i) + 0.5) / wp.float32(K)
    phi = wp.float32(i) * wp.pi * (3.0 - wp.sqrt(5.0)) + phi_offset
    directions[i] = vector_in_cone(z, phi, nozzle_direction)
    mass[i] = mass_ratio(wp.acos(z) / nozzle_angle) * droplet_mass


@wp.func
def mass_ratio(r: wp.float32):
    a_1 = 0.713
    a_2 = 0.207
    a_3 = 0.357
    b_1 = 0.711
    b_2 = -0.207
    b_3 = 0.357
    return (a_1 * wp.exp(-wp.pow((r - a_2) / a_3, 2.0)) + b_1 * wp.exp(-wp.pow((r - b_2) / b_3, 2.0))) * 4.0
