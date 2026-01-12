import warp as wp

SPRAY_COUNT = 1000


@wp.kernel
def update_cond_kernel(
    i: wp.array(dtype=int), drip_vel: int, adhesion_check: wp.array(dtype=int), drip: wp.array(dtype=int)
):
    adhesion_check[0] = wp.int32(i[0] % 10 == 0)
    drip[0] = wp.int32(i[0] % drip_vel == 0)
    i[0] = i[0] + 1


@wp.kernel
def solidify_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    tc: wp.float32,
    bbox: wp.array2d(dtype=wp.int32),
):
    widx, i, j, k = wp.tid()
    if (
        i < bbox[widx, 0] - 2
        or i > bbox[widx, 3] + 2
        or j < bbox[widx, 1] - 2
        or j > bbox[widx, 4] + 2
        or k < bbox[widx, 2] - 2
        or k > bbox[widx, 5] + 2
    ):
        return

    w = wet[widx, i, j, k]
    d = dry[widx, i, j, k]
    w = relu(wp.min(w + d, 1.0) - d)
    diff = wp.min(w, wp.float32(1.0) / tc)
    wet[widx, i, j, k] = relu(w - diff - wp.max(0.0, w + d - 1.0))
    dry[widx, i, j, k] = d + diff


@wp.kernel
def update_distances_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    distances: wp.array4d(dtype=wp.float32),
    indices: wp.array(dtype=wp.vec3i),
    positions: wp.array2d(dtype=wp.vec3i),
):
    widx, i, j = wp.tid()
    pos = positions[widx, i] + indices[j]
    if valid_pos(pos, wet.shape):
        wp.atomic_min(
            distances,
            widx,
            pos[0],
            pos[1],
            pos[2],
            wp.where(
                (wet[widx, pos[0], pos[1], pos[2]] + dry[widx, pos[0], pos[1], pos[2]]) < 0.1,
                100.0,
                wp.min(
                    wp.spatial_vector(
                        distances[widx, pos[0] + 1, pos[1], pos[2]] + 1.0,
                        distances[widx, pos[0] - 1, pos[1], pos[2]] + 1.0,
                        distances[widx, pos[0], pos[1] + 1, pos[2]] + 1.0,
                        distances[widx, pos[0], pos[1] - 1, pos[2]] + 1.0,
                        distances[widx, pos[0], pos[1], pos[2] + 1] + 5.0,
                        distances[widx, pos[0], pos[1], pos[2] - 1] + 0.1,
                    )
                ),
            ),
        )


@wp.kernel
def initialize_load_kernel(
    wet: wp.array4d(dtype=wp.float32), dry: wp.array4d(dtype=wp.float32), current_load: wp.array4d(dtype=wp.float32)
):
    widx, i, j, k = wp.tid()
    d = dry[widx, i, j, k]
    density = wet[widx, i, j, k] + d
    current_load[widx, i, j, k] = wp.where(d > 5.0, 1000.0, wp.where(density > 0.5, -density, 0.0))


@wp.func
def compression_strength(wet: wp.float32, dry: wp.float32, wsp: wp.float32, cs: wp.float32) -> wp.float32:
    return (wet * wsp + dry) * cs


@wp.func
def shear_strength(wet: wp.float32, dry: wp.float32, wsp: wp.float32, ss: wp.float32) -> wp.float32:
    return (wet * wsp + dry) * ss


@wp.func
def adhesion_strength(wet: wp.float32, dry: wp.float32, wsp: wp.float32, as_: wp.float32) -> wp.float32:
    return (wet * wsp + dry) * as_


@wp.func
def strength(
    wet: wp.float32,
    dry: wp.float32,
    direction: wp.vec3i,
    wsp: wp.float32,
    cs: wp.float32,
    ss: wp.float32,
    as_: wp.float32,
) -> wp.float32:
    return wp.where(
        direction[2] == 1,
        compression_strength(wet, dry, wsp, cs),
        wp.where(direction[2] == -1, adhesion_strength(wet, dry, wsp, as_), shear_strength(wet, dry, wsp, ss)),
    )


@wp.kernel
def drop_down_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    distance: wp.array4d(dtype=wp.float32),
    current_load: wp.array4d(dtype=wp.float32),
    bbox: wp.array2d(dtype=wp.int32),
    adhesion_failure_amount: wp.array(dtype=wp.float32),
):
    widx, i, j = wp.tid()
    write_pos = wp.int32(1)
    z_dim = wet.shape[3]
    if not in_bbox_all_height(bbox, wp.vec3i(i, j, 0)):
        return
    for k in range(z_dim):
        if not in_bbox(bbox, wp.vec3i(i, j, k)):
            continue
        if current_load[widx, i, j, k] < 0:
            w = wet[widx, i, j, k]
            d = dry[widx, i, j, k]

            # Direct writes (no atomics)
            wet[widx, i, j, k] = 0.0
            dry[widx, i, j, k] = 0.0
            distance[widx, i, j, k] = 1e6
            wet[widx, i, j, write_pos] = w + d
            wp.atomic_add(adhesion_failure_amount, widx, w + d)

            # Distance propagation
            distance[widx, i, j, write_pos] = distance[widx, i, j, write_pos - 1] + 0.1

            write_pos += 1
        elif (wet[widx, i, j, k] + dry[widx, i, j, k]) > 0.5:
            write_pos = k + 1


@wp.func
def in_bbox(bbox: wp.array(dtype=wp.int32), i: wp.vec3i):
    return (
        ((bbox[0] - 10) > i[0])
        and ((bbox[1] - 10) > i[1])
        and ((bbox[2] - 10) > i[2])
        and ((bbox[3] + 10) < i[0])
        and ((bbox[4] + 10) < i[1])
        and ((bbox[5] + 10) < i[2])
    ) or (
        ((bbox[6] - 10) > i[0])
        and ((bbox[7] - 10) > i[1])
        and ((bbox[8] - 10) > i[2])
        and ((bbox[9] + 10) < i[0])
        and ((bbox[10] + 10) < i[1])
        and ((bbox[11] + 10) < i[2])
    )


@wp.func
def in_bbox_all_height(bbox: wp.array(dtype=wp.int32), i: wp.vec3i):
    return (
        ((bbox[0] - 10) > i[0]) and ((bbox[1] - 10) > i[1]) and ((bbox[3] + 10) < i[0]) and ((bbox[4] + 10) < i[1])
    ) or (((bbox[6] - 10) > i[0]) and ((bbox[7] - 10) > i[1]) and ((bbox[9] + 10) < i[0]) and ((bbox[10] + 10) < i[1]))


@wp.kernel
def capacity_propagation_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    current_load: wp.array4d(dtype=wp.float32),
    distance: wp.array4d(dtype=wp.float32),
    bbox: wp.array2d(dtype=wp.int32),
    offset: wp.int32,
    length: wp.int32,
    direction: wp.vec3i,
    wsp: wp.float32,
    cs: wp.float32,
    ss: wp.float32,
    as_: wp.float32,
):
    widx, i, j, k = wp.tid()
    for l in range(length):
        indices = wp.vec3i(i + 1, j + 1, k + 1) + direction * (l + offset)
        if not in_bbox(bbox[widx], indices):
            continue

        other = wp.vec3i(i + 1, j + 1, k + 1) + direction * (l + offset + 1)
        wd = wet[widx, other[0], other[1], other[2]]
        dd = dry[widx, other[0], other[1], other[2]]

        if (wd + dd) > 0.5:
            dist = distance[widx, indices[0], indices[1], indices[2]]
            if distance[widx, other[0], other[1], other[2]] > dist:
                # pass capacity to neighbour
                w = wet[widx, indices[0], indices[1], indices[2]]
                d = dry[widx, indices[0], indices[1], indices[2]]
                load = current_load[widx, indices[0], indices[1], indices[2]]

                num_neighbours = 1.0
                if direction[2] == 0:
                    n1 = (
                        distance[widx, indices[0] + 1, indices[1], indices[2]] > dist
                        and (
                            wet[widx, indices[0] + 1, indices[1], indices[2]]
                            + dry[widx, indices[0] + 1, indices[1], indices[2]]
                        )
                        > 0.5
                    )
                    n2 = (
                        distance[widx, indices[0] - 1, indices[1], indices[2]] > dist
                        and (
                            wet[widx, indices[0] - 1, indices[1], indices[2]]
                            + dry[widx, indices[0] - 1, indices[1], indices[2]]
                        )
                        > 0.5
                    )
                    n3 = (
                        distance[widx, indices[0], indices[1] + 1, indices[2]] > dist
                        and (
                            wet[widx, indices[0], indices[1] + 1, indices[2]]
                            + dry[widx, indices[0], indices[1] + 1, indices[2]]
                        )
                        > 0.5
                    )
                    n4 = (
                        distance[widx, indices[0], indices[1] - 1, indices[2]] > dist
                        and (
                            wet[widx, indices[0], indices[1] - 1, indices[2]]
                            + dry[widx, indices[0], indices[1] - 1, indices[2]]
                        )
                        > 0.5
                    )

                    num_neighbours = wp.max(1.0, wp.float32(n1) + wp.float32(n2) + wp.float32(n3) + wp.float32(n4))

                new_val = wp.min(load / num_neighbours, strength(w, d, direction, wsp, cs, ss, as_)) - wd - dd

                current_load[widx, other[0], other[1], other[2]] = wp.max(
                    current_load[widx, other[0], other[1], other[2]], new_val
                )


@wp.kernel
def failure_spread_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    current_load: wp.array4d(dtype=wp.float32),
    indices: wp.array(dtype=wp.vec3i),
    positions: wp.array2d(dtype=wp.vec3i),
):
    widx, i, j = wp.tid()
    pos = positions[widx, i] + indices[j]
    if valid_pos(pos, wet.shape):
        if (wet[widx, pos[0], pos[1], pos[2]] + dry[widx, pos[0], pos[1], pos[2]]) > 0.5:
            dist = 2.0 - wp.length(wp.vec3f(indices[j]))
            wp.atomic_sub(
                current_load,
                widx,
                positions[widx, i][0] + indices[j][0],
                positions[widx, i][1] + indices[j][1],
                positions[widx, i][2] + indices[j][2],
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
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    distance: wp.array4d(dtype=wp.float32),
    max_z: wp.int32,
):
    widx, i, j = wp.tid()

    ii = i + 1
    jj = j + 1

    for k in range(max_z):
        w = wet[widx, ii, jj, k + 1]
        if w > 0.0:
            d = dry[widx, ii, jj, k + 1]
            wet_below = wet[widx, ii, jj, k]
            density_below = dry[widx, ii, jj, k] + wet_below

            drip_amount = wp.min(w, 1.0 - density_below)
            dist = distance[widx, ii, jj, k + 1]
            dist_below = distance[widx, ii, jj, k]

            if drip_amount > 0.0:
                wet[widx, ii, jj, k] = wet_below + drip_amount
                wet[widx, ii, jj, k + 1] = w - drip_amount
                distance[widx, ii, jj, k] = wp.min(distance[widx, ii, jj, k], dist + 1.0)
            else:
                d1 = 1.0 - dry[widx, ii + 1, jj, k] - wet[widx, ii + 1, jj, k]
                d2 = 1.0 - dry[widx, ii, jj + 1, k] - wet[widx, ii, jj + 1, k]
                d3 = 1.0 - dry[widx, ii - 1, jj, k] - wet[widx, ii - 1, jj, k]
                d4 = 1.0 - dry[widx, ii, jj - 1, k] - wet[widx, ii, jj - 1, k]

                density_side = d1 + d2 + d3 + d4

                if density_side > 0.0:
                    drip_amount = wp.min(w, density_side)

                    if d1 > 0.0:
                        val = drip_amount * d1 / density_side
                        wp.atomic_add(wet, widx, ii + 1, jj, k, val)
                        wp.atomic_min(distance, widx, ii + 1, jj, k, dist_below + 1.0)

                    if d2 > 0.0:
                        val = drip_amount * d2 / density_side
                        wp.atomic_add(wet, widx, ii, jj + 1, k, val)
                        wp.atomic_min(distance, widx, ii, jj + 1, k, dist_below + 1.0)

                    if d3 > 0.0:
                        val = drip_amount * d3 / density_side
                        wp.atomic_add(wet, widx, ii - 1, jj, k, val)
                        wp.atomic_min(distance, widx, ii - 1, jj, k, dist_below + 1.0)

                    if d4 > 0.0:
                        val = drip_amount * d4 / density_side
                        wp.atomic_add(wet, widx, ii, jj - 1, k, val)
                        wp.atomic_min(distance, widx, ii, jj - 1, k, dist_below + 1.0)

                    wet[widx, ii, jj, k + 1] = wet[widx, ii, jj, k + 1] - drip_amount

            w_new = wet[widx, ii, jj, k + 1]
            if w_new + d <= drip_amount and w_new > 0.0:
                distance[widx, ii, jj, k + 1] = 1e6


@wp.kernel
def out_of_bounds_spray_kernel(
    wet: wp.array4d(dtype=wp.float32),
    ray_trajectory: wp.array3d(dtype=wp.vec3i),
    out_of_bounds_spray: wp.array(dtype=wp.float32),
):
    widx, i = wp.tid()
    wp.atomic_add(out_of_bounds_spray, widx, wp.float32(not valid_pos(ray_trajectory[widx, i, 0], wet.shape)))


@wp.kernel
def spray_trajectory_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    ray_pos: wp.array2d(dtype=wp.vec3i),
    ray_dir: wp.array2d(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.float32),
    linear_spacing: wp.float32,
    h: wp.float32,
    ray_index: wp.array2d(dtype=wp.int32),
):
    widx, i, j = wp.tid()
    t = wp.float32(j + 10) * linear_spacing / velocities[i]
    pos = ray_pos[widx, i] + wp.vec3i(
        wp.int32(wp.rint(ray_dir[widx, i][0] * velocities[i] * t / h)),
        wp.int32(wp.rint(ray_dir[widx, i][1] * velocities[i] * t / h)),
        wp.int32(wp.rint((ray_dir[widx, i][2] * velocities[i] * t - 1.0 / 2.0 * 9.81 * t * t) / h)),
    )
    if valid_pos(pos, wet.shape):
        w = wp.float32(wet[widx, pos[0], pos[1], pos[2]])
        d = wp.float32(dry[widx, pos[0], pos[1], pos[2]])
        if (w + d) > 0.5:
            wp.atomic_max(ray_index, widx, i, SPRAY_COUNT - j - 10)


@wp.func
def valid_pos(pos: wp.vec3i, shape: wp._src.types.shape_t) -> bool:
    return pos[0] < shape[1] and pos[0] >= 0 and pos[1] < shape[2] and pos[1] >= 0 and pos[2] < shape[3] and pos[2] >= 0


@wp.kernel
def sum_kernel(data: wp.array2d(dtype=wp.int32), out_sum: wp.array(dtype=wp.int32)):
    widx, i = wp.tid()
    wp.atomic_add(out_sum, widx, data[widx, i])


@wp.kernel
def respreading_kernel(
    wet: wp.array4d(dtype=wp.float32),
    sigma: wp.float32,
    ray_index: wp.array2d(dtype=wp.int32),
    ray_pos: wp.array2d(dtype=wp.vec3i),
    ray_dir: wp.array2d(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.float32),
    linear_spacing: wp.float32,
    average_index: wp.array(dtype=wp.int32),
    droplet_mass: wp.array2d(dtype=wp.float32),
    h: wp.float32,
    k: wp.int32,
):
    widx, i, j = wp.tid()
    if wp.float32(ray_index[widx, i] - j) > (wp.float32(average_index[widx]) / wp.float32(k) + 5.0):
        t = wp.float32(SPRAY_COUNT - ray_index[widx, i] + j) * linear_spacing / velocities[i]
        pos = ray_pos[widx, i] + wp.vec3i(
            wp.int32(wp.rint(ray_dir[widx, i][0] * velocities[i] * t / h)),
            wp.int32(wp.rint(ray_dir[widx, i][1] * velocities[i] * t / h)),
            wp.int32(wp.rint((ray_dir[widx, i][2] * velocities[i] * t - 1.0 / 2.0 * 9.81 * t * t) / h)),
        )
        if valid_pos(pos, wet.shape):
            w = (
                wp.atomic_exch(wet, widx, pos[0], pos[1], pos[2], 0.0)
                + wp.atomic_exch(wet, widx, pos[0] + 1, pos[1], pos[2], 0.0)
                + wp.atomic_exch(wet, widx, pos[0] - 1, pos[1], pos[2], 0.0)
                + wp.atomic_exch(wet, widx, pos[0], pos[1], pos[2] + 1, 0.0)
                + wp.atomic_exch(wet, widx, pos[0], pos[1], pos[2] - 1, 0.0)
            )
            wp.atomic_add(droplet_mass, widx, i, w)


@wp.kernel
def spray_rebound_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    ray_pos: wp.array2d(dtype=wp.vec3i),
    ray_hit_pos: wp.array2d(dtype=wp.vec3i),
    ray_dir: wp.array2d(dtype=wp.vec3),
    ray_index: wp.array2d(dtype=wp.int32),
    velocities: wp.array(dtype=wp.float32),
    droplet_mass: wp.float32,
    linear_spacing: wp.float32,
    h: wp.float32,
    rebound_amount: wp.array2d(dtype=wp.float32),
    directions: wp.array2d(dtype=wp.vec3f),
):
    widx, i = wp.tid()
    if valid_pos(ray_hit_pos[widx, i], wet.shape):
        n = wp.normalize(
            wp.vec3f(
                wp.float32(
                    (
                        wet[widx, ray_hit_pos[widx, i][0] - 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]]
                        + dry[widx, ray_hit_pos[widx, i][0] - 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]]
                    )
                    > 0.5
                )
                - wp.float32(
                    (
                        wet[widx, ray_hit_pos[widx, i][0] + 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]]
                        + dry[widx, ray_hit_pos[widx, i][0] + 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]]
                    )
                    > 0.5
                ),
                wp.float32(
                    (
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] - 1, ray_hit_pos[widx, i][2]]
                        + dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] - 1, ray_hit_pos[widx, i][2]]
                    )
                    > 0.5
                )
                - wp.float32(
                    (
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] + 1, ray_hit_pos[widx, i][2]]
                        + dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] + 1, ray_hit_pos[widx, i][2]]
                    )
                    > 0.5
                ),
                wp.float32(
                    (
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] - 1]
                        + dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] - 1]
                    )
                    > 0.5
                )
                - wp.float32(
                    (
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] + 1]
                        + dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] + 1]
                    )
                    > 0.5
                )
                + 1e-4,
            )
        )
        t = wp.float32(SPRAY_COUNT - ray_index[widx, i]) * linear_spacing / velocities[i]
        v = wp.normalize(ray_dir[widx, i] * velocities[i] + wp.vec3f(0.0, 0.0, -9.81) * t)

        # calculate spraying factors that influence rebound
        angle = wp.acos(-wp.dot(v, n))
        distance = wp.length(wp.vec3f(ray_hit_pos[widx, i]) - wp.vec3f(ray_pos[widx, i])) * h + 1.2

        # calculate rebound rate
        rate = wp.min(0.1 + 0.8 * wp.abs(wp.sin(angle)) + 0.5 * (1.2 - distance) * (1.2 - distance), 1.0)
        rebound_amount[widx, i] = rate * droplet_mass
        directions[widx, i] = v - 2.0 * wp.dot(v, n) * n


@wp.kernel
def spray_backtrack_kernel(
    ray_pos: wp.array2d(dtype=wp.vec3i),
    ray_dir: wp.array2d(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.float32),
    linear_spacing: wp.float32,
    ray_index: wp.array2d(dtype=wp.int32),
    h: wp.float32,
    ray_trajectory: wp.array3d(dtype=wp.vec3i),
):
    widx, i, j = wp.tid()
    t = wp.float32(SPRAY_COUNT - ray_index[widx, i] - j) * linear_spacing / velocities[i]
    pos = ray_pos[widx, i] + wp.vec3i(
        wp.int32(wp.rint(ray_dir[widx, i][0] * velocities[i] * t / h)),
        wp.int32(wp.rint(ray_dir[widx, i][1] * velocities[i] * t / h)),
        wp.int32(wp.rint((ray_dir[widx, i][2] * velocities[i] * t - 1.0 / 2.0 * 9.81 * t * t) / h)),
    )
    ray_trajectory[widx, i, j] = pos


@wp.kernel
def spray_overlap_kernel(
    voxels: wp.array2d(dtype=wp.vec3i), overlap_distance: wp.float32, k: wp.int32, overlap: wp.array2d(dtype=wp.float32)
):
    widx, i = wp.tid()
    value = wp.float32(-1.0)
    for j in range(k):
        value += relu(
            (overlap_distance - wp.length(wp.vec3f(voxels[widx, i]) - wp.vec3f(voxels[widx, j]))) / overlap_distance
        )
    overlap[widx, i] = (value / wp.float32(k) * 2.0) * (value / wp.float32(k) * 2.0) * (value / wp.float32(k) * 2.0)


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
    voxels: wp.array2d(dtype=wp.vec3i),
    overlap: wp.array2d(dtype=wp.float32),
    mass: wp.array2d(dtype=wp.float32),
    transforms: wp.array(dtype=wp.transform),
    overlap_distance: wp.float32,
    anisotropic_distance_weight: wp.float32,
):
    widx, i, j, _ = wp.tid()
    direction = wp.transform_vector(transforms[widx], wp.vec3f(1.0, 0.0, 0.0))
    dist = anisotropic_distance(
        wp.vec3f(voxels[widx, i]), wp.vec3f(voxels[widx, j]), direction, anisotropic_distance_weight
    )
    ij_overlap = relu(((overlap_distance / 4.0) - dist) / (overlap_distance / 4.0))
    if overlap[widx, j] > overlap[widx, i]:
        # move mass from partner
        m = mass[widx, j] * (overlap[widx, j] - overlap[widx, i]) / overlap[widx, j] * 0.4 * ij_overlap
        wp.atomic_sub(mass, widx, j, m)
        wp.atomic_add(mass, widx, i, m)


@wp.kernel
def spray_neighbours_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    ball_indices: wp.array(dtype=wp.vec3i),
    voxels: wp.array2d(dtype=wp.vec3i),
    spray_neighbours: wp.array3d(dtype=wp.float32),
    density: wp.array2d(dtype=wp.float32),
    neighbour_count: wp.array2d(dtype=wp.float32),
):
    widx, i, j = wp.tid()
    pos = wp.vec3i(
        voxels[widx, i][0] + ball_indices[j][0],
        voxels[widx, i][1] + ball_indices[j][1],
        voxels[widx, i][2] + ball_indices[j][2],
    )
    if valid_pos(pos, wet.shape):
        return
    w = wp.float32(wet[widx, pos[0], pos[1], pos[2]])
    d = wp.float32(dry[widx, pos[0], pos[1], pos[2]])

    if (w + d) >= 1.0:
        spray_neighbours[widx, i, j] = 0.0
        return

    spray_neighbours[widx, i, j] = wp.float32(
        relu(1.0 - w - d)
        * wp.float32(
            ((dry[widx, pos[0] + 1, pos[1], pos[2]] + wet[widx, pos[0] + 1, pos[1], pos[2]]) > 0.5)
            or ((dry[widx, pos[0] - 1, pos[1], pos[2]] + wet[widx, pos[0] - 1, pos[1], pos[2]]) > 0.5)
            or ((dry[widx, pos[0], pos[1] + 1, pos[2]] + wet[widx, pos[0], pos[1] + 1, pos[2]]) > 0.5)
            or ((dry[widx, pos[0], pos[1] - 1, pos[2]] + wet[widx, pos[0], pos[1] - 1, pos[2]]) > 0.5)
            or ((dry[widx, pos[0], pos[1], pos[2] + 1] + wet[widx, pos[0], pos[1], pos[2] + 1]) > 0.5)
            or ((dry[widx, pos[0], pos[1], pos[2] - 1] + wet[widx, pos[0], pos[1], pos[2] - 1]) > 0.5)
            or (w + d) > 0.5
        )
    )
    wp.atomic_add(density, widx, i, spray_neighbours[widx, i, j])
    wp.atomic_add(neighbour_count, widx, i, wp.float32(spray_neighbours[widx, i, j] != 0.0))


@wp.kernel
def spray_distribution_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    ball_indices: wp.array(dtype=wp.vec3i),
    voxels: wp.array2d(dtype=wp.vec3i),
    spray_neighbours: wp.array3d(dtype=wp.float32),
    remaining_mass: wp.array2d(dtype=wp.float32),
    neighbour_count: wp.array2d(dtype=wp.float32),
):
    widx, i, j = wp.tid()

    weight = spray_neighbours[widx, i, j]
    if weight <= 0.0:
        return

    pos = voxels[widx, i] + ball_indices[j]
    if valid_pos(pos, wet.shape):
        w = wp.float32(wet[widx, pos[0], pos[1], pos[2]])
        d = wp.float32(dry[widx, pos[0], pos[1], pos[2]])
        diff = wp.min(
            (relu(remaining_mass[widx, i]) / (neighbour_count[widx, i] + 1.0)) * wp.float32(weight != 0.0),
            relu(1.0 - w - d),
        )
        wp.atomic_add(wet, widx, pos[0], pos[1], pos[2], diff)
        wp.atomic_sub(remaining_mass, widx, i, diff)


@wp.kernel
def randomize_directions_kernel(
    ray_dir: wp.array2d(dtype=wp.vec3), opening_angle: wp.float32, seed: wp.array(dtype=wp.int32)
):
    widx, i = wp.tid()
    state = wp.rand_init(seed[0], i)
    z = wp.cos(opening_angle) + wp.randf(state) * (1.0 - wp.cos(opening_angle))
    phi = wp.randf(state) * wp.pi * 2.0
    ray_dir[widx, i] = vector_in_cone(z, phi, ray_dir[widx, i])


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
    ee_transforms: wp.array(dtype=wp.transform),
    voxel_pos: wp.array(dtype=wp.vec3f),
    droplet_mass: wp.float32,
    seed: wp.array(dtype=wp.int32),
    k: wp.int32,
    h: wp.float32,
    width: wp.int32,
    positions: wp.array2d(dtype=wp.vec3i),
    directions: wp.array2d(dtype=wp.vec3f),
    mass: wp.array2d(dtype=wp.float32),
):
    widx, i = wp.tid()
    z = 1.0 - (1.0 - wp.cos(nozzle_angle)) * (wp.float32(i) + 0.5) / wp.float32(k)
    state = wp.rand_init(seed[0])
    phi = wp.float32(i) * wp.pi * (3.0 - wp.sqrt(5.0)) + wp.randf(state, 0.0, wp.pi * 2.0)
    positions[widx, i] = wp.vec3i((wp.transform_get_translation(ee_transforms[widx]) - voxel_pos[widx]) / h) + wp.vec3i(
        width // 2, 0, 0
    )
    directions[widx, i] = vector_in_cone(z, phi, wp.transform_vector(ee_transforms[widx], wp.vec3f(1.0, 0.0, 0.0)))
    mass[widx, i] = mass_ratio(wp.acos(z) / nozzle_angle) * droplet_mass


@wp.func
def mass_ratio(r: wp.float32):
    a_1 = 0.713
    a_2 = 0.207
    a_3 = 0.357
    b_1 = 0.711
    b_2 = -0.207
    b_3 = 0.357
    return (a_1 * wp.exp(-wp.pow((r - a_2) / a_3, 2.0)) + b_1 * wp.exp(-wp.pow((r - b_2) / b_3, 2.0))) * 4.0


@wp.kernel
def spray_reward_kernel(
    wet: wp.array4d(dtype=wp.float32),
    dry: wp.array4d(dtype=wp.float32),
    h: wp.float32,
    decimation: wp.int32,
    height: wp.array3d(dtype=wp.float32),
    height_sq: wp.array3d(dtype=wp.float32),
    air_gap: wp.array3d(dtype=wp.float32),
):
    widx, i, k = wp.tid()
    hit = wp.bool(False)
    for j in range(wet.shape[2]):
        w = wet[widx, i + 1, j, k + 1]
        d = dry[widx, i + 1, j, k + 1]
        if not hit and (w + d) > 0.5:
            wp.atomic_add(height, widx, i // decimation, k // decimation, wp.float32(j) * h)
            wp.atomic_add(height_sq, widx, i // decimation, k // decimation, wp.float32(j) * h * wp.float32(j) * h)
            hit = True
        if hit:
            wp.atomic_add(air_gap, widx, i // decimation, k // decimation, relu(1.0 - w - d))


@wp.kernel
def set_floor_kernel(
    dry: wp.array4d(dtype=wp.float32),
    distance: wp.array4d(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
):
    widx, i, j = wp.tid()
    dry[indices[widx], i, j, 0] = 10.0
    distance[indices[widx], i, j, 0] = 0.0


@wp.kernel
def set_wall_kernel(
    dry: wp.array4d(dtype=wp.float32),
    distance: wp.array4d(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
):
    widx, i, j = wp.tid()
    dry[indices[widx], i, dry.shape[2] - 1, j] = 10.0
    distance[indices[widx], i, dry.shape[2] - 1, j] = 0.0


@wp.kernel
def reset_global_bbox_kernel(global_bbox: wp.array2d(dtype=wp.int32), indices: wp.array(dtype=int)):
    i = wp.tid()
    idx = indices[i]
    global_bbox[idx, 0] = 100000
    global_bbox[idx, 1] = 100000
    global_bbox[idx, 2] = 100000
    global_bbox[idx, 3] = 0
    global_bbox[idx, 4] = 0
    global_bbox[idx, 5] = 0


@wp.kernel
def expand_global_bbox_kernel(global_bbox: wp.array2d(dtype=wp.int32), spray_bbox: wp.array2d(dtype=wp.int32)):
    widx = wp.tid()

    global_bbox[widx, 0] = wp.min(global_bbox[widx, 0], wp.min(spray_bbox[widx, 0], spray_bbox[widx, 6]))
    global_bbox[widx, 1] = wp.min(global_bbox[widx, 1], wp.min(spray_bbox[widx, 1], spray_bbox[widx, 7]))
    global_bbox[widx, 2] = wp.min(global_bbox[widx, 2], wp.min(spray_bbox[widx, 2], spray_bbox[widx, 8]))
    global_bbox[widx, 3] = wp.max(global_bbox[widx, 3], wp.max(spray_bbox[widx, 3], spray_bbox[widx, 9]))
    global_bbox[widx, 4] = wp.max(global_bbox[widx, 4], wp.max(spray_bbox[widx, 4], spray_bbox[widx, 10]))
    global_bbox[widx, 5] = wp.max(global_bbox[widx, 5], wp.max(spray_bbox[widx, 5], spray_bbox[widx, 11]))


@wp.kernel
def update_robot_position_kernel(
    j: wp.array(dtype=wp.float32),
    jq: wp.array(dtype=wp.float32),
    num_joints: int,
    vel_limit: wp.array(dtype=wp.float32),
    j_target: wp.array(dtype=wp.float32),
    jq_target: wp.array(dtype=wp.float32),
    dt: float,
    out_j: wp.array(dtype=wp.float32),
    out_jq: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    v_target = (j_target[i] - j[i]) / dt
    a_target = (v_target - jq[i]) / dt
    a_next = wp.clamp(a_target, -1.0, 1.0)
    v_next = wp.clamp(jq[i] + a_next * dt, -10.0, 10.0)
    out_j[i] = j[i] + v_next * dt
    out_jq[i] = v_next


@wp.kernel
def update_body_positions_kernel(body_q_in: wp.array(dtype=wp.transformf), body_q_out: wp.array(dtype=wp.transformf)):
    i = wp.tid()
    body_q_out[i] = body_q_in[i]


@wp.kernel
def update_bbox_kernel(
    ray_traj: wp.array3d(dtype=wp.vec3i), rebound_ray_traj: wp.array3d(dtype=wp.vec3i), bbox: wp.array2d(dtype=wp.int32)
):
    widx, i, k = wp.tid()
    wp.atomic_min(bbox, widx, 0, ray_traj[widx, i, k][0])
    wp.atomic_min(bbox, widx, 1, ray_traj[widx, i, k][1])
    wp.atomic_min(bbox, widx, 2, ray_traj[widx, i, k][2])
    wp.atomic_max(bbox, widx, 3, ray_traj[widx, i, k][0])
    wp.atomic_max(bbox, widx, 4, ray_traj[widx, i, k][1])
    wp.atomic_max(bbox, widx, 5, ray_traj[widx, i, k][2])
    wp.atomic_min(bbox, widx, 6, rebound_ray_traj[widx, i, k][0])
    wp.atomic_min(bbox, widx, 7, rebound_ray_traj[widx, i, k][1])
    wp.atomic_min(bbox, widx, 8, rebound_ray_traj[widx, i, k][2])
    wp.atomic_max(bbox, widx, 9, rebound_ray_traj[widx, i, k][0])
    wp.atomic_max(bbox, widx, 10, rebound_ray_traj[widx, i, k][1])
    wp.atomic_max(bbox, widx, 11, rebound_ray_traj[widx, i, k][2])


@wp.kernel
def reset_bbox_kernel(bbox: wp.array2d(dtype=wp.int32)):
    widx = wp.tid()
    bbox[widx, 0] = 10000
    bbox[widx, 1] = 10000
    bbox[widx, 2] = 10000
    bbox[widx, 3] = 10000
    bbox[widx, 4] = 10000
    bbox[widx, 5] = 10000
    bbox[widx, 6] = 0
    bbox[widx, 7] = 0
    bbox[widx, 8] = 0
    bbox[widx, 9] = 0
    bbox[widx, 10] = 0
    bbox[widx, 11] = 0
