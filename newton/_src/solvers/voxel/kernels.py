import warp as wp

from .utils import (
    is_full,
    is_wall,
    min_six_way,
    overflow_part,
    relu,
    saturating_add,
    saturating_add_4,
    saturating_sub,
    total_density_is_smaller,
)

SPRAY_COUNT = 1000
U8_ZERO = wp.uint8(0)
U8_ONE = wp.uint8(1)
U8_MAX = wp.uint8(255)
DENSITY_ZERO = wp.uint8(0)
DENSITY_10_PERCENT = wp.uint8(25)
DENSITY_HALF = wp.uint8(128)
DENSITY_MAX = wp.uint8(255)
DENSITY_MAX_F32 = wp.float32(255)
DISTANCE_ZERO = wp.uint8(0)
DISTANCE_MAX = wp.uint8(255)
DISTANCE_WALL = wp.int16(255 + 255)
LOAD_ZERO = wp.int16(0)
LOAD_MAX = wp.int16(32767)


@wp.kernel
def update_cond_kernel(
    i: wp.array(dtype=int), drip_vel: int, adhesion_check: wp.array(dtype=int), drip: wp.array(dtype=int)
):
    adhesion_check[0] = wp.int32(i[0] % 10 == 0)
    drip[0] = wp.int32(i[0] % drip_vel == 0)
    i[0] = i[0] + 1


@wp.kernel
def solidify_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    tc: wp.uint8,
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

    if is_wall(w, d):
        return

    # calculate part that solidifes
    w = saturating_add(w, d) - d
    diff = wp.min(w, tc)

    # account for if (w + d) > DENSITY_MAX
    wet[widx, i, j, k] = saturating_sub(w - diff, overflow_part(w, d))
    dry[widx, i, j, k] = d + diff


@wp.kernel
def update_distances_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    distance: wp.array4d(dtype=wp.uint8),
    indices: wp.array(dtype=wp.vec3i),
    positions: wp.array2d(dtype=wp.vec3i),
):
    widx, i, j = wp.tid()
    pos = positions[widx, i] + indices[j]
    if valid_pos(pos, wet.shape, 1):
        distance[widx, pos[0], pos[1], pos[2]] = wp.min(
            distance[widx, pos[0], pos[1], pos[2]],
            wp.where(
                total_density_is_smaller(
                    wet[widx, pos[0], pos[1], pos[2]], dry[widx, pos[0], pos[1], pos[2]], DENSITY_10_PERCENT
                ),
                DISTANCE_MAX,
                min_six_way(
                    distance[widx, pos[0] + 1, pos[1], pos[2]] + wp.uint8(2),
                    distance[widx, pos[0] - 1, pos[1], pos[2]] + wp.uint8(2),
                    distance[widx, pos[0], pos[1] + 1, pos[2]] + wp.uint8(2),
                    distance[widx, pos[0], pos[1] - 1, pos[2]] + wp.uint8(2),
                    distance[widx, pos[0], pos[1], pos[2] + 1] + wp.uint8(5),
                    distance[widx, pos[0], pos[1], pos[2] - 1] + wp.uint8(1),
                ),
            ),
        )


@wp.kernel
def initialize_load_kernel(
    wet: wp.array4d(dtype=wp.uint8), dry: wp.array4d(dtype=wp.uint8), current_load: wp.array4d(dtype=wp.int16)
):
    widx, i, j, k = wp.tid()
    w = wet[widx, i, j, k]
    d = dry[widx, i, j, k]
    current_load[widx, i, j, k] = wp.where(
        is_wall(w, d),
        LOAD_MAX,
        wp.where(total_density_is_smaller(w, d, DENSITY_HALF), wp.int16(DENSITY_ZERO), -wp.int16(w) - wp.int16(d)),
    )


@wp.func
def compression_strength(wet: wp.uint8, dry: wp.uint8, wsp: wp.int16, cs: wp.int16) -> wp.int16:
    return (wp.int16(wet) * wp.int16(10) / wsp + wp.int16(dry) * wp.int16(10)) / wp.int16(100) * cs


@wp.func
def shear_strength(wet: wp.uint8, dry: wp.uint8, wsp: wp.int16, ss: wp.int16) -> wp.int16:
    return (wp.int16(wet) * wp.int16(10) / wsp + wp.int16(dry) * wp.int16(10)) / wp.int16(100) * ss


@wp.func
def adhesion_strength(wet: wp.uint8, dry: wp.uint8, wsp: wp.int16, as_: wp.int16) -> wp.int16:
    return (wp.int16(wet) * wp.int16(10) / wsp + wp.int16(dry) * wp.int16(10)) / wp.int16(100) * as_


@wp.func
def strength(
    wet: wp.uint8,
    dry: wp.uint8,
    direction: wp.vec3i,
    wsp: wp.int16,
    cs: wp.int16,
    ss: wp.int16,
    as_: wp.int16,
) -> wp.int16:
    return wp.where(
        direction[2] == 1,
        compression_strength(wet, dry, wsp, cs),
        wp.where(direction[2] == -1, adhesion_strength(wet, dry, wsp, as_), shear_strength(wet, dry, wsp, ss)),
    )


@wp.kernel
def drop_down_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    distance: wp.array4d(dtype=wp.uint8),
    current_load: wp.array4d(dtype=wp.int16),
    bbox: wp.array2d(dtype=wp.int32),
    adhesion_failure_amount: wp.array(dtype=wp.float32),
):
    widx, i, j = wp.tid()
    write_pos = wp.int32(1)
    z_dim = wet.shape[3]
    if not in_bbox_all_height(bbox[widx], wp.vec3i(i, j, 0)):
        return
    for k in range(z_dim):
        if not in_bbox(bbox[widx], wp.vec3i(i, j, k)):
            continue
        if current_load[widx, i, j, k] < LOAD_ZERO:
            w = wet[widx, i, j, k]
            d = dry[widx, i, j, k]

            wet[widx, i, j, k] = DENSITY_ZERO
            dry[widx, i, j, k] = DENSITY_ZERO
            distance[widx, i, j, k] = DISTANCE_MAX
            wet[widx, i, j, write_pos] = saturating_add(w, d)
            wp.atomic_add(
                adhesion_failure_amount, widx, wp.float32(w) / DENSITY_MAX_F32 + wp.float32(d) / DENSITY_MAX_F32
            )

            # Distance propagation
            distance[widx, i, j, write_pos] = saturating_add(distance[widx, i, j, write_pos - 1], wp.uint8(1))

            write_pos += 1
        elif not total_density_is_smaller(wet[widx, i, j, k], dry[widx, i, j, k], DENSITY_HALF):
            write_pos = k + 1


@wp.func
def in_bbox(bbox: wp.array(dtype=wp.int32), i: wp.vec3i):
    return (
        (i[0] >= bbox[0] - 10)
        and (i[1] >= bbox[1] - 10)
        and (i[2] >= bbox[2] - 10)
        and (i[0] <= bbox[3] + 10)
        and (i[1] <= bbox[4] + 10)
        and (i[2] <= bbox[5] + 10)
    ) or (
        (i[0] >= bbox[6] - 10)
        and (i[1] >= bbox[7] - 10)
        and (i[2] >= bbox[8] - 10)
        and (i[0] <= bbox[9] + 10)
        and (i[1] <= bbox[10] + 10)
        and (i[2] <= bbox[11] + 10)
    )


@wp.func
def in_bbox_all_height(bbox: wp.array(dtype=wp.int32), i: wp.vec3i):
    return (
        (i[0] >= bbox[0] - 10) and (i[1] >= bbox[1] - 10) and (i[0] <= bbox[3] + 10) and (i[1] <= bbox[4] + 10)
    ) or ((i[0] >= bbox[6] - 10) and (i[1] >= bbox[7] - 10) and (i[0] <= bbox[9] + 10) and (i[1] <= bbox[10] + 10))


@wp.kernel
def capacity_propagation_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    current_load: wp.array4d(dtype=wp.int16),
    distance: wp.array4d(dtype=wp.uint8),
    bbox: wp.array2d(dtype=wp.int32),
    offset: wp.int32,
    length: wp.int32,
    direction: wp.vec3i,
    wsp: wp.int16,
    cs: wp.int16,
    ss: wp.int16,
    as_: wp.int16,
):
    widx, i, j, k = wp.tid()
    for l in range(length):
        indices = wp.vec3i(i + 1, j + 1, k + 1) + direction * (l + offset)
        if not in_bbox(bbox[widx], indices):
            continue

        other = wp.vec3i(i + 1, j + 1, k + 1) + direction * (l + offset + 1)
        wd = wp.int16(wet[widx, other[0], other[1], other[2]])
        dd = wp.int16(dry[widx, other[0], other[1], other[2]])

        if (wd + dd) > wp.int16(DENSITY_HALF):
            dist = distance[widx, indices[0], indices[1], indices[2]]
            if distance[widx, other[0], other[1], other[2]] > dist:
                # pass capacity to neighbour
                w = wet[widx, indices[0], indices[1], indices[2]]
                d = dry[widx, indices[0], indices[1], indices[2]]
                load = current_load[widx, indices[0], indices[1], indices[2]]

                num_neighbours = wp.int16(1)
                if direction[2] == 0:
                    n1 = distance[widx, indices[0] + 1, indices[1], indices[2]] > dist and not total_density_is_smaller(
                        wet[widx, indices[0] + 1, indices[1], indices[2]],
                        dry[widx, indices[0] + 1, indices[1], indices[2]],
                        DENSITY_HALF,
                    )
                    n2 = distance[widx, indices[0] - 1, indices[1], indices[2]] > dist and not total_density_is_smaller(
                        wet[widx, indices[0] - 1, indices[1], indices[2]],
                        dry[widx, indices[0] - 1, indices[1], indices[2]],
                        DENSITY_HALF,
                    )
                    n3 = distance[widx, indices[0], indices[1] + 1, indices[2]] > dist and not total_density_is_smaller(
                        wet[widx, indices[0], indices[1] + 1, indices[2]],
                        dry[widx, indices[0], indices[1] + 1, indices[2]],
                        DENSITY_HALF,
                    )
                    n4 = distance[widx, indices[0], indices[1] - 1, indices[2]] > dist and not total_density_is_smaller(
                        wet[widx, indices[0], indices[1] - 1, indices[2]],
                        dry[widx, indices[0], indices[1] - 1, indices[2]],
                        DENSITY_HALF,
                    )

                    num_neighbours = wp.int16(
                        wp.max(1.0, wp.float32(n1) + wp.float32(n2) + wp.float32(n3) + wp.float32(n4))
                    )

                new_val = wp.min(load / num_neighbours, strength(w, d, direction, wsp, cs, ss, as_)) - (
                    wd + dd
                ) / wp.int16(10)

                current_load[widx, other[0], other[1], other[2]] = wp.max(
                    current_load[widx, other[0], other[1], other[2]], new_val
                )


@wp.kernel
def failure_spread_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    current_load: wp.array4d(dtype=wp.int16),
    indices: wp.array(dtype=wp.vec3i),
    positions: wp.array2d(dtype=wp.vec3i),
):
    j, i, widx = wp.tid()
    pos = positions[widx, i] + indices[j]
    if valid_pos(pos, wet.shape):
        if not total_density_is_smaller(
            wet[widx, pos[0], pos[1], pos[2]], dry[widx, pos[0], pos[1], pos[2]], DENSITY_HALF
        ):
            dist = wp.int16((1.0 - 0.5 * wp.length(wp.vec3f(indices[j]))) * 25.0)
            current_load[
                widx,
                positions[widx, i][0] + indices[j][0],
                positions[widx, i][1] + indices[j][1],
                positions[widx, i][2] + indices[j][2],
            ] -= relu(dist)


@wp.kernel
def drip_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    distance: wp.array4d(dtype=wp.uint8),
    max_z: wp.int32,
):
    j, i, widx = wp.tid()

    ii = i + 1
    jj = j + 1

    for k in range(max_z):
        w = wet[widx, ii, jj, k + 1]
        if w == DENSITY_ZERO:
            continue
        d = dry[widx, ii, jj, k + 1]
        if is_wall(w, d):
            continue
        wet_below = wet[widx, ii, jj, k]
        density_below = saturating_add(dry[widx, ii, jj, k], wet_below)

        drip_amount = wp.min(w, DENSITY_MAX - density_below)
        dist = distance[widx, ii, jj, k + 1]
        dist_below = distance[widx, ii, jj, k]

        if drip_amount > 0:
            wet[widx, ii, jj, k] = saturating_add(wet_below, drip_amount)
            wet[widx, ii, jj, k + 1] = w - drip_amount
            distance[widx, ii, jj, k] = wp.min(distance[widx, ii, jj, k], saturating_add(dist, wp.uint8(1)))
        else:
            d1 = DENSITY_MAX - saturating_add(dry[widx, ii + 1, jj, k], wet[widx, ii + 1, jj, k])
            d2 = DENSITY_MAX - saturating_add(dry[widx, ii, jj + 1, k], wet[widx, ii, jj + 1, k])
            d3 = DENSITY_MAX - saturating_add(dry[widx, ii - 1, jj, k], wet[widx, ii - 1, jj, k])
            d4 = DENSITY_MAX - saturating_add(dry[widx, ii, jj - 1, k], wet[widx, ii, jj - 1, k])

            density_side = saturating_add_4(d1, d2, d3, d4)

            if density_side > DENSITY_ZERO:
                drip_amount = wp.min(w, density_side)

                if d1 > DENSITY_ZERO:
                    val = wp.uint8(wp.float32(drip_amount) * wp.float32(d1) / wp.float32(density_side))
                    w = wet[widx, ii + 1, jj, k]
                    wet[widx, ii + 1, jj, k] = saturating_add(w, val)
                    distance[widx, ii + 1, jj, k] = wp.min(distance[widx, ii + 1, jj, k], dist_below + wp.uint8(2))

                if d2 > DENSITY_ZERO:
                    val = wp.uint8(wp.float32(drip_amount) * wp.float32(d2) / wp.float32(density_side))
                    w = wet[widx, ii, jj + 1, k]
                    wet[widx, ii, jj + 1, k] = saturating_add(w, val)
                    distance[widx, ii, jj + 1, k] = wp.min(distance[widx, ii, jj + 1, k], dist_below + wp.uint8(2))

                if d3 > DENSITY_ZERO:
                    val = wp.uint8(wp.float32(drip_amount) * wp.float32(d3) / wp.float32(density_side))
                    w = wet[widx, ii - 1, jj, k]
                    wet[widx, ii - 1, jj, k] = saturating_add(w, val)
                    distance[widx, ii - 1, jj, k] = wp.min(distance[widx, ii - 1, jj, k], dist_below + wp.uint8(2))

                if d4 > DENSITY_ZERO:
                    val = wp.uint8(wp.float32(drip_amount) * wp.float32(d4) / wp.float32(density_side))
                    w = wet[widx, ii, jj - 1, k]
                    wet[widx, ii, jj - 1, k] = saturating_add(w, val)
                    distance[widx, ii, jj - 1, k] = wp.min(distance[widx, ii, jj - 1, k], dist_below + wp.uint8(2))

                wet[widx, ii, jj, k + 1] = saturating_sub(wet[widx, ii, jj, k + 1], drip_amount)

        w_new = wet[widx, ii, jj, k + 1]
        if w_new > 0 and total_density_is_smaller(w_new, d, drip_amount):
            distance[widx, ii, jj, k + 1] = DENSITY_MAX


@wp.kernel
def out_of_bounds_spray_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    ray_trajectory: wp.array3d(dtype=wp.vec3i),
    out_of_bounds_spray: wp.array(dtype=wp.float32),
):
    widx, i = wp.tid()
    wp.atomic_add(out_of_bounds_spray, widx, wp.float32(not valid_pos(ray_trajectory[widx, i, 0], wet.shape)))


@wp.kernel
def spray_trajectory_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
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
        w = wet[widx, pos[0], pos[1], pos[2]]
        d = dry[widx, pos[0], pos[1], pos[2]]
        if not total_density_is_smaller(w, d, DENSITY_HALF):
            wp.atomic_max(ray_index, widx, i, SPRAY_COUNT - j - 10)


@wp.func
def valid_pos(pos: wp.vec3i, shape: wp._src.types.shape_t, tolerance: int = 0) -> bool:
    return (
        pos[0] < shape[1] - tolerance
        and pos[0] >= tolerance
        and pos[1] < shape[2] - tolerance
        and pos[1] >= tolerance
        and pos[2] < shape[3] - tolerance
        and pos[2] >= tolerance
    )


@wp.kernel
def sum_kernel(data: wp.array2d(dtype=wp.int32), out_sum: wp.array(dtype=wp.int32)):
    widx, i = wp.tid()
    wp.atomic_add(out_sum, widx, data[widx, i])


@wp.kernel
def respreading_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
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
            w1 = wp.float32(wet[widx, pos[0], pos[1], pos[2]])
            d1 = wp.float32(dry[widx, pos[0], pos[1], pos[2]])
            if not is_wall(w1, d1):
                wet[widx, pos[0], pos[1], pos[2]] = DENSITY_ZERO
            else:
                w1 = 0.0
            w2 = wp.float32(wet[widx, pos[0] + 1, pos[1], pos[2]])
            d2 = wp.float32(dry[widx, pos[0] + 1, pos[1], pos[2]])
            if not is_wall(w2, d2):
                wet[widx, pos[0] + 1, pos[1], pos[2]] = DENSITY_ZERO
            else:
                w2 = 0.0
            w3 = wp.float32(wet[widx, pos[0] - 1, pos[1], pos[2]])
            d3 = wp.float32(dry[widx, pos[0] - 1, pos[1], pos[2]])
            if not is_wall(w3, d3):
                wet[widx, pos[0] - 1, pos[1], pos[2]] = DENSITY_ZERO
            else:
                w3 = 0.0
            w4 = wp.float32(wet[widx, pos[0], pos[1], pos[2] + 1])
            d4 = wp.float32(dry[widx, pos[0], pos[1], pos[2] + 1])
            if not is_wall(w4, d4):
                wet[widx, pos[0], pos[1], pos[2] + 1] = DENSITY_ZERO
            else:
                w4 = 0.0
            w5 = wp.float32(wet[widx, pos[0], pos[1], pos[2] - 1])
            d5 = wp.float32(dry[widx, pos[0], pos[1], pos[2] - 1])
            if not is_wall(w5, d5):
                wet[widx, pos[0], pos[1], pos[2] - 1] = DENSITY_ZERO
            else:
                w5 = 0.0
            wp.atomic_add(droplet_mass, widx, i, w1 + w2 + w3 + w4 + w5)


@wp.kernel
def spray_rebound_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    ray_pos: wp.array2d(dtype=wp.vec3i),
    ray_hit_pos: wp.array2d(dtype=wp.vec3i),
    ray_dir: wp.array2d(dtype=wp.vec3),
    ray_index: wp.array2d(dtype=wp.int32),
    velocities: wp.array(dtype=wp.float32),
    droplet_mass: wp.array2d(dtype=wp.float32),
    linear_spacing: wp.float32,
    h: wp.float32,
    rebound_amount: wp.array2d(dtype=wp.float32),
    directions: wp.array2d(dtype=wp.vec3f),
):
    widx, i = wp.tid()
    if valid_pos(ray_hit_pos[widx, i], wet.shape, 1):
        n = wp.normalize(
            wp.vec3f(
                wp.float32(
                    total_density_is_smaller(
                        wet[widx, ray_hit_pos[widx, i][0] - 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]],
                        dry[widx, ray_hit_pos[widx, i][0] - 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]],
                        DENSITY_HALF,
                    )
                )
                - wp.float32(
                    total_density_is_smaller(
                        wet[widx, ray_hit_pos[widx, i][0] + 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]],
                        dry[widx, ray_hit_pos[widx, i][0] + 1, ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2]],
                        DENSITY_HALF,
                    )
                ),
                wp.float32(
                    total_density_is_smaller(
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] - 1, ray_hit_pos[widx, i][2]],
                        dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] - 1, ray_hit_pos[widx, i][2]],
                        DENSITY_HALF,
                    )
                )
                - wp.float32(
                    total_density_is_smaller(
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] + 1, ray_hit_pos[widx, i][2]],
                        dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1] + 1, ray_hit_pos[widx, i][2]],
                        DENSITY_HALF,
                    )
                ),
                wp.float32(
                    total_density_is_smaller(
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] - 1],
                        dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] - 1],
                        DENSITY_HALF,
                    )
                )
                - wp.float32(
                    total_density_is_smaller(
                        wet[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] + 1],
                        dry[widx, ray_hit_pos[widx, i][0], ray_hit_pos[widx, i][1], ray_hit_pos[widx, i][2] + 1],
                        DENSITY_HALF,
                    )
                )
                + 1e-4,
            )
        )
        t = wp.float32(SPRAY_COUNT - ray_index[widx, i]) * linear_spacing / velocities[i]
        v = wp.normalize(ray_dir[widx, i] * velocities[i] + wp.vec3f(0.0, 0.0, -9.81) * t)

        # calculate spraying factors that influence rebound
        angle = wp.acos(-wp.dot(v, -n))
        distance = wp.length(wp.vec3f(ray_hit_pos[widx, i]) - wp.vec3f(ray_pos[widx, i])) * h

        # calculate rebound rate
        rate = wp.min(0.1 + 0.2 * wp.abs(wp.sin(angle)) + 0.3 * (1.2 - distance) * (1.2 - distance), 1.0)
        mass = droplet_mass[widx, i]
        rebound_amount[widx, i] = rate * mass
        droplet_mass[widx, i] -= rate * droplet_mass[widx, i]
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
    k: wp.int32,
):
    widx, i = wp.tid()
    direction = wp.transform_vector(transforms[widx], wp.vec3f(1.0, 0.0, 0.0))
    my_overlap = overlap[widx, i]
    for j in range(k):
        dist = anisotropic_distance(
            wp.vec3f(voxels[widx, i]), wp.vec3f(voxels[widx, j]), direction, anisotropic_distance_weight
        )
        other_overlap = overlap[widx, j]
        if other_overlap > my_overlap:
            ij_overlap = relu(((overlap_distance / 4.0) - dist) / (overlap_distance / 4.0))
            # move mass from partner
            m = mass[widx, j] * (overlap[widx, j] - overlap[widx, i]) / overlap[widx, j] * 0.4 * ij_overlap
            wp.atomic_sub(mass, widx, j, m)
            wp.atomic_add(mass, widx, i, m)


@wp.kernel
def spray_neighbours_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
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
    if not valid_pos(pos, wet.shape, 1):
        return
    w = wet[widx, pos[0], pos[1], pos[2]]
    d = dry[widx, pos[0], pos[1], pos[2]]

    if is_full(w, d):
        spray_neighbours[widx, i, j] = 0.0
        return

    spray_neighbours[widx, i, j] = relu(
        1.0 - wp.float32(w) / DENSITY_MAX_F32 - wp.float32(d) / DENSITY_MAX_F32
    ) * wp.float32(
        not total_density_is_smaller(
            dry[widx, pos[0] + 1, pos[1], pos[2]], wet[widx, pos[0] + 1, pos[1], pos[2]], DENSITY_HALF
        )
        or not total_density_is_smaller(
            dry[widx, pos[0] - 1, pos[1], pos[2]], wet[widx, pos[0] - 1, pos[1], pos[2]], DENSITY_HALF
        )
        or not total_density_is_smaller(
            dry[widx, pos[0], pos[1] + 1, pos[2]], wet[widx, pos[0], pos[1] + 1, pos[2]], DENSITY_HALF
        )
        or not total_density_is_smaller(
            dry[widx, pos[0], pos[1] - 1, pos[2]], wet[widx, pos[0], pos[1] - 1, pos[2]], DENSITY_HALF
        )
        or not total_density_is_smaller(
            dry[widx, pos[0], pos[1], pos[2] + 1], wet[widx, pos[0], pos[1], pos[2] + 1], DENSITY_HALF
        )
        or not total_density_is_smaller(
            dry[widx, pos[0], pos[1], pos[2] - 1], wet[widx, pos[0], pos[1], pos[2] - 1], DENSITY_HALF
        )
        or not total_density_is_smaller(w, d, DENSITY_HALF)
    )
    wp.atomic_add(density, widx, i, spray_neighbours[widx, i, j])
    wp.atomic_add(neighbour_count, widx, i, wp.float32(spray_neighbours[widx, i, j] != 0.0))


@wp.kernel
def spray_distribution_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    ball_indices: wp.array(dtype=wp.vec3i),
    voxels: wp.array2d(dtype=wp.vec3i),
    spray_neighbours: wp.array3d(dtype=wp.float32),
    remaining_mass: wp.array2d(dtype=wp.float32),
    neighbour_count: wp.array2d(dtype=wp.float32),
    seed: wp.array(dtype=wp.int32),
    seed2: wp.int32,
):
    j, i, widx = wp.tid()

    weight = spray_neighbours[widx, i, j]
    if weight <= 0.0:
        return

    pos = voxels[widx, i] + ball_indices[j]
    if not valid_pos(pos, wet.shape):
        return
    w = wet[widx, pos[0], pos[1], pos[2]]
    d = dry[widx, pos[0], pos[1], pos[2]]
    if total_density_is_smaller(w, d, DENSITY_MAX):
        w_f32 = wp.float32(w) / DENSITY_MAX_F32
        d_f32 = wp.float32(d) / DENSITY_MAX_F32
        state = wp.rand_init(wp.int32(wp.rand_init(seed[0], seed2)), wp.int32(wp.rand_init(i, j)))
        diff = (
            wp.min(
                (relu(remaining_mass[widx, i]) / (neighbour_count[widx, i] + 1.0)) * wp.float32(weight != 0.0) * 0.5,
                relu(1.0 - w_f32 - d_f32),
            )
            * DENSITY_MAX_F32
        )
        diff = wp.where((diff - wp.floor(diff)) > wp.randf(state), wp.ceil(diff), wp.floor(diff))
        wet[widx, pos[0], pos[1], pos[2]] += wp.uint8(diff)
        wp.atomic_sub(remaining_mass, widx, i, wp.floor(diff) / DENSITY_MAX_F32)


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
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    h: wp.float32,
    decimation: wp.int32,
    bbox: wp.array2d(dtype=wp.int32),
    height: wp.array3d(dtype=wp.float32),
    height_sq: wp.array3d(dtype=wp.float32),
    air_gap: wp.array3d(dtype=wp.float32),
):
    widx, i, k = wp.tid()
    hit = wp.bool(False)

    if i < bbox[widx, 0] - 2 or i > bbox[widx, 3] + 2 or k < bbox[widx, 2] - 2 or k > bbox[widx, 5] + 2:
        wp.atomic_add(height, widx, i // decimation, k // decimation, wp.float32(wet.shape[2] - 2) * h)
        wp.atomic_add(
            height_sq,
            widx,
            i // decimation,
            k // decimation,
            wp.float32(wet.shape[2] - 2) * h * wp.float32(wet.shape[2] - 2) * h,
        )
        return

    local_gap = wp.float32(0.0)

    for j in range(wet.shape[2]):
        w = wet[widx, i + 1, j, k + 1]
        d = dry[widx, i + 1, j, k + 1]
        if not hit and not total_density_is_smaller(w, d, DENSITY_HALF):
            wp.atomic_add(height, widx, i // decimation, k // decimation, wp.float32(j) * h)
            wp.atomic_add(height_sq, widx, i // decimation, k // decimation, wp.float32(j) * h * wp.float32(j) * h)
            hit = True
        if hit:
            local_gap += relu(1.0 - wp.float32(w) - wp.float32(d))
    wp.atomic_add(air_gap, widx, i // decimation, k // decimation, local_gap)


@wp.kernel
def set_floor_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    distance: wp.array4d(dtype=wp.uint8),
    indices: wp.array(dtype=wp.int32),
):
    widx, i, j = wp.tid()
    dry[indices[widx], i, j, 0] = DENSITY_MAX
    wet[indices[widx], i, j, 0] = DENSITY_MAX
    dry[indices[widx], i, j, 1] = DENSITY_MAX
    wet[indices[widx], i, j, 1] = DENSITY_MAX
    distance[indices[widx], i, j, 0] = DISTANCE_ZERO
    distance[indices[widx], i, j, 1] = DISTANCE_ZERO


@wp.kernel
def set_wall_kernel(
    wet: wp.array4d(dtype=wp.uint8),
    dry: wp.array4d(dtype=wp.uint8),
    distance: wp.array4d(dtype=wp.uint8),
    indices: wp.array(dtype=wp.int32),
):
    widx, i, j = wp.tid()
    dry[indices[widx], i, dry.shape[2] - 1, j] = DENSITY_MAX
    wet[indices[widx], i, dry.shape[2] - 1, j] = DENSITY_MAX
    dry[indices[widx], i, dry.shape[2] - 2, j] = DENSITY_MAX
    wet[indices[widx], i, dry.shape[2] - 2, j] = DENSITY_MAX
    distance[indices[widx], i, dry.shape[2] - 1, j] = DISTANCE_ZERO
    distance[indices[widx], i, dry.shape[2] - 2, j] = DISTANCE_ZERO


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
    if k == 0:
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
    bbox[widx, 3] = 0
    bbox[widx, 4] = 0
    bbox[widx, 5] = 0
    bbox[widx, 6] = 10000
    bbox[widx, 7] = 10000
    bbox[widx, 8] = 10000
    bbox[widx, 9] = 0
    bbox[widx, 10] = 0
    bbox[widx, 11] = 0
