import warp as wp


@wp.func
def total_density_is_smaller(wet: wp.uint8, dry: wp.uint8, density: wp.uint8) -> bool:
    return (density > dry) and (wet < density - dry)


@wp.func
def min_six_way(a: wp.uint8, b: wp.uint8, c: wp.uint8, d: wp.uint8, e: wp.uint8, f: wp.uint8) -> wp.uint8:
    return wp.min(wp.min(wp.min(a, b), c), wp.min(wp.min(d, e), f))


@wp.func
def saturating_add(a: wp.uint8, b: wp.uint8):
    total = a + b
    return wp.select(total < a, total, wp.uint8(255))


@wp.func
def saturating_add_4(a: wp.uint8, b: wp.uint8, c: wp.uint8, d: wp.uint8) -> wp.uint8:
    total = wp.uint16(a) + wp.uint16(b) + wp.uint16(c) + wp.uint16(d)
    return wp.uint8(wp.min(total, 255))


@wp.func
def relu(a: wp.int16) -> wp.int16:
    return wp.max(0, a)


@wp.func
def saturating_sub(a: wp.uint8, b: wp.uint8) -> wp.uint8:
    return wp.select(b > a, a - b, wp.uint8(0))
