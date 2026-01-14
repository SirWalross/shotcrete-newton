from typing import Any

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
    return wp.where(total >= a, total, wp.uint8(255))


@wp.func
def saturating_add_4(a: wp.uint8, b: wp.uint8, c: wp.uint8, d: wp.uint8) -> wp.uint8:
    total = wp.uint16(a) + wp.uint16(b) + wp.uint16(c) + wp.uint16(d)
    return wp.uint8(wp.min(total, wp.uint16(255)))


@wp.func
def relu(a: Any):
    return wp.max(type(a)(0), a)


@wp.func
def saturating_sub(a: wp.uint8, b: wp.uint8) -> wp.uint8:
    return wp.where(b > a, wp.uint8(0), a - b)


@wp.func
def is_full(wet: wp.uint8, dry: wp.uint8) -> bool:
    return wet >= (wp.uint8(255) - dry)


@wp.func
def overflow_part(a: wp.uint8, b: wp.uint8) -> wp.uint8:
    return wp.where((a + b) >= a, wp.uint8(0), a + b)
