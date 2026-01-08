import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.utils

if __name__ == "__main__":
    ur10 = newton.ModelBuilder()

    asset_path = newton.utils.download_asset("universal_robots_ur10")
    asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
    ur10.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    model = ur10.finalize()
    pos_obj = ik.IKPositionObjective(
        link_index=7,
        link_offset=wp.vec3(0.0),
        target_positions=wp.zeros((1,), dtype=wp.vec3),
    )
    rot_obj = ik.IKRotationObjective(
        link_index=7,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.zeros((1,), dtype=wp.vec4),
    )
    obj_joint_limits = ik.IKJointLimitObjective(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, obj_joint_limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianMode.ANALYTIC,
    )
    pos_obj.set_target_positions(wp.array([-0.1639, 0.6645, 0.6236], dtype=wp.vec3))
    rot_obj.set_target_rotations(wp.array([-0.5, -0.5, 0.5, 0.5], dtype=wp.vec4))
    joint_q = wp.array([[np.pi / 2, -np.pi / 2, np.pi / 2, 0, np.pi / 2, 0]], dtype=wp.float32)
    print(joint_q)
    solver.step(joint_q, joint_q)
    print(joint_q)
