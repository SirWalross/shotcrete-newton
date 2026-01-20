import numpy as np
import warp as wp

wp.config.mode = "debug"
wp.config.verify_cuda = True

import sys

import newton
import newton.examples
import newton.utils

np.set_printoptions(threshold=sys.maxsize)


@wp.kernel
def extract_particles(
    wet: wp.array4d(dtype=wp.uint8), dry: wp.array4d(dtype=wp.uint8), points: wp.array(dtype=wp.vec3)
):
    widx, i, j, k = wp.tid()
    w = wet[widx, i, j, k]
    d = dry[widx, i, j, k]
    idx = widx * wet.shape[1] * wet.shape[2] * wet.shape[3] + i * wet.shape[3] + j * wet.shape[1] * wet.shape[3] + k
    p = points[idx]
    if w > 5:
        points[idx] = p + wp.where(p[0] < -1e3, wp.vec3f(1e4, 0.0, 0.0), wp.vec3f(0.0))
    else:
        points[idx] = p + wp.where(p[0] < -1e3, wp.vec3f(0.0), -wp.vec3f(1e4, 0.0, 0.0))


class Example:
    def __init__(self, viewer, num_worlds=4):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        self.device = wp.get_device()

        ur10 = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(ur10)

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        ur10.add_usd(asset_file, collapse_fixed_joints=False, enable_self_collisions=False, hide_collision_shapes=True)

        distance = np.zeros((num_worlds, 128 + 2, 32 + 2, 256 + 2), dtype=np.float32) + 1e6

        ur10.add_particle_grid(
            pos=wp.vec3(-(130 / 2.0) * 0.005, 0.7 + 1.2, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=distance.shape[1],
            dim_y=distance.shape[2],
            dim_z=distance.shape[3],
            cell_x=0.005,
            cell_y=0.005,
            cell_z=0.005,
            mass=0.0,
            jitter=0.0,
            radius_mean=0.002,
        )

        builder = newton.ModelBuilder()
        print("replicating...")
        builder.replicate(ur10, self.num_worlds, spacing=(2, 2, 0))
        print("done")

        # set joint position
        builder.joint_q = np.tile([np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0], num_worlds).tolist()
        builder.joint_target_pos = np.tile(
            [np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0], num_worlds
        ).tolist()

        builder.add_ground_plane()

        self.model = builder.finalize()

        dry = np.zeros_like(distance, dtype=np.uint8)
        # dry[:, :, :, :7] = 10.0
        # distance[:, :, :, :7] = 0.0
        # dry[:, :, -8:, :] = 10.0
        # distance[:, :, -8:, :] = 0.0
        self.model.voxel_dry = wp.array(dry, dtype=wp.uint8, device=self.device)
        self.model.voxel_distance = wp.array(distance, dtype=wp.uint8, device=self.device)
        self.model.voxel_wet = wp.zeros_like(self.model.voxel_dry, device=self.device)
        self.model.voxel_load = wp.zeros(self.model.voxel_dry.shape, dtype=wp.int16, device=self.device)
        self.model.voxel_pos = wp.zeros((self.model.voxel_dry.shape[0],), dtype=wp.vec3f, device=self.device)
        transforms = wp.clone(self.model.body_q[::8]).numpy()
        transforms[:, 1] += 0.7
        self.model.voxel_transform = wp.array(transforms, device=self.device, dtype=wp.transform)

        self.state_0 = self.model.state()

        # print(self.state_0.particle_q[256*130:259*130])

        # exit(1)
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        self.rewards = newton.VoxelRewards(
            (
                num_worlds,
                self.model.voxel_wet.shape[1] - 2,
                self.model.voxel_wet.shape[2] - 2,
                self.model.voxel_wet.shape[3] - 2,
            ),
            16,
            self.device,
        )

        self.solver = newton.solvers.SolverVoxel(self.model, tcp_body_name="ur10/ee_link")

        self.viewer.set_model(self.model)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.rewards, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            extract_particles,
            dim=self.model.voxel_wet.shape,
            inputs=[self.model.voxel_wet, self.model.voxel_dry],
            outputs=[self.state_0.particle_q],
        )
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=4, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
