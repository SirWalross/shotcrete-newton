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

import warp as wp
import sys
import numpy as np

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from ..mujoco import SolverMuJoCo

class SolverVoxel(SolverBase):
    def __init__(
        self,
        model: Model,
        *,
        mujoco_config,
    ):
        super().__init__(model=model)
        print(f"{mujoco_config=}, {model=}")
        mujoco_config.pop("solver_type")
        print(model.num_worlds)
        print(model.joint_world)
        print(model.joint_X_p)
        print(model.articulation_count)
        print(model.joint_count)
        print(model.particle_count)
        print(model.edge_count)
        self.mujoco = SolverMuJoCo(model, **mujoco_config)

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        return self.mujoco.step(state_in, state_out, control, contacts, dt)
