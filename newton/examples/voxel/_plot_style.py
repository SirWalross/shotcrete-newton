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

"""Shared print-ready plot styling for the voxel evaluation examples.

Applies a subdued, thesis-friendly matplotlib style: Computer Modern typography
(New Computer Modern when installed, otherwise matplotlib's bundled ``cmr10``),
thin recessive axes, a light grid, and a colorblind-validated categorical palette.
All figures are written as vector PDFs.
"""

import subprocess

# categorical palette (fixed assignment order, colorblind-validated)
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#1a1a1a"
TEXT_SECONDARY = "#52514e"
GRID = "#dddcd8"
SPINE = "#b5b4b0"


def _find_new_computer_modern() -> list[str]:
    """Paths of installed NewComputerModern OTF files, if any (via fontconfig)."""
    try:
        out = subprocess.run(
            ["fc-list", "--format", "%{file}\n", ":family=NewComputerModern10"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        return [line for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def setup(plt) -> None:
    """Apply the shared style to a ``matplotlib.pyplot`` module (call before plotting)."""
    from matplotlib import font_manager  # noqa: PLC0415

    family = "cmr10"  # matplotlib's bundled Computer Modern Roman
    newcm = _find_new_computer_modern()
    if newcm:
        for path in newcm:
            try:
                font_manager.fontManager.addfont(path)
            except (OSError, RuntimeError):
                pass
        family = "NewComputerModern10"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family, "cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            # cmr10 has no Unicode minus glyph; tick labels go through mathtext
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "text.color": TEXT,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.6,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "xtick.color": SPINE,
            "ytick.color": SPINE,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed TrueType, keeps text selectable/editable
        }
    )
