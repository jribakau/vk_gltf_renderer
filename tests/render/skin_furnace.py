#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: measured skin lit from the front by a white furnace is its colour map.

FV_materials_subsurface makes the medium from the surface's colour: the base colour is taken as the
multiple-scattering albedo A and inverted to a single-scattering albedo (Chiang, Kutz and Burley
2016), and the walk leaves white through its diffuse exit (ptVolumeDiffuseExit). Under a uniform
white environment with no specular layer, then, a thick slab's front face should come back at about
A. With ptDiffuseEntry the walk is entered through a Lambertian lobe instead of by refraction (how
Unreal's path tracer enters subsurface): the script renders both entries and prints how far each
lands from A, per channel, and checks that the diffuse entry stays within 10% of it.

The script writes a glTF scene (a slab of skin-like medium filling the view, specular weight 0, no
lights), renders it headless to a linear .hdr under `--envSystem 3` (plain colour, white), and reads
the centre of the image.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/skin_furnace.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import TECHNIQUES, read_hdr
from volume_walk_exit import box_faces, make_gltf

IMAGE_SIZE = 128
SAMPLES    = 1024

SLAB_HALF       = (3.0, 3.0, 0.5)  # far wider than the view, thick against the scatter distance
BASE            = [0.8, 0.55, 0.45]  # a pale skin tone, as the multiple-scattering albedo
SCATTER_DIST    = [0.00367, 0.00137, 0.00068]  # figure/materials kSkinScatterDistance, metres
TOLERANCE       = 0.10


def skin_material():
    return {
        "name": "Skin",
        "pbrMetallicRoughness": {"baseColorFactor": BASE + [1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5},
        "extensions": {
            "KHR_materials_specular": {"specularFactor": 0.0},  # no reflection: all of the light enters
            "KHR_materials_ior": {"ior": 1.4},
            "KHR_materials_transmission": {"transmissionFactor": 1.0},
            "KHR_materials_volume": {"thicknessFactor": 1.0, "attenuationDistance": 1.0, "attenuationColor": [1.0, 1.0, 1.0]},
            "KHR_materials_scatter": {"scatterStrengthFactor": 1.0, "multiscatterColorFactor": [1.0, 1.0, 1.0],
                                      "scatterAnisotropy": 0.0},
            "FV_materials_subsurface": {"scatterDistance": SCATTER_DIST},
        },
    }


def scene():
    primitives = [(positions, normals, indices, 0) for _, _, positions, normals, indices in box_faces(SLAB_HALF)]
    gltf = make_gltf(primitives, [skin_material()], "skin_furnace.py")
    gltf["extensionsUsed"] += ["KHR_materials_specular", "KHR_materials_ior", "FV_materials_subsurface"]
    # no light: the environment alone
    gltf["scenes"][0]["nodes"] = [0, 1]
    del gltf["extensions"]
    gltf["extensionsUsed"].remove("KHR_lights_punctual")
    return gltf


def center_rgb(path):
    """Mean linear RGB over the central quarter of the image, which the slab's front face fills."""
    width, height, px = read_hdr(path)
    total, count = [0.0, 0.0, 0.0], 0
    for y in range(3 * height // 8, 5 * height // 8):
        for x in range(3 * width // 8, 5 * width // 8):
            i = 3 * (y * width + x)
            for c in range(3):
                total[c] += px[i + c]
            count += 1
    return [t / count for t in total]


def render(exe, scene_path, output_path, technique, diffuse_entry, walk_guiding):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "3", "--plainColor", "1", "1", "1", "--renderSystem", "0",
           "--ptTechnique", str(technique), "--ptVolumeDiffuseExit", "1",
           "--ptDiffuseEntry", "1" if diffuse_entry else "0", "--ptWalkGuiding", str(walk_guiding), "--ptMaxDepth", "20",
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(output_path).exists():
        print(result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"render failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "skin_furnace",
                        help="where the scene and images go")
    parser.add_argument("--walk-guiding", type=int, choices=(0, 1), default=1,
                        help="ptWalkGuiding: the walks' Dwivedi guiding (the result should not depend on it)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scene_path = args.out / "slab.gltf"
    scene_path.write_text(json.dumps(scene()))

    failures = 0
    for technique_name, technique in TECHNIQUES.items():
        for diffuse_entry in (False, True):
            image = args.out / f"{technique_name}_{'diffuse' if diffuse_entry else 'refraction'}.hdr"
            render(args.exe, scene_path, image, technique, diffuse_entry, args.walk_guiding)
            rgb   = center_rgb(image)
            ratio = [rgb[c] / BASE[c] for c in range(3)]
            label = "diffuse entry   " if diffuse_entry else "refraction entry"
            print(f"{technique_name:10s} {label}: rgb {', '.join(f'{v:.3f}' for v in rgb)}"
                  f"  / base: {', '.join(f'{r:.3f}' for r in ratio)}")
            if diffuse_entry and any(abs(r - 1.0) > TOLERANCE for r in ratio):
                print(f"FAIL: {technique_name}: the diffuse entry's front face is not its colour within {TOLERANCE:.0%}")
                failures += 1
    print("PASS" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
