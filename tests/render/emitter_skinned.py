#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: an emitter a skin has stretched lights like the surface it has become.

A floor is lit by one emissive panel hung above it, in two scenes: a panel one metre square, and a
panel half a metre square that a skin's joint scales to one metre. The two are the same surface when
drawn, so the floor must be lit the same in both.

The light table is built from the mesh's own positions, before its skin, so it weighs the skinned
panel's triangles by a quarter of the area they have when drawn. A light sample is picked by that
weight and then placed evenly over the drawn triangle; its pdf has to be of the drawn triangle's area
(emissiveTrianglePdf). When the pdf let the table's area cancel instead, the skinned panel's light
samples carried a quarter of its light: with light samples alone (ptEstimator 1) the floor was a
quarter as bright, and with both techniques weighed (ptEstimator 0) somewhere between.

The script writes the two glTF scenes, renders each headless to a linear .hdr with no sky, and
compares the images' mean. Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/emitter_skinned.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import base64
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import read_hdr

IMAGE_SIZE = 256
SAMPLES    = 512
PANEL_Y    = 1.0   # the panel's height above the floor, metres
STRETCH    = 2.0   # the joint's scale along the panel's two axes: four times the area
TOLERANCE  = 0.02  # the skinned panel's floor against the plain one's
MIN_MEAN   = 0.02  # the plain panel's image must be at least this bright, or the floor is unlit
ESTIMATORS = {"both weighed": 0, "light samples alone": 1, "BSDF rays alone": 2}


def scene(skinned):
    blob = bytearray()
    views, accessors = [], []

    def add(values, fmt, count, comp, typ, target=None, bounds=None):
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(struct.pack("<" + fmt * len(values), *values))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob) - offset}
        if target:
            view["target"] = target
        views.append(view)
        accessor = {"bufferView": len(views) - 1, "componentType": comp, "count": count, "type": typ}
        if bounds:
            accessor["min"], accessor["max"] = bounds
        accessors.append(accessor)
        return len(accessors) - 1

    def quad(half, y, normal_y, joints):
        """A square of side 2 * half at height y in the XZ plane, facing +-Y."""
        pos, nrm = [], []
        for dx, dz in ((-half, -half), (half, -half), (half, half), (-half, half)):
            pos += [dx, y, dz]
            nrm += [0.0, normal_y, 0.0]
        idx  = [0, 2, 1, 0, 3, 2] if normal_y > 0 else [0, 1, 2, 0, 2, 3]
        attr = {"POSITION": add(pos, "f", 4, 5126, "VEC3", 34962, ([-half, y, -half], [half, y, half])),
                "NORMAL": add(nrm, "f", 4, 5126, "VEC3", 34962)}
        if joints:
            attr["JOINTS_0"]  = add([0, 0, 0, 0] * 4, "H", 4, 5123, "VEC4", 34962)
            attr["WEIGHTS_0"] = add([1.0, 0.0, 0.0, 0.0] * 4, "f", 4, 5126, "VEC4", 34962)
        return attr, add(idx, "I", len(idx), 5125, "SCALAR", 34963)

    floor_attr, floor_idx = quad(0.5, 0.0, 1.0, False)
    # the skinned panel is in its place already, at half the size: its joint, at the origin, only stretches it
    panel_attr, panel_idx = quad(0.5 / STRETCH if skinned else 0.5, PANEL_Y, -1.0, skinned)

    nodes = [
        {"name": "Floor", "mesh": 0, "scale": [6, 1, 6]},
        {"name": "Panel", "mesh": 1},
        # looking down at the floor beside the panel, from the side the panel's back faces
        {"name": "Camera", "camera": 0, "translation": [0, 3.2, 3.2], "rotation": [-0.3826834, 0, 0, 0.9238795]},
    ]
    gltf = {
        "asset": {"version": "2.0", "generator": "emitter_skinned.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": floor_attr, "indices": floor_idx, "material": 0}]},
                   {"primitives": [{"attributes": panel_attr, "indices": panel_idx, "material": 1}]}],
        "materials": [{"name": "Floor", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0.0,
                                                                  "roughnessFactor": 1.0}},
                      {"name": "Panel", "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1], "metallicFactor": 0.0},
                       "emissiveFactor": [1, 1, 1], "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 5.0}}}],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 1.0, "znear": 0.05, "zfar": 100}}],
    }
    if skinned:
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        nodes.append({"name": "Joint", "scale": [STRETCH, 1, STRETCH]})
        nodes[1]["skin"] = 0
        gltf["scenes"][0]["nodes"].append(3)
        gltf["skins"] = [{"joints": [3], "inverseBindMatrices": add([float(v) for v in identity], "f", 1, 5126, "MAT4")}]
    gltf["accessors"]   = accessors
    gltf["bufferViews"] = views
    gltf["buffers"] = [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}]
    return gltf


def render(exe, scene_path, output_path, estimator):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "3", "--plainColor", "0", "0", "0",
           "--dlssEnable", "0", "--ptEstimator", str(estimator), "--ptMaxDepth", "3",
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(output_path).exists():
        print(result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"render failed: {' '.join(cmd)}")


def mean(path):
    _, _, px = read_hdr(path)
    return sum(px) / len(px)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "emitter_skinned",
                        help="where the scenes and images go")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = {}
    for name, skinned in (("skinned", True), ("plain", False)):
        scenes[name] = args.out / f"{name}.gltf"
        scenes[name].write_text(json.dumps(scene(skinned)))

    failures = 0
    for estimator_name, estimator in ESTIMATORS.items():
        means = {}
        for name, path in scenes.items():
            image = args.out / f"estimator{estimator}_{name}.hdr"
            render(args.exe, path, image, estimator)
            means[name] = mean(image)
        ratio = means["skinned"] / means["plain"] if means["plain"] > 0 else float("nan")
        print(f"{estimator_name:20s} skinned panel {means['skinned']:.5f}, plain panel {means['plain']:.5f}: ratio {ratio:.4f}")
        if means["plain"] < MIN_MEAN:
            print(f"FAIL: {estimator_name}: the floor is not lit (mean {means['plain']:.5f})")
            failures += 1
        elif not math.isfinite(ratio) or abs(ratio - 1.0) > TOLERANCE:
            print(f"FAIL: {estimator_name}: the skinned panel lights the floor {ratio:.3f}x the plain one's (tolerance {TOLERANCE:.0%})")
            failures += 1
    print("PASS" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
