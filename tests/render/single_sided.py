#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: a single-sided surface is solid to light from both sides; only the camera culls it.

A floor under a white sky, with a quad hung over it whose front faces the sky. A ray from the floor
meets the quad's back. Shadow rays have always stopped there; rays leaving the floor by its BSDF once
passed through (culled) and saw the sky, so the two estimators of the same light disagreed and the
floor under the quad was lit by the BSDF's share of the sky.

Three comparisons, each per ray technique, from the linear .hdr's mean:
  - Looking down at the floor from under the quad, a single-sided quad and a double-sided one: light
    must treat them alike, so the two floors match. This is where a leak shows.
  - Looking up at the quad, the same two: the camera culls the single-sided quad's back and sees the
    double-sided one, so the two must differ (glTF's single-sidedness is kept).
  - A one-sided emitter facing away from the floor, no sky: a ray meeting its back takes no light from
    it, as the light sampler never gives any, so the floor stays black.

The script writes the glTF scenes and renders each headless. Needs a GPU and a built renderer; uses
only the Python standard library.

  python tests/render/single_sided.py --exe _bin/Release/vk_gltf_renderer.exe
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

from volume_scatter_punctual import TECHNIQUES, read_hdr

IMAGE_SIZE = 256
SAMPLES    = 512
QUAD_Y     = 0.5   # the quad's height above the floor, metres
TOLERANCE  = 0.01  # the single-sided quad's image against the double-sided one's
MIN_APART  = 0.05  # from below, how far apart they must be
MAX_UNLIT  = 1e-4  # the floor's mean under an emitter facing away


def scene(double_sided, camera, emitter=False):
    blob = bytearray()
    views, accessors = [], []

    def add(values, fmt, count, comp, typ, target, bounds=None):
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(struct.pack("<" + fmt * len(values), *values))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob) - offset, "target": target})
        accessor = {"bufferView": len(views) - 1, "componentType": comp, "count": count, "type": typ}
        if bounds:
            accessor["min"], accessor["max"] = bounds
        accessors.append(accessor)
        return len(accessors) - 1

    # a unit quad in the XZ plane, its front facing +Y
    pos = [-0.5, 0.0, -0.5, 0.5, 0.0, -0.5, 0.5, 0.0, 0.5, -0.5, 0.0, 0.5]
    nrm = [0.0, 1.0, 0.0] * 4
    p = add(pos, "f", 4, 5126, "VEC3", 34962, ([-0.5, 0.0, -0.5], [0.5, 0.0, 0.5]))
    n = add(nrm, "f", 4, 5126, "VEC3", 34962)
    i = add([0, 2, 1, 0, 3, 2], "I", 6, 5125, "SCALAR", 34963)
    attributes = {"POSITION": p, "NORMAL": n}

    quad = {"name": "Quad", "doubleSided": double_sided,
            "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    if emitter:
        quad["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 0, 1]
        quad["emissiveFactor"] = [1, 1, 1]
        quad["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": 5.0}}
    # under the quad looking straight down at the floor, or beside it looking up at its back
    placed = {"floor": {"translation": [0, 0.3, 0], "rotation": [-0.7071068, 0, 0, 0.7071068]},
              "up": {"translation": [0, 0.05, 1.0], "rotation": [0.2588190, 0, 0, 0.9659258]}}[camera]
    gltf = {
        "asset": {"version": "2.0", "generator": "single_sided.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": [
            {"name": "Floor", "mesh": 0, "scale": [8, 1, 8]},
            {"name": "Quad", "mesh": 1, "translation": [0, QUAD_Y, 0]},
            {"name": "Camera", "camera": 0, **placed},
        ],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": i, "material": 0}]},
                   {"primitives": [{"attributes": attributes, "indices": i, "material": 1}]}],
        "materials": [{"name": "Floor", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0.0,
                                                                  "roughnessFactor": 1.0}}, quad],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 1.0, "znear": 0.01, "zfar": 100}}],
        "accessors": accessors,
        "bufferViews": views,
    }
    if emitter:
        gltf["extensionsUsed"] = ["KHR_materials_emissive_strength"]
    gltf["buffers"] = [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}]
    return gltf


def render(exe, scene_path, output_path, technique, sky):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "3", "--plainColor", str(sky), str(sky), str(sky), "--renderSystem", "0",
           # the path tracer compiled from shaders/, as emitter_cutout.py
           "--dlssEnable", "0", "--ptOptimalShader", "1",
           "--ptTechnique", str(technique), "--ptMaxDepth", "3",
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
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "single_sided", help="where the scenes and images go")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def rendered(name, technique_name, technique, sky, **kwargs):
        path = args.out / f"{name}.gltf"
        path.write_text(json.dumps(scene(**kwargs)))
        image = args.out / f"{technique_name}_{name}.hdr"
        render(args.exe, path, image, technique, sky)
        return mean(image)

    failures = 0
    for technique_name, technique in TECHNIQUES.items():
        one = rendered("floor_single", technique_name, technique, 1, double_sided=False, camera="floor")
        two = rendered("floor_double", technique_name, technique, 1, double_sided=True, camera="floor")
        ratio = one / two
        print(f"{technique_name:10s} the floor under the quad: single-sided {one:.5f}, double-sided {two:.5f}: ratio {ratio:.4f}")
        if not math.isfinite(ratio) or abs(ratio - 1.0) > TOLERANCE:
            print(f"FAIL: {technique_name}: light treats the two quads differently, {ratio:.3f}x (tolerance {TOLERANCE:.0%})")
            failures += 1

        one = rendered("up_single", technique_name, technique, 1, double_sided=False, camera="up")
        two = rendered("up_double", technique_name, technique, 1, double_sided=True, camera="up")
        apart = abs(one / two - 1.0)
        print(f"{technique_name:10s} looking up at it: single-sided {one:.5f}, double-sided {two:.5f}: {apart:.1%} apart")
        if apart < MIN_APART:
            print(f"FAIL: {technique_name}: the camera sees a single-sided quad's back")
            failures += 1

        unlit = rendered("emitter_away", technique_name, technique, 0, double_sided=False, camera="floor", emitter=True)
        print(f"{technique_name:10s} the floor under an emitter facing away: mean {unlit:.6f}")
        if unlit > MAX_UNLIT:
            print(f"FAIL: {technique_name}: a one-sided emitter lights what its back faces")
            failures += 1
    print("PASS" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
