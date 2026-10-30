#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: the backdrop is what the camera sees, and hair doesn't show it.

A fibre event spends the hair's own budget and not a surface bounce, so a ray leaving a fibre still has
surfaceDepth 0. It was taken for the camera ray: reaching the sky, it replaced the path's light with the
solid or blurred backdrop, whole, so black hair took the colour of whatever was behind the figure, under
the same light, and the same scene with the Environment background looked right.

The scene is a ribbon of black fibres (FV_materials_hair, an absorption nothing gets through) filling the
middle of the image, lit by a plain environment, rendered in front of a red backdrop and a black one. The
environment lights the hair the same in both, so the script checks:
  - the corners show the backdrop, so it is in use,
  - the ribbon is as bright in front of either backdrop,
  - and it is as dark as a fibre that only reflects its Fresnel, about a tenth of the light around it,
with both trace techniques.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/hair_backdrop.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import base64
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import TECHNIQUES, read_hdr

IMAGE_SIZE = 128
SAMPLES    = 128
ENV        = 1.0    # the plain environment's radiance
TOLERANCE  = 0.02   # the ribbon in front of the red backdrop against the black one, in radiance
MAX_RIBBON = 0.25   # a black fibre reflects 7.5 to 14% around a level view, and paths between its sides add little


def scene():
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

    pos = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0]
    nrm = [0.0, 0.0, 1.0] * 4
    uv = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    attributes = {
        "POSITION": add(pos, "f", 4, 5126, "VEC3", 34962, ([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0])),
        "NORMAL": add(nrm, "f", 4, 5126, "VEC3", 34962),
        "TEXCOORD_0": add(uv, "f", 4, 5126, "VEC2", 34962),
    }
    indices = add([0, 1, 2, 0, 2, 3], "I", 6, 5125, "SCALAR", 34963)
    black = [40.0, 40.0, 40.0]
    return {
        "asset": {"version": "2.0", "generator": "hair_backdrop.py"},
        "extensionsUsed": ["FV_materials_hair"],
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "Ribbon", "mesh": 0}, {"name": "Camera", "camera": 0, "translation": [0, 0, 3]}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": indices, "material": 0}]}],
        "materials": [{
            "name": "Hair",
            "doubleSided": True,
            "pbrMetallicRoughness": {"baseColorFactor": [0.01, 0.01, 0.01, 1], "metallicFactor": 0.0, "roughnessFactor": 0.3},
            "extensions": {"FV_materials_hair": {"longitudinalRoughness": 0.3, "azimuthalRoughness": 0.3,
                                                 "rootAbsorption": black, "tipAbsorption": black}},
        }],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.6, "znear": 0.05, "zfar": 100}}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}],
    }


def render(exe, scene_path, output_path, technique, backdrop):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "3", "--plainColor", str(ENV), str(ENV), str(ENV), "--renderSystem", "0",
           "--useSolidBackground", "1", "--solidBackgroundColor", *[str(c) for c in backdrop],
           # the path tracer compiled from shaders/: a run keeps the exe's embedded build unless the
           # variant it asks for differs, and the optimal shader is one that does
           "--dlssEnable", "0", "--ptOptimalShader", "1",
           "--ptTechnique", str(technique), "--ptMaxDepth", "4",
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(output_path).exists():
        print(result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"render failed: {' '.join(cmd)}")


def region(path, x0, y0, x1, y1):
    """The mean RGB of a rectangle given in fractions of the image."""
    w, h, px = read_hdr(path)
    channels = len(px) // (w * h)
    total, count = [0.0, 0.0, 0.0], 0
    for y in range(int(y0 * h), int(y1 * h)):
        for x in range(int(x0 * w), int(x1 * w)):
            at = (y * w + x) * channels
            for c in range(3):
                total[c] += px[at + c]
            count += 1
    return [t / count for t in total]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "hair_backdrop",
                        help="where the scene and images go")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scene_path = args.out / "ribbon.gltf"
    scene_path.write_text(json.dumps(scene()))

    failures = 0
    for technique_name, technique in TECHNIQUES.items():
        ribbon, corner = {}, {}
        for name, backdrop in (("red", (1.0, 0.0, 0.0)), ("black", (0.0, 0.0, 0.0))):
            image = args.out / f"{technique_name}_{name}.hdr"
            render(args.exe, scene_path, image, technique, backdrop)
            ribbon[name] = region(image, 0.42, 0.42, 0.58, 0.58)
            corner[name] = region(image, 0.0, 0.0, 0.1, 0.1)
        print(f"{technique_name:10s} ribbon on red {[round(c, 4) for c in ribbon['red']]}, on black {[round(c, 4) for c in ribbon['black']]};"
              f" corner on red {[round(c, 3) for c in corner['red']]}")
        if corner["red"][0] < 0.9 or corner["red"][1] > 0.05:
            print(f"  FAIL: the corners don't show the red backdrop, so this tests nothing")
            failures += 1
        difference = abs(ribbon["red"][0] - ribbon["black"][0])
        if difference > TOLERANCE:
            print(f"  FAIL: the ribbon shows the backdrop behind it: red {difference:.4f} brighter on the red one (at most {TOLERANCE})")
            failures += 1
        if not 0.0 < ribbon["black"][1] < MAX_RIBBON * ENV:
            print(f"  FAIL: the ribbon's radiance {ribbon['black'][1]:.4f} isn't a black fibre's (above 0, at most {MAX_RIBBON})")
            failures += 1
    print("PASS" if not failures else f"FAIL: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
