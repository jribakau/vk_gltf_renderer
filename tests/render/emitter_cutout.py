#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: an emitter's see-through parts emit nothing, to light samples as to rays.

A floor is lit by one emissive panel hung above it, in two scenes: the panel as one quad with a
checkerboard alpha mask (alphaMode MASK, half of it see-through, emissive all over), and the panel
with the see-through squares cut out of its geometry. A ray meets the masked panel only where it is
opaque, so the floor must be lit the same in both. The light samples once took light from the holes
too (sampleEmissiveTriangle ignored opacity) and the masked panel lit the floor more, by an amount
that depended on how often the light was sampled against the BSDF.

The script writes the two glTF scenes, renders each headless to a linear .hdr with no sky, and
compares the images' mean. Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/emitter_cutout.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import base64
import json
import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

from volume_scatter_punctual import TECHNIQUES, read_hdr

IMAGE_SIZE = 256
SAMPLES    = 512
CHECKER    = 8     # squares along a side of the panel
PANEL_Y    = 1.0   # the panel's height above the floor, metres
TOLERANCE  = 0.01  # the masked panel's floor against the cut one's
MIN_MEAN   = 0.02  # the cut panel's (about 0.055) image must be at least this bright, or the floor is unlit


def png_rgba(size, pixel):
    """A size x size RGBA PNG, pixel(x, y) -> (r, g, b, a)."""
    raw = b"".join(b"\x00" + bytes(c for x in range(size) for c in pixel(x, y)) for y in range(size))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def opaque(i, j):
    return (i + j) % 2 == 0


def scene(masked):
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

    def quads(squares, normal_y):
        """Unit-square cells (x0, z0, size) at y = 0 in the XZ plane, facing +-Y, with UVs over [-0.5, 0.5]^2."""
        pos, nrm, uv, idx = [], [], [], []
        for x0, z0, s in squares:
            base = len(pos) // 3
            for dx, dz in ((0, 0), (s, 0), (s, s), (0, s)):
                pos += [x0 + dx, 0.0, z0 + dz]
                nrm += [0.0, normal_y, 0.0]
                uv += [x0 + dx + 0.5, z0 + dz + 0.5]
            # wound so the face looks along +-Y
            idx += [base, base + 2, base + 1, base, base + 3, base + 2] if normal_y > 0 else [base, base + 1, base + 2, base, base + 2, base + 3]
        p = add(pos, "f", len(pos) // 3, 5126, "VEC3", 34962,
                ([min(pos[0::3]), 0.0, min(pos[2::3])], [max(pos[0::3]), 0.0, max(pos[2::3])]))
        n = add(nrm, "f", len(nrm) // 3, 5126, "VEC3", 34962)
        t = add(uv, "f", len(uv) // 2, 5126, "VEC2", 34962)
        i = add(idx, "I", len(idx), 5125, "SCALAR", 34963)
        return {"POSITION": p, "NORMAL": n, "TEXCOORD_0": t}, i

    floor_attr, floor_idx = quads([(-0.5, -0.5, 1.0)], 1.0)
    cell = 1.0 / CHECKER
    if masked:
        panel_attr, panel_idx = quads([(-0.5, -0.5, 1.0)], -1.0)
    else:
        panel_attr, panel_idx = quads([(-0.5 + i * cell, -0.5 + j * cell, cell)
                                       for j in range(CHECKER) for i in range(CHECKER) if opaque(i, j)], -1.0)

    panel = {"name": "Panel", "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1], "metallicFactor": 0.0},
             "emissiveFactor": [1, 1, 1], "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 5.0}}}
    gltf = {
        "asset": {"version": "2.0", "generator": "emitter_cutout.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": [
            {"name": "Floor", "mesh": 0, "scale": [6, 1, 6]},
            {"name": "Panel", "mesh": 1, "translation": [0, PANEL_Y, 0]},
            # looking down at the floor beside the panel, from the side the panel's back faces
            {"name": "Camera", "camera": 0, "translation": [0, 3.2, 3.2], "rotation": [-0.3826834, 0, 0, 0.9238795]},
        ],
        "meshes": [{"primitives": [{"attributes": floor_attr, "indices": floor_idx, "material": 0}]},
                   {"primitives": [{"attributes": panel_attr, "indices": panel_idx, "material": 1}]}],
        "materials": [{"name": "Floor", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0.0,
                                                                  "roughnessFactor": 1.0}}, panel],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 1.0, "znear": 0.05, "zfar": 100}}],
        "accessors": accessors,
        "bufferViews": views,
    }
    if masked:
        size = 8 * CHECKER
        image = png_rgba(size, lambda x, y: (255, 255, 255, 255 if opaque(x * CHECKER // size, y * CHECKER // size) else 0))
        panel["alphaMode"] = "MASK"
        panel["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        gltf["images"] = [{"uri": "data:image/png;base64," + base64.b64encode(image).decode()}]
        gltf["samplers"] = [{"magFilter": 9728, "minFilter": 9728}]
        gltf["textures"] = [{"source": 0, "sampler": 0}]
    gltf["buffers"] = [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}]
    return gltf


def render(exe, scene_path, output_path, technique):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "3", "--plainColor", "0", "0", "0", "--renderSystem", "0",
           # the path tracer compiled from shaders/: a run keeps the exe's embedded build unless the
           # variant it asks for differs, and the optimal shader is one that does
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
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "emitter_cutout",
                        help="where the scenes and images go")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = {}
    for name, masked in (("masked", True), ("cut", False)):
        scenes[name] = args.out / f"{name}.gltf"
        scenes[name].write_text(json.dumps(scene(masked)))

    failures = 0
    for technique_name, technique in TECHNIQUES.items():
        means = {}
        for name, path in scenes.items():
            image = args.out / f"{technique_name}_{name}.hdr"
            render(args.exe, path, image, technique)
            means[name] = mean(image)
        ratio = means["masked"] / means["cut"]
        print(f"{technique_name:10s} masked panel {means['masked']:.5f}, cut panel {means['cut']:.5f}: ratio {ratio:.4f}")
        if means["cut"] < MIN_MEAN:
            # the panel faces away or the floor is unlit: the ratio would compare nothing
            print(f"FAIL: {technique_name}: the floor is not lit (mean {means['cut']:.5f})")
            failures += 1
        elif not math.isfinite(ratio) or abs(ratio - 1.0) > TOLERANCE:
            print(f"FAIL: {technique_name}: the masked panel lights the floor {ratio:.3f}x the cut one's (tolerance {TOLERANCE:.0%})")
            failures += 1
    print("PASS" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
