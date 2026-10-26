#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: the rasterizer lets a directional light through a diffuse-transmitting surface.

singleLightContribution() drops a directional light below the hemisphere of the normal it is given;
point and spot lights it keeps. The raster light loop handed it the shading normal, so a leaf or a
curtain (KHR_materials_diffuse_transmission) lit from behind glowed under a point light and stayed
black under the sun. It now hands a surface that lets light through a normal facing the light, as
the path tracer's sampleLights() does.

The script writes a double-sided quad facing the camera, wholly diffuse-transmitting, lit from
behind by a sun or by a distant point light of the same irradiance, renders each with the
rasterizer to a linear .hdr, and checks that the two match.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/diffuse_transmission_sun.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import base64
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import read_hdr

IMAGE_SIZE = 256
FRAMES     = 16

CAMERA_Z       = 4.5
IRRADIANCE     = 3.0    # lux at the quad, from either light
POINT_DISTANCE = 100.0  # far enough that the point light's direction is the sun's across the quad


def quad_buffer():
    """A 2 x 2 quad in the XY plane facing +Z: 4 vertices, 6 indices."""
    positions = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)]
    normals   = [(0.0, 0.0, 1.0)] * 4
    indices   = [0, 1, 2, 0, 2, 3]
    data = b"".join(struct.pack("<3f", *p) for p in positions) + b"".join(struct.pack("<3f", *n) for n in normals)
    data += b"".join(struct.pack("<H", i) for i in indices)
    return data


def make_scene(light, translation):
    data = quad_buffer()
    # Lights shine along their node's -Z: a half turn about Y points them at +Z, through the quad
    # towards the camera.
    behind = [0.0, 1.0, 0.0, 0.0]
    return {
        "asset": {"version": "2.0", "generator": "diffuse_transmission_sun.py"},
        "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_diffuse_transmission"],
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": [
            {"name": "Leaf", "mesh": 0},
            {"name": "Camera", "camera": 0, "translation": [0.0, 0.0, CAMERA_Z]},
            {"name": "Light", "translation": translation, "rotation": behind,
             "extensions": {"KHR_lights_punctual": {"light": 0}}},
        ],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.75, "znear": 0.1, "zfar": 1000.0,
                                                            "aspectRatio": 1.0}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{
            "name": "Leaf",
            "doubleSided": True,
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                     "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_diffuse_transmission": {"diffuseTransmissionFactor": 1.0,
                                                                  "diffuseTransmissionColorFactor": [1.0, 1.0, 1.0]}},
        }],
        "buffers": [{"byteLength": len(data),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
            {"buffer": 0, "byteOffset": 48, "byteLength": 48, "target": 34962},
            {"buffer": 0, "byteOffset": 96, "byteLength": 12, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 0.0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "extensions": {"KHR_lights_punctual": {"lights": [light]}},
    }


SCENES = {
    "sun_behind":   make_scene({"type": "directional", "intensity": IRRADIANCE}, [0.0, 0.0, -5.0]),
    "point_behind": make_scene({"type": "point", "intensity": IRRADIANCE * POINT_DISTANCE ** 2},
                               [0.0, 0.0, -POINT_DISTANCE]),
}


def center_luminance(path):
    """Mean luminance over the central quarter of the image, which the quad fills."""
    width, height, px = read_hdr(path)
    total, count = 0.0, 0
    for y in range(3 * height // 8, 5 * height // 8):
        for x in range(3 * width // 8, 5 * width // 8):
            i = 3 * (y * width + x)
            total += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
            count += 1
    return total / count


def render(exe, scene_path, output_path):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "2", "--renderSystem", "1",
           "--frames", str(FRAMES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "diffuse_transmission_sun",
                        help="directory for the generated scenes and images")
    args = parser.parse_args()
    exe = args.exe.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    lum = {}
    for name, scene in SCENES.items():
        scene_path = args.out / f"{name}.gltf"
        scene_path.write_text(json.dumps(scene, indent=1))
        image_path = args.out / f"{name}.hdr"
        image_path.unlink(missing_ok=True)
        render(exe, scene_path.resolve(), image_path.resolve())
        lum[name] = center_luminance(image_path)
        print(f"{name:14s} mean luminance {lum[name]:.6f}")

    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    sun, point = lum["sun_behind"], lum["point_behind"]
    check(point > 1e-3, f"a point light behind the leaf shows through it ({point:.6f})")
    check(abs(sun - point) <= 0.05 * point, f"the sun behind it shows through as much ({sun:.6f} vs {point:.6f})")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
