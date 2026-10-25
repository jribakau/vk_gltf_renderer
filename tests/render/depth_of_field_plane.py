#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: the path tracer's depth of field focuses on a plane.

The thin lens took the focal point at the focal distance along each camera ray, a sphere around the
eye: a wall facing the camera at exactly the focal distance was sharp in the middle of the frame and
blurred towards its corners, which a lens never does. The focal distance is measured along the view
axis (as autofocus measures it), so the point is now taken where the ray meets the plane at that
depth.

The script writes a glTF scene (an emissive checkerboard facing a wide-angle camera, no lights, no
environment), renders it headless to a linear .hdr with a wide aperture focused on the board, and
checks that the corners of the frame are as sharp as its centre. The same scene turned as a whole
(the camera yawed and pitched, still square to the board) must be as sharp: the camera's axis was
once read through the transposed view matrix, which only an unturned camera does not tell apart. A
render with the focus pulled in checks that the aperture blurs at all.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/depth_of_field_plane.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import base64
import math
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import TECHNIQUES, read_hdr

IMAGE_SIZE = 256
SAMPLES    = 128

DISTANCE   = 4.0   # the board's distance from the camera, along the view axis
YFOV       = 1.2   # wide, so the corner rays are far off the axis (about 44 degrees)
TILE       = 0.1   # about 5 pixels on screen
BOARD_HALF = 3.0   # the frame's half-width at the board is DISTANCE * tan(YFOV / 2) = 2.74
APERTURE   = 0.3 ** 2  # ptAperture is the lens radius squared: a blur of over two tiles where focus misses by 30%
YAW, PITCH = math.radians(40.0), math.radians(-25.0)  # the turned scene


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def board_buffer():
    """Two primitives of square tiles at z = 0 facing +Z: the white ones and the black ones."""
    tiles = int(2 * BOARD_HALF / TILE)
    groups = ([], [])
    for j in range(tiles):
        for i in range(tiles):
            x0, y0 = -BOARD_HALF + i * TILE, -BOARD_HALF + j * TILE
            groups[(i + j) % 2].append([(x0, y0), (x0 + TILE, y0), (x0 + TILE, y0 + TILE), (x0, y0 + TILE)])

    data = bytearray()
    views, accessors, primitives = [], [], []
    for material, quads in enumerate(groups):
        positions = [(x, y, 0.0) for quad in quads for x, y in quad]
        indices = [4 * q + k for q in range(len(quads)) for k in (0, 1, 2, 0, 2, 3)]
        for values, fmt in ((positions, "<3f"), ([(0.0, 0.0, 1.0)] * len(positions), "<3f")):
            offset = len(data)
            data += b"".join(struct.pack(fmt, *v) for v in values)
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data) - offset, "target": 34962})
        accessors.append({"bufferView": len(views) - 2, "componentType": 5126, "count": len(positions), "type": "VEC3",
                          "min": [-BOARD_HALF, -BOARD_HALF, 0.0], "max": [BOARD_HALF, BOARD_HALF, 0.0]})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "count": len(positions), "type": "VEC3"})
        offset = len(data)
        data += b"".join(struct.pack("<H", i) for i in indices)
        data += b"\0" * (-len(data) % 4)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": 2 * len(indices), "target": 34963})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"})
        primitives.append({"attributes": {"POSITION": len(accessors) - 3, "NORMAL": len(accessors) - 2},
                           "indices": len(accessors) - 1, "material": material})
    return bytes(data), views, accessors, primitives


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def quat_rotate(q, v):
    x, y, z, w = quat_mul(quat_mul(q, [v[0], v[1], v[2], 0.0]), [-q[0], -q[1], -q[2], q[3]])
    return [x, y, z]


def make_scene(turn=(0.0, 0.0, 0.0, 1.0)):
    """The board at the origin and the camera DISTANCE in front of it, both turned by `turn`."""
    data, views, accessors, primitives = board_buffer()
    black = {"metallicFactor": 0.0, "roughnessFactor": 1.0, "baseColorFactor": [0.0, 0.0, 0.0, 1.0]}
    return {
        "asset": {"version": "2.0", "generator": "depth_of_field_plane.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "Board", "mesh": 0, "rotation": list(turn)},
            {"name": "Camera", "camera": 0, "rotation": list(turn),
             "translation": quat_rotate(turn, [0.0, 0.0, DISTANCE])},
        ],
        "cameras": [{"type": "perspective", "perspective": {"yfov": YFOV, "znear": 0.1, "zfar": 100.0,
                                                            "aspectRatio": 1.0}}],
        "meshes": [{"primitives": primitives}],
        # The white tiles are their own light, so what the image shows is the lens and nothing else.
        "materials": [{"name": "White", "pbrMetallicRoughness": black, "emissiveFactor": [1.0, 1.0, 1.0]},
                      {"name": "Black", "pbrMetallicRoughness": black}],
        "buffers": [{"byteLength": len(data),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
        "bufferViews": views,
        "accessors": accessors,
    }


# ---------------------------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------------------------

def sharpness(px, width, x0, y0, size):
    """Contrast over a square: the standard deviation of the luminance relative to its mean. Sharp
    tiles are about 1; a blur wider than a tile washes them out towards 0."""
    values = []
    for y in range(y0, y0 + size):
        for x in range(x0, x0 + size):
            i = 3 * (y * width + x)
            values.append(0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2])
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5 / max(mean, 1e-9)


def regions(path):
    """Sharpness of the centre and of each corner."""
    width, height, px = read_hdr(path)
    size, margin = 48, 8
    centre = sharpness(px, width, (width - size) // 2, (height - size) // 2, size)
    corners = [sharpness(px, width, x, y, size)
               for x in (margin, width - margin - size) for y in (margin, height - margin - size)]
    return centre, corners


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------

def render(exe, scene_path, output_path, technique, focal_distance):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "2", "--renderSystem", "0",
           "--ptTechnique", str(technique),
           "--ptAutoFocus", "0", "--ptFocalDistance", str(focal_distance), "--ptAperture", str(APERTURE),
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "depth_of_field_plane",
                        help="directory for the generated scene and images")
    args = parser.parse_args()
    exe = args.exe.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    yaw   = [0.0, math.sin(YAW / 2), 0.0, math.cos(YAW / 2)]
    pitch = [math.sin(PITCH / 2), 0.0, 0.0, math.cos(PITCH / 2)]
    scenes = {"board": make_scene(), "turned": make_scene(quat_mul(yaw, pitch))}
    for name, gltf in scenes.items():
        (args.out / f"{name}.gltf").write_text(json.dumps(gltf, indent=1))

    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    for technique_name, technique in TECHNIQUES.items():
        results = {}
        for name, scene, focus in (("focused", "board", DISTANCE), ("turned", "turned", DISTANCE),
                                   ("near", "board", 0.6 * DISTANCE)):
            image_path = (args.out / f"{name}_{technique_name}.hdr").resolve()
            image_path.unlink(missing_ok=True)
            render(exe, (args.out / f"{scene}.gltf").resolve(), image_path, technique, focus)
            results[name] = regions(image_path)
            centre, corners = results[name]
            print(f"{technique_name:10s} {name:8s} centre {centre:.4f} corners " + " ".join(f"{c:.4f}" for c in corners))

        centre, corners = results["focused"]
        check(min(corners) >= 0.85 * centre,
              f"{technique_name}: focused on the board, its corners are as sharp as its centre "
              f"(worst corner {min(corners):.4f} vs centre {centre:.4f})")
        turned_centre, turned_corners = results["turned"]
        check(min([turned_centre] + turned_corners) >= 0.85 * centre,
              f"{technique_name}: turned, the camera is as sharp, centre and corners "
              f"(worst {min([turned_centre] + turned_corners):.4f} vs {centre:.4f})")
        near_centre = results["near"][0]
        check(near_centre <= 0.5 * centre,
              f"{technique_name}: focused short of the board, its centre blurs ({near_centre:.4f} vs {centre:.4f})")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
