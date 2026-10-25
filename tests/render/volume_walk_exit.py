#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: a scatter walk's diffuse exit divides out the albedo of the medium it walked in.

With ptVolumeDiffuseExit a walk leaves the surface it reaches from inside as a Lambertian point in
that surface's base colour divided by the medium's multiple-scattering albedo. The divisor was read
from the surface the walk reached, not from the medium it was in, so:
  - a walk that met the back of a surface with no medium (a lash card under a lid, two-sided cloth)
    was divided by the 0.05 floor: an exit twenty times brighter than its colour,
  - a walk that left through a neighbouring material (face into lips) was divided by that material's
    albedo instead of its own.

The script writes small glTF scenes (a skin-like KHR_materials_scatter slab, back-lit by a point
light, seen from the front, no environment), renders each one headless to a linear .hdr, and checks:
  - a white two-sided quad with no medium, inside the slab, does not make it brighter,
  - giving the slab's hidden faces a material of another albedo (and the same colour) leaves the
    image as it was.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/volume_walk_exit.py --exe _bin/Release/vk_gltf_renderer.exe
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

IMAGE_SIZE = 256
SAMPLES    = 512

CAMERA_Z     = 4.5
LIGHT_Z      = -3.0  # behind the slab: what the camera sees is light that walked through it
SLAB_HALF    = (1.0, 1.0, 0.3)
QUAD_HALF    = 0.8


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def box_faces(half):
    """The six faces of a box as (axis, sign, positions, normals, indices), counter-clockwise from outside."""
    faces = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = [0.0, 0.0, 0.0]
            n[axis] = sign
            u, v = (axis + 1) % 3, (axis + 2) % 3
            positions = []
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[u], p[v] = sign * half[axis], du * half[u], dv * half[v]
                positions.append(p)
            quad = (0, 1, 2, 0, 2, 3) if sign > 0 else (0, 2, 1, 0, 3, 2)
            faces.append((axis, sign, positions, [n] * 4, list(quad)))
    return faces


def make_gltf(primitives, materials, generator):
    """One mesh of the given (positions, normals, indices, material) primitives, a camera on +Z and
    a point light behind the origin."""
    data = bytearray()
    views, accessors, prims = [], [], []
    for positions, normals, indices, material in primitives:
        attrs = {}
        for name, values in (("POSITION", positions), ("NORMAL", normals)):
            offset = len(data)
            data += b"".join(struct.pack("<3f", *p) for p in values)
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data) - offset, "target": 34962})
            accessor = {"bufferView": len(views) - 1, "componentType": 5126, "count": len(values), "type": "VEC3"}
            if name == "POSITION":
                accessor["min"] = [min(p[i] for p in values) for i in range(3)]
                accessor["max"] = [max(p[i] for p in values) for i in range(3)]
            accessors.append(accessor)
            attrs[name] = len(accessors) - 1
        offset = len(data)
        data += b"".join(struct.pack("<H", i) for i in indices)
        data += b"\0" * (-len(data) % 4)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": 2 * len(indices), "target": 34963})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"})
        prims.append({"attributes": attrs, "indices": len(accessors) - 1, "material": material})

    return {
        "asset": {"version": "2.0", "generator": generator},
        "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_transmission", "KHR_materials_volume",
                           "KHR_materials_scatter"],
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": [
            {"name": "Slab", "mesh": 0},
            {"name": "Camera", "camera": 0, "translation": [0.0, 0.0, CAMERA_Z]},
            {"name": "Light", "translation": [0.0, 0.0, LIGHT_Z], "extensions": {"KHR_lights_punctual": {"light": 0}}},
        ],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.5, "znear": 0.1, "zfar": 100.0,
                                                            "aspectRatio": 1.0}}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "buffers": [{"byteLength": len(data),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(data)).decode()}],
        "bufferViews": views,
        "accessors": accessors,
        "extensions": {"KHR_lights_punctual": {"lights": [{"type": "point", "intensity": 60.0}]}},
    }


def skin(name, multiscatter):
    """A dense, skin-like scattering medium behind a rough dielectric boundary. Dense enough that a
    straight shadow ray through the slab carries next to nothing: the light the camera sees has
    walked through it and left through the back face."""
    return {
        "name": name,
        "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.6, 0.5, 1.0], "metallicFactor": 0.0,
                                 "roughnessFactor": 0.5},
        "extensions": {
            "KHR_materials_transmission": {"transmissionFactor": 1.0},
            "KHR_materials_volume": {"thicknessFactor": 1.0, "attenuationDistance": 0.05,
                                     "attenuationColor": [0.5, 0.5, 0.5]},
            "KHR_materials_scatter": {"scatterStrengthFactor": 1.0, "multiscatterColorFactor": multiscatter,
                                      "scatterAnisotropy": 0.0},
        },
    }


# A white, opaque, two-sided surface with no medium: what a lash card or a two-sided garment is to a
# walk that meets it from behind.
CARD = {"name": "Card", "doubleSided": True,
        "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                 "roughnessFactor": 1.0}}


def slab_scene(hidden_material=0, card=False):
    """The slab: its front face (+Z, the one the camera sees) material 0, the other five
    `hidden_material`; with `card`, a quad of material 2 across its middle, facing the light, so a
    walk from the front meets its back."""
    primitives = []
    for axis, sign, positions, normals, indices in box_faces(SLAB_HALF):
        front = axis == 2 and sign > 0
        primitives.append((positions, normals, indices, 0 if front else hidden_material))
    if card:
        h = QUAD_HALF
        positions = [[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]]
        primitives.append((positions, [[0.0, 0.0, -1.0]] * 4, [0, 2, 1, 0, 3, 2], 2))  # faces -Z
    materials = [skin("Face", [0.8, 0.8, 0.8]), skin("Lips", [0.2, 0.2, 0.2]), CARD]
    return make_gltf(primitives, materials, "volume_walk_exit.py")


SCENES = {
    "plain":         slab_scene(),
    "card":          slab_scene(card=True),
    "other_albedo":  slab_scene(hidden_material=1),
}


def center_luminance(path):
    """Mean luminance over the central quarter of the image, inside the slab's outline."""
    width, height, px = read_hdr(path)
    total, count = 0.0, 0
    for y in range(3 * height // 8, 5 * height // 8):
        for x in range(3 * width // 8, 5 * width // 8):
            i = 3 * (y * width + x)
            total += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
            count += 1
    return total / count


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------

def render(exe, scene_path, output_path, technique):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "2", "--renderSystem", "0",
           "--ptTechnique", str(technique), "--ptShadowTransmission", "1",
           "--ptVolumeDiffuseExit", "1",
           # The bright exits this looks for are what a firefly clamp cuts: the load calibrates
           # ptFireflyClamp, and an accumulation uses the still's clamp instead when it is set.
           "--ptFireflyClampStill", "1e9",
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "volume_walk_exit",
                        help="directory for the generated scenes and images")
    args = parser.parse_args()
    exe = args.exe.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, gltf in SCENES.items():
        (args.out / f"{name}.gltf").write_text(json.dumps(gltf, indent=1))

    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    for technique_name, technique in TECHNIQUES.items():
        lum = {}
        for name in SCENES:
            image_path = args.out / f"{name}_{technique_name}.hdr"
            image_path.unlink(missing_ok=True)
            render(exe, (args.out / f"{name}.gltf").resolve(), image_path.resolve(), technique)
            lum[name] = center_luminance(image_path)
            print(f"{technique_name:10s} {name:14s} mean luminance {lum[name]:.6f}")

        plain = lum["plain"]
        check(plain > 1e-4, f"{technique_name}: the back-lit slab shows light through it ({plain:.6f})")
        # The card blocks some of the light and leaves the rest; it adds none. Dividing by the floor
        # made it several times brighter than the slab without it.
        check(lum["card"] <= 1.1 * plain,
              f"{technique_name}: a card with no medium inside the slab adds no light ({lum['card']:.6f} vs {plain:.6f})")
        # Walks enter through the front face only, so every exit carries the front material's medium.
        check(abs(lum["other_albedo"] - plain) <= 0.05 * plain,
              f"{technique_name}: hidden faces of another albedo leave the image as it was "
              f"({lum['other_albedo']:.6f} vs {plain:.6f})")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
