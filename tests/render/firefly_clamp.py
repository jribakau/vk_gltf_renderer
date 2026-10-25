#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: the firefly clamp cuts indirect light only, contribution by contribution.

The clamp scaled a camera sample's whole path once its radiance passed the threshold, so one bright
bounce also cut the direct light of the same path, and a tight clamp dulled every catchlight. With
ptFireflyClampMode 1 (the default) a contribution is clamped on its own, by its largest channel, and
only when the path before it has been rough (a diffuse or glossy bounce): the light of the first
surface, and of surfaces seen through a mirror or a cornea, is never clamped. A scatter walk entered
from such a surface is no bounce, so the skin's light, which leaves where the walk exits, is direct.

The script writes small glTF scenes (no environment), renders each one headless to a linear .hdr with
a tight clamp and with none, and checks:
  - a smooth sphere's highlight from a point light keeps its brightness under the tight clamp, and
    loses it with ptFireflyClampMode 0 (the whole path, as before),
  - a floor lit only by the light a wall bounces onto it is darkened by the tight clamp,
  - a front-lit skin slab keeps its brightness under the tight clamp, with the diffuse exit
    (ptVolumeDiffuseExit) and without it, where the walk gathers light inside and reflects off the
    inside of its boundary, which is no bounce either.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/firefly_clamp.py --exe _bin/Release/vk_gltf_renderer.exe
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
from volume_walk_exit import SLAB_HALF, box_faces, skin

IMAGE_SIZE = 192
SAMPLES    = 256
TIGHT      = 0.01  # far below every contribution in these scenes
OFF        = 1e9   # ptFireflyClampStill: what an accumulation clamps at (0 would fall back to ptFireflyClamp)

DOWN = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]  # a -90 degree turn about X: -Z turns to -Y


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def sphere(radius, rings=48, segments=96):
    """A UV sphere as (positions, normals, indices)."""
    positions, normals, indices = [], [], []
    for r in range(rings + 1):
        theta = math.pi * r / rings
        for s in range(segments + 1):
            phi = 2.0 * math.pi * s / segments
            n = [math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)]
            normals.append(n)
            positions.append([radius * c for c in n])
    for r in range(rings):
        for s in range(segments):
            a = r * (segments + 1) + s
            b = a + segments + 1
            indices += [a, a + 1, b, b, a + 1, b + 1]
    return positions, normals, indices


def quad(corner, u, v, normal):
    """A parallelogram from `corner` along `u` and `v`, counter-clockwise seen from `normal`'s side."""
    p = [corner, [corner[i] + u[i] for i in range(3)], [corner[i] + u[i] + v[i] for i in range(3)],
         [corner[i] + v[i] for i in range(3)]]
    return p, [normal] * 4, [0, 1, 2, 0, 2, 3]


def make_gltf(primitives, materials, camera, lights):
    """One mesh of (positions, normals, indices, material) primitives; `camera` and each light are
    (translation, rotation) and (light, translation, rotation)."""
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
        data += b"".join(struct.pack("<I", i) for i in indices)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": 4 * len(indices), "target": 34963})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"})
        prims.append({"attributes": attrs, "indices": len(accessors) - 1, "material": material})

    nodes = [{"name": "Mesh", "mesh": 0},
             {"name": "Camera", "camera": 0, "translation": camera[0], "rotation": camera[1]}]
    for i, (_, translation, rotation) in enumerate(lights):
        nodes.append({"name": f"Light{i}", "translation": translation, "rotation": rotation,
                      "extensions": {"KHR_lights_punctual": {"light": i}}})
    return {
        "asset": {"version": "2.0", "generator": "firefly_clamp.py"},
        "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_transmission", "KHR_materials_volume",
                           "KHR_materials_scatter"],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.6, "znear": 0.05, "zfar": 100.0,
                                                            "aspectRatio": 1.0}}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "buffers": [{"byteLength": len(data),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(data)).decode()}],
        "bufferViews": views,
        "accessors": accessors,
        "extensions": {"KHR_lights_punctual": {"lights": [light for light, _, _ in lights]}},
    }


def diffuse(name, color):
    return {"name": name, "pbrMetallicRoughness": {"baseColorFactor": color + [1.0], "metallicFactor": 0.0,
                                                   "roughnessFactor": 1.0}}


def catchlight_scene():
    """A black, smooth dielectric sphere: all it shows is the point light's highlight."""
    glossy = {"name": "Glossy", "pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.0, 0.0, 1.0],
                                                         "metallicFactor": 0.0, "roughnessFactor": 0.1}}
    return make_gltf([sphere(1.0) + (0,)], [glossy], ([0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 1.0]),
                     [({"type": "point", "intensity": 200.0}, [1.5, 1.5, 3.0], [0.0, 0.0, 0.0, 1.0])])


def bounce_scene():
    """A floor seen from straight above, lit only by what a white wall bounces onto it: the spot
    aims at the wall in a cone that misses the floor in view."""
    floor = quad([-3.0, 0.0, 3.0], [6.0, 0.0, 0.0], [0.0, 0.0, -6.0], [0.0, 1.0, 0.0])
    wall = quad([-3.0, 0.0, -2.0], [6.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0])
    to_wall = [0.0, -1.0, -3.5]  # from the spot at (0, 2.5, 1.5) to the wall's middle
    yaw_pitch = math.atan2(-to_wall[1], -to_wall[2])  # a turn about X from -Z towards the wall
    spot_rot = [-math.sin(yaw_pitch / 2), 0.0, 0.0, math.cos(yaw_pitch / 2)]
    spot = {"type": "spot", "intensity": 400.0, "spot": {"innerConeAngle": 0.1, "outerConeAngle": 0.2}}
    return make_gltf([floor + (0,), wall + (0,)], [diffuse("White", [0.8, 0.8, 0.8])],
                     ([0.0, 4.0, 1.5], DOWN), [(spot, [0.0, 2.5, 1.5], spot_rot)])


def skin_scene():
    """volume_walk_exit's skin slab, lit from the front: the light the camera sees has walked in the
    skin and left where it went in."""
    primitives = [(p, n, i, 0) for _, _, p, n, i in box_faces(SLAB_HALF)]
    return make_gltf(primitives, [skin("Skin", [0.8, 0.8, 0.8])], ([0.0, 0.0, 4.5], [0.0, 0.0, 0.0, 1.0]),
                     [({"type": "point", "intensity": 60.0}, [0.6, 0.6, 3.0], [0.0, 0.0, 0.0, 1.0])])


SCENES = {"catchlight": catchlight_scene(), "bounce": bounce_scene(), "skin": skin_scene()}


def mean_luminance(path, fraction=1.0):
    """Mean luminance over the central `fraction` of the image (by side)."""
    width, height, px = read_hdr(path)
    x0, x1 = int(width * (1 - fraction) / 2), int(width * (1 + fraction) / 2)
    y0, y1 = int(height * (1 - fraction) / 2), int(height * (1 + fraction) / 2)
    total, count = 0.0, 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = 3 * (y * width + x)
            total += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
            count += 1
    return total / count


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------

def render(exe, scene_path, output_path, technique, clamp, mode=None, diffuse_exit=True):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "2", "--renderSystem", "0",
           "--ptTechnique", str(technique), "--ptShadowTransmission", "1",
           "--ptVolumeDiffuseExit", "1" if diffuse_exit else "0",
           # The load recalibrates ptFireflyClamp; an accumulation's frames clamp at the still's value
           "--ptFireflyClampStill", str(clamp),
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    if mode is not None:
        cmd += ["--ptFireflyClampMode", str(mode)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "firefly_clamp",
                        help="directory for the generated scenes and images")
    parser.add_argument("--no-mode", action="store_true",
                        help="skip the ptFireflyClampMode 0 render (a renderer before the parameter rejects it)")
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
        runs = [(name, clamp, None, True) for name in SCENES for clamp in ("tight", "off")]
        runs += [("skin", clamp, None, False) for clamp in ("tight", "off")]
        if not args.no_mode:
            runs.append(("catchlight", "tight", 0, True))
        for name, clamp, mode, diffuse_exit in runs:
            key = f"{name}_{clamp}" + ("" if mode is None else f"_mode{mode}") + ("" if diffuse_exit else "_noexit")
            image_path = (args.out / f"{key}_{technique_name}.hdr").resolve()
            image_path.unlink(missing_ok=True)
            render(exe, (args.out / f"{name}.gltf").resolve(), image_path, technique, TIGHT if clamp == "tight" else OFF, mode,
                   diffuse_exit)
            lum[key] = mean_luminance(image_path, 0.5 if name == "skin" else 1.0)
            print(f"{technique_name:10s} {key:22s} mean luminance {lum[key]:.6f}")

        def ratio(a, b):
            return lum[a] / max(lum[b], 1e-12)

        r = ratio("catchlight_tight", "catchlight_off")
        check(r >= 0.98, f"{technique_name}: a highlight is direct light, which the clamp leaves alone ({r:.3f} of unclamped)")
        if not args.no_mode:
            r0 = ratio("catchlight_tight_mode0", "catchlight_off")
            check(r0 <= 0.5, f"{technique_name}: ptFireflyClampMode 0 clamps the whole path, highlight and all ({r0:.3f})")
        r = ratio("bounce_tight", "bounce_off")
        check(lum["bounce_off"] > 1e-4 and r <= 0.9,
              f"{technique_name}: light bounced off a wall is indirect and clamped ({r:.3f} of unclamped, {lum['bounce_off']:.5f})")
        r = ratio("skin_tight", "skin_off")
        check(r >= 0.95, f"{technique_name}: the light leaving a skin walk is direct, as is the surface's ({r:.3f} of unclamped)")
        r = ratio("skin_tight_noexit", "skin_off_noexit")
        check(r >= 0.95, f"{technique_name}: without the diffuse exit too, a walk's light under a surface seen directly "
                         f"is direct ({r:.3f} of unclamped)")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
