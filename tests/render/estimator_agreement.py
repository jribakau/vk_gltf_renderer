#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: light samples alone, BSDF rays alone and both (MIS) give the same image.

The path tracer reaches most light two ways, by a light sample and by a BSDF ray that hits the light,
and weighs the two by the power heuristic (ptEstimator 0). With ptEstimator 1 a light sample counts
every direction both reach, with 2 the BSDF ray does (shaders/light_pick.h.slang). Each is a whole
estimator on its own, so the three agree unless a pdf is wrong:
  - a light sampler's own (its radiance over a pdf that isn't the density it draws by): light samples
    alone split from BSDF rays alone;
  - a BSDF sampler's own: the same, the other way;
  - one side's account of the other's pdf (the light pick's, the tree's climb, a table's): MIS alone
    moves, as its weights no longer sum to one.
Weighing-only checks (one MIS against another) can't see the first two.

The scene has a light of each kind a BSDF ray can reach: an HDR dome with a bright window (the
environment's table and the compensated one), an emissive panel, 40 small emitters (the light tree,
and the table by power with --ptLightTree 0), a sized point light and a directional light the size of
a small sun. A floor, a diffuse sphere and a rough metal one see them. Each estimator renders it at
several seeds; the test compares the image means and the quadrants' between estimators against the
seeds' spread.

  python tests/render/estimator_agreement.py --exe _bin/Release/vk_gltf_renderer.exe
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

IMAGE_SIZE = 192
SEEDS      = 8
SAMPLES    = 1024  # a seed's; SEEDS x SAMPLES in all, an estimator
MAX_DEPTH  = 4
SIGMAS     = 4.5   # a difference this many standard errors apart fails (3 pairs x 5 regions)
ESTIMATORS = {"mis": 0, "light": 1, "bsdf": 2}
CONFIGS    = {"tree": [], "table": ["--ptLightTree", "0"]}


# ---------------------------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------------------------

def env_hdr(path, width=128, height=64):
    """An equirectangular dome: a blue-grey sky, darker below the horizon, and a bright window, as a
    flat (not run-length) Radiance file."""
    def rgbe(r, g, b):
        m = max(r, g, b)
        if m < 1e-32:
            return bytes((0, 0, 0, 0))
        e = math.frexp(m)[1]
        s = 256.0 / (2.0 ** e)
        return bytes((int(r * s), int(g * s), int(b * s), e + 128))

    rows = []
    for y in range(height):
        elev = math.pi * (0.5 - (y + 0.5) / height)  # +pi/2 at the top row
        row = bytearray()
        for x in range(width):
            if elev > 0.0:
                t = math.sin(elev)
                c = (0.25 + 0.2 * t, 0.3 + 0.25 * t, 0.4 + 0.35 * t)
            else:
                c = (0.08, 0.07, 0.06)
            if 20 <= x < 28 and 12 <= y < 17:  # the window, about 30 degrees up
                c = (30.0, 27.0, 24.0)
            row += rgbe(*c)
        rows.append(bytes(row))
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode()
    Path(path).write_bytes(header + b"".join(rows))


def uv_sphere(radius, stacks=16, slices=32):
    pos, nrm, idx = [], [], []
    for i in range(stacks + 1):
        v = math.pi * i / stacks
        for j in range(slices + 1):
            u = 2.0 * math.pi * j / slices
            n = (math.sin(v) * math.cos(u), math.cos(v), math.sin(v) * math.sin(u))
            nrm += n
            pos += [radius * c for c in n]
    for i in range(stacks):
        for j in range(slices):
            a, b = i * (slices + 1) + j, (i + 1) * (slices + 1) + j
            idx += [a, a + 1, b, b, a + 1, b + 1]
    return pos, nrm, idx


def quad(w, h, normal):
    """A w x h quad centred on the origin, in the plane its axis-aligned `normal` faces."""
    hw, hh = 0.5 * w, 0.5 * h
    if normal == "+y":
        pos = [-hw, 0, -hh, hw, 0, -hh, hw, 0, hh, -hw, 0, hh]
        n = (0, 1, 0)
        idx = [0, 2, 1, 0, 3, 2]
    elif normal == "-y":
        pos = [-hw, 0, -hh, hw, 0, -hh, hw, 0, hh, -hw, 0, hh]
        n = (0, -1, 0)
        idx = [0, 1, 2, 0, 2, 3]
    else:  # +z
        pos = [-hw, -hh, 0, hw, -hh, 0, hw, hh, 0, -hw, hh, 0]
        n = (0, 0, 1)
        idx = [0, 1, 2, 0, 2, 3]
    return pos, list(n) * 4, idx


def scene():
    blob = bytearray()
    views, accessors, meshes, nodes, materials = [], [], [], [], []

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

    def mesh(pos, nrm, idx, material):
        xs, ys, zs = pos[0::3], pos[1::3], pos[2::3]
        p = add(pos, "f", len(pos) // 3, 5126, "VEC3", 34962, ([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]))
        n = add(nrm, "f", len(nrm) // 3, 5126, "VEC3", 34962)
        i = add(idx, "I", len(idx), 5125, "SCALAR", 34963)
        meshes.append({"primitives": [{"attributes": {"POSITION": p, "NORMAL": n}, "indices": i, "material": material}]})
        return len(meshes) - 1

    def material(name, base=(0.8, 0.8, 0.8), metallic=0.0, roughness=1.0, emissive=None, strength=1.0):
        m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [*base, 1.0], "metallicFactor": metallic, "roughnessFactor": roughness}}
        if emissive:
            m["emissiveFactor"] = list(emissive)
            m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": strength}}
        materials.append(m)
        return len(materials) - 1

    def node(name, m, t, r=None):
        nodes.append({"name": name, "mesh": m, "translation": list(t), **({"rotation": list(r)} if r else {})})

    floor, wall = material("Floor", (0.7, 0.7, 0.7)), material("Wall", (0.6, 0.55, 0.5))
    node("Floor", mesh(*quad(8, 8, "+y"), floor), (0, 0, 0))
    node("Wall", mesh(*quad(8, 4, "+z"), wall), (0, 2, -2.5))
    node("Diffuse sphere", mesh(*uv_sphere(0.5), material("Diffuse", (0.75, 0.3, 0.25))), (-0.9, 0.5, -0.4))
    node("Metal sphere", mesh(*uv_sphere(0.5), material("Metal", (0.9, 0.85, 0.7), metallic=1.0, roughness=0.3)), (0.9, 0.5, -0.4))
    # an emissive panel overhead, facing down
    node("Panel", mesh(*quad(1.2, 0.6, "-y"), material("Panel", emissive=(1.0, 0.9, 0.8), strength=4.0)), (0, 2.6, 0.2))
    # 40 small emitters on the wall, each its own mesh and node: 40 emitters for the tree
    for k in range(40):
        i, j = k % 8, k // 8
        hue = (k * 0.618) % 1.0
        colour = (0.5 + 0.5 * math.cos(2 * math.pi * hue), 0.5 + 0.5 * math.cos(2 * math.pi * (hue - 1 / 3)),
                  0.5 + 0.5 * math.cos(2 * math.pi * (hue - 2 / 3)))
        m = mesh(*quad(0.18, 0.18, "+z"), material(f"Emitter {k}", emissive=colour, strength=6.0 + 4.0 * (k % 3)))
        node(f"Emitter {k}", m, (-2.1 + 0.6 * i, 0.6 + 0.5 * j, -2.49))
    # a sized point light and a directional one about 4 degrees across (radius in km at the sun's distance)
    nodes.append({"name": "Sphere light", "translation": [1.6, 1.9, 1.2], "extensions": {"KHR_lights_punctual": {"light": 0}}})
    nodes.append({"name": "Sun", "rotation": [-0.3535534, 0.3535534, 0.1464466, 0.8535534],
                  "extensions": {"KHR_lights_punctual": {"light": 1}}})
    nodes.append({"name": "Camera", "camera": 0, "translation": [0, 1.5, 3.6], "rotation": [-0.1305262, 0, 0, 0.9914449]})
    gltf = {
        "asset": {"version": "2.0", "generator": "estimator_agreement.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength", "KHR_lights_punctual"],
        "extensions": {"KHR_lights_punctual": {"lights": [
            {"type": "point", "color": [1.0, 0.95, 0.9], "intensity": 12.0, "extras": {"radius": 0.15}},
            {"type": "directional", "color": [1.0, 0.97, 0.9], "intensity": 2.0, "extras": {"radius": 5.0e6}},
        ]}},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.9, "znear": 0.05, "zfar": 100}}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}],
    }
    return gltf


# ---------------------------------------------------------------------------------------------
# Rendering and the comparison
# ---------------------------------------------------------------------------------------------

def read_pfm(path):
    """(width, height, [r, g, b, ...]) from an RGB float map, rows bottom first."""
    data = Path(path).read_bytes()
    parts, at = [], 0
    while len(parts) < 4:
        while data[at:at + 1].isspace():
            at += 1
        end = at
        while not data[end:end + 1].isspace():
            end += 1
        parts.append(data[at:end].decode())
        at = end
    at += 1
    width, height, scale = int(parts[1]), int(parts[2]), float(parts[3])
    floats = struct.unpack(("<" if scale < 0 else ">") + "f" * (width * height * 3), data[at:at + width * height * 12])
    return width, height, floats


def regions(path):
    """The image's mean luminance-ish (the channels' mean), then its four quadrants'."""
    w, h, px = read_pfm(path)
    sums, counts = [0.0] * 5, [0] * 5
    for y in range(h):
        for x in range(w):
            i = 3 * (y * w + x)
            v = (px[i] + px[i + 1] + px[i + 2]) / 3.0
            q = 1 + (1 if x >= w // 2 else 0) + (2 if y >= h // 2 else 0)
            sums[0] += v
            counts[0] += 1
            sums[q] += v
            counts[q] += 1
    return [s / c for s, c in zip(sums, counts)]


def render(exe, scene_path, env_path, out_path, estimator, seed, extra):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE), "--scenefile", str(scene_path),
           "--envSystem", "1", "--hdrfile", str(env_path), "--hdrIntensity", "1",
           "--ptTechnique", "1", "--dlssEnable", "0", "--optixEnable", "0", "--oidnEnable", "0",
           # no firefly clamp: it cuts one technique's rare bright samples more than the other's
           "--ptAdaptiveSampling", "0", "--ptAdaptiveStop", "0", "--ptFireflyClamp", "0", "--ptFireflyClampStill", "0",
           "--ptMaxDepth", str(MAX_DEPTH), "--ptEstimator", str(estimator), "--ptSeed", str(seed), *extra,
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(out_path).exists():
        print(result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"render failed: {' '.join(cmd)}")


def stats(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "estimator_agreement",
                        help="where the scene and images go")
    parser.add_argument("--configs", default=",".join(CONFIGS), help="which of " + ", ".join(CONFIGS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scene_path, env_path = args.out / "scene.gltf", args.out / "dome.hdr"
    scene_path.write_text(json.dumps(scene()))
    env_hdr(env_path)

    names = ["image", "lower left", "lower right", "upper left", "upper right"]
    failures = 0
    for config in args.configs.split(","):
        per = {}
        for est_name, est in ESTIMATORS.items():
            runs = []
            for seed in range(1, SEEDS + 1):
                image = args.out / f"{config}_{est_name}_{seed}.pfm"
                render(args.exe, scene_path, env_path, image, est, seed, CONFIGS[config])
                runs.append(regions(image))
            per[est_name] = [stats([r[k] for r in runs]) for k in range(5)]
            m, se = per[est_name][0]
            print(f"{config:5s} {est_name:5s} mean {m:.5f} +- {se:.5f} ({100 * se / m:.2f}%)", flush=True)
        if per["mis"][0][0] < 0.02:
            print(f"FAIL: {config}: the scene is not lit (mean {per['mis'][0][0]:.5f})")
            failures += 1
            continue
        for a, b in (("mis", "light"), ("mis", "bsdf"), ("light", "bsdf")):
            for k, name in enumerate(names):
                (ma, sa), (mb, sb) = per[a][k], per[b][k]
                z = abs(ma - mb) / max(math.hypot(sa, sb), 1e-12)
                line = f"{config:5s} {a:5s} vs {b:5s} {name:11s}: {100 * (mb / ma - 1):+.2f}% ({z:.1f} sigma)"
                if z > SIGMAS:
                    print("FAIL: " + line)
                    failures += 1
                elif k == 0:
                    print(line)
    print("PASS" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
