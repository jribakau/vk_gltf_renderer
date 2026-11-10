#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: a medium inside another medium meets that medium, not air.

The camera looks at an emitter through glass and a liquid, straight on, in three arrangements:
  - contained: a block of liquid inside a solid block of glass (oil in a bottle that is one solid),
  - overlapping: a block of liquid between two panes of glass and a little into each (a drink
    modelled into its glass's wall),
  - apart: the same with air between the liquid and the panes.

Straight on nothing bends, so what reaches the camera is the emitter times what each boundary lets
through (Fresnel at normal incidence, ((n1 - n2) / (n1 + n2))^2 reflected, between the two media that
meet there) and what each medium absorbs over the length the path runs in it, with the light that
bounces back and forth between the boundaries added (the layers' adding formula). The path tracer
keeps the medium a path is in and the one around it (shaders/medium_stack.h.slang): where two solids
overlap, the path is in the one it entered last, and the face of the first met inside the second is
no boundary.

With one inside flag, the liquid inside the glass was entered from air (2% reflected, not 0.4%), and
the glass's far face, met as if from outside, put the path inside glass from there to the emitter.

The script writes the glTF scenes, renders each headless to a linear .hdr with no environment, and
compares the emitter's brightness through the layers with the emitter alone. Needs a GPU and a built
renderer; uses only the Python standard library.

  python tests/render/medium_nested.py --exe _bin/Release/vk_gltf_renderer.exe

The ray-tracing technique alone by default, the one the path's medium rules run in for both (--all
adds ray query, whose pipeline takes minutes and tens of GB to build after a shader change).
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import IMAGE_SIZE, SAMPLES, TECHNIQUES
from volume_transmittance import emitter, make_scene, mean_luminance

N_GLASS, N_LIQUID   = 1.5, 1.33
SIGMA_GLASS         = 0.5   # extinction, 1/m
SIGMA_LIQUID        = 1.0
TOLERANCE           = 0.01
REGION              = 8     # half the side of the square measured at the image's centre, pixels


def solid(ior, sigma_t):
    """A clear, smooth solid of this index and extinction."""
    return {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.0},
            "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0},
                           "KHR_materials_ior": {"ior": ior},
                           "KHR_materials_volume": {"thicknessFactor": 1.0, "attenuationDistance": 1.0,
                                                    "attenuationColor": [math.exp(-sigma_t)] * 3}}}


def slab(material, z_far, z_near):
    """A box 2 m square across the view, from z_far to z_near."""
    return (material, [0.0, 0.0, 0.5 * (z_far + z_near)], [1.0, 1.0, 0.5 * (z_near - z_far)])


GLASS, LIQUID = solid(N_GLASS, SIGMA_GLASS), solid(N_LIQUID, SIGMA_LIQUID)
# behind everything, facing the camera
EMITTER = (emitter(1.0), [0.0, 0.0, -2.5], [0.5, 0.5, 0.25])

SCENES = {
    "alone":       [EMITTER],
    "contained":   [slab(GLASS, -1.0, 1.0), slab(LIQUID, -0.6, 0.6), EMITTER],
    "overlapping": [slab(GLASS, 0.8, 1.0), slab(LIQUID, -0.9, 0.9), slab(GLASS, -1.0, -0.8), EMITTER],
    "apart":       [slab(GLASS, 0.8, 1.0), slab(LIQUID, -0.7, 0.7), slab(GLASS, -1.0, -0.8), EMITTER],
}

# What the path crosses from the camera to the emitter: the index it is in, and how far, between
# boundaries. Overlapping: the path is in the liquid from its own face (in the near pane's wall) to the
# far pane's face, which it enters there.
AIR = (1.0, 0.0)
LAYERS = {
    "contained":   [(N_GLASS, SIGMA_GLASS * 0.4), (N_LIQUID, SIGMA_LIQUID * 1.2), (N_GLASS, SIGMA_GLASS * 0.4)],
    "overlapping": [(N_GLASS, SIGMA_GLASS * 0.1), (N_LIQUID, SIGMA_LIQUID * 1.7), (N_GLASS, SIGMA_GLASS * 0.2)],
    "apart":       [(N_GLASS, SIGMA_GLASS * 0.2), AIR, (N_LIQUID, SIGMA_LIQUID * 1.4), AIR, (N_GLASS, SIGMA_GLASS * 0.2)],
}


def transmitted(layers):
    """The share of light that crosses `layers` (index, optical depth), air on both sides, straight on:
    each boundary's Fresnel, each layer's absorption, and every bounce between boundaries."""
    indices = [1.0] + [n for n, _ in layers] + [1.0]
    depths  = [tau for _, tau in layers]
    # the stack so far: its transmission, and its reflection met from behind
    t, r_back = 1.0, 0.0
    for i in range(len(indices) - 1):
        r = ((indices[i] - indices[i + 1]) / (indices[i] + indices[i + 1])) ** 2
        between = 1.0 / (1.0 - r_back * r)
        t, r_back = t * (1.0 - r) * between, r + (1.0 - r) ** 2 * r_back * between
        if i < len(depths):
            a = math.exp(-depths[i])
            t, r_back = t * a, r_back * a * a
    return t


def render(exe, scene_path, output_path, technique):
    """The raw accumulation (no DLSS), with bounces enough for every boundary and the light between them."""
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE), "--scenefile", str(scene_path),
           "--envSystem", "2", "--dlssEnable", "0", "--ptTechnique", str(technique), "--ptShadowTransmission", "1",
           "--ptMaxDepth", "32", "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        print(result.stdout[-2000:], result.stderr[-2000:])
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "medium_nested",
                        help="where the scenes and images go")
    parser.add_argument("--all", action="store_true", help="both trace techniques, not ray tracing alone")
    args = parser.parse_args()
    exe = args.exe.resolve()
    techniques = TECHNIQUES if args.all else {"raytracing": TECHNIQUES["raytracing"]}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, boxes in SCENES.items():
        (args.out / f"{name}.gltf").write_text(json.dumps(make_scene(boxes), indent=1))

    failures = 0
    for technique_name, technique in techniques.items():
        lum = {}
        for name in SCENES:
            image = args.out / f"{name}_{technique_name}.hdr"
            image.unlink(missing_ok=True)
            render(exe, (args.out / f"{name}.gltf").resolve(), image.resolve(), technique)
            lum[name] = mean_luminance(image, REGION)
        for name, layers in LAYERS.items():
            measured, expected = lum[name] / lum["alone"], transmitted(layers)
            ok = abs(measured - expected) <= TOLERANCE * expected
            failures += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'}  {technique_name:10s} {name:12s} through the layers {measured:.4f}, expected {expected:.4f} "
                  f"(+-{TOLERANCE:.0%})")
    print("PASS" if failures == 0 else f"{failures} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
