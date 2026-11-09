"""Bakes nvshaders/ggx_transmission_table.h.slang: the share of the light a rough dielectric boundary
transmits, entering its denser side, that the path tracer's GGX refraction lobe (btdf_ggx_smith_sample:
VNDF sampling, each microfacet's 1 - F, the refracted ray's Smith masking G1) gives it.

The lobe is single scattering: light its refracted ray's masking blocks is lost, though on a real
microsurface it hits the surface again and mostly goes in. The truth is Heitz et al. 2016's
microsurface random walk ("Multiple-Scattering Microfacet BSDFs with the Smith Model"): GGX slopes,
uniform heights, every order of scattering, reflection and refraction by each microfacet's exact
Fresnel. The table holds the lobe's transmission over the walk's, the share of the light the lobe
lets in, which the shader divides by (with a second table, the lobe's transmission with no Fresnel at
all, which a boundary that reflects nothing lets in whole, for a specular weight below 1): interpolated, it holds the walk within 0.4% from index 1.2 up to
88 degrees and 1.1% nearer 1 up to 84, where the gain itself erred more. By sqrt(2 (eta - 1) / (eta + 1))
from 0 to 1 (eta 1 to 3 in 12 steps, dense near 1, where the gain grows without bound at grazing: steps
even in (eta - 1) / (eta + 1) let in up to 39% too much at index 1.05), perceptual roughness sqrt(alpha)
and sqrt(N.V) (16 each).

    python utils/ggx_transmission_table.py   (from native/vk_gltf_renderer; about 20 minutes on 16 cores)
"""

import os
from multiprocessing import Pool

import numpy as np

ETAS = 12
RES = 16
WALKS = 1000 * 1000  # per cell: the walk's transmission to about 3e-4
LOBE = 1000  # stratified per axis for the lobe's integral


def eta_of(k):
    x = 0.5 * (k / (ETAS - 1)) ** 2
    return (1.0 + x) / (1.0 - x)


def smith_lambda(w, alpha):
    """GGX's Lambda, signed as the walk needs it: below the horizon it is -1 - Lambda(-w)."""
    z = np.clip(w[:, 2], -1.0, 1.0)
    s = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = z / (np.maximum(s, 1e-12) * alpha)
        out = 0.5 * (-1.0 + np.sign(a) * np.sqrt(1.0 + 1.0 / (a * a)))
    out = np.where(z > 0.99999, 0.0, out)
    return np.where(z < -0.99999, -1.0, out)


def lobe_vndf(v, alpha, u1, u2):
    """hvd_ggx_sample_vndf (pbr_ggx_microfacet.h.slang), isotropic: the lobe's own sampler."""
    vh = np.array([alpha * v[0], alpha * v[1], v[2]])
    vh /= np.linalg.norm(vh)
    t1 = np.cross(vh, [0.0, 0.0, 1.0])
    t1 = t1 / np.linalg.norm(t1) if vh[2] < 0.99999 else np.array([1.0, 0.0, 0.0])
    t2 = np.cross(t1, vh)
    a = 1.0 / (1.0 + vh[2])
    r = np.sqrt(u1)
    phi = np.where(u2 < a, u2 / a * np.pi, np.pi + (u2 - a) / (1.0 - a) * np.pi)
    p1 = r * np.cos(phi)
    p2 = r * np.sin(phi) * np.where(u2 < a, 1.0, vh[2])
    h = p1[:, None] * t1 + p2[:, None] * t2 + np.sqrt(np.maximum(0.0, 1.0 - p1 * p1 - p2 * p2))[:, None] * vh
    h[:, 0] *= alpha
    h[:, 1] *= alpha
    h[:, 2] = np.maximum(h[:, 2], 0.0)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def walk_vndf(wi, alpha, u1, u2):
    """Visible normals from any direction, below the horizon too: Dupuy and Benyoub 2023's cap."""
    ws = np.stack([wi[:, 0] * alpha, wi[:, 1] * alpha, wi[:, 2]], -1)
    ws /= np.linalg.norm(ws, axis=-1, keepdims=True)
    phi = (2.0 * u1 - 1.0) * np.pi
    z = (1.0 - u2) * (1.0 + ws[:, 2]) - ws[:, 2]
    st = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    m = np.stack([st * np.cos(phi), st * np.sin(phi), z], -1) + ws
    m = np.stack([m[:, 0] * alpha, m[:, 1] * alpha, m[:, 2]], -1)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    m = m / np.maximum(n, 1e-12)
    m[n[:, 0] < 1e-12] = (0.0, 0.0, 1.0)
    return m


def fresnel(c, eta):
    """The exact dielectric Fresnel at cosine c to the microfacet, eta = n_t / n_i."""
    c = np.abs(c)
    g2 = eta * eta - 1.0 + c * c
    g = np.sqrt(np.maximum(g2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (g - c) / (g + c)
        b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0)
        f = 0.5 * a * a * (1.0 + b * b)
    return np.where(g2 < 0.0, 1.0, np.clip(np.nan_to_num(f, nan=1.0), 0.0, 1.0))


def refract(wi, m, eta):
    eta = np.broadcast_to(np.asarray(eta, float), (len(wi),))
    ci = np.sum(wi * m, -1)
    ct = -np.sqrt(np.maximum(1.0 - (1.0 - ci * ci) / (eta * eta), 0.0))
    wo = m * (ci / eta + ct)[:, None] - wi / eta[:, None]
    return wo / np.linalg.norm(wo, axis=-1, keepdims=True)


def reflect(wi, m):
    return -wi + 2.0 * np.sum(wi * m, -1)[:, None] * m


def lobe_transmission(eta, alpha, nov, fresnel_on=True):
    """The lobe's transmission: the VNDF sample's weight (1 - F(k1.h)) G1(k2), stratified; without its
    Fresnel, G1(k2) alone."""
    v = np.array([np.sqrt(1.0 - nov * nov), 0.0, nov])
    g = (np.arange(LOBE) + 0.5) / LOBE
    u1, u2 = np.meshgrid(g, g)
    h = lobe_vndf(v, alpha, u1.ravel(), u2.ravel())
    kh = h @ v
    f = fresnel(kh, eta)
    k2 = refract(np.tile(v, (len(h), 1)), h, eta)
    ok = (k2[:, 2] < 0.0) & (kh > 0.0) & (f < 1.0)
    g1 = 1.0 / (1.0 + smith_lambda(np.where(ok[:, None], -k2, [0.0, 0.0, 1.0]), alpha))
    return np.where(ok, ((1.0 - f) if fresnel_on else 1.0) * g1, 0.0).mean()


def C1(h):
    return np.clip(0.5 * (h + 1.0), 0.0, 1.0)


def next_height(wr, hr, u, alpha):
    """Heitz 2016's sampleHeight over uniform heights in [-1, 1]: where the ray along wr from height
    hr next meets the microsurface, or inf where it leaves."""
    z = wr[:, 2]
    lm = smith_lambda(wr, alpha)
    c = C1(hr)
    with np.errstate(over="ignore"):
        g1 = np.where(z > 0.0, np.power(np.maximum(c, 1e-30), lm), 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        h = 2.0 * (c / np.power(np.maximum(1.0 - u, 1e-30), 1.0 / np.where(lm == 0.0, 1e-30, lm))) - 1.0
    out = np.where(u > 1.0 - g1, np.inf, h)
    out = np.where(np.abs(z) < 1e-4, hr, out)
    out = np.where(z < -0.9999, 2.0 * (u * c) - 1.0, out)
    return np.where(z > 0.9999, np.inf, out)


def walk_transmission(eta, alpha, nov, rng, n=WALKS, orders=1000):
    """The share of light from cosine nov that the random walk leaves below the surface, and the share
    still on the surface after `orders` scatterings."""
    s = np.sqrt(1.0 - nov * nov)
    wr = np.tile([-s, 0.0, -nov], (n, 1))
    hr = np.full(n, 2.0)
    outside = np.ones(n, bool)
    alive = np.ones(n, bool)
    below = 0
    for _ in range(orders):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        w, o = wr[idx], outside[idx]
        u = rng.random(len(idx))
        hn = np.where(o, next_height(w, hr[idx], u, alpha), -next_height(-w, -hr[idx], u, alpha))
        left = ~np.isfinite(hn)
        below += int(np.count_nonzero(~o[left]))
        alive[idx[left]] = False
        idx, w, o, hn = idx[~left], w[~left], o[~left], hn[~left]
        if len(idx) == 0:
            break
        hr[idx] = hn
        wi = -w
        u1, u2 = rng.random(len(idx)), rng.random(len(idx))
        m = np.where(o[:, None], walk_vndf(wi, alpha, u1, u2), -walk_vndf(-wi, alpha, u1, u2))
        e = np.where(o, eta, 1.0 / eta)
        f = fresnel(np.sum(wi * m, -1), e)
        refl = rng.random(len(idx)) < f
        wn = reflect(wi, m)
        if (~refl).any():
            wn[~refl] = refract(wi[~refl], m[~refl], e[~refl])
        wr[idx] = wn
        outside[idx] = np.where(refl, o, ~o)
    return below / n, np.count_nonzero(alive) / n


def cell(args):
    k, i, j = args
    alpha = (i / (RES - 1)) ** 2
    if alpha <= 0.0:
        return k, i, j, 1.0, 1.0, 0.0
    eta = eta_of(k)
    nov = max((j / (RES - 1)) ** 2, 1e-4)
    rng = np.random.default_rng(1000003 * k + 1009 * i + j)
    lobe = lobe_transmission(eta, alpha, nov)
    # no boundary: every ray goes straight on, which the walk takes ever longer to show at grazing
    walk, stuck = (1.0, 0.0) if eta == 1.0 else walk_transmission(eta, alpha, nov, rng)
    return k, i, j, lobe / max(walk, 1e-6), lobe_transmission(eta, alpha, nov, False), stuck


def main():
    share = np.ones((ETAS, RES, RES))
    masked = np.ones((ETAS, RES, RES))
    jobs = [(k, i, j) for k in range(ETAS) for i in range(RES) for j in range(RES)]
    worst = 0.0
    with Pool() as pool:
        for k, i, j, g, m, stuck in pool.imap_unordered(cell, jobs, chunksize=4):
            share[k, i, j] = g
            masked[k, i, j] = m
            worst = max(worst, stuck)
    print(f"most light left on the surface by the walk's cap: {worst:.1e}")
    write(share, masked)


def block(values):
    rows = []
    for k in range(ETAS):
        rows.append(f"  // eta {eta_of(k):.4f}")
        for i in range(RES):
            rows.append("  " + ", ".join(f"{x:.5f}F" for x in values[k, i]) + ",")
    return chr(10).join(rows)


def write(share, masked):
    out = os.path.join(os.path.dirname(__file__), "..", "..", "nvpro_core2", "nvshaders", "ggx_transmission_table.h.slang")
    with open(out, "w", newline="\n") as f:
        f.write(f"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Figure Viewer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

// Generated by vk_gltf_renderer/utils/ggx_transmission_table.py: do not edit.
//
// The share of the light a rough dielectric boundary transmits, entering its denser side, that the GGX
// refraction lobe (btdf_ggx_smith_sample) gives it: the single-scattering lobe over Heitz et al. 2016's
// microsurface random walk, every order of scattering (SHARE); and the lobe's transmission with no
// Fresnel, its refracted ray's masking alone, which a boundary reflecting nothing would let in whole
// (MASKED). Blocks: sqrt(2 (eta - 1) / (eta + 1)) from 0 to 1 in {ETAS} steps (eta 1 to 3); rows:
// perceptual roughness sqrt(alpha) from 0 to 1; columns: sqrt(N.V) from 0 to 1; {RES} each.
#ifndef GGX_TRANSMISSION_TABLE_H
#define GGX_TRANSMISSION_TABLE_H

#define GGX_TRANSMISSION_TABLE_ETAS {ETAS}
#define GGX_TRANSMISSION_TABLE_RES {RES}

static const float GGX_TRANSMISSION_SHARE[{ETAS * RES * RES}] = {{
{block(share)}
}};

static const float GGX_TRANSMISSION_MASKED[{ETAS * RES * RES}] = {{
{block(masked)}
}};

#endif  // GGX_TRANSMISSION_TABLE_H
""")
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
