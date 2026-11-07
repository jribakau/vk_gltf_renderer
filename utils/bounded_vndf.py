"""The bounded VNDF sampler against the visible-normal one, for a white GGX reflection lobe (isotropic
alpha, height-correlated Smith masking): the share of samples that reflect below the surface, the
lobe's albedo each estimates, and the variance of that estimate. A numpy twin of nvshaders'
hvd_ggx_sample_vndf (Heitz 2018's distribution, drawn here by Dupuy and Benyoub's cap, the same one)
and hvd_ggx_sample_vndf_bounded (Eto and Tokuyoshi 2023), with brdf_ggx_smith_sample's weights.
tests/test_bsdf_lobes.cpp TheBoundedSamplerKeepsTheMeanWithLessVariance checks the C++ against it.

  python utils/bounded_vndf.py

2026-11-07: lost 3-41% -> 0-6%, variance x0.18-0.65, the albedo the same to 4 decimals (alpha 0.3-1).
"""
import numpy as np
rng = np.random.default_rng(5)
N = 2_000_000

def lam(k, a):
    return (-1 + np.sqrt(1 + a * a * (k[..., 0]**2 + k[..., 1]**2) / k[..., 2]**2)) / 2

def g1(k, a):
    return 1 / (1 + lam(k, a))

def g12(k1, k2, a):
    return 1 / (1 + lam(k1, a) + lam(k2, a))

for a in (0.3, 0.6, 1.0):
    for nov in (0.1, 0.3, 0.7):
        k = np.array([np.sqrt(1 - nov * nov), 0.0, nov])
        u1, u2 = rng.random(N), rng.random(N)
        v = np.array([k[0] * a, k[1] * a, k[2]]); v /= np.linalg.norm(v)
        # bounded cap
        s = 1 + np.hypot(k[0], k[1]); kb = (1 - a * a) * s * s / (s * s + a * a * k[2]**2)
        out = {}
        for name, b in (("vndf", v[2]), ("bounded", kb * v[2])):
            phi = 2 * np.pi * u1
            z = (1 - u2) * (1 + b) - b
            st = np.sqrt(np.clip(1 - z * z, 0, 1))
            m = v + np.stack([st * np.cos(phi), st * np.sin(phi), z], -1)
            h = np.stack([m[:, 0] * a, m[:, 1] * a, np.maximum(m[:, 2], 0)], -1)
            h /= np.linalg.norm(h, axis=1, keepdims=True)
            k2 = 2 * (h @ k)[:, None] * h - k
            ok = k2[:, 2] > 0
            t = np.sqrt((a * k[0])**2 + (a * k[1])**2 + k[2]**2)
            G1s = 2 * k[2] / (k[2] + t) if name == "vndf" else 2 * k[2] / (kb * k[2] + t)
            w = np.where(ok, g12(np.broadcast_to(k, k2.shape), np.where(ok[:, None], k2, [0, 0, 1]), a) / G1s, 0.0)
            out[name] = (1 - ok.mean(), w.mean(), w.var())
        lv, mv, vv = out["vndf"]; lb, mb, vb = out["bounded"]
        print(f"alpha {a:.1f} NoV {nov:.1f}: lost {lv*100:5.1f}% -> {lb*100:5.1f}%  albedo {mv:.4f} / {mb:.4f}  variance x{vb/vv:.2f}")
