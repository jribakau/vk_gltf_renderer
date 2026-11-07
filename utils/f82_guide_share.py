"""Checks shaders/dlss_specular_guide.h.slang's dlssEdgeTintShare: what a metal's edge tint (OpenPBR's
F82) leaves of its GGX lobe's directional albedo against the Schlick lobe's, which DLSS-RR's specular
albedo guide multiplies its Schlick fit (EnvBRDFApprox2) by.

The truth: each lobe's albedo, E(mu_o) = E[F(o.h) G2 / G1(o)] over GGX's visible normals (Heitz 2018,
height-correlated Smith), by Monte Carlo with the same samples for both Fresnels, so their ratio is
steadier than either. Against it, three stand-ins: the hemispherical average share alone
(f82AverageShare), the Fresnel's ratio at the view alone, and the shader's blend of the two: linear in
alpha up to 0.5, the view's cosine floored at 0.4 sqrt(alpha). Each named metal's darkest channel (figure/materials/metals.cpp), where the tint
takes most. Prints each one's worst relative error and the table behind it. About two minutes.

    python utils/f82_guide_share.py   (needs numpy)
"""
import numpy as np

N = 200_000
ROUGH_ALPHA = 0.5  # DLSS_EDGE_TINT_ROUGH_ALPHA
VIEW_FLOOR = 0.4  # DLSS_EDGE_TINT_VIEW_FLOOR
rng = np.random.default_rng(1)
U1, U2 = rng.random(N), rng.random(N)


def schlick(f0, mu):
    return f0 + (1 - f0) * (1 - mu) ** 5


def f82_b(f0, tint):
    return schlick(f0, 1 / 7) * (1 - tint) * (823543 / 46656)


def f82(f0, tint, mu):
    m = 1 - mu
    return np.clip(schlick(f0, mu) - f82_b(f0, tint) * mu * m ** 6, 0, 1)


def average_share(f0, tint):
    s = f0 + (1 - f0) / 21
    return max(s - f82_b(f0, tint) / 126, 0) / s


def smith_lambda(mu, a):
    t2 = (1 - mu * mu) / np.maximum(mu * mu, 1e-12)
    return 0.5 * (-1 + np.sqrt(1 + a * a * t2))


def albedos(fresnels, mu_o, a):
    """Each Fresnel's lobe albedo at view cosine mu_o, alpha a, over the same visible normals."""
    o = np.array([np.sqrt(1 - mu_o ** 2), 0.0, mu_o])
    vh = np.array([a * o[0], a * o[1], o[2]])
    vh /= np.linalg.norm(vh)
    if vh[2] > 0.9999:
        t1 = np.array([1.0, 0.0, 0.0])
    else:
        t1 = np.cross([0, 0, 1.0], vh)
        t1 /= np.linalg.norm(t1)
    t2 = np.cross(vh, t1)
    r, phi = np.sqrt(U1), 2 * np.pi * U2
    p1, p2 = r * np.cos(phi), r * np.sin(phi)
    s = 0.5 * (1 + vh[2])
    p2 = (1 - s) * np.sqrt(1 - p1 * p1) + s * p2
    nh = p1[:, None] * t1 + p2[:, None] * t2 + np.sqrt(np.maximum(0, 1 - p1 * p1 - p2 * p2))[:, None] * vh
    h = np.stack([a * nh[:, 0], a * nh[:, 1], np.maximum(1e-7, nh[:, 2])], 1)
    h /= np.linalg.norm(h, axis=1)[:, None]
    oh = np.clip(h @ o, 0, 1)
    mu_i = (2 * oh[:, None] * h - o)[:, 2]
    g = np.where(mu_i > 0, (1 + smith_lambda(mu_o, a)) / (1 + smith_lambda(mu_o, a) + smith_lambda(np.maximum(mu_i, 1e-6), a)), 0.0)
    return [float((f(oh) * g).mean()) for f in fresnels]


# name: (f0, edge tint) of the channel the tint takes most from
METALS = {
    "chrome": (0.5496, 0.7418),
    "iron": (0.5289, 0.7655),
    "titanium": (0.6170, 0.7696),
    "aluminium": (0.9067, 0.8891),
    "copper": (0.5208, 0.9455),
    "brass": (0.9012, 0.9529),
}
MUS = [1.0, 0.85, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.04]
ALPHAS = [0.0004, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 1.0]

worst = {"average": (0, None), "view": (0, None), "blend": (0, None)}
for name, (f0, tint) in METALS.items():
    avg = average_share(f0, tint)
    print(f"\n{name}: f0 {f0}, tint {tint}, average share {avg:.3f}")
    print(" alpha    mu   truth  average  view   blend")
    for a in ALPHAS:
        for mu in MUS:
            e_schlick, e_f82 = albedos([lambda m: schlick(f0, m), lambda m: f82(f0, tint, m)], mu, a)
            truth = e_f82 / e_schlick
            view = float(f82(f0, tint, mu) / schlick(f0, mu))
            k = min(a / ROUGH_ALPHA, 1.0)
            me = min(max(mu, VIEW_FLOOR * np.sqrt(a)), 1.0)
            blend = (1 - k) * float(f82(f0, tint, me) / schlick(f0, me)) + k * avg
            for key, v in (("average", avg), ("view", view), ("blend", blend)):
                err = abs(v - truth) / truth
                if err > worst[key][0]:
                    worst[key] = (err, f"{name} alpha {a} mu {mu}")
            print(f"{a:6.4f}  {mu:4.2f}  {truth:.3f}  {avg:.3f}    {view:.3f}  {blend:.3f}")
print("\nworst relative error:")
for key, (err, where) in worst.items():
    print(f"  {key:8s} {err:.3f}  ({where})")
