"""Prints the named metals of figure/materials/metals.cpp: each one's reflectance head-on (F0) and its
edge tint, OpenPBR's F82 tint, from measured optical constants.

Per wavelength, the exact unpolarised Fresnel reflectance of the conductor n + ik (the Fresnel
equations with Snell's law in complex numbers, as pbrt-v4's FrComplex has them) at normal incidence
and at cos 1/7 (about 82 degrees), integrated over 380-780 nm against the CIE 1931 observer (Wyman,
Sloan and Shirley 2013's fit) under a 6504 K blackbody (D65's colour temperature), to linear sRGB
over a perfect white's. The tint is that reflectance at cos 1/7 over Schlick's from F0 there.

Both are clamped to 1, as OpenPBR's colours are: gold's red head-on comes out at 1.025, outside sRGB,
and gold's green and brass's blue tints at 1.004 and 1.02. The F82 curve through them is within 0.03
of the measured metal at every angle; Schlick's alone is up to 0.11 off on aluminium and 0.18-0.20 on
chrome, iron and titanium, too bright toward grazing. Gold and silver gain little (each line's comment).

    python utils/metal_edge_tints.py   (from native/vk_gltf_renderer)

Optical constants, as refractiveindex.info tabulates them:
  gold, silver, copper: P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972)
  chrome, iron, titanium: P. B. Johnson and R. W. Christy, Phys. Rev. B 9, 5056 (1974)
  aluminium: K. M. McPeak et al., ACS Photonics 2, 326 (2015), every 10 nm
  brass (Cu 70, Zn 30): M. R. Querry, Optical constants, CRDC-CR-85034 (1985), every 10 nm
"""

import cmath
import math

# wavelength (um), n, k
GOLD = """0.3542 1.50 1.866  0.3679 1.48 1.895  0.3815 1.46 1.933  0.3974 1.47 1.952  0.4133 1.46 1.958
0.4305 1.45 1.948  0.4509 1.38 1.914  0.4714 1.31 1.849  0.4959 1.04 1.833  0.5209 0.62 2.081  0.5486 0.43 2.455
0.5821 0.29 2.863  0.6168 0.21 3.272  0.6595 0.14 3.697  0.7045 0.13 4.103  0.7560 0.14 4.542  0.8211 0.16 5.083"""
SILVER = """0.3542 0.10 1.419  0.3679 0.07 1.657  0.3815 0.05 1.864  0.3974 0.05 2.070  0.4133 0.05 2.275
0.4305 0.04 2.462  0.4509 0.04 2.657  0.4714 0.05 2.869  0.4959 0.05 3.093  0.5209 0.05 3.324  0.5486 0.06 3.586
0.5821 0.05 3.858  0.6168 0.06 4.152  0.6595 0.05 4.483  0.7045 0.04 4.838  0.7560 0.03 5.242  0.8211 0.04 5.727"""
COPPER = """0.3542 1.37 1.916  0.3679 1.36 1.975  0.3815 1.33 2.045  0.3974 1.32 2.116  0.4133 1.28 2.207
0.4305 1.25 2.305  0.4509 1.24 2.397  0.4714 1.25 2.483  0.4959 1.22 2.564  0.5209 1.18 2.608  0.5486 1.02 2.577
0.5821 0.70 2.704  0.6168 0.30 3.205  0.6595 0.22 3.747  0.7045 0.21 4.205  0.7560 0.24 4.665  0.8211 0.26 5.180"""
CHROME = """0.354 1.84 2.64  0.368 1.87 2.69  0.381 1.92 2.74  0.397 2.00 2.83  0.413 2.08 2.93  0.431 2.19 3.04
0.451 2.33 3.14  0.471 2.51 3.24  0.496 2.75 3.30  0.521 2.94 3.33  0.549 3.18 3.33  0.582 3.22 3.30
0.617 3.17 3.30  0.659 3.09 3.34  0.704 3.05 3.39  0.756 3.08 3.42  0.821 3.20 3.48"""
IRON = """0.354 1.93 2.35  0.368 2.02 2.43  0.381 2.12 2.50  0.397 2.24 2.58  0.413 2.35 2.65  0.431 2.48 2.71
0.451 2.59 2.77  0.471 2.67 2.82  0.496 2.74 2.88  0.521 2.86 2.91  0.549 2.95 2.93  0.582 2.94 2.99
0.617 2.88 3.05  0.659 2.92 3.10  0.704 2.86 3.19  0.756 2.87 3.28  0.821 2.94 3.39"""
TITANIUM = """0.354 1.82 2.87  0.368 1.90 2.90  0.381 1.99 2.93  0.397 2.08 2.95  0.413 2.14 2.98  0.431 2.21 3.01
0.451 2.27 3.04  0.471 2.32 3.10  0.496 2.36 3.19  0.521 2.44 3.30  0.549 2.54 3.43  0.582 2.60 3.58
0.617 2.67 3.72  0.659 2.76 3.84  0.704 2.86 3.96  0.756 3.00 4.01  0.821 3.21 4.01"""
ALUMINIUM = """0.35 0.28349792 3.663518946  0.36 0.300125667 3.776251503  0.37 0.317597538 3.889198914
0.38 0.335956002 4.002607455  0.39 0.354901676 4.114238107  0.40 0.375150842 4.226433266
0.41 0.396086448 4.336805792  0.42 0.417647849 4.447407079  0.43 0.440996226 4.559109307
0.44 0.464232752 4.669101846  0.45 0.489220122 4.778319404  0.46 0.514817248 4.887948117
0.47 0.53987657 4.997345967  0.48 0.568005038 5.105940631  0.49 0.596705366 5.212874187
0.50 0.625686295 5.320477736  0.51 0.655709839 5.428163169  0.52 0.688336416 5.535955836
0.53 0.720793584 5.641767754  0.54 0.75446839 5.746496546  0.55 0.789405353 5.851936501
0.56 0.829205097 5.958236408  0.57 0.867376853 6.060226283  0.58 0.908569739 6.165184423
0.59 0.948955518 6.267417058  0.60 0.992465612 6.368986418  0.61 1.038145667 6.46681888
0.62 1.088160063 6.566589093  0.63 1.136328574 6.66312559  0.64 1.190265203 6.759417967
0.65 1.246364405 6.852329839  0.66 1.304382818 6.942363757  0.67 1.365410391 7.031951759
0.68 1.426024482 7.116287362  0.69 1.493230683 7.197481356  0.70 1.559751729 7.27391404
0.71 1.631621525 7.34575992  0.72 1.706780893 7.414249862  0.73 1.786398844 7.475374294
0.74 1.872189153 7.528352557  0.75 1.958355454 7.571403838  0.76 2.054224115 7.599462923
0.77 2.144528834 7.61216158  0.78 2.231500036 7.603853787  0.79 2.314643646 7.572223927
0.80 2.373653298 7.522581337  0.81 2.409296598 7.449477169  0.82 2.410049326 7.369351244
0.83 2.375277499 7.288411569  0.84 2.301774531 7.225729444  0.85 2.204898553 7.188085811"""
BRASS = """0.35 1.504 1.815  0.36 1.503 1.815  0.37 1.497 1.818  0.38 1.487 1.818  0.39 1.471 1.813
0.40 1.445 1.805  0.41 1.405 1.794  0.42 1.350 1.786  0.43 1.278 1.784  0.44 1.191 1.797  0.45 1.094 1.829
0.46 0.994 1.883  0.47 0.900 1.957  0.48 0.816 2.046  0.49 0.745 2.145  0.50 0.686 2.250  0.51 0.639 2.358
0.52 0.602 2.464  0.53 0.573 2.568  0.54 0.549 2.668  0.55 0.527 2.765  0.56 0.505 2.860  0.57 0.484 2.958
0.58 0.468 3.059  0.59 0.460 3.159  0.60 0.450 3.253  0.61 0.452 3.345  0.62 0.449 3.434  0.63 0.445 3.522
0.64 0.444 3.609  0.65 0.444 3.695  0.66 0.445 3.778  0.67 0.444 3.860  0.68 0.444 3.943  0.69 0.445 4.025
0.70 0.446 4.106  0.71 0.448 4.186  0.72 0.450 4.266  0.73 0.452 4.346  0.74 0.455 4.424  0.75 0.457 4.501
0.76 0.458 4.579  0.77 0.460 4.657  0.78 0.464 4.737  0.79 0.469 4.814  0.80 0.473 4.890  0.81 0.478 4.965
0.82 0.481 5.039  0.83 0.483 5.115  0.84 0.486 5.192  0.85 0.490 5.269"""

# Edits::metal's keys, in the app's order (src/lib/materialLooks.ts METALS)
METALS = [
    ("gold", GOLD),
    ("silver", SILVER),
    ("copper", COPPER),
    ("brass", BRASS),
    ("aluminium", ALUMINIUM),
    ("chrome", CHROME),
    ("iron", IRON),
    ("titanium", TITANIUM),
]

MU_BAR = 1.0 / 7.0  # cos of about 82 degrees, where OpenPBR's F82 tint is defined
LAMBDAS = [380.0 + 5.0 * i for i in range(81)]  # nm
XYZ_TO_SRGB = [(3.2406, -1.5372, -0.4986), (-0.9689, 1.8758, 0.0415), (0.0557, -0.2040, 1.0570)]


def table(text):
    v = [float(x) for x in text.split()]
    return [(v[i] * 1000.0, v[i + 1], v[i + 2]) for i in range(0, len(v), 3)]


def at(rows, lam):
    """n and k at `lam` nm, linear between the measured rows."""
    if lam <= rows[0][0]:
        return rows[0][1], rows[0][2]
    for (w0, n0, k0), (w1, n1, k1) in zip(rows, rows[1:]):
        if lam <= w1:
            t = (lam - w0) / (w1 - w0)
            return n0 + t * (n1 - n0), k0 + t * (k1 - k0)
    return rows[-1][1], rows[-1][2]


def conductor(cos_i, n, k):
    """The exact unpolarised Fresnel reflectance of a conductor n + ik at incidence cosine cos_i."""
    eta = complex(n, k)
    cos_i = min(max(cos_i, 0.0), 1.0)
    sin2_t = (1.0 - cos_i * cos_i) / (eta * eta)
    cos_t = cmath.sqrt(1.0 - sin2_t)
    r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
    r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
    return (abs(r_parl) ** 2 + abs(r_perp) ** 2) / 2.0


def lobe(x, mu, below, above):
    s = below if x < mu else above
    return math.exp(-0.5 * ((x - mu) / s) ** 2)


def observer(lam):
    """CIE 1931 2-degree colour matching functions (Wyman, Sloan and Shirley 2013, multi-lobe fit)."""
    x = 1.056 * lobe(lam, 599.8, 37.9, 31.0) + 0.362 * lobe(lam, 442.0, 16.0, 26.7) - 0.065 * lobe(lam, 501.1, 20.4, 26.2)
    y = 0.821 * lobe(lam, 568.8, 46.9, 40.5) + 0.286 * lobe(lam, 530.9, 16.3, 31.1)
    z = 1.217 * lobe(lam, 437.0, 11.8, 36.0) + 0.681 * lobe(lam, 459.0, 26.0, 13.8)
    return x, y, z


def blackbody(lam_nm, kelvin=6504.0):
    lam = lam_nm * 1e-9
    h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return 1.0 / (lam**5 * (math.exp(h * c / (lam * kb * kelvin)) - 1.0))


def srgb_of(reflectance):
    """Linear sRGB of a reflectance spectrum (one value per LAMBDAS), unnormalised."""
    xyz = [0.0, 0.0, 0.0]
    for lam, r in zip(LAMBDAS, reflectance):
        s = blackbody(lam)
        for i, c in enumerate(observer(lam)):
            xyz[i] += r * s * c
    return [sum(m * c for m, c in zip(row, xyz)) for row in XYZ_TO_SRGB]


WHITE = srgb_of([1.0] * len(LAMBDAS))


def reflectance(rows, cos_i):
    c = srgb_of([conductor(cos_i, *at(rows, lam)) for lam in LAMBDAS])
    return [c[i] / WHITE[i] for i in range(3)]


def schlick(f0, mu):
    return f0 + (1.0 - f0) * (1.0 - mu) ** 5


def f82(f0, tint, mu):
    """OpenPBR's F82-tint Fresnel, as bsdf_functions.h.slang's f82Fresnel has it."""
    b = schlick(f0, MU_BAR) * (1.0 - tint) / (MU_BAR * (1.0 - MU_BAR) ** 6)
    return min(max(schlick(f0, mu) - b * mu * (1.0 - mu) ** 6, 0.0), 1.0)


def srgb_byte(c):
    c = min(max(c, 0.0), 1.0)
    c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055
    return round(c * 255.0)


def main():
    for key, text in METALS:
        rows = table(text)
        f0 = [min(c, 1.0) for c in reflectance(rows, 1.0)]
        edge = reflectance(rows, MU_BAR)
        tint = [min(edge[i] / schlick(f0[i], MU_BAR), 1.0) for i in range(3)]
        worst = {"schlick": 0.0, "f82": 0.0}
        for degrees in range(0, 90):
            mu = math.cos(math.radians(degrees))
            measured = reflectance(rows, mu)
            for i in range(3):
                worst["schlick"] = max(worst["schlick"], abs(schlick(f0[i], mu) - min(measured[i], 1.0)))
                worst["f82"] = max(worst["f82"], abs(f82(f0[i], tint[i], mu) - min(measured[i], 1.0)))
        hex_ = "#" + "".join("%02x" % srgb_byte(c) for c in f0)
        f = ", ".join("%.4ff" % c for c in f0)
        t = ", ".join("%.4ff" % c for c in tint)
        print('    {"%s", {%s}, {%s}},  // %s; off the measured metal by %.3f (Schlick: %.3f)' % (key, f, t, hex_, worst["f82"], worst["schlick"]))


if __name__ == "__main__":
    main()
