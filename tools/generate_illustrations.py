"""Render the decorative illustrations used on the public website.

These are procedurally generated ARTWORK (not patient images, not model output).
They are only used as decoration in templates and must never be sent to the model.

Run from the project folder:   python tools/generate_illustrations.py
Outputs: static/images/eye-hero.webp, static/images/retina-illustration.webp
"""
import os

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "images")
rng = np.random.default_rng(7)


def smooth_noise_1d(n, octaves=5, seed=0):
    """Periodic 1-D fractal noise of length n (values roughly -1..1)."""
    r = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = np.zeros(n)
    for o in range(octaves):
        k = 2 ** (o + 3)
        for _ in range(3):
            out += r.normal() / (o + 1) * np.sin(k * x * r.uniform(.8, 1.2) + r.uniform(0, 2 * np.pi))
    return out / np.abs(out).max()


def noise_2d(h, w, scale, seed):
    r = np.random.default_rng(seed)
    small = r.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- eye
def render_eye(size=1100, ss=2):
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy, R = S * .5, S * .5, S * .38
    x, y = (xx - cx) / R, (cy - yy) / R          # y up
    rr = x * x + y * y
    inside = rr <= 1
    z = np.sqrt(np.clip(1 - rr, 0, 1))
    n = np.stack([x, y, z], -1)

    # Gaze direction (slightly towards viewer's left/up for a 3/4 look)
    d = np.array([-.30, .12, 1.0]); d /= np.linalg.norm(d)
    u = np.cross([0, 1, 0], d); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    cosang = n @ d
    ang = np.arccos(np.clip(cosang, -1, 1))
    phi = np.arctan2(n @ v, n @ u)

    # Lighting
    L = np.array([-.55, .65, .55]); L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0, 1)
    view = np.array([0, 0, 1.0])
    H = (L + view); H /= np.linalg.norm(H)
    spec = np.clip(n @ H, 0, 1) ** 90
    rim = np.clip(1 - z, 0, 1) ** 2.6
    rimdir = np.clip(n @ np.array([.8, -.35, .2]) / 1.0, 0, 1)

    # --- Sclera
    sclera = np.empty((S, S, 3), np.float32)
    base = np.array([.93, .91, .90])
    warm = np.array([.90, .82, .82])
    edge_t = np.clip((ang - .55) / 1.0, 0, 1)[..., None]
    sclera[:] = base * (1 - edge_t * .35) + warm * edge_t * .35
    blotch = noise_2d(S, S, 90, 3)[..., None]
    sclera *= .96 + .06 * blotch

    # Blood vessels on the sclera: random walks in (ang, phi) space projected onto the sphere
    vess = np.zeros((S, S), np.float32)

    def sphere_walk(a, p, steps, da, turn):
        pts, pv = [], rng.normal(0, turn)
        for _ in range(steps):
            pts.append((a, p))
            pv = pv * .93 + rng.normal(0, turn * .35)
            p += pv
            a -= da * rng.uniform(.7, 1.3)
            if a < .60:
                break
        return np.array(pts)

    def project(pts):
        dirs = np.cos(pts[:, 0])[:, None] * d + np.sin(pts[:, 0])[:, None] * (
            np.cos(pts[:, 1])[:, None] * u + np.sin(pts[:, 1])[:, None] * v)
        vis = dirs[:, 2] > 0.04
        return np.stack([cx + dirs[:, 0] * R, cy - dirs[:, 1] * R], -1)[vis]

    def draw(px, w, val):
        # tapered polyline: thicker at the start (periphery), thinner towards the cornea
        n_ = len(px)
        for i in range(n_ - 1):
            ww = max(1, int(round(w * (1 - .7 * i / n_))))
            cv2.line(vess, tuple(px[i].astype(int)), tuple(px[i + 1].astype(int)), val * (1 - .6 * i / n_), ww, cv2.LINE_AA)

    for k in range(22):
        main = sphere_walk(rng.uniform(1.30, 1.55), rng.uniform(-np.pi, np.pi), 120, .0075, .006)
        px = project(main)
        if len(px) > 3:
            draw(px, ss * rng.uniform(1.6, 3.0), float(rng.uniform(.45, .8)))
        for j in range(8, len(main) - 8, 18):
            if rng.random() < .7:
                a_b, p_b = main[j]
                br = sphere_walk(a_b, p_b, int(rng.integers(20, 60)), .006, .02)
                pxb = project(br)
                if len(pxb) > 3:
                    draw(pxb, ss * rng.uniform(.9, 1.6), float(rng.uniform(.25, .5)))
    vess = cv2.GaussianBlur(vess, (0, 0), ss * .55)
    vess *= np.clip((ang - .6) / .5, 0, 1)
    vessel_col = np.array([.70, .22, .24])
    sclera = sclera * (1 - vess[..., None] * .6) + vessel_col * vess[..., None] * .6

    # --- Iris
    iris_r, pupil_r = .43, .155
    t = np.clip((ang - pupil_r) / (iris_r - pupil_r), 0, 1)       # 0 at pupil, 1 at limbus
    NPHI = 4096
    pidx = ((phi + np.pi) / (2 * np.pi) * NPHI).astype(np.int32) % NPHI
    fib1 = smooth_noise_1d(NPHI, 7, 11)[pidx]
    fib2 = smooth_noise_1d(NPHI, 7, 12)[pidx]
    fine = smooth_noise_1d(NPHI, 8, 13)[pidx]
    fibers = .55 * fib1 * (1 - t) + .55 * fib2 * t + .35 * fine
    radial_wave = np.sin(t * 22 + fib1 * 2.5) * .5 + .5
    crypts = noise_2d(S, S, 22, 21)
    crypts = cv2.GaussianBlur(np.clip((crypts - .74) * 4, 0, 1), (0, 0), 2 * ss) * (t > .3) * (t < .88)

    inner = np.array([.86, .62, .28])     # amber collarette around pupil
    mid = np.array([.12, .52, .55])       # teal stroma
    outer = np.array([.05, .26, .36])     # deep blue-teal near limbus
    w_in = np.clip(1 - t / .32, 0, 1)[..., None]
    w_out = np.clip((t - .55) / .45, 0, 1)[..., None]
    iris = mid * (1 - w_in - w_out) + inner * w_in + outer * w_out
    iris *= (.78 + .32 * fibers[..., None] + .10 * radial_wave[..., None])
    iris *= (1 - .28 * crypts[..., None])
    collarette = np.exp(-((t - .30) / .035) ** 2)[..., None]
    iris += collarette * np.array([.16, .12, .05])
    limbal = np.clip((t - .82) / .18, 0, 1)[..., None] ** 1.5
    iris *= (1 - .75 * limbal)
    pupil_edge = np.exp(-((ang - pupil_r) / .008) ** 2)[..., None]
    iris = iris * (1 - .5 * pupil_edge) + np.array([.20, .12, .06]) * .5 * pupil_edge

    pupil = np.array([.015, .03, .04])
    col = sclera.copy()
    iris_mask = ((ang < iris_r) & (ang >= pupil_r))[..., None]
    pupil_mask = (ang < pupil_r)[..., None]
    # soft limbus blend
    blend = np.clip((iris_r - ang) / .012, 0, 1)[..., None]
    col = col * (1 - blend) + np.where(pupil_mask, pupil, iris) * blend
    del iris_mask

    # --- Shading
    shade = .38 + .70 * lam[..., None]
    col *= shade
    # sub-surface warmth at the terminator
    col += np.array([.05, .03, .04]) * (np.clip(.35 - lam, 0, .35)[..., None]) * (ang > iris_r)[..., None]
    # teal rim light from lower-right/back
    teal = np.array([.18, .85, .86])
    col += teal * (rim * (.35 + .9 * rimdir))[..., None] * .55
    # cornea: glossy specular + soft window highlight
    corneal = (ang < iris_r * 1.08)
    col += spec[..., None] * .9
    hl_c = d * np.cos(.17) + (u * np.cos(2.3) + v * np.sin(2.3)) * np.sin(.17)
    hx, hy = cx + hl_c[0] * R, cy - hl_c[1] * R
    window = np.exp(-(((xx - hx) / (R * .085)) ** 4 + ((yy - hy) / (R * .055)) ** 4))
    col += window[..., None] * corneal[..., None] * .7
    small = np.exp(-(((xx - (cx + .30 * R)) / (R * .03)) ** 2 + ((yy - (cy + .05 * R)) / (R * .022)) ** 2))
    col += small[..., None] * corneal[..., None] * .35
    # cornea bulge: slight darkening ring at limbus
    col *= (1 - .18 * np.exp(-((ang - iris_r * 1.03) / .03) ** 2))[..., None]

    col = np.clip(col, 0, 1)

    # --- Background glow + shadow (transparent PNG)
    rgba = np.zeros((S, S, 4), np.float32)
    dist = np.sqrt(rr)
    fade = np.clip((1.26 - dist) / .24, 0, 1) ** 2          # guarantees zero alpha at the canvas edge
    glow = np.exp(-((dist - 1.0) / .14) ** 2) * (dist > 1) * .5 * fade
    glow_col = np.array([.30, .80, .82])
    shadow = np.exp(-(((xx - cx - .06 * R) / (R * .62)) ** 2 + ((yy - cy - 1.04 * R) / (R * .09)) ** 2)) * .32 * fade
    a_bg = np.clip(glow + shadow, 0, 1)
    bg_rgb = (glow_col * glow[..., None] + np.array([.04, .12, .20]) * shadow[..., None]) / np.maximum(a_bg[..., None], 1e-6)
    edge_aa = np.clip((1 - dist) * R / 1.5, 0, 1)
    alpha = edge_aa + a_bg * (1 - edge_aa)
    rgb = (col * edge_aa[..., None] + bg_rgb * (a_bg * (1 - edge_aa))[..., None]) / np.maximum(alpha[..., None], 1e-6)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    img = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- retina
def render_retina(size=900, ss=2):
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy, R = S / 2, S / 2, S * .47
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    disc = np.array([cx + .36 * R, cy - .02 * R])
    mac = np.array([cx - .10 * R, cy + .02 * R])

    base_c = np.array([.86, .40, .16])
    edge_c = np.array([.42, .12, .05])
    t = np.clip(dist, 0, 1)[..., None] ** 1.6
    col = base_c * (1 - t) + edge_c * t
    tex = (.6 * noise_2d(S, S, 6, 31) + .4 * noise_2d(S, S, 40, 32))[..., None]
    col *= .93 + .12 * tex
    dm = np.sqrt((xx - mac[0]) ** 2 + (yy - mac[1]) ** 2) / R
    col *= (1 - .38 * np.exp(-(dm / .16) ** 2))[..., None]
    col *= (1 - .25 * np.exp(-(dm / .035) ** 2))[..., None]

    # vessels: anatomical arcades around the macula + nasal branches (Bezier curves with gentle jitter)
    veins = np.zeros((S, S), np.float32)
    arts = np.zeros((S, S), np.float32)

    def bezier(p0, p1, p2, p3, n=160, wobble=.012):
        t_ = np.linspace(0, 1, n)[:, None]
        pts = ((1 - t_) ** 3) * p0 + 3 * ((1 - t_) ** 2) * t_ * p1 + 3 * (1 - t_) * t_ ** 2 * p2 + t_ ** 3 * p3
        tang = np.gradient(pts, axis=0)
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        jitter = np.cumsum(rng.normal(0, 1, n)); jitter -= np.linspace(jitter[0], jitter[-1], n)
        jitter = cv2.GaussianBlur(jitter.reshape(1, -1).astype(np.float32), (0, 0), 6).ravel()
        return pts + nrm * jitter[:, None] * wobble * R / (np.abs(jitter).max() + 1e-9)

    def stroke(canvas, pts, w0, w1):
        n_ = len(pts)
        for i in range(n_ - 1):
            ww = max(1, int(round(w0 + (w1 - w0) * i / n_)))
            cv2.line(canvas, tuple(np.int32(pts[i])), tuple(np.int32(pts[i + 1])), 1.0, ww, cv2.LINE_AA)

    def sprout(canvas, pts, w, depth, away_from):
        """Side branches heading away from `away_from` (the macula)."""
        for j in range(25, len(pts) - 10, int(rng.integers(22, 34))):
            p0 = pts[j]
            tang = pts[min(j + 3, len(pts) - 1)] - pts[j - 3]
            tang /= np.linalg.norm(tang) + 1e-9
            out = p0 - away_from; out /= np.linalg.norm(out) + 1e-9
            dirv = tang * .55 + out * .8; dirv /= np.linalg.norm(dirv)
            L = R * rng.uniform(.14, .32) * (.7 if depth else 1)
            p3 = p0 + dirv * L
            if np.hypot(*(p3 - [cx, cy])) > R * .97:
                p3 = np.array([cx, cy]) + (p3 - [cx, cy]) / np.hypot(*(p3 - [cx, cy])) * R * .96
            p1 = p0 + tang * L * .35
            p2 = p0 + dirv * L * .7 + rng.normal(0, .02 * R, 2)
            b = bezier(p0, p1, p2, p3, 70, .01)
            stroke(canvas, b, w * .6, max(1, w * .25))
            if depth == 0:
                sprout(canvas, b, w * .55, 1, away_from)

    D = disc
    for sign in (-1, 1):                     # superior (-1) and inferior (+1) arcades
        for canvas, w, off in ((veins, 10 * ss, 0), (arts, 7 * ss, .06)):
            p0 = D + [0, sign * .02 * R]
            p1 = D + [.04 * R, sign * (.42 + off) * R]
            p2 = mac + [.05 * R, sign * (.50 + off) * R]
            p3 = np.array([cx - .78 * R, cy + sign * (.30 + off * 1.5) * R])
            arc = bezier(p0, p1, p2, p3)
            stroke(canvas, arc, w, w * .35)
            sprout(canvas, arc, w, 0, mac)
    for ang_n, canvas, w in ((-.6, veins, 6.5 * ss), (-.2, arts, 5 * ss), (.25, veins, 6 * ss), (.65, arts, 4.6 * ss)):
        dirv = np.array([np.cos(ang_n), np.sin(ang_n)])
        p3 = D + dirv * R * .55
        b = bezier(D, D + dirv * R * .15, D + dirv * R * .35 + rng.normal(0, .03 * R, 2), p3, 100)
        stroke(canvas, b, w, w * .3)
        sprout(canvas, b, w, 0, D)
    # small macular twigs pointing towards (but not reaching) the fovea
    for k in range(9):
        a_ = rng.uniform(0, 2 * np.pi)
        p0 = mac + R * rng.uniform(.30, .36) * np.array([np.cos(a_), np.sin(a_)])
        p3 = mac + R * rng.uniform(.12, .18) * np.array([np.cos(a_ + rng.uniform(-.5, .5)), np.sin(a_ + rng.uniform(-.5, .5))])
        b = bezier(p0, p0 + (p3 - p0) * .3, p0 + (p3 - p0) * .7 + rng.normal(0, .02 * R, 2), p3, 40, .03)
        stroke(arts if k % 2 else veins, b, 2.2 * ss, 1)

    veins = cv2.GaussianBlur(np.clip(veins, 0, 1), (0, 0), ss * .7)
    arts = cv2.GaussianBlur(np.clip(arts, 0, 1), (0, 0), ss * .7)
    col = col * (1 - .82 * veins[..., None]) + np.array([.40, .05, .04]) * .82 * veins[..., None]
    col = col * (1 - .7 * arts[..., None]) + np.array([.66, .13, .06]) * .7 * arts[..., None]

    dd = np.sqrt((xx - disc[0]) ** 2 + (yy - disc[1]) ** 2) / R
    disc_mask = np.exp(-(dd / .075) ** 12)[..., None]
    disc_tex = (.9 + .1 * tex)
    col = col * (1 - disc_mask) + np.array([.98, .76, .46]) * disc_tex * disc_mask
    col *= (1 - .25 * np.exp(-((dd - .075) / .008) ** 2))[..., None]
    cup = np.exp(-(dd / .035) ** 4)[..., None]
    col = col * (1 - .6 * cup) + np.array([1.0, .93, .75]) * .6 * cup
    col += np.array([.25, .12, .02]) * np.exp(-(dd / .12) ** 2)[..., None] * .18

    col = col * (1 - .55 * veins[..., None] * disc_mask) + np.array([.45, .08, .05]) * .55 * veins[..., None] * disc_mask
    # vignette
    col *= (1 - .45 * np.clip(dist - .55, 0, 1) ** 1.5 * 2)[..., None]
    col = np.clip(col, 0, 1)
    edge = np.clip((1 - dist) * R / 1.5, 0, 1)
    out = np.zeros((S, S, 4), np.float32)
    out[..., :3] = col
    out[..., 3] = edge
    img = (out * 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- hero fundus
def render_retina_hero(size=1000, ss=2, seed=11):
    """A more detailed, naturally toned fundus illustration for the Home page hero (artwork only)."""
    r = np.random.default_rng(seed)
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy, R = S / 2, S / 2, S * .48
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    disc = np.array([cx + .40 * R, cy - .03 * R])
    mac = np.array([cx - .08 * R, cy + .02 * R])

    # Choroidal background: warm orange centre to deep red periphery, with fine tigroid texture
    t = np.clip(dist, 0, 1)[..., None] ** 1.4
    col = np.array([.88, .45, .19]) * (1 - t) + np.array([.48, .14, .06]) * t
    fine = noise_2d(S, S, 4, seed + 1)
    mid = noise_2d(S, S, 18, seed + 2)
    streak = cv2.GaussianBlur(noise_2d(S, S, 3, seed + 3), (0, 0), sigmaX=ss * 6, sigmaY=ss * 1.2)
    col *= (.93 + .07 * fine + .06 * mid + .05 * (streak - streak.mean()) * 4)[..., None]
    dm = np.sqrt((xx - mac[0]) ** 2 + (yy - mac[1]) ** 2) / R
    col *= (1 - .32 * np.exp(-(dm / .17) ** 2))[..., None]
    col = col * (1 - .25 * np.exp(-(dm / .05) ** 2))[..., None] + np.array([.16, .05, .02]) * .1 * np.exp(-(dm / .05) ** 2)[..., None]
    col += np.array([.35, .30, .22]) * np.exp(-(dm / .006) ** 2)[..., None] * .5      # foveal reflex

    veins = np.zeros((S, S), np.float32)
    arts = np.zeros((S, S), np.float32)

    def bez(p0, p1, p2, p3, n=200, wob=.010):
        tt = np.linspace(0, 1, n)[:, None]
        pts = (1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * p1 + 3 * (1 - tt) * tt ** 2 * p2 + tt ** 3 * p3
        tg = np.gradient(pts, axis=0); nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
        j = np.cumsum(r.normal(0, 1, n)); j -= np.linspace(j[0], j[-1], n)
        j = cv2.GaussianBlur(j.reshape(1, -1).astype(np.float32), (0, 0), 5).ravel()
        return pts + nr * j[:, None] * wob * R / (np.abs(j).max() + 1e-9)

    def stroke(cv, pts, w0, w1):
        n_ = len(pts)
        for i in range(n_ - 1):
            cv2.line(cv, tuple(np.int32(pts[i])), tuple(np.int32(pts[i + 1])), 1.0,
                     max(1, int(round(w0 + (w1 - w0) * i / n_))), cv2.LINE_AA)

    def clamp_in(p):
        d_ = np.hypot(*(p - [cx, cy]))
        return p if d_ < R * .98 else np.array([cx, cy]) + (p - [cx, cy]) / d_ * R * .98

    def sprout(cv, pts, w, depth, away):
        step = int(r.integers(24, 36)) if depth == 0 else int(r.integers(26, 40))
        for j in range(12, len(pts) - 6, step):
            p0 = pts[j]
            tg = pts[min(j + 3, len(pts) - 1)] - pts[max(j - 3, 0)]; tg /= np.linalg.norm(tg) + 1e-9
            out = p0 - away; out /= np.linalg.norm(out) + 1e-9
            side = r.choice([-1, 1])
            perp = np.array([-tg[1], tg[0]]) * side
            dv = tg * .6 + perp * .55 + out * .35; dv /= np.linalg.norm(dv)
            L = R * r.uniform(.16, .30) * (.55 if depth else 1)
            p3 = clamp_in(p0 + dv * L)
            b = bez(p0, p0 + tg * L * .35, p0 + dv * L * .7 + r.normal(0, .01 * R, 2), p3, 90, .006)
            stroke(cv, b, max(1, w * .62), 1)
            if depth < 1:
                sprout(cv, b, w * .6, depth + 1, away)

    D = disc
    for sign in (-1, 1):
        for cv, w, off in ((veins, 8 * ss, 0.0), (arts, 5.5 * ss, .07)):
            p0 = D + [0, sign * .03 * R]
            arc = bez(p0, D + [-.04 * R, sign * (.34 + off) * R], mac + [.16 * R, sign * (.44 + off) * R],
                      np.array([cx - .86 * R, cy + sign * (.16 + off * 1.6) * R]), 220, .006)
            stroke(cv, arc, w, w * .3)
            sprout(cv, arc, w, 0, mac)
    for ang_n, cv, w in ((-.75, veins, 5 * ss), (-.35, arts, 3.8 * ss), (.05, veins, 4.5 * ss),
                         (.4, arts, 3.6 * ss), (.8, veins, 4.6 * ss)):
        dv = np.array([np.cos(ang_n), np.sin(ang_n)])
        b = bez(D, D + dv * R * .12, D + dv * R * .3 + r.normal(0, .03 * R, 2), clamp_in(D + dv * R * .52), 140)
        stroke(cv, b, w, w * .3)
        sprout(cv, b, w, 0, D)
    for k in range(8):                                   # fine macular capillary twigs
        a_ = r.uniform(0, 2 * np.pi)
        p0 = mac + R * r.uniform(.26, .34) * np.array([np.cos(a_), np.sin(a_)])
        p3 = mac + R * r.uniform(.12, .17) * np.array([np.cos(a_ + r.uniform(-.6, .6)), np.sin(a_ + r.uniform(-.6, .6))])
        stroke(arts if k % 2 else veins, bez(p0, p0 + (p3 - p0) * .3, p0 + (p3 - p0) * .7 + r.normal(0, .02 * R, 2), p3, 50, .03), 1.6 * ss, 1)

    veins = cv2.GaussianBlur(np.clip(veins, 0, 1), (0, 0), ss * .6)
    arts = cv2.GaussianBlur(np.clip(arts, 0, 1), (0, 0), ss * .6)
    col = col * (1 - .80 * veins[..., None]) + np.array([.38, .05, .04]) * .80 * veins[..., None]
    col = col * (1 - .68 * arts[..., None]) + np.array([.68, .15, .07]) * .68 * arts[..., None]
    reflex = cv2.GaussianBlur(cv2.erode(arts, np.ones((3, 3), np.uint8), iterations=ss * 2), (0, 0), ss * .4)
    col += np.array([.10, .06, .04]) * np.clip(reflex, 0, 1)[..., None] * .6           # arterial light reflex

    # Optic disc: pale rim, brighter cup, soft peripapillary halo; vessels re-cross the rim
    dd = np.sqrt((xx - D[0]) ** 2 + (yy - D[1]) ** 2) / R
    halo = np.exp(-(dd / .15) ** 2)[..., None]
    col = col * (1 - .25 * halo) + np.array([.92, .58, .30]) * .25 * halo
    dmask = np.exp(-(dd / .088) ** 10)[..., None]
    col = col * (1 - dmask) + np.array([.97, .72, .46]) * (.94 + .06 * fine[..., None]) * dmask
    cupd = np.sqrt((xx - D[0] - .012 * R) ** 2 + (yy - D[1]) ** 2) / R
    cup = np.exp(-(cupd / .038) ** 4)[..., None]
    col = col * (1 - .65 * cup) + np.array([1.0, .90, .74]) * .65 * cup
    col = col * (1 - .85 * veins[..., None] * dmask) + np.array([.42, .07, .05]) * .85 * veins[..., None] * dmask
    col = col * (1 - .75 * arts[..., None] * dmask) + np.array([.70, .18, .09]) * .75 * arts[..., None] * dmask

    # Lens vignette and soft circular aperture
    col *= (1 - .55 * np.clip(dist - .55, 0, 1) ** 1.6 * 2.2)[..., None]
    col = np.clip(col, 0, 1)
    out = np.zeros((S, S, 4), np.float32)
    out[..., :3] = col
    out[..., 3] = np.clip((1 - dist) * R / (2.5 * ss), 0, 1)
    img = (out * 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- anatomical cutaway eye
def render_eye_cutaway(size=900, ss=2, seed=5):
    """Ray-traced anatomical eyeball with a quarter-section cutaway revealing the retina (artwork only).

    The inner back wall is textured with the procedurally generated retina-hero artwork.
    """
    r_ = np.random.default_rng(seed)
    S = size * ss
    fund = cv2.imread(os.path.join(OUT, "retina-hero.webp"), cv2.IMREAD_UNCHANGED)
    fund = cv2.cvtColor(fund, cv2.COLOR_BGRA2RGB).astype(np.float32) / 255

    def sample(img, mx, my):
        """Bilinear lookup of img (H, W[, C]) at float pixel coords."""
        H_, W_ = img.shape[:2]
        mx = np.clip(mx, 0, W_ - 1.001); my = np.clip(my, 0, H_ - 1.001)
        x0 = mx.astype(np.int32); y0 = my.astype(np.int32); fx = mx - x0; fy = my - y0
        if img.ndim == 3:
            fx = fx[:, None]; fy = fy[:, None]
        return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
                + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)

    def nrm(v):
        return v / np.linalg.norm(v)

    ext = 1.55
    lin = np.linspace(-ext, ext, S, dtype=np.float32)
    SX, SY = np.meshgrid(lin, -lin)
    SX = SX + .14                                                # shift the globe left to leave room for the nerve
    camd = nrm(np.array([0.74, 0.38, 0.56], np.float32))       # direction towards the camera
    f = -camd
    right = nrm(np.cross(f, [0, 1, 0])).astype(np.float32)
    up = np.cross(right, f).astype(np.float32)
    O = (camd * 6)[None, :] + SX.reshape(-1, 1) * right + SY.reshape(-1, 1) * up
    N = O.shape[0]
    d = f.astype(np.float32)

    L = nrm(-0.55 * right + 0.65 * up + 0.55 * camd)            # key light, upper left
    Lr = nrm(0.85 * right - 0.25 * up - 0.45 * camd)            # teal rim light, behind right
    Lf = nrm(0.4 * right - 0.8 * up + 0.4 * camd)              # soft cool fill from below
    Hh = nrm(L + camd)

    def P(t):
        return O + t[:, None] * d

    def sph(c, rad):
        oc = O - np.asarray(c, np.float32)
        b = oc @ d
        cc = (oc * oc).sum(1) - rad * rad
        disc = b * b - cc
        sq = np.sqrt(np.maximum(disc, 0))
        return -b - sq, -b + sq, disc > 0

    def inC(p):
        return (p[:, 0] > 0) & (p[:, 1] > 0)

    RI, ZJ, CC, RC = .90, .8667, np.array([0, 0, .45], np.float32), .65
    IRIS_Z, IRIS_R, PUP_R = .86, .44, .15
    LC, LS = np.array([0, 0, .70], np.float32), np.array([.34, .34, .17], np.float32)
    INF = np.float32(1e9)

    # ---- textures: scleral vessels (equirectangular), iris fibres
    Hc, Wc = 1024, 2048
    vtex = np.zeros((Hc, Wc), np.float32)
    for _ in range(60):
        th, ph = r_.uniform(.62, .75), r_.uniform(-np.pi, np.pi)
        pts, pv = [], 0
        for _ in range(160):
            pts.append((ph, th)); pv = pv * .92 + r_.normal(0, .006); ph += pv; th += r_.uniform(.004, .01)
            if th > 2.2:
                break
        pts = np.array(pts)
        px = np.stack([(pts[:, 0] + np.pi) / (2 * np.pi) * Wc, pts[:, 1] / np.pi * Hc], 1).astype(np.int32)
        cv2.polylines(vtex, [px], False, float(r_.uniform(.3, .7)), int(r_.integers(1, 3)), cv2.LINE_AA)
    vtex = cv2.GaussianBlur(vtex, (0, 0), 1.0)

    col = np.zeros((N, 3), np.float32)
    alpha = np.zeros(N, np.float32)

    def sclera_shade(p, idx):
        n = p / np.linalg.norm(p, axis=1, keepdims=True)
        th = np.arccos(np.clip(n[:, 2], -1, 1)); ph = np.arctan2(n[:, 1], n[:, 0])
        mx = ((ph + np.pi) / (2 * np.pi) * (Wc - 1)).astype(np.float32); my = (th / np.pi * (Hc - 1)).astype(np.float32)
        v = sample(vtex, mx, my)
        v *= np.clip((th - .62) / .4, 0, 1) * np.clip((2.2 - th) / .6, 0, 1)
        base = np.array([.94, .925, .915]) * (1 - .06 * np.clip((th - 1.2) / 1.5, 0, 1))[:, None]
        vw = .55 * (1 - .45 * np.clip((th - 1.5) / .6, 0, 1))[:, None]
        base = base * (1 - vw * v[:, None]) + np.array([.72, .24, .25]) * vw * v[:, None]
        lam = np.clip(n @ L, 0, 1)[:, None]
        spec = (np.clip(n @ Hh, 0, 1) ** 40)[:, None] * .22
        rim = ((1 - np.clip(n @ camd, 0, 1)) ** 3 * np.clip(n @ Lr + .35, 0, 1))[:, None]
        fill = np.clip(n @ Lf, 0, 1)[:, None]
        shade = .44 + .62 * lam + .16 * fill
        cool = np.array([.93, .97, 1.0]) * (1 - lam) + lam
        c = base * shade * cool + spec + np.array([.20, .78, .80]) * rim * .5
        col[idx] = c; alpha[idx] = 1

    def iris_color(p):
        rho = np.hypot(p[:, 0], p[:, 1]); t = np.clip((rho - PUP_R) / (IRIS_R - PUP_R), 0, 1)
        phi = np.arctan2(p[:, 1], p[:, 0])
        k = ((phi + np.pi) / (2 * np.pi) * 4096).astype(np.int32) % 4096
        fib = .6 * smooth_noise_1d(4096, 7, 31)[k] + .4 * smooth_noise_1d(4096, 8, 32)[k]
        inner, mid, outer = np.array([.80, .56, .26]), np.array([.13, .50, .53]), np.array([.05, .24, .33])
        w_in = np.clip(1 - t / .3, 0, 1)[:, None]; w_out = np.clip((t - .55) / .45, 0, 1)[:, None]
        c = mid * (1 - w_in - w_out) + inner * w_in + outer * w_out
        c *= (.78 + .3 * fib[:, None])
        c *= (1 - .7 * (np.clip((t - .85) / .15, 0, 1) ** 1.5))[:, None]
        c[rho < PUP_R] = [.01, .02, .03]
        return c

    def retina_color(p):
        n = p / np.linalg.norm(p, axis=1, keepdims=True)
        th = np.arccos(np.clip(-n[:, 2], -1, 1)); ph = np.arctan2(n[:, 1], n[:, 0]) + 2.55
        rn = th / (np.pi * .5) * .9
        mx = ((.5 + .5 * np.clip(rn, 0, .98) * np.cos(ph)) * (fund.shape[1] - 1)).astype(np.float32)
        my = ((.5 - .5 * np.clip(rn, 0, .98) * np.sin(ph)) * (fund.shape[0] - 1)).astype(np.float32)
        c = sample(fund, mx, my)
        per = np.array([.50, .15, .07]) * (1 - np.clip((th - 1.6) / .7, 0, 1))[:, None] + np.array([.10, .035, .03]) * np.clip((th - 1.6) / .7, 0, 1)[:, None]
        w = np.clip((rn - .80) / .15, 0, 1)[:, None]
        c = c * (1 - w) + per * w
        sh = .32 + .85 * np.clip((-n) @ camd, 0, 1) ** .8
        return c * sh[:, None]

    def face_color(rq, nf):
        c = np.where((rq > .955)[:, None], np.array([.93, .90, .86]),
            np.where((rq > .925)[:, None], np.array([.36, .09, .07]), np.array([.86, .50, .40])))
        lam = np.clip(nf @ L, 0, 1)
        return c * (.52 + .55 * lam)[:, None]

    def lens_overlay(c, t_from, t_to, mask):
        oc = (O[mask] - LC) / LS; dd = d / LS
        a = (dd * dd).sum(); b = oc @ dd; cc = (oc * oc).sum(1) - 1
        disc = b * b - a * cc; sq = np.sqrt(np.maximum(disc, 0))
        tl = (-b - sq) / a
        pl = O[mask] + tl[:, None] * d
        ok = (disc > 0) & (tl > t_from) & (tl < t_to) & ~inC(pl)
        nl = (pl - LC) / (LS * LS); nl /= np.linalg.norm(nl, axis=1, keepdims=True) + 1e-9
        spec = np.clip(nl @ Hh, 0, 1) ** 60
        out = c.copy()
        out[ok] = c[ok] * .78 + np.array([.80, .88, .92]) * .22 + spec[ok, None] * .5
        return out

    # ---- outer surface (eyeball ∪ corneal cap)
    e0, e1, eh = sph([0, 0, 0], 1.0)
    c0, c1, ch = sph(CC, RC)
    eye_ok = eh & (P(e0)[:, 2] <= ZJ)
    cor_ok = ch & (P(c0)[:, 2] > ZJ)
    t_out = np.where(eye_ok, e0, INF)
    is_cor = cor_ok & (c0 < t_out)
    t_out = np.where(is_cor, c0, t_out)
    hit = t_out < INF
    Pout = P(np.where(hit, t_out, 0))
    cutO = inC(Pout)
    w0, w1, wh = sph([0, 0, 0], RI)
    t_ir = (IRIS_Z - O[:, 2]) / d[2]
    Pi = P(t_ir)
    rho_i = np.hypot(Pi[:, 0], Pi[:, 1])
    Pw = P(w1)
    lens_from = np.zeros(N, np.float32)

    # A1: intact sclera
    m = hit & ~cutO & ~is_cor
    sclera_shade(Pout[m], m)

    # A2: intact cornea -> iris / pupil (or interior where the iris quadrant is cut away)
    m = hit & ~cutO & is_cor
    iris_here = m & (rho_i < IRIS_R) & ~inC(Pi)
    col[iris_here] = iris_color(Pi[iris_here]) * .95; alpha[iris_here] = 1
    limb = m & (rho_i >= IRIS_R)
    if limb.any():
        sclera_shade(Pout[limb], limb); col[limb] *= .82
    through = m & (rho_i < IRIS_R) & inC(Pi)
    col[through] = retina_color(Pw[through]); alpha[through] = 1
    nc = (Pout - CC) / RC
    corspec = np.clip(nc @ Hh, 0, 1) ** 140 * 1.4
    fres = (1 - np.clip(nc @ camd, 0, 1)) ** 4 * .35
    col[m] += corspec[m, None] + fres[m, None] * np.array([.80, .95, 1.0])

    # B: ray enters the removed quadrant
    m = hit & cutO
    tx = np.where(d[0] != 0, -O[:, 0] / d[0], INF); ty = np.where(d[1] != 0, -O[:, 1] / d[1], INF)
    tx = np.where(tx > t_out, tx, INF); ty = np.where(ty > t_out, ty, INF)
    t_ex = np.minimum(tx, ty)
    q = P(np.where(t_ex < INF, t_ex, 0))
    rq = np.linalg.norm(q, axis=1)
    nf = np.where((tx <= ty)[:, None], np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32))
    band = m & (t_ex < INF) & (rq >= RI) & (rq <= 1.0) & (q[:, 2] <= ZJ)
    dqc = np.linalg.norm(q - CC, axis=1)
    front = m & (t_ex < INF) & (q[:, 2] > ZJ) & (dqc <= RC)
    cornea_sec = front & (dqc >= RC - .035)
    col[band] = face_color(rq[band], nf[band]); alpha[band] = 1
    inner = m & (t_ex < INF) & (((rq < RI) & (q[:, 2] <= ZJ)) | front)
    iris_b = inner & (t_ir > t_ex) & (t_ir < w1) & (rho_i >= PUP_R) & (rho_i < IRIS_R) & ~inC(Pi)
    col[iris_b] = iris_color(Pi[iris_b]) * .55; alpha[iris_b] = 1
    wall = inner & ~iris_b & ~inC(Pw)
    col[wall] = retina_color(Pw[wall]); alpha[wall] = 1
    # vitreous depth haze
    haze = np.clip((w1 - t_ex) / 1.8, 0, 1)
    col[wall] = col[wall] * (1 - .10 * haze[wall, None]) + np.array([.55, .75, .80]) * .10 * haze[wall, None]
    lm = inner.copy()
    col[lm] = lens_overlay(col[lm], t_ex[lm], np.where(iris_b, t_ir, w1)[lm], lm)
    col[cornea_sec] = col[cornea_sec] * .55 + np.array([.86, .94, .96]) * .45
    # optic nerve: short sheathed cylinder leaving the back of the globe
    nx0, ny0, nr_ = .10, -.04, .16
    ox, oy = O[:, 0] - nx0, O[:, 1] - ny0
    qa = d[0] ** 2 + d[1] ** 2; qb = ox * d[0] + oy * d[1]; qc = ox ** 2 + oy ** 2 - nr_ ** 2
    qd = qb * qb - qa * qc
    tn = (-qb - np.sqrt(np.maximum(qd, 0))) / qa
    pn = P(tn)
    nerve = (qd > 0) & (pn[:, 2] < -.80) & (pn[:, 2] > -1.38) & (np.linalg.norm(pn, axis=1) > .995) & ((tn < t_out) | ~hit)
    if nerve.any():
        nn = np.stack([pn[nerve, 0] - nx0, pn[nerve, 1] - ny0, np.zeros(nerve.sum(), np.float32)], 1) / nr_
        lam_n = np.clip(nn @ L, 0, 1)
        stri = .94 + .06 * np.sin(np.arctan2(nn[:, 1], nn[:, 0]) * 40)
        col[nerve] = np.array([.93, .84, .78]) * ((.45 + .6 * lam_n) * stri)[:, None] + np.array([.2, .7, .75]) * ((1 - np.abs(nn @ camd)) ** 3 * .25)[:, None]
        alpha[nerve] = 1
    # optic nerve end cap (cut surface) with central retinal vessels
    zc = -1.38
    tcap = (zc - O[:, 2]) / d[2]
    pc = P(tcap)
    rcap = np.hypot(pc[:, 0] - nx0, pc[:, 1] - ny0)
    cap = (rcap < nr_) & ((tcap < t_out) | ~hit)
    if cap.any():
        cc_ = np.array([.90, .80, .74]) * .78
        cap_col = np.tile(cc_, (cap.sum(), 1))
        rc_ = rcap[cap] / nr_
        cap_col *= (1 - .25 * (rc_ > .82))[:, None]
        dots = (np.hypot(pc[cap, 0] - nx0 - .02, pc[cap, 1] - ny0) < .025) | (np.hypot(pc[cap, 0] - nx0 + .025, pc[cap, 1] - ny0 + .01) < .022)
        cap_col[dots] = [.55, .10, .08]
        col[cap] = cap_col; alpha[cap] = 1
    # subtle dark edge where the cut meets the outer surface
    edge = m & band & (rq > .993)
    col[edge] *= .8

    col = np.clip(col, 0, 1).reshape(S, S, 3)
    alpha = alpha.reshape(S, S)

    # ---- soft shadow + faint teal atmosphere (fades to zero at canvas edges)
    rr = np.sqrt((SX) ** 2 + SY ** 2)
    fade = np.clip((ext - .02 - np.maximum(np.abs(SX - .14), np.abs(SY))) / .25, 0, 1) ** 2
    shadow = np.exp(-(((SX - .05) / .78) ** 2 + ((SY + 1.12) / .10) ** 2)) * .30 * fade
    glow = np.exp(-((rr - 1.0) / .16) ** 2) * (rr > .95) * .16 * fade
    a_bg = np.clip(shadow + glow, 0, 1)
    bg = (np.array([.04, .12, .22]) * shadow[..., None] + np.array([.30, .78, .80]) * glow[..., None]) / np.maximum(a_bg[..., None], 1e-6)
    a = alpha + a_bg * (1 - alpha)
    rgb = (col * alpha[..., None] + bg * (a_bg * (1 - alpha))[..., None]) / np.maximum(a[..., None], 1e-6)
    out = np.dstack([rgb, a])
    img = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__" and "--anatomy-only" in os.sys.argv:
    eye = render_eye_cutaway()
    cv2.imwrite(os.path.join(OUT, "eye-anatomy.webp"), cv2.cvtColor(eye, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_WEBP_QUALITY, 90])
    print("Saved", os.path.join(OUT, "eye-anatomy.webp"))
elif __name__ == "__main__" and "--hero-only" in os.sys.argv:
    hero = render_retina_hero()
    cv2.imwrite(os.path.join(OUT, "retina-hero.webp"), cv2.cvtColor(hero, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_WEBP_QUALITY, 90])
    print("Saved", os.path.join(OUT, "retina-hero.webp"))
elif __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    eye = render_eye()
    cv2.imwrite(os.path.join(OUT, "eye-hero.webp"), cv2.cvtColor(eye, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_WEBP_QUALITY, 90])
    ret = render_retina()
    cv2.imwrite(os.path.join(OUT, "retina-illustration.webp"), cv2.cvtColor(ret, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_WEBP_QUALITY, 90])
    print("Saved illustrations to", OUT)
