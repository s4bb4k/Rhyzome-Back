import numpy as np


def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin_layer(width, height, res_x, res_y, rng):
    angles = rng.uniform(0, 2 * np.pi, (res_y + 1, res_x + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    yy, xx = np.mgrid[0:height, 0:width]
    gx = xx * res_x / width
    gy = yy * res_y / height
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    xf, yf = gx - x0, gy - y0

    def dot(ix, iy, dx, dy):
        g = gradients[iy, ix]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot(x0, y0, xf, yf)
    n10 = dot(x0 + 1, y0, xf - 1, yf)
    n01 = dot(x0, y0 + 1, xf, yf - 1)
    n11 = dot(x0 + 1, y0 + 1, xf - 1, yf - 1)
    u, v = _fade(xf), _fade(yf)
    return (1 - v) * ((1 - u) * n00 + u * n10) + v * ((1 - u) * n01 + u * n11)


def generate_perlin_map(width, height, scale=0.08, octaves=4,
                        persistence=0.5, seed=42, threshold=0.50):
    """Perlin usado por el experimento final. `scale` se conserva por compatibilidad API."""
    del scale
    seed = 42 if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    noise = np.zeros((height, width), dtype=float)
    amplitude, total = 1.0, 0.0
    base_x = int(rng.integers(2, 5))
    base_y = int(rng.integers(2, 5))
    for octave in range(octaves):
        rx = min(base_x * (2**octave), width)
        ry = min(base_y * (2**octave), height)
        noise += amplitude * _perlin_layer(width, height, rx, ry, rng)
        total += amplitude
        amplitude *= persistence
    noise /= max(total, 1e-12)
    lo, hi = noise.min(), noise.max()
    noise = (noise - lo) / (hi - lo + 1e-12)
    return (noise >= threshold).astype(np.uint8).tolist()
