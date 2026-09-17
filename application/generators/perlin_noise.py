import math
import random


def _fade(value):
    return value * value * (3.0 - 2.0 * value)


def _lerp(start, end, amount):
    return start + amount * (end - start)


def _value_noise(x, y, lattice):
    x0, y0 = math.floor(x), math.floor(y)
    tx, ty = _fade(x - x0), _fade(y - y0)
    top = _lerp(lattice(x0, y0), lattice(x0 + 1, y0), tx)
    bottom = _lerp(lattice(x0, y0 + 1), lattice(x0 + 1, y0 + 1), tx)
    return _lerp(top, bottom, ty)


def generate_perlin_map(
    width, height, scale=0.08, octaves=4, persistence=0.5, seed=None, threshold=0.5
):
    """Genera un mapa binario reproducible usando ruido fractal suavizado."""
    base_seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
    cache = {}

    def lattice(x, y, octave):
        key = (x, y, octave)
        if key not in cache:
            mixed = base_seed ^ (x * 73856093) ^ (y * 19349663) ^ (octave * 83492791)
            cache[key] = random.Random(mixed).random()
        return cache[key]

    values = []
    for y in range(height):
        row = []
        for x in range(width):
            amplitude = frequency = 1.0
            total = weight = 0.0
            for octave in range(octaves):
                sample = _value_noise(
                    x * scale * frequency,
                    y * scale * frequency,
                    lambda px, py, o=octave: lattice(px, py, o),
                )
                total += sample * amplitude
                weight += amplitude
                amplitude *= persistence
                frequency *= 2.0
            row.append(total / weight)
        values.append(row)

    return [[1 if value < threshold else 0 for value in row] for row in values]
