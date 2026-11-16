"""Generadores semánticos A de Rhyzome V2.

Experimentos 2/3, aislados de los generadores V1.

IMPORTANTE:
- ruido_suave utiliza interpolación bilineal.
- No equivale al algoritmo Perlin Noise de V1.
- Los perfiles de complejidad son experimentales.
- Sin perfil se conserva el comportamiento original.
"""

import numpy as np

TERRAIN = {
    "cesped": 0,
    "agua": 1,
    "arena": 2,
    "camino": 3,
    "piedra": 4,
    "calle": 5,
    "plaza": 6,
}

OBJECT = {
    "vacio": 0,
    "arbol": 1,
    "roca": 2,
    "cactus": 3,
    "edificio": 4,
    "muro": 5,
    "puerta": 6,
    "decoracion": 7,
}

BIOMAS = ("bosque", "desierto", "ciudad", "mazmorra")

WIDTH, HEIGHT = 50, 35


def ruido_suave(rng, h, w, escalas=(3, 7, 15)):
    """Genera ruido suave mediante interpolación bilineal."""

    yy, xx = np.mgrid[0:h, 0:w]

    total = np.zeros((h, w), dtype=float)
    suma = 0.0

    pesos = (0.55, 0.30, 0.15)

    for n, peso in zip(escalas, pesos):
        gh = max(2, int(np.ceil(h / n)) + 1)
        gw = max(2, int(np.ceil(w / n)) + 1)

        coarse = rng.random((gh, gw))

        yf = yy * (gh - 1) / max(1, h - 1)
        xf = xx * (gw - 1) / max(1, w - 1)

        y0 = np.minimum(yf.astype(int), gh - 2)
        x0 = np.minimum(xf.astype(int), gw - 2)

        a = yf - y0
        b = xf - x0

        interp = (
                (1 - a) * (1 - b) * coarse[y0, x0]
                + a * (1 - b) * coarse[y0 + 1, x0]
                + (1 - a) * b * coarse[y0, x0 + 1]
                + a * b * coarse[y0 + 1, x0 + 1]
        )

        total += peso * interp
        suma += peso

    total /= suma

    return (
            (total - total.min())
            / max(1e-12, total.max() - total.min())
    )


def cellular(rng, h, w, prob=0.43, iterations=4):
    """Genera muros mediante autómatas celulares."""

    walls = rng.random((h, w)) < prob

    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True

    for _ in range(iterations):
        padded = np.pad(
            walls.astype(int),
            1,
            constant_values=1,
        )

        count = sum(
            padded[
                1 + dy:h + 1 + dy,
                1 + dx:w + 1 + dx,
            ]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
        )

        walls = np.where(
            walls,
            count >= 4,
            count >= 5,
        )

        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True

    return walls


def generar_A(
        bioma,
        seed,
        h=HEIGHT,
        w=WIDTH,
        profile=None,
):
    """Genera un escenario semántico V2.

    Args:
        bioma: bosque, desierto, ciudad o mazmorra.
        seed: semilla para generación reproducible.
        h: altura del mapa.
        w: anchura del mapa.
        profile: parámetros experimentales opcionales.

    Returns:
        Diccionario con las capas terrain, objects y collision.

    Compatibilidad:
        Si profile es None, utiliza los valores originales
        del generador V2.
    """

    profile = {} if profile is None else profile.copy()

    # Inicialización reproducible del generador aleatorio.
    rng = np.random.default_rng(int(seed))

    terrain = np.zeros((h, w), dtype=np.uint8)
    objects = np.zeros((h, w), dtype=np.uint8)

    # ==================================================
    # BOSQUE
    # ==================================================
    if bioma == "bosque":
        noise = ruido_suave(rng, h, w)

        water_threshold = profile.get(
            "water_threshold", 0.17
        )
        tree_probability = profile.get(
            "tree_probability", 0.31
        )
        rock_probability = profile.get(
            "rock_probability", 0.035
        )

        terrain[noise < water_threshold] = TERRAIN["agua"]

        cy = h // 2

        for x in range(w):
            y = int(
                np.clip(
                    cy + 3 * np.sin(x / 7 + seed % 17),
                    1,
                    h - 2,
                )
            )

            terrain[
                max(0, y - 1):min(h, y + 2),
                x,
            ] = TERRAIN["camino"]

        mask = (
                (terrain == TERRAIN["cesped"])
                & (rng.random((h, w)) < tree_probability)
        )
        objects[mask] = OBJECT["arbol"]

        mask = (
                (terrain == TERRAIN["cesped"])
                & (objects == OBJECT["vacio"])
                & (rng.random((h, w)) < rock_probability)
        )
        objects[mask] = OBJECT["roca"]

    # ==================================================
    # DESIERTO
    # ==================================================
    elif bioma == "desierto":
        terrain[:] = TERRAIN["arena"]

        noise = ruido_suave(rng, h, w)

        water_threshold = profile.get(
            "water_threshold", 0.12
        )
        cactus_probability = profile.get(
            "cactus_probability", 0.09
        )
        rock_probability = profile.get(
            "rock_probability", 0.04
        )

        terrain[noise < water_threshold] = TERRAIN["agua"]

        y = h // 2
        terrain[y:y + 2, :] = TERRAIN["camino"]

        mask = (
                (terrain == TERRAIN["arena"])
                & (rng.random((h, w)) < cactus_probability)
        )
        objects[mask] = OBJECT["cactus"]

        mask = (
                (terrain == TERRAIN["arena"])
                & (objects == OBJECT["vacio"])
                & (rng.random((h, w)) < rock_probability)
        )
        objects[mask] = OBJECT["roca"]

    # ==================================================
    # CIUDAD
    # ==================================================
    elif bioma == "ciudad":
        terrain[:] = TERRAIN["cesped"]

        building_probability = profile.get(
            "building_probability", 0.73
        )
        tree_probability = profile.get(
            "tree_probability", 0.025
        )

        # Calles horizontales y verticales.
        for x in range(5, w, 10):
            terrain[:, x:x + 2] = TERRAIN["calle"]

        for y in range(5, h, 9):
            terrain[y:y + 2, :] = TERRAIN["calle"]

        # Plaza central.
        cy, cx = h // 2, w // 2

        terrain[
            cy - 3:cy + 4,
            cx - 4:cx + 5,
        ] = TERRAIN["plaza"]

        # Los edificios no ocupan calles ni plaza.
        for y in range(2, h - 2, 3):
            for x in range(2, w - 2, 4):
                area = terrain[
                    y:y + 2,
                    x:x + 3,
                ]

                if (
                        area.shape == (2, 3)
                        and np.all(area == TERRAIN["cesped"])
                        and rng.random() < building_probability
                ):
                    objects[
                        y:y + 2,
                        x:x + 3,
                    ] = OBJECT["edificio"]

        # Árboles en espacios verdes disponibles.
        mask = (
                (terrain == TERRAIN["cesped"])
                & (objects == OBJECT["vacio"])
                & (rng.random((h, w)) < tree_probability)
        )

        objects[mask] = OBJECT["arbol"]

    # ==================================================
    # MAZMORRA
    # ==================================================
    elif bioma == "mazmorra":
        terrain[:] = TERRAIN["piedra"]

        wall_probability = profile.get(
            "wall_probability", 0.43
        )
        iterations = profile.get(
            "iterations", 4
        )

        walls = cellular(
            rng,
            h,
            w,
            prob=wall_probability,
            iterations=iterations,
        )

        objects[walls] = OBJECT["muro"]

        # Corredor transitable central.
        y = h // 2

        objects[
            y - 1:y + 2,
            1:w - 1,
        ] = OBJECT["vacio"]

        # Puertas decorativas atravesables.
        for x in (w // 3, 2 * w // 3):
            objects[y, x] = OBJECT["puerta"]

        # Decoración de zonas transitables.
        mask = (
                (objects == OBJECT["vacio"])
                & (rng.random((h, w)) < 0.02)
        )

        objects[mask] = OBJECT["decoracion"]

    else:
        raise ValueError(
            "Bioma no reconocido: " + str(bioma)
        )

    # ==================================================
    # CAPA DE COLISIONES
    # ==================================================
    # V2:
    # 0 = transitable
    # 1 = obstáculo

    collision = np.isin(
        terrain,
        [TERRAIN["agua"]],
    ).astype(np.uint8)

    collision[
        np.isin(
            objects,
            [
                OBJECT["arbol"],
                OBJECT["roca"],
                OBJECT["cactus"],
                OBJECT["edificio"],
                OBJECT["muro"],
            ],
        )
    ] = 1

    collision[
        np.isin(
            objects,
            [
                OBJECT["puerta"],
                OBJECT["decoracion"],
            ],
        )
    ] = 0

    return {
        "bioma": bioma,
        "seed": int(seed),
        "terrain": terrain,
        "objects": objects,
        "collision": collision,
    }
