"""Generador B optimizado de Rhyzome V2.

Extiende el generador semántico A mediante mejoras
de conectividad y estructura del escenario.

Admite perfiles experimentales de complejidad
sin modificar los generadores de V1.
"""

from .semantic import generar_A, WIDTH, HEIGHT
from application.v2.services.connectivity import (
    reparar_conectividad,
)


def generar_B(
        bioma,
        seed,
        h=HEIGHT,
        w=WIDTH,
        profile=None,
):
    """Genera un mapa V2 optimizado.

    Args:
        bioma: tipo de escenario.
        seed: semilla reproducible.
        h: altura del mapa.
        w: anchura del mapa.
        profile: perfil experimental opcional.

    Returns:
        Tupla (mapa, cantidad de celdas reparadas).

    Compatibilidad:
        Sin profile mantiene el comportamiento original.
    """

    # Generación semántica con perfil opcional.
    original = generar_A(
        bioma,
        seed,
        h,
        w,
        profile=profile,
    )

    # Copias independientes de las capas.
    m = {
        **original,
        "terrain": original["terrain"].copy(),
        "objects": original["objects"].copy(),
        "collision": original["collision"].copy(),
    }

    cambios = 0

    # ==========================================
    # BOSQUE
    # ==========================================
    if bioma == "bosque":
        m, cambios = reparar_conectividad(m)

    # ==========================================
    # DESIERTO
    # ==========================================
    elif bioma == "desierto":
        x = w // 2

        m["terrain"][:, x:x + 2] = 3
        m["objects"][:, x:x + 2] = 0
        m["collision"][:, x:x + 2] = 0

    # ==========================================
    # CIUDAD
    # ==========================================
    elif bioma == "ciudad":
        cy, cx = h // 2, w // 2

        # Accesos centrales despejados.
        m["terrain"][cy, :] = 5
        m["terrain"][:, cx] = 5

        m["objects"][cy, :] = 0
        m["objects"][:, cx] = 0

        m["collision"][cy, :] = 0
        m["collision"][:, cx] = 0

    # ==========================================
    # MAZMORRA
    # ==========================================
    elif bioma == "mazmorra":
        reparado, cambios = reparar_conectividad(m)

        # Mantener suelo de piedra.
        reparado["terrain"][:] = 4

        m = reparado

    return m, cambios
