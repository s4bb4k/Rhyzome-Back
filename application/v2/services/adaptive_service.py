import math
import secrets
from time import perf_counter

from application.v2.services.map_service import generate

COMPLEXITIES = {"bajo", "medio", "alto"}
MAX_SEED = 2 ** 31 - 1


def _integer(value, name, minimum, maximum):
    """Valida enteros sin aceptar booleanos, decimales ni cadenas ambiguas."""
    if isinstance(value, bool):
        raise ValueError(f"'{name}' debe ser entero")

    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        raise ValueError(f"'{name}' debe ser entero")

    if not minimum <= result <= maximum:
        raise ValueError(
            f"'{name}' debe estar entre {minimum} y {maximum}"
        )

    return result


def _target_probability(ml, target):
    """Obtiene la probabilidad de la clase objetivo."""
    probabilities = ml.get("class_probabilities")

    if not isinstance(probabilities, dict):
        raise ValueError(
            "El modelo no devolvió probabilidades por clase"
        )

    value = probabilities.get(target, 0.0)

    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Probabilidad de complejidad inválida"
        ) from exc

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "La probabilidad debe estar entre 0 y 1"
        )

    return probability


def generate_adaptive(data):
    """
    Genera varios candidatos V2 y selecciona el mejor
    según la probabilidad ML de la complejidad objetivo.

    No modifica ni reentrena los modelos existentes.
    """
    if not isinstance(data, dict):
        raise ValueError("El cuerpo debe ser un objeto JSON")

    target = data.get("complexity")

    if not isinstance(target, str):
        raise ValueError(
            "'complexity' debe ser bajo, medio o alto"
        )

    target = target.strip().lower()

    if target not in COMPLEXITIES:
        raise ValueError(
            "'complexity' debe ser bajo, medio o alto"
        )

    attempts = _integer(
        data.get("attempts", 10),
        "attempts",
        1,
        50,
    )

    # Reutilizamos el contrato existente, pero la predicción
    # debe ejecutarse obligatoriamente para cada candidato.
    if "include_ml" in data and data["include_ml"] is not True:
        raise ValueError(
            "'include_ml' debe ser true en generación adaptativa"
        )

    # Validación de seed, independiente de la generación.
    base_seed = (
        _integer(data["seed"], "seed", 0, MAX_SEED)
        if "seed" in data
        else secrets.randbelow(MAX_SEED + 1)
    )

    if attempts > 1 and base_seed + attempts - 1 > MAX_SEED:
        raise ValueError(
            "La semilla y el número de intentos exceden el rango permitido"
        )

    started = perf_counter()
    best_map = None
    best_probability = -1.0
    best_attempt = None
    exact_matches = 0
    candidates = []

    for index in range(attempts):
        candidate_seed = base_seed + index

        request_data = {
            **data,
            "seed": candidate_seed,
            "include_ml": True,
        }

        # Usa los generadores V2 y el adaptador ML ya validados.
        candidate = generate(request_data)
        ml = candidate["ml"]

        predicted = ml["predicted_complexity"]
        probability = _target_probability(ml, target)
        matches = predicted == target

        if matches:
            exact_matches += 1

        candidates.append({
            "attempt": index + 1,
            "seed": candidate_seed,
            "predicted_complexity": predicted,
            "target_probability": probability,
            "matches_target": matches,
        })

        # Priorizamos una coincidencia de clase y, después,
        # la probabilidad de la clase solicitada.
        current_rank = (int(matches), probability)
        best_rank = (
            int(best_map["ml"]["predicted_complexity"] == target),
            best_probability,
        ) if best_map is not None else (-1, -1.0)

        if current_rank > best_rank:
            best_map = candidate
            best_probability = probability
            best_attempt = index + 1

    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    best_map["adaptive"] = {
        "enabled": True,
        "evaluation_mode": "experimental_v2",
        "selection_strategy": "class_match_then_target_probability",
        "target_complexity": target,
        "predicted_complexity": best_map["ml"]["predicted_complexity"],
        "target_probability": best_probability,
        "target_matched": (
                best_map["ml"]["predicted_complexity"] == target
        ),
        "attempts_requested": attempts,
        "attempts_executed": len(candidates),
        "exact_matches": exact_matches,
        "selected_attempt": best_attempt,
        "base_seed": base_seed,
        "selected_seed": best_map["seed"],
        "elapsed_ms": elapsed_ms,
        "compatibility_validated": False,
        "candidates": candidates,
    }

    return best_map
