import pytest

from application.v2.services import adaptive_service


@pytest.mark.parametrize(
    "complexity",
    ["bajo", "medio", "alto"],
)
def test_adaptive_transmite_perfil_al_generador(
        monkeypatch,
        complexity,
):
    received = []

    def fake_generate(data):
        received.append(data.copy())

        return {
            "seed": data["seed"],
            "layers": {
                "terrain": [[0]],
                "objects": [[0]],
                "collision": [[0]],
            },
            "ml": {
                "predicted_complexity": complexity,
                "class_probabilities": {
                    "bajo": 1.0 if complexity == "bajo" else 0.0,
                    "medio": 1.0 if complexity == "medio" else 0.0,
                    "alto": 1.0 if complexity == "alto" else 0.0,
                },
            },
        }

    monkeypatch.setattr(
        adaptive_service,
        "generate",
        fake_generate,
    )

    result = adaptive_service.generate_adaptive({
        "terrain_type": "bosque",
        "complexity": complexity,
        "seed": 12345,
        "attempts": 3,
        "variant": "B",
    })

    assert len(received) == 3

    for request in received:
        assert request["complexity"] == complexity
        assert request["include_ml"] is True

    assert result["adaptive"]["target_complexity"] == complexity
    assert result["adaptive"]["target_matched"] is True
    assert result["adaptive"]["attempts_executed"] == 3
