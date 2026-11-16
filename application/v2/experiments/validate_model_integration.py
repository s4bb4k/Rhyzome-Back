from application.v2.services.map_service import generate

BIOMES = ["bosque", "desierto", "ciudad", "mazmorra"]
COMPLEXITIES = ["bajo", "medio", "alto"]

total = 0
correct = 0

print("=== VALIDACIÓN DEL MODELO V2 ===")

for biome in BIOMES:
    for complexity in COMPLEXITIES:
        result = generate({
            "biome": biome,
            "complexity": complexity,
            "seed": 12345,
            "width": 50,
            "height": 35,
            "variant": "B",
            "include_ml": True,
        })

        ml = result["ml"]
        predicted = ml["predicted_complexity"]
        confidence = ml["confidence"]

        matches = predicted == complexity

        total += 1
        correct += int(matches)

        status = "OK" if matches else "REVISAR"

        print(
            f"{biome:10} | "
            f"Solicitada: {complexity:5} | "
            f"Predicha: {predicted:5} | "
            f"Confianza: {confidence:.2%} | "
            f"{status}"
        )

print()
print(f"Coincidencias: {correct}/{total}")
print(f"Porcentaje: {correct / total:.2%}")
