import unittest

from app import create_app


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

    def test_cellular_map_is_reproducible(self):
        payload = {
            "algorithm": "cellular", "width": 20, "height": 15,
            "seed": 1001, "probability": 0.45, "iterations": 4,
        }
        first = self.client.post("/api/maps/generate", json=payload)
        second = self.client.post("/api/maps/generate", json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.get_json()["map"], second.get_json()["map"])

    def test_perlin_map_dimensions(self):
        response = self.client.post(
            "/api/maps/generate",
            json={"algorithm": "perlin", "width": 24, "height": 18, "seed": 7},
        )
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(body["map"]), 18)
        self.assertTrue(all(len(row) == 24 for row in body["map"]))

    def test_dashboard_payload_is_accepted_and_echoed(self):
        payload = {
            "terrain_type": "Bosque", "width": 50, "height": 35,
            "seed": 539989, "algorithm": "perlin", "complexity": "medio",
            "tile_size": 16, "scale": 0.08, "octaves": 4,
            "persistence": 0.5, "threshold": 0.5,
        }
        response = self.client.post("/api/maps/generate", json=payload)
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["dimensions"], {"width": 50, "height": 35})
        self.assertEqual(body["ui_config"]["terrain_type"], "bosque")
        self.assertEqual(body["ui_config"]["complexity"], "medium")
        self.assertEqual(body["ui_config"]["tile_size"], 16)

    def test_dungeon_recommends_cellular_when_algorithm_is_omitted(self):
        response = self.client.post("/api/maps/generate", json={
            "map_type": "mazmorra", "width": 20, "height": 20, "seed": 10,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["algorithm"], "cellular")

    def test_evaluate_valid_map(self):
        response = self.client.post("/api/maps/evaluate", json={"map": [[0, 1], [0, 0]]})
        self.assertEqual(response.status_code, 200)
        self.assertIn("metrics", response.get_json())

    def test_invalid_dashboard_parameter(self):
        response = self.client.post("/api/maps/generate", json={"terrain_type": "mar"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["status"], "error")

    def test_adaptive_endpoint_loads_trained_model(self):
        response = self.client.post("/api/maps/generate-adaptive", json={
            "terrain_type": "mazmorra", "algorithm": "cellular",
            "width": 20, "height": 20, "seed": 539989,
            "complexity": "high", "tile_size": 16,
            "probability": 0.45, "iterations": 4, "attempts": 2,
        })
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["selected_model"], "svm")
        self.assertEqual(body["attempts"], 2)
        self.assertEqual(len(body["map"]), 20)

    def test_cors_preflight_for_vite(self):
        response = self.client.options("/api/maps/generate", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")

    def test_invalid_algorithm(self):
        response = self.client.post("/api/maps/generate", json={"algorithm": "unknown"})
        self.assertEqual(response.status_code, 400)

    def test_experiment_summary(self):
        response = self.client.get("/api/experiment")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["dataset"]["total_maps"], 1000)

    def test_training_map_can_be_inspected(self):
        response = self.client.get("/api/experiment/maps/cellular-0001")
        self.assertEqual(response.status_code, 200)
        training_map = response.get_json()["training_map"]
        self.assertIn("map", training_map)
        self.assertIn("features", training_map)


if __name__ == "__main__":
    unittest.main()
