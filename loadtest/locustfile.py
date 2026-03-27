import json
import os
from pathlib import Path

from locust import HttpUser, between, task


TARGET_IMAGE = os.getenv("TARGET_IMAGE", "data/viz_01_class_distribution.png")
THRESHOLD = os.getenv("PRED_THRESHOLD")


def _read_image_bytes() -> bytes:
    image_path = Path(TARGET_IMAGE)
    if not image_path.exists():
        raise FileNotFoundError(
            f"TARGET_IMAGE not found: {image_path}. "
            "Set TARGET_IMAGE env var to a valid .jpg/.jpeg/.png file."
        )
    return image_path.read_bytes()


class ModelUser(HttpUser):
    # Small wait to mimic real user pacing while still generating heavy load.
    wait_time = between(0.1, 0.3)

    def on_start(self):
        self.image_bytes = _read_image_bytes()
        self.filename = Path(TARGET_IMAGE).name

    @task(1)
    def health(self):
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status={response.status_code}")

    @task(3)
    def predict(self):
        files = {
            "file": (self.filename, self.image_bytes, "image/png"),
        }
        params = {}
        if THRESHOLD:
            params["threshold"] = THRESHOLD

        with self.client.post(
            "/predict",
            files=files,
            params=params,
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:300]}")
                return

            try:
                payload = response.json()
            except json.JSONDecodeError:
                response.failure("Response is not valid JSON")
                return

            required_keys = {"label", "confidence", "prob_benign", "prob_malignant", "inference_time_ms"}
            missing = required_keys - set(payload.keys())
            if missing:
                response.failure(f"Missing response keys: {sorted(missing)}")
