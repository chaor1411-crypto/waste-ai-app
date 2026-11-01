"""
TensorFlow 전용 분류 worker.
이 파일에서는 Ultralytics / PyTorch를 import하지 않습니다.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


RESULT_PREFIX = "__CLASSIFIER_RESULT__"


def emit_result(data):

    print(
        RESULT_PREFIX
        + json.dumps(
            data,
            ensure_ascii=False
        ),
        flush=True
    )


def main():

    try:

        payload_text = sys.stdin.read()

        if not payload_text.strip():
            raise ValueError(
                "분류 요청 데이터가 비어 있습니다."
            )

        payload = json.loads(payload_text)

        model_path = Path(
            payload["model_path"]
        )

        image_paths = [
            Path(path)
            for path in payload["image_paths"]
        ]

        input_size = int(
            payload.get(
                "input_size",
                320
            )
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"분류 모델을 찾을 수 없습니다: {model_path}"
            )

        if not image_paths:

            emit_result(
                {
                    "ok": True,
                    "predictions": []
                }
            )
            return

        try:

            tf.config.threading.set_inter_op_parallelism_threads(
                1
            )
            tf.config.threading.set_intra_op_parallelism_threads(
                1
            )

        except RuntimeError:
            pass

        classifier = tf.keras.models.load_model(
            str(model_path),
            compile=False
        )

        batch = []

        for image_path in image_paths:

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Crop 이미지를 찾을 수 없습니다: {image_path}"
                )

            with Image.open(image_path) as image:

                image = (
                    image
                    .convert("RGB")
                    .resize(
                        (
                            input_size,
                            input_size
                        )
                    )
                )

                batch.append(
                    np.array(
                        image,
                        dtype=np.float32
                    )
                )

        batch_array = np.stack(
            batch,
            axis=0
        )

        predictions = classifier.predict(
            batch_array,
            verbose=0
        )

        emit_result(
            {
                "ok": True,
                "predictions": (
                    np.asarray(
                        predictions,
                        dtype=np.float32
                    ).tolist()
                )
            }
        )

    except Exception as e:

        emit_result(
            {
                "ok": False,
                "error": (
                    f"{type(e).__name__}: {e}"
                )
            }
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
