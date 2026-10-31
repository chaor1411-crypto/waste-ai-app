import base64
import csv
import hashlib
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import requests
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="AI 생활폐기물 분리배출 시스템",
    page_icon="♻️",
    layout="centered"
)


# ============================================================
# 2. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).parent

YOLO_PATH = BASE_DIR / "models" / "best.pt"
CLASSIFIER_PATH = (
    BASE_DIR
    / "models"
    / "final_waste_classifier.keras"
)

FEEDBACK_DIR = BASE_DIR / "feedback_data"
FEEDBACK_CSV = FEEDBACK_DIR / "feedback_log.csv"


# ============================================================
# 3. 클래스 정보
# ============================================================

CLASS_NAMES = [
    "can",
    "glass",
    "paper",
    "pet",
    "plastic",
    "styrofoam"
]

KOREAN_NAMES = {
    "can": "캔",
    "glass": "유리병",
    "paper": "종이",
    "pet": "페트병",
    "plastic": "플라스틱",
    "styrofoam": "스티로폼"
}

DISPOSAL_GUIDE = {
    "can":
        "내용물을 비우고 이물질을 제거한 뒤 "
        "캔류로 배출하세요.",

    "glass":
        "내용물을 비우고 뚜껑을 분리한 뒤 "
        "유리병류로 배출하세요.",

    "paper":
        "비닐, 테이프, 금속 등 다른 재질을 제거한 뒤 "
        "종이류로 배출하세요.",

    "pet":
        "내용물을 비우고 깨끗하게 세척한 뒤 "
        "라벨과 뚜껑을 분리하여 페트병으로 배출하세요.",

    "plastic":
        "내용물을 비우고 깨끗하게 세척한 뒤 "
        "플라스틱류로 배출하세요.",

    "styrofoam":
        "내용물과 이물질을 제거한 뒤 "
        "깨끗한 상태로 스티로폼류에 배출하세요."
}


# ============================================================
# 4. 피드백 폴더 준비
# ============================================================

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

for class_name in CLASS_NAMES:
    (FEEDBACK_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 5. 모델 로딩
# ============================================================

@st.cache_resource
def load_models():

    yolo = YOLO(str(YOLO_PATH))

    classifier = tf.keras.models.load_model(
        str(CLASSIFIER_PATH)
    )

    return yolo, classifier


try:

    yolo_model, classifier = load_models()

except Exception as e:

    st.error("AI 모델을 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ============================================================
# 6. 중복 Bounding Box 제거 함수
# ============================================================

def intersection_over_smaller(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area1 = max(
        1,
        (box1[2] - box1[0])
        * (box1[3] - box1[1])
    )

    area2 = max(
        1,
        (box2[2] - box2[0])
        * (box2[3] - box2[1])
    )

    smaller = min(area1, area2)

    return intersection / smaller


def remove_duplicate_boxes(boxes):

    boxes = sorted(
        boxes,
        key=lambda x: x["yolo_confidence"],
        reverse=True
    )

    filtered = []

    for candidate in boxes:

        duplicate = False

        for kept in filtered:

            overlap = intersection_over_smaller(
                candidate["box"],
                kept["box"]
            )

            if overlap >= 0.70:
                duplicate = True
                break

        if not duplicate:
            filtered.append(candidate)

    return filtered


# ============================================================
# 7. 피드백 저장 함수
#    - 중앙 저장소가 설정되어 있으면 Google Drive/Sheets로 전송
#    - 설정이 없거나 전송 실패 시 로컬 feedback_data에 임시 저장
# ============================================================

def get_central_feedback_config():
    """Streamlit secrets에서 중앙 피드백 서버 설정을 읽습니다."""

    try:
        feedback_config = st.secrets["feedback"]

        webhook_url = str(
            feedback_config.get(
                "webhook_url",
                ""
            )
        ).strip()

        token = str(
            feedback_config.get(
                "token",
                ""
            )
        ).strip()

        return webhook_url, token

    except Exception:
        # secrets.toml이 아직 없더라도 앱 자체는 정상 실행
        return "", ""


def save_feedback_local(
    item,
    correct_label,
    feedback_type,
    now,
    image_name
):
    """중앙 저장이 불가능할 때 현재 PC에 피드백을 저장합니다."""

    save_dir = (
        FEEDBACK_DIR
        / correct_label
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        save_dir
        / image_name
    )

    crop = item["crop"]

    if not isinstance(crop, Image.Image):
        crop = Image.fromarray(
            np.asarray(crop).astype(np.uint8)
        )

    crop.convert("RGB").save(
        image_path,
        quality=95
    )

    file_exists = FEEDBACK_CSV.exists()

    with open(
        FEEDBACK_CSV,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow(
                [
                    "time",
                    "feedback_type",
                    "yolo_prediction",
                    "yolo_confidence",
                    "mobilenet_prediction",
                    "mobilenet_confidence",
                    "final_prediction",
                    "correct_label",
                    "image_path"
                ]
            )

        writer.writerow(
            [
                now.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                feedback_type,
                item["yolo_class"],
                round(
                    float(
                        item["yolo_confidence"]
                    ),
                    4
                ),
                item["classifier_class"],
                round(
                    float(
                        item[
                            "classifier_confidence"
                        ]
                    ),
                    4
                ),
                item["final_class"]
                if item["final_class"] is not None
                else "hold",
                correct_label,
                str(image_path)
            ]
        )

    return {
        "mode": "local",
        "location": str(image_path)
    }


def save_feedback(
    item,
    correct_label,
    feedback_type
):
    """
    사용자 피드백 저장.

    중앙 피드백 Web App URL이 설정되어 있으면
    Google Drive + Google Sheets로 전송합니다.
    전송에 실패하면 로컬 저장으로 자동 전환합니다.
    """

    now = datetime.now()

    timestamp = now.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_name = (
        f"{feedback_type}_{timestamp}.jpg"
    )

    webhook_url, token = (
        get_central_feedback_config()
    )

    # --------------------------------------------------------
    # 중앙 저장 시도
    # --------------------------------------------------------

    central_error = None

    if webhook_url:

        try:

            crop = item["crop"]

            if not isinstance(
                crop,
                Image.Image
            ):
                crop = Image.fromarray(
                    np.asarray(
                        crop
                    ).astype(
                        np.uint8
                    )
                )

            # 전송 용량을 줄이기 위해 최대 640px로 압축
            upload_crop = crop.convert(
                "RGB"
            ).copy()

            upload_crop.thumbnail(
                (640, 640)
            )

            buffer = io.BytesIO()

            upload_crop.save(
                buffer,
                format="JPEG",
                quality=85,
                optimize=True
            )

            image_base64 = (
                base64.b64encode(
                    buffer.getvalue()
                ).decode("ascii")
            )

            payload = {
                "token": token,
                "time": now.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "feedback_type": feedback_type,
                "yolo_prediction": item[
                    "yolo_class"
                ],
                "yolo_confidence": round(
                    float(
                        item[
                            "yolo_confidence"
                        ]
                    ),
                    4
                ),
                "mobilenet_prediction": item[
                    "classifier_class"
                ],
                "mobilenet_confidence": round(
                    float(
                        item[
                            "classifier_confidence"
                        ]
                    ),
                    4
                ),
                "final_prediction": (
                    item["final_class"]
                    if item[
                        "final_class"
                    ] is not None
                    else "hold"
                ),
                "correct_label": correct_label,
                "image_name": image_name,
                "image_base64": image_base64
            }

            response = requests.post(
                webhook_url,
                json=payload,
                timeout=25
            )

            response.raise_for_status()

            result = response.json()

            if not result.get(
                "ok",
                False
            ):
                raise RuntimeError(
                    result.get(
                        "error",
                        "중앙 저장소가 오류를 반환했습니다."
                    )
                )

            return {
                "mode": "central",
                "location": result.get(
                    "file_url",
                    "Google Drive"
                )
            }

        except Exception as e:
            central_error = str(e)

    # --------------------------------------------------------
    # 중앙 설정이 없거나 중앙 저장 실패 → 로컬 임시 저장
    # --------------------------------------------------------

    local_result = save_feedback_local(
        item=item,
        correct_label=correct_label,
        feedback_type=feedback_type,
        now=now,
        image_name=image_name
    )

    if central_error:
        local_result[
            "central_error"
        ] = central_error

    return local_result


# ============================================================
# 8. 이미지 분석 함수
# ============================================================

def analyze_image(image):

    image_np = np.array(image)

    try:

        results = yolo_model.predict(
            source=image_np,
            imgsz=640,
            conf=0.35,
            iou=0.45,
            agnostic_nms=True,
            max_det=10,
            verbose=False
        )

    except Exception as e:

        st.error(
            "폐기물 탐지 과정에서 오류가 발생했습니다."
        )

        st.exception(e)
        st.stop()

    result = results[0]

    raw_boxes = []

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            image.width,
            x2
        )

        y2 = min(
            image.height,
            y2
        )

        if x2 <= x1 or y2 <= y1:
            continue

        yolo_class_id = int(
            box.cls[0]
        )

        yolo_class = (
            yolo_model.names[
                yolo_class_id
            ]
        )

        yolo_confidence = float(
            box.conf[0]
        )

        raw_boxes.append(
            {
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "yolo_class":
                    yolo_class,
                "yolo_confidence":
                    yolo_confidence
            }
        )

    clean_boxes = remove_duplicate_boxes(
        raw_boxes
    )

    output_image = image.copy()

    draw = ImageDraw.Draw(
        output_image
    )

    detected_objects = []

    for item in clean_boxes:

        x1, y1, x2, y2 = (
            item["box"]
        )

        yolo_class = (
            item["yolo_class"]
        )

        yolo_confidence = (
            item["yolo_confidence"]
        )

        crop = image.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

        crop_resized = crop.resize(
            (
                320,
                320
            )
        )

        crop_array = np.array(
            crop_resized,
            dtype=np.float32
        )

        crop_array = np.expand_dims(
            crop_array,
            axis=0
        )

        try:

            probabilities = (
                classifier.predict(
                    crop_array,
                    verbose=0
                )[0]
            )

        except Exception as e:

            st.error(
                "재질 분류 과정에서 오류가 발생했습니다."
            )

            st.exception(e)
            st.stop()

        classifier_id = int(
            np.argmax(probabilities)
        )

        classifier_class = (
            CLASS_NAMES[
                classifier_id
            ]
        )

        classifier_confidence = float(
            probabilities[
                classifier_id
            ]
        )

        # ----------------------------------------------------
        # YOLO + MobileNet 결과 통합
        # ----------------------------------------------------

        final_class = None
        decision = ""

        if classifier_confidence < 0.60:

            final_class = None

            decision = (
                "AI 확신도가 낮아 "
                "판단을 보류했습니다."
            )

        elif yolo_class == classifier_class:

            final_class = classifier_class

            decision = (
                "두 AI 모델의 판단이 일치했습니다."
            )

        elif classifier_confidence >= 0.90:

            final_class = classifier_class

            decision = (
                "객체 탐지 모델과 재질 분류 모델의 "
                "판단은 달랐지만 재질 분류 확신도가 "
                "높아 최종 판정에 반영했습니다."
            )

        else:

            final_class = None

            decision = (
                "두 AI 모델의 판단이 달라 "
                "최종 판단을 보류했습니다."
            )

        detected_objects.append(
            {
                "final_class":
                    final_class,

                "classifier_class":
                    classifier_class,

                "classifier_confidence":
                    classifier_confidence,

                "yolo_class":
                    yolo_class,

                "yolo_confidence":
                    yolo_confidence,

                "decision":
                    decision,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                # 사용자가 정답을 수정하면
                # 실제 재학습용 이미지로 저장할 crop
                "crop":
                    crop.copy()
            }
        )

        # ----------------------------------------------------
        # Bounding Box 표시
        # ----------------------------------------------------

        if final_class is None:

            box_color = "orange"

            label_text = (
                "CHECK "
                f"{classifier_confidence * 100:.1f}%"
            )

        else:

            box_color = "lime"

            label_text = (
                f"{final_class} "
                f"{classifier_confidence * 100:.1f}%"
            )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline=box_color,
            width=5
        )

        draw.text(
            (
                x1 + 5,
                y1 + 5
            ),
            label_text,
            fill=box_color
        )

    return {
        "output_image": output_image,
        "detected_objects": detected_objects
    }


# ============================================================
# 9. 피드백 UI 함수
# ============================================================

def feedback_saved_message(
    result,
    correct_label=None
):
    """저장 위치에 따라 사용자에게 보여줄 메시지를 만듭니다."""

    if correct_label is None:
        prefix = "✅ 확인 결과"
    else:
        prefix = (
            "✅ 실제 품목 "
            f"'{KOREAN_NAMES[correct_label]}'"
        )

    if result["mode"] == "central":
        return (
            f"{prefix}가 중앙 학습 데이터에 저장되었습니다."
        )

    if "central_error" in result:
        return (
            f"{prefix}는 현재 PC에 임시 저장되었습니다. "
            "중앙 저장소 연결을 확인해주세요."
        )

    return (
        f"{prefix}가 현재 PC에 저장되었습니다."
    )


def show_feedback_ui(
    item,
    object_index,
    image_key
):

    key_base = (
        f"{image_key}_{object_index}"
    )

    wrong_key = (
        f"feedback_wrong_{key_base}"
    )

    message_key = (
        f"feedback_message_{key_base}"
    )

    if wrong_key not in st.session_state:
        st.session_state[wrong_key] = False

    st.markdown("---")
    st.markdown("#### 🤖 이 결과가 맞나요?")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # 맞아요
    # --------------------------------------------------------

    with col1:

        if item["final_class"] is not None:

            if st.button(
                "✅ 맞아요",
                key=f"correct_{key_base}",
                use_container_width=True
            ):

                result = save_feedback(
                    item=item,
                    correct_label=item[
                        "final_class"
                    ],
                    feedback_type="correct"
                )

                st.session_state[
                    message_key
                ] = feedback_saved_message(
                    result=result
                )

                st.session_state[
                    wrong_key
                ] = False

        else:

            st.button(
                "✅ 맞아요",
                key=f"correct_disabled_{key_base}",
                use_container_width=True,
                disabled=True
            )

    # --------------------------------------------------------
    # 틀렸어요
    # --------------------------------------------------------

    with col2:

        if st.button(
            "❌ 틀렸어요",
            key=f"wrong_{key_base}",
            use_container_width=True
        ):

            st.session_state[
                wrong_key
            ] = True

            st.session_state.pop(
                message_key,
                None
            )

    # --------------------------------------------------------
    # 실제 정답 선택
    # --------------------------------------------------------

    if st.session_state[
        wrong_key
    ]:

        st.info(
            "실제 폐기물 종류를 선택하면 "
            "해당 사진이 재학습용 데이터로 저장됩니다."
        )

        correct_label = st.selectbox(
            "실제 폐기물 종류",
            options=CLASS_NAMES,
            format_func=lambda x:
                KOREAN_NAMES[x],
            key=f"actual_{key_base}"
        )

        if st.button(
            "💾 수정 내용 저장",
            key=f"save_{key_base}",
            type="primary",
            use_container_width=True
        ):

            result = save_feedback(
                item=item,
                correct_label=correct_label,
                feedback_type="incorrect"
            )

            st.session_state[
                message_key
            ] = feedback_saved_message(
                result=result,
                correct_label=correct_label
            )

            st.session_state[
                wrong_key
            ] = False

    if message_key in st.session_state:

        st.success(
            st.session_state[
                message_key
            ]
        )


# ============================================================
# 10. 결과 화면 함수
# ============================================================

def show_analysis_result(
    analysis_result,
    image_key
):

    detected_objects = (
        analysis_result[
            "detected_objects"
        ]
    )

    output_image = (
        analysis_result[
            "output_image"
        ]
    )

    if len(detected_objects) == 0:

        st.warning(
            "⚠️ 폐기물을 탐지하지 못했습니다.\n\n"
            "폐기물이 화면 중앙에 크게 보이도록 "
            "다시 촬영해주세요."
        )

        return

    st.success(
        f"✅ 총 {len(detected_objects)}개의 "
        "폐기물 후보를 탐지했습니다."
    )

    st.markdown(
        "### AI 탐지 결과"
    )

    st.image(
        output_image,
        use_container_width=True
    )

    st.markdown(
        "## ♻️ 분리배출 안내"
    )

    for index, item in enumerate(
        detected_objects,
        start=1
    ):

        final_class = (
            item["final_class"]
        )

        confidence = (
            item[
                "classifier_confidence"
            ]
        )

        with st.container(
            border=True
        ):

            if final_class is not None:

                korean_name = (
                    KOREAN_NAMES[
                        final_class
                    ]
                )

                st.markdown(
                    f"### 폐기물 {index} : "
                    f"{korean_name}"
                )

                st.write(
                    f"**AI 확신도 : "
                    f"{confidence * 100:.1f}%**"
                )

                st.progress(
                    min(
                        max(
                            confidence,
                            0.0
                        ),
                        1.0
                    )
                )

                st.markdown(
                    "**분리배출 방법**"
                )

                st.success(
                    DISPOSAL_GUIDE[
                        final_class
                    ]
                )

            else:

                st.markdown(
                    f"### 폐기물 {index} : "
                    "판단 보류"
                )

                st.write(
                    f"**AI 확신도 : "
                    f"{confidence * 100:.1f}%**"
                )

                st.progress(
                    min(
                        max(
                            confidence,
                            0.0
                        ),
                        1.0
                    )
                )

                st.warning(
                    "⚠️ 현재 사진만으로는 "
                    "폐기물 종류를 확실하게 판단하기 어렵습니다.\n\n"
                    "아래의 '틀렸어요'를 눌러 실제 품목을 알려주거나, "
                    "폐기물이 화면 중앙에 크게 보이도록 다시 촬영해주세요."
                )

            with st.expander(
                "🔎 AI 분석 상세보기"
            ):

                st.write(
                    "YOLO 객체 탐지:",
                    f"{item['yolo_class']} "
                    f"({item['yolo_confidence'] * 100:.1f}%)"
                )

                st.write(
                    "MobileNet 재질 분류:",
                    f"{item['classifier_class']} "
                    f"({item['classifier_confidence'] * 100:.1f}%)"
                )

                st.write(
                    "판단 과정:",
                    item["decision"]
                )

            show_feedback_ui(
                item=item,
                object_index=index,
                image_key=image_key
            )


# ============================================================
# 11. 제목
# ============================================================

st.title(
    "♻️ AI 생활폐기물 분리배출 시스템"
)

st.write(
    "폐기물 사진을 촬영하거나 업로드하면 "
    "AI가 폐기물을 탐지하고 재질을 판별하여 "
    "올바른 분리배출 방법을 안내합니다."
)


# ============================================================
# 12. 사용 방법
# ============================================================

st.markdown("### 사용 방법")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "📷 **1. 사진 입력**\n\n"
        "폐기물을 촬영하거나 사진을 업로드하세요."
    )

with col2:

    st.info(
        "🤖 **2. AI 분석**\n\n"
        "폐기물의 위치와 재질을 분석합니다."
    )

with col3:

    st.info(
        "♻️ **3. 배출 안내**\n\n"
        "올바른 분리배출 방법을 확인하세요."
    )

st.divider()


# ============================================================
# 13. 사진 입력
# ============================================================

tab_camera, tab_upload = st.tabs(
    [
        "📷 카메라 촬영",
        "🖼️ 사진 업로드"
    ]
)

image_file = None

with tab_camera:

    camera_file = st.camera_input(
        "생활폐기물이 화면에 잘 보이도록 촬영해주세요."
    )

    if camera_file is not None:
        image_file = camera_file

with tab_upload:

    uploaded_file = st.file_uploader(
        "폐기물 사진을 선택해주세요.",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:
        image_file = uploaded_file


# ============================================================
# 14. 사진이 입력된 경우
# ============================================================

if image_file is not None:

    # 같은 이미지에서 피드백 버튼을 눌렀을 때
    # 분석 결과가 사라지지 않도록 이미지 고유값 생성
    file_bytes = image_file.getvalue()

    image_key = hashlib.md5(
        file_bytes
    ).hexdigest()[:12]

    image = Image.open(
        image_file
    )

    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    st.markdown(
        "### 입력 이미지"
    )

    st.image(
        image,
        use_container_width=True
    )

    # 새로운 사진이 들어오면
    # 이전 사진의 분석 결과를 제거
    if (
        st.session_state.get(
            "analysis_image_key"
        )
        != image_key
    ):

        st.session_state[
            "analysis_image_key"
        ] = image_key

        st.session_state.pop(
            "analysis_result",
            None
        )

    # --------------------------------------------------------
    # 분석 버튼
    # --------------------------------------------------------

    if st.button(
        "🔍 AI 분리배출 분석",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "AI가 폐기물을 분석하고 있습니다..."
        ):

            analysis_result = (
                analyze_image(
                    image
                )
            )

            st.session_state[
                "analysis_result"
            ] = analysis_result

            st.session_state[
                "analysis_image_key"
            ] = image_key

    # --------------------------------------------------------
    # 분석 결과는 session_state에서 출력
    # → 피드백 버튼 클릭 후에도 결과가 유지됨
    # --------------------------------------------------------

    if (
        st.session_state.get(
            "analysis_image_key"
        )
        == image_key
        and
        "analysis_result"
        in st.session_state
    ):

        show_analysis_result(
            analysis_result=st.session_state[
                "analysis_result"
            ],
            image_key=image_key
        )


# ============================================================
# 15. 하단 안내
# ============================================================

st.divider()

st.caption(
    "※ 본 시스템은 AI 기반 생활폐기물 분리배출 지원 프로토타입입니다. "
    "실제 배출 시에는 지역별 분리배출 기준을 함께 확인해주세요."
)
