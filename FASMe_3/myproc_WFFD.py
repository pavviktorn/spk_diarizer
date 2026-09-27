#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image


DEFAULT_IMAGE_DIR = Path("/datasets/work/vLLM/data/WFFD_database/Protocol1/Dev")
DEFAULT_LABEL_PATH = Path("/datasets/work/vLLM/data/WFFD_database/Protocol1/Dev_label.txt")
DEFAULT_DATASET_ROOT = Path("/datasets/work/vLLM/data/WFFD_database")
DEFAULT_OUTPUT_ROOT = Path("/datasets/work/vLLM/data/WFFD_database_out")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".jfif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop the real and wax faces from WFFD images using label-guided left/right assignment."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="WFFD dataset root. When --image-dir/--label-path are omitted, all protocol splits are processed.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional single image folder to process, such as Protocol1/Dev.",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=None,
        help="Optional single label file to process, such as Protocol1/Dev_label.txt.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--crop-scale", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument(
        "--input-sizes",
        type=int,
        nargs="+",
        default=[640, 320, 960],
        help="Square detector sizes used as multi-scale passes.",
    )
    parser.add_argument("--dedupe-iou", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing crops. By default, files are left untouched if already present.",
    )
    return parser.parse_args()


def read_image(image_path: Path) -> np.ndarray | None:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        with Image.open(image_path) as pil_image:
            rgb_image = pil_image.convert("RGB")
    except Exception:
        return None

    return cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)


def write_jpg(image_path: Path, image: np.ndarray) -> bool:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return False
    image_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(image_path))
    return True


def parse_label_file(label_path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with label_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid label line {line_number}: {raw_line.rstrip()}")
            filename, label = parts
            if label not in {"left_wax", "right_wax"}:
                raise ValueError(f"Unknown side label on line {line_number}: {label}")
            items.append((filename, label))
    return items


def split_name_from_label_path(label_path: Path) -> str:
    stem = label_path.stem
    if not stem.endswith("_label"):
        raise ValueError(f"Label file name must end with '_label.txt': {label_path}")
    return stem[:-6]


def discover_jobs(
    dataset_root: Path,
    output_root: Path,
    image_dir: Path | None,
    label_path: Path | None,
) -> List[Dict[str, Path | str]]:
    if (image_dir is None) != (label_path is None):
        raise ValueError("Use --image-dir and --label-path together, or omit both to process the whole dataset.")

    jobs: List[Dict[str, Path | str]] = []
    if image_dir is not None and label_path is not None:
        image_dir = image_dir.expanduser().resolve()
        label_path = label_path.expanduser().resolve()
        if dataset_root in image_dir.parents:
            rel_image_dir = image_dir.relative_to(dataset_root)
            job_output_dir = output_root / rel_image_dir
            job_name = str(rel_image_dir)
        else:
            job_output_dir = output_root
            job_name = image_dir.name

        jobs.append(
            {
                "image_dir": image_dir,
                "label_path": label_path,
                "output_dir": job_output_dir,
                "job_name": job_name,
            }
        )
        return jobs

    for job_label_path in sorted(dataset_root.glob("Protocol*/**/*_label.txt")):
        split_name = split_name_from_label_path(job_label_path)
        job_image_dir = job_label_path.parent / split_name
        rel_dir = job_image_dir.relative_to(dataset_root)
        jobs.append(
            {
                "image_dir": job_image_dir,
                "label_path": job_label_path,
                "output_dir": output_root / rel_dir,
                "job_name": str(rel_dir),
            }
        )

    if not jobs:
        raise RuntimeError(f"No label files found under {dataset_root}")
    return jobs


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = {}
    for image_path in image_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        key = image_path.name.lower()
        if key in index:
            duplicates.setdefault(key, [str(index[key])]).append(str(image_path))
            continue
        index[key] = image_path

    if duplicates:
        dup_preview = []
        for key, paths in sorted(duplicates.items()):
            dup_preview.append(f"{key}: {paths}")
        raise RuntimeError("Duplicate image names detected:\n" + "\n".join(dup_preview))
    return index


def download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as outfile:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            outfile.write(chunk)


def ensure_yunet_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    print(f"YuNet model not found. Downloading to {model_path} ...")
    try:
        download_file(DEFAULT_MODEL_URL, model_path)
    except URLError as exc:
        raise RuntimeError(
            f"Failed to download YuNet model from {DEFAULT_MODEL_URL}. "
            f"Set --model-path to an existing model file. Details: {exc}"
        ) from exc

    return model_path


def normalize_image(frame: np.ndarray) -> np.ndarray:
    if frame.ndim < 3:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _get_new_box(src_w: int, src_h: int, bbox: Sequence[int], scale: float) -> Tuple[int, int, int, int]:
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    new_width = box_w * scale
    new_height = box_h * scale
    center_x = box_w / 2 + x
    center_y = box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        left_top_x = 0
    if left_top_y < 0:
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        right_bottom_y = src_h - 1

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop_face(image: np.ndarray, bbox: Sequence[int], scale: float) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)
    return image[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]


def compute_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


def box_center(box: Sequence[int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_size(box: Sequence[int]) -> Tuple[float, float]:
    return (max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1]))


def box_touches_edge(box: Sequence[int], image_w: int, image_h: int, margin_ratio: float = 0.02) -> bool:
    margin = max(2.0, margin_ratio * max(image_w, image_h))
    return (
        box[0] <= margin
        or box[1] <= margin
        or image_w - box[2] <= margin
        or image_h - box[3] <= margin
    )


def prepare_square_image(frame: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, int, int]:
    frame = normalize_image(frame)
    exp_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    if frame.shape[0] >= frame.shape[1]:
        new_h = target_size
        new_w = int(frame.shape[1] * target_size / frame.shape[0])
    else:
        new_w = target_size
        new_h = int(frame.shape[0] * target_size / frame.shape[1])

    scale = new_h / frame.shape[0] if frame.shape[0] >= frame.shape[1] else new_w / frame.shape[1]
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    exp_img[top:top + new_h, left:left + new_w] = resized
    return exp_img, scale, top, left


class MultiScaleYuNetDetector:
    def __init__(
        self,
        model_path: Path,
        input_sizes: Iterable[int],
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
        dedupe_iou: float,
    ) -> None:
        self.input_sizes = [int(size) for size in input_sizes]
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.dedupe_iou = dedupe_iou
        self.detectors = {
            size: cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (size, size),
                score_threshold,
                nms_threshold,
                top_k,
            )
            for size in self.input_sizes
        }

    def _detect_at_size(self, frame: np.ndarray, target_size: int) -> List[Dict[str, float | List[int]]]:
        padded, scale, top, left = prepare_square_image(frame, target_size)
        detector = self.detectors[target_size]
        detector.setInputSize((target_size, target_size))
        _, faces = detector.detect(padded)
        if faces is None:
            return []

        image_h, image_w = frame.shape[:2]
        detections: List[Dict[str, float | List[int]]] = []
        for row in faces:
            x, y, w, h = row[:4]
            bbox = [
                int(round((x - left) / scale)),
                int(round((y - top) / scale)),
                int(round((x + w - left) / scale)),
                int(round((y + h - top) / scale)),
            ]
            bbox[0] = max(0, min(image_w - 1, bbox[0]))
            bbox[1] = max(0, min(image_h - 1, bbox[1]))
            bbox[2] = max(0, min(image_w - 1, bbox[2]))
            bbox[3] = max(0, min(image_h - 1, bbox[3]))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            detections.append(
                {
                    "bbox": bbox,
                    "score": float(row[14]),
                    "area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                }
            )
        return detections

    def _pair_score(
        self,
        left_det: Dict[str, float | List[int]],
        right_det: Dict[str, float | List[int]],
        image_w: int,
        image_h: int,
        max_area: float,
    ) -> Tuple[float, float, float, float]:
        left_box = left_det["bbox"]
        right_box = right_det["bbox"]
        left_area = float(left_det["area"])
        right_area = float(right_det["area"])
        total_area = max(1.0, float(image_w * image_h))

        left_cx, left_cy = box_center(left_box)
        right_cx, right_cy = box_center(right_box)
        left_w, left_h = box_size(left_box)
        right_w, right_h = box_size(right_box)

        combined_area_score = (left_area + right_area) / total_area
        size_balance = min(left_area, right_area) / max(left_area, right_area)
        vertical_alignment = max(0.0, 1.0 - abs(left_cy - right_cy) / max(left_h, right_h))
        horizontal_gap = abs(right_cx - left_cx) / max((left_w + right_w) / 2.0, 1.0)
        edge_penalty = 0.0
        for det in (left_det, right_det):
            box = det["bbox"]
            area_ratio = float(det["area"]) / max_area
            if area_ratio < 0.65 and box_touches_edge(box, image_w, image_h):
                edge_penalty += 0.35

        score = (
            3.0 * combined_area_score
            + 1.6 * size_balance
            + 1.2 * vertical_alignment
            + 0.2 * min(horizontal_gap, 3.0)
            + 0.3 * (float(left_det["score"]) + float(right_det["score"])) / 2.0
            - edge_penalty
        )
        return score, combined_area_score, size_balance, vertical_alignment

    def _select_best_pair(
        self, detections: List[Dict[str, float | List[int]]], image_w: int, image_h: int
    ) -> List[List[int]]:
        if len(detections) < 2:
            return []

        max_area = max(float(det["area"]) for det in detections)
        preferred = [det for det in detections if float(det["area"]) >= 0.45 * max_area]
        candidates = preferred if len(preferred) >= 2 else detections

        best_pair: Tuple[Dict[str, float | List[int]], Dict[str, float | List[int]]] | None = None
        best_score: Tuple[float, float, float, float] | None = None
        for left_idx in range(len(candidates) - 1):
            for right_idx in range(left_idx + 1, len(candidates)):
                pair = sorted(
                    (candidates[left_idx], candidates[right_idx]),
                    key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0,
                )
                score = self._pair_score(pair[0], pair[1], image_w, image_h, max_area)
                if best_score is None or score > best_score:
                    best_pair = (pair[0], pair[1])
                    best_score = score

        if best_pair is None:
            return []
        return [list(best_pair[0]["bbox"]), list(best_pair[1]["bbox"])]

    def detect_two_faces(self, frame: np.ndarray) -> List[List[int]]:
        merged: List[Dict[str, float | List[int]]] = []
        for size in self.input_sizes:
            merged.extend(self._detect_at_size(frame, size))

        merged.sort(key=lambda item: (item["score"], item["area"]), reverse=True)

        deduped: List[Dict[str, float | List[int]]] = []
        for detection in merged:
            bbox = detection["bbox"]
            if all(compute_iou(bbox, kept["bbox"]) < self.dedupe_iou for kept in deduped):
                deduped.append(detection)

        image_h, image_w = frame.shape[:2]
        return self._select_best_pair(deduped, image_w, image_h)


def resolve_face_roles(face_boxes: Sequence[Sequence[int]], side_label: str) -> Tuple[List[int], List[int]]:
    if len(face_boxes) != 2:
        raise ValueError(f"Expected exactly 2 face boxes, got {len(face_boxes)}")
    left_face, right_face = face_boxes
    if side_label == "left_wax":
        return list(right_face), list(left_face)
    return list(left_face), list(right_face)


def process_split(
    image_dir: Path,
    label_path: Path,
    output_dir: Path,
    detector: MultiScaleYuNetDetector,
    args: argparse.Namespace,
) -> Tuple[int, List[Tuple[str, str]]]:
    labels = parse_label_file(label_path)
    image_index = build_image_index(image_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped: List[Tuple[str, str]] = []
    total = len(labels) if args.limit is None else min(len(labels), args.limit)

    for index, (filename, side_label) in enumerate(labels[:total], start=1):
        image_path = image_index.get(filename.lower())
        if image_path is None:
            skipped.append((filename, "image not found"))
            print(f"[{index}/{total}] skip {filename}: image not found")
            continue

        image = read_image(image_path)
        if image is None:
            skipped.append((filename, "failed to read image"))
            print(f"[{index}/{total}] skip {filename}: failed to read image")
            continue

        try:
            face_boxes = detector.detect_two_faces(image)
            real_box, wax_box = resolve_face_roles(face_boxes, side_label)
        except Exception as exc:
            skipped.append((filename, str(exc)))
            print(f"[{index}/{total}] skip {filename}: {exc}")
            continue

        real_crop = crop_face(image, real_box, args.crop_scale)
        wax_crop = crop_face(image, wax_box, args.crop_scale)

        if real_crop.size == 0 or wax_crop.size == 0:
            skipped.append((filename, "empty crop"))
            print(f"[{index}/{total}] skip {filename}: empty crop")
            continue

        output_stem = Path(filename).stem
        real_output = output_dir / f"{output_stem}_real.jpg"
        wax_output = output_dir / f"{output_stem}_fake.jpg"

        if not args.overwrite and real_output.exists() and wax_output.exists():
            processed += 1
            print(f"[{index}/{total}] keep existing {filename}")
            continue

        ok_real = write_jpg(real_output, real_crop)
        ok_wax = write_jpg(wax_output, wax_crop)
        if not (ok_real and ok_wax):
            skipped.append((filename, "failed to write crop"))
            print(f"[{index}/{total}] skip {filename}: failed to write crop")
            continue

        processed += 1
        print(f"[{index}/{total}] processed {filename}")

    return processed, skipped


def process_dataset(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    model_path = ensure_yunet_model(args.model_path.expanduser().resolve())
    jobs = discover_jobs(dataset_root, output_root, args.image_dir, args.label_path)

    detector = MultiScaleYuNetDetector(
        model_path=model_path,
        input_sizes=args.input_sizes,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        dedupe_iou=args.dedupe_iou,
    )

    total_processed = 0
    total_skipped: List[Tuple[str, str]] = []

    for job in jobs:
        image_dir = job["image_dir"]
        label_path = job["label_path"]
        output_dir = job["output_dir"]
        job_name = job["job_name"]

        print()
        print(f"=== Processing {job_name} ===")
        processed, skipped = process_split(image_dir, label_path, output_dir, detector, args)
        total_processed += processed
        total_skipped.extend([(f"{job_name}/{filename}", reason) for filename, reason in skipped])
        print(f"Completed {job_name}: processed={processed}, skipped={len(skipped)}")

    print()
    print(f"Processed: {total_processed}")
    print(f"Skipped: {len(total_skipped)}")
    if total_skipped:
        print("Skipped details:")
        for filename, reason in total_skipped:
            print(f"  {filename}: {reason}")

    return 0 if not total_skipped else 1


def main() -> int:
    args = parse_args()
    try:
        return process_dataset(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
