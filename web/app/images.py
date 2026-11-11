"""Upload validation and the smaller copies shown in the interface."""

import cv2
import numpy as np

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# By content, not by the browser's Content-Type, which clients may omit.
SIGNATURES = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG\r\n\x1a\n": ".png",
    b"RIFF": ".webp",  # followed by size, then b"WEBP" (checked below)
}
# Long side of the copies: the result page and the history list.
VIEW_SIZE = 1280
THUMB_SIZE = 240


class InvalidImage(ValueError):
    pass


def extension(data: bytes) -> str:
    for signature, ext in SIGNATURES.items():
        if data.startswith(signature):
            if ext == ".webp" and data[8:12] != b"WEBP":
                break
            return ext
    raise InvalidImage("Formato não suportado: envie uma foto JPEG, PNG ou WebP.")


def decode(data: bytes) -> np.ndarray:
    """BGR image, rotated by its EXIF orientation (phones store photos
    sideways and set the flag; IMREAD_COLOR applies it)."""
    if not data:
        raise InvalidImage("Arquivo vazio.")
    if len(data) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise InvalidImage(f"Foto grande demais (máximo {mb} MB).")
    extension(data)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage(
            "Não foi possível abrir a imagem: o arquivo está corrompido?"
        )
    return image


def downscale(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = size / max(height, width)
    if scale >= 1:
        return image
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality=85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InvalidImage("Não foi possível converter a imagem.")
    return buffer.tobytes()
