"""세포 중심 crop 추출과 zarr 저장 (사양 A.1).

crop: centroid 중심 crop_size×crop_size, 전 채널 원본 dtype. crop 안의 모든 세포 instance mask와
같은 라벨 체계의 nucleus mask를 1..K로 재라벨해 함께 저장하고, 중심 세포의 새 라벨을 기록한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_crop(image: np.ndarray, nuc: np.ndarray, cell: np.ndarray, cy: float, cx: float, label: int,
                 crop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """image [C,H,W], 매칭된 nuc/cell 마스크 [H,W] → (img, nuc, cell, center_label). 경계를 넘으면 ValueError."""
    half = crop_size // 2
    y0, x0 = int(np.rint(cy)) - half, int(np.rint(cx)) - half
    h, w = cell.shape
    if y0 < 0 or x0 < 0 or y0 + crop_size > h or x0 + crop_size > w:
        raise ValueError("crop이 FOV 경계를 넘는다")
    sl = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
    c, n = cell[sl], nuc[sl]
    ids = np.union1d(np.unique(c), np.unique(n))
    ids = ids[ids > 0]
    lut = np.zeros(int(max(cell.max(), nuc.max())) + 1, np.uint16)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.uint16)
    return image[(slice(None),) + sl], lut[n], lut[c], int(lut[label])


class CropWriter:
    """데이터셋 crop zarr. 크기를 미리 알고 순서대로 쓴다(단일 프로세스 writer)."""

    def __init__(self, path: str | Path, n: int, n_channels: int, crop_size: int, dtype, chunk: int = 64):
        import zarr
        from zarr.codecs import BloscCodec

        comp = [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")]
        root = zarr.open_group(str(path), mode="w")
        self.images = root.create_array("images", shape=(n, n_channels, crop_size, crop_size), dtype=dtype,
                                        chunks=(chunk, n_channels, crop_size, crop_size), compressors=comp)
        self.nuc = root.create_array("nuc_mask", shape=(n, crop_size, crop_size), dtype="uint16",
                                     chunks=(chunk, crop_size, crop_size), compressors=comp)
        self.cell = root.create_array("cell_mask", shape=(n, crop_size, crop_size), dtype="uint16",
                                      chunks=(chunk, crop_size, crop_size), compressors=comp)
        self.root = root

    def write(self, start: int, images: np.ndarray, nuc: np.ndarray, cell: np.ndarray) -> None:
        end = start + len(images)
        self.images[start:end] = images
        self.nuc[start:end] = nuc
        self.cell[start:end] = cell
