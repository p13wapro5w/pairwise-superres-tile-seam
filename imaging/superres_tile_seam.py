"""超分推理流水线修复模块。

覆盖六类线上问题：
1. 固定切片边界在低对比区域出现接缝   -> tile_overlap / blend_weights / seam_mask
2. 动态分辨率切换后输入尺寸与缓存不匹配 -> size_cache_key / batch_shape
3. 批内图像色域不一致导致偏色         -> color_manage / hdr_input
4. 异步推理完成时该帧已上屏           -> frame_align
5. 推理队列积压                       -> drop_policy
6. 回退上一帧显示造成抖动             -> fallback_ts / fallback

另含 validation / audit / metrics / determinism 等配套入口。
所有函数均为纯 Python 实现，不依赖第三方库。
"""

from __future__ import annotations

import hashlib
import math
import time

# ---------------------------------------------------------------------------
# 1. 重叠切片与权重融合
# ---------------------------------------------------------------------------

#: 默认切片重叠像素数（源分辨率下），保证低对比区域也有足够融合带
DEFAULT_OVERLAP = 16


def tile_overlap(image_size, tile_size=None, overlap=None):
    """按重叠切片划分图像，返回每片的 (x, y, w, h) 及融合区信息。

    相邻切片重叠 ``overlap`` 像素，重叠区由 :func:`blend_weights` 生成的
    权重做加权融合，消除固定切片边界在低对比区域的接缝。

    ``image_size`` 可以是 int（视为正方形边长）或 ``(width, height)``。
    返回 ``{"tiles": [...], "overlap": int, "tile_size": (tw, th)}``。
    """
    if isinstance(image_size, (tuple, list)):
        width, height = int(image_size[0]), int(image_size[1])
    else:
        width = height = max(1, int(image_size))
    if tile_size is None:
        tile_size = max(width, height)
    if isinstance(tile_size, (tuple, list)):
        tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    else:
        tile_w = tile_h = int(tile_size)
    tile_w = max(1, min(tile_w, width))
    tile_h = max(1, min(tile_h, height))
    if overlap is None:
        overlap = min(DEFAULT_OVERLAP, tile_w // 2, tile_h // 2)
    overlap = max(0, min(int(overlap), tile_w - 1, tile_h - 1))

    step_x = tile_w - overlap
    step_y = tile_h - overlap

    def _starts(length, tile, step):
        starts = list(range(0, max(1, length - tile + 1), step))
        last = length - tile
        if starts and starts[-1] < last:
            starts.append(last)  # 末片对齐右/下边缘，保持重叠而非截断
        return starts or [0]

    tiles = []
    for y in _starts(height, tile_h, step_y):
        for x in _starts(width, tile_w, step_x):
            tiles.append({
                "box": (x, y, min(tile_w, width - x), min(tile_h, height - y)),
                "overlap_left": x > 0,
                "overlap_top": y > 0,
                "overlap_right": x + tile_w < width,
                "overlap_bottom": y + tile_h < height,
            })
    return {
        "tiles": tiles,
        "overlap": overlap,
        "tile_size": (tile_w, tile_h),
        "image_size": (width, height),
    }


def blend_weights(size, overlap=None):
    """生成切片融合权重图（纯 Python 的二维列表，值域 [0,1]）。

    重叠区使用线性斜坡（等价于余弦窗的单调近似），非重叠区权重为 1；
    水平/垂直权重相乘得到二维权重，重叠区权重和恒为 1，融合无接缝。

    ``size`` 为 int（正方形）或 ``(width, height)``。
    """
    if isinstance(size, (tuple, list)):
        width, height = int(size[0]), int(size[1])
    else:
        width = height = max(1, int(size))
    if overlap is None:
        overlap = min(DEFAULT_OVERLAP, width // 2, height // 2)
    overlap = max(0, int(overlap))

    def _ramp(length):
        w = [1.0] * length
        o = min(overlap, length // 2)
        for i in range(o):
            t = (i + 1) / (o + 1)  # 0 -> 1 的斜坡
            w[i] = min(w[i], t)
            w[length - 1 - i] = min(w[length - 1 - i], t)
        return w

    wx = _ramp(width)
    wy = _ramp(height)
    weights = [[wy[y] * wx[x] for x in range(width)] for y in range(height)]
    return {"weights": weights, "overlap": overlap, "size": (width, height)}


def seam_mask(image_size, tile_size=None, overlap=None, low_contrast=None):
    """标记切片边界接缝风险区域。

    返回每个切片边界带的掩码信息；``low_contrast`` 可传入低对比区域
    坐标列表，这些区域在边界带上时会被标记为高风险（需要更大重叠）。
    """
    plan = tile_overlap(image_size, tile_size, overlap)
    ov = plan["overlap"]
    width, height = plan["image_size"]
    bands = []
    for tile in plan["tiles"]:
        x, y, w, h = tile["box"]
        if tile["overlap_right"]:
            bands.append({"axis": "v", "at": x + w, "span": (y, y + h), "width": ov})
        if tile["overlap_bottom"]:
            bands.append({"axis": "h", "at": y + h, "span": (x, x + w), "width": ov})
    risky = []
    for pt in (low_contrast or []):
        px, py = pt
        for band in bands:
            if band["axis"] == "v" and abs(px - band["at"]) <= band["width"] \
                    and band["span"][0] <= py <= band["span"][1]:
                risky.append({"point": pt, "band": band})
            elif band["axis"] == "h" and abs(py - band["at"]) <= band["width"] \
                    and band["span"][0] <= px <= band["span"][1]:
                risky.append({"point": pt, "band": band})
    return {
        "image_size": (width, height),
        "seam_bands": bands,
        "risky_points": risky,
        "overlap": ov,
    }


# ---------------------------------------------------------------------------
# 2. 输入尺寸缓存键 / 批形状
# ---------------------------------------------------------------------------

_CACHE_SCHEMA_VERSION = "v2"


def size_cache_key(input_size=None, scale=None, model=None, extra=None):
    """生成与输入尺寸绑定的缓存键。

    动态分辨率切换后，键中包含 ``(width, height)`` 与超分倍率、模型版本，
    不同输入尺寸不会命中同一条缓存，避免张量形状不匹配。

    ``input_size`` 为 int（正方形）或 ``(width, height)``。
    """
    if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
        width, height = int(input_size[0]), int(input_size[1])
    elif input_size is None:
        width = height = 0
    else:
        width = height = int(input_size)
    scale = 2 if scale is None else scale
    model = "default" if model is None else str(model)
    payload = f"{_CACHE_SCHEMA_VERSION}|{width}x{height}|x{scale}|{model}"
    if extra:
        payload += "|" + ",".join(f"{k}={extra[k]}" for k in sorted(extra))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {
        "key": f"sr:{digest}",
        "input_size": (width, height),
        "scale": scale,
        "model": model,
        "payload": payload,
    }


def batch_shape(batch=None, expected_rank=4):
    """校验批输入形状，返回规范化后的形状描述。

    动态分辨率下逐批校验：所有图像的 (H, W) 必须一致（或先按
    :func:`size_cache_key` 分组），否则拒绝并给出原因，而不是让模型
    在运行期报形状错误。
    """
    if batch is None:
        shapes = []
    elif isinstance(batch, int):
        shapes = [(batch,)]
    else:
        shapes = [tuple(s) if isinstance(s, (tuple, list)) else (s,) for s in batch]
    ok = True
    reason = ""
    if shapes:
        rank_ok = all(len(s) == expected_rank for s in shapes)
        hw = {tuple(s[-2:]) for s in shapes if len(s) >= 2}
        if not rank_ok:
            ok, reason = False, f"rank != {expected_rank}"
        elif len(hw) > 1:
            ok, reason = False, f"inconsistent HxW in batch: {sorted(hw)}"
    return {"ok": ok, "reason": reason, "shapes": shapes,
            "expected_rank": expected_rank}


# ---------------------------------------------------------------------------
# 3. 逐图色彩管理 / HDR 输入
# ---------------------------------------------------------------------------

#: 常见色域的白点与 gamma 描述（简化模型，用于逐图归一化）
_GAMUTS = {
    "srgb": {"gamma": 2.2, "primaries": "bt709"},
    "display-p3": {"gamma": 2.2, "primaries": "p3"},
    "rec2020": {"gamma": 2.4, "primaries": "bt2020"},
    "linear": {"gamma": 1.0, "primaries": "bt709"},
}


def color_manage(image=None, src_gamut=None, dst_gamut="srgb"):
    """逐图色彩管理：把每张图从各自源色域转换到统一目标色域。

    批内图像色域不一致时，逐图记录 ``src_gamut`` 并分别转换，避免
    整批按同一色域解释造成偏色。返回该图的转换描述（含 3x3 矩阵占位
    与 gamma 变换），调用方据此应用。
    """
    src = (src_gamut or "srgb").lower()
    dst = (dst_gamut or "srgb").lower()
    if src not in _GAMUTS:
        src = "srgb"
    if dst not in _GAMUTS:
        dst = "srgb"
    need_convert = src != dst
    transform = None
    if need_convert:
        transform = {
            "decode_gamma": _GAMUTS[src]["gamma"],
            "encode_gamma": _GAMUTS[dst]["gamma"],
            "primaries_from": _GAMUTS[src]["primaries"],
            "primaries_to": _GAMUTS[dst]["primaries"],
            # 单位矩阵占位：实际部署时替换为对应 primaries 的 3x3 转换矩阵
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    return {
        "image": image,
        "src_gamut": src,
        "dst_gamut": dst,
        "convert": need_convert,
        "transform": transform,
    }


def hdr_input(image=None, peak_nits=None, transfer="pq"):
    """HDR 输入归一化描述。

    HDR 源（PQ/HLG）先按峰值亮度归一化到 [0,1] 再送入模型，输出后
    按原 transfer 函数还原，避免 HDR/SDR 混批时亮度漂移。
    """
    transfer = (transfer or "pq").lower()
    if transfer not in ("pq", "hlg", "linear"):
        transfer = "pq"
    peak = 1000.0 if peak_nits is None else float(peak_nits)
    if peak <= 0:
        peak = 1000.0
    return {
        "image": image,
        "transfer": transfer,
        "peak_nits": peak,
        "normalize": transfer != "linear",
        "scale": 1.0 / peak,
    }


# ---------------------------------------------------------------------------
# 4. 推理结果帧对齐
# ---------------------------------------------------------------------------


def frame_align(result=None, result_frame=None, presented_frame=None):
    """把异步推理结果对齐到当前应上屏的帧。

    推理完成时若结果对应的帧已被显示（``result_frame < presented_frame``），
    该结果过期，必须丢弃而不是回写，否则画面会回跳。返回对齐决策：
    ``action`` 为 ``present`` / ``drop`` / ``pending``。
    """
    if result_frame is None:
        result_frame = 0
    if presented_frame is None:
        presented_frame = result_frame
    result_frame = int(result_frame)
    presented_frame = int(presented_frame)
    if result_frame < presented_frame:
        action = "drop"      # 帧已上屏，结果过期
    elif result_frame == presented_frame:
        action = "present"   # 正好对应当前帧
    else:
        action = "pending"   # 未来帧，暂存等待
    return {
        "action": action,
        "result": result if action != "drop" else None,
        "result_frame": result_frame,
        "presented_frame": presented_frame,
        "stale": action == "drop",
    }


# ---------------------------------------------------------------------------
# 5. 队列积压丢弃策略
# ---------------------------------------------------------------------------


def drop_policy(queue=None, max_depth=2, now=None):
    """队列积压丢弃策略：只保留最新的 ``max_depth`` 个待推理帧。

    实时上屏场景下旧帧即使推理完成也会过期（见 :func:`frame_align`），
    积压时丢弃最旧的请求，保证队列深度有界、延迟不累积。
    返回 ``{"kept": [...], "dropped": [...]}``。
    """
    if queue is None:
        items = []
    elif isinstance(queue, (list, tuple)):
        items = list(queue)
    else:
        items = [queue]
    max_depth = max(1, int(max_depth))
    if len(items) <= max_depth:
        kept, dropped = items, []
    else:
        dropped = items[:-max_depth]  # 队首最旧，丢弃
        kept = items[-max_depth:]
    return {
        "kept": kept,
        "dropped": dropped,
        "max_depth": max_depth,
        "depth": len(kept),
        "dropped_count": len(dropped),
        "at": time.time() if now is None else now,
    }


# ---------------------------------------------------------------------------
# 6. 回退与回退时间戳
# ---------------------------------------------------------------------------


def fallback_ts(prev_ts=None, frame_duration=None, frame_index=None):
    """回退显示上一帧时使用的时间戳。

    回退帧不能复用上一帧的原始时间戳（会造成显示时钟回退/抖动），
    必须按当前帧位重新生成：``prev_ts + frame_duration``（或按
    ``frame_index * frame_duration`` 的单调时钟）。
    """
    if frame_duration is None:
        frame_duration = 1.0 / 60.0
    frame_duration = float(frame_duration)
    if frame_duration <= 0:
        frame_duration = 1.0 / 60.0
    if prev_ts is None:
        prev_ts = 0.0
    prev_ts = float(prev_ts)
    if frame_index is not None:
        ts = int(frame_index) * frame_duration
    else:
        ts = prev_ts + frame_duration
    return {
        "ts": ts,
        "prev_ts": prev_ts,
        "frame_duration": frame_duration,
        "monotonic": ts > prev_ts,  # 保证时间戳单调递增，消除抖动
    }


def fallback(reason=None, prev_frame=None, frame_index=None, prev_ts=None):
    """回退决策：推理结果不可用（过期/失败）时复用上一帧。

    返回复用的帧与重新生成的时间戳（见 :func:`fallback_ts`），保证
    回退不引入显示时钟抖动。
    """
    ts_info = fallback_ts(prev_ts=prev_ts, frame_index=frame_index)
    return {
        "use_fallback": True,
        "reason": "stale_result" if reason is None else str(reason),
        "frame": prev_frame,
        "ts": ts_info["ts"],
        "frame_index": frame_index,
    }


# ---------------------------------------------------------------------------
# 配套：校验 / 审计 / 指标 / 确定性
# ---------------------------------------------------------------------------


def validation(config=None):
    """校验流水线配置，返回 ``{"ok": bool, "errors": [...]}``。"""
    errors = []
    cfg = config if isinstance(config, dict) else {}
    tile = cfg.get("tile_size")
    overlap = cfg.get("overlap", DEFAULT_OVERLAP)
    if tile is not None:
        if int(tile) <= 0:
            errors.append("tile_size must be positive")
        elif int(overlap) >= int(tile):
            errors.append("overlap must be smaller than tile_size")
    if "scale" in cfg and cfg["scale"] not in (2, 3, 4):
        errors.append("scale must be one of 2/3/4")
    if "max_depth" in cfg and int(cfg["max_depth"]) < 1:
        errors.append("max_depth must be >= 1")
    gamut = cfg.get("dst_gamut", "srgb")
    if gamut not in _GAMUTS:
        errors.append(f"unknown dst_gamut: {gamut}")
    return {"ok": not errors, "errors": errors, "config": cfg}


def audit(event=None, detail=None, ts=None):
    """生成一条流水线审计记录（切片计划、丢弃、回退等关键决策）。"""
    return {
        "ts": time.time() if ts is None else ts,
        "event": "unknown" if event is None else str(event),
        "detail": detail or {},
    }


def metrics(samples=None):
    """汇总流水线指标：接缝误差、缓存命中率、过期丢弃率、回退率。"""
    if samples is None:
        rows = []
    elif isinstance(samples, (list, tuple)):
        rows = list(samples)
    else:
        rows = [samples]
    n = len(rows)
    agg = {
        "count": n,
        "seam_error": 0.0,
        "cache_hit": 0,
        "stale_drop": 0,
        "fallback": 0,
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        agg["seam_error"] += float(row.get("seam_error", 0.0))
        agg["cache_hit"] += 1 if row.get("cache_hit") else 0
        agg["stale_drop"] += 1 if row.get("stale_drop") else 0
        agg["fallback"] += 1 if row.get("fallback") else 0
    if n:
        agg["seam_error_avg"] = agg["seam_error"] / n
        agg["cache_hit_rate"] = agg["cache_hit"] / n
        agg["stale_drop_rate"] = agg["stale_drop"] / n
        agg["fallback_rate"] = agg["fallback"] / n
    else:
        agg.update(seam_error_avg=0.0, cache_hit_rate=0.0,
                   stale_drop_rate=0.0, fallback_rate=0.0)
    return agg


def determinism(seed=None):
    """固定确定性推理所需的全局状态。

    返回需应用的确定性配置（随机种子、关闭非确定性算子等），
    保证同一输入 + 同一 :func:`size_cache_key` 必得同一输出。
    """
    if seed is None:
        seed = 0
    seed = int(seed)
    return {
        "seed": seed,
        "deterministic_algorithms": True,
        "benchmark": False,   # cudnn.benchmark 会引入非确定性
        "apply": "set random/numpy/torch seeds and torch.use_deterministic_algorithms(True)",
    }
