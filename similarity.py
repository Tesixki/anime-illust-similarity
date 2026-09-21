"""イラスト類似度メトリクス計算モジュール

2枚のイラストについて、以下5系統の類似度を計算し 0〜100 のスコアに変換する。
すべてCPUで動作し、モデルは初回呼び出し時（Spaceでは起動時）に自動でダウンロードされる。

- SigLIP 2 : 画像内容のセマンティック類似度（OpenCLIP ViT-B-16-SigLIP2-256 / WebLI）
- DINOv2   : 自己教師あり視覚特徴の類似度（facebook/dinov2-small, CLS埋め込み）
- DreamSim : 知覚的類似度（CLIP+DINO+OpenCLIP アンサンブルを人間の判断で微調整）
- CCIP     : アニメキャラクターの同一性（deepghs/imgutils, ONNX）
- WD14     : イラストタグ埋め込みの一致度（WD SwinV2 tagger v3, ONNX）

スコア化（生値 -> 0〜100）は `calibration/calibrate.py` で実画像ペアを
4カテゴリ（近似複製 / 同一キャラ / 別キャラ / 無関係）に分けて実測した分布に
基づく区分線形マップ。アンカー値は `CALIBRATION` を参照。

単体での動作確認:
    python similarity.py image_a.png image_b.png
"""

from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# キャリブレーション（生値 -> スコア の区分線形マップ）
#
# 各メトリクスについて、実画像ペアの実測中央値（calibration/calibration_result.json、
# アニメ画像 8 キャラ × 6 枚 + 増強 60 枚、ペア約 460 組）をアンカーに使う。
# カテゴリとスコアの対応は全メトリクス共通:
#   無関係な画像（キャラ vs 風景/無関係イラスト）の中央値 ->   0
#   別キャラ（同作品内の別キャラ）の中央値                 ->  30
#   同一キャラ（別の絵）の中央値                           ->  70
#   近似複製（反転/軽いトリミング/明度/JPEG劣化）の中央値  -> 100
# 区間の外側はクリップ。cos 系は値が大きいほど、距離系は小さいほど高スコア。
#
# 例外: CCIP はキャラ同一性のモデルで「別キャラ」と「無関係画像」を区別しない
# （実測: 別キャラ中央値 0.327 / 無関係中央値 0.340、AUC 0.53）ため、
# 0 点のアンカーには別キャラの 95 パーセンタイル（0.461）を使う。
# この設定でモデル既定の判定閾値 0.178 は約 54 点に対応し、
# 「50 点前後 = 同一キャラかどうかの境界」という読み方ができる。
# ---------------------------------------------------------------------------
CATEGORY_SCORE = {"unrelated": 0.0, "diff_char": 30.0, "same_char": 70.0, "dup": 100.0}

# 生値アンカー: (unrelated, diff_char, same_char, dup) の順（calibrate.py の中央値）
CALIBRATION = {
    "siglip2": {"kind": "cosine", "anchors": (0.566, 0.800, 0.904, 0.988)},
    "dinov2": {"kind": "cosine", "anchors": (0.306, 0.546, 0.620, 0.977)},
    "dreamsim": {"kind": "distance", "anchors": (0.758, 0.532, 0.314, 0.023)},
    "ccip": {"kind": "distance", "anchors": (0.461, 0.327, 0.074, 0.004)},
    "wd14": {"kind": "cosine", "anchors": (0.451, 0.530, 0.748, 0.991)},
}


def _piecewise_score(x: float, anchors, kind: str) -> float:
    """アンカー（4点）の区分線形補間で 0〜100 化。区間外はクリップ。"""
    xs = np.asarray(anchors, dtype=np.float64)
    ys = np.asarray(
        [CATEGORY_SCORE[k] for k in ("unrelated", "diff_char", "same_char", "dup")]
    )
    if kind == "distance":  # 小さいほど類似 -> np.interp のため昇順に並べ替え
        xs, ys = xs[::-1], ys[::-1]
    return float(np.clip(np.interp(x, xs, ys), 0.0, 100.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


_LOCK = threading.Lock()
_CACHE: dict = {}


def _cached(key: str, builder):
    if key not in _CACHE:
        with _LOCK:
            if key not in _CACHE:
                _CACHE[key] = builder()
    return _CACHE[key]


# ---------------------------------------------------------------------------
# SigLIP 2
# ---------------------------------------------------------------------------

def _get_siglip():
    def build():
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2-256", pretrained="webli"
        )
        model.eval()
        return model, preprocess, torch

    return _cached("siglip", build)


def siglip2_embed(img: Image.Image) -> np.ndarray:
    model, preprocess, torch = _get_siglip()
    with torch.no_grad():
        f = model.encode_image(preprocess(img).unsqueeze(0))
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].numpy()


def siglip2_metric(e1: np.ndarray, e2: np.ndarray) -> dict:
    cos = _cosine(e1, e2)
    c = CALIBRATION["siglip2"]
    score = _piecewise_score(cos, c["anchors"], c["kind"])
    return {
        "raw": cos,
        "score": score,
        "interp": _interp_by_score(score, "内容"),
        "detail": f"cosine={cos:.3f}（1.0で完全一致）",
    }


# ---------------------------------------------------------------------------
# DINOv2（facebook/dinov2-small, CLS pooled 出力）
# ---------------------------------------------------------------------------

DINO_MODEL = os.environ.get("DINO_MODEL", "facebook/dinov2-small")


def _get_dino():
    def build():
        import torch
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
        model = AutoModel.from_pretrained(DINO_MODEL)
        model.eval()
        return model, proc, torch

    return _cached("dino", build)


def dinov2_embed(img: Image.Image) -> np.ndarray:
    model, proc, torch = _get_dino()
    with torch.no_grad():
        out = model(**proc(images=img, return_tensors="pt"))
    return out.pooler_output[0].numpy()


def dinov2_metric(e1: np.ndarray, e2: np.ndarray) -> dict:
    cos = _cosine(e1, e2)
    c = CALIBRATION["dinov2"]
    score = _piecewise_score(cos, c["anchors"], c["kind"])
    return {
        "raw": cos,
        "score": score,
        "interp": _interp_by_score(score, "構図・形状"),
        "detail": f"cosine={cos:.3f}（1.0で完全一致 / {DINO_MODEL.split('/')[-1]}）",
    }


# ---------------------------------------------------------------------------
# DreamSim（距離 = 1 - cos(埋め込み)）
# ---------------------------------------------------------------------------

DREAMSIM_CACHE = os.environ.get(
    "DREAMSIM_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "dreamsim")
)


def _get_dreamsim():
    def build():
        import torch
        from dreamsim import dreamsim

        model, preprocess = dreamsim(
            pretrained=True, device="cpu", cache_dir=DREAMSIM_CACHE
        )
        return model, preprocess, torch

    return _cached("dreamsim", build)


def dreamsim_embed(img: Image.Image) -> np.ndarray:
    model, preprocess, torch = _get_dreamsim()
    with torch.no_grad():
        return model.embed(preprocess(img))[0].numpy()


def dreamsim_metric(e1: np.ndarray, e2: np.ndarray) -> dict:
    dist = 1.0 - _cosine(e1, e2)
    c = CALIBRATION["dreamsim"]
    score = _piecewise_score(dist, c["anchors"], c["kind"])
    return {
        "raw": dist,
        "score": score,
        "interp": _interp_by_score(score, "見た目"),
        "detail": f"distance={dist:.3f}（0で完全一致）",
    }


# ---------------------------------------------------------------------------
# CCIP（キャラクター同一性）
# ---------------------------------------------------------------------------

def _ccip_threshold() -> float:
    def build():
        from imgutils.metrics import ccip_default_threshold

        return float(ccip_default_threshold())

    return _cached("ccip_thr", build)


def ccip_embed(img: Image.Image) -> np.ndarray:
    from imgutils.metrics import ccip_extract_feature

    return ccip_extract_feature(img)


def ccip_metric(e1: np.ndarray, e2: np.ndarray) -> dict:
    from imgutils.metrics import ccip_batch_differences

    diff = float(ccip_batch_differences([e1, e2])[0, 1])
    thr = _ccip_threshold()
    c = CALIBRATION["ccip"]
    score = _piecewise_score(diff, c["anchors"], c["kind"])
    if diff < thr * 0.7:
        interp = "同一キャラの可能性が非常に高い"
    elif diff < thr:
        interp = "同一キャラの可能性が高い"
    elif diff < thr * 1.3:
        interp = "同一キャラかどうかの境界付近"
    else:
        interp = "別キャラの可能性が高い"
    return {
        "raw": diff,
        "score": score,
        "interp": interp,
        "detail": f"difference={diff:.3f}（モデル閾値 {thr:.3f} 未満なら同一キャラ判定）",
        "same_character": diff < thr,
    }


# ---------------------------------------------------------------------------
# WD14 tagger（タグ埋め込み + 共通タグ）
# ---------------------------------------------------------------------------

def wd14_embed(img: Image.Image) -> dict:
    from imgutils.tagging import get_wd14_tags

    gen, char, emb = get_wd14_tags(img, fmt=("general", "character", "embedding"))
    return {
        "emb": np.asarray(emb, dtype=np.float32),
        "general": dict(gen),
        "character": dict(char),
    }


def wd14_metric(w1: dict, w2: dict) -> dict:
    cos = _cosine(w1["emb"], w2["emb"])
    c = CALIBRATION["wd14"]
    score = _piecewise_score(cos, c["anchors"], c["kind"])
    tags1 = set(w1["general"]) | set(w1["character"])
    tags2 = set(w2["general"]) | set(w2["character"])
    shared = sorted(tags1 & tags2, key=lambda t: -(w1["general"].get(t, 0) + w2["general"].get(t, 0)))
    return {
        "raw": cos,
        "score": score,
        "interp": _interp_by_score(score, "タグ構成"),
        "detail": f"cosine={cos:.3f}（1.0で完全一致）",
        "shared_tags": shared[:15],
        "char_tags_a": sorted(w1["character"]),
        "char_tags_b": sorted(w2["character"]),
    }


def _interp_by_score(score: float, subject: str) -> str:
    if score >= 85:
        return f"{subject}がほぼ同じ"
    if score >= 60:
        return f"{subject}がかなり近い"
    if score >= 40:
        return f"{subject}がやや近い"
    if score >= 20:
        return f"{subject}はあまり近くない"
    return f"{subject}が大きく異なる"


# ---------------------------------------------------------------------------
# メトリクス定義・重み・総合スコア
# ---------------------------------------------------------------------------

METRICS = {
    "ccip": {
        "label": "キャラクター類似度 (CCIP)",
        "desc": "描かれたキャラが同一かを判定するアニメ特化モデル。単一キャラの画像が前提。",
        "embed": ccip_embed,
        "fn": ccip_metric,
    },
    "wd14": {
        "label": "タグ類似度 (WD14 tagger v3)",
        "desc": "キャラ名・属性・服装などイラストタグ埋め込みの一致度。アニメ特化。",
        "embed": wd14_embed,
        "fn": wd14_metric,
    },
    "dreamsim": {
        "label": "知覚的類似度 (DreamSim)",
        "desc": "人間の類似度判断に近い画像間距離。色合い・質感・スタイル向け。",
        "embed": dreamsim_embed,
        "fn": dreamsim_metric,
    },
    "siglip2": {
        "label": "セマンティック類似度 (SigLIP 2)",
        "desc": "画像内容の意味的な近さ。汎用の画像テキスト対照学習モデル。",
        "embed": siglip2_embed,
        "fn": siglip2_metric,
    },
    "dinov2": {
        "label": "視覚特徴類似度 (DINOv2)",
        "desc": "自己教師あり学習の視覚特徴。構図・形状・オブジェクトの類似に強い。",
        "embed": dinov2_embed,
        "fn": dinov2_metric,
    },
}

# 総合スコアの重み。calibration の AUC（同一キャラ vs 別キャラ の識別力）に基づく:
#   CCIP 0.999 / SigLIP2 0.949 / WD14 0.944 / DreamSim 0.941 / DINOv2 0.678
# 最も識別力の高い CCIP を厚めにし、同程度の 3 つは均等、
# キャラ識別に弱い DINOv2（構図・形状向け）は補助として軽くしている。
WEIGHTS = {
    "ccip": 0.30,
    "wd14": 0.20,
    "siglip2": 0.20,
    "dreamsim": 0.20,
    "dinov2": 0.10,
}


def total_score(results: dict):
    """利用可能なメトリクスの重み付き平均（重みは利用分で再正規化）"""
    num, den = 0.0, 0.0
    for key, w in WEIGHTS.items():
        r = results.get(key)
        if r and "score" in r:
            num += r["score"] * w
            den += w
    return round(num / den) if den else None


def band_label(score: float) -> str:
    if score >= 80:
        return "かなり似ている"
    if score >= 60:
        return "ある程度似ている"
    if score >= 40:
        return "部分的に似ている"
    return "あまり似ていない"


def score_breakdown(results: dict):
    """総合スコアの計算内訳（表示用）。Returns (rows, total, den)."""
    rows, num, den = [], 0.0, 0.0
    for key, w in WEIGHTS.items():
        info = METRICS[key]
        r = results.get(key)
        if r and "score" in r:
            contrib = r["score"] * w
            rows.append([info["label"], round(r["score"], 1), w, round(contrib, 1)])
            num += contrib
            den += w
        else:
            rows.append([info["label"], "-", w, "対象外" if r and "skipped" in r else "計算失敗"])
    return rows, (round(num / den) if den else None), den


def make_comment(results: dict) -> str:
    total = results.get("total")
    if total is None:
        return "計算可能なメトリクスがありませんでした"

    def ok(k):
        return k in results and "score" in results[k]

    lines = [f"総合評価: {band_label(total)}"]
    if "skipped" in results.get("ccip", {}):
        lines.append(f"CCIP（キャラ判定）は対象外のため総合スコアから除外: {results['ccip']['skipped']}")
    if ok("ccip"):
        ccip = results["ccip"]
        others = [results[k]["score"] for k in ("siglip2", "dreamsim", "wd14") if ok(k)]
        if ccip.get("same_character") and ccip["score"] >= 60:
            lines.append("同一キャラクターの可能性が高いです（CCIP基準）")
        elif ccip.get("same_character") is False:
            if others and max(others) >= 60:
                lines.append("画風・構図は近いですが、キャラクターは別人の可能性があります")
            else:
                lines.append("キャラクターも内容も異なる画像と見られます")
    if ok("wd14"):
        common = set(results["wd14"].get("char_tags_a") or []) & set(
            results["wd14"].get("char_tags_b") or []
        )
        if common:
            lines.append(f"共通キャラタグ検出: {', '.join(sorted(common))}")
    return "\n".join(lines)


def preload(progress=None) -> None:
    """全モデルをロードする（Space起動時に呼ぶ）。"""
    loaders = [
        ("SigLIP 2", _get_siglip),
        ("DINOv2", _get_dino),
        ("DreamSim", _get_dreamsim),
        ("CCIP", _ccip_threshold),
    ]
    for name, fn in loaders:
        if progress:
            progress(name)
        fn()
    # ONNX モデル（CCIP / WD14）は 1 回推論して重みを取得しておく
    dummy = Image.new("RGB", (64, 64), (255, 255, 255))
    ccip_embed(dummy)
    wd14_embed(dummy)


PERSON_TAGS = {"solo", "1girl", "1boy", "2girls", "2boys", "multiple_girls", "multiple_boys"}
MULTI_TAGS = {"2girls", "2boys", "multiple_girls", "multiple_boys"}


def _ccip_applicability(w: dict | None) -> tuple[bool, str | None]:
    """WD14 タグから CCIP（単一キャラ前提）が適用可能かを判定する。

    Returns (applicable, note)。人物タグが無く no_humans が付いた画像は対象外。
    複数人物タグがある場合は適用するが注意書きを返す。
    """
    if not w:
        return True, None
    tags = set(w["general"])
    if "no_humans" in tags and not (tags & PERSON_TAGS):
        return False, "人物が検出されませんでした（no_humans）"
    if tags & MULTI_TAGS:
        return True, "複数キャラが検出されました。CCIPは単一キャラ前提のため精度が落ちます"
    return True, None


def compute_all(img1: Image.Image, img2: Image.Image, on_progress=None) -> dict:
    """全メトリクスを計算。個別の失敗は error として記録し、他は続行する。"""
    results = {}
    embeds = {}
    for i, (key, info) in enumerate(METRICS.items()):
        if on_progress:
            on_progress(i, len(METRICS), info["label"])
        try:
            e1 = info["embed"](img1)
            e2 = info["embed"](img2)
            embeds[key] = (e1, e2)
            results[key] = info["fn"](e1, e2)
        except Exception as e:  # モデルDL失敗なども1メトリクスの失敗に留める
            results[key] = {"error": f"{type(e).__name__}: {e}"}

    # CCIP の適用可否ガード（WD14 のタグを利用）
    if "wd14" in embeds and "score" in results.get("ccip", {}):
        notes = []
        applicable = True
        for label, w in zip("AB", embeds["wd14"]):
            ok, note = _ccip_applicability(w)
            applicable &= ok
            if note:
                notes.append(f"画像{label}: {note}")
        if not applicable:
            results["ccip"] = {"skipped": "、".join(notes), "raw": results["ccip"]["raw"]}
        elif notes:
            results["ccip"]["note"] = "、".join(notes)

    results["total"] = total_score(results)
    results["comment"] = make_comment(results)
    return results


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 3:
        sys.exit("usage: python similarity.py <image_a> <image_b>")
    a = Image.open(sys.argv[1]).convert("RGB")
    b = Image.open(sys.argv[2]).convert("RGB")
    res = compute_all(a, b)
    for key, info in METRICS.items():
        r = res[key]
        if "error" in r:
            print(f"{info['label']}: ERROR {r['error']}")
        else:
            print(f"{info['label']}: {r['score']:.1f} / 100  {r['interp']}  ({r['detail']})")
    print(f"total: {res['total']}")
    print(json.dumps(res["comment"], ensure_ascii=False))
