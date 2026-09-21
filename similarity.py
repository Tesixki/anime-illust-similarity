"""イラスト類似度メトリクス計算モジュール

2枚のイラストについて、以下8系統の類似度を計算し 0〜100 のスコアに変換する。
CPU / ZeroGPU の両方で動作し、モデルは初回呼び出し時（Spaceでは起動時）に自動でダウンロードされる。

- CCIP     : アニメキャラクターの同一性（deepghs/imgutils, ONNX）
- WD14     : イラストタグ埋め込みの一致度（WD SwinV2 tagger v3, ONNX）
- PixAI    : 大規模アニメタガー PixAI Tagger v1.0 の内部埋め込みの一致度（SAM3系 ViTDet, 1008px）
- DreamSim : 知覚的類似度（CLIP+DINO+OpenCLIP アンサンブルを人間の判断で微調整）
- SigLIP 2 : 画像内容のセマンティック類似度（OpenCLIP ViT-B-16-SigLIP2-256 / WebLI）
- DINOv2   : 自己教師あり視覚特徴の類似度（facebook/dinov2-small, CLS埋め込み）
- Depth    : 構図類似度（Depth Anything V2 Small の深度マップ相関）
- DWPose   : ポーズ類似度（rtmlib DWPose, OpenPose 18 点の四肢向き比較）

スコア化（生値 -> 0〜100）は `calibration/calibrate.py`（キャラ同一性系: 近似複製 / 同一キャラ /
別キャラ / 無関係）と `calibration/calibrate_composition.py`（構図・ポーズ: 同構図 / 反転・トリミング /
別カット / 無関係）で実画像ペアの分布を実測し、その中央値をアンカーにした区分線形マップ。
アンカー値は `CALIBRATION` を参照。

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
    "pixai": {"kind": "cosine", "anchors": (0.282, 0.508, 0.742, 0.978)},
    # 構図・ポーズは較正カテゴリが異なる（無関係=0 / 別カット=30 / 反転・トリミング=70 / 同構図=100）。
    # 値は calibration/calibrate_composition.py の実測中央値（calibration_composition.json）。
    # ポーズは無関係画像がほぼ対象外になるため、0 点アンカーは別カットの 5 パーセンタイル。
    "depth": {"kind": "cosine", "anchors": (0.285, 0.451, 0.805, 0.999)},
    "pose": {"kind": "cosine", "anchors": (-0.074, 0.734, 0.978, 0.999)},
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


_DEVICE = "cpu"


def set_device(device: str) -> str:
    """torch 系モデル（SigLIP 2 / DINOv2 / DreamSim）の推論デバイスを切り替える。

    CUDA が使えない場合は cpu にフォールバックする。ONNX 系（CCIP / WD14）は常に CPU。
    ZeroGPU Space では @spaces.GPU 関数の中で "cuda" を指定する。
    """
    global _DEVICE
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    for key in ("siglip", "dino", "dreamsim", "pixai", "depth"):
        if key in _CACHE:
            model = _CACHE[key][0]
            model.to(device)
            # DreamSim の PerceptualModel / ViTExtractor は `device` 属性を持ち、
            # forward 内で投影行列を self.device に移すため、属性も同期させる。
            for m in model.modules():
                if isinstance(getattr(m, "device", None), str):
                    m.device = device
    _DEVICE = device
    return device


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
        model.eval().to(_DEVICE)
        return model, preprocess, torch

    return _cached("siglip", build)


def siglip2_embed(img: Image.Image) -> np.ndarray:
    model, preprocess, torch = _get_siglip()
    with torch.no_grad():
        f = model.encode_image(preprocess(img).unsqueeze(0).to(_DEVICE))
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].float().cpu().numpy()


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
        model.eval().to(_DEVICE)
        return model, proc, torch

    return _cached("dino", build)


def dinov2_embed(img: Image.Image) -> np.ndarray:
    model, proc, torch = _get_dino()
    with torch.no_grad():
        inputs = {k: v.to(_DEVICE) for k, v in proc(images=img, return_tensors="pt").items()}
        out = model(**inputs)
    return out.pooler_output[0].float().cpu().numpy()


def dinov2_metric(e1: np.ndarray, e2: np.ndarray) -> dict:
    cos = _cosine(e1, e2)
    c = CALIBRATION["dinov2"]
    score = _piecewise_score(cos, c["anchors"], c["kind"])
    return {
        "raw": cos,
        "score": score,
        "interp": _interp_by_score(score, "全体の視覚特徴"),
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

        from unittest.mock import patch

        # ZeroGPU では起動時に torch.cuda.is_available() が True に patch され、
        # peft のアダプタ読み込みが CUDA を選んで失敗する（実 GPU は @spaces.GPU 内でのみ使用可）。
        # ロード中だけ False に固定し、確実に CPU へ読み込む。
        with patch.object(torch.cuda, "is_available", lambda: False):
            model, preprocess = dreamsim(
                pretrained=True, device="cpu", cache_dir=DREAMSIM_CACHE
            )
        model.to(_DEVICE)
        return model, preprocess, torch

    return _cached("dreamsim", build)


def dreamsim_embed(img: Image.Image) -> np.ndarray:
    model, preprocess, torch = _get_dreamsim()
    with torch.no_grad():
        return model.embed(preprocess(img).to(_DEVICE))[0].float().cpu().numpy()


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


# ---------------------------------------------------------------------------
# PixAI Tagger v1.0（pixai-labs/pixai-tagger-v1.0, SAM3 系 ViTDet 486M, 1008px）
#   分類ヘッド直前の attention-pool 出力（1024次元）を埋め込みとして使い、
#   同じ forward からタグ確率（30,877 タグ / 6 カテゴリ）も取る。
#   非常に重いため CPU では 1 枚数十秒かかる。ENABLE_PIXAI=0 で無効化できる。
# ---------------------------------------------------------------------------

PIXAI_MODEL = os.environ.get("PIXAI_MODEL", "pixai-labs/pixai-tagger-v1.0")
ENABLE_PIXAI = os.environ.get("ENABLE_PIXAI", "1") != "0"
PIXAI_TAG_CATEGORIES = ("general", "character", "copyright", "style")


def _get_pixai():
    def build():
        import torch
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(PIXAI_MODEL, trust_remote_code=True)
        model = AutoModel.from_pretrained(PIXAI_MODEL, trust_remote_code=True)
        model.eval().to(_DEVICE)
        cfg = model.config
        thr = cfg.category_best_threshold or {}
        # カテゴリごとの (name, start, end, threshold)
        spans, st = [], 0
        for cat, n in cfg.tags_split:
            spans.append((cat, st, st + n, float(thr.get(cat, 0.2))))
            st += n
        return model, proc, torch, {"tags": list(cfg.tags), "spans": spans}

    return _cached("pixai", build)


def pixai_embed(img: Image.Image) -> dict:
    model, proc, torch, meta = _get_pixai()
    with torch.no_grad():
        px = proc(img, return_tensors="pt")["pixel_values"].to(_DEVICE, model.dtype)
        feats = model.forward_feature(px)[-1]  # [1, C, h, w]
        tokens = feats.view(feats.shape[0], feats.shape[1], -1).permute(0, 2, 1)
        pooled = model.head_pool(tokens)  # [1, embed_dim]
        probs = torch.sigmoid(model.head(pooled))[0].float().cpu().numpy()
    out = {"emb": pooled[0].float().cpu().numpy()}
    tags = meta["tags"]
    for cat, st, en, thr in meta["spans"]:
        if cat not in PIXAI_TAG_CATEGORIES:
            continue
        idx = np.nonzero(probs[st:en] > thr)[0]
        out[cat] = {tags[st + i]: float(probs[st + i]) for i in idx}
    return out


def pixai_metric(w1: dict, w2: dict) -> dict:
    cos = _cosine(w1["emb"], w2["emb"])
    c = CALIBRATION["pixai"]
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
        "style_tags_a": sorted(w1.get("style", {})),
        "style_tags_b": sorted(w2.get("style", {})),
    }


# ---------------------------------------------------------------------------
# 構図類似度（Depth Anything V2 Small の深度マップ比較）
#   単眼深度（相対値）を画像ごとに正規化 -> 64x64 に縮小 -> ピアソン相関。
#   「前景/背景の分離」「被写体の画面内配置と大きさ」「引き/寄り」を捉える。
#   左右反転は別構図として扱う（相関が下がる）。
# ---------------------------------------------------------------------------

DEPTH_MODEL = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
DEPTH_GRID = 64


def _get_depth():
    def build():
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
        model.eval().to(_DEVICE)
        return model, proc, torch

    return _cached("depth", build)


def _depth_colorize(d: np.ndarray) -> Image.Image:
    import cv2

    dn = (d - d.min()) / (d.max() - d.min() + 1e-6)
    vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))


def depth_embed(img: Image.Image) -> dict:
    import cv2

    model, proc, torch = _get_depth()
    with torch.no_grad():
        inputs = {k: v.to(_DEVICE) for k, v in proc(images=img, return_tensors="pt").items()}
        d = model(**inputs).predicted_depth[0].float().cpu().numpy()
    # 元画像のアスペクトに戻してから固定グリッドへ（正規化で相対深度のスケール/オフセットを消す）
    d_img = cv2.resize(d, img.size, interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(d_img, (DEPTH_GRID, DEPTH_GRID), interpolation=cv2.INTER_AREA)
    small = (small - np.median(small)) / (small.std() + 1e-6)
    return {"map": small.astype(np.float32), "vis": _depth_colorize(d_img)}


def depth_metric(d1: dict, d2: dict) -> dict:
    a, b = d1["map"].ravel(), d2["map"].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    corr_flip = float(np.corrcoef(a, d2["map"][:, ::-1].ravel())[0, 1])
    c = CALIBRATION["depth"]
    score = _piecewise_score(corr, c["anchors"], c["kind"])
    detail = f"depth corr={corr:.3f}（1.0で完全一致）"
    note = None
    if corr_flip > corr + 0.15:
        note = f"左右反転すると一致度が上がります（反転時 corr={corr_flip:.3f}）: 鏡像構図の可能性"
    return {
        "raw": corr,
        "score": score,
        "interp": _interp_by_score(score, "構図（奥行き配置）"),
        "detail": detail,
        "note": note,
        "vis_a": d1["vis"],
        "vis_b": d2["vis"],
    }


# ---------------------------------------------------------------------------
# ポーズ類似度（DWPose / rtmlib, OpenPose 18 点形式）
#   四肢（関節ペア）の向きベクトルのコサインを平均する角度ベースの指標。
#   位置・スケールに不変で「同じポーズか」を測る（画面内配置は Depth 側が担当）。
#   両画像で信頼度 >= POSE_KPT_THR の関節ペアが POSE_MIN_LIMBS 未満なら「対象外」。
# ---------------------------------------------------------------------------

POSE_MODE = os.environ.get("POSE_MODE", "performance")  # rtmlib: lightweight / balanced / performance
# 関節信頼度の閾値。performance モード（rtmw-dw-x-l 384x288）の ONNX は信頼度が未正規化
# （可視の目/鼻で 5〜7、可視の手首で 3〜6、首・肩で 2.6〜3.8、隠れた手首や膝で 1.6〜2.8、
# 人物なし画像で 0.9〜3.3）なので、実測に基づき 2.8 を使う。balanced / lightweight は 0〜1 で 0.5。
POSE_KPT_THR = float(os.environ.get("POSE_KPT_THR", "2.8" if POSE_MODE == "performance" else "0.5"))
POSE_MIN_LIMBS = 4
POSE_TTA = os.environ.get("POSE_TTA", "1") != "0"  # 左右反転して 2 回推定し、座標を信頼度で加重平均
# 人物ボックスを複数スケールで切り出して推定を平均する（位置ノイズの低減）。"1.0" で無効
POSE_BOX_SCALES = tuple(float(x) for x in os.environ.get("POSE_BOX_SCALES", "1.0,1.25").split(","))
# 短辺がこれ未満の画像は推定前に拡大する（0 で無効）
POSE_UPSCALE_MIN_SIDE = int(os.environ.get("POSE_UPSCALE_MIN_SIDE", "0"))
# 手（指）: OpenPose 134 点形式の 92..112 が左手、113..133 が右手（各 root + 5 指 × 4 関節）
POSE_HAND_THR = float(os.environ.get("POSE_HAND_THR", "2.5" if POSE_MODE == "performance" else "0.4"))
POSE_HAND_MIN_LIMBS = 6      # 片手あたり、両画像で共通して取れた指の骨がこれ以上なら手を比較に含める
POSE_HAND_WEIGHT = 0.5       # 総合 limb cos における手（左右まとめて）の重み。体は 1.0
POSE_LHAND, POSE_RHAND = 92, 113
POSE_HAND_LIMBS = []  # (i, j) 片手 20 本。root->指1->指2->指3->指4
for _f in range(5):
    _b = 1 + _f * 4
    POSE_HAND_LIMBS += [(0, _b), (_b, _b + 1), (_b + 1, _b + 2), (_b + 2, _b + 3)]
# 左右対応（反転 TTA 用）: 体 + 足 + 手
POSE_LR_SWAP = (
    [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]
    + [(18, 21), (19, 22), (20, 23)]
    + [(POSE_LHAND + i, POSE_RHAND + i) for i in range(21)]
)
# OpenPose 18 点: 0鼻 1首 2右肩 3右肘 4右手首 5左肩 6左肘 7左手首 8右腰 9右膝 10右足首
#                11左腰 12左膝 13左足首 14右目 15左目 16右耳 17左耳
POSE_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
POSE_LIMB_NAMES = [
    "首→右肩", "首→左肩", "右上腕", "右前腕", "左上腕", "左前腕",
    "首→右腰", "右大腿", "右下腿", "首→左腰", "左大腿", "左下腿",
    "首→鼻", "鼻→右目", "右目→右耳", "鼻→左目", "左目→左耳",
]


def _get_pose():
    def build():
        from rtmlib import Wholebody

        return (Wholebody(to_openpose=True, mode=POSE_MODE, backend="onnxruntime", device="cpu"),)

    return _cached("pose", build)


def _pose_pick_person(kpts: np.ndarray, scores: np.ndarray) -> int:
    """有効な体の関節のバウンディングボックスが最大の人物のインデックスを返す。"""
    best, best_area = 0, -1.0
    for i in range(kpts.shape[0]):
        v = kpts[i, :18][scores[i, :18] >= POSE_KPT_THR]
        area = float(np.prod(v.max(0) - v.min(0))) if len(v) >= 2 else 0.0
        if area > best_area:
            best, best_area = i, area
    return best


def _pose_flip_back(kpts: np.ndarray, scores: np.ndarray, width: int):
    """反転画像で推定した関節（134 点）を元画像の座標系に戻し、左右のラベルを入れ替える。"""
    kpts = kpts.copy()
    scores = scores.copy()
    kpts[:, 0] = width - 1 - kpts[:, 0]
    for a, b in POSE_LR_SWAP:
        kpts[[a, b]] = kpts[[b, a]]
        scores[[a, b]] = scores[[b, a]]
    return kpts, scores


def pose_embed(img: Image.Image) -> dict:
    """DWPose で OpenPose 134 点（体 18 / 足 6 / 顔 68 / 両手 42）を推定する。

    精度対策: 人物ボックスの複数スケール切り出し + 左右反転 TTA を信頼度で加重平均、
    人物検出に失敗したら画面全体を 1 人として推定、小さい画像は事前拡大（任意）。
    """
    import cv2
    from rtmlib import draw_skeleton

    (wb,) = _get_pose()
    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    up = 1.0
    if POSE_UPSCALE_MIN_SIDE and min(bgr.shape[:2]) < POSE_UPSCALE_MIN_SIDE:
        up = POSE_UPSCALE_MIN_SIDE / min(bgr.shape[:2])
        bgr = cv2.resize(bgr, None, fx=up, fy=up, interpolation=cv2.INTER_LANCZOS4)
    h, w = bgr.shape[:2]
    bboxes = wb.det_model(bgr)
    if len(bboxes) == 0:  # 人物検出に失敗した場合は画面全体を 1 人として推定（バストアップ等の保険）
        bboxes = [[0, 0, w, h]]
    kpts, scores = wb.pose_model(bgr, bboxes=bboxes)  # (N, 134, 2), (N, 134)
    out = {"n_people": int(kpts.shape[0]), "kpts": None, "scores": None, "vis": img}
    if kpts.shape[0] == 0:
        return out
    best = _pose_pick_person(kpts, scores)
    b0 = bboxes[best]
    cx, cy = (b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2
    bw, bh = b0[2] - b0[0], b0[3] - b0[1]

    flipped = np.ascontiguousarray(bgr[:, ::-1]) if POSE_TTA else None
    acc_kp = np.zeros((134, 2), dtype=np.float64)
    acc_w = np.zeros(134, dtype=np.float64)
    sc_list = []
    for s in POSE_BOX_SCALES:
        box = [cx - bw * s / 2, cy - bh * s / 2, cx + bw * s / 2, cy + bh * s / 2]
        kp, sc = wb.pose_model(bgr, bboxes=[box])
        kp, sc = kp[0].astype(np.float64), sc[0].astype(np.float64)
        acc_kp += kp * sc[:, None]
        acc_w += sc
        sc_list.append(sc)
        if flipped is not None:
            fb = [w - 1 - box[2], box[1], w - 1 - box[0], box[3]]
            kf, sf = wb.pose_model(flipped, bboxes=[fb])
            kf, sf = _pose_flip_back(kf[0].astype(np.float64), sf[0].astype(np.float64), w)
            acc_kp += kf * sf[:, None]
            acc_w += sf
            sc_list.append(sf)
    kp = (acc_kp / (acc_w[:, None] + 1e-6)).astype(np.float32)
    sc = np.mean(sc_list, axis=0).astype(np.float32)

    # 可視化（拡大した座標系のまま描き、最後に元サイズへ）。体は POSE_KPT_THR、手は POSE_HAND_THR で描画
    vis = draw_skeleton(bgr.copy(), kp[None, :18], sc[None, :18], openpose_skeleton=True, kpt_thr=POSE_KPT_THR)
    hand_kp = np.concatenate([kp[None, POSE_LHAND:POSE_LHAND + 21], kp[None, POSE_RHAND:POSE_RHAND + 21]], axis=0)
    hand_sc = np.concatenate([sc[None, POSE_LHAND:POSE_LHAND + 21], sc[None, POSE_RHAND:POSE_RHAND + 21]], axis=0)
    vis = draw_skeleton(vis, hand_kp, hand_sc, openpose_skeleton=False, kpt_thr=POSE_HAND_THR, radius=2, line_width=2)
    if up != 1.0:
        vis = cv2.resize(vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        kp = kp / up
    out["kpts"] = kp
    out["scores"] = sc
    out["vis"] = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return out


def _limb_cosines(p1: dict, p2: dict, limbs, offset: int, thr: float):
    """関節ペア（骨）の向きベクトルのコサインと重み（両画像の最小信頼度）を返す。"""
    cos_list, weights, idx = [], [], []
    for n, (i, j) in enumerate(limbs):
        a, b = i + offset, j + offset
        conf = min(p1["scores"][a], p1["scores"][b], p2["scores"][a], p2["scores"][b])
        if conf < thr:
            continue
        v1 = p1["kpts"][b] - p1["kpts"][a]
        v2 = p2["kpts"][b] - p2["kpts"][a]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        cos_list.append(float(v1 @ v2 / (n1 * n2)))
        weights.append(float(conf))
        idx.append(n)
    return cos_list, weights, idx


def pose_metric(p1: dict, p2: dict) -> dict:
    base = {"vis_a": p1["vis"], "vis_b": p2["vis"]}
    if p1["kpts"] is None or p2["kpts"] is None:
        who = "両方" if p1["kpts"] is None and p2["kpts"] is None else ("画像A" if p1["kpts"] is None else "画像B")
        return {**base, "skipped": f"{who}で人物（ポーズ）を検出できませんでした", "raw": None}

    body_cos, body_w, body_idx = _limb_cosines(p1, p2, POSE_LIMBS, 0, POSE_KPT_THR)
    if len(body_cos) < POSE_MIN_LIMBS:
        return {
            **base,
            "skipped": f"両画像で共通して検出できた関節ペアが {len(body_cos)} 本のみ（{POSE_MIN_LIMBS} 本以上必要）",
            "raw": None,
        }
    body_raw = float(np.average(body_cos, weights=body_w))

    # 手（指）: 片手ごとに共通の骨が POSE_HAND_MIN_LIMBS 本以上あれば比較に含める
    hands = {}
    for name, off in (("左手", POSE_LHAND), ("右手", POSE_RHAND)):
        hc, hw, _ = _limb_cosines(p1, p2, POSE_HAND_LIMBS, off, POSE_HAND_THR)
        if len(hc) >= POSE_HAND_MIN_LIMBS:
            hands[name] = (float(np.average(hc, weights=hw)), len(hc))
    if hands:
        hand_raw = float(np.mean([v[0] for v in hands.values()]))
        raw = (body_raw * 1.0 + hand_raw * POSE_HAND_WEIGHT) / (1.0 + POSE_HAND_WEIGHT)
    else:
        hand_raw = None
        raw = body_raw

    c = CALIBRATION["pose"]
    score = _piecewise_score(raw, c["anchors"], c["kind"])
    worst = sorted(zip(body_cos, [POSE_LIMB_NAMES[i] for i in body_idx]))[:3]
    detail = f"limb cos={raw:.3f}（1.0で完全一致, 体の関節ペア {len(body_cos)} 本"
    if hands:
        detail += "、" + " / ".join(f"{k}の指 {v[1]} 本 (cos {v[0]:.2f})" for k, v in hands.items())
    else:
        detail += "、指は両画像で共通して取れず"
    detail += "）"
    if worst and worst[0][0] < 0.7:
        detail += " / 差が大きい部位: " + ", ".join(f"{n}({cv:.2f})" for cv, n in worst)
    note = None
    if p1["n_people"] > 1 or p2["n_people"] > 1:
        note = f"複数人物を検出（A: {p1['n_people']} / B: {p2['n_people']}）。最も大きい人物同士で比較しています"
    return {
        **base,
        "raw": raw,
        "score": score,
        "interp": _interp_by_score(score, "ポーズ"),
        "detail": detail,
        "note": note,
        "n_limbs": len(body_cos),
        "body_raw": body_raw,
        "hand_raw": hand_raw,
        "hands": {k: v[1] for k, v in hands.items()},
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
    "pixai": {
        "label": "タグ類似度 (PixAI Tagger v1.0)",
        "desc": "30,877タグの大規模アニメタガー（SAM3系 ViTDet, 1008px）の内部埋め込みの一致度。",
        "embed": pixai_embed,
        "fn": pixai_metric,
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
        "desc": "自己教師あり学習の全体特徴（CLS）。物体・形状の類似に強いが位置情報は持たない。",
        "embed": dinov2_embed,
        "fn": dinov2_metric,
    },
    "depth": {
        "label": "構図類似度 (Depth Anything V2)",
        "desc": "深度マップの相関。被写体の画面内配置・大きさ・前景/背景の分離を比較する。",
        "embed": depth_embed,
        "fn": depth_metric,
    },
    "pose": {
        "label": "ポーズ類似度 (DWPose)",
        "desc": "OpenPose 形式の関節から四肢の向きを比較。位置・スケール不変。人物が取れない画像は対象外。",
        "embed": pose_embed,
        "fn": pose_metric,
    },
}

# 総合スコアの重み。calibration の AUC（同一キャラ vs 別キャラ の識別力）に基づく:
#   CCIP 0.999 / PixAI 0.960 / SigLIP2 0.949 / WD14 0.944 / DreamSim 0.941 / DINOv2 0.678
# 最も識別力の高い CCIP を厚めにし、次点の PixAI をやや厚め、同程度の 3 つは均等、
# キャラ識別に弱い DINOv2（構図・形状向け）は補助として軽くしている。
# ENABLE_PIXAI=0 のときは pixai を除いて残りの重みで再正規化される（total_score 参照）。
WEIGHTS = {
    "ccip": 0.25,
    "pixai": 0.20,
    "wd14": 0.10,
    "siglip2": 0.10,
    "dreamsim": 0.10,
    "dinov2": 0.05,
    "depth": 0.10,
    "pose": 0.10,
}


if not ENABLE_PIXAI:
    METRICS.pop("pixai", None)
    WEIGHTS.pop("pixai", None)


def total_score(results: dict):
    """利用可能なメトリクスの重み付き平均（重みは利用分で再正規化）"""
    num, den = 0.0, 0.0
    for key, w in WEIGHTS.items():
        r = results.get(key)
        if r and "score" in r:
            num += r["score"] * w
            den += w
    return round(num / den, 3) if den else None


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
            rows.append([info["label"], round(r["score"], 3), w, round(contrib, 3)])
            num += contrib
            den += w
        else:
            rows.append([info["label"], "-", w, "対象外" if r and "skipped" in r else "計算失敗"])
    return rows, (round(num / den, 3) if den else None), den


def make_comment(results: dict) -> str:
    total = results.get("total")
    if total is None:
        return "計算可能なメトリクスがありませんでした"

    def ok(k):
        return k in results and "score" in results[k]

    lines = [f"総合評価: {band_label(total)}"]
    if "skipped" in results.get("ccip", {}):
        lines.append(f"CCIP（キャラ判定）は対象外のため総合スコアから除外: {results['ccip']['skipped']}")

    # タグ系（PixAI / WD14）の平均スコア。CCIP の判定との整合チェックに使う
    tag_scores = [results[k]["score"] for k in ("pixai", "wd14") if ok(k)]
    tag_avg = sum(tag_scores) / len(tag_scores) if tag_scores else None

    if ok("ccip"):
        ccip = results["ccip"]
        others = [results[k]["score"] for k in ("siglip2", "dreamsim", "wd14", "pixai") if ok(k)]
        if ccip.get("same_character") and ccip["score"] >= 60:
            if tag_avg is not None and tag_avg < 40:
                lines.append(
                    "CCIPは同一キャラ判定ですが、タグ系（PixAI / WD14）は別キャラを示唆しています"
                    "（髪色・獣耳など属性が似た別キャラの可能性）"
                )
            else:
                lines.append("同一キャラクターの可能性が高いです（CCIP基準）")
        elif ccip.get("same_character") is False:
            if others and max(others) >= 60:
                lines.append("画風・構図は近いですが、キャラクターは別人の可能性があります")
            else:
                lines.append("キャラクターも内容も異なる画像と見られます")

    # キャラタグ（PixAI 優先、無ければ WD14）
    common, ta, tb = set(), set(), set()
    for k in ("pixai", "wd14"):
        if ok(k):
            a = set(results[k].get("char_tags_a") or [])
            b = set(results[k].get("char_tags_b") or [])
            common |= a & b
            ta |= a
            tb |= b
    if common:
        lines.append(f"共通キャラタグ検出: {', '.join(sorted(common))}")
    elif ta or tb:
        fa = ", ".join(sorted(ta)[:3]) or "(なし)"
        fb = ", ".join(sorted(tb)[:3]) or "(なし)"
        lines.append(f"検出キャラタグ: 画像A = {fa} / 画像B = {fb}")
    return "\n".join(lines)


def preload(progress=None) -> None:
    """全モデルをロードする（Space起動時に呼ぶ）。"""
    loaders = [
        ("SigLIP 2", _get_siglip),
        ("DINOv2", _get_dino),
        ("DreamSim", _get_dreamsim),
        ("CCIP", _ccip_threshold),
        ("Depth Anything V2", _get_depth),
        ("DWPose", _get_pose),
    ] + ([("PixAI Tagger", _get_pixai)] if "pixai" in METRICS else [])
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
