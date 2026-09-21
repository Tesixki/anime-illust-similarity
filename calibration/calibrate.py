"""スコア較正用スクリプト: 実画像ペアで各メトリクスの生値分布を実測する。

使い方:
    python calibration/calibrate.py <画像ディレクトリ> [出力JSON]

<画像ディレクトリ>/<キャラID>/*.jpg|png のように、キャラクターごとにサブディレクトリを
分けて画像を置く。以下 4 カテゴリのペアを作り、各メトリクスの生値のパーセンタイルと
同一キャラ vs 別キャラ の AUC を出力する。

    dup        : 同じ絵に軽い変換（左右反転 / 85%トリミング / 明度0.8 / JPEG q=30 / 半解像度）
    same_char  : 同じキャラの別の絵
    diff_char  : 別キャラ（同じディレクトリ群の中の別キャラ）
    unrelated  : キャラ画像 vs <画像ディレクトリ>/_unrelated/ の風景・無関係画像（任意）

出力 JSON の stats[<cat>][<metric>].pct["50"] が similarity.CALIBRATION のアンカー値。
"""

from __future__ import annotations

import io
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import similarity  # noqa: E402

AUGS = ["hflip", "crop85", "bright", "jpeg30", "small256"]
PCT = [5, 25, 50, 75, 95]


def aug(img: Image.Image, kind: str) -> Image.Image:
    if kind == "hflip":
        return ImageOps.mirror(img)
    if kind == "crop85":
        w, h = img.size
        m = 0.075
        return img.crop((int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m)))).resize(
            (w, h), Image.BICUBIC
        )
    if kind == "bright":
        return ImageEnhance.Brightness(img).enhance(0.8)
    if kind == "jpeg30":
        b = io.BytesIO()
        img.save(b, "JPEG", quality=30)
        b.seek(0)
        return Image.open(b).convert("RGB")
    if kind == "small256":
        w, h = img.size
        return img.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    raise ValueError(kind)


def raw_value(key: str, e1, e2) -> float:
    return similarity.METRICS[key]["fn"](e1, e2)["raw"]


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main(img_dir: Path, out_path: Path, n_aug_images: int = 12, n_diff_pairs: int = 200, seed: int = 0):
    random.seed(seed)
    imgs: dict[str, tuple[str, Image.Image]] = {}
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.parent != img_dir:
            if p.parent.name == "_unrelated":  # 無関係画像は 1 枚ずつ別グループ扱い
                imgs[f"_unrelated/{p.stem}"] = ("_unrelated_" + p.stem, Image.open(p).convert("RGB"))
            else:
                imgs[f"{p.parent.name}/{p.stem}"] = (p.parent.name, Image.open(p).convert("RGB"))
    anime = [n for n in imgs if not n.startswith("_unrelated/")]
    proc = [n for n in imgs if n.startswith("_unrelated/")]
    if len(anime) < 4:
        sys.exit(f"画像が足りません: {img_dir}")

    aug_pairs = [(n, k) for n in random.sample(anime, min(n_aug_images, len(anime))) for k in AUGS]
    all_imgs = {n: v[1] for n, v in imgs.items()}
    all_imgs.update({f"{n}|{k}": aug(imgs[n][1], k) for n, k in aug_pairs})
    print(f"images={len(imgs)} (+{len(aug_pairs)} augmented)", flush=True)

    feats: dict[str, dict] = {k: {} for k in similarity.METRICS}
    timing = {k: 0.0 for k in similarity.METRICS}
    for i, (n, im) in enumerate(all_imgs.items()):
        for k, info in similarity.METRICS.items():
            t = time.time()
            feats[k][n] = info["embed"](im)
            timing[k] += time.time() - t
        if i % 10 == 0:
            print(f"  embedded {i + 1}/{len(all_imgs)}", flush=True)

    groups: dict[str, list[str]] = {}
    for n in anime:
        groups.setdefault(imgs[n][0], []).append(n)
    pairs = {
        "dup": [(n, f"{n}|{k}") for n, k in aug_pairs],
        "same_char": [p for g in groups.values() for p in itertools.combinations(g, 2)],
    }
    for k in AUGS:
        pairs[f"dup_{k}"] = [(n, f"{n}|{kk}") for n, kk in aug_pairs if kk == k]
    diff = [(a, b) for a, b in itertools.combinations(anime, 2) if imgs[a][0] != imgs[b][0]]
    pairs["diff_char"] = random.sample(diff, min(n_diff_pairs, len(diff)))
    if proc:
        pairs["unrelated"] = [(a, b) for a in random.sample(anime, min(20, len(anime))) for b in proc]
        pairs["proc_pairs"] = list(itertools.combinations(proc, 2))

    out = {
        "n_images": len(imgs),
        "n_characters": len(groups),
        "timing_per_image_sec": {k: round(v / len(all_imgs), 3) for k, v in timing.items()},
        "stats": {},
        "auc": {},
        "suggested_anchors": {},
    }
    for cat, ps in pairs.items():
        out["stats"][cat] = {}
        for m in similarity.METRICS:
            v = np.array([raw_value(m, feats[m][a], feats[m][b]) for a, b in ps])
            out["stats"][cat][m] = {
                "n": len(v),
                "pct": dict(zip(map(str, PCT), np.percentile(v, PCT).round(4).tolist())),
                "mean": round(float(v.mean()), 4),
            }
    for m, c in similarity.CALIBRATION.items():
        sgn = -1 if c["kind"] == "distance" else 1
        s = [sgn * raw_value(m, feats[m][a], feats[m][b]) for a, b in pairs["same_char"]]
        d = [sgn * raw_value(m, feats[m][a], feats[m][b]) for a, b in pairs["diff_char"]]
        out["auc"][m] = {"same_char_vs_diff_char": round(auc(s, d), 4)}
        if "unrelated" in pairs:
            u = [sgn * raw_value(m, feats[m][a], feats[m][b]) for a, b in pairs["unrelated"]]
            out["auc"][m]["diff_char_vs_unrelated"] = round(auc(d, u), 4)
        out["suggested_anchors"][m] = [
            out["stats"][cat][m]["pct"]["50"] for cat in ("unrelated", "diff_char", "same_char", "dup") if cat in pairs
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for cat in pairs:
        print(f"\n== {cat} (n={len(pairs[cat])})")
        for m in similarity.METRICS:
            p = out["stats"][cat][m]["pct"]
            print(f"  {m:9s} " + " ".join(f"p{q}={p[str(q)]:.3f}" for q in PCT))
    print("\nAUC:", json.dumps(out["auc"], indent=1))
    print("suggested CALIBRATION anchors (unrelated, diff_char, same_char, dup):")
    for m, a in out["suggested_anchors"].items():
        print(f"  {m}: {tuple(a)}")
    print("->", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "calibration" / "calibration_result.json")
