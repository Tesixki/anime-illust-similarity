"""構図（Depth）・ポーズ（DWPose）メトリクスの較正スクリプト。

使い方:
    python calibration/calibrate_composition.py <画像ディレクトリ> [出力JSON]

<画像ディレクトリ>/<キャラID>/*.jpg と、任意で <画像ディレクトリ>/_unrelated/*.png|jpg（風景・ロゴなど）。
キャラの同一性ではなく「配置・ポーズが同じか」を測るため、calibrate.py とは別のカテゴリで実測する:

    same_comp : 同じ絵に構図を変えない変換（明度 0.8 / JPEG q=30 / 半解像度）          -> 100 点
    shifted   : 同じ絵に構図を変える変換（左右反転 / 85% トリミング）                     ->  70 点
    other_cut : 別の絵（同一キャラの別カット + 別キャラ）                                 ->  30 点
    unrelated : キャラ画像 vs _unrelated/ の画像（ポーズは多くが対象外になる）             ->   0 点

出力 JSON の suggested_anchors を similarity.CALIBRATION の depth / pose に貼る。
ポーズは unrelated がほぼ対象外になるため、0 点アンカーには other_cut の 5 パーセンタイルを使う。
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

SAME_AUGS = ["bright", "jpeg30", "small256"]
SHIFT_AUGS = ["hflip", "crop85"]
PCT = [5, 25, 50, 75, 95]
METRICS = ["depth", "pose"]


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


def raw_value(key: str, e1, e2):
    r = similarity.METRICS[key]["fn"](e1, e2)
    return r.get("raw") if "score" in r else None  # 対象外は None


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main(img_dir: Path, out_path: Path, n_aug_images: int = 12, n_diff_pairs: int = 200, seed: int = 0):
    random.seed(seed)
    imgs: dict[str, tuple[str, Image.Image]] = {}
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.parent != img_dir:
            if p.parent.name == "_unrelated":
                imgs[f"_unrelated/{p.stem}"] = ("_unrelated_" + p.stem, Image.open(p).convert("RGB"))
            else:
                imgs[f"{p.parent.name}/{p.stem}"] = (p.parent.name, Image.open(p).convert("RGB"))
    anime = [n for n in imgs if not n.startswith("_unrelated/")]
    unrel = [n for n in imgs if n.startswith("_unrelated/")]
    if len(anime) < 4:
        sys.exit(f"画像が足りません: {img_dir}")

    aug_src = random.sample(anime, min(n_aug_images, len(anime)))
    aug_pairs = [(n, k) for n in aug_src for k in SAME_AUGS + SHIFT_AUGS]
    all_imgs = {n: v[1] for n, v in imgs.items()}
    all_imgs.update({f"{n}|{k}": aug(imgs[n][1], k) for n, k in aug_pairs})
    print(f"images={len(imgs)} (+{len(aug_pairs)} augmented)", flush=True)

    feats: dict[str, dict] = {k: {} for k in METRICS}
    timing = {k: 0.0 for k in METRICS}
    for i, (n, im) in enumerate(all_imgs.items()):
        for k in METRICS:
            t = time.time()
            feats[k][n] = similarity.METRICS[k]["embed"](im)
            timing[k] += time.time() - t
        if i % 10 == 0:
            print(f"  embedded {i + 1}/{len(all_imgs)}", flush=True)

    groups: dict[str, list[str]] = {}
    for n in anime:
        groups.setdefault(imgs[n][0], []).append(n)
    same_char = [p for g in groups.values() for p in itertools.combinations(g, 2)]
    diff = [(a, b) for a, b in itertools.combinations(anime, 2) if imgs[a][0] != imgs[b][0]]
    pairs = {
        "same_comp": [(n, f"{n}|{k}") for n, k in aug_pairs if k in SAME_AUGS],
        "shifted": [(n, f"{n}|{k}") for n, k in aug_pairs if k in SHIFT_AUGS],
        "shifted_hflip": [(n, f"{n}|{k}") for n, k in aug_pairs if k == "hflip"],
        "shifted_crop85": [(n, f"{n}|{k}") for n, k in aug_pairs if k == "crop85"],
        "other_cut": same_char + random.sample(diff, min(n_diff_pairs, len(diff))),
        "same_char": same_char,
    }
    if unrel:
        pairs["unrelated"] = [(a, b) for a in random.sample(anime, min(20, len(anime))) for b in unrel]

    out = {
        "description": "構図（depth）・ポーズ（pose）の較正。same_comp=100 / shifted=70 / other_cut=30 / unrelated=0 点。",
        "n_images": len(imgs), "n_characters": len(groups), "n_unrelated": len(unrel),
        "timing_per_image_sec": {k: round(v / len(all_imgs), 3) for k, v in timing.items()},
        "stats": {}, "auc": {}, "suggested_anchors": {},
    }
    vals: dict[str, dict[str, list]] = {}
    for cat, ps in pairs.items():
        out["stats"][cat] = {}
        for m in METRICS:
            v = [raw_value(m, feats[m][a], feats[m][b]) for a, b in ps]
            valid = np.array([x for x in v if x is not None])
            vals.setdefault(m, {})[cat] = valid
            out["stats"][cat][m] = {
                "n": len(v), "n_valid": int(len(valid)),
                "pct": dict(zip(map(str, PCT), np.percentile(valid, PCT).round(4).tolist())) if len(valid) else None,
                "mean": round(float(valid.mean()), 4) if len(valid) else None,
            }
    for m in METRICS:
        v = vals[m]
        out["auc"][m] = {
            "same_comp_vs_other_cut": round(auc(v["same_comp"], v["other_cut"]), 4),
            "shifted_vs_other_cut": round(auc(v["shifted"], v["other_cut"]), 4),
        }
        if "unrelated" in v and len(v["unrelated"]):
            out["auc"][m]["other_cut_vs_unrelated"] = round(auc(v["other_cut"], v["unrelated"]), 4)
        p = lambda cat, q: float(np.percentile(v[cat], q))  # noqa: E731
        zero = p("unrelated", 50) if m == "depth" and len(v.get("unrelated", [])) else p("other_cut", 5)
        out["suggested_anchors"][m] = [round(x, 4) for x in (zero, p("other_cut", 50), p("shifted", 50), p("same_comp", 50))]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for cat in pairs:
        print(f"\n== {cat} (n={len(pairs[cat])})")
        for m in METRICS:
            s = out["stats"][cat][m]
            print(f"  {m:6s} valid={s['n_valid']:3d} " + (" ".join(f"p{q}={s['pct'][str(q)]:.3f}" for q in PCT) if s["pct"] else "-"))
    print("\nAUC:", json.dumps(out["auc"], indent=1))
    print("suggested CALIBRATION anchors (unrelated/other5, other_cut, shifted, same_comp):")
    for m, a in out["suggested_anchors"].items():
        print(f"  {m}: {tuple(a)}")
    print("->", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "calibration" / "calibration_composition.json")
