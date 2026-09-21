"""イラスト一致度スコア - Hugging Face Space (Gradio)

ローカル起動:
    python app.py
"""

import os
from pathlib import Path

try:  # ZeroGPU Space では @spaces.GPU が必須。CPU Space / ローカルでは no-op
    import spaces  # noqa: F401  (torch より先に import する必要がある)

    # 実測は GPU で 1 組 30 秒前後。予約時間が長いと匿名ユーザーの ZeroGPU quota で弾かれるため短めにする
    _gpu = spaces.GPU(duration=60)
except Exception:  # spaces 未インストール等
    def _gpu(fn):
        return fn

import gradio as gr
from PIL import Image

import similarity

SAMPLES_DIR = Path(__file__).parent / "samples"
EXAMPLES = [
    # ご注文はうさぎですか？ の保登心愛 vs 保登モカ（姉妹。衣装・構図がほぼ同じ別キャラの例）
    [str(SAMPLES_DIR / "hoto-cocoa.png"), str(SAMPLES_DIR / "hoto-mocha.png")],
    # ブルーアーカイブのシロコ vs ホロライブの白上フブキ（別キャラだが銀髪・獣耳など属性が近い例）
    [str(SAMPLES_DIR / "shiroko.png"), str(SAMPLES_DIR / "fubuki.png")],
    # パピルス vs マリ（別作品・別キャラだが、橙髪・フード・胸元で手を組むポーズと構図が近い例）
    [str(SAMPLES_DIR / "papyrus.png"), str(SAMPLES_DIR / "mari.png")],
    # 天下一品ロゴ vs 進入禁止標識（似ていると言われる有名な組み合わせ。非アニメ画像の例）
    [str(SAMPLES_DIR / "tenkaippin_01.jpg"), str(SAMPLES_DIR / "shinnyu_kinshi_02.jpg")],
]


def _is_image(path: str) -> bool:
    """実体のある画像のみ許可（git-lfs 未導入で clone すると LFS ポインタのテキストになるため）"""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


EXAMPLES = [e for e in EXAMPLES if all(_is_image(p) for p in e)]

BREAKDOWN_HEADERS = ["メトリクス", "スコア", "重み", "寄与"]

MAX_SIDE = 1536  # 巨大画像はメモリ節約のため事前に縮小


def _shrink(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def _score_bar(score: float) -> str:
    pct = max(0, min(100, int(round(score))))
    return (
        '<div style="background:#e5e7eb;border-radius:6px;height:10px;width:100%">'
        f'<div style="background:#f97316;width:{pct}%;height:10px;border-radius:6px"></div></div>'
    )


def _metric_card(key: str, r: dict) -> str:
    info = similarity.METRICS[key]
    head = f"**{info['label']}**  \n<small>{info['desc']}</small>"
    if "error" in r:
        return f"{head}\n\n⚠️ 計算失敗: `{r['error']}`"
    if "skipped" in r:
        ref = f"参考値 {r['raw']:.3f} / " if r.get("raw") is not None else ""
        return f"{head}\n\n### 対象外\n\n{r['skipped']}  \n<small>{ref}総合スコアには含めません</small>"
    body = (
        f"{head}\n\n"
        f"### {r['score']:.3f} / 100\n"
        f"{_score_bar(r['score'])}\n\n"
        f"**→ {r['interp']}**  \n<small>{r['detail']}</small>"
    )
    if r.get("note"):
        body += f"\n\n<small>⚠️ {r['note']}</small>"
    if r.get("shared_tags"):
        body += "\n\n<small>共通タグ: " + ", ".join(r["shared_tags"][:12]) + "</small>"
    return body


@_gpu
def run(img_a, img_b, progress=gr.Progress()):
    if img_a is None or img_b is None:
        raise gr.Error("画像Aと画像Bの両方を指定してください")
    a, b = _shrink(img_a), _shrink(img_b)
    try:  # GPU があれば torch 系モデルを CUDA で推論（ZeroGPU では関数内でのみ利用可）
        similarity.set_device("cuda")
    except Exception:
        similarity.set_device("cpu")

    def on_progress(i, n, label):
        progress((i, n), desc=f"計算中: {label}")

    res = similarity.compute_all(a, b, on_progress=on_progress)
    total = res["total"]
    if total is None:
        summary = "## ⚠️ すべてのメトリクスで計算に失敗しました"
    else:
        summary = (
            f"## 総合スコア: {total:.3f} / 100（{similarity.band_label(total)}）\n"
            f"{_score_bar(total)}\n\n"
            + "\n".join(f"- {line}" for line in res["comment"].splitlines())
        )
    rows, _, den = similarity.score_breakdown(res)
    note = "" if den >= 0.999 else "※対象外・計算失敗のメトリクスを除き、残りの重みで正規化しています"
    cards = [_metric_card(k, res.get(k, {})) for k in similarity.METRICS]
    vis = [
        res.get("depth", {}).get("vis_a"),
        res.get("depth", {}).get("vis_b"),
        res.get("pose", {}).get("vis_a"),
        res.get("pose", {}).get("vis_b"),
    ]
    return [summary, rows, note, *cards, *vis]


WEIGHT_TEXT = " / ".join(
    f"{similarity.METRICS[k]['label'].split(' (')[-1].rstrip(')')} {w}"
    for k, w in similarity.WEIGHTS.items()
)

CALIB_TABLE = "\n".join(
    f"| {k} | {c['kind']} | " + " | ".join(f"{v:.3f}" for v in c["anchors"]) + " |"
    for k, c in similarity.CALIBRATION.items()
)

HOW_TO_READ = f"""
### スコアの見方

総合スコアは各メトリクスの **重み付き平均** です（重み: {WEIGHT_TEXT}）。

| スコア | 目安 |
|---|---|
| 80〜 | かなり似ている（同一キャラの別カット〜ほぼ同じ絵） |
| 60〜 | ある程度似ている（同一キャラの可能性が高い） |
| 40〜 | 部分的に似ている（画風・構図は近いがキャラは別など） |
| 〜40 | あまり似ていない |

各メトリクスの生値は、実画像ペア（アニメ画像 8 キャラ × 6 枚 + 増強画像）を 4 カテゴリに分けて
実測した中央値をアンカーとする区分線形マップで 0〜100 に変換しています
（近似複製=100 / 同一キャラ=70 / 別キャラ=30 / 無関係=0）。
つまり「同一キャラの別カットなら各メトリクスとも 70 点前後、別キャラなら 30 点前後」が目盛りの基準です。

| メトリクス | 生値 | 無関係(0点) | 別キャラ(30点) | 同一キャラ(70点) | 近似複製(100点) |
|---|---|---|---|---|---|
{CALIB_TABLE}

**構図・ポーズの較正カテゴリ**: この 2 つは「キャラが同じか」ではなく「配置・ポーズが同じか」を測るため、
較正カテゴリを 無関係=0 / 別カット（同一・別キャラ問わず別の絵）=30 / 反転・トリミング=70 / 同構図（明度・JPEG・縮小のみ）=100 に置き換えています。

**生値の意味**
- SigLIP 2 / DINOv2 / WD14 / PixAI `cosine`: 埋め込みのコサイン類似度。1.0 で完全一致
- DreamSim `distance`: 知覚的距離。0 で完全一致
- Depth `corr`: 正規化した深度マップ（64×64）のピアソン相関。1.0 で完全一致。左右反転は別構図扱い
- DWPose `limb cos`: 両画像で検出できた関節ペア（四肢）の向きベクトルのコサイン平均。1.0 で完全一致。
  人物が検出できない、または共通の関節ペアが 4 本未満なら「対象外」
- CCIP `difference`: キャラ間の距離。モデル既定の閾値 0.178 未満なら同一キャラ判定（このスケールで約 54 点）。
  CCIP は「別キャラ」と「無関係画像」を区別しないため、0 点のアンカーだけ別キャラの 95 パーセンタイルを使用

**CCIP の適用ガード**: WD14 で `no_humans` が付き人物タグが無い画像（風景など）が含まれる場合、
CCIP は「対象外」として総合スコアから除外し、残りの重みで正規化します。
複数キャラが検出された場合はスコアを出しつつ注意書きを表示します。

注意: CCIP / WD14 / PixAI はアニメイラスト前提のモデルで、キャリブレーションもアニメ画像で行っています。
実写などでは目盛りがずれます。
"""

with gr.Blocks(title="イラスト一致度スコア") as demo:
    gr.Markdown(
        "# 🎨 イラスト一致度スコア\n"
        "2枚のイラストを入力すると、**キャラクター・タグ（2種）・知覚・意味・視覚特徴** の"
        f"{len(similarity.METRICS)}観点から一致度を 0〜100 でスコア化します。"
    )
    with gr.Row():
        img_a = gr.Image(type="pil", label="画像A", height=360)
        img_b = gr.Image(type="pil", label="画像B", height=360)
    btn = gr.Button("類似度を計算", variant="primary")

    summary = gr.Markdown()
    cards = []
    keys = list(similarity.METRICS)
    for i in range(0, len(keys), 4):  # 4 列ずつ並べる
        with gr.Row():
            for _ in keys[i : i + 4]:
                cards.append(gr.Markdown())
    with gr.Accordion("構図・ポーズの可視化（Depth マップ / OpenPose 骨格）", open=True):
        with gr.Row():
            vis_depth_a = gr.Image(label="Depth 画像A", interactive=False, height=300)
            vis_depth_b = gr.Image(label="Depth 画像B", interactive=False, height=300)
            vis_pose_a = gr.Image(label="ポーズ 画像A", interactive=False, height=300)
            vis_pose_b = gr.Image(label="ポーズ 画像B", interactive=False, height=300)
        gr.Markdown(
            "<small>Depth: 明るいほど手前（Depth Anything V2 の相対深度）。"
            "ポーズ: 信頼度 0.5 以上の関節のみ描画（DWPose, OpenPose 18 点形式）。</small>"
        )
    with gr.Accordion("スコアの内訳", open=False):
        breakdown = gr.Dataframe(
            headers=BREAKDOWN_HEADERS, datatype=["str", "str", "number", "str"],
            interactive=False,
        )
        note = gr.Markdown()
    with gr.Accordion("スコアの見方・キャリブレーション", open=False):
        gr.Markdown(HOW_TO_READ)

    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=[img_a, img_b],
            label="サンプル（ココア vs モカ / シロコ vs フブキ / パピルス vs マリ / 天下一品ロゴ vs 進入禁止標識）",
            cache_examples=False,
        )

    btn.click(
        run,
        inputs=[img_a, img_b],
        outputs=[summary, breakdown, note, *cards, vis_depth_a, vis_depth_b, vis_pose_a, vis_pose_b],
    )


if __name__ == "__main__":
    print("モデルを事前ロード中...", flush=True)
    similarity.preload(progress=lambda name: print(f"  loading {name}", flush=True))
    print("ロード完了", flush=True)
    # OPEN_BROWSER=1（start.bat が設定）のときだけブラウザを自動で開く。Space では未設定
    demo.queue(default_concurrency_limit=1).launch(
        inbrowser=os.environ.get("OPEN_BROWSER") == "1"
    )
