---
title: Anime Illust Similarity
emoji: 🎨
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
models:
  - timm/ViT-B-16-SigLIP2-256
  - facebook/dinov2-small
  - deepghs/ccip
  - SmilingWolf/wd-swinv2-tagger-v3
---

# イラスト一致度スコア

2枚のイラスト画像を入力すると、一致度を **0〜100 のスコア** で返す Hugging Face Space です。
主にアニメ・二次元イラストを対象とし、CPU のみで動作します（1組あたり数十秒）。

## 計算するメトリクス

| メトリクス | 観点 | 使用モデル | 生値 |
|---|---|---|---|
| キャラクター類似度 (CCIP) | 描かれているキャラが同一か | [deepghs/imgutils](https://github.com/deepghs/imgutils) CCIP caformer（ONNX） | difference（小さいほど同一） |
| タグ類似度 (WD14 tagger v3) | キャラ名・属性・服装タグの一致 | WD SwinV2 tagger v3 の内部埋め込み | cosine |
| 知覚的類似度 (DreamSim) | 人間の知覚に近い画像間距離 | DreamSim ensemble（CLIP+DINO+OpenCLIP, LoRA微調整） | distance = 1 − cosine |
| セマンティック類似度 (SigLIP 2) | 画像全体の意味・内容の近さ | OpenCLIP ViT-B-16-SigLIP2-256 (WebLI) | cosine |
| 視覚特徴類似度 (DINOv2) | 構図・形状・オブジェクトの近さ | facebook/dinov2-small（CLS 埋め込み） | cosine |

総合スコアは各メトリクスの重み付き平均
（CCIP 0.30 / WD14 0.20 / SigLIP 2 0.20 / DreamSim 0.20 / DINOv2 0.10）です。

## スコアの妥当性（キャリブレーション）

生値 → 0〜100 の変換は、実画像ペアを 4 カテゴリに分けて実測した分布に基づく **区分線形マップ** です。
各カテゴリの生値の **中央値** をアンカーに使い、カテゴリとスコアの対応は全メトリクスで共通にしています。

| カテゴリ | 内容 | スコア |
|---|---|---|
| 近似複製 (dup) | 同じ絵に 左右反転 / 85% トリミング / 明度 0.8 / JPEG q=30 / 半解像度 を適用 | 100 |
| 同一キャラ (same_char) | 同じキャラクターの別の絵（別カット） | 70 |
| 別キャラ (diff_char) | 同じ作品内の別キャラクター（画風は同じ） | 30 |
| 無関係 (unrelated) | アニメ画像 vs 風景・無関係な簡易イラスト | 0 |

こうすることで「同一キャラの別カットなら概ね 70 点、別キャラなら 30 点前後」という
共通の目盛りが全メトリクスで揃い、重み付き平均に意味が出ます。

実測に使ったのはキャラ別にグループ化された公開アニメ画像（8 キャラ × 6 枚）と、
そのうち 12 枚に 5 種の変換を加えた増強画像 60 枚です（ペア総数 約 460）。
実測値（中央値）と AUC は次のとおりです（詳細は `calibration/calibration_result.json`）。

| メトリクス | 生値 | 無関係 (0点) | 別キャラ (30点) | 同一キャラ (70点) | 近似複製 (100点) | AUC 同一 vs 別キャラ |
|---|---|---|---|---|---|---|
| CCIP | difference | 0.461 ※ | 0.327 | 0.074 | 0.004 | 0.999 |
| SigLIP 2 | cosine | 0.566 | 0.800 | 0.904 | 0.988 | 0.949 |
| WD14 | cosine | 0.451 | 0.530 | 0.748 | 0.991 | 0.944 |
| DreamSim | distance | 0.758 | 0.532 | 0.314 | 0.023 | 0.941 |
| DINOv2 | cosine | 0.306 | 0.546 | 0.620 | 0.977 | 0.678 |

※ CCIP はキャラ同一性のモデルで「別キャラ」と「無関係画像」を区別しません
（実測: 別キャラ中央値 0.327 / 無関係中央値 0.340、AUC 0.53）。そのため CCIP だけ 0 点のアンカーに
別キャラの 95 パーセンタイルを使っています。この目盛りではモデル既定の判定閾値 0.178 が約 54 点に当たり、
「50 点前後 = 同一キャラかどうかの境界」と読めます。

**重みの根拠**: 同一キャラ vs 別キャラ の AUC に基づき、最も識別力の高い CCIP を 0.30、
同程度の SigLIP 2 / WD14 / DreamSim を各 0.20、キャラ識別に弱く構図・形状向けの DINOv2 を 0.10 にしています。

**CCIP の適用ガード**: WD14 で `no_humans` が付き人物タグ（solo / 1girl / 1boy など）が無い画像が含まれる場合、
CCIP は「対象外」として総合スコアから除外し、残りの重みで正規化します（風景に対して CCIP が
境界付近の値を返してしまう問題への対処）。複数キャラが検出された場合はスコアを出しつつ注意書きを表示します。

**POC からの主な変更点**
- 生値→スコアの変換を「勘で置いた線形アンカー」から「実測中央値による区分線形マップ」に変更
- CCIP の閾値取得失敗時のフォールバックが 0.35 だった誤りを修正（モデル既定値は 0.178）
- DINOv3（ゲート付き）→ 公開モデルの DINOv2-small に固定し、torch.hub 依存を廃止
- 画像ごとに埋め込みを 1 回だけ計算する構成に整理（DreamSim も埋め込みから距離を計算）

> **注意**: CCIP / WD14 は「単一キャラクターが写ったアニメイラスト」を前提としたモデルです。
> キャリブレーションもアニメ画像で行っているため、実写などでは目盛りがずれます。

## ローカルでの起動

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  /  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt gradio
python app.py
```

初回起動時にモデル（合計約 2.5GB）が `~/.cache` にダウンロードされます。

CLI での確認:

```bash
python similarity.py image_a.png image_b.png
```

## キャリブレーションの再実行

```bash
python calibration/calibrate.py <画像ディレクトリ> [出力JSON]
```

`<画像ディレクトリ>/<キャラID>/*.jpg` の構成で画像を置くと、同じ 4 カテゴリで分布を再計測し、
`CALIBRATION` に貼るためのアンカー値を出力します。自分のデータで目盛りを調整したい場合に使ってください。

## ファイル構成

```
app.py                              # Gradio UI
similarity.py                       # 埋め込み抽出・スコア化・総合スコア（CALIBRATION / WEIGHTS）
calibration/calibrate.py            # 実画像ペアで生値の分布を実測するスクリプト
calibration/calibration_result.json # 実測結果（アンカー値・AUC・処理時間）
samples/                            # プロシージャル生成の簡易サンプル（実イラストではない）
requirements.txt
```

## License

MIT（各モデルはそれぞれのライセンスに従います）
