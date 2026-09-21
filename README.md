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
  - pixai-labs/pixai-tagger-v1.0
  - timm/ViT-B-16-SigLIP2-256
  - facebook/dinov2-small
  - deepghs/ccip
  - SmilingWolf/wd-swinv2-tagger-v3
---

# イラスト一致度スコア

2枚のイラスト画像を入力すると、一致度を **0〜100 のスコア** で返す Hugging Face Space です。
主にアニメ・二次元イラストを対象とし、ZeroGPU（`@spaces.GPU`）と CPU の両方で動作します。

## 計算するメトリクス

| メトリクス | 観点 | 使用モデル | 生値 |
|---|---|---|---|
| キャラクター類似度 (CCIP) | 描かれているキャラが同一か | [deepghs/imgutils](https://github.com/deepghs/imgutils) CCIP caformer（ONNX） | difference（小さいほど同一） |
| タグ類似度 (WD14 tagger v3) | キャラ名・属性・服装タグの一致 | WD SwinV2 tagger v3 の内部埋め込み（ONNX） | cosine |
| タグ類似度 (PixAI Tagger v1.0) | 30,877 タグの大規模タガーによるタグ・キャラ・画風の一致 | [pixai-labs/pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0)（SAM3 系 ViTDet 486M, 1008px）の attention-pool 埋め込み | cosine |
| 知覚的類似度 (DreamSim) | 人間の知覚に近い画像間距離 | DreamSim ensemble（CLIP+DINO+OpenCLIP, LoRA微調整） | distance = 1 − cosine |
| セマンティック類似度 (SigLIP 2) | 画像全体の意味・内容の近さ | OpenCLIP ViT-B-16-SigLIP2-256 (WebLI) | cosine |
| 視覚特徴類似度 (DINOv2) | 物体・形状など全体的な視覚特徴（位置情報は持たない） | facebook/dinov2-small（CLS 埋め込み） | cosine |
| 構図類似度 (Depth Anything V2) | 被写体の画面内配置・大きさ・前景/背景の分離 | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) の深度マップ（正規化 → 64×64） | Pearson corr |
| ポーズ類似度 (DWPose) | 四肢の向き（位置・スケール不変） | [rtmlib](https://github.com/Tau-J/rtmlib) の DWPose（OpenPose 18 点形式, ONNX） | limb cos |

PixAI Tagger は分類ヘッド直前の 1024 次元ベクトルを埋め込みに使い、同じ forward で得たタグ確率から
共通タグ・キャラタグも表示します（閾値はモデルカード推奨の general 0.17 / character 0.27 / style 0.15 / copyright 0.24）。
非常に重いモデルのため、CPU では 1 枚あたり数十秒かかります（`ENABLE_PIXAI=0` で無効化可能）。

構図とポーズは UI に可視化を出します（Depth マップの色分け画像と、OpenPose 形式の骨格を重ねた画像を A/B それぞれ表示）。
Depth は左右反転を別構図として扱い、反転すると相関が大きく上がる場合は「鏡像構図の可能性」を注記します。

ポーズ推定の精度対策（バストアップ等で関節が部分的にしか取れない問題への対応）:
- rtmlib の `performance` モード（RTMW-DW x-l, 入力 384×288, 検出器 yolox_x）を使用。`balanced`（256×192）より関節位置が安定します。
  このモードの ONNX は信頼度が未正規化（可視の目・鼻で 5〜7、隠れた手首・膝で 1.6〜2.8）なので、閾値は実測に基づき 2.8 にしています。
- 左右反転 TTA: 同じ人物ボックスを反転して再推定し、左右ラベルを戻してから座標を信頼度で加重平均。
- 人物検出に失敗した場合は画面全体を 1 人として推定（バストアップの保険）。
- 四肢ごとの比較は両画像の関節信頼度の最小値で加重平均し、境界付近の関節の影響を抑制。
- それでも共通して取れた関節ペアが 4 本未満なら「対象外」として総合から外し、カードに比較した関節ペア数を表示します。
  見えない関節（バストアップの下半身など）はどのモデルでも推定できないため、これは仕様上の限界です。

総合スコアは各メトリクスの重み付き平均
（CCIP 0.25 / PixAI 0.20 / WD14 0.10 / SigLIP 2 0.10 / DreamSim 0.10 / DINOv2 0.05 / 構図 0.10 / ポーズ 0.10）です。

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
| PixAI Tagger | cosine | 0.282 | 0.508 | 0.742 | 0.978 | 0.960 |
| SigLIP 2 | cosine | 0.566 | 0.800 | 0.904 | 0.988 | 0.949 |
| WD14 | cosine | 0.451 | 0.530 | 0.748 | 0.991 | 0.944 |
| DreamSim | distance | 0.758 | 0.532 | 0.314 | 0.023 | 0.941 |
| DINOv2 | cosine | 0.306 | 0.546 | 0.620 | 0.977 | 0.678 |

※ CCIP はキャラ同一性のモデルで「別キャラ」と「無関係画像」を区別しません
（実測: 別キャラ中央値 0.327 / 無関係中央値 0.340、AUC 0.53）。そのため CCIP だけ 0 点のアンカーに
別キャラの 95 パーセンタイルを使っています。この目盛りではモデル既定の判定閾値 0.178 が約 54 点に当たり、
「50 点前後 = 同一キャラかどうかの境界」と読めます。

### 構図・ポーズの較正

構図（Depth）とポーズ（DWPose）は「キャラが同じか」ではなく「配置・ポーズが同じか」を測るので、
較正カテゴリを次のように置き換えて同じ画像セット（48 枚 + 増強 60 枚）で実測しました
（`calibration/calibrate_composition.py` → `calibration/calibration_composition.json`）。

| カテゴリ | 内容 | スコア |
|---|---|---|
| same_comp | 同じ絵に構図を変えない変換（明度 0.8 / JPEG q=30 / 半解像度） | 100 |
| shifted | 同じ絵に構図を変える変換（左右反転 / 85% トリミング） | 70 |
| other_cut | 別の絵（同一キャラの別カット + 別キャラ） | 30 |
| unrelated | アニメ画像 vs 風景・ロゴ（ポーズはほぼ「対象外」） | 0 |

| メトリクス | 生値 | unrelated (0点) | other_cut (30点) | shifted (70点) | same_comp (100点) | AUC same_comp vs other_cut |
|---|---|---|---|---|---|---|
| Depth | Pearson corr | 0.285 | 0.451 | 0.805 | 0.999 | 1.000 |
| DWPose | limb cos | −0.164 ※ | 0.806 | 0.978 | 1.000 | 0.995 |

※ ポーズは無関係画像がほぼ対象外になるため、0 点アンカーは other_cut の 5 パーセンタイルです。
別カット同士でも limb cos の中央値が 0.81 と高い（アニメのフレームは正面バストアップが多く、首→肩・鼻→目などの向きが揃う）ため、
ポーズの目盛りは「ほぼ一致しないと高得点にならない」厳しめになっています。
左右反転は Depth では別構図（中央値 0.689）、ポーズでも四肢の向きが反転するため下がります（中央値 0.875）。

**重みの根拠**: 同一キャラ vs 別キャラ の AUC に基づき、最も識別力の高い CCIP を 0.25、次点の PixAI Tagger を 0.20、
同程度の SigLIP 2 / WD14 / DreamSim を各 0.10、キャラ識別に弱い DINOv2 を 0.05 にし、
キャラ同一性とは独立な軸である構図とポーズに各 0.10 を割り当てています。
`ENABLE_PIXAI=0` の場合や対象外のメトリクスがある場合は、残りの重みで再正規化されます。

**CCIP の適用ガード**: WD14 で `no_humans` が付き人物タグ（solo / 1girl / 1boy など）が無い画像が含まれる場合、
CCIP は「対象外」として総合スコアから除外し、残りの重みで正規化します（風景に対して CCIP が
境界付近の値を返してしまう問題への対処）。複数キャラが検出された場合はスコアを出しつつ注意書きを表示します。

**POC からの主な変更点**
- 構図類似度（Depth Anything V2 の深度マップ相関）とポーズ類似度（DWPose の四肢向き比較）を追加し、Depth マップと骨格を UI に可視化
- PixAI Tagger v1.0 をタグ系メトリクスとして追加（同一 vs 別キャラ AUC 0.960 で WD14 より高い）
- ZeroGPU 対応（`@spaces.GPU` 内で torch 系モデルを CUDA へ移動、CPU でもそのまま動作）
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

初回起動時にモデル（合計約 4.5GB、うち PixAI Tagger が約 2GB）が `~/.cache` にダウンロードされます。

環境変数:

| 変数 | 既定値 | 説明 |
|---|---|---|
| `ENABLE_PIXAI` | `1` | `0` で PixAI Tagger を無効化（CPU 環境で軽くしたい場合） |
| `DINO_MODEL` | `facebook/dinov2-small` | DINO 系モデル（変更した場合は再較正が必要） |
| `DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | 深度推定モデル（変更した場合は再較正が必要） |
| `POSE_MODE` | `performance` | rtmlib のモード（`lightweight` / `balanced` / `performance`）。変更した場合は再較正が必要 |
| `POSE_KPT_THR` | `2.8`（performance 時）/ `0.5` | 関節信頼度の閾値 |
| `POSE_TTA` | `1` | `0` で左右反転 TTA を無効化 |
| `DREAMSIM_CACHE` | `~/.cache/dreamsim` | DreamSim 重みの保存先 |

OpenCV（rtmlib の依存）のため Linux では `libgl1` と `libglib2.0-0` が必要です（Space では `packages.txt` で導入）。

## Space のハードウェア

ZeroGPU では `run()` が `@spaces.GPU` で包まれ、torch 系モデル（SigLIP 2 / DINOv2 / DreamSim / PixAI）が
関数内で CUDA に移されます。CPU Space やローカルでは同じコードがそのまま CPU で動きます
（ONNX 系の CCIP / WD14 は常に CPU）。

CLI での確認:

```bash
python similarity.py image_a.png image_b.png
```

## キャリブレーションの再実行

```bash
python calibration/calibrate.py <画像ディレクトリ> [出力JSON]
```

`<画像ディレクトリ>/<キャラID>/*.jpg` の構成で画像を置くと、同じ 4 カテゴリで分布を再計測し、
`CALIBRATION` に貼るためのアンカー値を出力します。無関係画像（風景など）は `<画像ディレクトリ>/_unrelated/` に置きます
（省略時は unrelated カテゴリを計測しません）。自分のデータで目盛りを調整したい場合に使ってください。

## ファイル構成

```
app.py                              # Gradio UI
similarity.py                       # 埋め込み抽出・スコア化・総合スコア（CALIBRATION / WEIGHTS）
calibration/calibrate.py            # 実画像ペアで生値の分布を実測するスクリプト（キャラ同一性系）
calibration/calibration_result.json # 実測結果（アンカー値・AUC・処理時間）
calibration/calibrate_composition.py      # 構図・ポーズ用の較正スクリプト（カテゴリが異なる）
calibration/calibration_composition.json  # 構図・ポーズの実測結果
packages.txt                        # Space 用 apt パッケージ（OpenCV の libGL）
samples/                            # サンプル画像（ココア / モカ、シロコ / フブキ、パピルス / マリ、天下一品ロゴ / 進入禁止標識）
requirements.txt
```

## License

MIT（各モデルはそれぞれのライセンスに従います）
