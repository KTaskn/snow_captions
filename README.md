日本語の文章を，"やさしい日本語"の文章に変換する，平易化するためのモデルをつくってください
* t5-largeの日本語pretrainedモデルをバックボーン，またはベースモデルとする
* 下記のデータセットを学習する
    * ./dataset/T15-2020.1.7.xlsx
    * ./dataset/T23-2020.1.7.xlsx
* 1/20をバリデーションデータにする
* バリデーションデータに対して予測の精度評価を実施する
* "やさしい日本語"にたいする評価を都度する

## 手順（例）
### 1. データ準備
```
python scripts/prepare_data.py --input-xlsx dataset/T15-2020.1.7.xlsx dataset/T23-2020.1.7.xlsx
```
- 出力: `data/processed/train.jsonl`, `data/processed/valid.jsonl`
- 統計: `data/processed/stats.json`, `data/processed/stats_by_file.json`

### 2. 学習
```
python scripts/train.py --bf16 --gradient-checkpointing
```
- 既定モデル: `sonoisa/t5-base-japanese-v1.1`
- 再開: `--resume-from outputs/t5-simplify`（最新のcheckpointから）または `--resume-from outputs/t5-simplify/final`（finalを初期値として再開）
```
python scripts/train.py \
    --bf16 \
    --per-device-train-batch-size 192 \
    --per-device-eval-batch-size 192 \
    --gradient-accumulation-steps 4 \
    --dataloader-num-workers 4 \
    --dataloader-pin-memory \
    --save-total-limit 5 \
    --logging-steps 50 \
    --num-train-epochs 15 \
    --learning-rate 2e-4 \
    --eval-metrics sari_bleu \
    --metrics-tokenize fugashi \
    --mecabrc /etc/mecabrc \
    --mecab-dicdir /var/lib/mecab/dic/ipadic-utf8
```

### 3. 評価
```
python scripts/evaluate.py --model outputs/t5-simplify/final --data-file data/processed/valid.jsonl --tokenize fugashi
```
- 指標: SARI（主）、BLEU/ROUGE-L（補助）
```
python scripts/evaluate.py \
    --model outputs/t5-simplify/final \
    --data-file data/processed/valid.jsonl \
    --tokenize fugashi \
    --mecabrc /etc/mecabrc \
    --mecab-dicdir /var/lib/mecab/dic/ipadic-utf8 \
    --batch-size 1024 \
    --num-beams 1 \
    --max-length 192 \
    --bf16 \
    --dataloader-num-workers 4 \
    --dataloader-pin-memory
```

### 4. 損失曲線の可視化
```
python scripts/plot_loss.py \
    --log-file outputs/logs/loss_curve.json \
    --output outputs/logs/loss_curve.png
```

### 5. 予測
```
python scripts/predict.py --model outputs/t5-simplify/final --text "海辺のデッキでサーフボードのようなものをしようとしている人がいる"
```
```
python scripts/predict.py --model outputs/t5-simplify/final --input-file data/input.txt
```
```
python scripts/predict.py --model outputs/t5-simplify/final --input-file data/input.txt --json
```

### 6. STAIR-captionsのキャプション生成・スコアリング・トークン検証
#### 6.1 STAIR-captionsのtrain/valに対してキャプション生成
```
python scripts/generate_stair_simplified.py \
  --model outputs/t5-simplify/final \
  --input-json STAIR-captions/stair_captions_v1.2_train.json \
  --output-json outputs/stair_simplified_train.json \
  --batch-size 1024 \
  --num-beams 4 \
  --no-repeat-ngram-size 3 \
  --repetition-penalty 1.2 \
  --bf16
```
```
python scripts/generate_stair_simplified.py \
  --model outputs/t5-simplify/final \
  --input-json STAIR-captions/stair_captions_v1.2_val.json \
  --output-json outputs/stair_simplified_val.json \
  --batch-size 512 \
  --num-beams 4 \
  --no-repeat-ngram-size 3 \
  --repetition-penalty 1.2 \
  --bf16
```
- 既定で `caption` を生成結果で上書きし，元のキャプションは `source_caption` に保存
- `--caption-field` や `--output-caption-field` で入出力フィールド名を変更可能
- `--num-workers` と `--prefetch-factor` でトークナイズを並列化可能

推奨パラメータ（速度重視）
```
--batch-size 512 --num-beams 1 --bf16 --num-workers 4 --prefetch-factor 2 --pin-memory
```
推奨パラメータ（品質バランス）
```
--batch-size 128 --num-beams 4 --no-repeat-ngram-size 3 --repetition-penalty 1.2 --bf16
```

#### 6.1.1 シャーディングで並列実行
```
python scripts/generate_stair_simplified.py \
  --model outputs/t5-simplify/final \
  --input-json STAIR-captions/stair_captions_v1.2_train.json \
  --output-json outputs/stair_simplified_train.part0.json \
  --num-shards 4 \
  --shard-id 0
```
- `--num-shards` と `--shard-id` で分割実行（各プロセスで別 `--output-json` を指定）

#### 6.2 より高精度な日本語LMで尤度スコアリング
```
python scripts/score_captions.py \
  --lm-model <lm-path-or-name> \
  --input-json outputs/stair_simplified_train.json \
  --output-json outputs/stair_simplified_train_scored.json
```
- 追加されるフィールド: `lm_total_logprob`, `lm_avg_logprob`, `lm_ppl`

#### 6.2.1 source/target の自然さ比較スコアリング
```
python scripts/score_fluency.py \
  --lm-model rinna/japanese-gpt-1b \
  --input-json outputs/stair_simplified_train.json \
  --output-json outputs/stair_simplified_train_scored.json \
  --summary-json outputs/stair_simplified_train_fluency_summary.json \
  --source-field source_caption \
  --target-field caption_replaced \
  --batch-size 8 \
  --max-length 128 \
```
- 追加されるフィールド: `lm_source_ppl`, `lm_target_ppl`, `lm_delta_ppl`
- ルートに `lm_fluency_scoring` を追加（source の percentile 閾値など）

#### 6.2.2 source/target の意味類似度スコアリング
```
python scripts/score_semantic_similarity.py \
  --model sonoisa/sentence-bert-base-ja-mean-tokens \
  --input-json outputs/stair_simplified_train.json \
  --output-json outputs/stair_simplified_train_scored.json \
  --summary-json outputs/stair_simplified_train_similarity_summary.json \
  --source-field source_caption \
  --target-field caption_replaced \
  --batch-size 16 \
  --max-length 256 \
  --device cpu
```
- 追加されるフィールド: `semantic_similarity` （必要なら `--threshold` で `semantic_below_threshold` も付与）
- ルートに `semantic_scoring` を追加（類似度の percentile 閾値など）

#### 6.3 TOKENS.txt制約の検証
```
python scripts/check_tokens.py \
  --input-json outputs/stair_simplified_train.json \
  --failures-output outputs/stair_simplified_train_failures.jsonl \
  --include-missing-tokens-in-failures \
  --tokenize fugashi \
  --mecabrc /etc/mecabrc \
  --mecab-dicdir /var/lib/mecab/dic/unidic \
  --missing-tokens-output outputs/stair_simplified_train_failures_missing_tokens.jsonl \
  --caption-field caption_replaced
```

```
python scripts/check_tokens.py \
  --input-json outputs/stair_simplified_val.json \
  --failures-output outputs/stair_simplified_val_failures.jsonl \
  --include-missing-tokens-in-failures \
  --tokenize fugashi \
  --mecabrc /etc/mecabrc \
  --mecab-dicdir /var/lib/mecab/dic/unidic \
  --missing-tokens-output outputs/stair_simplified_val_failures_missing_tokens.jsonl \
  --caption-field caption_replaced
```

- 失敗例はJSONLで保存（`index`, `id`, `image_id`, `caption`）
- `--missing-tokens-output` を付けると、未登録トークンの一覧（出現回数付き）をJSONで保存
- `--include-missing-tokens-in-failures` を付けると、失敗例に `missing_tokens` を追加
- `--tokenize fugashi` を付けると、MeCab (UniDicなど) の形態素の原形でTOKENSに一致するか判定（UniDicのグロスや用法メモを除去、読みがTOKENSにあれば許容、漢数字は算用数字に正規化）

#### 6.4 TOKENS.txt内語彙への置換
```
python scripts/replace_tokens.py \
  --input-json outputs/stair_simplified_train.json \
  --output-json outputs/stair_simplified_train_replaced.json \
  --replacements scripts/token_replacements.json \
  --output-field caption_replaced \
  --tokenize fugashi \
  --mecabrc /etc/mecabrc \
  --mecab-dicdir /var/lib/mecab/dic/unidic
```

#### 6.5 キャプションの手動レビュー用Web UI
```
python scripts/review_captions_web.py \
  --input-json stair/stair_simplified_train_replaced_snow.json \
  --merge-json stair/stair_simplified_train_replaced_snow_fluency.json \
  --merge-json stair/stair_simplified_train_replaced_snow_similarity.json \
  --failures-jsonl stair/stair_simplified_train_failures_snow.jsonl \
  --output-json stair/stair_simplified_train_replaced_snow_reviewed.json \
  --host 127.0.0.1 \
  --port 8000
```
- `--merge-json` で `lm_*` や `semantic_*` など追加フィールドを統合（`id` でマージ）
- `--failures-jsonl` に `missing_tokens` が含まれている場合はUIに表示
- ブラウザで `caption_replaced` を編集し、`Save JSON` で出力を保存

```
python scripts/replace_tokens.py \
  --input-json outputs/stair_simplified_val.json \
  --output-json outputs/stair_simplified_val_replaced.json \
  --replacements scripts/token_replacements.json \
  --output-field caption_replaced \
  --tokenize fugashi \
  --mecabrc /etc/mecabrc \
  --mecab-dicdir /var/lib/mecab/dic/unidic
```
- `scripts/token_replacements.json` の置換辞書を使って語彙をTOKENS内に寄せる

we need fugashi
```
apt update
apt install -y mecab libmecab-dev mecab-ipadic-utf8 unidic-mecab
```
