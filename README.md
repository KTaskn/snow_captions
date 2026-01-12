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

we need fugashi
```
apt update
apt install -y mecab libmecab-dev mecab-ipadic-utf8
```