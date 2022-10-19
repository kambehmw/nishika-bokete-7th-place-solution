# 7th place solution

まずは、コンペを開催してくださった皆様&コンペに参加してくださった皆様ありがとうございました。

Nishikaのディスカッションの解法もご参考ください。
https://www.nishika.com/competitions/33/topics/395

## 学習方法

- `exp/bokete/`ディレクトリにコンペデータを配置する
- 以下のようなコマンドで学習を実行する

```bash
cd exp/ex{ID}
python ex{ID}.py
```

## 最終的なsubmissionファイルの作成方法

- ex55, ex59, ex61のensemble.ipynbをそれぞれ実行する

## Summary

最終的に暫定Best（Best Public）とBest CVのsubを選択しました。

|  | CV | 暫定 | 最終 |
| ---- | ---- | --- | --- |
| Best Public | 0.63243 | 0.635165 | **0.633456** |
| Best CV | 0.631605 | 0.635746 | 0.633468 |

解法の概要は以下です。

解法ではBest CVのsubを解説します。（Best PublicはBest CVと1つ前のBest Publicのサブを平均したものになっているため）

- 11個のモデルをアンサンブル
- Best CVに対して、nelder-meadでアンサンブルの重みを最適化
- 予測値を調整する後処理を適用

## Cross Validatioin

`is_laugh`のラベルに対して、StratifiedKFoldを使用し7foldsでCVを計算していました。

## Model & Weight

Tutorialで公開されていたnotebookを改良して使用していました。
notebookと同じMMBTのモデルがベースになっています。Tutorialのご共有ありがとうございました。
https://www.nishika.com/competitions/33/topics/355

学習済みモデルはtorchvision or Huggingfaceのものをそれぞれ表しています。

| No | Vision model | Text model | CV | ensemble weight |
| --- | ---- | ---- | --- | --- |
| 1. | ResNet152_Weights.IMAGENET1K_V2 | cl-tohoku/bert-base-japanese-v2 | 0.6426 | 0.156 |
| 2. | ResNet152_Weights.IMAGENET1K_V2 | cl-tohoku/bert-base-japanese-whole-word-masking | 0.6454 | 0.129 |
| 3. | Swin_B_Weights.IMAGENET1K_V1 | cl-tohoku/bert-base-japanese-v2 | 0.6407 | 0.127 | 
| 4. | ViT_B_16_Weights.IMAGENET1K_V1 | cl-tohoku/bert-base-japanese-v2 | 0.6514 | -0.109 |
| 5. | Swin_B_Weights.IMAGENET1K_V1 | sonoisa/sentence-bert-base-ja-mean-tokens-v2 | 0.6458 | 0.113 |
| 6. | EfficientNet_V2_M_Weights.IMAGENET1K_V1 | cl-tohoku/bert-base-japanese-v2 | 0.6439 | 0.144 |
| 7. | Swin_B_Weights.IMAGENET1K_V1 | cl-tohoku/bert-base-japanese-char-v2 | 0.6485 | 0.093 |
| 8. | openai/clip-vit-base-patch32 | cl-tohoku/bert-base-japanese-v2 | 0.6454 | 0.108 |
| 9. | sonoisa/clip-vit-b-32-japanese-v1 | sonoisa/sentence-bert-base-ja-mean-tokens-v2 | 0.6515 | 0.055 |
| 10. | Swin_B_Weights.IMAGENET1K_V1 | sonoisa/t5-base-japanese-v1.1 | 0.6574 | -0.126 |
| 11. | Swin_B_Weights.IMAGENET1K_V1 | cl-tohoku/bert-base-japanese-v2 | 0.6503 | 0.289 |

No.11のモデルについては、テストデータに対して擬似ラベリング(soft label)したものを学習データに含めて学習しました。

CV(ensemble): 0.632142

## Post process

アンサンブルの予測値をnelder-meadで求めた係数をかけることで調整しました。
閾値の間隔は適当に決定しています。

- preds < 0.05 → preds * 0.9858
- 0.05 <= preds < 0.1 → preds * 0.9817
- 0.1 <= preds < 0.15 → preds * 0.9805
- 0.15 <= preds < 0.2 → preds * 0.9894
- 0.2 <= preds < 0.25 → preds * 0.9386
- 0.25 <= preds < 0.3 → preds * 0.9151
- 0.3 <= preds < 0.35 → preds * 1.0362
- 0.35 <= preds < 0.4 → preds * 1.0127
- 0.4 <= preds < 0.45 → preds * 1.0132
- 0.45 <= preds < 0.5 → preds * 0.9979
- 0.5 <= preds < 0.55 → preds * 1.0238
- 0.55 <= preds < 0.6 → preds * 1.0171
- 0.6 <= preds < 0.65 → preds * 1.0121
- 0.65 <= preds < 0.7 → preds * 1.0047
- 0.7 <= preds < 0.75 → preds * 1.0076
- 0.75 <= preds < 0.8 → preds * 1.0087
- 0.8 <= preds < 0.85 → preds * 1.0059
- 0.85 <= preds < 0.9 → preds * 1.0429
- 0.9 <= preds < 0.95 → preds * 1.0716
- 0.95 <= preds < 1.0 → preds * 1.0124

CV(ensemble+post process): 0.632142 → 0.631605