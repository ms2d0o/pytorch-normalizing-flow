# pytorch-normalizing-flow

- 以下を参考にpytorchでnormalizing flowを実装しました。
  - 間違っている箇所が多々あるかと思いますので気づいた点はissueやpull requestなどでお知らせくださいませ。
  - [Normalizing Flowの理論と実装](https://qiita.com/opeco17/items/62192b4dd1cd9cbaa170)
  - [深層生成モデルを巡る旅(1): Flowベース生成モデル](https://qiita.com/shionhonda/items/0fb7f91a150dff604cc5)
  - [e-hulten/planar-flows](https://github.com/e-hulten/planar-flows/blob/master/utils.py)

## requirements
---
torch >= 1.0.0<br>
numpy >= 1.15.2<br>
matplotlib >= 3.0.0