# PhaseContrastImageAnalysis

## 非染色液滴の円近似
- 画像の局所平均の分布がフラットではない場合、閾値による界面の検出が困難である。本プログラムは局所勾配から界面を検出するので局所平均の分布にむらがあっても界面を検出可能である。
- 検出した界面を円で近似する。

## 部分的な界面の曲率計算
液滴の変形を分析可能。

## 制約
- 同時に複数の液滴を円近似することはできない。液滴が1個のみ写るように画像を切り取ること。
- 画像はmultittiff形式にのみ対応している。他の画像形式を読み込みたい場合はprocess.pyを変更しておく。

## 依存関係
- 使用ライブラリ： `numpy cv2 skimage matplotlib numba jupyterlab`
- 動作確認環境：Ubuntu 22.04.2 LTS on WSL2, Python 3.10.12

## 使い方
- template directoryを同じ階層にコピーして使う。
1. Modify conf.json. Make sure to overwrite path of images.
2. Check and set a crop area
3. Run main files. Interface of the droplet can be detected by Otsu threshold or My bright2dark detection.

- main_b2d.ipynbでは、`test_run 関数`で処理過程の画像を確認しながらパラメータを調整する。



