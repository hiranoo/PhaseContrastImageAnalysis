# PhaseContrastImageAnalysis

## 非染色液滴の円近似
- 画像の局所平均の分布がフラットではない場合、閾値による界面の検出が困難である。本プログラムは局所勾配から界面を検出するので局所平均の分布にむらがあっても界面を検出可能である。
- 検出した界面を円で近似する。

## 部分的な界面の曲率計算
液滴の変形を分析可能

## 依存関係
- 使用ライブラリ： `numpy cv2 skimage matplotlib numba jupyterlab`
- 動作確認環境：Ubuntu 22.04.2 LTS on WSL2, Python 3.10.12

## 使い方
- main.ipynbを雛形として用意している。
- `test_run 関数`で処理過程の画像を確認しながらパラメータを調整する。
- 画像はmultittiff形式にのみ対応している。他の画像形式を読み込みたい場合はprocess.pyを変更しておく。



