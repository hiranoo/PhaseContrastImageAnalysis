# PhaseContrastImageAnalysis

非染色液滴の輪郭を抽出してトラックするプログラムです。

## 使い方

## 処理内容
1. Sensor dust mask
2. Roughly compute mass center of edge candidates
3. Bright to dark mask
- brightness gradient vector using Sobel filter
- the angle of brightness gradient vector and relative spatial vector from mass center as cosin
- Filter for large enough brightness gradient vector
- Filter for large enough cosin
4. Select edge points
- gaussianblur
- Laplacian
- remove sensor dust and neighbors
- remove small bg-vectors
- remove small cosins

