# Ichimatsu pipeline

Implemented an image transformation algorithm with less result variation caused by noise compared to the i2i algorithm in Diffusers.

# English (English Translation by ChatGPT)

## Recommended Environment
Python 3.10.11

A GPU is required for generating results within a realistic timeframe.

Install CUDA Toolkit and cuDNN.

## How to Use
After setting up the recommended environment, run setup.bat to configure the necessary settings.

Once completed, use img2img.ipynb or mov2mov.ipynb according to your purpose.

### Example Usage of img2img.ipynb
Place a pre-trained model of Stable Diffusion in the model directory and specify its path in the ```pretrained_model_name_or_path``` field.

Add the input images to the input directory and specify the path to the desired image in the ```input_path``` field.
If using the Ichimatsu pipeline, enable the ```use_ichimatsu_pipeline``` option.

After executing all the steps, the results will be saved in the result directory.

### Version History
ver1.0.0
Implemented an image transformation algorithm with less result variation caused by noise compared to the i2i algorithm in Diffusers.

# 日本語

## 推奨環境
Python 3.10.11

現実的な時間で生成するためにGPUが必要

cuda toolkit, cudnnのインストール

## 使用方法
推奨環境を整えた後、setup.batを実行し、環境構築を構築します

完了後、img2img.ipynb、mov2mov.ipynbを目的に合わせ使用してください

### img2img.ipynb使用例
modelディレクトリにStable Diffusionの学習済みモデルを入れ、```pretrained_model_name_or_path```にパスを指定してください。

inputディレクトリに入力に使用する画像を入れ、```input_path```でその画像のパスを指定してください。

Ichimatsu pipelineを使用する場合は、```use_ichimatsu_pipeline```をtrueにしてください。

すべて実行した後、resultディレクトリに生成結果が保存されます。

### バージョン履歴
ver1.0.0 ノイズによる結果の変動がdiffusersのi2iより小さい画像変換アルゴリズムの実装
