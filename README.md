# LDWLip

## 1. 环境配置

### 1.1 下载项目

```bash
git clone https://github.com/Davispeech/LDWLip.git
cd LDWLip
```

也可以直接下载：[GitHub ZIP](https://github.com/Davispeech/LDWLip/archive/refs/heads/main.zip)。

### 1.2 创建 Conda 环境

```bash
conda create -y -n LDWLip3 python=3.9
conda activate LDWLip3
```

### 1.3 安装 PyTorch 和依赖

根据 CUDA 版本选择 PyTorch。以下为 CUDA 11.8 示例：

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install sentencepiece av six opencv-python scikit-image tqdm thop tensorboard pyyaml tiktoken chardet decord soundfile einops tensorboardX matplotlib scikit-learn pypinyin
pip install timm openai-whisper transformers==4.30.0 accelerate pandas loralib
pip install "numpy<2"
python -m pip install pip==24.0
pip install fairseq
python -m pip install openmim
python -m mim install mmcv-full
```

### 1.4 预处理模型

人脸检测和对齐工具参考：[face_detection](https://github.com/hhj1897/face_detection)。

```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
```

## 2. 数据集准备

GitHub 中提供的是 CSV 标签文件；视频数据需要根据 CSV 中的路径自行准备，并在数据配置中设置 `data_path`。

### 2.1 CMLR

CMLR 来源于《新闻联播》视频。标签目录：[labels/cmlr](https://github.com/Davispeech/LDWLip/tree/main/labels/cmlr)。

常规划分：

```text
labels/cmlr/cmlr_train.csv
labels/cmlr/cmlr_val.csv
labels/cmlr/cmlr_test.csv
```

Unseen speaker 划分：

```text
labels/cmlr/cmlr_unseen_train.csv
labels/cmlr/cmlr_unseen_test.csv
```

数据配置：[cmlr.yaml](https://github.com/Davispeech/LDWLip/blob/main/ldw_cfg/dataset/cmlr.yaml)、[cmlr_unseen.yaml](https://github.com/Davispeech/LDWLip/blob/main/ldw_cfg/dataset/cmlr_unseen.yaml)。

适应集目录：

```text
labels/cmlr/cmlr_unseen_adapt_s2_1.csv
labels/cmlr/cmlr_unseen_adapt_s2_3.csv
labels/cmlr/cmlr_unseen_adapt_s2_5.csv
labels/cmlr/cmlr_unseen_adapt_s6_1.csv
labels/cmlr/cmlr_unseen_adapt_s6_3.csv
labels/cmlr/cmlr_unseen_adapt_s6_5.csv
```

CSV 下载链接：[CMLR 标签目录](https://github.com/Davispeech/LDWLip/tree/main/labels/cmlr)；原始视频下载链接：`to be continue`。

### 2.2 CVSRBILI

CVSRBILI 是本项目使用的中文唇读数据集。标签目录：[labels/cvsrbili](https://github.com/Davispeech/LDWLip/tree/main/labels/cvsrbili)。

常规划分包含 57 名说话人：

```text
labels/cvsrbili/cvsrbili_train.csv
labels/cvsrbili/cvsrbili_val.csv
labels/cvsrbili/cvsrbili_test.csv
```

Unseen speaker 划分：

```text
labels/cvsrbili/cvsrbili_unseen_train.csv
labels/cvsrbili/cvsrbili_unseen_val.csv
labels/cvsrbili/cvsrbili_unseen_test.csv
```

其中训练集包含 41 名说话人，验证集和测试集各包含 8 名未见说话人。测试说话人为 `b001`、`b017`、`b029`、`b032`、`b037`、`b040`、`b042` 和 `b052`。

1/3/5 分钟 speaker adaptation 标签：

```text
labels/cvsrbili/cvsrbili_unseen_adapt_<speaker>_<minutes>.csv
```

例如：`cvsrbili_unseen_adapt_b001_1.csv`。数据配置：[cvsrbili_unseen.yaml](https://github.com/Davispeech/LDWLip/blob/main/ldw_cfg/dataset/cvsrbili_unseen.yaml)。

适应集目录：

```text
labels/cvsrbili/cvsrbili_unseen_adapt_b001_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b001_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b001_5.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b017_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b017_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b017_5.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b029_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b029_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b029_5.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b032_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b032_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b032_5.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b037_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b037_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b037_5.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b040_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b040_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b040_5.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b042_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b042_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b042_5.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b052_1.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b052_3.csv
labels/cvsrbili/cvsrbili_unseen_adapt_b052_5.csv
```

CSV 下载链接：[CVSRBILI 标签目录](https://github.com/Davispeech/LDWLip/tree/main/labels/cvsrbili)；原始视频下载链接：`to be continue`。

### 2.3 ICSLR

ICSLR 来源于实验室录制数据。标签目录：[labels/icslr](https://github.com/Davispeech/LDWLip/tree/main/labels/icslr)。

```text
labels/icslr/icslr_train.csv
labels/icslr/icslr_val_112.csv
labels/icslr/icslr_test.csv
```

CSV 下载链接：[ICSLR 标签目录](https://github.com/Davispeech/LDWLip/tree/main/labels/icslr)；原始视频下载链接：`to be continue`。

## 3. 快速启动

以下命令使用仓库根目录作为 `root_dir`，并假设基础模型权重位于 `path/to/model.pth`。

### 3.1 基本训练

```bash
python ldw_main.py vsr train data=cmlr.yaml model=Lipv1_aux.yaml root_dir=. model_name=baseline_cmlr weights=path/to/model.pth max_epochs=75
```

### 3.2 模型评估

```bash
python ldw_main.py vsr eval data=cmlr.yaml model=Lipv1_aux.yaml root_dir=. model_name=baseline_cmlr test_data=labels/cmlr/cmlr_test.csv weights=path/to/model.pth
```

### 3.3 指定视频预测

```bash
python ldw_main.py vsr predict data=cmlr.yaml model=Lipv1_aux.yaml root_dir=. weights=path/to/model.pth source=path/to/video.mp4
```

## 4. 训练、评估与预测识别

以下命令使用 Windows CMD 的 `^` 进行换行。每个参数单独列出，执行前请替换占位路径。

### 4.0 参数说明

- `vsr` / `lipauth`：任务类型，分别对应视觉语音识别和唇部身份识别。
- `train` / `eval` / `predict`：训练、评估和指定视频预测子命令。
- `data`：数据集 YAML 配置路径。
- `model`：模型 YAML 配置路径。
- `root_dir`：项目根目录，从项目根目录执行时使用 `.`。
- `model_name`：实验名称，决定输出目录名。
- `train_data`：speaker adaptation 使用的适应集 CSV。
- `val_data`：训练过程中的验证集 CSV。
- `test_data`：评估时使用的测试集 CSV。
- `weights`：基础模型权重路径。
- `lora_weights`：speaker adaptation 训练生成的 LoRA 权重路径。
- `max_epochs`：最大训练轮数。
- `save_every_epoch`：每隔多少轮保存一次模型。
- `lora=True`：启用 LoRA 权重注入。
- `source`：预测时输入的视频文件路径。

### 4.1 VSR（Overlap）

Overlap 使用同一说话人集合中的常规训练、验证和测试划分。

训练：

```bat
python ldw_main.py vsr train ^
  data=cmlr.yaml ^
  model=Lipv1_aux.yaml ^
  root_dir=. ^
  model_name=vsr_overlap_cmlr ^
  weights=path/to/model.pth ^
  max_epochs=75
```

评估：

```bat
python ldw_main.py vsr eval ^
  data=cmlr.yaml ^
  model=Lipv1_aux.yaml ^
  root_dir=. ^
  model_name=vsr_overlap_cmlr ^
  test_data=labels/cmlr/cmlr_test.csv ^
  weights=output/train--vsr_overlap_cmlr/model_last.pth
```

指定视频预测：

```bat
python ldw_main.py vsr predict ^
  data=cmlr.yaml ^
  model=Lipv1_aux.yaml ^
  root_dir=. ^
  weights=output/train--vsr_overlap_cmlr/model_last.pth ^
  source=path/to/video.mp4
```

### 4.2 VSR（Unseen）

Unseen 使用按说话人隔离的划分。以下以 CVSRBILI 为例。

训练：

```bat
python ldw_main.py vsr train ^
  data=cvsrbili_unseen.yaml ^
  model=Lipv1_aux.yaml ^
  root_dir=. ^
  model_name=vsr_unseen_cvsrbili ^
  weights=path/to/model.pth ^
  max_epochs=75
```

评估：

```bat
python ldw_main.py vsr eval ^
  data=cvsrbili_unseen.yaml ^
  model=Lipv1_aux.yaml ^
  root_dir=. ^
  model_name=vsr_unseen_cvsrbili ^
  test_data=labels/cvsrbili/cvsrbili_unseen_test.csv ^
  weights=output/train--vsr_unseen_cvsrbili/model_last.pth
```

指定视频预测：

```bat
python ldw_main.py vsr predict ^
  data=cvsrbili_unseen.yaml ^
  model=Lipv1_aux.yaml ^
  root_dir=. ^
  weights=output/train--vsr_unseen_cvsrbili/model_last.pth ^
  source=path/to/video.mp4
```

#### Speaker adaptation（V-LoRA / LoRA-MoE）

以 `b001` 的 1 分钟适应集为例：

```bat
python ldw_main.py vsr train ^
  data=cvsrbili_unseen.yaml ^
  model=PhD003/AdaptVSR3LoRA_NoAug_cvsrbili.yaml ^
  root_dir=. ^
  model_name=adapt_b001_1 ^
  train_data=labels/cvsrbili/cvsrbili_unseen_adapt_b001_1.csv ^
  val_data=labels/cvsrbili/cvsrbili_unseen_val.csv ^
  weights=path/to/cvsrbili_backbone.pth ^
  max_epochs=6 ^
  save_every_epoch=5 ^
  lora=True
```

评估适配权重：

```bat
python ldw_main.py vsr eval ^
  data=cvsrbili_unseen.yaml ^
  model=PhD003/AdaptVSR3LoRA_NoAug_cvsrbili.yaml ^
  root_dir=. ^
  model_name=adapt_b001_1 ^
  test_data=labels/cvsrbili/cvsrbili_unseen_test.csv ^
  weights=path/to/cvsrbili_backbone.pth ^
  lora_weights=output/train--adapt_b001_1/model_5.pth ^
  lora=True
```

将 `b001` 替换为其他 unseen speaker，将 `1` 替换为 `3` 或 `5`，即可运行其他适应实验。

### 4.3 LipAuth

LipAuth 使用 `lipauth` 任务入口。仓库当前未包含特定实验的 LipAuth 模型 YAML，请将 `path/to/lipauth_*.yaml` 替换为实际配置文件。

训练：

```bat
python ldw_main.py lipauth train ^
  data=path/to/lipauth_data.yaml ^
  model=path/to/lipauth_model.yaml ^
  root_dir=. ^
  model_name=lipauth_baseline ^
  max_epochs=20 ^
  save_every_epoch=1
```

评估：

```bat
python ldw_main.py lipauth eval ^
  data=path/to/lipauth_data.yaml ^
  model=path/to/lipauth_model.yaml ^
  root_dir=. ^
  model_name=lipauth_baseline ^
  weights=output/train--lipauth_baseline/model_last.pth
```

预测/识别：

```bat
python ldw_main.py lipauth predict ^
  data=path/to/lipauth_data.yaml ^
  model=path/to/lipauth_model.yaml ^
  root_dir=. ^
  weights=output/train--lipauth_baseline/model_last.pth ^
  source=path/to/video.mp4
```

V-LoRA 和 LoRA-MoE 的 speaker adaptation 配置位于 [`ldw_cfg/model/PhD003`](https://github.com/Davispeech/LDWLip/tree/main/ldw_cfg/model/PhD003)，核心实现位于 [`ldw_nets/lora`](https://github.com/Davispeech/LDWLip/tree/main/ldw_nets/lora)。
