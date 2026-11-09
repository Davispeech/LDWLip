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

### 2.3 ICSLR

ICSLR 来源于实验室录制数据。标签目录：[labels/icslr](https://github.com/Davispeech/LDWLip/tree/main/labels/icslr)。

```text
labels/icslr/icslr_train.csv
labels/icslr/icslr_val_112.csv
labels/icslr/icslr_test.csv
```

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

### 4.1 VSR（Overlap）

Overlap 使用同一说话人集合中的常规训练、验证和测试划分。

训练：

```bash
python ldw_main.py vsr train data=cmlr.yaml model=Lipv1_aux.yaml root_dir=. model_name=vsr_overlap_cmlr weights=path/to/model.pth max_epochs=75
```

评估：

```bash
python ldw_main.py vsr eval data=cmlr.yaml model=Lipv1_aux.yaml root_dir=. model_name=vsr_overlap_cmlr test_data=labels/cmlr/cmlr_test.csv weights=output/train--vsr_overlap_cmlr/model_last.pth
```

预测：

```bash
python ldw_main.py vsr predict data=cmlr.yaml model=Lipv1_aux.yaml root_dir=. weights=output/train--vsr_overlap_cmlr/model_last.pth source=path/to/video.mp4
```

### 4.2 VSR（Unseen）

Unseen 使用按说话人隔离的划分。以下以 CVSRBILI 为例。

训练：

```bash
python ldw_main.py vsr train data=cvsrbili_unseen.yaml model=Lipv1_aux.yaml root_dir=. model_name=vsr_unseen_cvsrbili weights=path/to/model.pth max_epochs=75
```

评估：

```bash
python ldw_main.py vsr eval data=cvsrbili_unseen.yaml model=Lipv1_aux.yaml root_dir=. model_name=vsr_unseen_cvsrbili test_data=labels/cvsrbili/cvsrbili_unseen_test.csv weights=output/train--vsr_unseen_cvsrbili/model_last.pth
```

预测：

```bash
python ldw_main.py vsr predict data=cvsrbili_unseen.yaml model=Lipv1_aux.yaml root_dir=. weights=output/train--vsr_unseen_cvsrbili/model_last.pth source=path/to/video.mp4
```

### 4.3 LipAuth

LipAuth 使用 `lipauth` 任务入口。仓库当前未包含特定实验的 LipAuth 模型 YAML，请将下面的 `path/to/lipauth_model.yaml` 替换为实际配置文件，并将数据配置替换为对应的 LipAuth 数据配置。

训练：

```bash
python ldw_main.py lipauth train data=path/to/lipauth_data.yaml model=path/to/lipauth_model.yaml root_dir=. model_name=lipauth_baseline max_epochs=20 save_every_epoch=1
```

评估：

```bash
python ldw_main.py lipauth eval data=path/to/lipauth_data.yaml model=path/to/lipauth_model.yaml root_dir=. model_name=lipauth_baseline weights=output/train--lipauth_baseline/model_last.pth
```

预测/识别：

```bash
python ldw_main.py lipauth predict data=path/to/lipauth_data.yaml model=path/to/lipauth_model.yaml root_dir=. weights=output/train--lipauth_baseline/model_last.pth source=path/to/video.mp4
```

V-LoRA 和 LoRA-MoE 的 speaker adaptation 配置位于 [`ldw_cfg/model/PhD003`](https://github.com/Davispeech/LDWLip/tree/main/ldw_cfg/model/PhD003)，核心实现位于 [`ldw_nets/lora`](https://github.com/Davispeech/LDWLip/tree/main/ldw_nets/lora)。
