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

CSV 标签下载：[CMLR 标签目录](https://github.com/Davispeech/LDWLip/tree/main/labels/cmlr)。

CMLR 官方数据主页：[https://vipazoo.cn/CMLR.html](https://vipazoo.cn/CMLR.html)；直接下载链接：[http://t.cn/A6waiog1](http://t.cn/A6waiog1)；提取码：`emqx`。该数据集仅限高校和科研机构用于研究。

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

CSV 下载链接：[CVSRBILI 标签目录](https://github.com/Davispeech/LDWLip/tree/main/labels/cvsrbili)；原始视频下载链接：[百度网盘](https://pan.baidu.com/s/1_aXb7OE166TB9q1EprYHMg?pwd=8keb)，提取码：`8keb`。

### 2.3 ICSLR

ICSLR 来源于实验室录制数据。标签目录：[labels/icslr](https://github.com/Davispeech/LDWLip/tree/main/labels/icslr)。

```text
labels/icslr/icslr_train.csv
labels/icslr/icslr_val_112.csv
labels/icslr/icslr_test.csv
```

CSV 标签下载：[ICSLR 标签目录](https://github.com/Davispeech/LDWLip/tree/main/labels/icslr)。

ICSLR 数据集论文及获取参考：[Variable Structure and Modeling Units for Chinese Lipreading](https://doi.org/10.1142/S0218001422560213)。

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

以下命令只展示参数名称和命令结构，参数值全部使用占位符。执行前请将尖括号中的内容替换为实际值。命令使用 Windows CMD 的 `^` 进行换行。

### 4.0 参数说明

- `data`：数据集 YAML 配置路径，替换 `<DATA_CONFIG>`。
- `model`：模型 YAML 配置路径，替换 `<MODEL_CONFIG>`。
- `root_dir`：项目根目录，替换 `<ROOT_DIR>`。
- `model_name`：实验名称，替换 `<MODEL_NAME>`。
- `train_data`：speaker adaptation 适应集 CSV，替换 `<TRAIN_DATA>`。
- `val_data`：训练过程验证集 CSV，替换 `<VAL_DATA>`。
- `test_data`：评估测试集 CSV，替换 `<TEST_DATA>`。
- `weights`：基础模型权重，替换 `<BASE_WEIGHTS>`。
- `lora_weights`：speaker adaptation 生成的 LoRA 权重，替换 `<LORA_WEIGHTS>`。
- `max_epochs`：训练轮数，替换 `<MAX_EPOCHS>`。
- `save_every_epoch`：保存间隔，替换 `<SAVE_INTERVAL>`。
- `lora`：LoRA 开关，替换 `<LORA_ENABLED>`。
- `source`：输入视频路径，替换 `<VIDEO_SOURCE>`。

### 4.1 VSR（Overlap）

训练：

```bat
python ldw_main.py vsr train ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  weights=<BASE_WEIGHTS> ^
  max_epochs=<MAX_EPOCHS>
```

评估：

```bat
python ldw_main.py vsr eval ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  test_data=<TEST_DATA> ^
  weights=<BASE_WEIGHTS>
```

指定视频预测：

```bat
python ldw_main.py vsr predict ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  weights=<BASE_WEIGHTS> ^
  source=<VIDEO_SOURCE>
```

### 4.2 VSR（Unseen）

训练：

```bat
python ldw_main.py vsr train ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  weights=<BASE_WEIGHTS> ^
  max_epochs=<MAX_EPOCHS>
```

评估：

```bat
python ldw_main.py vsr eval ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  test_data=<TEST_DATA> ^
  weights=<BASE_WEIGHTS>
```

指定视频预测：

```bat
python ldw_main.py vsr predict ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  weights=<BASE_WEIGHTS> ^
  source=<VIDEO_SOURCE>
```

#### Speaker adaptation（V-LoRA / LoRA-MoE）

训练：

```bat
python ldw_main.py vsr train ^
  data=<DATA_CONFIG> ^
  model=<ADAPTATION_MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  train_data=<TRAIN_DATA> ^
  val_data=<VAL_DATA> ^
  weights=<BASE_WEIGHTS> ^
  max_epochs=<MAX_EPOCHS> ^
  save_every_epoch=<SAVE_INTERVAL> ^
  lora=<LORA_ENABLED>
```

评估适配权重：

```bat
python ldw_main.py vsr eval ^
  data=<DATA_CONFIG> ^
  model=<ADAPTATION_MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  test_data=<TEST_DATA> ^
  weights=<BASE_WEIGHTS> ^
  lora_weights=<LORA_WEIGHTS> ^
  lora=<LORA_ENABLED>
```

### 4.3 LipAuth

训练：

```bat
python ldw_main.py lipauth train ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  max_epochs=<MAX_EPOCHS> ^
  save_every_epoch=<SAVE_INTERVAL>
```

评估：

```bat
python ldw_main.py lipauth eval ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  model_name=<MODEL_NAME> ^
  weights=<BASE_WEIGHTS>
```

预测/识别：

```bat
python ldw_main.py lipauth predict ^
  data=<DATA_CONFIG> ^
  model=<MODEL_CONFIG> ^
  root_dir=<ROOT_DIR> ^
  weights=<BASE_WEIGHTS> ^
  source=<VIDEO_SOURCE>
```

V-LoRA 和 LoRA-MoE 的配置位于 [`ldw_cfg/model/PhD003`](https://github.com/Davispeech/LDWLip/tree/main/ldw_cfg/model/PhD003)，核心实现位于 [`ldw_nets/lora`](https://github.com/Davispeech/LDWLip/tree/main/ldw_nets/lora)。
