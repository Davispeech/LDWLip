# 1.3 快速启动

## （1）环境配置

### 1. 下载项目

项目文件：

`LDWLip.tar.gz`（414.40 MB）

### 2. 创建 Conda 环境

```bash
conda create -y -n DWLip python=3.9
conda activate DWLip
```

### 3. 安装 PyTorch

根据实际运行环境选择对应版本。

#### PyTorch 2.0.1 + CUDA 11.7

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### PyTorch 2.1.0 + CUDA 11.8

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. 安装依赖库

```bash
pip install sentencepiece av six opencv-python scikit-image tqdm thop tensorboard pyyaml tiktoken chardet decord soundfile einops tensorboardX matplotlib scikit-learn pypinyin
pip install timm openai-whisper
pip install "numpy<2" --force-reinstall
pip install transformers==4.30.0
pip install accelerate pandas loralib
```

### 5. 安装 fairseq

注意：最新版 `pip` 与 `fairseq` 可能存在兼容性问题。建议将 `pip` 降级至 24.0 后再安装 `fairseq`。

```bash
python.exe -m pip install pip==24.0
pip install fairseq
```

### 6. 安装 mmcv

```bash
python -m pip install openmim
python -m mim install mmcv-full
```

### 7. 安装预处理模型

预处理模型用于人脸检测与人脸对齐，不影响模型训练和评估。

相关工具参考：

https://github.com/hhj1897/

#### 人脸检测

```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
```

#### 人脸对齐

```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
```

> 注意：`git lfs pull` 可以不执行。

## （2）数据准备

### 1. CMLR 数据集

数据来源：爬取《新闻联播》。

#### 常规训练与评估

```text
cmlr_train.csv
cmlr_test.csv
cmlr_val.csv
```


#### Unseen 训练与评估

```text
cmlr_train_person.csv
extraSet.csv
```

其中：

- `cmlr_train_person.csv` 包含 11 人；
- `extraSet.csv` 包含 5 人。

### 2. CVSRBILI 数据集

`cvsrbili` 是本项目整理的中文唇读数据集，标签文件位于
`labels/cvsrbili/`。

#### 常规训练与评估

```text
labels/cvsrbili/cvsrbili_train.csv
labels/cvsrbili/cvsrbili_val.csv
labels/cvsrbili/cvsrbili_test.csv
```

常规划分包含 57 名说话人：

- 训练集：213,051 条记录；
- 验证集：30,454 条记录；
- 测试集：60,869 条记录。

#### Unseen speaker 训练与测试

Unseen speaker 划分按说话人隔离，训练、验证和测试说话人互不重叠：

```text
labels/cvsrbili/cvsrbili_unseen_train.csv
labels/cvsrbili/cvsrbili_unseen_val.csv
labels/cvsrbili/cvsrbili_unseen_test.csv
```

- `cvsrbili_unseen_train.csv`：41 名说话人，229,196 条记录；
- `cvsrbili_unseen_val.csv`：8 名说话人，40,909 条记录；
- `cvsrbili_unseen_test.csv`：8 名说话人，17,381 条记录。

Unseen speaker 测试集由 8 名未出现在训练集中的说话人组成：
`b001`、`b017`、`b029`、`b032`、`b037`、`b040`、`b042` 和 `b052`。

## （3）Speaker adaptation 训练与评估

以下命令以 CVSRBILI 的 `b001` 说话人、1 分钟适应集为例。将
`weights` 替换为已经训练好的基础模型权重。

### 1. V-LoRA + speaker identity loss

该配置使用视觉卷积 LoRA（V-LoRA），不启用 LoRA-MoE：

```bash
python ldw_main.py vsr train ^
  data=cvsrbili_unseen.yaml ^
  model=PhD003/AdaptVSR3LoRA_VEid_cvsrbili.yaml ^
  root_dir=. ^
  model_name=AdaptVSR3LoRA_VEidCE_b001_1_cvsrbili ^
  train_data=labels/cvsrbili/cvsrbili_unseen_adapt_b001_1.csv ^
  val_data=labels/cvsrbili/cvsrbili_unseen_val.csv ^
  max_epochs=7 ^
  weights=path/to/cvsrbili_backbone.pth ^
  warmup_epochs=0 ^
  lr=0.01 ^
  save_every_epoch=1 ^
  lora=True
```

评估第 6 个 epoch 的适配权重：

```bash
python ldw_main.py vsr eval ^
  data=cvsrbili_unseen.yaml ^
  model=PhD003/AdaptVSR3LoRA_VEid_cvsrbili.yaml ^
  root_dir=. ^
  model_name=AdaptVSR3LoRA_VEidCE_b001_1_ep6_cvsrbili ^
  test_data=labels/cvsrbili/cvsrbili_unseen_test.csv ^
  weights=path/to/cvsrbili_backbone.pth ^
  lora_weights=output/train--AdaptVSR3LoRA_VEidCE_b001_1_cvsrbili/model_6.pth ^
  lora=True
```

### 2. LoRA-MoE（本文方法，w/o augmentation）

该配置在 V-LoRA 基础上加入 `LoRAMoEPaper`。训练 6 个 epoch，
第 5 个 epoch 的权重用于最终评估：

```bash
python ldw_main.py vsr train ^
  data=cvsrbili_unseen.yaml ^
  model=PhD003/AdaptVSR3LoRA_NoAug_cvsrbili.yaml ^
  root_dir=. ^
  model_name=AdaptVSR3LoRA_NoAug_b001_1_cvsrbili ^
  train_data=labels/cvsrbili/cvsrbili_unseen_adapt_b001_1.csv ^
  val_data=labels/cvsrbili/cvsrbili_unseen_val.csv ^
  max_epochs=6 ^
  weights=path/to/cvsrbili_backbone.pth ^
  warmup_epochs=0 ^
  lr=0.01 ^
  save_every_epoch=5 ^
  lora=True
```

```bash
python ldw_main.py vsr eval ^
  data=cvsrbili_unseen.yaml ^
  model=PhD003/AdaptVSR3LoRA_NoAug_cvsrbili.yaml ^
  root_dir=. ^
  model_name=AdaptVSR3LoRA_NoAug_b001_1_cvsrbili ^
  test_data=labels/cvsrbili/cvsrbili_unseen_test.csv ^
  weights=path/to/cvsrbili_backbone.pth ^
  lora_weights=output/train--AdaptVSR3LoRA_NoAug_b001_1_cvsrbili/model_5.pth ^
  lora=True
```

将 `b001` 替换为其他 unseen speaker（`b017`、`b029`、`b032`、
`b037`、`b040`、`b042`、`b052`），将 `1` 替换为 `3` 或 `5`，
即可运行对应的 3/5 分钟适应实验。批量运行也可以使用
`run_vlora_noaug_bili.bat` 或 `run_vlora_idce_cmlrbili.bat`。

### 3. CNCVS 数据集

数据来源：网络爬取。

#### 常规训练与评估

```text
cncvs_train.csv
cncvs_test.csv
cncvs_val.csv
```

### 4. ICSLR 数据集

数据来源：实验室录制。

#### 常规训练与评估

```text
icslr_train.csv
icslr_test.csv
```

#### Unseen 训练与评估

```text
icslr_train_person.csv
icslr_test_person.csv
```

其中：

- `icslr_train_person.csv` 包含 25 人；
- `icslr_test_person.csv` 包含 2 人。




## （4）训练、评估和预测识别

### 1. 基本训练

```bash
python main.py vsr train data=cmlr.yaml model=Lipv1.yaml weights= run_exp/debug/model_last.pth epochs=75
```

### 2. 模型评估

评估模型，并将结果保存至：

`output/Lipv1_cer8.0/`

```bash
python main.py vsr eval data=cmlr.yaml model=Lipv1.yaml weights= run_exp/debug/model_last.pth model_name=Lipv1_cer8.0
```

### 3. 指定视频预测

使用训练好的模型对指定视频进行预测：

```bash
python main.py vsr predict data=cmlr.yaml model=Lipv1.yaml weights= run_exp/debug/model_last.pth source=./data/ldw.mp4
```

# 1.4 验证环境配置是否成功

## 1. 训练

### VSR 模型训练

```bash
python ldw_main.py vsr train data=icslrAuth.yaml model=Pro_001_Displays/VSR/LipTMP_VSR_icslr.yaml root_dir=E:/Sproject/DaviLip/LDWLip weights=run_exp/export/Encoder_lrw_snv05x_tcn1x.pth model_name=LipTMP_VSR_icslr_pre max_epochs=5 save_every_epoch=1
```

### LipAuth 模型训练

```bash
python ldw_main.py lipauth train data=icslrAuth.yaml model=Pro_001_Displays/icslr/Static_ResNet18_icslr.yaml root_dir=E:/Sproject/DaviLip/LDWLip model_name=Static_ResNet18_icslr max_epochs=20 save_every_epoch=1
```

## 2. 评估

### VSR 模型评估

```bash
python ldw_main.py vsr eval data=icslrAuth.yaml model=Pro_001_Displays/VSR/LipTMP_VSR_icslr.yaml root_dir=E:/Sproject/DaviLip/LDWLip weights=run_exp/LipTMP_VSR_icslr_pre/model_last.pth
```

