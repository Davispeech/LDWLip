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

### 2. CNCVS 数据集

数据来源：网络爬取。

#### 常规训练与评估

```text
cncvs_train.csv
cncvs_test.csv
cncvs_val.csv
```

### 3. ICSLR 数据集

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




## （3）训练、评估和预测识别

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

