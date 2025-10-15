# Wave-Mamba-KD: Knowledge Distillation for Image Restoration Tasks

<p align="center">
    <img src="Figures/logo.png" width="200">
</p>

This is a modified implementation based on ***Wave-Mamba: Wavelet State Space Model for Ultra-High-Definition Low-Light Image Enhancement, ACMMM2024.***

## 📝 Overview

本项目基于 [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba) 开源代码进行修改和扩展，采用知识蒸馏技术，将模型应用于多个图像恢复任务：

- ✨ **图像去雨** (Image Deraining)
- 🌫️ **图像去雾** (Image Dehazing)
- 🔦 **低光增强** (Low-Light Enhancement)
- 🏃 **运动去模糊** (Motion Deblurring)

### 主要改进

- 🎓 引入知识蒸馏框架，提升模型性能
- 🔧 适配多种图像恢复任务
- 📊 优化训练和推理流程
- 🚀 支持 UHD 超高清图像处理

<hr />

## 🔗 原始项目

本项目基于以下工作：

**Wave-Mamba: Wavelet State Space Model for Ultra-High-Definition Low-Light Image Enhancement**

<a href="https://alexzou14.github.io">Wenbin Zou*,</a> Hongxia Gao <sup>✉️</sup>, Weipeng Yang, and Tongtong Liu

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2408.01276)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/AlexZou14/Wave-Mamba)

<hr />

## 📋 TODO List

- [x] 知识蒸馏框架实现
- [x] 图像去雨任务适配
- [x] 图像去雾任务适配
- [x] 低光增强任务适配
- [x] 运动去模糊任务适配
- [x] 测试代码和预训练模型
- [ ] 多任务联合训练
- [ ] 更多实验结果和对比

## 🛠️ Dependencies and Installation

### 环境要求

- Ubuntu >= 22.04
- CUDA >= 11.8
- Pytorch >= 2.0.1
- Python >= 3.8

### 安装步骤

```bash
cd Wave-Mamba-kd

# 创建 conda 环境
conda create -n wavemamba_kd python=3.8
conda activate wavemamba_kd

# 安装依赖
pip3 install -r requirements.txt
python setup.py develop
```

## 📦 Datasets Download

### 去雨数据集
- [Rain100L/H](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
- [Rain13K](https://github.com/megvii-research/MIMO-UNet)

### 去雾数据集
- [RESIDE](https://sites.google.com/view/reside-dehaze-datasets)
- [Dense-Haze](https://github.com/hendrycks/robustness)

### 低光增强数据集
- [LOL](https://daooshee.github.io/BMVC2018website/)
- [UHD-LL](https://li-chongyi.github.io/UHDFour/)
- [UHDLOL4K](https://taowangzj.github.io/projects/LLFormer)

### 运动去模糊数据集
- [GoPro](https://seungjunnah.github.io/Datasets/gopro)
- [HIDE](https://github.com/joanshen0508/HA_deblur)

## 🎯 Pre-trained Models

预训练模型下载链接：

| 任务 | 数据集 | PSNR | SSIM | 下载链接 |
|------|--------|------|------|----------|
| 去雨 | Rain100L | TBD | TBD | [Google Drive](#) |
| 去雾 | RESIDE | TBD | TBD | [Google Drive](#) |
| 低光增强 | LOL | TBD | TBD | [Google Drive](#) |
| 运动去模糊 | GoPro | TBD | TBD | [Google Drive](#) |

## 🚀 Quick Inference

### 使用脚本推理

```bash
bash test.sh
```

### 使用 Python 命令

```bash
# 图像去雨
python inference_wavemamba.py -i input_path -g gt_path -w weight_path -o output_path

# 指定任务类型
python inference_wavemamba.py -i input_path -w weight_path -o output_path --task deraining
```

### 参数说明

- `-i, --input`: 输入图像或文件夹路径
- `-g, --gt`: Ground truth 图像路径（用于计算指标）
- `-w, --weight`: 模型权重路径
- `-o, --output`: 输出文件夹路径
- `-s, --out_scale`: 最终上采样比例（默认: 1）
- `--max_size`: 全图推理的最大图像尺寸（默认: 600）
- `--task`: 任务类型 (deraining/dehazing/enhancement/deblurring)

## 🎓 Knowledge Distillation

本项目采用知识蒸馏技术，包括：

- **Teacher-Student 架构**: 使用大模型作为教师网络，指导小模型学习
- **特征蒸馏**: 在特征层面进行知识传递
- **响应蒸馏**: 在输出层面进行知识传递
- **多尺度蒸馏**: 结合小波变换的多尺度特性

### 蒸馏损失

```python
L_total = L_task + λ_feat * L_feat + λ_resp * L_resp
```

## 🏋️ Train the Model

### 使用脚本训练

```bash
bash train.sh
```

### 使用命令行训练

```bash
# 单 GPU 训练
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train_wavemamba_deraining.yml

# 多 GPU 分布式训练
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=4324 \
    basicsr/train.py \
    -opt options/train_wavemamba_deraining.yml \
    --launcher pytorch
```

### 配置文件

训练配置文件位于 `options/` 目录：
- `train_wavemamba_deraining.yml` - 去雨任务
- `train_wavemamba_dehazing.yml` - 去雾任务
- `train_wavemamba_enhancement.yml` - 低光增强任务
- `train_wavemamba_deblurring.yml` - 运动去模糊任务

## 📊 Main Results

### 定量结果

| 任务 | 数据集 | PSNR ↑ | SSIM ↑ | 参数量 (M) | FLOPs (G) |
|------|--------|--------|--------|-----------|-----------|
| 去雨 | Rain100L | TBD | TBD | TBD | TBD |
| 去雾 | RESIDE | TBD | TBD | TBD | TBD |
| 低光增强 | LOL | TBD | TBD | TBD | TBD |
| 运动去模糊 | GoPro | TBD | TBD | TBD | TBD |

### 可视化结果

![results](Figures/results.png)

## 🗂️ Project Structure

```
Wave-Mamba-kd/
├── basicsr/                 # 基础训练框架
│   ├── archs/              # 模型架构
│   ├── data/               # 数据加载器
│   ├── models/             # 训练模型
│   └── train.py            # 训练脚本
├── options/                # 配置文件
├── inference_wavemamba.py  # 推理脚本
├── requirements.txt        # 依赖列表
└── README.md              # 说明文档
```

## 📖 Citation

如果本项目对您的研究有帮助，请引用：

```bibtex
@inproceedings{zou2024wavemamba,
  title={Wave-Mamba: Wavelet State Space Model for Ultra-High-Definition Low-Light Image Enhancement},
  author={Wenbin Zou and Hongxia Gao and Weipeng Yang and Tongtong Liu},
  booktitle={ACM Multimedia 2024},
  year={2024},
  url={https://openreview.net/forum?id=oQahsz6vWe}
}
```

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 🙏 Acknowledgement

本项目基于以下开源项目：
- [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba) - 原始实现
- [BasicSR](https://github.com/xinntao/BasicSR) - 训练框架

感谢原作者的杰出工作！

## 📧 Contact

如有问题，请提交 Issue 或联系：[your-email@example.com]

---

⭐ 如果本项目对您有帮助，欢迎 Star 支持！