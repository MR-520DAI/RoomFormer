# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目规则
- 每次启动时必须检索项目内的.claude目录，重点关注agents、skills文件夹下的文件，了解存在哪些子代理，哪些技能。

## 项目概览

**RoomFormer** 是基于 PyTorch 的深度学习项目，用于从 3D 扫描重建 2D 楼层平面图，发表于 CVPR 2023。它使用新颖的 Transformer 架构和两级查询，在单阶段中直接预测可变大小的房间多边形集合。

- **任务**：从 3D 点云重建楼层平面图 → 输出房间多边形角点坐标、语义房间类型、门和窗
- **架构**：基于 Transformer（受 Deformable DETR 启发），使用两级查询处理房间和角点
- **论文**：https://arxiv.org/abs/2211.15658
- **项目页面**：https://ywyue.github.io/RoomFormer/

## 环境配置

要求：Python 3.8、PyTorch 1.9.0、CUDA 11.1

```bash
# 创建 conda 环境
conda create -n roomformer python=3.8
conda activate roomformer

# 安装 PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 安装依赖
pip install -r requirements.txt

# 编译可变形注意力模块
cd models/ops
sh make.sh

# 编译可微光栅化模块
cd ../../diff_ras
python setup.py build develop
```

## 常用命令

### 训练

```bash
# 在 Structured3D 上训练
./tools/train_stru3d.sh

# 在 SceneCAD 上训练
./tools/train_scenecad.sh

# 训练语义丰富的楼层平面图（房间 + 门 + 窗）
./tools/train_stru3d_sem_rich.sh
```

### 评估

```bash
# 在 Structured3D 测试集上评估
./tools/eval_stru3d.sh

# 在 Structured3D 上评估紧凑布局模型
./tools/eval_stru3d_tight.sh

# 在 SceneCAD 上评估
./tools/eval_scenecad.sh

# 评估语义丰富的楼层平面图
./tools/eval_stru3d_sem_rich.sh
```

### 关键参数

- `--num_queries`：角点查询总数（默认 800）
- `--num_polys`：最大房间数量（默认 20）
- `--semantic_classes`：-1 = 无语义，19 = 有语义（16 种房间类型 + 门 + 窗 + 空）
- `--backbone`：默认 resnet50
- `--enc_layers`、`--dec_layers`：默认各 6 层

## 代码架构

### 目录结构

```
RoomFormer/
├── main.py                 # 训练入口
├── eval.py                 # 评估/推理入口
├── engine.py               # 训练循环和评估逻辑
├── models/
│   ├── roomformer.py       # RoomFormer 核心模型类
│   ├── deformable_transformer.py  # 可变形 Transformer
│   ├── backbone.py         # ResNet 特征提取
│   ├── matcher.py          # 训练用多边形匹配
│   ├── losses.py           # 损失函数（L1、光栅化）
│   └── ops/                # 编译的 CUDA 可变形注意力
├── datasets/
│   └── poly_data.py        # COCO 格式数据集加载
├── data_preprocess/        # 数据预处理脚本
├── diff_ras/               # 可微光栅化 CUDA 扩展
├── util/                   # 工具（多边形操作、可视化）
├── tools/                  # 训练/评估 shell 脚本
├── checkpoints/            # 预训练模型检查点
└── detectron2/             # 数据处理用 Detectron2 子模块
```

### 关键组件

| 文件 | 用途 |
|------|------|
| `main.py` | 训练入口，参数解析 |
| `eval.py` | 评估/推理入口 |
| `engine.py` | 训练循环、评估、后处理 |
| `models/roomformer.py` | RoomFormer 核心 Transformer 模型 |
| `models/deformable_transformer.py` | 可变形 Transformer 实现 |
| `datasets/poly_data.py` | COCO 格式数据集加载 |

### 数据流程

1. **输入**：3D 点云 → 投影到俯视图密度图（256×256）
2. **骨干网络**：ResNet 提取多尺度特征
3. **Transformer**：编码器处理特征，解码器使用两级查询（房间 + 角点）
4. **输出**：
   - `pred_logits`：角点分类（有效/无效）
   - `pred_coords`：角点坐标（归一化 [0,1]）
   - `pred_room_logits`：房间语义类别（可选）

### 预训练检查点

`checkpoints/` 目录包含：
- `roomformer_stru3d.pth`：在 Structured3D 上训练
- `roomformer_stru3d_tight.pth`：在紧凑房间布局上训练
- `roomformer_stru3d_semantic_rich.pth`：用于语义丰富的楼层平面图
- `roomformer_scenecad.pth`：在 SceneCAD 上训练

### 数据期望结构

```
data/
├── stru3d/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── scenecad/
    ├── train/
    ├── val/
    └── annotations/
        ├── train.json
        └── val.json
```

Structured3D 评估需要 MonteFloor 真实数据位于 `./s3d_floorplan_eval/montefloor_data`。
