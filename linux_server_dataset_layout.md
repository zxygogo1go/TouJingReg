# Linux 服务器数据集结构与新项目复用说明

更新时间：2026-07-21

本文记录当前 MUSA 项目在 Linux 服务器上的原始数据与预处理数据组织方式，并说明 GAM-Reg 的复用边界。图像、标签和病例划分可以复用，但必须先完成第 7 节的数据兼容检查与 manifest 转换。当前服务器项目目录为：

```text
/root/autodl-tmp/MUSA
```

## 1. 服务器总体结构

```text
/root/autodl-tmp/
├── MUSA/                                      # 当前项目
│   ├── data/                                  # SegRap 预处理数据，120 例
│   ├── data_hanseg/                           # HaN-Seg 预处理数据，42 例
│   ├── HaN-Seg/set_1/                         # HaN-Seg 原始 NRRD 数据（若仍保留）
│   ├── outputs/                               # SegRap 模型与实验输出
│   └── outputs_hanseg/                        # HaN-Seg 模型与实验输出
└── SegRap2023_Training_Set_120cases/          # SegRap 原始 NIfTI 数据
    ├── segrap_0000/
    ├── segrap_0001/
    └── ...
```

SegRap 的 metadata 中记录的原始数据绝对路径是：

```text
/root/autodl-tmp/SegRap2023_Training_Set_120cases
```

新项目训练时通常只需要复用 `data/` 或 `data_hanseg/`，不需要再次读取原始 NIfTI/NRRD 数据。

## 2. 统一的预处理数据协议

两个数据集都遵循下面的目录结构：

```text
<DATA_ROOT>/
├── images/
│   └── <case_id>.npy                         # 归一化 CT
├── seg_o/
│   └── <case_id>.npy                         # 多类别 OAR 标签
├── seg_b/
│   └── <case_id>.npy                         # 二值骨性结构标签
├── metadata/
│   └── <case_id>.json                        # 标签映射与预处理信息
└── lists/
    └── paper_split/
        ├── trn_list_inter.txt                 # 训练 case ID
        ├── val_list_inter.txt                 # 验证 moving/fixed ID
        ├── val_pairs.csv                      # 验证配准对
        ├── test_pairs.csv                     # 独立测试配准对
        └── README.md                          # 数据划分说明
```

同一个病例必须在 `images`、`seg_o`、`seg_b` 和 `metadata` 下使用完全相同的文件名。例如：

```text
images/segrap_0000.npy
seg_o/segrap_0000.npy
seg_b/segrap_0000.npy
metadata/segrap_0000.json
```

### 数组约定

| 内容 | 磁盘形状 | dtype | 数值约定 |
|---|---:|---|---|
| `images/*.npy` | `(160,160,192)` | `float32` | CT 经 `[-1024,3000]` 截断后归一化到 `[0,1]` |
| `seg_o/*.npy` | `(160,160,192)` | `int16` | `0` 为背景，正整数为 OAR 标签 |
| `seg_b/*.npy` | `(160,160,192)` | `int16` | 只允许 `0/1` |

预处理空间为 2 mm 各向同性。数据加载器会自动增加通道维，因此进入网络后的单例形状是 `(1,160,160,192)`，组成 batch 后是 `(B,1,160,160,192)`。

### metadata 的作用

每个 JSON 至少记录：

- `case_id`；
- 原始尺寸和 spacing；
- `target_shape=[160,160,192]`；
- `target_spacing=[2.0,2.0,2.0]`；
- CT 截断范围；
- `label_map`，即结构名称到整数标签的映射；
- 骨性结构列表和预处理后的实际标签。

用于三维配准时，建议 metadata 进一步明确记录：

- `array_axis_order="zyx"`，确认 `.npy` 的三个轴对应网络的 `D,H,W`；
- `target_direction`，记录统一后的 3x3 方向矩阵；
- `target_origin` 或统一 crop frame 的定义；
- `prealigned=true`，或保存实际使用的 `rigid_transform`/`affine_transform`。

旧 metadata 缺少这些字段时，不能仅为了通过检查而填写固定值。应先回查 MUSA 的预处理代码和抽样病例，确认数组轴、方向和粗配准确实满足要求；否则应从原始 NIfTI/NRRD 重新预处理。

新项目中建议保留整个 `metadata/`。其中 `case_dir`、`image_path` 等原始绝对路径在移动项目后可能失效，但当前小 OAR 标签解析只依赖 `label_map`，不依赖这些旧绝对路径。

## 3. SegRap 数据

预处理数据根目录：

```text
/root/autodl-tmp/MUSA/data
```

病例范围与文件数：

```text
case_id: segrap_0000 ... segrap_0119
images:  120
seg_o:   120
seg_b:   120
metadata:120
```

当前 paper split：

| 子集 | 病例 | 数量 |
|---|---|---:|
| test | `segrap_0000` 至 `segrap_0009` | 10 例，5 对 |
| val | `segrap_0010` 至 `segrap_0019` | 10 例，5 对 |
| train | `segrap_0020` 至 `segrap_0119` | 100 例 |

对应路径：

```text
/root/autodl-tmp/MUSA/data/lists/paper_split/trn_list_inter.txt
/root/autodl-tmp/MUSA/data/lists/paper_split/val_list_inter.txt
/root/autodl-tmp/MUSA/data/lists/paper_split/val_pairs.csv
/root/autodl-tmp/MUSA/data/lists/paper_split/test_pairs.csv
```

## 4. HaN-Seg 数据

预处理数据根目录：

```text
/root/autodl-tmp/MUSA/data_hanseg
```

服务器上应包含：

```text
data_hanseg/
├── images/                                   # hanseg_0001.npy ... hanseg_0042.npy
├── seg_o/                                    # 同名多类别 OAR 标签
├── seg_b/                                    # 同名下颌骨二值标签
├── metadata/                                 # 同名 JSON
└── lists/paper_split/
```

HaN-Seg 共 42 例，case ID 为 `hanseg_0001` 至 `hanseg_0042`。预处理使用固定的全局 OAR 标签顺序，因此某个病例缺少结构时不会改变其他标签的编号。默认将 `OAR_Bone_Mandible` 从 `seg_o` 排除，并单独写入二值 `seg_b`。

当前 paper split：

| 子集 | 病例规则 | 数量 |
|---|---|---:|
| test | case 01-10，相邻病例配成 5 对 | 10 例 |
| val | case 11-18、20-21，配成 5 对 | 10 例 |
| train | 其余病例，包括 case 19 | 22 例 |

case 19 缺少 `OAR_OpticChiasm`，因此保留在训练集，不进入验证和测试指标。

对应路径：

```text
/root/autodl-tmp/MUSA/data_hanseg/lists/paper_split/trn_list_inter.txt
/root/autodl-tmp/MUSA/data_hanseg/lists/paper_split/val_list_inter.txt
/root/autodl-tmp/MUSA/data_hanseg/lists/paper_split/val_pairs.csv
/root/autodl-tmp/MUSA/data_hanseg/lists/paper_split/test_pairs.csv
```

说明：`data_hanseg/images`、`seg_o`、`seg_b` 和 `metadata` 已被 `.gitignore` 排除，所以不会随 Git 仓库上传；新项目必须在服务器上通过绝对路径或软链接复用。

## 5. 列表文件格式

### 训练列表

`trn_list_inter.txt` 每行一个不带扩展名的 case ID：

```text
segrap_0020
segrap_0021
segrap_0022
```

当前训练数据集会对这些 ID 构造笛卡尔积配准对，包括同一病例到自身的配准对。

### 验证列表

`val_list_inter.txt` 的前半部分是 moving，后半部分是与其逐项对应的 fixed。例如：

```text
segrap_0010
segrap_0012
segrap_0014
segrap_0016
segrap_0018
segrap_0011
segrap_0013
segrap_0015
segrap_0017
segrap_0019
```

它会生成 `0010->0011`、`0012->0013` 等 5 个验证对。文件行数必须为偶数。

### pair CSV

`val_pairs.csv` 和 `test_pairs.csv` 没有表头，每行严格为：

```text
moving_id,fixed_id
```

例如：

```text
segrap_0000,segrap_0001
segrap_0002,segrap_0003
```

这种 CSV 是 MUSA 的病例 ID 协议，不是 GAM-Reg 可直接读取的 manifest。GAM-Reg 要求第一行包含 `moving,fixed,moving_seg,fixed_seg`，且每个单元格是相对 `--data-root` 的文件路径。第 8 节的工具负责转换，不能直接把这里的 `val_pairs.csv` 传给 `train.py` 或 `validate.py`。

## 6. 新项目推荐复用方式

推荐使用软链接，避免在 AutoDL 磁盘中复制两份大体积数组。假设新项目目录是：

```text
/root/autodl-tmp/NEW_PROJECT
```

在服务器执行：

```bash
cd /root/autodl-tmp/NEW_PROJECT
ln -s /root/autodl-tmp/MUSA/data data
ln -s /root/autodl-tmp/MUSA/data_hanseg data_hanseg
```

之后新项目可继续使用相对路径：

```text
data/images
data/seg_o
data/seg_b
data/metadata
data/lists/paper_split
```

如果新项目只使用 HaN-Seg，则只创建 `data_hanseg` 软链接即可。

也可以完全不创建链接，直接在配置中使用绝对路径：

```bash
--vol-path /root/autodl-tmp/MUSA/data_hanseg/images \
--seg-path-o /root/autodl-tmp/MUSA/data_hanseg/seg_o \
--seg-path-b /root/autodl-tmp/MUSA/data_hanseg/seg_b \
--metadata-path /root/autodl-tmp/MUSA/data_hanseg/metadata \
--trn-list /root/autodl-tmp/MUSA/data_hanseg/lists/paper_split/trn_list_inter.txt \
--val-list /root/autodl-tmp/MUSA/data_hanseg/lists/paper_split/val_list_inter.txt
```

如果需要让新项目的数据完全独立，可以使用：

```bash
rsync -a --info=progress2 /root/autodl-tmp/MUSA/data_hanseg/ /root/autodl-tmp/NEW_PROJECT/data_hanseg/
```

## 7. GAM-Reg 复用结论与兼容边界

| 项目 | 是否可直接复用 | GAM-Reg 处理方式 |
|---|---|---|
| `images/*.npy` | 有条件 | 数值是 `[0,1]`，必须使用 `zero_one` 模式映射到模型要求的 `[-1,1]`，不能再次按 HU 截断 |
| `seg_o/*.npy` | 有条件 | 可作为多类解剖监督；类别数必须由全数据标签和 `label_map` 推导，不能保留默认 5 类 |
| `seg_b/*.npy` | 是 | 可作为背景/骨性结构两类监督，配置中的类别数应为 2 |
| `metadata/*.json` | 是 | 用于校验 shape、spacing、标签映射和物理空间信息；旧绝对路径不参与训练 |
| `trn_list_inter.txt` | 需转换 | 转换为带文件路径的 `train_pairs.csv`；默认构造有方向的笛卡尔积并排除 self-pair |
| 原 `val_pairs.csv`、`test_pairs.csv` | 需转换 | 从无表头 ID 对转换为 GAM-Reg manifest |
| 旧 MUSA 命令行参数 | 否 | `--vol-path`、`--seg-path-o` 等不是本工程参数，应使用 `--manifest` 和 `--data-root` |

统一到 `(160,160,192)` 和 2 mm 各向同性只是必要条件，不是充分条件。GAM-Reg 规格还要求 moving/fixed 采用相同方向、相同轴语义，并已有刚性或仿射粗配准。当前预检工具能发现 metadata 中缺少这些证明，但仅凭 `.npy` 无法恢复物理方向或判断粗配准质量。

训练和验证必须按病例划分，不能先生成全数据配准对再随机拆分，否则同一病例会同时出现在训练和验证/测试中，造成数据泄漏。现有 paper split 是按病例隔离的，可以继续使用。

`seg_o` 的 one-hot 张量显存占用随类别数线性增长。若 `dataset_summary.json` 检测到类别数较多并导致显存不足，应明确选择论文所需的 OAR 子集并重新映射为连续标签；不能直接把不同结构合并成同一标签。仅验证配准链路时可以先用 `seg_b`。

## 8. 生成 GAM-Reg manifests 并预检

在 GAM-Reg 项目根目录运行。以下以 HaN-Seg 多类 OAR 为例：

```bash
python prepare_dataset.py \
  --data-root /root/autodl-tmp/MUSA/data_hanseg \
  --output-dir manifests/hanseg \
  --seg-dir seg_o \
  --expected-shape 160 160 192
```

工具会完整检查被 train/val/test 引用的病例，并生成：

```text
manifests/hanseg/
├── train_pairs.csv
├── val_pairs.csv
├── test_pairs.csv
├── dataset_summary.json
└── dataset_config.yaml
```

其中：

- 三个 CSV 已带 GAM-Reg 所需表头，文件路径相对 `--data-root`；
- `dataset_summary.json` 记录病例数、pair 数、shape、spacing、标签集合、类别数和 warnings；
- `dataset_config.yaml` 自动写入 `zero_one`、`target_shape` 和检测到的 `num_anatomy_classes`；
- `dataset_config.yaml` 同时写入 `spacing_dhw`，供物理坐标 smoothness 使用；
- Jacobian 安全损失同时使用全局 RMS 与最差 `0.1%` tail-RMS，并约束正向和逆向形变，避免少量严重折叠被全体素均值稀释；
- 训练 pair 默认不包含 `case->same case`；只有明确需要恒等样本时才加 `--include-self`。

若先使用骨性二值标签验证链路，应写到独立目录，避免覆盖多类配置：

```bash
python prepare_dataset.py \
  --data-root /root/autodl-tmp/MUSA/data_hanseg \
  --output-dir manifests/hanseg_bone \
  --seg-dir seg_b \
  --expected-shape 160 160 192
```

无分割监督时使用 `--seg-dir none`。这会生成空的 segmentation 列，并在数据配置中关闭 anatomy head、Dice loss 和 anatomy loss。

不要把 `--skip-array-check` 用于第一次接入；它只适合数据已完整验证后的快速重建 manifest。`--allow-missing-metadata` 同样会降低可审计性，不推荐用于正式实验。

## 9. GAM-Reg 训练、验证与测试命令

先用一个真实 pair 做显存和数据链路检查：

```bash
python train.py \
  --config manifests/hanseg/dataset_config.yaml \
  --stage registration-warmup \
  --manifest manifests/hanseg/train_pairs.csv \
  --data-root /root/autodl-tmp/MUSA/data_hanseg \
  --epochs 1 \
  --steps-per-epoch 1 \
  --device cuda \
  --output-dir runs/hanseg_real_smoke
```

检查通过后再增加 epoch 和 steps。验证集命令：

```bash
python validate.py \
  --config manifests/hanseg/dataset_config.yaml \
  --checkpoint runs/hanseg_real_smoke/checkpoints/latest.pt \
  --manifest manifests/hanseg/val_pairs.csv \
  --data-root /root/autodl-tmp/MUSA/data_hanseg \
  --max-batches 5 \
  --output-json runs/hanseg_real_smoke/validation.json \
  --device cuda
```

独立测试必须使用从未参与调参的 `test_pairs.csv`：

```bash
python validate.py \
  --config manifests/hanseg/dataset_config.yaml \
  --checkpoint runs/gam_reg/checkpoints/latest.pt \
  --manifest manifests/hanseg/test_pairs.csv \
  --data-root /root/autodl-tmp/MUSA/data_hanseg \
  --max-batches 5 \
  --device cuda
```

日志中应重点检查总 loss 是否有限、`folding_ratio_metric` 是否接近 0，以及输出是否出现 NaN/Inf。正式实验前还需抽样可视化 moving、fixed、warped moving 的三正交切片，确认方向与配准关系符合预期。

## 10. 服务器补充完整性检查

复用前先在服务器检查四类文件数量是否一致：

```bash
for root in /root/autodl-tmp/MUSA/data /root/autodl-tmp/MUSA/data_hanseg; do
  echo "DATA_ROOT=$root"
  for folder in images seg_o seg_b metadata; do
    printf '%-10s ' "$folder"
    find "$root/$folder" -maxdepth 1 -type f | wc -l
  done
done
```

预期结果：

```text
data:        images=120, seg_o=120, seg_b=120, metadata=120
data_hanseg: images=42,  seg_o=42,  seg_b=42,  metadata=42
```

使用当前 MUSA 校验脚本：

```bash
cd /root/autodl-tmp/MUSA
python scripts/preprocess/validate_prepared_data.py --data-root data
python scripts/preprocess/validate_prepared_data.py --data-root data_hanseg
```

检查软链接是否正确：

```bash
readlink -f /root/autodl-tmp/NEW_PROJECT/data
readlink -f /root/autodl-tmp/NEW_PROJECT/data_hanseg
```

## 11. 通用项目读取数据时的最小约束

新项目不必使用 MUSA 的 Dataset 类，但必须保证：

1. 使用 case ID 拼接 `.npy`，并在三个数组目录中读取同名文件；
2. CT 转为 `float32`，标签转为整数或所需的 one-hot/mask；
3. 增加通道维，将 `(160,160,192)` 转为 `(1,160,160,192)`；
4. moving、fixed、OAR 和 bone 的空间形状完全一致；
5. 训练时不要把 `test_pairs.csv` 用于调参或 checkpoint 选择；
6. 需要按结构名称获取标签时，从 `metadata/<case_id>.json` 的 `label_map` 解析，不要硬编码不同数据集之间不一致的标签编号。
