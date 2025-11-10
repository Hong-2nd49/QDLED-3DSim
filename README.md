# QDLED-3DSim
3D carrier transport and recombination analysis toolkit for QLEDs with micro/nano structuring. Parses TCAD/COMSOL data, builds geometry-aware features, and trains surrogate models to support rapid architecture screening.

---

**3D Carrier Transport & Recombination Toolkit for Microstructured QLEDs**  
面向微/纳结构 QLED 的三维载流子输运与复合分析工具包

---

## 1. Overview | 项目简介

QDLED-3DSim is a toolkit for analyzing **3D carrier transport, recombination profiles, and EQE proxies** in QLED devices with internal micro/nano structuring.

It is designed to:

- Ingest **TCAD / COMSOL / Lumerical** simulation outputs
- Represent **2D/3D device geometries** (layer stacks, lateral ZnO–QD patterns, multi-EML, etc.)
- Extract physically meaningful metrics: carrier maps, recombination heatmaps, internal EQE proxies
- Train **surrogate models** that predict device behavior from geometry

QDLED-3DSim 专注于将三维器件仿真结果转化为：
- 可解释的物理参数（载流子分布、复合分布、EQE 近似）
- 面向结构优化的几何特征和代理模型
并可与 QLED-RLopt 联动，作为强化学习优化的高保真“oracle”。

---

## 2. Features | 功能特性

- 📥 **Simulator Parsing**
  - 解析 COMSOL / Lumerical / TCAD 导出的 CSV / 数据文件
- 🧱 **Geometry Handling**
  - 接收 2D / 3D 结构定义（层结构 + 平面图案参数）
- 🧠 **Surrogate Modeling (Optional)**
  - 使用 3D CNN / GNN（可拓展）拟合仿真映射，辅助快速筛选结构
- 📊 **Visualization**
  - 3D / 2D 载流子与复合分布可视化
  - 对比不同结构设计下的性能指标

---

## 3. Repository Structure | 仓库结构

```text
QDLED-3DSim/
├── README.md                      # 项目说明（本文件）
├── requirements.txt               # 依赖配置
├── config/
│   └── default_materials.yaml     # 材料参数示例（ZnO, QD, HTL 等）
├── simulator/
│   ├── __init__.py
│   ├── comsol_parser.py           # 通用 COMSOL/TCAD 输出解析示例
│   ├── geometry_builder.py        # 用于描述/生成器件结构网格
│   └── mesh_configurator.py       # 网格与边界设置（示例/预留）
├── ai_model/
│   ├── __init__.py
│   ├── featurize_geometry.py      # 将几何与材料信息编码为特征
│   ├── train_model.py             # 训练代理模型（如 MLP / 3D CNN / GNN）
│   └── evaluate_model.py          # 评价代理模型性能与误差
├── data/
│   ├── raw_simulations/           # 原始仿真文件（CSV / HDF5 等）
│   └── preprocessed/              # 处理后的特征与标签
├── visualization/
│   ├── __init__.py
│   └── render_3d_carriers.py      # 三维载流子与复合分布绘图
├── scripts/
│   └── run_full_simulation.py     # 读取配置 + 解析数据 + 输出指标示例
├── notebooks/
│   ├── 01_inspect_simulation_data.ipynb
│   └── 02_compare_structures.ipynb
└── LICENSE
```
## 4. Usage | 使用方式

Run device simulations in COMSOL / Lumerical / TCAD for your QLED structures

Export spatial data (e.g. x,y,z,n_electron,n_hole,R_rad,R_nrad) as CSV

Place files under data/raw_simulations/

Use:

simulator/comsol_parser.py to parse

visualization/render_3d_carriers.py to plot maps

ai_model/train_model.py to fit surrogate models (optional)

## 5. Integration with QLED-RLopt | 与 QLED-RLopt 的联动

QDLED-3DSim 提供高保真三维仿真解析与指标抽取

QLED-RLopt 使用这些指标作为 RL 奖励信号

二者可组成一条完整链路：
结构参数 → 3D 仿真 → 指标 → RL 优化 → 新结构候选

## 6. License | 许可

建议 MIT License，便于科研协作与交叉使用。
