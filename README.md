# Occ_Orin

面向 `Jetson AGX Orin 32GB` 车端环境的 OCC 项目仓库。

当前仓库主要包含三部分内容：

- `orin_occ/`
  - 面向当前车辆配置重写的 OCC baseline
  - 采用多相机主干、稠密 BEV、可扩展 LiDAR BEV 增强的工程路线
  - 已打通公开 `Occ3D-nuScenes-mini` 的训练数据链路
- `当前环境/`
  - 当前车辆硬件环境说明
  - 其中 `大车.md` 作为最新硬件情况基线
- `调研/`
  - OCC 部署、Flash-OCC、Orin 方案和当前车端配置判断的调研文档

## 当前车辆环境

以 [大车.md](./当前环境/大车.md) 为准，当前已确认：

- 域控：`Jetson AGX ORIN 32G`
- 相机：`10` 路
  - 前向 `3` 路
  - 后向微光 `3` 路
  - 环视 `4` 路
- 激光雷达：`5` 个
  - 前向 `Robosense-M1P`
  - 前后左右 `4` 个 `Robosense-E1R`
- 毫米波：`大陆 ARS548`
- 组合导航：`导远 INS-5711（含 RTK）`

## 当前工程状态

`orin_occ/` 不是对 FlashOCC 的直接搬运，而是为当前车端环境整理的一套更轻量、更容易继续开发和导出的 baseline：

- 支持自定义 JSONL manifest
- 支持 `Occ3D labels.npz`
- 支持 `nuScenes + Occ3D` 公共数据启动
- 已在远端 `RTX 4090` 环境完成：
  - smoke test
  - synthetic training sanity check
  - 真实 `Occ3D-nuScenes-mini` 单步训练 sanity check

## 目录说明

```text
.
├─ orin_occ/
├─ 当前环境/
└─ 调研/
```

## 说明

- 本仓库不包含大体量数据集与压缩包。
- 第三方参考代码 `FlashOCC-master` 与下载数据未纳入版本库。
- 如果继续推进，优先从 `orin_occ/README.md` 开始。
