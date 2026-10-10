# Pan-Tompkins QRS 检测复现（MIT-BIH 全库评估）

从零复现 Pan-Tompkins（1985）经典 QRS 检测算法及其前置滤波链，在 PhysioNet MIT-BIH 心律失常数据库全库 48 条记录、108,512 个双专家标注心拍上量化评估，并与开源库 py-ecg-detectors 对拍。

> 《生物医学传感检测与仪器》课程选做作业 · 周增飏

## 结果摘要

| 配置 | Se | +P | F1 |
|---|---|---|---|
| **本文复现（完整滤波链）** | **99.50%** | 97.47% | **98.48%** |
| py-ecg-detectors（参照） | 99.20% | 97.78% | 98.49% |

- 与成熟开源实现 F1 持平，Se 反超 0.30 个百分点
- R 波定位误差中位数 −5.6 ms（|误差| < 25 ms 占 99.8%）
- HRV 参数（SDNN/RMSSD/pNN50/LF-HF）误差传递 0.16%–5.82%
- 算法边界：起搏尖峰（104 号）与重度噪声（108 号）记录，与文献一致

## 复现步骤

```bash
pip install -r requirements.txt
# 从 PhysioNet 下载 mitdb 1.0.0 解压至 data/mit-bih-arrhythmia-database-1.0.0/
# https://physionet.org/content/mitdb/1.0.0/
python code/01_explore.py        # 数据探查与读取验证
python code/02_pan_tompkins.py   # 单记录（100 号）三配置验证 + 算法各阶段图
python code/03_batch_evaluate.py # 全库 48 条批评估 + 参照库对拍（约 30 s）
python code/04_hrv.py            # HRV 时域/频域分析 + 误差传递
python code/05_figures.py        # 报告全部图表与汇总报表生成
```

所有图表输出至 `docs/`（仓库已预置一份生成产物）。

## 代码结构

| 文件 | 内容 |
|---|---|
| `code/ecg_lib.py` | 算法库：滤波链 / Pan-Tompkins / beat 匹配评估 |
| `code/01_explore.py` | 数据探查与读取验证 |
| `code/02_pan_tompkins.py` | 单记录三配置验证 + 阶段图 |
| `code/03_batch_evaluate.py` | 全库批评估 + 参照库对拍 + CSV + 失败图 |
| `code/04_hrv.py` | HRV 时域/频域分析 + 误差传递 |
| `code/05_figures.py` | 报告全部图表与汇总报表生成 |

## 实现要点

- **评估口径**：beat-level 一对一匹配，容差 ±150 ms（AAMI EC57 惯例）；pooled 指标由 TP/FP/FN 加总计算
- **预处理**：60 Hz IIR 陷波 + 两级中值滤波（0.2 s/0.6 s）去基线
- **算法链**：一阶因果带通（5–15 Hz）→ 微分 → 平方 → 150 ms 滑动积分 → 自适应双阈值 + search-back
- **因果性语义**：全链 `lfilter` 因果实现（复现中发现零相位滤波与中心对称滑窗会产生幻影峰，是复现实时算法最常见的隐性失效）
- **R 波精修**：在带通信号上取前倾窗 [p−220 ms, p+40 ms] 内最大斜率点定位

## 参考

- Pan J, Tompkins W J. A Real-Time QRS Detection Algorithm. IEEE Trans. Biomed. Eng., 1985, BME-32(3): 230–236.
- Moody G B, Mark R G. The impact of the MIT-BIH Arrhythmia Database. IEEE EMB Magazine, 2001, 20(3): 45–50.
- 依赖库：[wfdb-python](https://github.com/MIT-LCP/wfdb-python) · [py-ecg-detectors](https://github.com/berndporr/py-ecg-detectors)

## 许可

代码以 [MIT License](LICENSE) 发布。MIT-BIH 数据库版权归 PhysioNet/麻省理工学院所有，请遵循其使用条款自行下载。
