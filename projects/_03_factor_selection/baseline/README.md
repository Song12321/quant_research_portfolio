# 第一阶段正确性基线

精简前的冻结包位于项目外：

`D:\lqs\codeAbout\py\Quantitative\stock\quant_research_baselines\legacy_d8387bcf_20230710_20250710`

冻结契约：

- 股票池：`ZZ800`；
- 日期：`2023-07-10` 至 `2025-07-10`；
- 收益：`o2o`；
- 周期：5D、21D；
- 分层数：5；
- 案例：`volatility_40d`、`operating_accruals`、`three_low_one_high`。

验证脚本先核验冻结 manifest 与全部冻结产物的 SHA-256，再重算三个案例的 prepared、processed、IC、分层和换手结果并精确比较：

```powershell
D:\lqs\codeAbout\py\env\env_quant_py_3.11__reload_dium\Scripts\python.exe projects\_03_factor_selection\baseline\verify_phase1_parity.py
```

旧冻结包只作为本阶段的 Legacy 正确性证据，不作为后续 Inner / Out / Finalout 的盲测数据。
