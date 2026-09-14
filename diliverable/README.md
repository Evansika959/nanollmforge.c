# Hardware predictor delivery — dynamic energy

本目录按用户指定的 `diliverable` 命名，保留在 Git ignore 之外。内容是已有数据和 checkpoint 的校验副本，没有重新训练、改写原始实验或启动硬件。

导出快照日期：**2026-09-14**。本说明的轮次和指标对应导出时点，不随后续实验自动更新。模块说明见 [prediction README](../scripts/prediction/README.md)，实际续跑步骤见 [active-learning guide](../scripts/prediction/active_learning/README.md)。

## 交付内容

| 文件 | 用途 |
|---|---|
| `data/dataset_latest.json` | 最新完整有效数据：dynamic 第 9 轮，1,552 train / 150 validation / 314 test，共 2,016 条 |
| `models/predictor_best.joblib` | 当前 dynamic 实测阶段验证集最优模型：第 1 轮 XGBoost，1,472 条训练数据 |
| `data/dataset_best_model.json` | 与最佳模型指纹严格匹配的数据版本，1,472 train / 150 validation / 314 test |
| `models/predictor_latest.joblib` | 第 9 轮最新模型，与 `dataset_latest.json` 严格匹配，用于后续训练工作 |
| `evaluation/model_selection.json` | 初始模型及第 1–9 轮共 10 个 checkpoint 的重新验证结果与选择规则 |
| `manifest.json` | 文件 SHA-256、原始位置、数据指纹、样本数量、协议和 Python 包版本 |
| `provenance/` | 原运行配置、状态、代码指纹、能耗标签迁移记录及本阶段各已完成轮次的测量 CSV |

“最佳”限定为**当前真实手表 dynamic 阶段**，按同一 150 条固定验证集上三个目标 MAPE 的算术平均值选择；不是所有历史实验的全局冠军。没有用测试集选模型，也没有重新评估测试集。反复使用验证集选择 checkpoint，结果不等同于独立最终测试性能。

最佳模型固定验证集结果：

| 目标 | MAPE | MAE | R² |
|---|---:|---:|---:|
| 解码吞吐率 | 22.024% | 5.490 tokens/s | 0.5857 |
| TTFT | 22.384% | 197.712 ms | 0.7337 |
| Dynamic energy | 43.558% | 5.940 mJ/output token | 0.8423 |

三个目标平均 MAPE 为 **29.322%**。最新模型不等于最佳模型：第 9 轮能耗 MAPE 为 44.100%。没有将最佳模型伪装成已使用最新全部数据训练的模型。

## 数据与协议

- 输入只有架构信息；温度、电量、基线功率不作为 predictor 输入。
- 能耗为扣基线后的 `max(active_power_w - baseline_power_w, 0) * duration_s * 1000 / 32`，包含 prefill 窗口，不是纯 decode 能耗。
- 工作负载：名义 48 / 实际 49 个 prompt tokens，32 个输出 tokens，31 次 decode forward。
- 历史协议对象保留 40°C；实际后续测量使用的 `<45°C` 推理前准入规则见 `provenance/settings.json` 和 observation 的测量上下文。这不是推理期间峰值温度保证。
- `AL_56edd3ffbca9b338` 扣后能耗为零，无法用于 log 训练；其原记录保存在 `provenance/energy_transition.json` 的隔离清单中，没有删除或钳成伪正值。该条不计入 2,016 条有效数据。
- 中断的第 10 轮没有完整新测量纳入本数据集。完整原始 traces、候选池及中断状态仍在原 outputs 中，本目录不是整个实验工作区的备份。

## 使用模型

从仓库根目录执行，使用现有 `nanollmforge` 环境；此 bundle 依赖仓库的 `scripts.prediction` 模块，并非无需代码的独立应用。准确包版本见 `manifest.json`。

```python
from scripts.prediction.data.dataset import MeasurementDataset
from scripts.prediction.models.serialization import load_bundle
from scripts.prediction.inference.predictor import predict_bundle

dataset = MeasurementDataset.load('diliverable/data/dataset_latest.json')
model = load_bundle('diliverable/models/predictor_best.joblib')
configs = [dataset.observations[0]['architecture']]  # 替换为待预测架构
predictions = predict_bundle(model, configs)
print(model['targets'])  # decode_tok_s, ttft_ms, dynamic_energy_per_token_mj
print(predictions)
```

也可使用现有 CLI，提供包含完整架构列的 CSV，并选择尚不存在的输出文件：

```bash
python -m scripts.prediction predict \
  --model diliverable/models/predictor_best.joblib \
  --configs YOUR_ARCHITECTURES.csv \
  --output scripts/prediction/outputs/delivery_predictions.csv
```

仅加载可信 joblib；pickle/joblib 反序列化可以执行代码。

## 继续实验与重新打包

充电后继续原实验时，仍使用 `scripts/prediction/outputs/active_learning_watch5_dynamic_under45`，不要把本目录当作可恢复的 AL workspace。若初始化一个全新工作区，checkpoint 与 dataset 必须匹配：最佳模型配 `dataset_best_model.json`，最新模型配 `dataset_latest.json`。

重新导出时指定新目录，工具拒绝覆盖已有交付物，不访问手表，也不训练模型：

```bash
python -m scripts.package_prediction_deliverable --destination diliverable_next
```

`scripts/prediction/outputs/` 与 `scripts/sweep/outputs/` 现已加入 `.gitignore`，原来被跟踪的输出仅从 Git 索引移除，本地文件仍在。此操作不会删除 Git 历史中的旧输出，也不会自动提交任何更改。
