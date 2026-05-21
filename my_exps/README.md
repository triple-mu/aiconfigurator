# GLM-4.5-Air @ GB200 PD-Disagg 搜索实验

## 目标

为 SWEBench 多轮 agent 工作负载找出 GLM-4.5-Air 在 GB200 上的最优 PD-disagg 配置,
使 8-GPU disagg 集群吞吐 ≥ 2× 4-GPU agg baseline。

## 3 个实验文件

| 文件 | 拓扑 | total_gpus | 角色 |
|---|---|---|---|
| `01_agg_4gpu_baseline.yaml` | 单 worker, TP=4 | 4 | **基线**(分母) |
| `02_disagg_pd_4_4_locked.yaml` | 1P(TP4) + 1D(TP4) | 8 | 你指定的拓扑,在此空间内找最优 server 参数 |
| `03_disagg_open_8gpu.yaml` | 任意 P/D × TP 组合 | 8 | 不锁拓扑,看 aic 是否找到更优 |

每个文件内含 **3 个工作负载点**(短/中/长 multi-turn),分别独立搜索。

## 工作负载点说明

| 点 | ISL | OSL | prefix | 含义 |
|---|---|---|---|---|
| short | 2000 | 800 | 500 | 第 1 轮,sys+tool+Q1,prefix 只到 sys |
| mid | 5000 | 1200 | 3500 | 第 3-4 轮,prefix 是前轮历史 |
| long | 12000 | 1500 | 10500 | 第 8+ 轮,绝大部分是 cached prefix |

→ **请根据你 SWEBench dataset 的实际 ISL/OSL 分布调整这些数字**(在每个文件的
`exp_*_short/mid/long` 段下的 `isl/osl/prefix` 字段)。

## 关键设置

- **Backend**: sglang 0.5.10(GB200 上对 non-DSA MoE 支持最完整)
- **Database mode**: **HYBRID**(GB200 sglang 0.5.10 缺 `moe_inter_size=1408` 的真实采集点,
  HYBRID 用 1024 和 1536 两个点做插值,精度损失 < 5%)
- **Quant**: 全程 bf16(`gemm/moe/kvcache/fmha` 全 `bfloat16`)
- **SLA**: `TTFT=1000ms, TPOT=60ms`(宽松,避免过滤掉高吞吐配置;
  你后面跑 SWEBench 时按业务实际 SLA 收紧再 sweep 一次)
- **MTP**: 暂关(`nextn: 0`)——GLM-4.5-Air 有 `num_nextn_predict_layers=1`,
  可后续单开一份 exp 测 MTP 加速(需要先实测 accept rate)

## 跑实验

```bash
# 在仓库根目录,激活 venv 后:

# 1. 跑 agg baseline (4 卡,~分钟级)
.venv/bin/aiconfigurator cli exp \
  --yaml-path my_exps/01_agg_4gpu_baseline.yaml \
  --save-dir results/agg_baseline 2>&1 | tee logs/01_agg.log

# 2. 跑 disagg locked (8 卡,~分钟级)
.venv/bin/aiconfigurator cli exp \
  --yaml-path my_exps/02_disagg_pd_4_4_locked.yaml \
  --save-dir results/disagg_pd44 2>&1 | tee logs/02_disagg_pd44.log

# 3. 跑 disagg open (8 卡,搜索空间更大,~10分钟级)
.venv/bin/aiconfigurator cli exp \
  --yaml-path my_exps/03_disagg_open_8gpu.yaml \
  --save-dir results/disagg_open 2>&1 | tee logs/03_disagg_open.log
```

每次跑完会在 `--save-dir` 下生成:
- `<exp_name>/<mode>/best_config_topn.csv` — top-5 候选(排序按 tokens/s/gpu)
- `<exp_name>/<mode>/pareto.csv` — Pareto 前沿
- `<exp_name>/<mode>/top1/<mode>/k8s_deploy.yaml` — top-1 的 K8s manifest(可直接 apply)
- `<exp_name>/<mode>/top1/<mode>/node_0_run.sh` — bare-metal 启动脚本
- `<exp_name>/<mode>/top1/<mode>/bench_run.sh` — aiperf 跑分脚本
- `pareto_frontier.png` — 可视化

## 分析方法

跑完三套后,对比:

### A. 同点对比(每个 workload 点都比一次)

| 工作负载 | 01 agg cluster tps | 02 disagg-locked cluster tps | 03 disagg-open cluster tps | locked/agg | open/agg |
|---|---|---|---|---|---|
| short  | ? | ? | ? | x | x |
| mid    | ? | ? | ? | x | x |
| long   | ? | ? | ? | x | x |

- locked/agg ≥ 2.0  → 你的 P=4+D=4 限制能达成 2x 目标
- open  > locked    → 别的拓扑更好(读 03 的 top-1 看是什么拓扑)
- open == locked    → 你的限制本身就是最优,继续 02 即可

### B. 在 02 内部找"鲁棒"配置

读 `02_disagg_pd44_short/mid/long` 各自的 `best_config_topn.csv`,
找**在 3 个 workload 点都进 top-3** 的 (moe_tp, moe_ep, prefill_bs, decode_bs)
组合 —— 这是对你 multi-turn workload 最稳健的 server 参数。

### C. 拿这个鲁棒配置去跑 SWEBench

`02_disagg_pd44_mid/top1/disagg/` 下的 `k8s_deploy.yaml` 或 `node_0_run.sh`,
直接部署到你的 GB200 集群,跑你的 SWEBench 多轮 bench 脚本,实测 end-to-end 吞吐。

## 已知限制

1. **HYBRID 模式精度损失**:绝对 tokens/s 数字可能偏差 ~5%,但**相对排序可信**。
2. **`first_k_dense_replace=1`**:GLM-4.5-Air 第 0 层是 dense FFN(`inter=10944`),
   aic 当前会把它当 MoE 算,~2-3% 算力高估。不影响相对排序。
3. **SGLang 0.5.10 模板 fallback 到 0.5.9**:generator 没专门为 0.5.10 出模板,
   floor-match 用 0.5.9 模板。SGLang 0.5.9 → 0.5.10 server args 基本兼容,
   部署时如发现某个 flag 报错,手改一下即可。
4. **MTP 没启用**:GLM-4.5-Air 支持 MTP(`num_nextn_predict_layers=1`),
   SGLang 部署时可开启 `--speculative-num-steps 1` 等,但 aic 估算需要单独 sweep
   `nextn` 并提供 accept_rate。当前 3 套 sweep 都是 MTP-off 的保守估计。

## 下一步迭代(如果首轮不达标)

如果 02 的 locked/agg 比值 < 1.5x,可能原因和应对:

- **prefill 太小**:在 `02_disagg_pd_4_4_locked.yaml` 把 `prefill_max_batch_size` 加大到 32
- **SLA 太紧**:`tpot` 从 60 放到 100,看是否找到吞吐更高的解
- **MoE 并行没选对**:看 `best_config_topn.csv` 里 top-1 的 `(moe_tp, moe_ep)`,
  如果是 (1,4) 或 (4,1) 而不是均衡的 (2,2),说明搜索空间还可以扩
- **拓扑限制太严**:跑 03_open,看是不是 `2P(TP2) + 1D(TP4)` 这类异构拓扑更好
