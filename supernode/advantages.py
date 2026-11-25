"""Data for the '算法优势' page: comparison with published results and the ablation on the backtrack window.

Every number of 'ours' is read from experiment result files (experiments/uce_unified/results) or the demo
build metadata at request time; published numbers are those reported in the papers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_bundled = Path(__file__).resolve().parent / "benchmarks"          # results bundled with a release
RES = _bundled if _bundled.exists() else ROOT / "experiments" / "uce_unified" / "results"


def _csv(name):
    f = RES / name
    return pd.read_csv(f) if f.exists() else None


def _calchas():
    out = {}
    for backend, f in (("LightGBM", "calchas_compare_lgbm.csv"), ("HGB", "hgb/calchas_compare.csv")):
        p = RES / f
        if not p.exists():
            continue
        d = pd.read_csv(p, header=[0, 1], index_col=[0, 1])
        out[backend] = {(proto, model): {k: float(d.loc[(proto, model), (k, "mean")]) for k in ("P", "R", "F1", "PR_AUC")}
                        for proto, model in d.index}
    return out


def _pooled(name, scenario=None, model="hybrid", horizon="1d"):
    d = _csv(name)
    if d is None:
        return None
    q = d[(d.model == model) & (d.variant == "full") & (d.horizon == horizon)]
    if scenario:
        q = q[q.scenario == scenario]
    return q.iloc[0].to_dict() if len(q) else None


def comparison() -> list[dict]:
    rows = []
    c = _calchas()
    lg = c.get("LightGBM", {})
    hgb = c.get("HGB", {})
    a_ours = lg.get(("A_calchas", "Hybrid_lgbm_valF1"))
    a_rf = lg.get(("A_calchas", "Calchas_RF_thr0.6"))
    b_ours = lg.get(("B_strict", "Hybrid_lgbm_valF1"))
    b_rf = lg.get(("B_strict", "Calchas_RF_thr0.6"))
    if a_ours:
        rows.append({"dataset": "Calchas HBM（加速器显存）", "paper": "Calchas（USENIX ATC'24）", "task": "服务器级，1 天内不可纠正错误",
                     "protocol": "论文同口径（论文代码的降采样与切分方式），30 次平均", "metric": "F1",
                     "paper_value": 0.42, "ours": a_ours["F1"], "fmt": "f3",
                     "detail": f"本框架 P {a_ours['P']:.3f} / R {a_ours['R']:.3f}；论文 P 0.34 / R 0.56；同口径重跑 Calchas 随机森林 F1 {a_rf['F1']:.3f}",
                     "verdict": "超越" if a_ours["F1"] > 0.42 else "持平"})
        rows.append({"dataset": "Calchas HBM（加速器显存）", "paper": "Calchas（USENIX ATC'24）", "task": "同上",
                     "protocol": "论文同口径", "metric": "PR-AUC（排序能力）", "paper_value": a_rf["PR_AUC"], "paper_note": "论文方法同口径重跑",
                     "ours": a_ours["PR_AUC"], "fmt": "f3",
                     "detail": f"比 Calchas 随机森林高 {100 * (a_ours['PR_AUC'] / a_rf['PR_AUC'] - 1):.0f}%",
                     "verdict": "超越" if a_ours["PR_AUC"] > a_rf["PR_AUC"] else "持平"})
    if b_ours:
        rows.append({"dataset": "Calchas HBM（加速器显存）", "paper": "Calchas（USENIX ATC'24）", "task": "同上",
                     "protocol": "严格口径（测试集保持真实正负比例）", "metric": "PR-AUC", "paper_value": b_rf["PR_AUC"],
                     "paper_note": "论文方法同口径重跑", "ours": b_ours["PR_AUC"], "fmt": "f3",
                     "detail": f"高 {100 * (b_ours['PR_AUC'] / b_rf['PR_AUC'] - 1):.0f}%；F1 {b_ours['F1']:.3f} vs {b_rf['F1']:.3f}",
                     "verdict": "超越" if b_ours["PR_AUC"] > b_rf["PR_AUC"] else "持平"})
    for tag, backend in (("_cov_lgbm", "LightGBM"), ("_cov_hgb", "HGB")):
        r = _pooled(f"dram{tag}_pooled.csv")
        if r and r.get("oracle_F1_cov_mean") == r.get("oracle_F1_cov_mean"):
            v = r["oracle_F1_cov_mean"]
            rows.append({"dataset": "阿里 DRAM（服务器内存，天池公开）", "paper": "SRDS'22（Cheng 等）", "task": "1 天内不可纠正错误引起的故障",
                         "protocol": f"按月时间切分，测试集取最优阈值，召回按有可预测样本的服务器计（论文常用的报告方式）；本框架 {backend}",
                         "metric": "F1", "paper_value": 0.175, "paper_note": "论文 LightGBM 17.5%（随机森林约 16%）", "ours": v,
                         "fmt": "pct", "detail": f"阈值在验证月确定（更严格）时 F1 {r['valF1_F1_cov']:.1%}；样本级 PR-AUC {r['PR_AUC']:.3f}",
                         "verdict": ("超越" if v > 0.18 else "略高（同等水平）") if v > 0.175 else ("持平" if v > 0.16 else "低于")})
    r = _pooled("dram_pooled.csv", horizon="1h")
    if r:
        rows.append({"dataset": "阿里 DRAM（服务器内存，天池公开）", "paper": "SRDS'22（Cheng 等）", "task": "1 小时 / 4 小时内不可纠正错误",
                     "protocol": "严格口径", "metric": "F1（1 小时）", "paper_value": None, "paper_note": "论文未提供 1 小时、4 小时预测",
                     "ours": r["valF1_F1_cov"], "fmt": "pct", "detail": "同时输出 1/4/24 小时三个一致的、校准过的概率",
                     "verdict": "新增能力"})
    r = _pooled("summit_pooled.csv")
    if r:
        rows.append({"dataset": "OLCF Summit（hpc_dataset_artifact）", "paper": "ISSRE'26（Lu 等）", "task": "论文：10 分钟内任意错误，只用遥测；本框架：1 天内显存不可纠正错误",
                     "protocol": "任务不同，仅作参考", "metric": "事件召回 / 精确率", "paper_value": 0.0035,
                     "paper_note": "论文最好的事件级 F1 0.35%，AUROC 0.44–0.59", "ours": r["valF1_ep_recall"], "fmt": "pct",
                     "detail": f"本框架事件召回 {r['valF1_ep_recall']:.1%}（复发 {r['valF1_recur_recall']:.0%}），设备级精确率 {r['valF1_P']:.0%}；首次故障召回 {r['valF1_first_recall']:.0%}",
                     "verdict": "不同任务"})
    return rows


def ablation(meta: dict) -> dict:
    return meta.get("ablation", {})
