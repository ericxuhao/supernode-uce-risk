"""用自己的数据运行算法：生成一份示例数据（字段名故意与默认不同），然后识别字段、训练、预测。

    python examples/byo_quickstart.py            # 在发布包根目录运行

把 make_example_data() 换成读取你自己的表即可。真实数据通常是：
  errors.csv     节点名, 卡号, 时间, 错误编码（含不可纠正错误的编码，如 48 / 94 / 95）
  sbe.csv        节点名, 卡号, 时间, 可纠正错误累计计数（可选）
  telemetry.csv  节点名, 卡号, 时间, 温度 / 功率等数值列（可选）
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "byo_data"
DAY = 86400


def make_example_data(seed=0, hosts=6, cards=4, days=90):
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
    weak = rng.random(hosts * cards) < 0.35
    err, sbe, tel = [], [], []
    for k in range(hosts * cards):
        h, g = f"node{k // cards:02d}", k % cards
        for tu in rng.uniform(t0 + 5 * DAY, t0 + days * DAY, rng.poisson(6 if weak[k] else 0.3)):
            err.append((h, g, pd.Timestamp(tu, unit="s", tz="UTC"), 48))
            if rng.random() < 0.7:
                err.append((h, g, pd.Timestamp(tu - rng.uniform(600, 6 * 3600), unit="s", tz="UTC"), 63))
        for tx in rng.uniform(t0, t0 + days * DAY, rng.poisson(20)):
            err.append((h, g, pd.Timestamp(tx, unit="s", tz="UTC"), 31))
        ts = np.sort(rng.uniform(t0, t0 + days * DAY, rng.poisson((3.0 if weak[k] else 0.2) * days)))
        sbe += [(h, g, int(t * 1000), i + 1) for i, t in enumerate(ts)]
        for t in range(t0, t0 + days * DAY, 3600):
            tel.append((h, g, pd.Timestamp(t, unit="s", tz="UTC").isoformat(), 60 + 5 * weak[k] + rng.normal(0, 2),
                        250 + rng.normal(0, 20)))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(err, columns=["Hostname", "GPU_Index", "Timestamp", "XID"]).to_csv(OUT / "errors.csv", index=False)
    pd.DataFrame(sbe, columns=["Host", "GPU", "time_ms", "SBE_Count"]).to_csv(OUT / "sbe.csv", index=False)
    pd.DataFrame(tel, columns=["hostname", "gpu_index", "collect_time", "mem_temp_c", "power_w"]).to_csv(
        OUT / "telemetry.csv", index=False)


def cli(*args):
    cmd = [sys.executable, "-m", "uce_risk.cli", *args]
    print("\n$ uce-risk", " ".join(args), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE.parent)


if __name__ == "__main__":
    make_example_data()
    d = "examples/byo_data/"
    inputs = ["--errors", d + "errors.csv", "--uce-codes", "48", "--ce", d + "sbe.csv", "--telemetry", d + "telemetry.csv"]
    cli("describe", *inputs)
    cli("train", *inputs, "--val-start", "2024-03-01", "--out", d + "model.joblib", "--bags", "2")
    cli("predict", "--model", d + "model.joblib", *inputs, "--at", "2024-03-25", "--out", d + "risk.csv")
    print(pd.read_csv(OUT / "risk.csv").head(8).to_string(index=False))
