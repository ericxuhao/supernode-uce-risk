"""Web service of the supernode demo.

    python supernode/server.py        # then open http://127.0.0.1:8765
(run supernode/build.py once before)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, DAY, DEMO_END, DEMO_START, HOUR, LIVE_START, N_CARDS, CAUSES, cause_zh  # noqa: E402
import fta  # noqa: E402
from engine import Engine, HZ  # noqa: E402
from cluster import ClusterEngine, capability_summary, load_settings, save_settings  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).with_name("static")), static_url_path="/static")
E: Engine | None = None            # engine behind the monitoring pages (replay or cluster)
REPLAY: Engine | None = None       # None when the replay data is not deployed (cluster-only)
META: dict = {}                    # model metadata (thresholds, fault-tree weights, validation results)
CLUSTER: ClusterEngine | None = None
SOURCE_ZH = {"log": "运行日志模型（错误事件 + 历史）", "ce": "显存 RAS 模型（地址级可纠正错误 + 事件 + 历史）",
             "": "未接入"}
EMPTY = pd.DataFrame({"card": pd.Series(dtype=int), "s": pd.Series(dtype=int), "p_1h": pd.Series(dtype=float),
                      "p_4h": pd.Series(dtype=float), "p_1d": pd.Series(dtype=float)})


def is_cluster():
    return E is CLUSTER
RULE_ZH = {"recur_time": "距上次不可纠正错误", "recur_count": "不可纠正错误次数", "node_uce": "同节点不可纠正错误",
           "app_event": "应用类错误", "ce_burst": "可纠正错误突发", "ce_accel": "可纠正错误加速", "ce_onset": "首次可纠正错误",
           "ce_storm": "可纠正错误风暴", "bank_fault": "Bank 级故障", "rowcol_fault": "行/列故障", "scatter": "错误单元分散",
           "hard_fault": "硬错误单元", "hi_any": "不可纠正错误历史", "ev_any": "错误事件", "ce_any": "可纠正错误"}
VAL_ZH = {"yes": "是", "no": "否", "none": "无", "rising": "上升", "flat": "平稳", "scattered": "分散", "other": "其他",
          "top10%": "前10%", "normal": "正常", "missing": "缺失", "older/none": "较早/无", ">30d/none": "30天以上/无",
          "low": "低", "mid": "中", "high": "高"}


def reason_zh(s: str) -> str:
    if not s or s == "baseline":
        return "无显著规则命中"
    k, _, v = s.partition("=")
    v = VAL_ZH.get(v, v.replace("h", "小时").replace("d", "天").replace("<=", "≤").replace(">=", "≥"))
    if k.startswith("hw_event_"):
        code = k[len("hw_event_"):]
        return f"硬件事件：{cause_zh(code)}（{v}）"
    if k.startswith("tel_high_"):
        return f"遥测偏高：{k[9:]}（{v}）"
    return f"{RULE_ZH.get(k, k)}：{v}"


def t_iso(t):
    return pd.Timestamp(int(t), unit="s", tz="UTC").tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")


def risk_at(t):
    """per-card risk at (the scan nearest before) time t, from history or live scans."""
    if is_cluster():
        with E.lock:
            cand = [x for x in E.live if int(x.s.iloc[0]) <= t]
            R = cand[-1] if cand else (E.live[-1] if E.live else EMPTY)
    elif t >= LIVE_START and E.live:
        with E.lock:
            cand = [x for x in E.live if int(x.s.iloc[0]) <= t]
        R = cand[-1] if cand else E.live[0]
    else:
        hh = (t // HOUR) * HOUR
        R = E.history[E.history.s == hh]
        if R.empty:
            R = E.history[E.history.s == E.history.s[E.history.s <= hh].max()]
    R = R.drop(columns=[c for c in ("source",) if c in R]).merge(E.cards[["card", "source"]], on="card")
    return E.levels(R.copy())


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    import time as _t
    last = E.scans[-1] if E.scans else {}
    ag = CLUSTER.agent_rows()
    cl = {"nodes": len(ag), "online": sum(a["online"] for a in ag), "cards": int((CLUSTER.cards.source != "").sum()),
          "interval_s": CLUSTER.settings["interval_s"], "next_scan_in": max(0, int(CLUSTER.next_scan - _t.time()))}
    mode = (f"集群实时采集（{cl['online']}/{cl['nodes']} 个节点在线，{cl['cards']} 张计算卡）" if is_cluster()
            else "开源数据回放" + (" + 本机实时采集" if REPLAY is not None and REPLAY.local.ok else ""))
    return jsonify({"title": CFG["title"], "supernode": CFG["supernode"], "sim_t": E.sim_t, "sim_time": t_iso(E.sim_t),
                    "playing": E.playing, "tick_s": E.tick_s, "live_start": LIVE_START, "demo_start": DEMO_START,
                    "demo_end": DEMO_END, "scans": len(E.scans), "last_scan": last, "cards": N_CARDS,
                    "source": "cluster" if is_cluster() else "replay", "cluster": cl, "mode": mode,
                    "replay_available": REPLAY is not None,
                    "horizons": [{"key": h, "label": CFG["horizon_labels"][h]} for h in HZ]})


def grid(R, t, truth=False):
    R = R.set_index("card")
    ev = pd.concat([E.events, E.future[E.future.t <= E.sim_t]], ignore_index=True)
    u = ev[(ev.cause == "MEMORY_UNCORRECTABLE_ERROR")]
    prev = set(map(tuple, u[(u.t > t - DAY) & (u.t <= t)][["host", "device"]].to_numpy()))
    nxt = set(map(tuple, u[(u.t > t) & (u.t <= t + DAY)][["host", "device"]].to_numpy())) if truth else set()
    racks = []
    for (ri, rack), cr in E.cards.groupby(["rack_i", "rack"]):
        nodes = []
        for (ni, node), cn in cr.groupby(["node_i", "node"]):
            cells = []
            for c in cn.itertuples():
                r = R.loc[c.card] if c.card in R.index else None
                cells.append({"card": int(c.card), "name": c.name, "src": c.source, "host": c.ohost,
                              "p": {h: (float(r[f"p_{h}"]) if r is not None else 0.0) for h in HZ},
                              "lvl": {h: (int(r[f"lvl_{h}"]) if r is not None else 0) for h in HZ},
                              "uce_prev": (c.node, c.device) in prev, "uce_next": (c.node, c.device) in nxt})
            nodes.append({"node": node, "cards": cells})
        racks.append({"rack": rack, "nodes": nodes})
    return racks


@app.route("/api/supernode")
def supernode():
    t = int(request.args.get("t", E.sim_t))
    truth = t < E.sim_t
    return jsonify({"t": t, "time": t_iso(t), "racks": grid(risk_at(t), t, truth=truth), "truth": truth})


@app.route("/api/alarms")
def alarms():
    h = request.args.get("h", "1d")
    with E.lock:
        if E.current is None:
            return jsonify({"t": E.sim_t, "alarms": [], "nodes": []})
        R, A = E.current.copy(), dict(E.attrib)
    R = R[R[f"lvl_{h}"] > 0].sort_values(f"p_{h}", ascending=False).merge(E.cards, on=["card", "source"])
    out = []
    for r in R.itertuples():
        a = A.get(int(r.card), {})
        out.append({"card": int(r.card), "name": r.name, "rack": r.rack, "node": r.node,
                    "p": {x: float(getattr(r, f"p_{x}")) for x in HZ}, "lvl": {x: int(getattr(r, f"lvl_{x}")) for x in HZ},
                    "category": a.get("category", ""), "root_cause": a.get("root_cause", ""), "reason": reason_zh(r.reason),
                    "model": SOURCE_ZH[r.source]})
    nodes = R.groupby("node").card.count().sort_values(ascending=False)
    return jsonify({"t": E.sim_t, "alarms": out, "nodes": [{"node": n, "cards": int(k)} for n, k in nodes.items()]})


@app.route("/api/card/<int:card>")
def card(card):
    t = int(request.args.get("t", E.sim_t))
    c = E.cards[E.cards.card == card].iloc[0]
    after = DAY if t < E.sim_t else 0
    rs = E.risk_series(card, t - 7 * DAY, min(t + after, E.sim_t))
    src = c.source
    thr = E.thr[src or "log"]
    x = E.explain(card, t)
    x["fta"]["gates"] = x["fta"]["gates"]
    return jsonify({
        "card": card, "name": c["name"], "rack": c.rack, "node": c.node, "t": t, "time": t_iso(t),
        "model": SOURCE_ZH[src], "thresholds": thr, "host": c.get("ohost", ""),
        "risk_series": {"t": rs.s.astype(int).tolist(), **{h: rs[f"p_{h}"].round(6).tolist() for h in HZ}},
        "events": E.card_events(card, t - 7 * DAY, min(t + after, E.sim_t)),
        "telemetry": E.card_telemetry(card, t - DAY, t),
        "now": {"risk": x["risk"], "level": x["level"], "reason": reason_zh(x["reason"]), "fta": x["fta"],
                "tel": x["telemetry"]}})


@app.route("/api/scans")
def scans():
    return jsonify([{**s, "time": t_iso(s["t"])} for s in E.scans[-240:]])


@app.route("/api/backtrack")
def backtrack():
    E = REPLAY                         # validation on open data, whichever source the live pages show
    if E is None:
        return jsonify({"available": False})
    eff = E.meta["effectiveness"]
    ep = E.episodes[(E.episodes.t > DEMO_START) & (E.episodes.t <= max(LIVE_START, E.sim_t))].merge(
        E.cards[["card", "name", "rack", "node", "source"]], on=["card", "source"])
    H = E.history
    rows = []
    for e in ep.sort_values("t").itertuples():
        th = E.thr[e.source]["1d"]["high"]
        w = H[(H.card == e.card) & (H.s >= e.t - DAY) & (H.s < e.t)]
        al = w[w.p_1d > th]
        rows.append({"card": int(e.card), "name": e.name, "t": int(e.t), "time": t_iso(e.t), "first": bool(e.first),
                     "model": SOURCE_ZH[e.source], "max_p1d": float(w.p_1d.max()) if len(w) else None,
                     "lead_h": float((e.t - al.s.min()) / HOUR) if len(al) else None,
                     "caught_1h": bool(len(w[(w.s >= e.t - HOUR) & (w.p_1h > E.thr[e.source]["1h"]["high"])])),
                     "caught_4h": bool(len(w[(w.s >= e.t - 4 * HOUR) & (w.p_4h > E.thr[e.source]["4h"]["high"])])),
                     "caught_1d": bool(len(al))})
    return jsonify({"effectiveness": eff, "episodes": rows, "window": [t_iso(DEMO_START), t_iso(LIVE_START)],
                    "t0": DEMO_START, "t1": LIVE_START, "models": {k: SOURCE_ZH[k] for k in eff},
                    "cards": {k: int((E.cards.source == k).sum()) for k in eff}})


@app.route("/api/collect")
def collect():
    from common import SPOOL
    E = REPLAY or CLUSTER
    m = META["models"]
    spool = sorted(SPOOL.glob("*.jsonl")) if SPOOL.exists() else []
    rep = {}
    for src, v in m.items():
        f = v["report"].get("fields", {})
        rep[src] = {"model": SOURCE_ZH[src], "modules": [x.split(" (")[0] for x in v["report"].get("modules", [])],
                    "fields": {"设备": f.get("device"), "节点": f.get("host"), "时间": f.get("time"),
                               "错误类型": f.get("error_code"), "地址层级": ", ".join((f.get("ce_location") or {}).keys()) or "无",
                               "事件类别": [cause_zh(c.replace("bg_", "")) + ("（背景）" if c.startswith("bg_") else "")
                                        for c in f.get("event_codes", []) if c != "other"]},
                    "info": {k: v["info"].get(k) for k in ("n_features", "groups", "fusion_weight_ai", "val_positives",
                                                            "tree_params", "backend")},
                    "thresholds": E.thr[src]}
    from sources import LOCAL_FIELDS
    local = {"available": E.local.ok, "host": E.local.host if E.local.ok else "",
             "fields": [{"key": k, "label": lab} for _, k, lab in LOCAL_FIELDS], "series": E.local_series[-120:]}
    return jsonify({"probes": E.probes, "mode": "开源数据回放" + (" + 本机实时采集" if E.local.ok else ""), "local": local,
                    "replay_sources": [] if REPLAY is None else [
                        {"name": "错误日志", "from": "OLCF Summit 超算公开错误日志（逐卡、真实时间）", "cards": int((E.cards.source == "log").sum())},
                        {"name": "显存 RAS 日志", "from": "Calchas 公开 HBM 错误日志（地址级 CE/UER/UEO，时间平移）",
                         "cards": int((E.cards.source == "ce").sum())},
                        {"name": "遥测与作业日志", "from": "OLCF Summit 公开 10 秒遥测（2020-01，按日回放到各节点）", "cards": N_CARDS}],
                    "spool": {"files": len(spool), "latest": spool[-1].name if spool else "", **E.ingest_stats},
                    "telemetry_fields": {"核心温度": "core_temp", "显存温度": "mem_temp", "功率": "power"},
                    "models": rep, "fta_weights": E.meta["fta_weights"],
                    "fta_basic": [{"id": b[0], "gate": fta.GATES[b[1]]["name"], "name": b[2],
                                   "source": "模型特征" if b[3] == "model" else "实时遥测", "prior": b[6]} for b in fta.BASIC],
                    "gates": {k: v["name"] for k, v in fta.GATES.items()}, "built": E.meta.get("built")})


@app.route("/api/control", methods=["POST"])
def control():
    d = request.get_json(force=True)
    a = d.get("action")
    if a == "play":
        E.playing = True
    elif a == "pause":
        E.playing = False
    elif a == "step":
        E.step()                        # replay: advance one hour; cluster: scan now
    elif a == "speed":
        E.tick_s = float(max(0.5, min(30, d.get("value", 3))))
    return jsonify({"playing": E.playing, "tick_s": E.tick_s})


def _clean(o):
    """JSON-safe: NaN / inf -> None."""
    import math
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


DEMO_VERSION = "1.5.0"


@app.route("/api/version")
def version_api():
    import uce_risk
    m = META
    info = {k: m["models"][k]["info"] for k in ("log", "ce")}
    items = [
        ("产品", CFG["title"]),
        ("原型版本", DEMO_VERSION),
        ("算法库", f"uce_risk {getattr(uce_risk, '__version__', '')}（AI + 规则混合，分段风险模型）"),
        ("树模型", "LightGBM" if info["log"].get("backend") == "lgbm" else str(info["log"].get("backend"))),
        ("模型构建时间", str(m.get("built", ""))),
        ("运行日志模型", f"{info['log']['n_features']} 个特征；训练数据截至 {t_iso(DEMO_START)}"),
        ("显存 RAS 模型", f"{info['ce']['n_features']} 个特征"),
        ("预测时间窗", "1 小时 / 4 小时 / 24 小时"),
        ("监测对象", f"超节点 {CFG['supernode']}：{CFG['topology']['racks']} 机柜 × {CFG['topology']['nodes_per_rack']} 节点 × "
                    f"{CFG['topology']['cards_per_node']} 计算卡"),
        ("回放数据", "OLCF Summit 公开错误日志与遥测、Calchas 公开 HBM 错误日志"),
        ("集群采集", f"采集频率 {CLUSTER.settings['interval_s']} 秒，数据保留 {CLUSTER.settings['retention_days']} 天"),
    ]
    return jsonify({"product": CFG["title"], "version": DEMO_VERSION, "items": items})


@app.route("/api/advantages")
def advantages_api():
    import advantages
    m = META
    return jsonify(_clean({"comparison": advantages.comparison(), "ablation": advantages.ablation(m),
                           "models": {k: v for k, v in SOURCE_ZH.items() if k},
                           "cards": {k: int((REPLAY.cards.source == k).sum()) if REPLAY is not None else 0 for k in ("log", "ce")},
                           "window": [t_iso(DEMO_START), t_iso(LIVE_START)],
                           "fusion": {k: m["models"][k]["info"].get("fusion_weight_ai") for k in ("log", "ce")}}))


# ----------------------------------------------------------------------------- cluster collection API
@app.route("/api/source", methods=["GET", "POST"])
def source():
    global E
    if request.method == "POST":
        want = request.get_json(force=True).get("source")
        E = CLUSTER if (want == "cluster" or REPLAY is None) else REPLAY
    return jsonify({"source": "cluster" if is_cluster() else "replay"})


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    s = CLUSTER.settings
    if request.method == "POST":
        d = request.get_json(force=True)
        if "interval_s" in d:
            s["interval_s"] = int(max(10, min(3600, float(d["interval_s"]))))
        if "retention_days" in d:
            s["retention_days"] = int(max(1, min(3650, float(d["retention_days"]))))
        save_settings(s)
        CLUSTER.next_scan = min(CLUSTER.next_scan, __import__("time").time() + s["interval_s"])
    return jsonify({**s, "storage": CLUSTER.storage_estimate()})


@app.route("/api/agent/register", methods=["POST"])
def agent_register():
    d = request.get_json(force=True)
    a = CLUSTER.register(d["host"], d.get("capabilities") or {}, d.get("interval_s"), d.get("agent_version"))
    return jsonify({"ok": True, "node": a.get("node"), "interval_s": CLUSTER.settings["interval_s"]})


@app.route("/api/agent/config")
def agent_config():
    s = CLUSTER.settings
    return jsonify({"interval_s": s["interval_s"], "retention_days": s["retention_days"]})


@app.route("/api/ingest", methods=["POST"])
def ingest():
    d = request.get_json(force=True)
    n = CLUSTER.ingest(d["host"], d.get("records", []))
    return jsonify({"ok": True, "accepted": n})


@app.route("/api/cluster")
def cluster_info():
    host = request.host_url.rstrip("/")
    return jsonify({"agents": CLUSTER.agent_rows(), "settings": CLUSTER.settings, "storage": CLUSTER.storage_estimate(),
                    "ingest": CLUSTER.ingest_stats, "scans": len(CLUSTER.scans),
                    "deploy": f"curl -fsSL {host}/agent/install.sh | sudo bash",
                    "last_scan": CLUSTER.scans[-1] if CLUSTER.scans else None})


def _agent_files():
    here = Path(__file__).resolve().parent
    for d in (here / "collector", here, here.parent / "collectors" / "gpu_ecc"):
        if (d / "gpu_ecc_collector.py").exists():
            return {"agent.py": here / "agent.py", "gpu_ecc_collector.py": d / "gpu_ecc_collector.py"}
    return {"agent.py": here / "agent.py"}


@app.route("/agent/<name>")
def agent_file(name):
    """node agent served by the monitoring server itself, so intranet nodes need no copy and no internet:
        curl -fsSL http://<server>:<port>/agent/install.sh | sudo bash"""
    from flask import Response
    if name == "install.sh":
        host = request.host_url.rstrip("/")
        return Response(AGENT_INSTALL.replace("__SERVER__", host), mimetype="text/x-shellscript")
    f = _agent_files().get(name)
    if f is None:
        return jsonify({"error": "not found"}), 404
    return Response(f.read_text(encoding="utf-8"), mimetype="text/x-python")


AGENT_INSTALL = r"""#!/usr/bin/env bash
# 计算节点 Agent 一键安装（由监测服务 __SERVER__ 生成）。只依赖 Python 3 标准库与厂商管理工具。
#   curl -fsSL __SERVER__/agent/install.sh | sudo bash
#   卸载：curl -fsSL __SERVER__/agent/install.sh | sudo bash -s -- --uninstall
set -euo pipefail
SERVER="__SERVER__"
DIR=/opt/supernode-agent
DATA=/var/lib/supernode-agent
[ "$(id -u)" -eq 0 ] || { echo "需要 root：curl -fsSL $SERVER/agent/install.sh | sudo bash"; exit 1; }
if [ "${1:-}" = "--uninstall" ]; then
  systemctl disable --now supernode-agent 2>/dev/null || true
  rm -f /etc/systemd/system/supernode-agent.service; systemctl daemon-reload
  rm -rf "$DIR" "$DATA"; echo "已卸载 supernode-agent"; exit 0
fi
command -v python3 >/dev/null || { echo "未找到 python3"; exit 1; }
command -v systemctl >/dev/null || { echo "需要 systemd"; exit 1; }
NVSMI="$(command -v nvidia-smi || true)"
for c in /usr/bin/nvidia-smi /usr/local/nvidia/bin/nvidia-smi /usr/lib/wsl/lib/nvidia-smi; do
  [ -z "$NVSMI" ] && [ -x "$c" ] && NVSMI="$c"
done
[ -n "$NVSMI" ] || { echo "提示：未找到 nvidia-smi，只能采集部分指标"; NVSMI=nvidia-smi; }
fetch() { if command -v curl >/dev/null; then curl -fsSL "$1" -o "$2"
          else python3 -c "import sys,urllib.request; open(sys.argv[2],'wb').write(urllib.request.urlopen(sys.argv[1],timeout=30).read())" "$1" "$2"; fi; }
mkdir -p "$DIR/collector" "$DATA"
fetch "$SERVER/agent/agent.py" "$DIR/agent.py"
fetch "$SERVER/agent/gpu_ecc_collector.py" "$DIR/collector/gpu_ecc_collector.py"
python3 "$DIR/agent.py" --print-capabilities --base-dir "$DATA" --nvidia-smi "$NVSMI" > "$DATA/capabilities.json" 2>/dev/null || true
cat > /etc/systemd/system/supernode-agent.service <<EOF
[Unit]
Description=Supernode node agent (compute-card telemetry / error collection)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$(command -v python3) $DIR/agent.py --server $SERVER --base-dir $DATA --nvidia-smi $NVSMI
Restart=always
RestartSec=10
User=root
Nice=10
CPUQuota=20%
MemoryMax=512M

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable supernode-agent >/dev/null 2>&1
systemctl restart supernode-agent
sleep 2
if systemctl is-active --quiet supernode-agent; then
  echo "✔ supernode-agent 已启动，向 $SERVER 上报；几十秒后在页面“数据采集与处理”中可见本节点 $(hostname)"
else
  journalctl -u supernode-agent -n 20 --no-pager; exit 1
fi
"""


def start_local_agent(port):
    """the server's own machine is a node too: probe it and collect through the same agent path."""
    import shutil
    import threading
    smi = shutil.which("nvidia-smi") or next((c for c in ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi",
                                                          "/usr/lib/wsl/lib/nvidia-smi") if Path(c).exists()), None)
    if not CFG.get("local_agent", True) or not smi:
        return
    from agent import Agent
    from common import DATA
    a = Agent(f"http://127.0.0.1:{port}", str(DATA / "local_agent"), nvidia_smi=smi)
    threading.Thread(target=a.run, name="local-agent", daemon=True).start()


def main():
    global E, REPLAY, CLUSTER, META
    import json
    import os

    import joblib
    from common import MODEL_DIR, replay_available
    models = joblib.load(MODEL_DIR / "models.joblib")
    META = json.loads((MODEL_DIR / "build_meta.json").read_text(encoding="utf-8"))
    use_replay = CFG.get("replay", True) and replay_available()
    REPLAY = Engine() if use_replay else None
    CLUSTER = ClusterEngine(models, META)
    E = CLUSTER if (REPLAY is None or CLUSTER.settings.get("default_source") == "cluster") else REPLAY
    if REPLAY is not None:
        REPLAY.start()
    CLUSTER.start()
    print(f"[supernode] models {MODEL_DIR} | replay {'on' if REPLAY is not None else 'off (cluster only)'} | "
          f"data {CLUSTER.settings} ", flush=True)
    port = int(os.environ.get("PORT", CFG["server"]["port"]))
    start_local_agent(port)
    host = os.environ.get("HOST", CFG["server"]["host"])
    for u in access_urls(host, port):
        print(f"[supernode] 访问地址 {u}", flush=True)
    try:
        from waitress import serve          # production WSGI server when installed (requirements.txt)
    except ImportError:
        app.run(host=host, port=port, debug=False, threaded=True)
    else:
        serve(app, host=host, port=port, threads=int(os.environ.get("THREADS", 16)), ident="supernode")


def access_urls(host, port):
    """the addresses intranet users can type into a browser."""
    import socket
    if host not in ("0.0.0.0", "", "::"):
        return [f"http://{host}:{port}"]
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:   # no packet is sent; picks the default route
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    ips = sorted(i for i in ips if not i.startswith("127.")) or ["127.0.0.1"]
    return [f"http://{i}:{port}" for i in ips]


if __name__ == "__main__":
    main()
