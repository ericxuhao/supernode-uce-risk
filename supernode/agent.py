"""Node agent: automatic capability detection + scheduled collection + push to the monitoring server.

Runs on every compute node (e.g. an 8-card A100 server). It wraps the field-probing collector in
collectors/gpu_ecc (ECC / row-remap counters, temperatures, power, clocks, PCIe / interconnect errors,
kernel error log, counter-change events) and adds:

  * capability detection: which metrics this driver/device supports, whether the kernel error log,
    device-event listener and telemetry exporter are reachable -> reported to the server, which decides
    from these fields which model inputs exist;
  * a user-set collection interval (default 300 s, pulled from the server; slower tiers scale with it) to
    bound storage;
  * upload of new samples / events to POST /api/ingest; data stays on local disk until accepted, so a
    server outage loses nothing.

    python supernode/agent.py --server http://<server>:8765            # on each node
    python supernode/agent.py --print-capabilities                    # what this machine can collect
    python supernode/agent.py --server http://127.0.0.1:8765 --simulate 4   # 4 simulated 8-card nodes (test)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_here = Path(__file__).resolve().parent
# the collector core: next to this file in a release (collector/), or collectors/gpu_ecc in the repository
for _p in (_here / "collector", _here, ROOT / "collectors" / "gpu_ecc"):
    if (_p / "gpu_ecc_collector.py").exists():
        sys.path.insert(0, str(_p))
        break
import gpu_ecc_collector as gc  # noqa: E402

AGENT_VERSION = "1.0"
DEFAULT_INTERVAL = 300


def collector_cfg(base_dir: str, interval: float, nvidia_smi: str = "nvidia-smi") -> dict:
    cfg = dict(gc.DEFAULT_CONFIG)
    cfg.update(base_dir=base_dir, nvidia_smi=nvidia_smi, fast_interval_s=interval,
               slow_interval_s=max(2 * interval, 60), q_parse_interval_s=max(interval, 300),
               raw_q_interval_s=max(3600, 12 * interval), enable_nvml_events=False, snapshot_on_sbe=False,
               retention_days=30, min_free_pct=5.0, critical_free_pct=1.0)
    return cfg


# ----------------------------------------------------------------------------- capability detection
def detect_capabilities(collector: gc.Collector) -> dict:
    """field support of this driver / device plus reachable side channels."""
    c = gc.Collector.__new__(type(collector))
    c.__dict__.update({k: v for k, v in collector.__dict__.items()})
    c.unsupported, c.resolved = {}, {}
    c.detect_fields()
    supported = sorted({sp["id"] for tier in ("fast", "slow") for _, sp in c.resolved.get(tier, [])})
    devices = []
    rc, out, _ = c.smi("--query-gpu=index,uuid", "--format=csv,noheader,nounits")
    if rc == 0:
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 2 and p[0].isdigit():
                devices.append({"index": int(p[0]), "uuid": p[1]})
    # a probe query: fields the driver lists may still return N/A on this device
    na = set()
    for tier in ("fast", "slow"):
        specs = c.resolved.get(tier, [])
        if not specs or not devices:
            continue
        seen_ok = set()
        q = ",".join(["index"] + [f for f, _ in specs])
        rc, out, _ = c.smi("--query-gpu=" + q, "--format=csv,noheader,nounits")
        if rc != 0:
            continue
        for line in out.strip().splitlines():
            vals = [x.strip() for x in line.split(",")][1:]
            for (f, sp), v in zip(specs, vals):
                if gc.parse_value(v, sp["kind"])[0] is not None:
                    seen_ok.add(sp["id"])
        na |= {sp["id"] for _, sp in specs} - seen_ok       # N/A on every card of this node
    kernel = None
    if shutil.which("journalctl"):
        rc, _, _ = gc.run_cmd(["journalctl", "-k", "-n", "1", "--no-pager", "-q"], 10)
        kernel = "journalctl" if rc == 0 else None
    if not kernel and shutil.which("dmesg"):
        rc, _, _ = gc.run_cmd(["dmesg", "-T"], 10)
        kernel = "dmesg" if rc == 0 else None
    try:
        import pynvml  # noqa: F401
        nvml = True
    except Exception:
        nvml = False
    rc, out, _ = c.smi("nvlink", "-s") if devices else (1, "", "")
    return {"tool": bool(devices), "devices": devices,
            "metrics_supported": [m for m in supported if m not in na], "metrics_na_on_device": sorted(na),
            "metrics_unsupported": sorted(c.unsupported), "kernel_log": kernel, "device_events": nvml,
            "interconnect": rc == 0 and "Link" in (out or "")}


# ----------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, server: str | None, base_dir: str, nvidia_smi: str = "nvidia-smi", interval: float | None = None,
                 host: str | None = None, collector_cls=gc.Collector):
        self.server = server.rstrip("/") if server else None
        self.base_dir, self.nvidia_smi = base_dir, nvidia_smi
        self.interval = interval or DEFAULT_INTERVAL
        self.fixed_interval = interval is not None
        self.host = host or socket.gethostname()
        self.collector_cls = collector_cls
        self.stop = threading.Event()
        self.state_path = os.path.join(base_dir, "agent_state.json")
        os.makedirs(base_dir, exist_ok=True)
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"sent": {}}
        self.col = None
        self.caps = None
        self.stats = {"uploaded": 0, "upload_errors": 0, "last_upload": None}

    # -------------------------------------------------- server I/O
    def _post(self, path, payload, timeout=30):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.server + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path, timeout=10):
        with urllib.request.urlopen(self.server + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def pull_settings(self):
        if not self.server or self.fixed_interval:
            return
        try:
            s = self._get(f"/api/agent/config?host={urllib.request.quote(self.host)}")
            new = float(s.get("interval_s", self.interval))
            if new != self.interval:
                gc.LOG.info("collection interval %s -> %s s", self.interval, new)
                self.interval = new
                self.restart_collector()
        except Exception as e:
            gc.LOG.warning("settings pull failed: %s", e)

    def register(self):
        if self.server:
            try:
                self._post("/api/agent/register", {"host": self.host, "agent_version": AGENT_VERSION,
                                                   "interval_s": self.interval, "capabilities": self.caps})
            except Exception as e:
                gc.LOG.warning("register failed: %s", e)

    # -------------------------------------------------- collection
    def make_collector(self):
        c = self.collector_cls(collector_cfg(self.base_dir, self.interval, self.nvidia_smi))
        c.host = self.host
        return c

    def start_collector(self):
        self.col = self.make_collector()
        threading.Thread(target=self.col.run, name=f"collector-{self.host}", daemon=True).start()

    def restart_collector(self):
        if self.col:
            self.col.stop.set()
            time.sleep(1.5)
        self.start_collector()
        self.register()

    def new_records(self, limit=5000):
        """lines of samples (fast tier) and events not yet accepted by the server."""
        out, marks = [], {}
        files = sorted(glob.glob(os.path.join(self.base_dir, "samples", "*", "fast-*.jsonl*")) +
                       glob.glob(os.path.join(self.base_dir, "events", "*", "events-*.jsonl*")))
        for f in files:
            key = os.path.basename(f).replace(".gz", "")
            done = self.state["sent"].get(key, 0)
            opener = gzip.open if f.endswith(".gz") else open
            try:
                with opener(f, "rt", encoding="utf-8") as fh:
                    lines = fh.readlines()
            except (OSError, EOFError):
                continue
            new = [ln for ln in lines[done:] if ln.endswith("\n")]       # skip a line still being written
            take = new[: max(0, limit - len(out))]
            for ln in take:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
            if take:
                marks[key] = done + len(take)
            if len(out) >= limit:
                break
        return out, marks

    def upload(self):
        if not self.server:
            return
        while True:
            recs, marks = self.new_records()
            if not recs:
                return
            try:
                self._post("/api/ingest", {"host": self.host, "records": recs})
            except Exception as e:
                self.stats["upload_errors"] += 1
                gc.LOG.warning("upload failed (%d records kept on disk): %s", len(recs), e)
                return
            self.state["sent"].update(marks)
            self.stats["uploaded"] += len(recs)
            self.stats["last_upload"] = time.time()
            with open(self.state_path + ".tmp", "w") as f:
                json.dump(self.state, f)
            os.replace(self.state_path + ".tmp", self.state_path)

    def run(self):
        probe = self.make_collector()
        self.caps = detect_capabilities(probe)
        self.pull_settings() if self.server else None
        self.start_collector()
        self.register()
        last_settings = time.time()
        while not self.stop.is_set():
            self.upload()
            if time.time() - last_settings > 60:
                self.pull_settings()
                self.register()                 # heartbeat
                last_settings = time.time()
            self.stop.wait(5)
        if self.col:
            self.col.stop.set()


# ----------------------------------------------------------------------------- simulation (testing without hardware)
class SimulatedNodeCollector(gc.Collector):
    """an 8-card node with a full A100-class field set: drifting temperature / power, correctable-error
    counters, occasional row remapping, uncorrectable errors and kernel error lines. For testing only."""
    CARDS = 8

    def __init__(self, cfg, seed=0, risky=()):
        gc.Collector.__init__(self, cfg)
        import random
        self.rng = random.Random(seed)
        self.risky = set(risky)
        self.sbe = [0] * self.CARDS
        self.dbe = [0] * self.CARDS
        self.remap = [0] * self.CARDS
        self.names = set(gc.KEY_FIELDS)
        for specs in (gc.FAST_SPECS, gc.SLOW_SPECS, gc.INVENTORY_SPECS):
            for sp in specs:
                self.names.add(sp["fields"][0])

    def tick(self):
        for i in range(self.CARDS):
            hot = i in self.risky
            if self.rng.random() < (0.6 if hot else 0.02):
                self.sbe[i] += self.rng.randint(1, 40 if hot else 3)
            if hot and self.rng.random() < 0.08:
                self.remap[i] += 1
                self.q.put(("kernel", {"type": "xid", "xid": 63, "pci": f"00000000:{i + 7:02X}:00.0",
                                       "message": "row remapping event", "event_time": gc.iso(time.time())}))
            if hot and self.rng.random() < 0.03:
                self.dbe[i] += 1
                self.q.put(("kernel", {"type": "xid", "xid": 48, "pci": f"00000000:{i + 7:02X}:00.0",
                                       "message": "uncorrectable memory error", "event_time": gc.iso(time.time())}))

    def smi(self, *args):
        if args[0] == "--help-query-gpu":
            return 0, "\n".join('"%s"' % n for n in sorted(self.names)), ""
        if args[0].startswith("--query-gpu="):
            if "pci.bus_id" in args[0] and "temperature.gpu" in args[0]:
                self.tick()
            fields = args[0].split("=", 1)[1].split(",")
            rows = []
            for i in range(self.CARDS):
                hot = i in self.risky
                base = {"index": str(i), "uuid": f"DEV-{self.host}-{i}", "pci.bus_id": f"00000000:{i + 7:02X}:00.0",
                        "temperature.gpu": f"{55 + 3 * hot + self.rng.gauss(0, 2):.0f}",
                        "temperature.memory": f"{60 + 8 * hot + self.rng.gauss(0, 2):.0f}",
                        "power.draw": f"{300 + self.rng.gauss(0, 30):.1f}",
                        "utilization.gpu": f"{self.rng.randint(40, 100)}",
                        "ecc.errors.corrected.volatile.total": str(self.sbe[i]),
                        "ecc.errors.corrected.aggregate.total": str(self.sbe[i]),
                        "ecc.errors.uncorrected.volatile.total": str(self.dbe[i]),
                        "ecc.errors.uncorrected.aggregate.total": str(self.dbe[i]),
                        "remapped_rows.correctable": str(self.remap[i]), "remapped_rows.uncorrectable": str(self.dbe[i]),
                        "remapped_rows.pending": "No", "remapped_rows.failure": "No", "ecc.mode.current": "Enabled",
                        "name": "compute card", "serial": f"SN{i:04d}"}
                rows.append(", ".join(base.get(f, "0") for f in fields))
            return 0, "\n".join(rows) + "\n", ""
        if args[0] == "-q":
            return 0, "", ""
        if args[0] == "nvlink":
            return 0, "\n".join(f"GPU {i}: card\n\t Link 0: CRC Errors: 0" for i in range(self.CARDS)), ""
        if args[0].startswith("--query-compute-apps"):
            return 0, "", ""
        if args[0] == "--version":
            return 0, "fake", ""
        return 1, "", "unexpected"


def simulate(server: str, n: int, base: str, interval: float | None):
    agents = []
    for k in range(n):
        host = f"sim-node-{k + 1:02d}"
        risky = (k % 8, (k * 3 + 1) % 8) if k % 2 == 0 else ()

        def cls(cfg, k=k, risky=risky):
            return SimulatedNodeCollector(cfg, seed=k, risky=risky)
        a = Agent(server, os.path.join(base, host), interval=interval, host=host, collector_cls=cls)
        threading.Thread(target=a.run, name=host, daemon=True).start()
        agents.append(a)
    return agents


def main():
    ap = argparse.ArgumentParser(description="supernode monitoring node agent")
    ap.add_argument("--server", help="monitoring server, e.g. http://10.0.0.5:8765")
    ap.add_argument("--base-dir", default=os.environ.get("AGENT_BASE_DIR", os.path.join(os.path.expanduser("~"), ".supernode-agent")))
    ap.add_argument("--nvidia-smi", default="nvidia-smi", help="path of the vendor management tool")
    ap.add_argument("--interval", type=float, help="collection interval in seconds (default: server setting, 300)")
    ap.add_argument("--print-capabilities", action="store_true")
    ap.add_argument("--simulate", type=int, default=0, help="run N simulated 8-card nodes (testing)")
    a = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gc.LOG.setLevel(logging.INFO)
    if a.print_capabilities:
        ag = Agent(None, a.base_dir, a.nvidia_smi)
        print(json.dumps(detect_capabilities(ag.make_collector()), ensure_ascii=False, indent=1))
        return
    if a.simulate:
        simulate(a.server, a.simulate, os.path.join(a.base_dir, "simulated"), a.interval)
        while True:
            time.sleep(60)
    Agent(a.server, a.base_dir, a.nvidia_smi, a.interval).run()


if __name__ == "__main__":
    main()
