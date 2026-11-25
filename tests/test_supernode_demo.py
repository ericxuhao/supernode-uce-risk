"""End-to-end test of the supernode demo service (needs supernode/build.py to have been run).

Checks every API used by the page: topology (16 x 8 x 8 = 1,024 cards), valid and ordered risks
(p_1h <= p_4h <= p_24h), scheduled scans advancing the clock, alarms with fault-tree attribution, card
detail, history backtrack, collection page, and that no displayed text names a vendor device or code.
"""
import re
import sys
import time
from pathlib import Path

_R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R / ("supernode" if (_R / "supernode").exists() else "supernode_demo")))
import server  # noqa: E402
from engine import Engine  # noqa: E402

FORBIDDEN = re.compile(r"gpu|xid|nvidia|dcgm", re.I)
DISPLAY_KEYS = {"name", "rack", "node", "category", "root_cause", "reason", "model", "cause", "source", "detail", "probe",
                "evidence", "action", "label", "mode", "title", "from", "time", "modules", "事件类别"}


def displayed_strings(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from displayed_strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from displayed_strings(v, key)
    elif isinstance(obj, str) and key in DISPLAY_KEYS:
        yield key, obj


def check_text(js, where):
    bad = [(k, s) for k, s in displayed_strings(js) if FORBIDDEN.search(s)]
    assert not bad, (where, bad[:5])


def test_demo():
    t0 = time.time()
    server.E = server.REPLAY = Engine()
    from cluster import ClusterEngine
    server.META = server.REPLAY.meta
    server.CLUSTER = ClusterEngine(server.REPLAY.models, server.REPLAY.meta)
    print(f"engine ready in {time.time() - t0:.1f} s")
    c = server.app.test_client()
    s = c.get("/api/status").get_json()
    assert s["cards"] == 1024 and s["scans"] >= 1
    check_text(s, "status")

    g = c.get("/api/supernode").get_json()
    assert len(g["racks"]) == 16 and all(len(r["nodes"]) == 8 for r in g["racks"])
    cards = [x for r in g["racks"] for n in r["nodes"] for x in n["cards"]]
    assert len(cards) == 1024 and all(len(n["cards"]) == 8 for r in g["racks"] for n in r["nodes"])
    for x in cards:
        p = x["p"]
        assert 0 <= p["1h"] <= p["4h"] + 1e-9 <= p["1d"] + 2e-9 <= 1 + 2e-9, x
    check_text(g, "supernode")

    t_before = s["sim_t"]
    for _ in range(3):                                   # scheduled scans: collect -> process -> scan -> attribute
        c.post("/api/control", json={"action": "step"})
    s2 = c.get("/api/status").get_json()
    assert s2["sim_t"] == t_before + 3 * 3600 and s2["scans"] == s["scans"] + 3
    assert not s2["last_scan"].get("error"), s2["last_scan"]

    for h in ("1h", "4h", "1d"):
        a = c.get(f"/api/alarms?h={h}").get_json()
        check_text(a, f"alarms {h}")
        for x in a["alarms"]:
            assert x["lvl"][h] >= 1 and x["category"], x
    a = c.get("/api/alarms?h=1d").get_json()
    assert a["alarms"], "expected alarmed cards"
    top = a["alarms"][0]
    d = c.get(f"/api/card/{top['card']}").get_json()
    assert d["now"]["fta"]["gates"] and len(d["now"]["fta"]["gates"]) == 5
    assert d["risk_series"]["t"] and len(d["telemetry"]["t"]) == 288
    check_text(d, "card")

    b = c.get("/api/backtrack").get_json()
    assert b["episodes"] and set(b["effectiveness"]) == {"log", "ce"}
    for src, e in b["effectiveness"].items():
        for h in ("1h", "4h", "1d"):
            assert {"model", "rule"} <= set(e[h])
    check_text(b, "backtrack")
    ep = b["episodes"][0]
    d2 = c.get(f"/api/card/{ep['card']}?t={ep['t'] - 3600}").get_json()       # replay one past failure
    assert any(x["code"] == "MEMORY_UNCORRECTABLE_ERROR" for x in d2["events"])
    hist = c.get(f"/api/supernode?t={ep['t'] - 3600}").get_json()
    assert hist["truth"] and any(x["uce_next"] for r in hist["racks"] for n in r["nodes"] for x in n["cards"])

    k = c.get("/api/collect").get_json()
    assert k["spool"]["records"] > 0 and set(k["models"]) == {"log", "ce"}
    assert "CE spatial / fault modes" in " ".join(k["models"]["ce"]["modules"])          # field-driven modules
    assert "CE spatial / fault modes" not in " ".join(k["models"]["log"]["modules"])
    check_text({kk: v for kk, v in k.items() if kk != "local"}, "collect")
    return s2, a, b


if __name__ == "__main__":
    s, a, b = test_demo()
    print("clock", s["sim_time"], "| scans", s["scans"], "| 24h alarms", len(a["alarms"]), "| episodes", len(b["episodes"]))
    for src, e in b["effectiveness"].items():
        for h, v in e.items():
            m, r, sb = v["model"], v["rule"], v.get("model_same_budget", {})
            print(f"{src} {h}: model R {m['recall']:.2f} P {m['precision']:.3f} | same budget R {sb.get('recall', 0):.2f} "
                  f"P {sb.get('precision', 0):.3f} | rule R {r['recall']:.2f} P {r['precision']:.3f}")
    print("ok")
