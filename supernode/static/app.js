"use strict";
const $ = (s) => document.querySelector(s);
const api = (p, o) => fetch(p, o).then((r) => r.json());
const LV = ["正常", "关注", "高风险"];
let ST = { hzLive: "1d", hzBack: "1d", backT: null, horizons: [], lastScans: -1, back: null, collect: null };

const pct = (p) => (p == null ? "—" : p >= 0.1 ? (p * 100).toFixed(0) + "%" : p >= 0.001 ? (p * 100).toFixed(2) + "%" : (p * 100).toFixed(3) + "%");
const fmtT = (t) => {
  const d = new Date((t + 8 * 3600) * 1000);
  return d.toISOString().slice(0, 16).replace("T", " ");
};
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// ---------------------------------------------------------------- tabs / controls
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tabpane").forEach((x) => x.classList.toggle("active", x.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "back") loadBack();
  if (b.dataset.tab === "collect" || b.dataset.tab === "model") loadCollect();
});
const ctl = (action, value) => api("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, value }) }).then(refreshStatus);
$("#btnPlay").onclick = () => ctl("play");
$("#btnPause").onclick = () => ctl("pause");
$("#btnStep").onclick = () => ctl("step").then(refreshLive);
$("#speed").onchange = (e) => ctl("speed", +e.target.value);
document.querySelectorAll("#srcSeg button").forEach((b) => b.onclick = async () => {
  await api("/api/source", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: b.dataset.src }) });
  ST.lastScans = -1; ST.source = null; await refreshStatus(); refreshLive();
  if (document.querySelector("#tab-collect").classList.contains("active")) loadCollect();
});
$("#dClose").onclick = () => $("#drawer").classList.add("hidden");
$("#btnHelp").onclick = () => { $("#help").classList.remove("hidden"); loadHelp(); };
$("#helpClose").onclick = () => $("#help").classList.add("hidden");
$("#help").onclick = (e) => { if (e.target.id === "help") $("#help").classList.add("hidden"); };
document.querySelectorAll(".htab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".htab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".hpane").forEach((x) => x.classList.toggle("active", x.id === "h-" + b.dataset.h));
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { $("#help").classList.add("hidden"); $("#drawer").classList.add("hidden"); } });
let helpLoaded = false;
async function loadHelp() {
  if (helpLoaded) return;
  helpLoaded = true;
  loadAdv();
  const v = await api("/api/version");
  $("#helpSub").textContent = `${v.product} · 版本 ${v.version}`;
  $("#verInfo").innerHTML = `<table>${v.items.map(([k, x]) => `<tr><td style="width:180px" class="muted">${esc(k)}</td><td>${esc(x)}</td></tr>`).join("")}</table>`;
}
$("#drawer").onclick = (e) => { if (e.target.id === "drawer") $("#drawer").classList.add("hidden"); };

function seg(el, key, onchange) {
  el.innerHTML = ST.horizons.map((h) => `<button data-h="${h.key}" class="${ST[key] === h.key ? "on" : ""}">${h.label}</button>`).join("");
  el.querySelectorAll("button").forEach((b) => b.onclick = () => { ST[key] = b.dataset.h; seg(el, key, onchange); onchange(); });
}

// ---------------------------------------------------------------- status
async function refreshStatus() {
  const s = await api("/api/status");
  $("#title").textContent = s.title; document.title = s.title;
  $("#sn").textContent = "超节点 " + s.supernode;
  $("#mode").textContent = "数据来源：" + s.mode;
  $("#simtime").textContent = fmtT(s.sim_t);
  const pill = $("#runpill");
  pill.textContent = s.playing ? "监测中" : "已暂停"; pill.className = "pill " + (s.playing ? "run" : "stop");
  const ls = s.last_scan || {};
  const cl = s.source === "cluster";
  document.querySelector('#srcSeg [data-src="replay"]').style.display = s.replay_available ? "" : "none";
  document.querySelector('.tab[data-tab="back"]').style.display = s.replay_available ? "" : "none";
  document.querySelectorAll("#srcSeg button").forEach((b) => b.classList.toggle("on", b.dataset.src === s.source));
  $("#speedBox").style.display = cl ? "none" : "";
  $("#clusterInfo").style.display = cl ? "" : "none";
  $("#btnStep").title = cl ? "立即扫描" : "单步扫描";
  $(".clock-label").textContent = cl ? "当前时间（实时）" : "监测时钟（回放）";
  if (cl) {
    const m = Math.floor(s.cluster.next_scan_in / 60), sec = s.cluster.next_scan_in % 60;
    $("#clusterInfo").textContent = `采集频率 ${s.cluster.interval_s >= 60 ? s.cluster.interval_s / 60 + " 分钟" : s.cluster.interval_s + " 秒"} · 下次扫描 ${m}:${String(sec).padStart(2, "0")}`;
    $("#scaninfo").textContent = ls.error ? "扫描出错：" + ls.error : s.cluster.cards ? `第 ${s.scans} 次扫描 · 本次耗时 ${ls.seconds ?? "—"} 秒` : "等待节点接入…";
  } else {
    $("#scaninfo").textContent = ls.error ? "扫描出错：" + ls.error : `第 ${s.scans} 次扫描 · 每 ${s.tick_s} 秒推进 1 小时 · 本次耗时 ${ls.seconds ?? "—"} 秒`;
  }
  if (ST.source !== s.source) { ST.source = s.source; ST.lastScans = -1; }
  if (!ST.horizons.length) {
    ST.horizons = s.horizons;
    seg($("#hzLive"), "hzLive", refreshLive);
    seg($("#hzBack"), "hzBack", () => loadSnapshot(ST.backT));
  }
  if (s.scans !== ST.lastScans) { ST.lastScans = s.scans; refreshLive(); }
  return s;
}

// ---------------------------------------------------------------- live
function drawGrid(el, data, hz, opts = {}) {
  el.innerHTML = data.racks.map((r) => {
    let bad = 0;
    const nodes = r.nodes.map((n) => {
      const alarm = n.cards.some((c) => c.lvl[hz] === 2);
      const cells = n.cards.map((c) => {
        const l = c.lvl[hz]; if (l === 2) bad++;
        const off = !c.src;
        const cls = ["cell", off ? "off" : "l" + l, c.uce_prev ? "uce" : "", opts.truth && c.uce_next ? "next" : "", c.src === "ce" ? "ce" : ""].join(" ");
        const tip = off ? `${c.name}  未接入` : `${c.name}${c.host && ST.source === "cluster" ? "（" + c.host + "）" : ""}  1小时 ${pct(c.p["1h"])} · 4小时 ${pct(c.p["4h"])} · 24小时 ${pct(c.p["1d"])}${c.src === "ce" ? "（显存 RAS 模型）" : ""}`;
        return `<div class="${cls}" data-card="${c.card}" data-off="${off ? 1 : 0}" title="${esc(tip)}"></div>`;
      }).join("");
      return `<div class="node ${alarm ? "alarm" : ""}"><span class="nl">${n.node.slice(-2)}</span>${cells}</div>`;
    }).join("");
    return `<div class="rack"><div class="rack-h"><b>机柜 ${r.rack}</b><span class="${bad ? "bad" : "muted"}">${bad ? bad + " 张高风险" : "正常"}</span></div>${nodes}</div>`;
  }).join("");
  el.querySelectorAll(".cell").forEach((c) => c.onclick = () => { if (c.dataset.off !== "1") openCard(+c.dataset.card, opts.t); });
}

async function refreshLive() {
  const [g, a, sc] = await Promise.all([api("/api/supernode"), api("/api/alarms?h=" + ST.hzLive), api("/api/scans")]);
  drawGrid($("#gridLive"), g, ST.hzLive);
  const cards = g.racks.flatMap((r) => r.nodes.flatMap((n) => n.cards));
  const cnt = (h, l) => cards.filter((c) => c.lvl[h] === l).length;
  const last = sc[sc.length - 1] || {};
  const k = [
    [ST.source === "cluster" ? "已接入计算卡" : "计算卡总数", ST.source === "cluster" ? cards.filter((c) => c.src).length + " / " + cards.length : cards.length, ""], ["24小时高风险", cnt("1d", 2), "high"], ["4小时高风险", cnt("4h", 2), "high"],
    ["1小时高风险", cnt("1h", 2), "high"], ["关注（24小时）", cnt("1d", 1), "watch"],
    ["近24小时不可纠正错误", cards.filter((c) => c.uce_prev).length, ""], ["本次扫描耗时", (last.seconds ?? "—") + " 秒", ""]];
  $("#kpis").innerHTML = k.map(([l, v, c]) => `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  $("#alarmCount").textContent = `${a.alarms.length} 张（${ST.horizons.find((h) => h.key === ST.hzLive)?.label}）`;
  $("#alarmNodes").innerHTML = a.nodes.slice(0, 12).map((n) => `<span>节点 ${n.node} · ${n.cards}</span>`).join("");
  $("#alarmList").innerHTML = a.alarms.slice(0, 80).map((x) => `
    <div class="alarm" data-card="${x.card}">
      <div class="a1"><span class="nm">${x.name}</span><span class="ps">
        <span class="lv${x.lvl["1h"]}">1h ${pct(x.p["1h"])}</span><span class="lv${x.lvl["4h"]}">4h ${pct(x.p["4h"])}</span><span class="lv${x.lvl["1d"]}">24h ${pct(x.p["1d"])}</span></span></div>
      <div class="cat">${esc(x.category || "—")} · ${esc(x.root_cause || x.reason)}</div>
    </div>`).join("") || `<div class="muted" style="padding:10px">当前没有告警</div>`;
  $("#alarmList").querySelectorAll(".alarm").forEach((e) => e.onclick = () => openCard(+e.dataset.card));
  drawScanChart(sc);
}

function drawScanChart(sc) {
  const svg = $("#scanChart"); const W = 1200, H = 140, P = 24;
  if (!sc.length) { svg.innerHTML = ""; return; }
  const n = sc.length, mx = Math.max(1, ...sc.map((s) => s.high_1d || 0));
  const x = (i) => P + (W - 2 * P) * (n === 1 ? 0.5 : i / (n - 1)), y = (v) => H - 20 - (H - 40) * v / mx;
  const line = (k, col) => `<polyline fill="none" stroke="${col}" stroke-width="1.6" points="${sc.map((s, i) => x(i) + "," + y(s[k] || 0)).join(" ")}"/>`;
  const uce = sc.map((s, i) => s.uce ? `<line x1="${x(i)}" x2="${x(i)}" y1="${H - 18}" y2="14" stroke="#fff" stroke-dasharray="2,3" opacity=".5"/>` : "").join("");
  svg.innerHTML = `${uce}${line("high_1d", "#42a5f5")}${line("high_4h", "#ffa726")}${line("high_1h", "#ef5350")}
    <text x="${P}" y="12" fill="#8b98a9" font-size="11">高风险计算卡数（蓝 24小时 / 橙 4小时 / 红 1小时）；白色虚线 = 该小时发生不可纠正错误</text>
    <text x="${P}" y="${H - 4}" fill="#8b98a9" font-size="10">${sc[0].time}</text><text x="${W - P}" y="${H - 4}" fill="#8b98a9" font-size="10" text-anchor="end">${sc[n - 1].time}</text>
    <text x="4" y="${y(mx) + 4}" fill="#8b98a9" font-size="10">${mx}</text>`;
}

// ---------------------------------------------------------------- card drawer
function lineChart(svg, W, H, xs, series, opts = {}) {
  const P = { l: 46, r: opts.right ? 46 : 12, t: 10, b: 22 };
  if (!xs.length) { svg.innerHTML = `<text x="20" y="40" fill="#8b98a9">无数据</text>`; return; }
  const x0 = opts.x0 ?? Math.min(...xs), x1 = opts.x1 ?? Math.max(...xs);
  const X = (v) => P.l + (W - P.l - P.r) * (v - x0) / Math.max(1, x1 - x0);
  const left = series.filter((s) => !s.right), right = series.filter((s) => s.right);
  const rng = (ss) => { const v = ss.flatMap((s) => s.y.filter((q) => q != null && isFinite(q))); return v.length ? [opts.ymin ?? Math.min(...v), Math.max(...v)] : [0, 1]; };
  const [l0, l1] = opts.log ? [Math.log10(opts.floor), 0] : rng(left), [r0, r1] = rng(right);
  const Y = (v, r) => {
    if (opts.log) { const q = Math.log10(Math.max(v, opts.floor)); return P.t + (H - P.t - P.b) * (1 - (q - l0) / (l1 - l0)); }
    const [a, b] = r ? [r0, r1] : [l0, l1]; return P.t + (H - P.t - P.b) * (1 - (v - a) / Math.max(1e-9, b - a));
  };
  let g = "";
  if (opts.log) for (let e = Math.ceil(l0); e <= 0; e++) g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(10 ** e)}" y2="${Y(10 ** e)}" stroke="#222b36"/><text x="4" y="${Y(10 ** e) + 3}" fill="#8b98a9" font-size="10">${pct(10 ** e)}</text>`;
  else g += [l0, (l0 + l1) / 2, l1].map((v) => `<text x="4" y="${Y(v) + 3}" fill="#8b98a9" font-size="10">${v.toFixed(0)}</text>`).join("");
  if (right.length) g += [r0, r1].map((v) => `<text x="${W - 4}" y="${Y(v, true) + 3}" fill="#8b98a9" font-size="10" text-anchor="end">${v.toFixed(0)}</text>`).join("");
  (opts.bands || []).forEach((b) => { g += `<rect x="${X(b[0])}" width="${Math.max(1, X(b[1]) - X(b[0]))}" y="${H - P.b - 4}" height="4" fill="#455a64"/>`; });
  (opts.hlines || []).forEach((h) => { g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(h.v)}" y2="${Y(h.v)}" stroke="${h.c}" stroke-dasharray="4,4" opacity=".7"/>`; });
  (opts.vlines || []).forEach((v) => { g += `<line x1="${X(v.t)}" x2="${X(v.t)}" y1="${P.t}" y2="${H - P.b}" stroke="${v.c}" stroke-width="${v.w || 1.5}" ${v.dash ? 'stroke-dasharray="3,3"' : ""}/>` + (v.label ? `<text x="${X(v.t) + 3}" y="${P.t + 10}" fill="${v.c}" font-size="10">${v.label}</text>` : ""); });
  series.forEach((s) => {
    const pts = xs.map((t, i) => (s.y[i] == null ? null : `${X(t).toFixed(1)},${Y(s.y[i], s.right).toFixed(1)}`)).filter(Boolean).join(" ");
    g += `<polyline fill="none" stroke="${s.c}" stroke-width="${s.w || 1.5}" points="${pts}"/>`;
  });
  g += `<text x="${P.l}" y="${H - 6}" fill="#8b98a9" font-size="10">${fmtT(x0)}</text><text x="${W - P.r}" y="${H - 6}" fill="#8b98a9" font-size="10" text-anchor="end">${fmtT(x1)}</text>`;
  svg.innerHTML = g;
}

function ftaTree(f) {
  const gates = f.gates.map((g) => {
    const hot = g.p > 0;
    const be = g.events.map((e) => `<div class="be ${e.active ? "on" : ""} ${e.active && e.name === f.root_cause ? "root" : ""}" title="P(24小时内不可纠正错误 | 该事件)，依据：${e.basis}${e.rr ? "，相对风险 RR=" + e.rr : ""}">
      ${esc(e.name)}<span class="af">${pct(e.p)}${e.rr ? " · RR " + e.rr : ""}</span>${e.active ? `<span class="ev">${esc(e.evidence)}</span>` : ""}</div>`).join("");
    return `<div class="gate ${hot ? "hot" : ""}"><div class="gh"><span>${esc(g.name)}</span><span>${pct(g.p)}</span></div><div class="gate-op">或门 · P(24小时内不可纠正错误 | 事件) · 相对风险</div>${be}</div>`;
  }).join("");
  return `<div class="tree"><div class="tree-top">顶事件：计算卡不可纠正错误 · 前兆证据给出的 24 小时概率 ${pct(f.top)}（全体基线 ${pct(f.base)}） · 故障类别：${esc(f.category)} · 根因：${esc(f.root_cause)}</div>
    <div class="gates">${gates}</div><div class="note">处置建议：${esc(f.action)}</div></div>`;
}

async function openCard(card, t) {
  $("#drawer").classList.remove("hidden");
  $("#dTitle").textContent = "加载中…"; $("#dFta").innerHTML = ""; $("#dEvents").innerHTML = "";
  const d = await api(`/api/card/${card}` + (t ? `?t=${t}` : ""));
  $("#dTitle").textContent = `计算卡 ${d.name}`;
  $("#dSub").textContent = `机柜 ${d.rack} · 节点 ${d.node}${d.host && ST.source === "cluster" ? "（" + d.host + "）" : ""} · ${d.model} · 评估时刻 ${d.time}`;
  const n = d.now;
  $("#dKpis").innerHTML = ST.horizons.map((h) => `<div class="kpi ${n.level[h.key] === 2 ? "high" : n.level[h.key] === 1 ? "watch" : ""}">
      <div class="v">${pct(n.risk[h.key])}</div><div class="l">未来${h.label}风险 · ${LV[n.level[h.key]]}（高风险阈值 ${pct(d.thresholds[h.key].high)}）</div></div>`).join("")
    + `<div class="kpi"><div class="v" style="font-size:15px">${esc(n.reason)}</div><div class="l">规则评分卡主要依据</div></div>`;
  const rs = d.risk_series, uce = d.events.filter((e) => e.code === "MEMORY_UNCORRECTABLE_ERROR");
  lineChart($("#dRisk"), 900, 200, rs.t, [{ y: rs["1d"], c: "#42a5f5" }, { y: rs["4h"], c: "#ffa726" }, { y: rs["1h"], c: "#ef5350" }],
    { log: true, floor: 1e-5, x0: d.t - 7 * 86400, x1: Math.max(d.t, rs.t[rs.t.length - 1] || d.t),
      hlines: [{ v: d.thresholds["1d"].high, c: "#42a5f5" }],
      vlines: [...uce.map((e) => ({ t: e.t, c: "#ffffff", w: 2, label: "不可纠正错误" })), { t: d.t, c: "#b388ff", dash: true, label: "评估时刻" }] });
  $("#dCat").textContent = n.fta.category;
  $("#dFta").innerHTML = ftaTree(n.fta);
  const tl = d.telemetry, bands = [];
  for (let i = 0; i < tl.t.length; i++) if (tl.busy[i]) bands.push([tl.t[i] - 300, tl.t[i]]);
  lineChart($("#dTel"), 600, 170, tl.t, [{ y: tl.core_temp, c: "#ffa726" }, { y: tl.mem_temp, c: "#ef5350" }, { y: tl.power, c: "#42a5f5", right: true }], { right: true, bands });
  $("#dEvents").innerHTML = d.events.slice().reverse().map((e) => `<div class="alarm" style="cursor:default">
      <div class="a1"><span class="${e.severity === "critical" ? "lv2" : e.severity === "major" ? "lv1" : "muted"}">${esc(e.cause)}${e.node_level ? "（节点）" : ""}</span><span class="muted mono">${fmtT(e.t)}</span></div></div>`).join("") || `<div class="muted">无事件</div>`;
}

// ---------------------------------------------------------------- backtrack
async function loadBack() {
  if (!ST.back) ST.back = await api("/api/backtrack");
  const b = ST.back;
  $("#backWindow").textContent = `${b.window[0]} 至 ${b.window[1]}`;
  const P = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%"), Hh = (v) => (v == null ? "—" : v.toFixed(1) + " 小时");
  const cmp = (a, b) => `<td class="num ${(a ?? 0) > (b ?? 0) ? "better" : ""}">${P(a)}</td><td class="num">${P(b)}</td>`;
  const rowsFor = (r) => ST.horizons.map((h) => {
    const m = r[h.key].model, q = r[h.key].rule, e = r[h.key].model_same_budget || {};
    return `<tr><td>${h.label}</td><td class="num">${P(m.recall)}</td><td class="num">${P(m.precision)}</td><td class="num">${Hh(m.median_lead_h)}</td>
      <td class="num">${m.alarm_card_days}</td>${cmp(e.recall, q.recall)}${cmp(e.precision, q.precision)}<td class="num">${q.alarm_card_days}</td>
      <td class="num">${P(m.recur_recall)} / ${P(q.recur_recall)}</td><td class="num">${P(m.first_recall)} / ${P(q.first_recall)}</td><td class="num">${m.caught}/${m.episodes}</td></tr>`;
  }).join("");
  // horizon-matched rule: hardware-type error in the previous H hours, vs the model at the same alarm volume
  const winRows = (r) => ST.horizons.map((h) => {
    const w = r[h.key].rule_window, e = r[h.key].model_same_budget_window || {};
    return w ? `<tr><td>${h.label}</td>${cmp(e.recall, w.recall)}${cmp(e.precision, w.precision)}<td class="num">${w.alarm_card_days}</td>
      <td class="num">${e.caught ?? "—"} / ${w.caught}</td></tr>` : "";
  }).join("");
  $("#effTables").innerHTML = `<div class="eff">` + Object.entries(b.effectiveness).map(([src, r]) => `
    <div><h4>${b.models[src]} · ${b.cards[src]} 张计算卡</h4>
    <table><tr><th rowspan="2">时间窗</th><th colspan="4">算法（高风险阈值）</th><th colspan="5">同等告警量对比 · 规则 A：近 30 天出现过硬件类错误</th><th rowspan="2">复发召回<br>算法/规则 A</th><th rowspan="2">首次召回<br>算法/规则 A</th><th rowspan="2">命中/事件</th></tr>
      <tr><th>召回</th><th>精确率</th><th>提前量中位数</th><th>告警卡·天</th><th>算法召回</th><th>规则召回</th><th>算法精确率</th><th>规则精确率</th><th>告警卡·天</th></tr>${rowsFor(r)}</table>
    ${winRows(r) ? `<table class="wintab"><tr><th rowspan="2">时间窗</th><th colspan="5">同等告警量对比 · 规则 B：前 1 / 4 / 24 小时内出现硬件类错误（与时间窗对应）</th><th rowspan="2">命中数<br>算法/规则 B</th></tr>
      <tr><th>算法召回</th><th>规则召回</th><th>算法精确率</th><th>规则精确率</th><th>告警卡·天</th></tr>${winRows(r)}</table>` : ""}</div>`).join("") + `</div>`;
  $("#epCount").textContent = `${b.episodes.length} 次（含实时阶段已发生）`;
  $("#epList").innerHTML = b.episodes.slice().reverse().map((e) => `<div class="ep" data-card="${e.card}" data-t="${e.t}">
      <div><span class="mono">${e.name}</span><span class="badge ${e.first ? "first" : "rec"}">${e.first ? "首次" : "复发"}</span>
        <span class="badge ${e.caught_1d ? "hit" : "miss"}">${e.caught_1d ? "24h 预警命中" : "未预警"}</span>${e.caught_1h ? '<span class="badge hit">1h</span>' : ""}${e.caught_4h ? '<span class="badge hit">4h</span>' : ""}</div>
      <div class="muted mono">${e.time}</div>
      <div class="muted" style="grid-column:1/3">提前量 ${e.lead_h == null ? "—" : e.lead_h.toFixed(0) + " 小时"} · 前24小时最高风险 ${pct(e.max_p1d)}</div></div>`).join("");
  $("#epList").querySelectorAll(".ep").forEach((e) => e.onclick = () => { const t = +e.dataset.t; setBack(t - 3600); openCard(+e.dataset.card, t - 3600); });
  const s = $("#backSlider"); s.min = b.t0 / 3600; s.max = b.t1 / 3600 - 1; s.step = 1;
  if (ST.backT == null) setBack(b.t1 - 30 * 86400);
  s.oninput = () => setBack(+s.value * 3600, true);
  s.onchange = () => setBack(+s.value * 3600);
  $("#backPrev").onclick = () => setBack(ST.backT - 3600);
  $("#backNext").onclick = () => setBack(ST.backT + 3600);
}
let snapTimer = null;
function setBack(t, lazy) {
  ST.backT = t; $("#backSlider").value = t / 3600; $("#backTime").textContent = fmtT(t);
  clearTimeout(snapTimer); snapTimer = setTimeout(() => loadSnapshot(t), lazy ? 250 : 0);
}
async function loadSnapshot(t) {
  if (t == null) return;
  const g = await api("/api/supernode?t=" + t);
  drawGrid($("#gridBack"), g, ST.hzBack, { truth: true, t });
  const cards = g.racks.flatMap((r) => r.nodes.flatMap((n) => n.cards));
  const next = cards.filter((c) => c.uce_next), hit = next.filter((c) => c.lvl[ST.hzBack] === 2);
  const high = cards.filter((c) => c.lvl[ST.hzBack] === 2).length;
  $("#backHit").textContent = `此刻高风险 ${high} 张；此后 24 小时内实际发生不可纠正错误 ${next.length} 张，其中 ${hit.length} 张已被标为高风险。`;
}

// ---------------------------------------------------------------- collection / model pages
function fmtIv(v) { return v >= 60 && v % 60 === 0 ? v / 60 + " 分钟" : v + " 秒"; }
async function loadCluster() {
  const k = await api("/api/cluster");
  const s = k.settings, st = k.storage;
  const sel = $("#setInterval");
  if (!ST.settingsInit) {
    const opt = [...sel.options].find((o) => +o.value === s.interval_s);
    sel.value = opt ? String(s.interval_s) : "custom";
    $("#setCustom").style.display = opt ? "none" : ""; $("#setCustom").value = s.interval_s;
    $("#setRetention").value = s.retention_days; ST.settingsInit = true;
  }
  $("#storage").innerHTML = [
    ["当前频率", fmtIv(s.interval_s)], ["已接入计算卡", st.cards], ["单条记录", st.bytes_per_record + " 字节"],
    ["当前规模每天", st.mb_per_day + " MB"], [`保留 ${s.retention_days} 天合计`, st.gb_retained + " GB"],
    ["1024 卡每天（估算）", st.mb_per_day_1024 + " MB"]].map(([l, v]) => `<div class="kpi"><div class="v" style="font-size:18px">${v}</div><div class="l">${l}</div></div>`).join("");
  $("#deployCmd").textContent = k.deploy;
  $("#agentCount").textContent = `${k.agents.filter((a) => a.online).length} / ${k.agents.length} 个节点在线 · 已接收 ${k.ingest.records.toLocaleString()} 条记录`;
  $("#agents").innerHTML = k.agents.length ? `<table><tr><th>节点</th><th>拓扑位置</th><th>计算卡</th><th>状态</th><th>最近上报</th><th>采集频率</th><th>记录数</th><th>可构建的模型输入</th></tr>${k.agents.map((a) => `
    <tr><td class="mono">${esc(a.host)}</td><td>${a.node || "—"}${a.warning ? " · " + a.warning : ""}</td><td class="num">${a.cards}</td>
      <td class="${a.online ? "ok" : "no"}">${a.online ? "在线" : "离线"}</td><td class="mono">${a.last_seen ? fmtT(a.last_seen) : "—"}</td>
      <td>${a.interval_s ? fmtIv(a.interval_s) : "—"}</td><td class="num">${a.records.toLocaleString()}</td><td>${esc(a.summary ? a.summary.scenario : "—")}</td></tr>
    <tr><td colspan="8"><div class="capgrid">${(a.summary ? a.summary.groups : []).map((g) => `<span class="cap ${g.status === "支持" ? "yes" : "no"}" title="${g.supported != null ? "可采指标 " + g.supported + "，本机不可用 " + g.unavailable : ""}">${g.group}</span>`).join("")}</div></td></tr>`).join("")}</table>`
    : `<div class="muted">尚无节点接入。在计算节点上运行下面的命令即可自动探测采集能力并开始上报。</div>`;
}
$("#setInterval").onchange = (e) => { $("#setCustom").style.display = e.target.value === "custom" ? "" : "none"; };
$("#setSave").onclick = async () => {
  const v = $("#setInterval").value === "custom" ? +$("#setCustom").value : +$("#setInterval").value;
  const r = await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ interval_s: v, retention_days: +$("#setRetention").value }) });
  $("#setMsg").textContent = `已保存：采集频率 ${fmtIv(r.interval_s)}，保留 ${r.retention_days} 天；各节点将在 1 分钟内生效`;
  ST.settingsInit = false; loadCluster(); refreshStatus();
};
async function loadCollect() {
  loadCluster();
  const c = ST.collect = await api("/api/collect");
  const sp = c.spool;
  const steps = [
    ["采集", "本机探测 / 采集器目录 / 开源数据回放", `${c.mode}${c.spool.local_records ? " · 本机样本 " + c.spool.local_records : ""}`],
    ["缓冲", "按小时写入采集器格式 JSONL", `${sp.files} 个文件 · 最新 ${sp.latest}`],
    ["解析", "遥测样本、错误事件、显存 RAS、作业日志", `累计 ${sp.records.toLocaleString()} 条（本小时 ${sp.last_records.toLocaleString()} 条）`],
    ["标准化", "错误码 → 与厂商无关的故障原因", `事件 ${sp.event_rows.toLocaleString()} 条 · 遥测 ${sp.telemetry_rows.toLocaleString()} 行`],
    ["字段识别", "按字段自动启用特征模块", Object.keys(c.models).length + " 组输入"],
    ["推理", "1/4/24 小时风险（AI + 规则混合）", "每小时一次"],
    ["归因", "故障树：类别、根因、处置建议", "高于关注阈值的计算卡"]];
  $("#flow").innerHTML = steps.map((s, i) => `<div class="step"><div class="n">步骤 ${i + 1}</div><div class="t">${s[0]}</div><div class="d">${s[1]}</div><div class="v">${s[2]}</div></div>`).join("");
  $("#probes").innerHTML = `<table><tr><th>采集源</th><th>探测项</th><th>状态</th><th>说明</th></tr>${c.probes.map((p) => `<tr><td>${p.source}</td><td class="mono">${esc(p.probe)}</td><td class="${p.usable ? "ok" : "no"}">${p.usable ? "可用" : "不可用"}</td><td>${esc(p.detail)}</td></tr>`).join("")}</table>
    <p class="note">本机没有可用的采集源时，自动切换为开源数据回放；回放数据按采集器的输出格式写入缓冲目录，与真实采集走同一条处理链路。</p>`;
  $("#replay").innerHTML = `<table><tr><th>数据</th><th>来源</th><th>覆盖计算卡</th></tr>${c.replay_sources.map((r) => `<tr><td>${r.name}</td><td>${esc(r.from)}</td><td class="num">${r.cards}</td></tr>`).join("")}</table>
    <p class="note">遥测字段映射：${Object.entries(c.telemetry_fields).map(([k, v]) => k + " ← " + v).join("，")}。为便于演示，选卡时纳入了回放期内发生过不可纠正错误和硬件错误的设备，故障密度高于真实集群。</p>`;
  const L = c.local;
  if (!L.available) {
    $("#localHost").textContent = "本机未检测到可访问的计算卡";
    $("#localTable").innerHTML = ""; $("#localChart").innerHTML = "";
  } else {
    $("#localHost").textContent = `主机 ${L.host} · 已采集 ${L.series.length} 个样本`;
    const last = {}; L.series.forEach((r) => (last[r.gpu] = r));
    $("#localTable").innerHTML = `<table><tr><th>本机计算卡</th><th>时间（UTC）</th>${L.fields.map((f) => `<th>${f.label}</th>`).join("")}</tr>${Object.values(last).map((r) =>
      `<tr><td>卡 ${r.gpu}</td><td class="mono">${r.ts.replace("T", " ").replace("Z", "")}</td>${L.fields.map((f) => `<td class="num">${r[f.key] == null ? '<span class="muted">不支持</span>' : r[f.key]}</td>`).join("")}</tr>`).join("")}</table>`;
    const S = L.series.filter((r) => r.gpu === 0), ts = S.map((r) => Date.parse(r.ts) / 1000);
    lineChart($("#localChart"), 1200, 140, ts, [{ y: S.map((r) => r.DCGM_FI_DEV_GPU_TEMP), c: "#ffa726" }, { y: S.map((r) => r.DCGM_FI_DEV_POWER_USAGE), c: "#42a5f5", right: true }], { right: true });
  }
  $("#fields").innerHTML = Object.values(c.models).map((m) => `<div style="margin-bottom:10px"><h4 style="margin:4px 0">${m.model}</h4>
    <table><tr><th>识别字段</th><th>值</th></tr>${Object.entries(m.fields).map(([k, v]) => `<tr><td>${k}</td><td>${Array.isArray(v) ? `<div class="chips">${v.map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</div>` : esc(v ?? "—")}</td></tr>`).join("")}
    <tr><td>启用的模块</td><td><div class="chips">${m.modules.map((x) => `<span class="chip on">${esc(x)}</span>`).join("")}</div></td></tr></table></div>`).join("");
  // model page
  $("#models").innerHTML = Object.values(c.models).map((m) => {
    const p = m.info.tree_params || {};
    return `<div style="margin-bottom:12px"><h4 style="margin:4px 0">${m.model}</h4><table>
      <tr><th>项目</th><th>值</th></tr>
      <tr><td>特征数 / 特征组</td><td>${m.info.n_features} · ${(m.info.groups || []).join("、")}</td></tr>
      <tr><td>树模型</td><td>${m.info.backend === "lgbm" ? "LightGBM" : m.info.backend}；自动调参：${p._val_ap ? `验证期 PR-AUC ${p._val_ap_default.toFixed(3)} → ${p._val_ap.toFixed(3)}` : "未调参（验证期正样本不足或关闭）"}</td></tr>
      <tr><td>AI 融合权重（1/4/24 小时）</td><td>${Object.values(m.info.fusion_weight_ai || {}).map((v) => v.toFixed(2)).join(" / ")}</td></tr>
      <tr><td>高风险阈值（1/4/24 小时）</td><td>${ST.horizons.map((h) => pct(m.thresholds[h.key].high)).join(" / ")}</td></tr>
      <tr><td>关注阈值（1/4/24 小时）</td><td>${ST.horizons.map((h) => pct(m.thresholds[h.key].watch)).join(" / ")}</td></tr></table></div>`;
  }).join("");
  const W = c.fta_weights;
  const gates = {};
  c.fta_basic.forEach((b) => (gates[b.gate] = gates[b.gate] || []).push(b));
  $("#ftaStatic").innerHTML = `<div class="tree"><div class="tree-top">顶事件：计算卡不可纠正错误</div><div class="gates">${Object.entries(gates).map(([g, bs]) => `
    <div class="gate"><div class="gh"><span>${g}</span></div><div class="gate-op">或门</div>${bs.map((b) => {
      const w1 = W.log[b.id], w2 = W.ce[b.id];
      const f = (w) => (w && w.p_uce != null ? pct(w.p_uce) + (w.rr ? `（RR ${w.rr.toFixed(1)}）` : "") : "—");
      return `<div class="be on" style="background:#1f2a38;border-color:#2a3442">${esc(b.name)}<span class="ev" style="color:#9fb3c8">${b.source}：${b.prior != null ? "专家先验 RR " + b.prior : "日志模型 " + f(w1) + " · RAS 模型 " + f(w2)}</span></div>`;
    }).join("")}</div>`).join("")}</div>
    <p class="note">基本事件权重为“出现该事件时 24 小时内发生不可纠正错误”的条件概率，括号内为相对风险 RR（相对于未出现该事件），均在各模型的训练数据上估计；或门概率 = 1 − Π(1 − p)。遥测类基本事件缺少带标签的历史数据，使用专家先验的 RR × 基线概率。“—”表示训练数据中出现次数不足。</p></div>`;
}

// ---------------------------------------------------------------- algorithm advantages
const fmtV = (v, f) => (v == null ? "—" : f === "pct" ? (v * 100).toFixed(1) + "%" : v.toFixed(3));
const VCLS = { "超越": "v-up", "略高（同等水平）": "v-eq", "持平": "v-eq", "新增能力": "v-new", "不同任务": "v-na", "低于": "v-down" };
async function loadAdv() {
  const a = await api("/api/advantages");
  const up = a.comparison.filter((r) => r.verdict === "超越");
  const hero = up.slice(0, 3).map((r) => `<div class="hero"><div class="hv">${fmtV(r.ours, r.fmt)}</div>
      <div class="hp">${esc(r.paper)} · ${esc(r.metric)}：${r.paper_note ? esc(r.paper_note) : "论文报告"} ${fmtV(r.paper_value, r.fmt)}</div><div class="ht">${esc(r.dataset)} · ${esc(r.protocol.split("，")[0])}</div></div>`);
  const abl = a.ablation.log && a.ablation.log.hybrid ? a.ablation.log : null;
  if (abl && abl.ai) {
    const h = abl.hybrid["1h"].pr_auc, ai = abl.ai["1h"].pr_auc;
    hero.push(`<div class="hero"><div class="hv">${(h / Math.max(ai, 1e-9)).toFixed(1)} 倍</div><div class="hp">1 小时预警排序能力：AI + 规则混合 vs 纯 AI（PR-AUC ${h.toFixed(4)} vs ${ai.toFixed(4)}）</div>
      <div class="ht">本原型回溯期 · 运行日志模型</div></div>`);
  }
  $("#advHero").innerHTML = hero.join("");
  $("#advCompare").innerHTML = `<table><tr><th>数据集</th><th>对比论文</th><th>任务</th><th>评测口径</th><th>指标</th><th>论文 / 论文方法</th><th>本框架</th><th>结论</th><th>说明</th></tr>${a.comparison.map((r) => `
    <tr class="${r.verdict === "超越" ? "best" : ""}"><td>${esc(r.dataset)}</td><td>${esc(r.paper)}</td><td>${esc(r.task)}</td><td>${esc(r.protocol)}</td><td>${esc(r.metric)}</td>
      <td class="num">${fmtV(r.paper_value, r.fmt)}${r.paper_note ? `<div class="muted" style="font-size:11px;text-align:left">${esc(r.paper_note)}</div>` : ""}</td>
      <td class="num"><b>${fmtV(r.ours, r.fmt)}</b></td><td><span class="verdict ${VCLS[r.verdict] || "v-na"}">${r.verdict}</span></td><td>${esc(r.detail)}</td></tr>`).join("")}</table>`;
  $("#advWindow").textContent = `${a.window[0]} 至 ${a.window[1]}`;
  const NAMES = { hybrid: "AI + 规则混合（本框架）", ai: "纯 AI（LightGBM）", rules: "规则评分卡", rule: "简单规则（近30天硬件错误）" };
  const P = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
  $("#advAblation").innerHTML = Object.entries(a.ablation).map(([src, m]) => {
    const keys = ["hybrid", "ai", "rules"].filter((k) => m[k]);
    const cols = [
      ["PR-AUC 1小时", (k) => m[k]["1h"].pr_auc, "max", (v) => v.toFixed(4)],
      ["PR-AUC 4小时", (k) => m[k]["4h"].pr_auc, "max", (v) => v.toFixed(4)],
      ["PR-AUC 24小时", (k) => m[k]["1d"].pr_auc, "max", (v) => v.toFixed(3)],
      ["校准比 4小时", (k) => m[k]["4h"].calib, "one", (v) => v.toFixed(2)],
      ["校准比 24小时", (k) => m[k]["1d"].calib, "one", (v) => v.toFixed(2)],
      ["1小时 高风险阈值 召回", (k) => m[k]["1h"].model.recall, "max", P],
      ["1小时 高风险阈值 精确率", (k) => m[k]["1h"].model.precision, "max", P],
      ["24小时 同等告警量 召回", (k) => (m[k]["1d"].model_same_budget || {}).recall, "max", P],
      ["24小时 同等告警量 精确率", (k) => (m[k]["1d"].model_same_budget || {}).precision, "max", P]];
    const bestOf = (f, how) => {
      const v = keys.map((k) => f(k)).filter((x) => x != null);
      return how === "one" ? v.reduce((b, x) => (Math.abs(x - 1) < Math.abs(b - 1) ? x : b), v[0]) : Math.max(...v);
    };
    const rule = m.hybrid["1d"].rule;
    return `<h4 style="margin:6px 0">${a.models[src]} · ${a.cards[src]} 张计算卡</h4><table><tr><th>方法</th>${cols.map((c) => `<th>${c[0]}</th>`).join("")}</tr>
      ${keys.map((k) => `<tr class="${k === "hybrid" ? "best" : ""}"><td>${NAMES[k]}</td>${cols.map(([, f, how, fmt]) => {
        const v = f(k), b = bestOf(f, how);
        return `<td class="num ${v != null && v === b ? "better" : ""}">${v == null ? "—" : fmt(v)}</td>`;
      }).join("")}</tr>`).join("")}
      <tr><td>${NAMES.rule}</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td>
        <td class="num">${P(m.hybrid["1h"].rule.recall)}</td><td class="num">${P(m.hybrid["1h"].rule.precision)}</td><td class="num">${P(rule.recall)}</td><td class="num">${P(rule.precision)}</td></tr></table>`;
  }).join("") || `<div class="muted">尚未生成消融结果（运行 build.py --ablation-only）。</div>`;
  const caps = [
    ["按字段自适应输入", "不按数据集写适配器：自动识别设备、时间、错误编码、地址层级、CE 计数、遥测字段，按字段开关特征模块；本原型中两类数据源的卡同时扫描。"],
    ["1/4/24 小时一致的风险概率", "分段风险模型一次给出三个时间窗，结构上保证 1 小时 ≤ 4 小时 ≤ 24 小时；论文方法通常只有单一时间窗。"],
    ["概率经过校准", "先验校正后可直接当作发生概率使用，用于预估故障数和备件；校准比见上表。"],
    ["AI + 规则混合", "规则评分卡 + 带单调约束的 LightGBM，验证集上选融合权重。1 小时、4 小时窗口的排序明显优于单独的 AI 或规则，概率校准更接近 1；24 小时窗口与纯 AI 各有胜负（见上表）。"],
    ["可解释的报警", "每次报警给出规则依据和故障树归因（故障类别、根因、处置建议），基本事件权重由历史数据估计。"],
    ["实时、可扩展", "1,024 张卡一次扫描约 1 秒；7,500 万条级日志可分片处理；集群模式按设定频率实时采集与扫描。"]];
  $("#advCaps").innerHTML = caps.map(([t, d]) => `<div class="capc"><b>${t}</b><span>${d}</span></div>`).join("");
}

// ---------------------------------------------------------------- loop
refreshStatus();
setInterval(refreshStatus, 1000);
setInterval(() => { if (document.querySelector("#tab-collect").classList.contains("active")) loadCluster(); }, 10000);
