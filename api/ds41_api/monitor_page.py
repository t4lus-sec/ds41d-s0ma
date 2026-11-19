"""s0ma change 131: the monitor's page (GET /monitor). Self-contained: no external scripts, fonts or styles, so it
works offline; it polls /monitor/data every two seconds and draws its charts on canvases."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ds41d monitor</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1b1f24; --muted: #5d6673; --faint: #8a93a0; --line: #e3e6ea;
  --accent: #2f6fdf; --accent2: #c2581f; --accent3: #2c9a6a; --accent4: #8b5cd6; --warn: #b7791f; --bad: #c53030;
  --good: #2c9a6a; --chip: #eef1f5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1215; --panel: #171b20; --ink: #e6e9ed; --muted: #a2abb7; --faint: #6e7783; --line: #262c33;
    --accent: #6ea1ff; --accent2: #f08a4b; --accent3: #4cc38a; --accent4: #b18cff; --warn: #e0a84a; --bad: #ff6b6b;
    --good: #4cc38a; --chip: #20262d;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.mono, td.n, .tile .v { font-variant-numeric: tabular-nums; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 18px; padding: 18px 24px 8px; }
header h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: .2px; }
header .meta { color: var(--muted); font-size: 13px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--faint); margin-right: 6px; vertical-align: 1px; }
.dot.live { background: var(--good); } .dot.busy { background: var(--accent); } .dot.down { background: var(--bad); }
main { padding: 8px 24px 32px; display: grid; gap: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
.tile { padding: 12px 14px; }
.tile .k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }
.tile .v { font-size: 26px; font-weight: 650; margin-top: 2px; }
.tile .v small { font-size: 13px; font-weight: 500; color: var(--muted); margin-left: 3px; }
.tile .s { color: var(--faint); font-size: 12px; margin-top: 2px; min-height: 17px; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
.panel { padding: 14px 16px 12px; min-width: 0; }
.panel h2 { font-size: 13px; margin: 0 0 8px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .5px;
  display: flex; justify-content: space-between; gap: 8px; }
.panel h2 span { text-transform: none; letter-spacing: 0; font-weight: 500; color: var(--faint); }
canvas { width: 100%; height: 190px; display: block; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin-top: 4px; }
.legend i { display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 5px; vertical-align: 3px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 12px; padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 5px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td.n, th.n { text-align: right; }
tr:last-child td { border-bottom: 0; }
.scroll { overflow-x: auto; }
.chip { display: inline-block; padding: 0 7px; border-radius: 999px; background: var(--chip); color: var(--muted); font-size: 12px; }
.chip.err { color: var(--bad); } .chip.warn { color: var(--warn); } .chip.ok { color: var(--good); }
.empty { color: var(--faint); padding: 10px 0; }
.bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--chip); margin: 6px 0 4px; }
.bar div { height: 100%; }
.kv { display: grid; grid-template-columns: 1fr auto; gap: 3px 16px; font-size: 13px; }
.kv div:nth-child(odd) { color: var(--muted); }
.kv div:nth-child(even) { text-align: right; font-variant-numeric: tabular-nums; }
footer { color: var(--faint); font-size: 12px; padding: 0 24px 24px; }
@media (max-width: 520px) { header, main, footer { padding-left: 16px; padding-right: 16px; } }
</style>
</head>
<body>
<header>
  <h1>ds41d monitor</h1>
  <div class="meta" id="status"><span class="dot"></span>connecting…</div>
  <div class="meta" id="model"></div>
</header>
<main>
  <section class="tiles" id="tiles"></section>
  <section class="panel">
    <h2>Running now <span id="queued"></span></h2>
    <div class="scroll"><table id="active"></table></div>
  </section>
  <section class="grid2">
    <div class="panel"><h2>Decode speed <span>tok/s per request</span></h2><canvas id="c_decode"></canvas>
      <div class="legend"><span><i style="background:var(--accent)"></i>decode</span><span><i style="background:var(--accent3)"></i>draft acceptance ×100</span></div></div>
    <div class="panel"><h2>Prompt <span>first token, s · prefill, thousand tok/s</span></h2><canvas id="c_prompt"></canvas>
      <div class="legend"><span><i style="background:var(--accent2)"></i>time to first token</span><span><i style="background:var(--accent4)"></i>prefill speed</span></div></div>
    <div class="panel"><h2>GPU load <span>%</span></h2><canvas id="c_gpu"></canvas>
      <div class="legend"><span><i style="background:var(--accent)"></i>SM busy</span><span><i style="background:var(--accent2)"></i>memory controller busy</span></div></div>
    <div class="panel"><h2>GPU power and heat <span>W · °C</span></h2><canvas id="c_power"></canvas>
      <div class="legend"><span><i style="background:var(--accent4)"></i>power</span><span><i style="background:var(--warn)"></i>temperature</span></div></div>
  </section>
  <section class="grid2">
    <div class="panel"><h2>Experts <span>where decode found them, last 50 requests</span></h2><div id="experts"></div></div>
    <div class="panel"><h2>Saved conversations <span id="conv_at"></span></h2><div id="convs"></div></div>
  </section>
  <section class="panel">
    <h2>Recent requests <span>the last 25, newest first</span></h2>
    <div class="scroll"><table id="recent"></table></div>
  </section>
</main>
<footer>Numbers only: no prompt or completion text is recorded, and conversation ids show as an 8-character hash. Refreshes every 2 s; the GPU is sampled only while this page is open.</footer>
<script>
"use strict";
const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 0) => v == null || !isFinite(v) ? "–" : Number(v).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
const pct = v => v == null ? "–" : fmt(v * 100, 1) + "%";
const gib = b => b == null ? "–" : fmt(b / 1073741824, 1) + " GiB";
const dur = s => s == null ? "–" : s < 60 ? fmt(s, 1) + " s" : s < 3600 ? fmt(s / 60, 1) + " min" : fmt(s / 3600, 1) + " h";
const clock = t => new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const median = a => { const v = a.filter(x => x != null && isFinite(x)).sort((x, y) => x - y); if (!v.length) return null;
  const m = v.length >> 1; return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2; };

function chart(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const L = 40, R = 8, T = 8, B = 22, pw = w - L - R, ph = h - T - B;
  const pts = series.flatMap(s => s.points.filter(p => p[1] != null && isFinite(p[1])));
  g.font = "11px ui-sans-serif, system-ui, sans-serif"; g.fillStyle = css("--faint"); g.strokeStyle = css("--line"); g.lineWidth = 1;
  if (!pts.length) { g.fillText("no data yet", L, T + ph / 2); return; }
  let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  if (x1 - x0 < 60) x0 = x1 - 60;
  let y1 = opts.max != null ? opts.max : Math.max(...pts.map(p => p[1])) * 1.1 || 1, y0 = 0;
  const step = Math.pow(10, Math.floor(Math.log10(y1 / 4 || 1))), nice = [1, 2, 2.5, 5, 10].map(m => m * step).find(s => y1 / s <= 5) || step * 10;
  if (opts.max == null) y1 = Math.ceil(y1 / nice) * nice;
  const X = x => L + (x - x0) / (x1 - x0) * pw, Y = y => T + ph - (y - y0) / (y1 - y0) * ph;
  for (let v = 0; v <= y1 + 1e-9; v += nice) {
    g.beginPath(); g.moveTo(L, Y(v) + .5); g.lineTo(L + pw, Y(v) + .5); g.stroke();
    g.textAlign = "right"; g.fillText(fmt(v, nice < 1 ? 1 : 0), L - 6, Y(v) + 4);
  }
  g.textAlign = "center";
  for (let i = 0; i <= 3; ++i) { const t = x0 + (x1 - x0) * i / 3; g.fillText(new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}), Math.min(Math.max(X(t), L + 20), L + pw - 20), h - 6); }
  for (const s of series) {
    const p = s.points.filter(q => q[1] != null && isFinite(q[1]));
    g.strokeStyle = g.fillStyle = css(s.color); g.lineWidth = 1.75;
    if (s.dots) { for (const q of p) { g.beginPath(); g.arc(X(q[0]), Y(Math.min(q[1], y1)), 2.6, 0, 7); g.fill(); } continue; }
    g.beginPath(); p.forEach((q, i) => i ? g.lineTo(X(q[0]), Y(Math.min(q[1], y1))) : g.moveTo(X(q[0]), Y(Math.min(q[1], y1)))); g.stroke();
  }
}

function tiles(d) {
  const r = d.recent.filter(x => !x.probe && x.completion_tokens > 8), last = r.slice(-20), g = d.gpu[d.gpu.length - 1] || {};
  const drafted = last.reduce((a, x) => a + (x.drafted || 0), 0), accepted = last.reduce((a, x) => a + (x.accepted || 0), 0);
  const prompt = d.recent.slice(-20).reduce((a, x) => a + (x.prompt_tokens || 0), 0), cached = d.recent.slice(-20).reduce((a, x) => a + (x.cached_tokens || 0), 0);
  const items = [
    ["Decode", fmt(median(last.map(x => x.decode_tok_s)), 1), "tok/s", r.length ? `last ${fmt(r[r.length - 1].decode_tok_s, 1)} · median of ${last.length}` : "no requests yet"],
    ["First token", fmt(median(d.recent.slice(-20).map(x => x.ttft_ms / 1000)), 2), "s", "median, last 20"],
    ["Draft acceptance", drafted ? fmt(accepted / drafted * 100, 1) : "–", "%", `${fmt(median(last.map(x => x.tokens_per_step)), 2)} tokens a step`],
    ["Verify step", fmt(median(last.map(x => x.verify_ms_step)), 1), "ms", `drafting ${fmt(median(last.map(x => x.draft_ms_step)), 2)} ms a step`],
    ["Prompt reuse", prompt ? fmt(cached / prompt * 100, 0) : "–", "%", "of prompt tokens from cache, last 20"],
    ["GPU", g.util != null ? fmt(g.util) : "–", "%", g.mem_used_mib != null ? `${fmt(g.mem_used_mib / 1024, 1)} GiB · ${fmt(g.power_w)} W · ${fmt(g.temp_c)} °C` : (d.gpu.length ? "no nvidia-smi" : "sampling…")],
    ["Served", fmt(d.totals.requests), "requests", `${fmt(d.totals.completion_tokens)} tokens out · ${fmt(d.totals.prompt_tokens)} in`],
  ];
  $("tiles").innerHTML = items.map(([k, v, u, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}<small>${u}</small></div><div class="s">${esc(s)}</div></div>`).join("");
}

function active(d) {
  $("queued").textContent = d.queued ? `${d.queued} waiting at the API` : "";
  if (!d.active.length) { $("active").innerHTML = `<tr><td class="empty">Idle</td></tr>`; return; }
  $("active").innerHTML = `<tr><th>request</th><th>conversation</th><th>state</th><th class="n">prompt</th><th class="n">generated</th><th class="n">tok/s</th><th class="n">elapsed</th><th class="n">priority</th></tr>` +
    d.active.sort((a, b) => b.elapsed_s - a.elapsed_s).map(a => `<tr><td class="mono">${a.id}</td><td class="mono">${a.conversation || "–"}</td>
      <td>${a.prefilling ? '<span class="chip warn">prefill</span>' : '<span class="chip ok">decode</span>'}${a.images ? ' <span class="chip">image</span>' : ""}</td>
      <td class="n">${fmt(a.prompt_tokens)}</td><td class="n">${fmt(a.tokens)}${a.max_tokens ? ` <span style="color:var(--faint)">/ ${fmt(a.max_tokens)}</span>` : ""}</td>
      <td class="n">${fmt(a.tok_s, 1)}</td><td class="n">${dur(a.elapsed_s)}</td><td class="n">${a.priority ?? "–"}</td></tr>`).join("");
}

function recent(d) {
  const rows = d.recent.slice(-25).reverse();
  if (!rows.length) { $("recent").innerHTML = `<tr><td class="empty">No finished requests yet</td></tr>`; return; }
  const fin = r => r.error || r.finish === "error" ? '<span class="chip err">error</span>' : r.finish === "cancelled" ? '<span class="chip warn">cancelled</span>' :
    `<span class="chip">${esc(r.finish)}</span>`;
  $("recent").innerHTML = `<tr><th>finished</th><th>conversation</th><th class="n">prompt</th><th class="n">cached</th><th>cache</th><th class="n">first token</th>
    <th class="n">prefill tok/s</th><th class="n">out</th><th class="n">decode tok/s</th><th class="n">accept</th><th class="n">tok/step</th><th class="n">verify ms</th><th class="n">RAM experts</th><th>end</th></tr>` +
    rows.map(r => `<tr><td class="mono">${clock(r.t_end)}</td><td class="mono">${r.conversation || "–"}${r.probe ? ' <span class="chip">probe</span>' : ""}${r.images ? ' <span class="chip">image</span>' : ""}</td>
      <td class="n">${fmt(r.prompt_tokens)}</td><td class="n">${fmt(r.cached_tokens)}</td><td>${r.prompt_cache ? `<span class="chip">${esc(r.prompt_cache)}</span>` : ""}</td>
      <td class="n">${r.ttft_ms != null ? fmt(r.ttft_ms / 1000, 2) + " s" : "–"}</td><td class="n">${fmt(r.prefill_tok_s)}</td><td class="n">${fmt(r.completion_tokens)}</td>
      <td class="n">${fmt(r.decode_tok_s, 1)}</td><td class="n">${pct(r.acceptance)}</td><td class="n">${fmt(r.tokens_per_step, 2)}</td><td class="n">${fmt(r.verify_ms_step, 1)}</td>
      <td class="n">${fmt(r.experts_ram)}${r.experts_nvme ? ` <span class="chip err">${fmt(r.experts_nvme)} NVMe</span>` : ""}</td>
      <td>${fin(r)}${r.loop_guard ? ' <span class="chip warn">loop guard</span>' : ""}</td></tr>`).join("");
}

function experts(d) {
  const r = d.recent.slice(-50), v = r.reduce((a, x) => a + (x.experts_vram || 0), 0), m = r.reduce((a, x) => a + (x.experts_ram || 0), 0),
        n = r.reduce((a, x) => a + (x.experts_nvme || 0), 0), all = v + m + n, e = d.engine;
  const bar = all ? `<div class="bar"><div style="width:${v / all * 100}%;background:var(--accent3)"></div><div style="width:${m / all * 100}%;background:var(--accent)"></div><div style="width:${n / all * 100}%;background:var(--bad)"></div></div>
    <div class="legend"><span><i style="background:var(--accent3)"></i>VRAM ${pct(v / all)}</span><span><i style="background:var(--accent)"></i>RAM ${pct(m / all)}</span><span><i style="background:var(--bad)"></i>NVMe ${pct(n / all)}</span></div>` : `<div class="empty">No decode yet</div>`;
  const h2d = r.reduce((a, x) => a + (x.h2d_bytes || 0), 0), steps = r.reduce((a, x) => a + (x.steps || 0), 0);
  $("experts").innerHTML = bar + `<div class="kv" style="margin-top:10px">
    <div>Copied RAM → GPU, last 50</div><div>${fmt(h2d / 1e9, 1)} GB${steps ? ` · ${fmt(h2d / steps / 1e6)} MB a step` : ""}</div>
    <div>Engine life: VRAM / RAM / NVMe</div><div>${fmt(e.expert_vram_hits)} / ${fmt(e.expert_host_hits)} / ${fmt(e.expert_nvme_misses)}</div>
    <div>Read from NVMe, engine life</div><div>${e.expert_nvme_bytes != null ? fmt(e.expert_nvme_bytes / 1e9, 1) + " GB" : "–"}</div>
    <div>Live slots</div><div>${fmt(e.live_slots)}</div></div>`;
}

function convs(d) {
  const c = d.conversations, e = d.engine;
  if (!c) { $("convs").innerHTML = `<div class="empty">Waiting for the engine…</div>`; return; }
  $("conv_at").textContent = "as of " + clock(c.at);
  const used = c.ram_budget ? c.ram_bytes / c.ram_budget : null;
  $("convs").innerHTML = `<div class="kv">
    <div>In RAM</div><div>${fmt(c.ram_count)} · ${fmt(c.ram_tokens)} tokens · ${gib(c.ram_bytes)}${c.ram_budget ? ` of ${gib(c.ram_budget)}` : ""}</div>
    <div>On disk</div><div>${fmt(c.disk_count)} · ${gib(c.disk_bytes)}</div>
    <div>Saves / loads / evictions</div><div>${fmt(e.conversation_saves)} / ${fmt(e.conversation_loads)} / ${fmt(e.conversation_evictions)}</div></div>` +
    (used != null ? `<div class="bar"><div style="width:${Math.min(100, used * 100)}%;background:${used > .9 ? "var(--warn)" : "var(--accent)"}"></div></div>` : "") +
    (c.ram.length ? `<table style="margin-top:6px"><tr><th>conversation</th><th class="n">tokens</th><th class="n">size</th><th class="n">idle</th></tr>` +
      c.ram.map(x => `<tr><td class="mono">${x.conversation}</td><td class="n">${fmt(x.tokens)}</td><td class="n">${gib(x.bytes)}</td><td class="n">${dur(x.idle_s)}</td></tr>`).join("") + "</table>" : "");
}

let failures = 0;
async function tick() {
  try {
    const res = await fetch("/monitor/data", {cache: "no-store"});
    if (!res.ok) throw new Error(res.status);
    const d = await res.json(); failures = 0;
    const state = d.active.length ? `<span class="dot busy"></span>${d.active.length} running` : `<span class="dot live"></span>idle`;
    $("status").innerHTML = `${state} · up ${dur(d.uptime_s)} · updated ${clock(d.now)}`;
    $("model").textContent = d.model;
    tiles(d); active(d); recent(d); experts(d); convs(d);
    const r = d.recent.filter(x => !x.probe);
    chart($("c_decode"), [{color: "--accent", dots: true, points: r.filter(x => x.completion_tokens > 8).map(x => [x.t_end, x.decode_tok_s])},
                          {color: "--accent3", dots: true, points: r.map(x => [x.t_end, x.acceptance != null ? x.acceptance * 100 : null])}]);
    chart($("c_prompt"), [{color: "--accent2", dots: true, points: d.recent.map(x => [x.t_end, x.ttft_ms != null ? x.ttft_ms / 1000 : null])},
                          {color: "--accent4", dots: true, points: d.recent.filter(x => (x.prefill_tokens || 0) > 256).map(x => [x.t_end, x.prefill_tok_s / 1000])}]);
    chart($("c_gpu"), [{color: "--accent", points: d.gpu.map(s => [s.t, s.util])}, {color: "--accent2", points: d.gpu.map(s => [s.t, s.mem_busy])}], {max: 100});
    chart($("c_power"), [{color: "--accent4", points: d.gpu.map(s => [s.t, s.power_w])}, {color: "--warn", points: d.gpu.map(s => [s.t, s.temp_c])}]);
  } catch (err) {
    if (++failures > 1) $("status").innerHTML = `<span class="dot down"></span>cannot reach the API (${esc(err.message || err)})`;
  }
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"""
