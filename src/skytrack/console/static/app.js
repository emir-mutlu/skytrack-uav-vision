/* SkyTrack Ground Control - live operator console.
 * State arrives over Server-Sent Events (or polling with ?poll), video as MJPEG
 * (or polled snapshots). Commands go to POST /api/command as JSON. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const root = document.documentElement;
  const POLL = new URLSearchParams(location.search).has("poll");
  const WINDOW_S = 30;
  const M_PER_DEG = 111320;
  const S = { last: null, err: [], lat: [], trail: [], hover: {}, online: false, seq: -1, html: {} };
  let dirty = true;
  let toastTimer = 0;

  const tok = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const signed = (v, d = 2) => (v == null ? "—" : (v < 0 ? "−" : "+") + Math.abs(v).toFixed(d));
  const setText = (id, text) => { const el = $(id); if (el.textContent !== text) el.textContent = text; };
  const setHTML = (id, html) => { if (S.html[id] !== html) { S.html[id] = html; $(id).innerHTML = html; } };

  /* ------------------------------------------------------------------ theme */
  function setTheme(t) {
    root.dataset.theme = t;
    const b = $("theme-toggle");
    b.textContent = t === "day" ? "Night" : "Day";
    b.setAttribute("aria-label", t === "day" ? "Switch to night display" : "Switch to day display");
    try { localStorage.setItem("skytrack-theme", t); } catch (_) { /* storage unavailable */ }
    dirty = true;
  }
  let saved = null;
  try { saved = localStorage.getItem("skytrack-theme"); } catch (_) { /* storage unavailable */ }
  const forced = new URLSearchParams(location.search).get("theme"); // ?theme=day for screenshots / kiosks
  setTheme((forced || saved) === "day" ? "day" : "night");
  $("theme-toggle").addEventListener("click", () => setTheme(root.dataset.theme === "day" ? "night" : "day"));

  /* ------------------------------------------------------------------ toast + commands */
  function toast(msg, sticky = false) {
    const t = $("toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    if (!sticky) toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
  }

  async function send(cmd) {
    try {
      const r = await fetch("/api/command", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cmd),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.ok) toast(j.error || `Command failed (HTTP ${r.status})`);
    } catch (_) {
      toast("SkyTrack is not reachable. Check that `skytrack console` is still running.");
    }
  }

  /* ------------------------------------------------------------------ video */
  const video = $("video");
  function startVideo() { if (!POLL) video.src = "/stream.mjpg?ts=" + Date.now(); }
  video.addEventListener("load", () => { $("video-offline").hidden = true; });
  video.addEventListener("error", () => {
    $("video-offline").hidden = false;
    if (!POLL) setTimeout(startVideo, 1500);
  });
  $("video-wrap").addEventListener("click", (e) => {
    const s = S.last;
    if (!s) return;
    const r = video.getBoundingClientRect();
    const [iw, ih] = s.size;
    const k = Math.min(r.width / iw, r.height / ih);
    const dw = iw * k, dh = ih * k;
    const x = (e.clientX - r.left - (r.width - dw) / 2) / dw;
    const y = (e.clientY - r.top - (r.height - dh) / 2) / dh;
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) send({ cmd: "designate", x, y });
  });

  /* ------------------------------------------------------------------ controls */
  $("btn-release").addEventListener("click", () => send({ cmd: "release" }));
  $("btn-pause").addEventListener("click", () => send({ cmd: "toggle_pause" }));
  $("tg-follow").addEventListener("click", () => send({ cmd: "follow", on: !(S.last && S.last.gimbal.follow) }));
  $("tg-auto").addEventListener("click", () => send({ cmd: "auto_designation", on: !(S.last && S.last.lock.auto) }));
  $("tracks-body").addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-id]");
    if (row) send({ cmd: "designate_id", id: Number(row.dataset.id) });
  });
  $("tracks-body").addEventListener("keydown", (e) => {
    const row = e.target.closest("tr[data-id]");
    if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); send({ cmd: "designate_id", id: Number(row.dataset.id) }); }
  });
  addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest("input, textarea, select, tr")) return;
    if (e.target.closest("button") && (e.code === "Space" || e.key === "Enter")) return; // native click
    const k = e.key.toLowerCase();
    if (k === "r") send({ cmd: "release" });
    else if (e.code === "Space") { e.preventDefault(); send({ cmd: "toggle_pause" }); }
    else if (k === "f" && S.last && S.last.gimbal.available) send({ cmd: "follow", on: !S.last.gimbal.follow });
    else if (k === "a" && S.last) send({ cmd: "auto_designation", on: !S.last.lock.auto });
  });
  for (const id of ["sp-err", "sp-lat"]) {
    const cv = $(id);
    cv.addEventListener("mousemove", (e) => { S.hover[id] = e.offsetX; dirty = true; });
    cv.addEventListener("mouseleave", () => { S.hover[id] = null; dirty = true; });
  }
  addEventListener("resize", () => { dirty = true; });

  /* ------------------------------------------------------------------ data */
  function setOnline(on) {
    if (on === S.online) return;
    S.online = on;
    if (on) { $("toast").hidden = true; startVideo(); }
    else toast("Connection to SkyTrack lost. Reconnecting…", true);
  }

  async function loadHistory() {
    try {
      const h = await (await fetch("/api/history", { cache: "no-store" })).json();
      const now = performance.now() / 1000;
      S.err = h.err.filter((r) => r[0] <= WINDOW_S).map(([age, v]) => [now - age, v]);
      S.lat = h.lat.filter((r) => r[0] <= WINDOW_S).map(([age, v]) => [now - age, v]);
      S.trail = h.trail.map(([age, la, lo]) => [now - age, la, lo]);
      dirty = true;
    } catch (_) { /* charts simply start empty */ }
  }

  async function connect() {
    await loadHistory();
    if (POLL) {
      const tick = async () => {
        try {
          const r = await fetch("/api/state", { cache: "no-store" });
          const s = await r.json();
          setOnline(true);
          ingest(s);
          if (s.seq != null && s.seq !== S.seq) { S.seq = s.seq; video.src = "/api/snapshot.jpg?seq=" + s.seq; }
        } catch (_) { setOnline(false); }
        setTimeout(tick, 150);
      };
      tick();
      return;
    }
    const es = new EventSource("/events");
    es.onmessage = (m) => { setOnline(true); try { ingest(JSON.parse(m.data)); } catch (_) { /* partial frame */ } };
    es.onerror = () => setOnline(false);
  }

  function push(arr, t, v) {
    arr.push([t, v]);
    while (arr.length && t - arr[0][0] > WINDOW_S) arr.shift();
  }

  function ingest(s) {
    if (!s || !s.lock) return; // no frame processed yet
    S.last = s;
    const now = performance.now() / 1000;
    const e = s.lock.error;
    push(S.err, now, e ? Math.hypot(e[0], e[1]) : null);
    push(S.lat, now, s.latency_ms);
    const v = s.vehicle;
    if (v && v.connected && v.lat != null) {
      const last = S.trail[S.trail.length - 1];
      if (!last || now - last[0] > 0.5) S.trail.push([now, v.lat, v.lon]);
      while (S.trail.length && now - S.trail[0][0] > 120) S.trail.shift();
    }
    renderTop(s);
    renderVideo(s);
    renderLock(s);
    renderVehicle(s);
    renderTracks(s);
    renderEvents(s);
    dirty = true;
  }

  /* ------------------------------------------------------------------ panels */
  function renderTop(s) {
    setText("chip-source", s.source || "Source");
    setText("chip-profile", "Profile " + s.profile);
    const chip = $("chip-link"), v = s.vehicle;
    let status = "off", text = "No vehicle link";
    if (v && v.connected) { status = "ok"; text = `${v.mode || "?"} · ${v.armed ? "Armed" : "Disarmed"}`; }
    else if (s.gimbal.available) { status = "warn"; text = "No heartbeat"; }
    chip.dataset.status = status;
    setText("chip-link-text", text);
    setText("m-fps", fmt(s.fps, 1));
    setText("m-lat", fmt(s.latency_ms, 1) + " ms");
  }

  function lockSummary(L, nTracks) {
    if (!L.enabled) return `${nTracks} active tracks`;
    if (L.state === "LOCKED") return `Target #${L.track_id} · ${fmt(L.duration, 1)} s`;
    if (L.state === "ACQUIRING") return `Target #${L.track_id} · ${Math.round(L.progress * 100)} %`;
    if (L.state === "LOST") return `Target #${L.track_id} · reacquiring`;
    return L.candidate_id != null ? `Candidate #${L.candidate_id}` : "No target";
  }

  function renderVideo(s) {
    const L = s.lock, pill = $("state-pill");
    pill.dataset.state = L.enabled ? L.state : "OFF";
    setText("state-pill", L.enabled ? L.state.toLowerCase() : "Track only");
    setText("state-sub", lockSummary(L, s.tracks.length));
    $("video-paused").hidden = !s.paused;
    $("video-hint").hidden = !(L.enabled && !L.auto && s.tracks.length && L.designated_id == null);
    setText("frame-meta", `F${String(s.frame).padStart(6, "0")} · ${s.detector.toUpperCase()} · ${s.size[0]}×${s.size[1]}`);
    const pause = $("btn-pause");
    pause.setAttribute("aria-pressed", String(s.paused));
    pause.querySelector(".lbl").textContent = s.paused ? "Resume" : "Pause";
    const follow = $("tg-follow");
    follow.setAttribute("aria-checked", String(s.gimbal.follow && s.gimbal.available));
    follow.disabled = !s.gimbal.available;
    follow.title = s.gimbal.available ? "Keep the designated target centred with the gimbal (F)" : "Needs a flight-controller link";
    $("tg-auto").setAttribute("aria-checked", String(L.auto));
    $("tg-auto").disabled = !L.enabled;
    $("btn-release").disabled = !L.enabled;
  }

  function renderLock(s) {
    const L = s.lock;
    $("lock-card").dataset.state = L.enabled ? L.state : "OFF";
    setText("lock-state", L.enabled ? L.state.toLowerCase() : "Lock off");
    setText("lock-mode", !L.enabled ? "Disabled" : L.auto ? "Auto designation" : "Operator designation");
    setText("lock-tgt", L.track_id != null ? `#${L.track_id}` : L.candidate_id != null ? `#${L.candidate_id} cand.` : "—");
    setText("lock-time", L.state === "LOCKED" ? `${fmt(L.duration, 1)} s`
      : L.state === "ACQUIRING" ? `${Math.round(L.progress * 100)} % of ${fmt(L.acquire_s, 1)} s` : "—");
    setText("lock-ex", L.error ? signed(L.error[0]) : "—");
    setText("lock-ey", L.error ? signed(L.error[1]) : "—");
    const C = 2 * Math.PI * 52;
    const prog = L.state === "LOCKED" ? 1 : L.state === "ACQUIRING" ? L.progress : 0;
    $("scope-ring").style.strokeDashoffset = (C * (1 - prog)).toFixed(1);
    const dot = $("scope-dot"), line = $("scope-line");
    if (L.error) {
      const x = 60 + clamp(L.error[0], -1, 1) * 52, y = 60 + clamp(L.error[1], -1, 1) * 52;
      dot.setAttribute("cx", x.toFixed(1)); dot.setAttribute("cy", y.toFixed(1));
      line.setAttribute("x2", x.toFixed(1)); line.setAttribute("y2", y.toFixed(1));
      dot.style.opacity = 1; line.style.opacity = 1;
    } else { dot.style.opacity = 0; line.style.opacity = 0; }
  }

  function renderVehicle(s) {
    const v = s.vehicle;
    $("tele").hidden = !v;
    $("tele-empty").hidden = !!v;
    setText("tele-meta", !v ? "No link" : v.connected ? "MAVLink" : "No heartbeat");
    if (!v) return;
    setText("t-mode", v.mode || "—");
    setText("t-armed", v.armed ? "Armed" : "Disarmed");
    setText("t-alt", v.alt_rel != null ? `${fmt(v.alt_rel, 1)} m AGL` : "—");
    setText("t-gs", v.groundspeed != null ? `${fmt(v.groundspeed, 1)} m/s` : "—");
    setText("t-hdg", v.heading != null ? `${String(Math.round(v.heading) % 360).padStart(3, "0")}°` : "—");
    setText("t-cmd", s.gimbal.cmd ? `${signed(s.gimbal.cmd[0], 1)} / ${signed(s.gimbal.cmd[1], 1)} °/s` : "—");
    setText("t-gmb", v.gimbal_pitch != null
      ? `Pitch ${signed(v.gimbal_pitch, 1)}°  Yaw ${signed(v.gimbal_yaw, 1)}°${s.gimbal.active ? "  · following" : ""}`
      : "No gimbal attitude reported");
    setText("t-pos", v.lat != null ? `${v.lat.toFixed(6)}, ${v.lon.toFixed(6)}` : "Waiting for GPS");
    const g = s.target_geo, link = $("t-map");
    setText("t-tgt", g ? `${g.lat.toFixed(6)}, ${g.lon.toFixed(6)}` : "—");
    setText("t-rng", g ? `${fmt(g.range_m, 0)} m ground · ${g.method === "dem" ? "terrain model" : "flat ground"}` : "—");
    link.hidden = !g;
    if (g) link.href = `https://www.openstreetmap.org/?mlat=${g.lat}&mlon=${g.lon}#map=18/${g.lat}/${g.lon}`;
  }

  const ROLE = { target: "Target", candidate: "Candidate", track: "Tracked" };
  function renderTracks(s) {
    setText("tracks-meta", `${s.tracks.length} active`);
    const html = !s.tracks.length
      ? `<tr class="empty"><td colspan="7">No confirmed tracks yet. A target appears here after the detector has seen it in three frames.</td></tr>`
      : s.tracks.map((t) => `<tr data-id="${t.id}" data-role="${t.role}" tabindex="0" title="${
        t.geo ? `${t.geo.lat.toFixed(6)}, ${t.geo.lon.toFixed(6)} · ` : ""}Click to designate track #${t.id}">`
        + `<td>#${t.id}</td><td>${esc(t.label)}</td><td class="num">${t.score.toFixed(2)}</td>`
        + `<td class="num">${fmt(t.age_s, 1)} s</td><td class="num">${t.geo ? fmt(t.geo.range_m, 0) + " m" : "—"}</td>`
        + `<td class="num">${fmt(t.speed_px_s, 0)}</td>`
        + `<td><span class="role role-${t.role}">${ROLE[t.role]}</span></td></tr>`).join("");
    setHTML("tracks-body", html);
  }

  const EVENTS = {
    lock_acquired: ["Lock acquired", "lock"], lock_lost: ["Lock lost", "acq"], designated: ["Designated", "cand"],
    released: ["Released", "muted"], no_track: ["No track", "muted"], link_up: ["Link up", "track"],
    link_lost: ["Link lost", "acq"], paused: ["Paused", "muted"], resumed: ["Resumed", "muted"],
    follow_on: ["Gimbal follow on", "accent"], follow_off: ["Gimbal follow off", "muted"],
    auto_on: ["Auto designation on", "accent"], auto_off: ["Auto designation off", "muted"],
    session_start: ["Session started", "muted"], source_end: ["Source ended", "acq"],
  };
  function renderEvents(s) {
    setText("events-meta", `${s.events.length} logged`);
    const html = !s.events.length ? `<li class="empty">Lock, designation and link events appear here.</li>`
      : s.events.slice().reverse().map((e) => {
        const [label, tone] = EVENTS[e.kind] || [e.kind, "muted"];
        const id = e.track_id != null ? ` #${e.track_id}` : "";
        return `<li><time>${esc(e.utc)}</time><span class="ev ev-${tone}">${label}${id}</span>`
          + `<span class="ev-detail">${esc(e.detail || "")}</span></li>`;
      }).join("");
    setHTML("events", html);
  }

  /* ------------------------------------------------------------------ canvas helpers */
  function fit(cv) {
    const r = cv.getBoundingClientRect();
    const d = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(r.width * d)), H = Math.max(1, Math.round(r.height * d));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(d, 0, 0, d, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }
  const niceMax = (v) => {
    const p = Math.pow(10, Math.floor(Math.log10(v || 1)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  };

  /* Single-series sparkline: faint grid, area fill, 2px line, emphasised endpoint, hover readout. */
  function drawSpark(id, series, o) {
    const { ctx, w, h } = fit($(id));
    ctx.clearRect(0, 0, w, h);
    const now = performance.now() / 1000;
    const L = 40, R = 8, T = 6, B = 16, pw = w - L - R, ph = h - T - B;
    const vals = series.filter((p) => p[1] != null).map((p) => p[1]);
    const vmax = niceMax(Math.max(o.floor, ...vals) * 1.1);
    const X = (t) => L + pw * (1 - (now - t) / WINDOW_S);
    const Y = (v) => T + ph * (1 - Math.min(v, vmax) / vmax);
    const ink3 = tok("--ink-3"), accent = tok("--accent"), mono = tok("--font-data");
    ctx.font = `10px ${mono}`;
    ctx.lineWidth = 1;
    for (const f of [0, 0.5, 1]) {
      const y = Math.round(T + ph * (1 - f)) + 0.5;
      ctx.strokeStyle = f === 0 ? tok("--line") : tok("--grid");
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
      ctx.fillStyle = ink3; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(o.fmt(vmax * f), L - 6, y);
    }
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left"; ctx.fillText("−30 s", L, h - 3);
    ctx.textAlign = "center"; ctx.fillText("−15 s", L + pw / 2, h - 3);
    ctx.textAlign = "right"; ctx.fillText("now", w - R, h - 3);

    const segs = [];
    let cur = [];
    for (const [t, v] of series) {
      if (v == null) { if (cur.length) segs.push(cur); cur = []; } else cur.push([X(t), Y(v)]);
    }
    if (cur.length) segs.push(cur);
    for (const sg of segs) {
      if (sg.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(sg[0][0], T + ph);
      for (const [x, y] of sg) ctx.lineTo(x, y);
      ctx.lineTo(sg[sg.length - 1][0], T + ph);
      ctx.closePath();
      ctx.globalAlpha = 0.16; ctx.fillStyle = accent; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath();
      sg.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
    }
    const tail = series[series.length - 1];
    if (tail && tail[1] != null) {
      ctx.beginPath(); ctx.arc(X(tail[0]), Y(tail[1]), 4, 0, 2 * Math.PI);
      ctx.fillStyle = accent; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = tok("--surface"); ctx.stroke();
    }
    setText(id + "-v", tail && tail[1] != null ? o.fmt(tail[1], true) : o.none);

    const hx = S.hover[id];
    if (hx != null && hx >= L && hx <= w - R && series.length) {
      const tt = now - (1 - (hx - L) / pw) * WINDOW_S;
      let best = series[0];
      for (const p of series) if (Math.abs(p[0] - tt) < Math.abs(best[0] - tt)) best = p;
      ctx.strokeStyle = ink3; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx + 0.5, T); ctx.lineTo(hx + 0.5, T + ph); ctx.stroke(); ctx.setLineDash([]);
      if (best[1] != null) {
        ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 4, 0, 2 * Math.PI);
        ctx.fillStyle = accent; ctx.fill(); ctx.strokeStyle = tok("--surface"); ctx.lineWidth = 2; ctx.stroke();
      }
      const label = `${best[1] != null ? o.fmt(best[1], true) : o.none} · ${(now - best[0]).toFixed(1)} s ago`;
      ctx.font = `11px ${mono}`;
      const tw = ctx.measureText(label).width + 12;
      const bx = clamp(hx + 8, L, w - R - tw);
      ctx.fillStyle = tok("--surface-2"); ctx.strokeStyle = tok("--line");
      ctx.beginPath(); ctx.rect(bx, T, tw, 18); ctx.fill(); ctx.stroke();
      ctx.fillStyle = tok("--ink"); ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(label, bx + 6, T + 9.5);
    }
  }

  /* Tactical plot: north up, vehicle at the centre, range rings, camera footprint, targets. */
  const RINGS = [60, 90, 150, 300, 600, 900, 1500, 3000, 6000];
  function drawMap() {
    const { ctx, w, h } = fit($("map"));
    ctx.clearRect(0, 0, w, h);
    const s = S.last, v = s && s.vehicle;
    const label = tok("--font-label"), mono = tok("--font-data");
    if (!v || !v.connected || v.lat == null) {
      ctx.textAlign = "center";
      ctx.fillStyle = tok("--ink-2"); ctx.font = `600 12px ${label}`;
      ctx.fillText(s && s.gimbal.available ? "WAITING FOR VEHICLE POSITION" : "NO VEHICLE LINK", w / 2, h / 2 - 8);
      ctx.fillStyle = tok("--ink-3"); ctx.font = `12px ${tok("--font-body")}`;
      ctx.fillText(s && s.gimbal.available ? "No heartbeat or GPS position yet" : "Start with --mavlink or the sim source", w / 2, h / 2 + 14);
      setText("map-meta", "North up");
      return;
    }
    const k = Math.cos(v.lat * Math.PI / 180);
    const ne = (lat, lon) => [(lat - v.lat) * M_PER_DEG, (lon - v.lon) * M_PER_DEG * k];
    // Scale to the targets; a shallow camera footprint can reach kilometres, so it only sets
    // the scale when nothing is tracked and is otherwise clipped to the plot circle.
    let ext = 0;
    for (const t of s.tracks) if (t.geo) ext = Math.max(ext, t.geo.range_m * 1.35);
    if (!ext && s.footprint) {
      const d = s.footprint.filter(Boolean).map((p) => Math.hypot(p[0], p[1]));
      if (d.length) ext = Math.min(...d) * 1.6;
    }
    ext = Math.max(ext, 50);
    const R = RINGS.find((x) => x >= ext) || RINGS[RINGS.length - 1];
    const cx = w / 2, cy = h / 2, sc = (Math.min(w, h) / 2 - 24) / R;
    const P = (n, e) => [cx + e * sc, cy - n * sc];

    ctx.lineWidth = 1;
    ctx.strokeStyle = tok("--grid");
    ctx.beginPath();
    ctx.moveTo(cx, cy - R * sc); ctx.lineTo(cx, cy + R * sc);
    ctx.moveTo(cx - R * sc, cy); ctx.lineTo(cx + R * sc, cy);
    ctx.stroke();
    ctx.font = `10px ${mono}`; ctx.fillStyle = tok("--ink-3"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    for (const f of [1 / 3, 2 / 3, 1]) {
      ctx.strokeStyle = f === 1 ? tok("--line") : tok("--grid");
      ctx.beginPath(); ctx.arc(cx, cy, R * f * sc, 0, 2 * Math.PI); ctx.stroke();
      const a = -Math.PI / 4;
      ctx.fillText(`${Math.round(R * f)} m`, cx + Math.cos(a) * R * f * sc + 3, cy + Math.sin(a) * R * f * sc - 1);
    }
    ctx.font = `600 11px ${label}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const rr = R * sc + 12;
    [["N", 0, -rr, "--ink"], ["E", rr, 0, "--ink-3"], ["S", 0, rr, "--ink-3"], ["W", -rr, 0, "--ink-3"]]
      .forEach(([t, dx, dy, c]) => { ctx.fillStyle = tok(c); ctx.fillText(t, cx + dx, cy + dy); });

    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, R * sc, 0, 2 * Math.PI); ctx.clip();
    if (S.trail.length > 1) {
      ctx.beginPath();
      S.trail.forEach(([, la, lo], i) => { const [x, y] = P(...ne(la, lo)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.strokeStyle = tok("--ink-3"); ctx.setLineDash([2, 4]); ctx.stroke(); ctx.setLineDash([]);
    }
    if (s.footprint && s.footprint.every(Boolean)) {
      ctx.beginPath();
      s.footprint.forEach(([n, e], i) => { const [x, y] = P(n, e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath();
      ctx.globalAlpha = 0.14; ctx.fillStyle = tok("--accent"); ctx.fill(); ctx.globalAlpha = 1;
      ctx.strokeStyle = tok("--accent"); ctx.lineWidth = 1; ctx.stroke();
    }
    const stateCol = { LOCKED: "--lock", ACQUIRING: "--acq", LOST: "--acq" }[s.lock.state] || "--cand";
    const tgt = s.tracks.find((t) => t.role === "target" && t.geo);
    if (tgt) {
      const [x, y] = P(...ne(tgt.geo.lat, tgt.geo.lon));
      ctx.strokeStyle = tok(stateCol); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.restore();
    for (const t of s.tracks) {
      if (!t.geo) continue;
      const [x, y] = P(...ne(t.geo.lat, t.geo.lon));
      ctx.lineWidth = 2; ctx.strokeStyle = tok("--plot");
      if (t.role === "target") {
        ctx.fillStyle = tok(stateCol);
        ctx.beginPath(); ctx.moveTo(x, y - 8); ctx.lineTo(x + 8, y); ctx.lineTo(x, y + 8); ctx.lineTo(x - 8, y); ctx.closePath();
        ctx.fill(); ctx.stroke();
      } else if (t.role === "candidate") {
        ctx.strokeStyle = tok("--cand"); ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
      } else {
        ctx.fillStyle = tok("--track"); ctx.fillRect(x - 4.5, y - 4.5, 9, 9); ctx.strokeRect(x - 4.5, y - 4.5, 9, 9);
      }
      ctx.fillStyle = tok("--ink"); ctx.font = `11px ${mono}`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(t.role === "target" ? `#${t.id}  ${Math.round(t.geo.range_m)} m` : `#${t.id}`, x + 11, y - 8);
    }
    const hd = ((v.heading != null ? v.heading : v.yaw) || 0) * Math.PI / 180;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(hd);
    ctx.beginPath(); ctx.moveTo(0, -11); ctx.lineTo(8, 9); ctx.lineTo(0, 4); ctx.lineTo(-8, 9); ctx.closePath();
    ctx.fillStyle = tok("--ink"); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = tok("--plot"); ctx.stroke();
    ctx.restore();
    setText("map-meta", `North up · rings ${Math.round(R / 3)} m`);
  }

  /* ------------------------------------------------------------------ main loop */
  const errFmt = (v, unit) => v.toFixed(unit ? 3 : 2);
  const latFmt = (v, unit) => (unit ? `${v.toFixed(1)} ms` : `${Math.round(v)}`);
  function frame() {
    if (dirty) {
      dirty = false;
      drawSpark("sp-err", S.err, { fmt: errFmt, floor: 0.1, none: "no target" });
      drawSpark("sp-lat", S.lat, { fmt: latFmt, floor: 10, none: "—" });
      drawMap();
    }
    requestAnimationFrame(frame);
  }
  setInterval(() => { setText("clock", new Date().toISOString().slice(11, 19) + " UTC"); }, 1000);
  connect();
  requestAnimationFrame(frame);
})();
