/* SkyTrack GCS - operator console.
 * State arrives over Server-Sent Events (or polling with ?poll), video as MJPEG
 * (or polled snapshots). Commands go to POST /api/command as JSON.
 * With <html data-replay> the page instead replays a recorded run (states.json + replay.mp4),
 * as built by scripts/export_replay.py for static hosting. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const root = document.documentElement;
  const params = new URLSearchParams(location.search);
  const POLL = params.has("poll");
  const REPLAY = root.hasAttribute("data-replay");
  const WINDOW_S = 30;
  const M_PER_DEG = 111320;
  const D2R = Math.PI / 180;
  const S = { last: null, err: [], trail: [], online: false, seq: -1, html: {}, swapped: false, hover: null };
  let dirty = true;
  let toastTimer = 0;

  const tok = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const sgn = (v, d = 1) => (v == null ? "—" : (v < 0 ? "-" : "+") + Math.abs(v).toFixed(d));
  const deg3 = (v) => String(Math.round(((v % 360) + 360) % 360) % 360).padStart(3, "0");
  const setText = (id, text) => { const el = $(id); if (el.textContent !== text) el.textContent = text; };
  const setHTML = (id, html) => { if (S.html[id] !== html) { S.html[id] = html; $(id).innerHTML = html; } };
  const setLed = (id, s) => { const el = $(id); if (el.dataset.s !== s) el.dataset.s = s; };

  function dms(value, pos, neg, width) {
    const v = Math.abs(value), d = Math.floor(v), m = Math.floor((v - d) * 60);
    const s = (v - d - m / 60) * 3600;
    return `${String(d).padStart(width, "0")}°${String(m).padStart(2, "0")}'${s.toFixed(2).padStart(5, "0")}"${value >= 0 ? pos : neg}`;
  }
  const latDms = (v) => dms(v, "N", "S", 2);
  const lonDms = (v) => dms(v, "E", "W", 3);
  function bearingRange(lat0, lon0, lat, lon) {
    const n = (lat - lat0) * M_PER_DEG, e = (lon - lon0) * M_PER_DEG * Math.cos(lat0 * D2R);
    return [((Math.atan2(e, n) / D2R) + 360) % 360, Math.hypot(n, e)];
  }

  /* ------------------------------------------------------------------ theme */
  function setTheme(t) {
    root.dataset.theme = t;
    $("theme-toggle").textContent = t === "day" ? "Night" : "Day";
    try { localStorage.setItem("skytrack-theme", t); } catch (_) { /* storage unavailable */ }
    dirty = true;
  }
  let saved = null;
  try { saved = localStorage.getItem("skytrack-theme"); } catch (_) { /* storage unavailable */ }
  setTheme((params.get("theme") || saved) === "day" ? "day" : "night");
  $("theme-toggle").addEventListener("click", () => setTheme(root.dataset.theme === "day" ? "night" : "day"));

  /* ------------------------------------------------------------------ commands */
  function toast(msg, sticky = false) {
    const t = $("toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    if (!sticky) toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
  }
  async function send(cmd) {
    if (REPLAY || (S.last && S.last.read_only)) {
      toast(`${REPLAY ? "Recorded run" : "Read-only demo"}. Run SkyTrack locally to designate targets and steer the gimbal.`);
      return;
    }
    try {
      const r = await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cmd) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.ok) toast(j.error || `Command rejected (HTTP ${r.status})`);
    } catch (_) {
      toast("SkyTrack is not reachable. Check that `skytrack console` is still running.");
    }
  }

  /* ------------------------------------------------------------------ video / map views */
  const video = $("video"), map = $("map");
  // MJPEG by default; polled snapshots with ?poll, or when a reverse proxy holds the stream back
  let snap = POLL, snapBusy = false;
  function startVideo() {
    if (snap || REPLAY) return;
    video.src = "/stream.mjpg?ts=" + Date.now();
    setTimeout(() => { if (!snap && S.online && !video.naturalWidth) snap = true; }, 8000);
  }
  function nextSnapshot(seq) {
    if (!snap || snapBusy || seq == null || seq === S.seq) return;
    S.seq = seq;
    snapBusy = true;
    video.src = "/api/snapshot.jpg?seq=" + seq;
  }
  video.addEventListener("load", () => { snapBusy = false; $("video-offline").hidden = true; });
  video.addEventListener("loadeddata", () => { $("video-offline").hidden = true; });  // replay <video>
  video.addEventListener("error", () => {
    snapBusy = false;
    $("video-offline").hidden = false;
    if (!snap) setTimeout(startVideo, 1500);
  });

  function layoutViews() {
    const main = $("main-slot"), pip = $("pip-slot");
    main.appendChild(S.swapped ? map : video);
    pip.appendChild(S.swapped ? video : map);
    main.classList.toggle("is-video", !S.swapped);
    setText("pip-name", S.swapped ? "Video" : "Map");
    $("video-offline").style.display = S.swapped ? "none" : "";
    dirty = true;
  }
  $("pip-swap").addEventListener("click", (e) => { e.stopPropagation(); S.swapped = !S.swapped; layoutViews(); });
  layoutViews();

  $("main-slot").addEventListener("click", (e) => {
    const s = S.last;
    if (!s || S.swapped) return;
    const r = video.getBoundingClientRect();
    const [iw, ih] = s.size;
    const k = Math.min(r.width / iw, r.height / ih), dw = iw * k, dh = ih * k;
    const x = (e.clientX - r.left - (r.width - dw) / 2) / dw, y = (e.clientY - r.top - (r.height - dh) / 2) / dh;
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) send({ cmd: "designate", x, y });
  });

  /* ------------------------------------------------------------------ controls */
  $("btn-release").addEventListener("click", () => send({ cmd: "release" }));
  $("btn-pause").addEventListener("click", () => send({ cmd: "toggle_pause" }));
  $("seg-gimbal").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-v]");
    if (b) send({ cmd: "follow", on: b.dataset.v === "follow" });
  });
  $("seg-acq").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-v]");
    if (b) send({ cmd: "auto_designation", on: b.dataset.v === "auto" });
  });
  $("tracks-body").addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-id]");
    if (row) send({ cmd: "designate_id", id: Number(row.dataset.id) });
  });
  addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest("input, textarea, select")) return;
    if (e.target.closest("button") && (e.code === "Space" || e.key === "Enter")) return;
    const k = e.key.toLowerCase(), s = S.last;
    if (k === "r") send({ cmd: "release" });
    else if (e.code === "Space") { e.preventDefault(); send({ cmd: "toggle_pause" }); }
    else if (k === "f" && s && s.gimbal.available) send({ cmd: "follow", on: !s.gimbal.follow });
    else if (k === "a" && s) send({ cmd: "auto_designation", on: !s.lock.auto });
    else if (k === "m") { S.swapped = !S.swapped; layoutViews(); }
  });
  const strip = $("sp-err");
  strip.addEventListener("mousemove", (e) => { S.hover = e.offsetX; dirty = true; });
  strip.addEventListener("mouseleave", () => { S.hover = null; dirty = true; });
  addEventListener("resize", () => { dirty = true; });

  /* ------------------------------------------------------------------ data */
  function setOnline(on) {
    if (on === S.online) return;
    S.online = on;
    if (on) { $("toast").hidden = true; startVideo(); } else toast("Connection to SkyTrack lost. Reconnecting…", true);
  }
  async function loadHistory() {
    try {
      const h = await (await fetch("/api/history", { cache: "no-store" })).json();
      const now = performance.now() / 1000;
      S.err = h.err.filter((r) => r[0] <= WINDOW_S).map(([age, v]) => [now - age, v]);
      S.trail = h.trail.map(([age, la, lo]) => [now - age, la, lo]);
      dirty = true;
    } catch (_) { /* start with empty charts */ }
  }
  async function replay() {
    let states;
    try { states = await (await fetch("states.json")).json(); } catch (_) { toast("The recorded run could not be loaded.", true); return; }
    if (!states.length) return;
    const rate = 10;  // states per second of video
    let last = -1;
    setOnline(true);
    video.play().catch(() => { /* autoplay blocked: starts on the first click */ });
    document.addEventListener("click", () => { if (video.paused) video.play().catch(() => {}); }, { once: true });
    const step = () => {
      const i = Math.min(states.length - 1, Math.floor((video.currentTime || 0) * rate));
      if (i !== last) {
        if (i < last) { S.err.length = 0; S.trail.length = 0; }  // video looped
        last = i;
        ingest(states[i]);
      }
      requestAnimationFrame(step);
    };
    step();
  }
  async function connect() {
    if (REPLAY) return replay();
    await loadHistory();
    if (POLL) {
      const tick = async () => {
        try {
          const s = await (await fetch("/api/state", { cache: "no-store" })).json();
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
    es.onmessage = (m) => {
      setOnline(true);
      let s;
      try { s = JSON.parse(m.data); } catch (_) { return; /* partial */ }
      nextSnapshot(s.seq);
      ingest(s);
    };
    es.onerror = () => setOnline(false);
  }
  function ingest(s) {
    if (!s || !s.lock) return;
    S.last = s;
    const now = performance.now() / 1000;
    const e = s.lock.error;
    S.err.push([now, e ? Math.hypot(e[0], e[1]) : null]);
    while (S.err.length && now - S.err[0][0] > WINDOW_S) S.err.shift();
    const v = s.vehicle;
    if (v && v.connected && v.lat != null) {
      const last = S.trail[S.trail.length - 1];
      if (!last || now - last[0] > 0.5) S.trail.push([now, v.lat, v.lon]);
      while (S.trail.length && now - S.trail[0][0] > 180) S.trail.shift();
    }
    renderToolbar(s);
    renderPayload(s);
    renderTelemetry(s);
    renderTarget(s);
    renderTracks(s);
    renderLog(s);
    dirty = true;
  }

  /* ------------------------------------------------------------------ panes */
  const GPS_FIX = { 0: "No GPS", 1: "No fix", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK float", 6: "RTK fixed" };
  function renderToolbar(s) {
    const v = s.vehicle, up = v && v.connected;
    setText("v-mode", up ? v.mode || "—" : "—");
    const arm = $("v-arm");
    setText("v-arm", up ? (v.armed ? "ARMED" : "Disarmed") : "—");
    arm.dataset.armed = String(!!(up && v.armed));
    if (up && v.gps_fix != null) {
      setText("v-gps", `${GPS_FIX[v.gps_fix] || v.gps_fix}${v.gps_sats != null ? " · " + v.gps_sats + " sat" : ""}`);
      setLed("led-gps", v.gps_fix >= 3 ? "ok" : v.gps_fix === 2 ? "warn" : "crit");
    } else { setText("v-gps", "—"); setLed("led-gps", ""); }
    if (up && v.battery_v != null) {
      setText("v-bat", `${fmt(v.battery_v, 1)} V${v.battery_pct != null ? " · " + v.battery_pct + "%" : ""}`);
      const p = v.battery_pct;
      setLed("led-bat", p == null ? "ok" : p > 30 ? "ok" : p > 15 ? "warn" : "crit");
    } else { setText("v-bat", "—"); setLed("led-bat", ""); }
    if (up) { setText("v-link", "OK"); setLed("led-link", "ok"); }
    else if (s.gimbal.available) { setText("v-link", "No heartbeat"); setLed("led-link", "crit"); }
    else { setText("v-link", "None"); setLed("led-link", ""); }
    setText("v-src", s.source || "—");
    setText("v-proc", `${fmt(s.fps, 1)} fps · ${fmt(s.latency_ms, 1)} ms`);
    setText("video-meta", `EO · ${s.detector} · ${s.size[0]}x${s.size[1]} · F${String(s.frame).padStart(6, "0")}`);
  }

  const MODE = { SEARCHING: ["SEARCH", ""], ACQUIRING: ["ACQUIRE", "warn"], LOCKED: ["TRACK", "ok"], LOST: ["COAST", "warn"] };
  function renderPayload(s) {
    const L = s.lock;
    const [label, led] = L.enabled ? MODE[L.state] : ["OFF", ""];
    setText("trk-state", label + (s.paused ? " · FROZEN" : ""));
    setLed("led-trk", led);
    const pause = $("btn-pause");
    pause.setAttribute("aria-pressed", String(s.paused));
    pause.querySelector(".lbl").textContent = s.paused ? "Unfreeze" : "Freeze";
    const ro = !!s.read_only;
    document.body.classList.toggle("ro", ro);
    $("btn-release").disabled = ro || !L.enabled;
    pause.disabled = ro;
    for (const b of $("seg-gimbal").querySelectorAll("button")) {
      b.disabled = ro || !s.gimbal.available;
      b.setAttribute("aria-pressed", String(s.gimbal.available && (b.dataset.v === "follow") === s.gimbal.follow));
    }
    for (const b of $("seg-acq").querySelectorAll("button")) {
      b.disabled = ro || !L.enabled;
      b.setAttribute("aria-pressed", String((b.dataset.v === "auto") === L.auto));
    }
    const hint = $("pb-hint");
    hint.textContent = !ro ? "Click a target in the video to designate it"
      : REPLAY ? "Recorded simulation run, operator controls are off" : "Read-only demo, operator controls are off";
    hint.hidden = !(ro || (L.enabled && !L.auto && L.designated_id == null && s.tracks.length));
  }

  function renderTelemetry(s) {
    const v = s.vehicle, up = v && v.connected;
    setText("tele-meta", !v ? "No link" : up ? "MAVLink" : "No heartbeat");
    setText("t-agl", up && v.alt_rel != null ? `${fmt(v.alt_rel, 1)} m` : "—");
    setText("t-msl", up && v.alt_amsl != null ? `${fmt(v.alt_amsl, 0)} m` : "—");
    setText("t-gs", up && v.groundspeed != null ? `${fmt(v.groundspeed, 1)} m/s` : "—");
    setText("t-hdg", up && v.heading != null ? `${deg3(v.heading)}°` : "—");
    const az = up && v.heading != null ? v.heading + (v.gimbal_yaw || 0) : null;
    setText("t-az", az != null ? `${deg3(az)}°` : "—");
    setText("t-el", up && v.gimbal_pitch != null ? `${sgn(v.gimbal_pitch, 1)}°` : "—");
    setText("t-pos", up && v.lat != null ? `${latDms(v.lat)}  ${lonDms(v.lon)}` : "—");
    setText("inst-meta", up ? `R ${sgn(v.roll, 1)}°  P ${sgn(v.pitch, 1)}°` : "—");
  }

  function renderTarget(s) {
    const L = s.lock, g = s.target_geo, v = s.vehicle;
    const t = s.tracks.find((x) => x.id === L.track_id);
    setText("tgt-meta", L.enabled ? (L.auto ? "Auto acquire" : "Manual") : "Lock disabled");
    setText("g-track", t ? `${t.id} · ${t.label} · ${t.score.toFixed(2)}` : L.candidate_id != null ? `${L.candidate_id} (candidate)` : "—");
    const st = $("g-status");
    const status = !L.enabled ? "Disabled" : L.state === "LOCKED" ? `TRACK ${fmt(L.duration, 1)} s`
      : L.state === "ACQUIRING" ? `ACQUIRE ${Math.round(L.progress * 100)}% of ${fmt(L.acquire_s, 1)} s`
        : L.state === "LOST" ? "COAST (reacquiring)" : "SEARCH";
    setText("g-status", status);
    st.dataset.s = L.state === "LOCKED" ? "lock" : L.state === "ACQUIRING" || L.state === "LOST" ? "acq" : "";
    setText("g-err", L.error ? `X ${sgn(L.error[0], 3)}  Y ${sgn(L.error[1], 3)}` : "—");
    setText("g-lat", g ? latDms(g.lat) : "—");
    setText("g-lon", g ? lonDms(g.lon) : "—");
    setText("g-rng", g ? `${fmt(g.range_m, 0)} m ground${g.slant_m != null ? " · " + fmt(g.slant_m, 0) + " m slant" : ""}` : "—");
    if (g && v && v.lat != null) setText("g-brg", `${deg3(bearingRange(v.lat, v.lon, g.lat, g.lon)[0])}° true`);
    else setText("g-brg", "—");
    const link = $("g-map");
    link.hidden = !g;
    if (g) link.href = `https://www.openstreetmap.org/?mlat=${g.lat}&mlon=${g.lon}#map=18/${g.lat}/${g.lon}`;
  }

  const ROLE = { target: "Target", candidate: "Candidate", track: "Track" };
  function renderTracks(s) {
    setText("tracks-meta", `${s.tracks.length} active`);
    const v = s.vehicle;
    const html = !s.tracks.length
      ? `<tr class="empty"><td colspan="7">No confirmed tracks. A track is listed after the detector has seen it in three frames.</td></tr>`
      : s.tracks.map((t) => {
        const br = t.geo && v && v.lat != null ? deg3(bearingRange(v.lat, v.lon, t.geo.lat, t.geo.lon)[0]) : "—";
        return `<tr data-id="${t.id}" data-role="${t.role}" title="Designate track ${t.id}">`
          + `<td>${t.id}</td><td>${esc(t.label)}</td><td class="num">${t.score.toFixed(2)}</td>`
          + `<td class="num">${fmt(t.age_s, 1)}</td><td class="num">${t.geo ? fmt(t.geo.range_m, 0) : "—"}</td>`
          + `<td class="num">${br}</td><td class="st-${t.role}">${ROLE[t.role]}</td></tr>`;
      }).join("");
    setHTML("tracks-body", html);
  }

  const MSG = {
    lock_acquired: ["NOTICE", (e) => `Lock acquired on track ${e.track_id}`],
    lock_lost: ["WARNING", (e) => `Lock lost on track ${e.track_id}${e.detail ? " after " + e.detail : ""}`],
    designated: ["INFO", (e) => `Operator designated track ${e.track_id}`],
    released: ["INFO", (e) => `Lock released${e.track_id != null ? " (track " + e.track_id + ")" : ""}`],
    no_track: ["INFO", (e) => e.detail || "No track at that position"],
    link_up: ["INFO", (e) => `Vehicle link up${e.detail ? ", mode " + e.detail : ""}`],
    link_lost: ["CRITICAL", () => "Vehicle link lost: no heartbeat"],
    paused: ["INFO", () => "Video frozen"], resumed: ["INFO", () => "Video resumed"],
    follow_on: ["INFO", () => "Gimbal follow on"], follow_off: ["INFO", () => "Gimbal follow off, gimbal holding"],
    auto_on: ["INFO", () => "Auto acquisition on"], auto_off: ["INFO", () => "Auto acquisition off, operator designates"],
    session_start: ["INFO", (e) => `Session started (${e.detail || "video"})`],
    source_end: ["WARNING", () => "Video source ended"],
  };
  function renderLog(s) {
    setText("msg-meta", `${s.events.length}`);
    const html = !s.events.length ? `<li class="empty">Lock, designation and link messages appear here.</li>`
      : s.events.slice().reverse().map((e) => {
        const [sev, text] = MSG[e.kind] || ["INFO", () => e.kind];
        return `<li><time>${esc(e.utc)}</time><span class="sev-${sev.toLowerCase()}">${sev}</span><span class="msg">${esc(text(e))}</span></li>`;
      }).join("");
    setHTML("log", html);
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
  function seg(ctx, x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }

  /* Attitude indicator: sky / ground ball, pitch ladder, roll scale, fixed aircraft symbol. */
  function drawADI() {
    const { ctx, w, h } = fit($("adi"));
    ctx.clearRect(0, 0, w, h);
    const v = S.last && S.last.vehicle, up = v && v.connected;
    const roll = up ? v.roll || 0 : 0, pitch = up ? v.pitch || 0 : 0;
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 2, ppd = R / 32;
    const mono = tok("--mono");
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.clip();
    ctx.translate(cx, cy); ctx.rotate(-roll * D2R); ctx.translate(0, pitch * ppd);
    ctx.fillStyle = up ? tok("--sky") : "#3a3d40"; ctx.fillRect(-2 * R, -4 * R, 4 * R, 4 * R);
    ctx.fillStyle = up ? tok("--ground") : "#2c2e30"; ctx.fillRect(-2 * R, 0, 4 * R, 4 * R);
    ctx.strokeStyle = "#ffffff"; ctx.fillStyle = "#ffffff"; ctx.lineWidth = 1.5;
    seg(ctx, -2 * R, 0, 2 * R, 0);
    ctx.lineWidth = 1; ctx.font = `9px ${mono}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let p = -30; p <= 30; p += 5) {
      if (!p) continue;
      const y = -p * ppd, half = p % 10 ? R * 0.1 : R * 0.22;
      seg(ctx, -half, y, half, y);
      if (p % 10 === 0) { ctx.fillText(String(Math.abs(p)), -half - 9, y); ctx.fillText(String(Math.abs(p)), half + 9, y); }
    }
    ctx.restore();

    ctx.save(); ctx.translate(cx, cy);
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
    const rr = R * 0.84;
    ctx.beginPath(); ctx.arc(0, 0, rr, -150 * D2R, -30 * D2R); ctx.stroke();
    for (const a of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {
      const t = (a - 90) * D2R, len = a % 30 === 0 ? R * 0.1 : R * 0.06;
      seg(ctx, Math.cos(t) * rr, Math.sin(t) * rr, Math.cos(t) * (rr - len), Math.sin(t) * (rr - len));
    }
    ctx.rotate(-roll * D2R);
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.moveTo(0, -rr + 2); ctx.lineTo(-5, -rr + 10); ctx.lineTo(5, -rr + 10); ctx.closePath(); ctx.fill();
    ctx.restore();

    ctx.strokeStyle = "#f2b01e"; ctx.lineWidth = 3; ctx.lineCap = "square";
    seg(ctx, cx - R * 0.55, cy, cx - R * 0.2, cy); seg(ctx, cx - R * 0.2, cy, cx - R * 0.2, cy + R * 0.08);
    seg(ctx, cx + R * 0.2, cy, cx + R * 0.55, cy); seg(ctx, cx + R * 0.2, cy, cx + R * 0.2, cy + R * 0.08);
    ctx.fillStyle = "#f2b01e"; ctx.fillRect(cx - 2, cy - 2, 4, 4);
    ctx.lineCap = "butt"; ctx.lineWidth = 2; ctx.strokeStyle = tok("--edge");
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
    if (!up) { ctx.fillStyle = "#c8ccd0"; ctx.font = `11px ${tok("--ui")}`; ctx.textAlign = "center"; ctx.fillText("No attitude", cx, cy + R * 0.45); }
  }

  /* Heading indicator: rotating compass card, sensor field of view wedge, target bearing bug. */
  function drawHSI() {
    const { ctx, w, h } = fit($("hsi"));
    ctx.clearRect(0, 0, w, h);
    const s = S.last, v = s && s.vehicle, up = v && v.connected;
    const hdg = up ? (v.heading != null ? v.heading : v.yaw || 0) : 0;
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 2, mono = tok("--mono");
    ctx.fillStyle = "#151719"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.fill();
    ctx.save(); ctx.translate(cx, cy);
    if (up && v.gimbal_yaw != null) {
      const half = (s.camera_hfov || 60) / 2, a = (v.gimbal_yaw - 90) * D2R;
      ctx.fillStyle = "rgba(61, 134, 216, 0.28)"; ctx.strokeStyle = "rgba(110, 170, 235, 0.9)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, R * 0.78, a - half * D2R, a + half * D2R); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    ctx.rotate(-hdg * D2R);
    ctx.strokeStyle = "#e8e8e8"; ctx.fillStyle = "#e8e8e8"; ctx.lineWidth = 1;
    ctx.font = `10px ${mono}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let d = 0; d < 360; d += 5) {
      const t = (d - 90) * D2R, len = d % 10 ? R * 0.06 : R * 0.11;
      seg(ctx, Math.cos(t) * R * 0.97, Math.sin(t) * R * 0.97, Math.cos(t) * (R * 0.97 - len), Math.sin(t) * (R * 0.97 - len));
      if (d % 30 === 0) {
        const label = { 0: "N", 90: "E", 180: "S", 270: "W" }[d] || String(d / 10);
        ctx.save(); ctx.translate(Math.cos(t) * R * 0.74, Math.sin(t) * R * 0.74); ctx.rotate(d * D2R);
        ctx.font = d % 90 ? `10px ${mono}` : `bold 11px ${mono}`; ctx.fillText(label, 0, 0); ctx.restore();
      }
    }
    const g = s && s.target_geo;
    if (up && g && v.lat != null) {
      const b = bearingRange(v.lat, v.lon, g.lat, g.lon)[0], t = (b - 90) * D2R, r = R * 0.9;
      ctx.fillStyle = "#e5484d"; ctx.save(); ctx.translate(Math.cos(t) * r, Math.sin(t) * r); ctx.rotate(b * D2R);
      ctx.beginPath(); ctx.moveTo(0, -6); ctx.lineTo(5, 0); ctx.lineTo(0, 6); ctx.lineTo(-5, 0); ctx.closePath(); ctx.fill(); ctx.restore();
    }
    ctx.restore();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
    seg(ctx, cx, cy - R * 0.22, cx, cy + R * 0.18); seg(ctx, cx - R * 0.18, cy - R * 0.02, cx + R * 0.18, cy - R * 0.02);
    seg(ctx, cx - R * 0.08, cy + R * 0.16, cx + R * 0.08, cy + R * 0.16);
    ctx.fillStyle = "#f2b01e"; ctx.beginPath(); ctx.moveTo(cx, cy - R + 1); ctx.lineTo(cx - 6, cy - R - 0 + 10); ctx.lineTo(cx + 6, cy - R + 10); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#000000"; ctx.fillRect(cx - 19, cy + R * 0.42 - 9, 38, 18);
    ctx.strokeStyle = "#e8e8e8"; ctx.lineWidth = 1; ctx.strokeRect(cx - 19, cy + R * 0.42 - 9, 38, 18);
    ctx.fillStyle = "#ffffff"; ctx.font = `12px ${mono}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(up ? deg3(hdg) : "---", cx, cy + R * 0.42 + 1);
    ctx.lineWidth = 2; ctx.strokeStyle = tok("--edge"); ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
  }

  /* Map: north up, metric grid, sensor footprint, track symbols, vehicle and its track. */
  const GRID = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000];
  function drawMap() {
    const { ctx, w, h } = fit(map);
    const s = S.last, v = s && s.vehicle;
    ctx.fillStyle = tok("--bg"); ctx.fillRect(0, 0, w, h);
    const small = w < 420, mono = tok("--mono"), fs = small ? 9 : 11;
    if (!v || !v.connected || v.lat == null) {
      ctx.fillStyle = tok("--text-3"); ctx.font = `${small ? 11 : 13}px ${tok("--ui")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(s && s.gimbal.available ? "Waiting for vehicle position" : "No vehicle link", w / 2, h / 2);
      return;
    }
    const k = Math.cos(v.lat * D2R);
    const ne = (lat, lon) => [(lat - v.lat) * M_PER_DEG, (lon - v.lon) * M_PER_DEG * k];
    let ext = 0;
    for (const t of s.tracks) if (t.geo) ext = Math.max(ext, t.geo.range_m * 1.3);
    if (!ext && s.footprint) {
      const d = s.footprint.filter(Boolean).map((p) => Math.hypot(p[0], p[1]));
      if (d.length) ext = Math.min(...d) * 1.6;
    }
    ext = Math.max(ext, 60);
    const sc = (Math.min(w, h) / 2 - 14) / ext;
    const cx = w / 2, cy = h / 2;
    const P = (n, e) => [cx + e * sc, cy - n * sc];
    const step = GRID.find((g) => g * sc >= (small ? 34 : 48)) || 5000;
    ctx.strokeStyle = tok("--grid"); ctx.lineWidth = 1;
    const e0 = Math.floor((-cx / sc) / step) * step, n0 = Math.floor((-(h - cy) / sc) / step) * step;
    for (let e = e0; e * sc + cx <= w; e += step) { const x = Math.round(cx + e * sc) + 0.5; seg(ctx, x, 0, x, h); }
    for (let n = n0; cy - n * sc >= 0; n += step) { const y = Math.round(cy - n * sc) + 0.5; seg(ctx, 0, y, w, y); }

    if (S.trail.length > 1) {
      ctx.strokeStyle = tok("--text-3"); ctx.lineWidth = 1; ctx.beginPath();
      S.trail.forEach(([, la, lo], i) => { const [x, y] = P(...ne(la, lo)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    if (s.footprint && s.footprint.every(Boolean)) {
      ctx.beginPath();
      s.footprint.forEach(([n, e], i) => { const [x, y] = P(n, e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fillStyle = "rgba(61, 134, 216, 0.16)"; ctx.fill();
      ctx.strokeStyle = tok("--accent"); ctx.lineWidth = 1; ctx.stroke();
    }
    const tgt = s.tracks.find((t) => t.role === "target" && t.geo);
    if (tgt) {
      const [x, y] = P(...ne(tgt.geo.lat, tgt.geo.lon));
      ctx.strokeStyle = tok("--text-2"); ctx.setLineDash([4, 3]); seg(ctx, cx, cy, x, y); ctx.setLineDash([]);
    }
    ctx.font = `${fs}px ${mono}`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (const t of s.tracks) {
      if (!t.geo) continue;
      const [x, y] = P(...ne(t.geo.lat, t.geo.lon));
      if (t.role === "target") {
        ctx.fillStyle = tok("--crit"); ctx.strokeStyle = "#000000"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.closePath(); ctx.fill(); ctx.stroke();
      } else {  // unknown track: quatrefoil frame
        ctx.fillStyle = tok("--track");
        for (const [dx, dy] of [[-2.6, 0], [2.6, 0], [0, -2.6], [0, 2.6]]) { ctx.beginPath(); ctx.arc(x + dx, y + dy, 3.2, 0, 2 * Math.PI); ctx.fill(); }
      }
      ctx.fillStyle = tok("--text");
      ctx.fillText(t.role === "target" ? `${t.id} ${Math.round(t.geo.range_m)}m` : `${t.id}`, x + 9, y - 7);
    }
    const hd = (v.heading != null ? v.heading : v.yaw || 0) * D2R;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(hd);
    ctx.beginPath(); ctx.moveTo(0, -10); ctx.lineTo(7, 8); ctx.lineTo(0, 4); ctx.lineTo(-7, 8); ctx.closePath();
    ctx.fillStyle = "#e5484d"; ctx.fill(); ctx.strokeStyle = "#000000"; ctx.lineWidth = 1; ctx.stroke();
    ctx.restore();
    // scale bar and north arrow
    const bar = step * sc, bx = 8, by = h - 10;
    ctx.strokeStyle = tok("--text"); ctx.lineWidth = 2; seg(ctx, bx, by, bx + bar, by);
    ctx.lineWidth = 1; seg(ctx, bx, by - 4, bx, by + 2); seg(ctx, bx + bar, by - 4, bx + bar, by + 2);
    ctx.fillStyle = tok("--text"); ctx.textBaseline = "bottom"; ctx.fillText(`${step} m`, bx + 2, by - 4);
    ctx.textAlign = "center"; ctx.font = `bold ${fs + 1}px ${mono}`; ctx.fillText("N", w - 14, 22);
    ctx.beginPath(); ctx.moveTo(w - 14, 24); ctx.lineTo(w - 18, 34); ctx.lineTo(w - 10, 34); ctx.closePath(); ctx.fill();
  }

  /* Tracking error strip chart: one series, recessive grid, hover readout. */
  function drawStrip() {
    const { ctx, w, h } = fit(strip);
    ctx.fillStyle = tok("--bg"); ctx.fillRect(0, 0, w, h);
    const now = performance.now() / 1000, L = 28, T = 4, B = 4, R = 4, pw = w - L - R, ph = h - T - B;
    const vmax = Math.max(0.2, ...S.err.filter((p) => p[1] != null).map((p) => p[1])) > 0.5 ? 1 : 0.5;
    const X = (t) => L + pw * (1 - (now - t) / WINDOW_S), Y = (v) => T + ph * (1 - Math.min(v, vmax) / vmax);
    const mono = tok("--mono");
    ctx.strokeStyle = tok("--grid"); ctx.lineWidth = 1; ctx.font = `9px ${mono}`; ctx.fillStyle = tok("--text-3");
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const f of [0, 0.5, 1]) { const y = Math.round(T + ph * (1 - f)) + 0.5; seg(ctx, L, y, w - R, y); ctx.fillText((vmax * f).toFixed(2), L - 4, y); }
    for (let k = 1; k < 6; k++) { const x = Math.round(L + pw * k / 6) + 0.5; seg(ctx, x, T, x, T + ph); }
    ctx.strokeStyle = tok("--accent"); ctx.lineWidth = 1.5; ctx.beginPath();
    let pen = false;
    for (const [t, v] of S.err) {
      if (v == null) { pen = false; continue; }
      const x = X(t), y = Y(v);
      if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
    }
    ctx.stroke();
    if (S.hover != null && S.err.length && S.hover >= L) {
      const tt = now - (1 - (S.hover - L) / pw) * WINDOW_S;
      let best = S.err[0];
      for (const p of S.err) if (Math.abs(p[0] - tt) < Math.abs(best[0] - tt)) best = p;
      ctx.strokeStyle = tok("--text-3"); ctx.setLineDash([2, 2]); seg(ctx, S.hover + 0.5, T, S.hover + 0.5, T + ph); ctx.setLineDash([]);
      const label = `${best[1] != null ? best[1].toFixed(3) : "no target"}  ${(now - best[0]).toFixed(1)} s ago`;
      ctx.font = `10px ${mono}`;
      const tw = ctx.measureText(label).width + 8, bx = clamp(S.hover + 6, L, w - R - tw);
      ctx.fillStyle = tok("--pane-2"); ctx.fillRect(bx, T, tw, 14); ctx.strokeStyle = tok("--line"); ctx.strokeRect(bx, T, tw, 14);
      ctx.fillStyle = tok("--text"); ctx.textAlign = "left"; ctx.fillText(label, bx + 4, T + 7.5);
    }
  }

  /* ------------------------------------------------------------------ loop */
  function frame() {
    if (dirty) { dirty = false; drawADI(); drawHSI(); drawMap(); drawStrip(); }
    requestAnimationFrame(frame);
  }
  setInterval(() => { setText("v-clock", new Date().toISOString().slice(11, 19) + "Z"); }, 1000);
  connect();
  requestAnimationFrame(frame);
})();
