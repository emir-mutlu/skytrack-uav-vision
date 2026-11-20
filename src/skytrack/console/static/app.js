/* SkyTrack GCS - sensor operator station.
 * Live: state over Server-Sent Events (or polling with ?poll), video as MJPEG (or polled
 * snapshots), operator commands as JSON to POST /api/command.
 * <html data-replay>: replays a recorded run (states.json + replay.mp4), as built by
 * scripts/export_replay.py for static hosting.
 * Colour follows flight-deck practice and is always paired with text or shape. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const root = document.documentElement;
  const params = new URLSearchParams(location.search);
  const POLL = params.has("poll");
  const REPLAY = root.hasAttribute("data-replay");
  const D2R = Math.PI / 180;
  const M_PER_DEG = 111320;
  const RANGES = [125, 250, 500, 1000, 2000, 4000];
  const C = {};
  const S = {
    last: null, online: false, seq: -1, swapped: false, trail: [], vs: null, prevAlt: null,
    rangeIdx: 2, rangeAuto: true, northUp: false, centred: false, html: {}, videoOk: false,
  };
  let dirty = true;
  let toastTimer = 0;

  /* ------------------------------------------------------------------ helpers */
  function readColours() {
    const cs = getComputedStyle(root);
    for (const k of ["txt", "txt-2", "txt-3", "line", "line-2", "grn", "cyn", "mag", "amb", "red", "yel",
      "sky", "gnd", "ac", "tape"]) C[k] = cs.getPropertyValue(`--${k}`).trim();
  }
  const font = (px, bold) => `${bold ? 700 : 400} ${px}px "B612 Mono", Consolas, monospace`;
  const pad = (n, w) => String(n).padStart(w, "0");
  const wrap360 = (d) => ((d % 360) + 360) % 360;
  const sgn = (x, d) => (x < 0 ? "-" : "+") + Math.abs(x).toFixed(d);
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const rangeLabel = (m) => (m >= 1000 ? `${m / 1000}K` : String(m));

  function setText(id, text, colour) {
    const el = $(id);
    if (el.textContent !== text) el.textContent = text;
    const c = colour || "";
    if ((el.dataset.c || "") !== c) {
      if (c) el.dataset.c = c;
      else delete el.dataset.c;
    }
  }
  function dms(v, pos, neg) {
    if (v == null) return "---";
    const a = Math.abs(v), d = Math.floor(a), mf = (a - d) * 60, m = Math.floor(mf);
    return `${v < 0 ? neg : pos}${d}°${pad(m, 2)}'${((mf - m) * 60).toFixed(1).padStart(4, "0")}"`;
  }
  function offsetNE(v, lat, lon) {
    return [(lat - v.lat) * M_PER_DEG, (lon - v.lon) * M_PER_DEG * Math.cos(v.lat * D2R)];
  }
  function bearingTo(v, lat, lon) {
    const [n, e] = offsetNE(v, lat, lon);
    return wrap360(Math.atan2(e, n) / D2R);
  }

  /* MGRS grid reference (WGS-84, Transverse Mercator series; same maths as skytrack.geo.mgrs).
   * digits per axis: 4 = 10 m. Truncated, never rounded. */
  const MG_BANDS = "CDEFGHJKLMNPQRSTUVWX";
  const MG_COLS = ["ABCDEFGH", "JKLMNPQR", "STUVWXYZ"];
  const MG_ROWS = "ABCDEFGHJKLMNPQRSTUV";
  function utm(lat, lon) {
    if (!(lat >= -80 && lat < 84)) return null;
    lon = wrap360(lon + 180) - 180;
    let zone = Math.floor((lon + 180) / 6) + 1;
    if (lat >= 56 && lat < 64 && lon >= 3 && lon < 12) zone = 32;
    else if (lat >= 72 && lon >= 0 && lon < 42) zone = lon < 9 ? 31 : lon < 21 ? 33 : lon < 33 ? 35 : 37;
    const a = 6378137, f = 1 / 298.257223563, k0 = 0.9996, e2 = f * (2 - f), ep2 = e2 / (1 - e2);
    const phi = lat * D2R, sp = Math.sin(phi), cp = Math.cos(phi), tp = Math.tan(phi);
    const n = a / Math.sqrt(1 - e2 * sp * sp), t = tp * tp, c = ep2 * cp * cp;
    const q = cp * (lon - ((zone - 1) * 6 - 177)) * D2R;
    const m = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
      - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * Math.sin(2 * phi)
      + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * Math.sin(4 * phi)
      - (35 * e2 ** 3 / 3072) * Math.sin(6 * phi));
    const east = 500000 + k0 * n * (q + (1 - t + c) * q ** 3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * q ** 5 / 120);
    let north = k0 * (m + n * tp * (q * q / 2 + (5 - t + 9 * c + 4 * c * c) * q ** 4 / 24
      + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * q ** 6 / 720));
    if (lat < 0) north += 10000000;
    return { zone, band: MG_BANDS[Math.min(Math.floor((lat + 80) / 8), 19)], east, north };
  }
  function mgrs(lat, lon, digits = 4) {
    const u = lat == null || lon == null ? null : utm(lat, lon);
    if (!u) return "---";
    const col = MG_COLS[(u.zone - 1) % 3][Math.floor(u.east / 100000) - 1];
    const row = MG_ROWS[(Math.floor(u.north / 100000) + (u.zone % 2 === 0 ? 5 : 0)) % 20];
    const div = 10 ** (5 - digits);
    const e = Math.floor(Math.floor(u.east % 100000) / div), nn = Math.floor(Math.floor(u.north % 100000) / div);
    return `${u.zone}${u.band} ${col}${row} ${pad(e, digits)} ${pad(nn, digits)}`;
  }

  /* ------------------------------------------------------------------ brightness */
  function setBrt(mode) {
    root.dataset.brt = mode;
    $("brt").textContent = mode === "ngt" ? "NGT" : "DAY";
    try { localStorage.setItem("skytrack-brt", mode); } catch (_) { /* storage unavailable */ }
  }
  let savedBrt = null;
  try { savedBrt = localStorage.getItem("skytrack-brt"); } catch (_) { /* storage unavailable */ }
  setBrt(savedBrt === "ngt" ? "ngt" : "day");
  $("brt").addEventListener("click", () => setBrt(root.dataset.brt === "ngt" ? "day" : "ngt"));

  /* ------------------------------------------------------------------ commands */
  function toast(msg, sticky) {
    const t = $("toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    if (!sticky) toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
  }
  const viewOnly = () => REPLAY || !!(S.last && S.last.read_only);
  async function send(cmd) {
    if (viewOnly()) {
      toast(`${REPLAY ? "Playback" : "View only"} - controls inhibited. Run SkyTrack locally for control`);
      return;
    }
    try {
      const r = await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cmd) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.ok) toast(j.error || `Command rejected (HTTP ${r.status})`);
    } catch (_) {
      toast("SkyTrack not reachable - check that skytrack console is running");
    }
  }

  /* ------------------------------------------------------------------ video */
  // MJPEG by default; polled snapshots with ?poll, or when a reverse proxy holds the stream back
  const video = $("video");
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
  function videoUp() { snapBusy = false; S.videoOk = true; $("video-offline").hidden = true; }
  video.addEventListener("load", videoUp);
  video.addEventListener("loadeddata", videoUp);  // replay <video>
  video.addEventListener("error", () => {
    snapBusy = false;
    S.videoOk = false;
    $("video-offline").hidden = false;
    if (!snap) setTimeout(startVideo, 1500);
  });

  /* ------------------------------------------------------------------ displays and keys */
  const pageSens = $("page-sens"), pageHsd = $("page-hsd");
  function layoutViews() {  // SWAP moves whole pages, legends included, between the two displays
    $(S.swapped ? "mfd-b" : "mfd-a").appendChild(pageSens);
    $(S.swapped ? "mfd-a" : "mfd-b").appendChild(pageHsd);
    if (REPLAY && video.paused) video.play().catch(() => {});
    dirty = true;
  }
  function swap() { S.swapped = !S.swapped; layoutViews(); }

  $("scr-sens").addEventListener("click", (e) => {
    const s = S.last;
    if (!s) return;
    const r = video.getBoundingClientRect();
    const [iw, ih] = s.size;
    const k = Math.min(r.width / iw, r.height / ih), dw = iw * k, dh = ih * k;
    const x = (e.clientX - r.left - (r.width - dw) / 2) / dw, y = (e.clientY - r.top - (r.height - dh) / 2) / dh;
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) send({ cmd: "designate", x, y });
  });
  $("osb-frz").addEventListener("click", () => send({ cmd: "toggle_pause" }));
  $("osb-brk").addEventListener("click", () => send({ cmd: "release" }));
  $("osb-acq").addEventListener("click", () => { if (S.last) send({ cmd: "auto_designation", on: !S.last.lock.auto }); });
  $("osb-gmb").addEventListener("click", () => { if (S.last) send({ cmd: "follow", on: !S.last.gimbal.follow }); });
  $("osb-swap-a").addEventListener("click", swap);
  $("osb-swap-b").addEventListener("click", swap);
  $("osb-rng-up").addEventListener("click", () => setRange(S.rangeIdx + 1));
  $("osb-rng-dn").addEventListener("click", () => setRange(S.rangeIdx - 1));
  $("osb-orient").addEventListener("click", () => { S.northUp = !S.northUp; renderHsdLegends(); dirty = true; });
  $("osb-dep").addEventListener("click", () => { S.centred = !S.centred; renderHsdLegends(); dirty = true; });
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
    else if (k === "m") swap();
  });
  function setRange(idx) {
    S.rangeAuto = false;
    S.rangeIdx = Math.max(0, Math.min(RANGES.length - 1, idx));
    renderHsdLegends();
    dirty = true;
  }
  if (typeof ResizeObserver === "function") {
    const ro = new ResizeObserver(() => { dirty = true; });
    ro.observe($("pfd"));
    ro.observe($("map"));
  }
  addEventListener("resize", () => { dirty = true; });

  /* ------------------------------------------------------------------ data */
  function setOnline(on) {
    if (on === S.online) return;
    S.online = on;
    if (on) { $("toast").hidden = true; startVideo(); } else toast("Connection to SkyTrack lost - reconnecting", true);
  }
  async function loadHistory() {
    try {
      const h = await (await fetch("/api/history", { cache: "no-store" })).json();
      S.trail = (h.trail || []).map(([, la, lo]) => [la, lo]);
    } catch (_) { /* start with an empty track */ }
  }
  async function replay() {
    let states;
    try { states = await (await fetch("states.json")).json(); } catch (_) { toast("The recorded run could not be loaded", true); return; }
    if (!states.length) return;
    const rate = 10;  // states per second of video
    let last = -1;
    setOnline(true);
    video.play().catch(() => { /* autoplay blocked: starts on the first click */ });
    document.addEventListener("click", () => { if (video.paused) video.play().catch(() => {}); }, { once: true });
    const step = () => {
      const i = Math.min(states.length - 1, Math.floor((video.currentTime || 0) * rate));
      if (i !== last) {
        if (i < last) { S.trail.length = 0; S.prevAlt = null; S.vs = null; }  // video looped
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
          nextSnapshot(s.seq);
          ingest(s);
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
    const v = s.vehicle;
    if (v && v.connected && v.lat != null) {
      const prev = S.trail[S.trail.length - 1];
      if (!prev || Math.hypot(...offsetNE(v, prev[0], prev[1])) > 4) S.trail.push([v.lat, v.lon]);
      if (S.trail.length > 600) S.trail.splice(0, S.trail.length - 600);
      if (v.alt_rel != null) {
        if (S.prevAlt && s.t > S.prevAlt[0]) {
          const raw = (v.alt_rel - S.prevAlt[1]) / (s.t - S.prevAlt[0]);
          S.vs = S.vs == null ? raw : 0.85 * S.vs + 0.15 * raw;
        }
        if (!S.prevAlt || s.t !== S.prevAlt[0]) S.prevAlt = [s.t, v.alt_rel];
      }
    }
    S.last = s;
    renderStatus(s);
    renderOsb(s);
    renderTarget(s);
    renderTracks(s);
    renderCas(s);
    dirty = true;
  }

  /* ------------------------------------------------------------------ text panes */
  const GPS_FIX = ["NO GPS", "NO FIX", "2D", "3D", "DGPS", "RTK FLT", "RTK FIX", "STATIC", "PPP"];
  function renderStatus(s) {
    const v = s.vehicle, up = !!(v && v.connected);
    $("a-sim").hidden = !/^(simulation|synthetic)$/i.test(s.source || "");
    $("a-play").hidden = !REPLAY;
    $("a-view").hidden = REPLAY || !s.read_only;
    document.body.classList.toggle("ro", viewOnly());
    setText("v-mode", v ? (up ? v.mode || "---" : "NO LINK") : "---", v ? (up ? "grn" : "red") : "dim");
    setText("v-arm", v && up ? (v.armed ? "ARMED" : "DISARMED") : "---", v && up ? "" : "dim");
    setText("v-link", v ? (up ? "UP" : "LOST") : "OFF", v ? (up ? "grn" : "red") : "dim");
    if (up && v.gps_fix != null) {
      setText("v-gps", `${GPS_FIX[v.gps_fix] || "?"}${v.gps_sats != null ? " " + v.gps_sats : ""}`, v.gps_fix >= 3 ? "" : "amb");
    } else setText("v-gps", "---", "dim");
    if (up && v.battery_v != null) {
      const p = v.battery_pct;
      setText("v-bat", `${v.battery_v.toFixed(1)}V${p != null ? " " + p + "%" : ""}`, p == null ? "" : p < 15 ? "red" : p < 30 ? "amb" : "");
    } else setText("v-bat", "---", "dim");
    setText("v-proc", `${Math.round(s.fps)} FPS ${s.latency_ms.toFixed(1)} MS`);
    setText("v-clock", `${s.utc || new Date().toISOString().slice(11, 19)}Z`);
  }

  function renderOsb(s) {
    const ro = viewOnly(), L = s.lock, G = s.gimbal;
    const frz = $("osb-frz");
    frz.setAttribute("aria-pressed", String(!!s.paused));
    frz.disabled = ro;
    setText("lg-acq", L.auto ? "AUTO" : "MAN");
    $("osb-acq").disabled = ro || !L.enabled;
    $("osb-brk").disabled = ro || !L.enabled;
    setText("lg-gmb", !G.available ? "---" : G.follow ? "FLW" : "HOLD");
    $("osb-gmb").disabled = ro || !G.available;
  }
  function renderHsdLegends() {
    setText("lg-rng", rangeLabel(RANGES[S.rangeIdx]));
    setText("lg-orient", S.northUp ? "N" : "HDG");
    setText("lg-dep", S.centred ? "CEN" : "DEP");
    $("osb-rng-up").disabled = S.rangeIdx >= RANGES.length - 1;
    $("osb-rng-dn").disabled = S.rangeIdx <= 0;
  }

  const TRK_STATE = { LOCKED: ["TRACK", "grn"], ACQUIRING: ["ACQUIRE", ""], LOST: ["COAST", "amb"], SEARCHING: ["SEARCH", ""] };
  function renderTarget(s) {
    const L = s.lock, tg = s.target_geo, v = s.vehicle;
    const tr = (s.tracks || []).find((t) => t.id === L.track_id);
    setText("g-trk", L.track_id != null ? `${L.track_id}${tr ? " " + tr.label.toUpperCase() : ""}` : "---", L.track_id != null ? "" : "dim");
    if (!L.enabled) setText("g-stat", "LOCK OFF", "dim");
    else {
      const [t, c] = TRK_STATE[L.state] || ["---", "dim"];
      setText("g-stat", L.state === "ACQUIRING" ? `${t} ${Math.round(L.progress * 100)}%` : t, c);
    }
    setText("g-lock", L.state === "LOCKED" ? `${L.duration.toFixed(1)} S` : "---", L.state === "LOCKED" ? "" : "dim");
    const has = !!tg, dim = has ? "" : "dim";
    setText("g-grid", has ? mgrs(tg.lat, tg.lon) : "---", dim);
    setText("g-lat", has ? dms(tg.lat, "N", "S") : "---", dim);
    setText("g-lon", has ? dms(tg.lon, "E", "W") : "---", dim);
    setText("g-elev", has && tg.elev_m != null ? `${Math.round(tg.elev_m)} M` : "---", has && tg.elev_m != null ? "" : "dim");
    setText("g-mthd", has ? tg.method.toUpperCase() : "---", dim);
    setText("g-slr", has ? `${Math.round(tg.slant_m)} M` : "---", dim);
    setText("g-gr", has ? `${Math.round(tg.range_m)} M` : "---", dim);
    const posOk = has && v && v.lat != null;
    setText("g-brg", posOk ? `${pad(Math.round(bearingTo(v, tg.lat, tg.lon)) % 360, 3)}°T` : "---", posOk ? "" : "dim");
    const los = v && v.connected && v.gimbal_yaw != null && v.gimbal_pitch != null;
    setText("g-los", los ? `AZ ${pad(Math.round(wrap360(v.heading + v.gimbal_yaw)) % 360, 3)} EL ${sgn(v.gimbal_pitch, 1)}` : "---", los ? "" : "dim");
    setText("g-bse", L.error ? `X ${sgn(L.error[0], 3)} Y ${sgn(L.error[1], 3)}` : "---", L.error ? "" : "dim");
  }

  function renderTracks(s) {
    const v = s.vehicle, rows = s.tracks || [];
    setText("trk-count", `${rows.length} ACTIVE`);
    const html = !rows.length ? '<tr class="empty"><td colspan="8">NO TRACKS</td></tr>' : rows.map((t) => {
      const g = t.geo, posOk = g && v && v.lat != null;
      const brg = posOk ? pad(Math.round(bearingTo(v, g.lat, g.lon)) % 360, 3) : "---";
      const grid = g ? mgrs(g.lat, g.lon).split(" ").slice(1).join(" ") : "---";
      const stat = t.role === "target" ? "TGT" : t.role === "candidate" ? "CAND" : "TRK";
      return `<tr data-id="${t.id}"${t.role === "target" ? ' class="tgt"' : ""}><td>${t.id}</td><td>${esc(t.label.toUpperCase())}</td>`
        + `<td class="n">${t.score.toFixed(2)}</td><td class="n">${Math.round(t.age_s)}</td>`
        + `<td class="n">${g ? Math.round(g.range_m) : "---"}</td><td class="n">${brg}</td><td>${grid}</td><td>${stat}</td></tr>`;
    }).join("");
    if (S.html.tracks !== html) { S.html.tracks = html; $("tracks-body").innerHTML = html; }
  }

  const MSG = {
    session_start: (e) => [`SESSION START ${e.detail}`, "sts"],
    lock_acquired: (e) => [`TRK ${e.track_id} LOCK ACQUIRED`, "adv"],
    lock_lost: (e) => [`TRK ${e.track_id} LOCK LOST${e.detail ? " AFTER " + e.detail : ""}`, "cau"],
    designated: (e) => [`TRK ${e.track_id} DESIGNATED`, "adv"],
    released: (e) => [e.track_id != null ? `TRK ${e.track_id} LOCK RELEASED` : "LOCK RELEASED", "adv"],
    no_track: (e) => [`DESIGNATION REJECTED - ${e.detail}`, "cau"],
    paused: () => ["VIDEO FROZEN", "adv"],
    resumed: () => ["VIDEO LIVE", "adv"],
    follow_on: () => ["GIMBAL FOLLOW", "adv"],
    follow_off: () => ["GIMBAL HOLD", "adv"],
    auto_on: () => ["ACQUISITION AUTO", "adv"],
    auto_off: () => ["ACQUISITION MANUAL", "adv"],
    source_end: () => ["VIDEO SOURCE ENDED", "cau"],
    link_up: (e) => [`FC LINK UP ${e.detail}`, "adv"],
    link_lost: () => ["FC LINK LOST", "wrn"],
  };
  function annunciate(id, text, cls) {
    const el = $(id);
    if (el.textContent !== text) el.textContent = text;
    if (el.className !== cls) el.className = cls;
  }
  function renderCas(s) {
    const v = s.vehicle, up = !!(v && v.connected), L = s.lock, G = s.gimbal;
    // dark cockpit: normal conditions stay unlit, only abnormal ones light up (with their name)
    annunciate("c-link", up || !v ? "LINK" : "LINK LOST", !v ? "off" : up ? "" : "wrn");
    if (!up || v.gps_fix == null) annunciate("c-gps", "GPS", "off");
    else annunciate("c-gps", v.gps_fix >= 3 ? "GPS" : v.gps_fix === 2 ? "GPS 2D" : "GPS NO FIX", v.gps_fix >= 3 ? "" : "cau");
    const p = up ? v.battery_pct : null;
    annunciate("c-batt", p == null ? "BATT" : p < 15 ? "BATT CRIT" : p < 30 ? "BATT LOW" : "BATT", p == null ? "off" : p < 15 ? "wrn" : p < 30 ? "cau" : "");
    annunciate("c-gmb", G.available && !G.follow ? "GMB HOLD" : "GMB", !G.available ? "off" : G.follow ? "" : "adv");
    if (!L.enabled) annunciate("c-trk", "TRK OFF", "off");
    else annunciate("c-trk", { LOCKED: "TRK", ACQUIRING: "TRK ACQ", LOST: "TRK COAST", SEARCHING: "TRK SRCH" }[L.state] || "TRK",
      { LOCKED: "", ACQUIRING: "adv", LOST: "cau", SEARCHING: "adv" }[L.state] || "");
    annunciate("c-vid", !S.videoOk ? "NO VIDEO" : s.paused ? "VID FRZ" : "VID", !S.videoOk ? "wrn" : s.paused ? "adv" : "");

    const ev = s.events || [];
    setText("msg-count", String(ev.length));
    const html = !ev.length ? '<li class="empty">NO MESSAGES</li>' : ev.slice().reverse().map((e) => {
      const [text, cls] = (MSG[e.kind] || ((x) => [`${x.kind}${x.track_id != null ? " " + x.track_id : ""}`, "sts"]))(e);
      return `<li><time>${esc(e.utc)}</time><span class="${cls}">${esc(String(text).toUpperCase())}</span></li>`;
    }).join("");
    if (S.html.log !== html) { S.html.log = html; $("log").innerHTML = html; }
  }

  /* ------------------------------------------------------------------ canvas helpers */
  function fit(cv) {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [g, r.width, r.height];
  }
  function line(g, x1, y1, x2, y2) {
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x2, y2);
    g.stroke();
  }
  function flag(g, x, y, text) {  // invalid-data flag: amber text in an amber box
    g.font = font(12, true);
    const w = Math.round(g.measureText(text).width + 12);
    g.strokeStyle = C.amb;
    g.lineWidth = 1;
    g.strokeRect(Math.round(x - w / 2) + 0.5, Math.round(y - 10) + 0.5, w, 19);
    g.fillStyle = C.amb;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(text, x, y);
    g.textBaseline = "alphabetic";
  }

  /* ------------------------------------------------------------------ PFD */
  function fmaCells(s) {
    const v = s.vehicle, up = !!(v && v.connected), L = s.lock, G = s.gimbal;
    const trk = !L.enabled ? ["TRK OFF", "txt-3"]
      : { LOCKED: ["TRACK", "grn"], ACQUIRING: ["ACQ", "txt"], LOST: ["COAST", "amb"], SEARCHING: ["SEARCH", "txt"] }[L.state] || ["---", "txt-3"];
    return [
      v ? (up ? [v.mode || "---", "grn"] : ["NO LINK", "red"]) : ["NO FC", "txt-3"],
      up ? [v.armed ? "ARMED" : "DISARMED", "txt"] : ["---", "txt-3"],
      G.available ? (G.follow ? ["GMB FLW", "grn"] : ["GMB HOLD", "txt"]) : ["GMB ---", "txt-3"],
      trk,
    ];
  }
  function drawFMA(g, s, W, h) {  // flight mode annunciator row
    const cells = fmaCells(s), cw = W / cells.length;
    g.strokeStyle = C.line;
    g.lineWidth = 1;
    line(g, 0, h + 0.5, W, h + 0.5);
    for (let i = 1; i < cells.length; i++) line(g, Math.round(i * cw) + 0.5, 5, Math.round(i * cw) + 0.5, h - 5);
    g.font = font(11, true);
    g.textAlign = "center";
    g.textBaseline = "middle";
    cells.forEach(([t, c], i) => { g.fillStyle = C[c]; g.fillText(t, i * cw + cw / 2, h / 2 + 1); });
    g.textBaseline = "alphabetic";
  }
  function tape(g, o) {  // vertical scale with a boxed readout pointing at the attitude display
    const { x, y, w, h, value, step, major, span, side } = o;
    const ppu = h / span, cy = Math.round(y + h / 2);
    g.save();
    g.beginPath();
    g.rect(x, y, w, h);
    g.clip();
    g.fillStyle = C.tape;
    g.fillRect(x, y, w, h);
    g.strokeStyle = C.txt;
    g.fillStyle = C.txt;
    g.lineWidth = 1;
    g.font = font(10);
    g.textBaseline = "middle";
    for (let u = Math.ceil((value - span / 2 - step) / step) * step; u <= value + span / 2 + step; u += step) {
      const yy = Math.round(cy - (u - value) * ppu) + 0.5, isMajor = Math.abs(u % major) < 1e-6, len = isMajor ? 8 : 4;
      if (side === "L") {
        line(g, x + w - len, yy, x + w, yy);
        if (isMajor) { g.textAlign = "right"; g.fillText(String(u), x + w - 11, yy); }
      } else {
        line(g, x, yy, x + len, yy);
        if (isMajor) { g.textAlign = "left"; g.fillText(String(u), x + 11, yy); }
      }
    }
    g.restore();
    const bh = 22, bw = w - 9, bx = side === "L" ? x + 1 : x + 8, t = cy - bh / 2, b = cy + bh / 2;
    g.beginPath();
    if (side === "L") {
      g.moveTo(bx, t); g.lineTo(bx + bw, t); g.lineTo(bx + bw, cy - 5); g.lineTo(bx + bw + 6, cy);
      g.lineTo(bx + bw, cy + 5); g.lineTo(bx + bw, b); g.lineTo(bx, b);
    } else {
      g.moveTo(bx + bw, t); g.lineTo(bx, t); g.lineTo(bx, cy - 5); g.lineTo(bx - 6, cy);
      g.lineTo(bx, cy + 5); g.lineTo(bx, b); g.lineTo(bx + bw, b);
    }
    g.closePath();
    g.fillStyle = "#000";
    g.fill();
    g.strokeStyle = C.txt;
    g.lineWidth = 1.2;
    g.stroke();
    g.fillStyle = C.txt;
    g.font = font(13, true);
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(o.fmt(value), bx + bw / 2, cy + 1);
    g.textBaseline = "alphabetic";
    g.font = font(10);
    g.fillStyle = C["txt-2"];
    g.fillText(o.label, x + w / 2, y - 5);
    g.fillText(o.unit, x + w / 2, y + h + 13);
  }
  function hdgTape(g, x, y, w, h, hdg, sensAz, tgtBrg) {
    const cx = Math.round(x + w / 2), span = 50, ppd = w / span;
    g.save();
    g.beginPath();
    g.rect(x, y, w, h);
    g.clip();
    g.fillStyle = C.tape;
    g.fillRect(x, y, w, h);
    g.strokeStyle = C.txt;
    g.fillStyle = C.txt;
    g.lineWidth = 1;
    g.font = font(10);
    g.textAlign = "center";
    for (let d = Math.floor((hdg - span / 2 - 5) / 5) * 5; d <= hdg + span / 2 + 5; d += 5) {
      const xx = Math.round(x + w / 2 + (d - hdg) * ppd) + 0.5, n = wrap360(d), isMajor = n % 10 === 0;
      line(g, xx, y + h, xx, y + h - (isMajor ? 8 : 4));
      if (isMajor && Math.abs(xx - cx) > 36 && xx - 13 > x && xx + 13 < x + w) g.fillText(pad(n, 3), xx, y + h - 11);
    }
    const bug = (az, colour, kind) => {  // sensor azimuth (cyan triangle), target bearing (magenta diamond)
      if (az == null) return;
      const d = ((az - hdg + 540) % 360) - 180, xx = x + w / 2 + Math.max(-span / 2, Math.min(span / 2, d)) * ppd;
      g.fillStyle = colour;
      g.strokeStyle = colour;
      g.lineWidth = 1.5;
      g.beginPath();
      if (kind === "tri") { g.moveTo(xx, y + 9); g.lineTo(xx - 5, y + 1); g.lineTo(xx + 5, y + 1); g.closePath(); g.fill(); } else { g.moveTo(xx, y + 1); g.lineTo(xx + 5, y + 6); g.lineTo(xx, y + 11); g.lineTo(xx - 5, y + 6); g.closePath(); g.stroke(); }
    };
    bug(sensAz, C.cyn, "tri");
    bug(tgtBrg, C.mag, "dia");
    g.restore();
    g.fillStyle = "#000";
    g.fillRect(cx - 20, y - 3, 40, 17);
    g.strokeStyle = C.txt;
    g.lineWidth = 1.2;
    g.strokeRect(cx - 19.5, y - 2.5, 39, 16);
    line(g, cx + 0.5, y + 13, cx + 0.5, y + h);
    g.fillStyle = C.txt;
    g.font = font(12, true);
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(pad(Math.round(wrap360(hdg)) % 360, 3), cx, y + 6);
    g.textBaseline = "alphabetic";
  }
  function aircraftSymbol(g, cx, cy, aw) {  // fixed reference: black-filled wings outlined in yellow
    const L = Math.max(24, aw * 0.12), gap = Math.max(14, aw * 0.07);
    const shapes = [
      [[-gap - L, -2], [-gap, -2], [-gap, 9], [-gap - 5, 9], [-gap - 5, 3], [-gap - L, 3]],
      [[gap + L, -2], [gap, -2], [gap, 9], [gap + 5, 9], [gap + 5, 3], [gap + L, 3]],
      [[-3, -3], [3, -3], [3, 3], [-3, 3]],
    ];
    g.save();
    g.translate(cx, cy);
    g.lineJoin = "miter";
    for (const pts of shapes) {
      g.beginPath();
      pts.forEach(([px, py], i) => (i ? g.lineTo(px, py) : g.moveTo(px, py)));
      g.closePath();
      g.fillStyle = "#000";
      g.fill();
      g.strokeStyle = C.ac;
      g.lineWidth = 2;
      g.stroke();
    }
    g.restore();
  }
  function drawPFD(s) {
    const [g, W, H] = fit($("pfd"));
    g.fillStyle = "#000";
    g.fillRect(0, 0, W, H);
    if (W < 160 || H < 150) return;
    const v = s.vehicle, up = !!(v && v.connected && v.roll != null);
    const fmaH = 24, hdgH = 30, tw = Math.round(Math.max(50, Math.min(70, W * 0.15)));
    const ax = tw + 12, aw = W - 2 * tw - 24, ay = fmaH + 18, ah = H - fmaH - hdgH - 48;
    const cx = Math.round(ax + aw / 2), cy = Math.round(ay + ah / 2);
    drawFMA(g, s, W, fmaH);
    if (!up) {
      g.strokeStyle = C.line;
      g.lineWidth = 1;
      g.strokeRect(ax + 0.5, ay + 0.5, aw, ah);
      flag(g, cx, cy, "ATT");
      flag(g, 4 + tw / 2, cy, "SPD");
      flag(g, W - 4 - tw / 2, cy, "ALT");
      flag(g, cx, H - hdgH / 2 - 4, "HDG");
      return;
    }
    // attitude: sky and ground move with pitch and roll behind the fixed aircraft symbol
    const ppd = ah / 40;
    g.save();
    g.beginPath();
    g.rect(ax, ay, aw, ah);
    g.clip();
    g.translate(cx, cy);
    g.rotate(-v.roll * D2R);
    g.translate(0, (v.pitch || 0) * ppd);
    const big = (aw + ah) * 2;
    g.fillStyle = C.sky;
    g.fillRect(-big, -big, 2 * big, big);
    g.fillStyle = C.gnd;
    g.fillRect(-big, 0, 2 * big, big);
    g.strokeStyle = C.txt;
    g.lineWidth = 1.5;
    line(g, -big, 0, big, 0);
    g.lineWidth = 1;
    g.fillStyle = C.txt;
    g.font = font(10);
    g.textBaseline = "middle";
    for (let p = -40; p <= 40; p += 2.5) {
      if (p === 0) continue;
      const y = -p * ppd, isMajor = p % 10 === 0, mid = p % 5 === 0;
      const half = aw * (isMajor ? 0.13 : mid ? 0.075 : 0.035);
      line(g, -half, y, half, y);
      if (isMajor) {
        g.textAlign = "right"; g.fillText(String(Math.abs(p)), -half - 4, y);
        g.textAlign = "left"; g.fillText(String(Math.abs(p)), half + 4, y);
      }
    }
    g.textBaseline = "alphabetic";
    g.restore();
    // bank scale fixed to the display, sky pointer turning with the horizon
    const rr = Math.min(aw, ah) * 0.44;
    g.save();
    g.translate(cx, cy);
    g.strokeStyle = C.txt;
    g.fillStyle = C.txt;
    g.lineWidth = 1.5;
    g.beginPath();
    g.arc(0, 0, rr, (-90 - 60) * D2R, (-90 + 60) * D2R);
    g.stroke();
    for (const a of [-60, -45, -30, -20, -10, 10, 20, 30, 45, 60]) {
      const len = Math.abs(a) % 30 === 0 ? 10 : 6, sa = Math.sin(a * D2R), ca = -Math.cos(a * D2R);
      line(g, sa * rr, ca * rr, sa * (rr + len), ca * (rr + len));
    }
    g.beginPath();
    g.moveTo(0, -rr - 1); g.lineTo(-6, -rr - 11); g.lineTo(6, -rr - 11);
    g.closePath();
    g.stroke();
    g.rotate(-v.roll * D2R);
    g.beginPath();
    g.moveTo(0, -rr + 1); g.lineTo(-6, -rr + 11); g.lineTo(6, -rr + 11);
    g.closePath();
    g.fill();
    g.restore();
    aircraftSymbol(g, cx, cy, aw);
    // speed and altitude tapes
    tape(g, { x: 4, y: ay, w: tw, h: ah, value: v.groundspeed ?? 0, step: 1, major: 5, span: 16, side: "L", label: "GS", unit: "M/S", fmt: (x) => x.toFixed(1) });
    tape(g, { x: W - tw - 4, y: ay, w: tw, h: ah, value: v.alt_amsl ?? v.alt_rel ?? 0, step: 2, major: 10, span: 60, side: "R", label: "ALT", unit: "M MSL", fmt: (x) => String(Math.round(x)) });
    g.font = font(10);
    g.textAlign = "center";
    g.fillStyle = C.txt;
    const rx = W - 4 - tw / 2;
    g.fillText(`AGL ${v.alt_rel != null ? Math.round(v.alt_rel) : "---"}`, rx, ay + ah + 27);
    g.fillText(`VS ${S.vs != null ? sgn(S.vs, 1) : "---"}`, rx, ay + ah + 39);
    // heading scale with the sensor line-of-sight and target bearing bugs
    const az = v.gimbal_yaw != null ? wrap360(v.heading + v.gimbal_yaw) : null;
    const tg = s.target_geo, brg = tg && v.lat != null ? bearingTo(v, tg.lat, tg.lon) : null;
    hdgTape(g, ax, H - hdgH - 6, aw, hdgH, v.heading ?? 0, az, brg);
    g.font = font(10);
    g.textAlign = "left";
    g.fillStyle = C.cyn;
    g.fillText(az != null ? `SENS ${pad(Math.round(az) % 360, 3)}` : "", 4, H - 18);
    g.fillStyle = C.mag;
    g.fillText(brg != null ? `TGT  ${pad(Math.round(brg) % 360, 3)}` : "", 4, H - 6);
  }

  /* ------------------------------------------------------------------ HSD */
  function hostile(g, x, y, r) {  // MIL-STD-2525 hostile frame: diamond
    g.beginPath();
    g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y);
    g.closePath();
    g.fillStyle = "#000";
    g.fill();
    g.strokeStyle = C.red;
    g.lineWidth = 1.6;
    g.stroke();
  }
  function unknown(g, x, y, r) {  // MIL-STD-2525 unknown frame: quatrefoil
    const q = r * 0.55;
    g.beginPath();
    g.arc(x, y - q, q, Math.PI, 2 * Math.PI);
    g.arc(x + q, y, q, 1.5 * Math.PI, 2.5 * Math.PI);
    g.arc(x, y + q, q, 0, Math.PI);
    g.arc(x - q, y, q, 0.5 * Math.PI, 1.5 * Math.PI);
    g.closePath();
    g.fillStyle = "#000";
    g.fill();
    g.strokeStyle = C.yel;
    g.lineWidth = 1.6;
    g.stroke();
  }
  function ownship(g, x, y, ang) {
    g.save();
    g.translate(x, y);
    g.rotate(ang);
    g.strokeStyle = C.cyn;
    g.lineWidth = 2;
    g.lineCap = "square";
    line(g, 0, -10, 0, 10);
    line(g, -9, -2, 9, -2);
    line(g, -4, 8, 4, 8);
    g.restore();
  }
  function drawHSD(s) {
    const [g, W, H] = fit($("map"));
    g.fillStyle = "#000";
    g.fillRect(0, 0, W, H);
    if (W < 100 || H < 100) return;
    const v = s.vehicle;
    if (!v || !v.connected || v.lat == null) { flag(g, W / 2, H / 2, v ? "NO NAV" : "NO FC"); return; }
    if (S.rangeAuto) {  // smallest scale that holds every geolocated track, until the operator picks one
      let far = 0;
      for (const t of s.tracks || []) if (t.geo) far = Math.max(far, t.geo.range_m);
      let idx = RANGES.findIndex((r) => r >= far * 1.25);
      if (idx < 0) idx = RANGES.length - 1;
      if (idx !== S.rangeIdx) { S.rangeIdx = idx; renderHsdLegends(); }
    }
    const hdg = v.heading ?? 0, rot = S.northUp ? 0 : -hdg * D2R;
    const ox = Math.round(W / 2), oy = Math.round(S.centred ? H / 2 + 8 : H * 0.7);
    const Rpx = Math.max(40, S.centred ? Math.min(W, H) / 2 - 30 : Math.min(W / 2 - 12, oy - 34));
    const R = RANGES[S.rangeIdx], k = Rpx / R, cr = Math.cos(rot), sr = Math.sin(rot);
    const xy = (n, e) => { const x = e * k, y = -n * k; return [ox + x * cr - y * sr, oy + x * sr + y * cr]; };
    const ll = (lat, lon) => { const [n, e] = offsetNE(v, lat, lon); return xy(n, e); };

    // range rings and compass rose
    g.lineWidth = 1;
    g.strokeStyle = C.line;
    g.setLineDash([3, 4]);
    g.beginPath();
    g.arc(ox, oy, Rpx / 2, 0, 2 * Math.PI);
    g.stroke();
    g.setLineDash([]);
    g.strokeStyle = C["line-2"];
    g.beginPath();
    g.arc(ox, oy, Rpx, 0, 2 * Math.PI);
    g.stroke();
    g.font = font(10);
    g.textBaseline = "middle";
    for (let d = 0; d < 360; d += 5) {
      const a = d * D2R + rot, sx = Math.sin(a), sy = -Math.cos(a), len = d % 30 === 0 ? 9 : d % 10 === 0 ? 6 : 3;
      g.strokeStyle = d % 10 === 0 ? C["txt-2"] : C["line-2"];
      line(g, ox + sx * Rpx, oy + sy * Rpx, ox + sx * (Rpx - len), oy + sy * (Rpx - len));
      if (d % 30 === 0) {
        g.fillStyle = d % 90 === 0 ? C.txt : C["txt-2"];
        g.textAlign = "center";
        g.fillText({ 0: "N", 90: "E", 180: "S", 270: "W" }[d] || String(d / 10), ox + sx * (Rpx - 19), oy + sy * (Rpx - 19));
      }
    }
    g.fillStyle = C["txt-2"];
    g.textAlign = "left";
    g.fillText(rangeLabel(R / 2), ox + Rpx * 0.36 + 4, oy - Rpx * 0.36 - 4);

    // own track, sensor footprint and line of sight (own sensor: cyan)
    g.fillStyle = C["txt-3"];
    for (const [la, lo] of S.trail) { const [x, y] = ll(la, lo); g.fillRect(x - 1, y - 1, 2, 2); }
    const fp = s.footprint;
    if (fp && fp.length === 4) {
      g.strokeStyle = C.cyn;
      g.lineWidth = 1.2;
      g.beginPath();
      let pen = false;
      for (let i = 0; i <= 4; i++) {
        const c = fp[i % 4];
        if (!c) { pen = false; continue; }
        const [x, y] = xy(c[0], c[1]);
        if (pen) g.lineTo(x, y); else { g.moveTo(x, y); pen = true; }
      }
      g.stroke();
      if (fp.every(Boolean)) {
        const [x, y] = xy(fp.reduce((a, c) => a + c[0], 0) / 4, fp.reduce((a, c) => a + c[1], 0) / 4);
        g.setLineDash([3, 3]);
        line(g, ox, oy, x, y);
        g.setLineDash([]);
      }
    }
    // tracks: unknown = yellow quatrefoil, designated target = red diamond on a magenta line of sight
    const tg = s.target_geo;
    if (tg) {
      const [x, y] = ll(tg.lat, tg.lon);
      g.strokeStyle = C.mag;
      g.lineWidth = 1.2;
      line(g, ox, oy, x, y);
    }
    g.font = font(10);
    g.textBaseline = "middle";
    g.textAlign = "left";
    for (const t of s.tracks || []) {
      if (!t.geo) continue;
      const [x, y] = ll(t.geo.lat, t.geo.lon);
      if (t.role === "target") hostile(g, x, y, 8); else unknown(g, x, y, 8);
      g.fillStyle = t.role === "target" ? C.txt : C["txt-2"];
      g.fillText(t.role === "target" ? `TGT ${t.id}` : String(t.id), x + 11, y);
    }
    ownship(g, ox, oy, S.northUp ? hdg * D2R : 0);
    // heading readout and lubber mark at the top of the rose
    const hy = Math.max(10, oy - Rpx - 14);
    if (!S.northUp) {
      g.fillStyle = C.txt;
      g.beginPath();
      g.moveTo(ox, oy - Rpx + 1); g.lineTo(ox - 5, oy - Rpx - 7); g.lineTo(ox + 5, oy - Rpx - 7);
      g.closePath();
      g.fill();
    }
    g.fillStyle = "#000";
    g.fillRect(ox - 20, hy - 9, 40, 17);
    g.strokeStyle = C.txt;
    g.lineWidth = 1.2;
    g.strokeRect(ox - 19.5, hy - 8.5, 39, 16);
    g.fillStyle = C.txt;
    g.font = font(12, true);
    g.textAlign = "center";
    g.fillText(pad(Math.round(wrap360(hdg)) % 360, 3), ox, hy);
    // corner data, on a black ground so the rose never runs through it
    g.textBaseline = "alphabetic";
    g.font = font(10);
    const corner = (text, colour, right, bottom) => {
      const tw = g.measureText(text).width, x = right ? W - 6 - tw : 6, y = bottom ? H - 6 : 14;
      g.fillStyle = "#000";
      g.fillRect(x - 3, y - 11, tw + 6, 15);
      g.fillStyle = colour;
      g.textAlign = "left";
      g.fillText(text, x, y);
    };
    corner(S.northUp ? "N UP" : "HDG UP", C["txt-2"], false, false);
    corner(`${rangeLabel(R)} M`, C["txt-2"], true, false);
    corner(`OWN ${mgrs(v.lat, v.lon)}`, C["txt-2"], false, true);
    if (tg) corner(`TGT ${pad(Math.round(bearingTo(v, tg.lat, tg.lon)) % 360, 3)}°/${Math.round(tg.range_m)}M`, C.mag, true, true);
  }

  /* ------------------------------------------------------------------ main loop */
  function frame() {
    if (dirty && S.last) {
      dirty = false;
      drawPFD(S.last);
      drawHSD(S.last);
    }
    requestAnimationFrame(frame);
  }
  readColours();
  layoutViews();
  renderHsdLegends();
  if (document.fonts && document.fonts.load) document.fonts.load('11px "B612 Mono"').then(() => { dirty = true; }).catch(() => {});
  requestAnimationFrame(frame);
  connect();
})();
