// Headless scenario tests for app.js. Run with: node test/run-tests.mjs
//
// app.js is a browser module (top-level DOM access, a CDN-backed import in
// backends.js), so it cannot be imported in Node directly. This harness
// builds a testable copy: stub DOM and globals, the real app.js source with
// its backends import replaced, and a test-access export appended. The tests
// then drive the REAL state machines (hand-over-face detection, the teach
// flow, pose picking, smoothing, trails) with synthetic landmark data.
//
// No dependencies, no framework: node test/run-tests.mjs exits 0 on green.
import { readFileSync, writeFileSync } from "node:fs";

// ---------- build the testable module ----------
const SRC = new URL("../app.js", import.meta.url);
const OUT = new URL("./.app.under-test.mjs", import.meta.url); // gitignored

const stubs = `
// ---------- headless stubs (test harness, not app code) ----------
function makeCtx() {
  return new Proxy({}, {
    get(t, k) {
      if (typeof k === "symbol") return undefined;
      if (!(k in t)) t[k] = (..._a) => makeCtx();
      return t[k];
    },
    set(t, k, v) { t[k] = v; return true; },
  });
}
function makeEl(tag = "div") {
  const classes = new Set();
  return {
    tagName: tag.toUpperCase(),
    children: [], dataset: {}, attributes: {},
    style: new Proxy({ setProperty() {} }, { set(t, k, v) { t[k] = v; return true; }, get(t, k) { return t[k]; } }),
    classList: {
      add(...c) { c.forEach((x) => classes.add(x)); },
      remove(...c) { c.forEach((x) => classes.delete(x)); },
      toggle(c, f) { const on = f === undefined ? !classes.has(c) : !!f; on ? classes.add(c) : classes.delete(c); return on; },
      contains(c) { return classes.has(c); },
    },
    value: "", textContent: "", innerHTML: "", title: "",
    // className writes sync into classList: the app sets one and reads the
    // other (e.g. the big-status countdown classes in updateCaptureOverlay).
    get className() { return [...classes].join(" "); },
    set className(v) { classes.clear(); String(v).split(/\\s+/).filter(Boolean).forEach((c) => classes.add(c)); },
    hidden: false, disabled: false, checked: false, offsetWidth: 0, firstChild: null,
    width: 0, height: 0, videoWidth: 640, videoHeight: 480, readyState: 4, clientHeight: 640,
    addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
    appendChild(c) { this.children.push(c); return c; },
    insertBefore(c) { this.children.unshift(c); return c; },
    querySelector() { return makeEl(); },
    querySelectorAll() { return []; },
    setAttribute() {}, getAttribute() { return null; },
    focus() {}, select() {}, click() {}, play() { return Promise.resolve(); },
    getContext() { return makeCtx(); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
  };
}
const __els = new Map();
globalThis.document = {
  getElementById(id) { if (!__els.has(id)) __els.set(id, makeEl()); return __els.get(id); },
  querySelector() { return makeEl(); },
  querySelectorAll() { return []; },
  createElement(t) { return makeEl(t); },
  addEventListener() {},
  body: makeEl("body"), documentElement: makeEl("html"),
  fullscreenElement: null, activeElement: null, hidden: false,
};
const __store = new Map();
globalThis.localStorage = {
  getItem(k) { return __store.has(k) ? __store.get(k) : null; },
  setItem(k, v) { __store.set(k, String(v)); },
  removeItem(k) { __store.delete(k); },
};
globalThis.window = {
  addEventListener() {}, innerWidth: 1280, innerHeight: 800,
  AudioContext: undefined, webkitAudioContext: undefined,
};
Object.defineProperty(globalThis, "navigator", {
  value: { mediaDevices: { getUserMedia: () => Promise.reject(Object.assign(new Error("headless"), { name: "NotFoundError" })) } },
  configurable: true,
});
globalThis.history = { replaceState() {}, pushState() {}, state: null };
globalThis.requestAnimationFrame = () => 0;
// Stand-in for backends.js (its CDN import cannot load headless).
const BLAZE_CONNECTIONS = [], COCO_CONNECTIONS = [];
function createBackend() { throw new Error("no backend in harness"); }
// ---------- end stubs ----------
`;

const exports_ = `
globalThis.__T = {
  get teach() { return teach; },
  get templates() { return templates; },
  get lastMainMid() { return lastMainMid; },
  get restInfo() { return restInfo; },
  get trails() { return trails; },
  get markPos() { return markPos; },
  overlay, wordInput, teachMsg, recordBtn, statusEl, threshInput, teachHint,
  startTeach, teachStep, updateHandsOnFace, clearHandsOnFace, handOnFace, isNearFace,
  updateTeachUI, cancelTeach,
  get tut() { return tut; },
  startTutorial, endTutorial, tutNotify, tutLockEl, tutNextBtn,
  updateCaptureOverlay, countLabel, bigStatus, stripIndices, pbTempo, judgeBeat,
  pickMainPose, smoothPose, drawPresence, resampleSeq, migrateSq, dtw, poseDist,
  trimStillEnds, trimLeadIn, travelOf, thresholdFor, recomputeWordStats,
  setLowLight, lightNote,
  get lowLight() { return lowLight; },
};
`;

let src = readFileSync(SRC, "utf8");
const importLine = 'import { createBackend, BLAZE_CONNECTIONS, COCO_CONNECTIONS } from "./backends.js";';
if (!src.includes(importLine)) throw new Error("backends import line not found; update the harness");
src = src.replace(importLine, "// import replaced by harness stubs");
writeFileSync(OUT, stubs + src + exports_);

await import(OUT.href);
const T = globalThis.__T;

// ---------- tests ----------
let pass = 0, fail = 0;
function check(name, cond, detail = "") {
  if (cond) { pass++; console.log("  ok    " + name); }
  else { fail++; console.log("  FAIL  " + name + (detail ? "  [" + detail + "]" : "")); }
}

// A 33-slot landmark frame for a standing body, image-normalized coords.
// Shoulder width 0.2, so the face anchor sits at (0.5, 0.35).
function standing() {
  const lm = (x, y, v = 0.9) => ({ x, y, visibility: v });
  const lms = new Array(33).fill(null).map(() => lm(0.5, 0.5, 0.1));
  lms[0] = lm(0.5, 0.30);                           // nose
  lms[7] = lm(0.55, 0.30); lms[8] = lm(0.45, 0.30); // ears
  lms[11] = lm(0.6, 0.5); lms[12] = lm(0.4, 0.5);   // shoulders
  lms[13] = lm(0.65, 0.62); lms[14] = lm(0.35, 0.62); // elbows
  lms[15] = lm(0.68, 0.74); lms[16] = lm(0.32, 0.74); // wrists
  lms[23] = lm(0.55, 0.78); lms[24] = lm(0.45, 0.78); // hips
  return lms;
}

console.log("T1: updateHandsOnFace latch / reject / release");
{
  T.clearHandsOnFace();
  let now = 1000;
  const step = (lms) => { const h = T.updateHandsOnFace(lms, now); now += 33; return h; };
  for (let i = 0; i < 10; i++) step(standing());
  check("idle: neither hand on face", !T.handOnFace.left.on && !T.handOnFace.right.on);

  // Subject right wrist (16) to the anchor, forearm vertical (elbow below).
  const cover = standing();
  cover[16] = { x: 0.5, y: 0.35, visibility: 0.9 };
  cover[14] = { x: 0.5, y: 0.52, visibility: 0.9 };
  let h = null;
  for (let i = 0; i < 6; i++) h = step(cover);
  check("palm over face latches right hand", h.right === true && h.left === false);
  check("isNearFace true while covering", T.isNearFace());

  // Wrist drops out behind the palm: the sticky window keeps it latched.
  const occluded = standing();
  occluded[16] = { x: 0.5, y: 0.35, visibility: 0.05 };
  occluded[14] = { x: 0.5, y: 0.52, visibility: 0.9 };
  for (let i = 0; i < 8; i++) h = step(occluded); // ~264ms < UNSEEN_STICKY_MS
  check("unseen wrist stays latched (sticky)", h.right === true);

  // Seen clearly away from the face: releases after the release debounce.
  for (let i = 0; i < 10; i++) h = step(standing());
  check("wrist seen away releases the latch", h.right === false);

  // Hand BESIDE the face: inside the radius but offset sideways -> no latch.
  T.clearHandsOnFace();
  const side = standing();
  side[16] = { x: 0.59, y: 0.35, visibility: 0.9 };
  side[14] = { x: 0.59, y: 0.52, visibility: 0.9 };
  for (let i = 0; i < 10; i++) h = step(side);
  check("hand beside the face never latches", h.right === false);

  // Crossed arms: wrist near the face but the forearm is horizontal.
  T.clearHandsOnFace();
  const crossed = standing();
  crossed[16] = { x: 0.5, y: 0.37, visibility: 0.9 };
  crossed[14] = { x: 0.62, y: 0.38, visibility: 0.9 };
  for (let i = 0; i < 10; i++) h = step(crossed);
  check("crossed-arm wrist never latches", h.right === false);
}

console.log("T2: full teach flow through the real state machine");
{
  T.clearHandsOnFace();
  const before = T.templates.length;
  T.wordInput.value = "hello";
  T.startTeach();
  check("teach starts in prime", T.teach && T.teach.state === "prime");
  let now = performance.now() + 1000;
  const vecAt = (w) => { const v = new Array(66).fill(0); v[30] = w; v[31] = -w; return v; };
  const go = (hands, vec, frames, dtm = 33) => {
    // runFrame only calls teachStep while a teach is active; mirror that guard.
    for (let i = 0; i < frames; i++) { if (!T.teach) return; T.teachStep(vec, hands, false, now, true); now += dtm; }
  };
  go({ left: false, right: false }, vecAt(0), 3);    // arm: right hand off first
  go({ left: false, right: true }, vecAt(0), 16);    // cover ~530ms -> starting
  check("sustained right cover starts countdown", T.teach.state === "starting");
  go({ left: false, right: true }, vecAt(0), 95);    // ~3.1s hold -> capturing
  check("3s hold begins capture", T.teach.state === "capturing");
  // The move: wrist sweeps 0 -> 0.8 fast, then holds still.
  for (let i = 0; i <= 16; i++) { T.teachStep(vecAt(i * 0.05), { left: false, right: false }, false, now, true); now += 33; }
  go({ left: false, right: false }, vecAt(0.8), 10); // settle
  go({ left: true, right: false }, vecAt(0.8), 16);  // left cover ~530ms arms stop
  check("still capturing until stop hold completes", T.teach.state === "capturing");
  go({ left: true, right: false }, vecAt(0.8), 95);  // ~3.1s -> finishTeach
  check("left-hand hold finishes the teach", T.teach === null);
  check("a template was saved", T.templates.length === before + 1, "len " + T.templates.length);
  const t = T.templates[T.templates.length - 1];
  check("saved word is 'hello'", t && t.word === "hello");
  check("seq resampled to 20 frames", t && t.seq.length === 20);
  check("coordinates were quantized", t && t.seq.every((f) => f.every((n) => n === Math.round(n * 1e4) / 1e4)));
  check("teach message announces take 1 of 3", /Take 1 of 3/.test(T.teachMsg.textContent), T.teachMsg.textContent.slice(0, 60));
  check("stored durMs is plausible (move only)", t && t.durMs >= 300 && t.durMs < 4000, "durMs " + (t && t.durMs));
}

console.log("T3: pickMainPose stickiness and bystander recovery");
{
  const mk = (cx, size, vis = 0.9) => {
    const p = new Array(33).fill(null);
    p[11] = { x: cx + size / 2, y: 0.5, visibility: vis };
    p[12] = { x: cx - size / 2, y: 0.5, visibility: vis };
    p[23] = { x: cx + size / 4, y: 0.5 + size * 1.4, visibility: vis };
    p[24] = { x: cx - size / 4, y: 0.5 + size * 1.4, visibility: vis };
    return p;
  };
  // Duplicate poses of one person, alternating a 5% size advantage: the old
  // memoryless pick flip-flopped every frame, a whole-skeleton jump each time.
  T.pickMainPose([]); // reset lastMainMid
  let switches = 0, last = null;
  for (let i = 0; i < 60; i++) {
    const A = mk(0.50, i % 2 ? 0.20 : 0.21);
    const B = mk(0.53, i % 2 ? 0.21 : 0.20);
    const got = T.pickMainPose([A, B]);
    const id = got === A ? "A" : "B";
    if (last && id !== last) switches++;
    last = id;
  }
  check("duplicate poses no longer flip-flop", switches <= 1, switches + " switches");

  // Bystander latch: performer drops out briefly, must win back on return.
  T.pickMainPose([]);
  const performer = mk(0.5, 0.3), bystander = mk(0.12, 0.1);
  for (let i = 0; i < 5; i++) T.pickMainPose([performer, bystander]);
  for (let i = 0; i < 3; i++) T.pickMainPose([bystander]); // performer lost
  const back = T.pickMainPose([performer, bystander]);
  check("performer wins back after a dropout", back === performer);
}

console.log("T4: smoothPose does not freeze on a mid-move visibility dip");
{
  const mk = (x, vis) => { const l = new Array(33).fill(null); l[11] = { x, y: 0.5, visibility: vis }; return l; };
  let now = 500000; // a time gap > 0.5s reinitializes the filter
  T.smoothPose(mk(0, 0.8), now);
  let worst = 0;
  for (let i = 1; i < 60; i++) {
    now += 33;
    const target = i * 0.05; // a brisk sweep, 0.05 units/frame
    const vis = i === 30 || i === 31 ? 0.05 : 0.8; // motion-blur dip
    const out = T.smoothPose(mk(target, vis), now);
    worst = Math.max(worst, Math.abs(target - out[11].x));
  }
  check("worst detachment through the dip < 0.12 units", worst < 0.12, worst.toFixed(3));
}

console.log("T5: trails survive a slow dance and break on tracking loss");
{
  T.overlay.width = 1280; T.overlay.height = 720;
  T.clearHandsOnFace();
  let now = 900000;
  // Slow continuous wrist drift, ~3px/frame against a ~2px travel gate. The
  // old timestamp-gap heuristic dropped EVERY segment of a dance this slow.
  for (let i = 0; i < 90; i++) {
    const lms = standing();
    lms[15] = { x: 0.68 + i * 0.0025, y: 0.74, visibility: 0.9 };
    T.drawPresence(lms, now);
    now += 33;
  }
  const arr = T.trails.get(15) || [];
  const fresh = arr.filter((p) => now - p.t <= 900);
  const drawable = fresh.filter((p, k) => k > 0 && !fresh[k - 1].brk).length;
  check("slow movement still records a drawable trail", drawable >= 2, drawable + " segments");
  check("no spurious breaks while continuously tracked", fresh.every((p, k) => k === fresh.length - 1 || !p.brk));
}

console.log("T6: migrateSq renormalizes an old 4:3 code to torso length 1");
{
  const f = new Array(66).fill(0);
  f[22] = 0.3; f[23] = -1.2; f[24] = -0.3; f[25] = -1.2; // shoulders 11, 12
  f[46] = 0.2; f[47] = 0; f[48] = -0.2; f[49] = 0;       // hips 23, 24
  const t = T.migrateSq({ word: "x", seq: [f.slice()] });
  const v = t.seq[0];
  const sx = (v[22] + v[24]) / 2, sy = (v[23] + v[25]) / 2;
  const hx = (v[46] + v[48]) / 2, hy = (v[47] + v[49]) / 2;
  const torso = Math.hypot(sx - hx, sy - hy);
  check("migrated torso length is 1", Math.abs(torso - 1) < 1e-9, torso.toFixed(6));
  check("migration marks sq:1", t.sq === 1);
}

console.log("T7: low light hides guessed nodes and shows the notice");
{
  T.clearHandsOnFace();
  let now = 1200000;
  const frame = (vis) => {
    const lms = standing();
    lms[15] = { x: 0.68, y: 0.74, visibility: vis };
    return lms;
  };
  // First make the wrist node hidden (vis under the normal 0.22 keep gate).
  for (let i = 0; i < 5; i++) { T.drawPresence(frame(0.05), now); now += 33; }
  const trailLen = () => (T.trails.get(15) || []).filter((p) => !p.brk && now - p.t <= 900).length;

  T.setLowLight(true);
  check("low-light notice is shown", T.lowLight === true && T.lightNote.hidden === false);
  // A half-confident wrist (0.5) clears the normal 0.38 show gate but must
  // NOT clear the low-light 0.6 gate: no node, so no trail points either.
  const before = trailLen();
  for (let i = 0; i < 20; i++) {
    const lms = frame(0.5);
    lms[15].x = 0.68 + i * 0.004; // moving, so the travel gate is not the reason
    T.drawPresence(lms, now); now += 33;
  }
  check("half-confident wrist is not drawn in low light", trailLen() <= before, trailLen() + " pts");

  T.setLowLight(false);
  check("notice clears with the light", T.lowLight === false && T.lightNote.hidden === true);
  for (let i = 0; i < 20; i++) {
    const lms = frame(0.5);
    lms[15].x = 0.68 + i * 0.004;
    T.drawPresence(lms, now); now += 33;
  }
  check("same wrist draws again in normal light", trailLen() >= 2, trailLen() + " pts");
}

console.log("T8: a full palm cover (face AND wrist occluded) completes both 3s holds");
{
  // The video-reproduced bug: covering the face properly hides the nose, the
  // ears and the covering wrist. The face anchor then expired mid-hold
  // (1500ms stickiness + 700ms grace < 3000ms hold) and the countdown
  // canceled right around "1". The whole teach must now survive it.
  T.clearHandsOnFace();
  const before = T.templates.length;
  T.wordInput.value = "occlusion";
  T.startTeach();
  let now = performance.now() + 120000;
  const vecAt = (w) => { const v = new Array(66).fill(0); v[30] = w; v[31] = -w; return v; };
  const step = (lms, vec) => {
    const h = T.updateHandsOnFace(lms, now);
    if (T.teach) T.teachStep(vec, h, T.isNearFace(), now, true);
    now += 33;
    return h;
  };
  // Right hand covers, fully visible at first (that is how the latch forms).
  const coverR = standing();
  coverR[16] = { x: 0.5, y: 0.35, visibility: 0.9 };
  coverR[14] = { x: 0.5, y: 0.52, visibility: 0.9 };
  // Then the palm hides the face AND its own wrist: only shoulders stay seen.
  const hiddenR = standing();
  hiddenR[0] = { x: 0.5, y: 0.3, visibility: 0.05 };
  hiddenR[7] = { x: 0.55, y: 0.3, visibility: 0.05 };
  hiddenR[8] = { x: 0.45, y: 0.3, visibility: 0.05 };
  hiddenR[16] = { x: 0.5, y: 0.35, visibility: 0.05 };
  hiddenR[14] = { x: 0.5, y: 0.52, visibility: 0.9 };

  for (let i = 0; i < 5; i++) step(standing(), vecAt(0));   // arm: right hand off
  for (let i = 0; i < 20; i++) step(coverR, vecAt(0));      // visible cover -> starting
  check("visible cover starts the countdown", T.teach && T.teach.state === "starting");
  let rHeld = true;
  for (let i = 0; i < 100 && T.teach.state === "starting"; i++) {
    const h = step(hiddenR, vecAt(0));
    if (T.teach.state === "starting" && !h.right) rHeld = false;
  }
  check("latch survives 3s of full occlusion", rHeld);
  check("occluded hold still begins the capture", T.teach && T.teach.state === "capturing");

  // The move: sweep, then settle; the right hand is seen away and releases.
  for (let i = 0; i <= 16; i++) step(standing(), vecAt(i * 0.05));
  for (let i = 0; i < 10; i++) step(standing(), vecAt(0.8));
  check("wrist seen away releases after the hold", !T.handOnFace.right.on);

  // Left hand covers to stop, then is swallowed by the palm the same way.
  const coverL = standing();
  coverL[15] = { x: 0.5, y: 0.35, visibility: 0.9 };
  coverL[13] = { x: 0.5, y: 0.52, visibility: 0.9 };
  const hiddenL = standing();
  hiddenL[0] = { x: 0.5, y: 0.3, visibility: 0.05 };
  hiddenL[7] = { x: 0.55, y: 0.3, visibility: 0.05 };
  hiddenL[8] = { x: 0.45, y: 0.3, visibility: 0.05 };
  hiddenL[15] = { x: 0.5, y: 0.35, visibility: 0.05 };
  hiddenL[13] = { x: 0.5, y: 0.52, visibility: 0.9 };
  for (let i = 0; i < 20; i++) step(coverL, vecAt(0.8));
  for (let i = 0; i < 110 && T.teach; i++) step(hiddenL, vecAt(0.8));
  check("occluded left hold stops and saves", T.teach === null);
  check("the occluded take was saved", T.templates.length === before + 1, "len " + T.templates.length);
}

console.log("T9: a single-frame skeleton collapse no longer bypasses smoothing");
{
  const mk = (x, vis = 0.8) => { const l = new Array(33).fill(null); l[11] = { x, y: 0.5, visibility: vis }; return l; };
  let now = 3000000; // time gap reinitializes the filter
  T.smoothPose(mk(0.5), now);
  for (let i = 0; i < 20; i++) { now += 33; T.smoothPose(mk(0.5), now); }
  now += 33;
  const garbage = T.smoothPose(mk(0.95), now); // one collapsed frame
  check("one garbage jump is damped, not followed", garbage[11].x < 0.76, garbage[11].x.toFixed(3));
  let out = null;
  for (let i = 0; i < 5; i++) { now += 33; out = T.smoothPose(mk(0.5), now); }
  check("recovers onto the body right after", Math.abs(out[11].x - 0.5) < 0.1, out[11].x.toFixed(3));
  // A REAL person switch holds its new position: two jumped frames reset.
  for (let i = 0; i < 20; i++) { now += 33; T.smoothPose(mk(0.5), now); }
  now += 33;
  T.smoothPose(mk(0.95), now);
  now += 33;
  const switched = T.smoothPose(mk(0.95), now);
  check("a persistent jump resets onto the new person", switched[11].x === 0.95, switched[11].x.toFixed(3));
}

console.log("T11: pose-card frames are spread over travel, not time");
{
  // 20 frames: the body lingers for the first half, then sweeps a wrist.
  const frame = (w) => { const v = new Array(66).fill(0); v[30] = w; return v; };
  const seq = [];
  for (let i = 0; i < 10; i++) seq.push(frame(0));
  for (let i = 0; i < 10; i++) seq.push(frame(i * 0.1));
  const idxs = T.stripIndices(seq, 5);
  check("five indices, first and last frame included", idxs.length === 5 && idxs[0] === 0 && idxs[4] === 19, idxs.join(","));
  check("indices strictly increasing", idxs.every((v, k) => k === 0 || v > idxs[k - 1]));
  const inMove = idxs.filter((i) => i >= 10).length;
  check("cells concentrate where the body moves", inMove >= 3, idxs.join(","));
  const still = T.stripIndices(seq.slice(0, 10), 5);
  check("a still seq falls back to even spacing", still.length === 5 && still[0] === 0 && still[4] === 9, still.join(","));
}

console.log("T12: playback tempo rides the taught BPM, never the raw take length");
{
  // A short sharp take (900ms) used to count at dur/8 ~ 112ms per count.
  const fast = T.pbTempo({ bpm: 80, durMs: 900 });
  check("short take snaps to one bar at its BPM", fast.bars === 1 && fast.nCounts === 4 && fast.dur === 3000, JSON.stringify(fast));
  check("counts tick on the taught quarter note", fast.beat === 750);
  const long = T.pbTempo({ bpm: 110, durMs: 5200 });
  check("a long take rounds to two bars, eight counts", long.bars === 2 && long.nCounts === 8);
  const legacy = T.pbTempo({ durMs: 1000 });
  check("codes without a stored bpm fall back to the default", legacy.beat === 750 && legacy.bars === 1);
  check("fastest allowed bpm still counts at a danceable rate", T.pbTempo({ bpm: 160, durMs: 600 }).beat >= 375);
}

console.log("T13: per-count judging grades and combos like DDR");
{
  const pb = {};
  T.judgeBeat(pb, 1000);
  check("no accumulator: no verdict", !pb.judgeFx);
  pb.beatAcc = { live: 2, on: 2 };
  T.judgeBeat(pb, 1000);
  check("too few live frames: no verdict (empty room never scolded)", !pb.judgeFx);
  pb.beatAcc = { live: 10, on: 8 };
  T.judgeBeat(pb, 1000);
  check("mostly on-ring grades HIT and starts a combo", pb.judgeFx?.grade === "HIT" && pb.combo === 1);
  pb.beatAcc = { live: 10, on: 7 };
  T.judgeBeat(pb, 2000);
  check("another HIT extends the combo", pb.judgeFx?.grade === "HIT" && pb.combo === 2);
  pb.beatAcc = { live: 10, on: 3 };
  T.judgeBeat(pb, 3000);
  check("partially on grades CLOSE and holds the combo", pb.judgeFx?.grade === "CLOSE" && pb.combo === 2);
  pb.beatAcc = { live: 10, on: 0 };
  T.judgeBeat(pb, 4000);
  check("off the rings grades MISS and breaks the combo", pb.judgeFx?.grade === "MISS" && pb.combo === 0);
  check("the accumulator is consumed each time", pb.beatAcc === null);
}

console.log("T10: a canceled countdown takes its 'recording in' label with it");
{
  T.clearHandsOnFace();
  T.wordInput.value = "label";
  T.startTeach();
  let now = performance.now() + 300000;
  const step = (lms) => {
    const h = T.updateHandsOnFace(lms, now);
    if (T.teach) T.teachStep(new Array(66).fill(0), h, T.isNearFace(), now, true);
    now += 33;
  };
  const coverR = standing();
  coverR[16] = { x: 0.5, y: 0.35, visibility: 0.9 };
  coverR[14] = { x: 0.5, y: 0.52, visibility: 0.9 };
  for (let i = 0; i < 5; i++) step(standing());
  for (let i = 0; i < 20; i++) step(coverR);
  check("countdown running", T.teach && T.teach.state === "starting");
  T.updateCaptureOverlay(now);
  check("label shows during the countdown", !T.countLabel.hidden && T.countLabel.textContent === "recording in");
  // Hand drops and is SEEN away: past the grace window the countdown cancels.
  for (let i = 0; i < 35; i++) step(standing());
  check("sustained uncover cancels back to prime", T.teach && T.teach.state === "prime");
  T.updateCaptureOverlay(now);
  check("canceled countdown hides its label", T.countLabel.hidden === true);
}

console.log("T14: the panel hint follows the teach state instead of saying Ready forever");
{
  if (T.teach) T.cancelTeach();
  check("idle hint says Ready", T.teachHint.innerHTML.includes("Ready"));
  T.wordInput.value = "hintword";
  T.startTeach();
  check("priming hint asks for the RIGHT hand", T.teachHint.innerHTML.includes("RIGHT"));
  T.teach.state = "capturing";
  T.updateTeachUI(false); // the hint reflects the state even while untracked
  check("capturing hint asks for the LEFT hand", T.teachHint.innerHTML.includes("LEFT"));
  T.cancelTeach();
  check("cancel restores the Ready hint", T.teachHint.innerHTML.includes("Ready"));
}

console.log("T15: guide step 1 confirms tracking on the card instead of auto-advancing");
{
  T.startTutorial();
  check("guide opens on step 1", T.tut && T.tut.i === 0);
  check("Next is enabled from the start", T.tutNextBtn.disabled === false);
  T.tutNotify("tracked");
  T.tutNotify("tracked"); // the poller repeats the event; the card confirms once
  check("tracked does not advance the step", T.tut && T.tut.i === 0);
  check("the card confirms and points at Next",
    T.tutLockEl.hidden === false && /Next/.test(T.tutLockEl.textContent || ""));
  T.endTutorial();
  check("guide closed", T.tut === null);
}

console.log("T16: a confidently misplaced wrist cannot cancel an active hold");
{
  // The IMG_3167 failure: the palm is on the face, but the model keeps
  // reporting the hidden wrist CONFIDENTLY beside the head. That used to be
  // "seen away" evidence and released the latch mid-countdown.
  T.clearHandsOnFace();
  T.wordInput.value = "halluc";
  T.startTeach();
  let now = performance.now() + 240000;
  const vecAt = (w) => { const v = new Array(66).fill(0); v[30] = w; v[31] = -w; return v; };
  const step = (lms, vec) => {
    const h = T.updateHandsOnFace(lms, now);
    if (T.teach) T.teachStep(vec, h, T.isNearFace(), now, true);
    now += 33;
    return h;
  };
  const coverR = standing();
  coverR[16] = { x: 0.5, y: 0.35, visibility: 0.9 };
  coverR[14] = { x: 0.5, y: 0.52, visibility: 0.9 };
  const hallucR = standing();
  hallucR[16] = { x: 0.64, y: 0.42, visibility: 0.9 }; // beside the head, high vis
  hallucR[14] = { x: 0.62, y: 0.55, visibility: 0.9 }; // garbage elbow too
  for (let i = 0; i < 5; i++) step(standing(), vecAt(0));
  for (let i = 0; i < 20; i++) step(coverR, vecAt(0));
  check("cover starts the countdown", T.teach && T.teach.state === "starting");
  let held = true;
  for (let i = 0; i < 100 && T.teach && T.teach.state === "starting"; i++) {
    const h = step(hallucR, vecAt(0));
    if (T.teach.state === "starting" && !h.right) held = false;
  }
  check("misplaced wrist does not break the start hold", held);
  check("the hold still begins the capture", T.teach && T.teach.state === "capturing");
  // Once the capture is running the right hand is no longer the acting hand,
  // so the SAME wrong wrist releases it at the normal pace.
  for (let i = 0; i < 10; i++) step(hallucR, vecAt(0.4));
  check("outside its hold the same wrist releases normally", !T.handOnFace.right.on);
  // A real uncover during a hold still cancels: wrist clearly seen far away.
  for (let i = 0; i < 20; i++) step(standing(), vecAt(0.6));
  for (let i = 0; i < 20; i++) step(coverR, vecAt(0.8)); // right hand back on face
  check("no restart from the right hand mid-capture", T.teach && T.teach.state === "capturing");
  T.cancelTeach();
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
