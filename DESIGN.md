# AlgoDance: Design

A browser app that turns body movement into language. Using a webcam and
MediaPipe pose tracking, you can **teach** the app a movement (a static pose or
a short dance) and bind it to a word or phrase, then **perform** that movement
to make the word appear on screen.

The name is the premise: an algorithm reading a dance, movement as a private,
embodied code.

---

## Goals

1. **Teach** — record a body movement and assign it a word/phrase.
2. **Perform** — recognize a performed movement live and display the word.
3. Runs entirely in the browser, webcam-enabled, no install.
4. Stores only pose **coordinates**, never webcam video.

Non-goals (for v1): multi-person, full sign-language grammar, cloud sync
(see [SHARING.md](SHARING.md) for the future shared-database plan).

---

## Architecture

```
webcam ─▶ MediaPipe PoseLandmarker ─▶ 33 landmarks/frame
                                         │
                    normalize (torso-relative)  ─▶ pose vector (66 numbers)
                                         │
        ┌────────────── Teach ──────────┴────────── Perform ──────────────┐
        │ record frames between rest      rest-delimited segmentation      │
        │ poses, trim, resample to 20     capture movement between rests   │
        │ ─▶ template stored in           ─▶ resample ─▶ DTW vs templates  │
        │    localStorage                 ─▶ best < threshold ⇒ show word  │
        └─────────────────────────────────────────────────────────────────┘
```

Single-page, no build step. Files: `index.html`, `style.css`, `app.js`, and
`backends.js` (ES modules). MediaPipe Tasks-Vision loads from CDN; MoveNet
(TensorFlow.js) is loaded on demand via dynamic `import()` only when selected,
so a failure disables just that algorithm rather than the whole app.

---

## Pose pipeline

- **Algorithms:** user-selectable across two families (default
  `blaze-full`, persisted in localStorage), all normalized to the same 33-slot
  layout by `backends.js`:
  - **BlazePose** (MediaPipe): 33 native points, GPU delegate with CPU
    fallback, `numPoses: 2` then keep the most prominent person.
  - **MoveNet** (TensorFlow.js): 17 COCO points mapped into the 33 slots.
  Because the two disagree on scale, codes are tagged with their family and
  matched only within it (switching algorithm means re-teaching). The mapped
  key indices (nose 0, ears 7/8, shoulders 11/12, wrists 15/16, hips 23/24)
  line up across all families, so rest detection and normalization are shared.
- **Visibility gate:** we only use a frame when the shoulders (landmarks 11,
  12) have visibility > 0.35. Hips are NOT required to be visible: the model
  estimates their position even out of frame, which is enough for
  normalization, and requiring them blocked close face-and-torso framings.

### Normalization (why matching is position/size invariant)

Raw landmark coordinates depend on where you stand and how close you are. We
remove that:

- Translate so the **hip midpoint** is the origin.
- Scale by **torso length** (hip-midpoint to shoulder-midpoint).
- Keep `x, y` only (MediaPipe `z` is noisier). Result: a 66-number vector per
  frame that is the same whether you are near/far, left/right in frame.

---

## The resting pose (gesture boundaries)

Gestures are **bracketed by the hand-over-face rest pose**: covering the face
arms a capture, moving the hand away starts it, and covering the face again
ends it. Nothing else can start a recording, so ordinary movement never fires
by accident. Frames at either end where a wrist is still near the face (the
trip to and from the pose) are trimmed, and a capture whose trimmed frames
never travel a minimum distance is dropped as a false start. For anyone who
prefers explicit control, a **Manual trigger** mode (hold a button or Spacebar)
captures instead.

`isResting()` is true when either wrist sits near the face (distance normalized
by shoulder width) AND is roughly under the face centre horizontally. The
horizontal-centring gate exists because the pose tracks the wrist, which sits
below the face when the palm covers it, so a radius alone cannot distinguish a
palm over the face from a hand beside it. Both conditions use hysteresis (easier
to enter than to leave) to stop boundary flicker. The face anchor averages whichever of nose/ears are visible,
with a low visibility gate, because the covering hand itself occludes the nose,
and is then dropped a quarter shoulder-width toward the chin: the wrist hangs
at chin level when a palm covers the face, and an eye-level target made people
reach up to their forehead to trigger it.
The face was chosen over the hips because close framings often crop the hips or
track them weakly, while the face stays solid. The app draws a **target circle
over the face** that lights up when a wrist is close enough, so the threshold
is visible rather than guessed. Only the upper body (face, shoulders, wrists,
and roughly-estimated hips for normalization) needs to be in frame, so close
framing and seated use both work.

- **Perform:** a small state machine: `idle` (waiting for the face to be
  covered) -> `armed` (hand on face) -> `recording` (hand moved away) -> back
  to `armed` when the face is covered again, which trims and matches the
  capture. A capture that never returns to the face is abandoned after 10 s.
- **Teach:** the identical bracket. After the countdown it waits for the hand
  on the face, records from the moment it leaves, and saves when the face is
  covered again. Near-face frames are trimmed from both ends, so every stored
  template spans only the movement itself. In Manual trigger mode the record
  button becomes a start/stop toggle.

---

## Matching (Dynamic Time Warping)

- Each captured movement is **resampled to 20 frames** so lengths are uniform.
- Distance between two poses = mean per-landmark euclidean distance.
- Distance between two sequences = **DTW**, normalized by path length. DTW is
  robust to tempo differences (fast vs slow performance of the same move).
- The performed movement is compared against every saved template; the lowest
  distance wins if it is below the **sensitivity threshold**.

### Calibration

The Perform panel shows the live **best match** and its **distance**, plus a
confidence bar. The sensitivity slider is the threshold. Lower = stricter.
Users watch the number while performing to find a good threshold for their
codes and their space.

---

Ghost playback anchors on the live body when one is in frame: because the
stored seq is normalized to hip-centre/torso-length, it is reprojected using
the performer's current hip midpoint and torso length so the ghost dances on
them at their size, falling back to a fixed centre-screen spot when no body is
detected.

## Data model

A template (one code):

```json
{
  "id": "c_ab12cd3",
  "word": "hello",
  "seq": [[x0,y0, x1,y1, ... 66 numbers], ... 20 frames],
  "durMs": 2000,
  "createdAt": 1720600000000
}
```

`seq` is the only movement data: normalized pose coordinates. No imagery.

**Storage:** `localStorage` under `algodance.templates.v1`. Export/import as a
JSON file is supported for backup and manual sharing today.

---

## UI / theme

- Six tabs: **How it works** (default), **Teach**, **Perform**, **Codes**,
  **About**, **Zine**.
- Big webcam stage with skeleton overlay (mirrored, selfie-style); matched word
  animates large over the video; a phrase strip collects matched words.
- Perform shows only its core (your codes, the match meter, the phrase);
  routine building and tuning sit behind disclosures.
- **Instructions are stamped, not typed:** the teach/perform loop is a strip of
  one-imperative scraps (`.stamp-steps`), with prose demoted to "fine print"
  disclosures. The critical R-then-L instruction lives in exactly one voice.
- **Terminology:** a *move* says a *word or phrase*; repeated recordings of the
  same word are *takes*. "Example"/"template" never appear in UI copy.

### The mimetic first lesson / attract loop

The first thing a visitor meets is the seed ghost, not a paragraph: on the
opening tab (and on a kiosk idle for 45 s) the AlgoDance ghost loops with a
taped caption ("Dance along. This move says ALGODANCE"). Each ghost cycle
scores whatever the visitor danced alongside it (lenient threshold, the word is
unambiguous), and a good copy fires the word: the first exchange with the
piece is danced. On a kiosk, two cycles of a body dancing without matching
hands the room back to normal matching, so the loop never traps a performer.

### Rejection is never silent

A completed move that matches nothing gets a quiet paper scrap near the figure
("Saw that. Nothing matched yet; make your move bigger"), rate-limited to one
per 4 s. The same element carries the closest-match coaching; both render in
kiosk mode, where the match meter does not exist. Silence reading as "it can't
see me" is the top abandonment driver for gesture installations.

### The back of the sheet (operator side)

Paste-ups have a front (the work) and a back (the pencil notes). Everything
that can reconfigure or wipe the installation lives on the back: pose
algorithm, layout rotation, background cutout, idle calibration, kiosk entry,
delete-all, and the word-wall clear. It opens by holding the registration mark
in the corner for ~2 s (the wordmark on phones, where the crop marks are
hidden), or Shift+O. Switching algorithm family asks an in-world confirm that
names the consequence (the other family's moves stop matching). The browser's
`confirm()`/`prompt()` are never used; a paper-dialog scrap asks instead.

### The word wall

Every word the installation has ever spoken is taped to the tile wall, UNDER
the paper surfaces (z-index 0), so language peeks out of the gutters and the
wall slowly disappears beneath it. One scrap per distinct word, placed once
and persisted (`algodance.wall.v1`); saying a word again grows its scrap
(log-scaled, capped), so the wall records what the room says most. Hidden on
phones and in kiosk, where the video owns the screen.

### Hands-only word picking

On the idle Teach tab the dictionary's recent words hang along the video edge
as scraps; holding a wrist over one for ~1.2 s picks it into the word field and
arms the hands-free re-arm, so a whole take can be taught without touching the
keyboard. **Privacy constraint: the browser's built-in speech recognition (Web
Speech API) must never be used for word entry; in Chrome it ships audio to
Google's servers, which breaks the app's core claim. On-device WASM speech
(Vosk/Whisper) is acceptable future work.**

### Accessibility floor (verified)

- Red as *text* on paper uses `--accent-2` `#D40021` (4.75:1) or `#B3001B` on
  the post-it; the bright `--accent` red is reserved for fills, edges, and
  second passes. Muted text on the darker paper uses `--ink-soft-2` `#4E463A`
  (5.4:1 on paper-3).
- Buttons have a 40 px touch floor (36 px small, 30 px tiny, 28 px chip-x);
  tabs 38 px.
- Nothing glows: the REC dot and the current count dot print with off-register
  second passes, and the former glassy pills (playback controls, zine chrome,
  kiosk exit, kiosk phrase chips, closest-match hint) are paper scraps.
- The locked guide step states its reason on the card (never a tooltip) and
  unlocks Next after 45 s or two failed takes; the manual-trigger fallback is
  reachable from the Teach panel itself ("Record with a button instead").
- A recording interrupted by a lock screen cancels with a plain explanation
  instead of resurfacing as a stale REC.

### Look: mixed media on a tiled wall

The page is a paste-up, not a screen. The wall stays ceramic tile (the bathroom
motif nods to queer space) and everything else is a scrap of paper taped to it,
cut and tilted a fraction off square. The livestream is the one photographic
element, torn-edge clipped and screened with halftone so it belongs to the same
print run as the page around it.

- **Substrate:** white ceramic tiles with grime in the grout and photographic
  grain over the whole wall. Paper scraps sit on top with a real cast shadow.
- **Inks:** riso red `#FF002A` and yellow `#ECFF00`, printed on warm paper
  rather than glowing on black, plus ballpoint blue `#14417A` for hand-drawn
  marks. Ink `#1C1815` is the text colour.
- **Type, three voices:** Bodoni Moda for headlines (cut from a magazine),
  Courier Prime for labels and marginalia, Archivo for reading. Self-hosted
  as latin-subset woff2 in `fonts/` (117KB, variable files where the family
  has one): a webfont request would tell a third party the page was opened,
  from which IP and when, which is exactly the claim the app makes about
  itself.
- **Joint markers print rather than glow:** each is a disc punched out of
  paper with a halftone screen inside, a cut ink edge, and a second ink pass
  a hair off register. Trails are opaque marker strokes, not additive light.
- Paste-up angles and torn cuts are removed wherever a surface goes full-bleed
  (phone, kiosk, the zine), because a paper edge stretched edge to edge just
  leaks gaps at the viewport.

### Background cutout

Opt-in, off by default, toggled under the video so it is reachable from Teach
as well as Perform. BlazePose emits a person mask from the same inference it
already runs, so the model cost is an option rather than a second network; the
compositing is the real cost. Masks feed an exponential average (a raw mask
boils at the edge) and the composite runs every animation frame off the latest
one (compositing only on mask arrival makes the body stutter at detection rate
while the background moves smoothly). A black outline is drawn from a dilated
silhouette built at mask resolution. Matching never sees any of it; it works on
landmarks, not pixels.

---

## Limitations / known trade-offs

- 2D only (ignores depth); movements that differ mainly in depth can collide.
- Single pose; no two-person duets.
- Static poses work but must still be bracketed by rest to trigger.
- Recognition quality depends on lighting, framing, and threshold calibration.
- Similar movements bound to different words will be hard to disambiguate.

---

## Decisions (confirmed)

1. **Use / audience:** audience-facing — performance and installation. Favors a
   clean big-word display, a match sound, and reliable triggering in front of
   people.
2. **Dynamic** capture (not static-only): gestures are movement sequences
   delimited by the rest pose. Implemented.
3. **Full body — arms and legs:** matching compares all 33 landmarks, so leg and
   whole-body dance movement counts, not just arms. Implemented.
4. **Output:** on-screen text **plus a ping** (a short confirmation tone) on each
   match. A "Ping on match" toggle is in the Perform panel. Implemented.
5. **Multiple examples per word:** yes. Record the same word several times; each
   take is stored as an example and the best (nearest) match across a word's
   examples wins. The Codes list groups examples under one word. Implemented.
6. **Seated / accessibility:** yes. The rest pose (a hand over the face) only
   needs the head, shoulders, and one wrist in frame, so it works standing or
   seated. A **Manual trigger** mode (hold a button or Spacebar) is provided as a
   fallback for anyone who cannot reliably reach or hold the rest pose.
   Implemented.
7. **Shared library:** yes, planned for later. See [SHARING.md](SHARING.md).
   v1 ships with local storage + JSON export/import.

Still open: text-to-speech output (in addition to the ping), a dedicated
full-screen performance view, and per-code thresholds.

---

## Roadmap

- [x] v1: teach + perform, rest segmentation, localStorage, export/import
- [x] Ping on match; sound toggle
- [x] Multiple examples per word (nearest-of-N matching)
- [x] Seated support + manual (hold) trigger for accessibility
- [x] Ghost playback: click a saved code to replay its skeleton movement
  (reprojected from stored coordinates, no video involved)
- [x] Full-screen performance/installation view (kiosk mode)
- [x] Mixed-media visual world; optional background cutout
- [x] Operator "back of the sheet" (hidden settings surface; model-switch guard)
- [x] Mimetic first lesson + kiosk attract loop (copy the seed ghost)
- [x] Persistent word wall (everything the installation has said)
- [x] Visible rejection feedback (miss scraps, also in kiosk)
- [x] Hands-only word picking (dwell over a word scrap)
- [x] Stamped instruction strips; in-world dialogs replace confirm()/prompt()
- [ ] On-device (WASM) speech-to-text for word entry; never the Web Speech API
- [ ] Per-code threshold
- [ ] Shared code library (see [SHARING.md](SHARING.md))
