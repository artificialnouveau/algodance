// Pose-estimation backends. Each backend exposes the SAME interface so the app
// never cares which algorithm is running:
//
//   backend.family                 "blaze" | "movenet"
//   backend.connections            [[i,j], ...] slot pairs to draw as bones
//   await backend.load()           fetch weights / build the session
//   await backend.detect(video,ts) -> array of poses; each pose is a 33-slot
//                                     array of {x, y, visibility} in image-
//                                     normalized coords (0..1, origin top-left)
//   backend.close()                release resources
//
// Every backend outputs the SAME 33-slot layout as MediaPipe BlazePose, so the
// rest of the pipeline (normalization, rest detection, matching, drawing) is
// identical regardless of algorithm. 17-keypoint models (MoveNet) fill only the
// slots they have; the unused slots stay at visibility 0. Because a code is
// only ever matched against codes from the SAME family, the constant empty
// slots do not affect recognition.
//
// Only BlazePose is imported statically (it is the default and must always
// work). MoveNet is loaded with dynamic import() the first time it is
// selected, so a CDN failure disables just that algorithm instead of breaking
// the whole app.

import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

export const NUM_LMS = 33;

// COCO-17 keypoint index -> BlazePose 33-slot index. This places COCO joints
// into the same slots BlazePose uses, so the app's key indices (nose 0, ears
// 7/8, shoulders 11/12, wrists 15/16, hips 23/24) line up for every backend.
const COCO_TO_BLAZE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

// Bones to draw for a 17-keypoint (COCO) skeleton, in BlazePose slot indices.
export const COCO_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 2], [0, 5], [2, 7], [5, 8],
];

// Full BlazePose bone list, derived from the model's own connection table.
export const BLAZE_CONNECTIONS = (PoseLandmarker.POSE_CONNECTIONS || []).map(
  (c) => [c.start, c.end]
);

function emptyPose() {
  const p = new Array(NUM_LMS);
  for (let i = 0; i < NUM_LMS; i++) p[i] = { x: 0, y: 0, visibility: 0 };
  return p;
}

// Turn 17 normalized COCO keypoints ({x,y,score}) into a 33-slot pose.
function cocoToPose(kpts) {
  const p = emptyPose();
  for (let i = 0; i < 17; i++) {
    const k = kpts[i];
    if (!k) continue;
    p[COCO_TO_BLAZE[i]] = { x: k.x, y: k.y, visibility: k.score ?? 0 };
  }
  return p;
}

// ---------- BlazePose (MediaPipe Tasks-Vision) ----------
const MP_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/";
const BLAZE_URLS = {
  "blaze-lite": MP_BASE + "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  "blaze-full": MP_BASE + "pose_landmarker_full/float16/1/pose_landmarker_full.task",
  "blaze-heavy": MP_BASE + "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
};
let mpFileset = null;

class BlazeBackend {
  constructor(key) { this.key = key; this.family = "blaze"; this.connections = BLAZE_CONNECTIONS; this.lm = null; }
  async load() {
    mpFileset = mpFileset || await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    const make = (delegate) => PoseLandmarker.createFromOptions(mpFileset, {
      baseOptions: { modelAssetPath: BLAZE_URLS[this.key], delegate },
      runningMode: "VIDEO",
      // 2 is enough to still see and reject a background bystander (the app
      // keeps the larger, nearer skeleton) while roughly halving the tracking
      // cost of 3.
      numPoses: 2,
    });
    try { this.lm = await make("GPU"); }
    catch (e) { console.warn("BlazePose GPU failed, using CPU:", e); this.lm = await make("CPU"); }
  }
  // MediaPipe returns 33 landmarks {x,y,z,visibility} already in image space.
  async detect(video, ts) { return this.lm.detectForVideo(video, ts).landmarks || []; }
  close() { try { this.lm?.close(); } catch {} }
}

// ---------- MoveNet (TensorFlow.js) ----------
class MoveNetBackend {
  constructor(key) { this.key = key; this.family = "movenet"; this.connections = COCO_CONNECTIONS; this.detector = null; }
  async load() {
    const tf = await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.22.0/+esm");
    await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@4.22.0/+esm");
    await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.22.0/+esm");
    await tf.setBackend("webgl");
    await tf.ready();
    const pd = await import("https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/+esm");
    const modelType = this.key === "movenet-thunder"
      ? pd.movenet.modelType.SINGLEPOSE_THUNDER
      : pd.movenet.modelType.SINGLEPOSE_LIGHTNING;
    this.detector = await pd.createDetector(pd.SupportedModels.MoveNet, { modelType });
  }
  async detect(video) {
    const w = video.videoWidth || 1, h = video.videoHeight || 1;
    const poses = await this.detector.estimatePoses(video, { flipHorizontal: false });
    return poses.map((p) =>
      cocoToPose(p.keypoints.map((k) => ({ x: k.x / w, y: k.y / h, score: k.score })))
    );
  }
  close() { try { this.detector?.dispose(); } catch {} }
}

export function createBackend(key) {
  if (key.startsWith("blaze")) return new BlazeBackend(key);
  if (key.startsWith("movenet")) return new MoveNetBackend(key);
  throw new Error("unknown algorithm: " + key);
}
