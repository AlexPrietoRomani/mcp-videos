(function (global) {
  "use strict";

  if (global.__MCP_VIDEOS_HARNESS__) {
    return;
  }

  var virtualNow = 0;
  var performanceObject = global.performance;

  if (performanceObject && typeof performanceObject.now === "function") {
    try {
      performanceObject.now = function () {
        return virtualNow;
      };
    } catch (error) {
      virtualNow = Number(performanceObject.now()) || 0;
    }
  }

  function clamp(value, min, max) {
    var number = Number(value);
    var lower = Number(min);
    var upper = Number(max);
    if (!isFinite(number)) {
      number = 0;
    }
    if (!isFinite(lower)) {
      lower = 0;
    }
    if (!isFinite(upper)) {
      upper = 1;
    }
    if (lower > upper) {
      var swap = lower;
      lower = upper;
      upper = swap;
    }
    return Math.max(lower, Math.min(upper, number));
  }

  function lerp(start, end, amount) {
    var from = Number(start) || 0;
    var to = Number(end) || 0;
    return from + (to - from) * clamp(amount, 0, 1);
  }

  function easeOutCubic(value) {
    var x = clamp(value, 0, 1);
    return 1 - Math.pow(1 - x, 3);
  }

  function easeInOutQuad(value) {
    var x = clamp(value, 0, 1);
    if (x < 0.5) {
      return 2 * x * x;
    }
    return 1 - Math.pow(-2 * x + 2, 2) / 2;
  }

  function spring(value, options) {
    var time = Math.max(0, Number(value) || 0);
    var config = options && typeof options === "object" ? options : {};
    var stiffness = Math.max(1, Number(config.stiffness) || 170);
    var damping = Math.max(0, Number(config.damping) || 26);
    var mass = Math.max(0.0001, Number(config.mass) || 1);
    var omega0 = Math.sqrt(stiffness / mass);
    var zeta = damping / (2 * Math.sqrt(stiffness * mass));
    var result;

    if (zeta < 1) {
      var omegaD = omega0 * Math.sqrt(1 - zeta * zeta);
      var envelope = Math.exp(-zeta * omega0 * time);
      result =
        1 -
        envelope *
          (Math.cos(omegaD * time) + ((zeta * omega0) / omegaD) * Math.sin(omegaD * time));
    } else if (Math.abs(zeta - 1) < 1e-6) {
      result = 1 - Math.exp(-omega0 * time) * (1 + omega0 * time);
    } else {
      var root = Math.sqrt(zeta * zeta - 1);
      var r1 = -omega0 * (zeta - root);
      var r2 = -omega0 * (zeta + root);
      result = 1 - (r2 * Math.exp(r1 * time) - r1 * Math.exp(r2 * time)) / (r2 - r1);
    }

    return clamp(result, 0, 1);
  }

  function progress(timeMs, startMs, endMs) {
    var time = Number(timeMs) || 0;
    var start = Number(startMs) || 0;
    var end = Number(endMs) || 0;
    if (end <= start) {
      return 1;
    }
    return clamp((time - start) / (end - start), 0, 1);
  }

  function readTime(word, keys) {
    for (var index = 0; index < keys.length; index += 1) {
      var value = word ? word[keys[index]] : undefined;
      if (value !== undefined && value !== null) {
        var number = Number(value);
        if (isFinite(number)) {
          return number;
        }
      }
    }
    return null;
  }

  function cueList(words) {
    if (Array.isArray(words)) {
      return words;
    }
    if (words && Array.isArray(words.cues)) {
      return words.cues;
    }
    return [];
  }

  function activeWordIndexAt(timeMs, words, options) {
    var time = Number(timeMs) || 0;
    var inclusiveEnd = Boolean(options && options.inclusiveEnd);
    var cues = cueList(words);

    for (var index = 0; index < cues.length; index += 1) {
      var cue = cues[index];
      if (Array.isArray(cue)) {
        var arrayStart = Number(cue[0]);
        var arrayEnd = Number(cue[1]);
        if (time >= arrayStart && (time < arrayEnd || (inclusiveEnd && time <= arrayEnd))) {
          return index;
        }
        continue;
      }
      var start = readTime(cue, ["start_ms", "startMs", "start"]);
      var end = readTime(cue, ["end_ms", "endMs", "end"]);
      if (start === null || end === null) {
        continue;
      }
      if (time >= start && (time < end || (inclusiveEnd && time <= end))) {
        return index;
      }
    }
    return -1;
  }

  function activeWordAt(timeMs, words, options) {
    var index = activeWordIndexAt(timeMs, words, options);
    if (index < 0) {
      return null;
    }
    return cueList(words)[index];
  }

  function syncAnimations(timeMs) {
    var documentObject = global.document;
    if (!documentObject || typeof documentObject.getAnimations !== "function") {
      return;
    }

    var animations;
    try {
      animations = documentObject.getAnimations({ subtree: true });
    } catch (error) {
      animations = documentObject.getAnimations();
    }

    for (var index = 0; index < animations.length; index += 1) {
      var animation = animations[index];
      try {
        if (typeof animation.pause === "function") {
          animation.pause();
        }
        animation.currentTime = timeMs;
      } catch (error) {
        continue;
      }
    }
  }

  function stepDampedAt(timeMs, keyframes, tauSec) {
    var rawKeys = Array.isArray(keyframes) ? keyframes : [];
    if (!rawKeys.length) {
      return { value: 0, target: 0, index: -1 };
    }
    var normalized = [];
    for (var i = 0; i < rawKeys.length; i++) {
      var kf = rawKeys[i];
      if (Array.isArray(kf)) {
        normalized.push({ atMs: Number(kf[0]) || 0, value: Number(kf[1]) || 0 });
      } else if (kf && typeof kf === "object") {
        var at = Number(kf.atMs !== undefined ? kf.atMs : kf.at) || 0;
        var val = Number(kf.value !== undefined ? kf.value : kf.val) || 0;
        normalized.push({ atMs: at, value: val });
      }
    }
    normalized.sort(function (a, b) { return a.atMs - b.atMs; });

    var t = Math.max(0, Number(timeMs) || 0);
    if (t <= normalized[0].atMs) {
      return { value: normalized[0].value, target: normalized[0].value, index: 0 };
    }

    var tau = Math.max(1e-6, Number(tauSec !== undefined ? tauSec : 0.28) || 0.28);
    var curVal = normalized[0].value;
    var curTime = normalized[0].atMs;
    var curTarget = normalized[0].value;
    var curIdx = 0;

    for (var j = 1; j < normalized.length; j++) {
      var nextTime = normalized[j].atMs;
      var nextVal = normalized[j].value;
      if (t <= nextTime) {
        var dtSec = (t - curTime) / 1000;
        curVal = curTarget + (curVal - curTarget) * Math.exp(-dtSec / tau);
        return { value: curVal, target: curTarget, index: curIdx };
      }
      var segDt = (nextTime - curTime) / 1000;
      curVal = curTarget + (curVal - curTarget) * Math.exp(-segDt / tau);
      curTime = nextTime;
      curTarget = nextVal;
      curIdx = j;
    }

    var finalDt = (t - curTime) / 1000;
    curVal = curTarget + (curVal - curTarget) * Math.exp(-finalDt / tau);
    return { value: curVal, target: curTarget, index: curIdx };
  }

  function stepSpringAt(timeMs, keyframesOrTargetFn, options) {
    var opts = options && typeof options === "object" ? options : {};
    var stiffness = Math.max(0.1, Number(opts.stiffness !== undefined ? opts.stiffness : 420) || 420);
    var damping = Math.max(0, Number(opts.damping !== undefined ? opts.damping : 26) || 26);
    var mass = Math.max(0.0001, Number(opts.mass !== undefined ? opts.mass : 1) || 1);

    var getTarget;
    var initTarget = 0;

    if (typeof keyframesOrTargetFn === "function") {
      getTarget = keyframesOrTargetFn;
      try {
        initTarget = Number(getTarget(0)) || 0;
      } catch (e) {
        initTarget = 0;
      }
    } else if (Array.isArray(keyframesOrTargetFn) && keyframesOrTargetFn.length > 0) {
      var normalized = [];
      for (var i = 0; i < keyframesOrTargetFn.length; i++) {
        var kf = keyframesOrTargetFn[i];
        if (Array.isArray(kf)) {
          normalized.push({ atMs: Number(kf[0]) || 0, value: Number(kf[1]) || 0 });
        } else if (kf && typeof kf === "object") {
          var at = Number(kf.atMs !== undefined ? kf.atMs : kf.at) || 0;
          var val = Number(kf.value !== undefined ? kf.value : kf.val) || 0;
          normalized.push({ atMs: at, value: val });
        }
      }
      normalized.sort(function (a, b) { return a.atMs - b.atMs; });
      initTarget = normalized[0].value;

      getTarget = function (tMs) {
        if (tMs <= normalized[0].atMs) {
          return normalized[0].value;
        }
        var targetVal = normalized[0].value;
        for (var k = 1; k < normalized.length; k++) {
          if (tMs >= normalized[k].atMs) {
            targetVal = normalized[k].value;
          } else {
            break;
          }
        }
        return targetVal;
      };
    } else {
      getTarget = function () { return 0; };
    }

    var totalSec = Math.max(0, Number(timeMs) || 0) / 1000;
    var x = opts.initialValue !== undefined ? Number(opts.initialValue) || 0 : initTarget;
    var v = opts.initialVelocity !== undefined ? Number(opts.initialVelocity) || 0 : 0;

    if (totalSec <= 0) {
      return { value: x, velocity: v, target: getTarget(0) };
    }

    var dt = 1 / 240;
    var steps = Math.floor(totalSec / dt);
    var remainder = totalSec - steps * dt;

    for (var s = 0; s < steps; s++) {
      var currentTms = s * dt * 1000;
      var target = Number(getTarget(currentTms)) || 0;
      var fSpring = -stiffness * (x - target);
      var fDamping = -damping * v;
      var a = (fSpring + fDamping) / mass;
      v += a * dt;
      x += v * dt;
    }

    if (remainder > 1e-6) {
      var remTms = steps * dt * 1000;
      var remTarget = Number(getTarget(remTms)) || 0;
      var remFSpring = -stiffness * (x - remTarget);
      var remFDamping = -damping * v;
      var remA = (remFSpring + remFDamping) / mass;
      v += remA * remainder;
      x += v * remainder;
    }

    return {
      value: x,
      velocity: v,
      target: Number(getTarget(totalSec * 1000)) || 0,
    };
  }
  var MotionKit = {
    clamp: clamp,
    lerp: lerp,
    easeOutCubic: easeOutCubic,
    easeInOutQuad: easeInOutQuad,
    spring: spring,
    stepDampedAt: stepDampedAt,
    stepSpringAt: stepSpringAt,
    progress: progress,
    progressInRange: progress,
    activeWordAt: activeWordAt,
    activeWordIndexAt: activeWordIndexAt,
  };

  // =========================================================================
  // Anatomy IsoKit integration (wheresryan22/anatomy - MIT License)
  // =========================================================================
const AZIMUTH = 45;
const ELEVATION = 30;
const SHADES = 4;
const RADIANS = Math.PI / 180;
const UNIT = Math.sqrt(1.6);
function cameraOf({ scale, azimuth = AZIMUTH, elevation = ELEVATION }) {
    const a = azimuth * RADIANS;
    const e = elevation * RADIANS;
    return { sinA: Math.sin(a), cosA: Math.cos(a), sinE: Math.sin(e), cosE: Math.cos(e), k: scale * UNIT };
}
function iso([x, y, z], projection) {
    const { origin } = projection;
    const { sinA, cosA, sinE, cosE, k } = cameraOf(projection);
    return [origin[0] + (x * sinA - y * cosA) * k, origin[1] + ((x * cosA + y * sinA) * sinE - z * cosE) * k];
}
function towardViewer(projection) {
    const { sinA, cosA } = cameraOf(projection);
    return [cosA, sinA];
}
function depthOf([x, y, z], projection) {
    const { sinA, cosA, sinE, cosE } = cameraOf(projection);
    return (x * cosA + y * sinA) * cosE + z * sinE;
}
const tenth = (value) => {
    const text = (Math.round(value * 10) / 10).toFixed(1).replace(/\.0$/, "");
    return text === "-0" ? "0" : text;
};
function pathOf(points, closed = false) {
    if (!points.length)
        return "";
    return points.map(([x, y], at) => `${at ? "L" : "M"}${tenth(x)} ${tenth(y)}`).join("") + (closed ? "Z" : "");
}
const joinPaths = (paths) => paths.filter(Boolean).join("");
function roundedPlan({ x, y, w, d, r }, steps = 8) {
    const radius = Math.max(0, Math.min(r, w / 2, d / 2));
    const corners = [
        [x + w - radius, y + radius, -90],
        [x + w - radius, y + d - radius, 0],
        [x + radius, y + d - radius, 90],
        [x + radius, y + radius, 180],
    ];
    const out = [];
    for (const [cx, cy, start] of corners) {
        const count = radius > 0 ? steps : 0;
        for (let step = 0; step <= count; step++) {
            const angle = (start + (count ? (90 * step) / count : 45)) * RADIANS;
            out.push([cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)]);
        }
    }
    return out;
}
function signedArea(points) {
    let area = 0;
    for (let index = 0; index < points.length; index++) {
        const [ax, ay] = points[index];
        const [bx, by] = points[(index + 1) % points.length];
        area += ax * by - bx * ay;
    }
    return area / 2;
}
const oriented = (points) => (signedArea(points) < 0 ? [...points].reverse() : points);
function convexHull(points) {
    const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (sorted.length < 3)
        return sorted;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const point of sorted) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0)
            lower.pop();
        lower.push(point);
    }
    const upper = [];
    for (let index = sorted.length - 1; index >= 0; index--) {
        const point = sorted[index];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0)
            upper.pop();
        upper.push(point);
    }
    return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}
function runsOf(flags, want) {
    const count = flags.length;
    const runs = [];
    const start = flags.findIndex((flag, index) => flag !== flags[(index + count - 1) % count]);
    if (start < 0)
        return flags[0] === want ? [Array.from({ length: count + 1 }, (_, index) => index % count)] : [];
    let run = null;
    for (let step = 0; step < count; step++) {
        const index = (start + step) % count;
        if (flags[index] === want) {
            if (!run)
                run = [index];
            run.push((index + 1) % count);
        }
        else if (run) {
            runs.push(run);
            run = null;
        }
    }
    if (run)
        runs.push(run);
    return runs;
}
function extrude(ring, z, height, projection, options = {}) {
    const convex = options.convex ?? true;
    const count = ring.length;
    const counter = signedArea(ring) > 0;
    const [vx, vy] = towardViewer(projection);
    const { sinA, cosA } = cameraOf(projection);
    const normals = ring.map(([ax, ay], index) => {
        const [bx, by] = ring[(index + 1) % count];
        const dx = bx - ax;
        const dy = by - ay;
        const length = Math.hypot(dx, dy) || 1;
        return (counter ? [dy / length, -dx / length] : [-dy / length, dx / length]);
    });
    const facing = normals.map(([nx, ny]) => nx * vx + ny * vy > 1e-9);
    const top = ring.map(([x, y]) => iso([x, y, z + height], projection));
    const bottom = ring.map(([x, y]) => iso([x, y, z], projection));
    const outline = [];
    const crease = [];
    for (const run of runsOf(facing, false))
        outline.push(pathOf(run.map((index) => top[index])));
    for (const run of runsOf(facing, true)) {
        crease.push(pathOf(run.map((index) => top[index])));
        if (height > 0)
            outline.push(pathOf(run.map((index) => bottom[index])));
    }
    if (height > 0) {
        for (let index = 0; index < count; index++) {
            if (facing[(index + count - 1) % count] !== facing[index])
                outline.push(pathOf([top[index], bottom[index]]));
        }
    }
    let fill;
    if (convex) {
        fill = pathOf(convexHull([...top, ...bottom]), true);
    }
    else {
        const parts = [pathOf(oriented(top), true), pathOf(oriented(bottom), true)];
        for (let index = 0; index < count; index++) {
            if (!facing[index])
                continue;
            const next = (index + 1) % count;
            parts.push(pathOf(oriented([top[index], top[next], bottom[next], bottom[index]]), true));
        }
        fill = parts.join("");
    }
    const buckets = Array.from({ length: SHADES }, () => []);
    if (height > 0) {
        const [lx, ly] = [-sinA, cosA];
        for (let index = 0; index < count; index++) {
            if (!facing[index])
                continue;
            const [nx, ny] = normals[index];
            const lit = (nx * lx + ny * ly + 1) / 2;
            const shade = Math.min(SHADES - 1, Math.max(0, Math.floor(lit * SHADES)));
            const next = (index + 1) % count;
            buckets[shade].push(pathOf(oriented([top[index], top[next], bottom[next], bottom[index]]), true));
        }
    }
    let bevel;
    if (options.bevel && options.bevel > 0) {
        const by = options.bevel;
        const inset = ring.map(([x, y], index) => {
            const [ax, ay] = normals[(index + count - 1) % count];
            const [bx, by2] = normals[index];
            const mx = ax + bx;
            const my = ay + by2;
            const length = Math.hypot(mx, my) || 1;
            return [x - (mx / length) * by, y - (my / length) * by];
        });
        const insetTop = inset.map(([x, y]) => iso([x, y, z + height], projection));
        bevel = runsOf(facing, true)
            .map((run) => pathOf([top[run[0]], ...run.slice(1, -1).map((index) => insetTop[index]), top[run[run.length - 1]]]))
            .join("");
    }
    return {
        fill,
        outline: outline.join(""),
        crease: crease.join(""),
        top: pathOf(oriented(top), true),
        shades: buckets.map((paths) => paths.join("")),
        bevel,
    };
}
const ringOf = (plan, steps = 8) => roundedPlan(plan, steps);
function slabOf(plan, z, height, projection, steps = 8, bevel = 0) {
    return extrude(ringOf(plan, steps), z, height, projection, { convex: true, bevel });
}
function circleRing(cx, cy, radius, steps = 40) {
    return Array.from({ length: steps }, (_, step) => {
        const angle = (step / steps) * Math.PI * 2;
        return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
    });
}
function cylinder(cx, cy, radius, z, height, projection, steps = 32, bevel = 0) {
    return extrude(circleRing(cx, cy, radius, steps), z, height, projection, { convex: true, bevel });
}
const insetPlan = ({ x, y, w, d, r }, by) => ({ x: x + by, y: y + by, w: w - by * 2, d: d - by * 2, r: Math.max(0, r - by) });
function onTop(points, z, projection, closed = false) {
    return pathOf(points.map(([x, y]) => iso([x, y, z], projection)), closed);
}
const planOutline = (plan, z, projection, steps = 8) => onTop(ringOf(plan, steps), z, projection, true);
const lineOnTop = (from, to, z, projection) => onTop([from, to], z, projection);
const segment = (from, to, projection) => pathOf([iso(from, projection), iso(to, projection)]);
const rise = (at, from, to, projection) => segment([at[0], at[1], from], [at[0], at[1], to], projection);
const ring = (cx, cy, radius, z, projection, steps = 40) => onTop(circleRing(cx, cy, radius, steps), z, projection, true);
const pointsAt = (points, projection) => points.map((point) => iso(point, projection));
function dotGrid(x, y, columns, rows, gap, z, projection) {
    const dots = [];
    for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++)
            dots.push(iso([x + column * gap, y + row * gap, z], projection));
    }
    return dots;
}
function corners({ x, y, w, d }, inset = 0) {
    return [
        [x + inset, y + inset],
        [x + w - inset, y + inset],
        [x + inset, y + d - inset],
        [x + w - inset, y + d - inset],
    ];
}
function boxCorners({ x, y, w, d }, z0, z1) {
    const out = [];
    for (const px of [x, x + w])
        for (const py of [y, y + d])
            for (const z of [z0, z1])
                out.push([px, py, z]);
    return out;
}
function knurl(cx, cy, radius, z0, z1, count, projection, inset = 0.4) {
    const [vx, vy] = towardViewer(projection);
    const lines = [];
    for (let index = 0; index < count; index++) {
        const angle = (index / count) * Math.PI * 2;
        const nx = Math.cos(angle);
        const ny = Math.sin(angle);
        if (nx * vx + ny * vy < 0.15)
            continue;
        lines.push(segment([cx + radius * nx, cy + radius * ny, z0 + inset], [cx + radius * nx, cy + radius * ny, z1 - inset], projection));
    }
    return lines.join("");
}
function sideArc(cx, cy, radius, z, projection, steps = 24) {
    const [vx, vy] = towardViewer(projection);
    const facing = Math.atan2(vy, vx);
    const points = [];
    for (let step = 0; step <= steps; step++) {
        const angle = facing - Math.PI / 2 + (step / steps) * Math.PI;
        points.push(iso([cx + radius * Math.cos(angle), cy + radius * Math.sin(angle), z], projection));
    }
    return pathOf(points);
}
function ringSeam(ring, z, projection) {
    const count = ring.length;
    const counter = signedArea(ring) > 0;
    const [vx, vy] = towardViewer(projection);
    const facing = ring.map(([ax, ay], index) => {
        const [bx, by] = ring[(index + 1) % count];
        const dx = bx - ax;
        const dy = by - ay;
        const [nx, ny] = counter ? [dy, -dx] : [-dy, dx];
        return nx * vx + ny * vy > 1e-9;
    });
    return runsOf(facing, true).map((run) => pathOf(run.map((index) => iso([ring[index][0], ring[index][1], z], projection)))).join("");
}
const sideSeam = (plan, z, projection, steps = 8) => ringSeam(ringOf(plan, steps), z, projection);
function sideRing(center, radius, axis, projection, steps = 20) {
    const [px, py, pz] = center;
    const points = [];
    for (let step = 0; step < steps; step++) {
        const angle = (step / steps) * Math.PI * 2;
        const u = radius * Math.cos(angle);
        const v = radius * Math.sin(angle);
        points.push(iso(axis === "x" ? [px, py + u, pz + v] : [px + u, py, pz + v], projection));
    }
    return pathOf(points, true);
}
function coil(from, to, center, radius, turns, axis, projection, perTurn = 16) {
    const count = Math.max(1, Math.round(turns * perTurn));
    const back = [];
    const front = [];
    let run = [];
    let near = null;
    for (let index = 0; index <= count; index++) {
        const share = index / count;
        const angle = share * turns * Math.PI * 2;
        const u = radius * Math.cos(angle);
        const v = radius * Math.sin(angle);
        const along = from + (to - from) * share;
        const point = axis === "x" ? [along, center[0] + u, center[1] + v] : [center[0] + u, along, center[1] + v];
        const normal = axis === "x" ? [0, Math.cos(angle), Math.sin(angle)] : [Math.cos(angle), 0, Math.sin(angle)];
        const facing = depthOf(normal, projection) > 0;
        const flat = iso(point, projection);
        if (near === null)
            near = facing;
        if (facing !== near) {
            run.push(flat);
            (near ? front : back).push(pathOf(run));
            run = [flat];
            near = facing;
            continue;
        }
        run.push(flat);
    }
    if (run.length > 1)
        (near ? front : back).push(pathOf(run));
    return { back: back.join(""), front: front.join("") };
}
function topTicks(from, to, step, every, edge, z, lengths, projection, axis = "x") {
    const minor = [];
    const major = [];
    const first = Math.ceil(from / step) * step;
    for (let at = first, index = Math.round(first / step); at <= to + 1e-9; at += step, index++) {
        const big = index % every === 0;
        const length = big ? lengths[1] : lengths[0];
        const a = axis === "x" ? [at, edge] : [edge, at];
        const b = axis === "x" ? [at, edge + length] : [edge + length, at];
        (big ? major : minor).push(lineOnTop(a, b, z, projection));
    }
    return { minor: minor.join(""), major: major.join("") };
}
function sideTicks(from, to, step, every, face, top, lengths, projection, axis = "x") {
    const minor = [];
    const major = [];
    const first = Math.ceil(from / step) * step;
    for (let at = first, index = Math.round(first / step); at <= to + 1e-9; at += step, index++) {
        const big = index % every === 0;
        const length = big ? lengths[1] : lengths[0];
        const a = axis === "x" ? [at, face, top] : [face, at, top];
        const b = axis === "x" ? [at, face, top - length] : [face, at, top - length];
        (big ? major : minor).push(segment(a, b, projection));
    }
    return { minor: minor.join(""), major: major.join("") };
}
function radialTicks(cx, cy, radius, count, every, z, lengths, projection, from = 0, span = Math.PI * 2) {
    const minor = [];
    const major = [];
    const closed = Math.abs(span - Math.PI * 2) < 1e-9;
    const steps = closed ? count : count + 1;
    for (let index = 0; index < steps; index++) {
        const angle = from + (index / count) * span;
        const big = index % every === 0;
        const inner = radius - (big ? lengths[1] : lengths[0]);
        const sx = Math.sin(angle);
        const sy = -Math.cos(angle);
        (big ? major : minor).push(lineOnTop([cx + sx * inner, cy + sy * inner], [cx + sx * radius, cy + sy * radius], z, projection));
    }
    return { minor: minor.join(""), major: major.join("") };
}
const haloOf = (plan, z, height, projection) => slabOf(plan, z, height, projection).fill;
function axisVector(projection, [dx, dy, dz]) {
    const origin = iso([0, 0, 0], projection);
    const moved = iso([dx, dy, dz], projection);
    return [moved[0] - origin[0], moved[1] - origin[1]];
}
const fine = (value) => Math.round(value * 100) / 100;
function translateAlong(projection, by) {
    const [x, y] = axisVector(projection, by);
    return `translate(${fine(x)} ${fine(y)})`;
}
const thousandth = (value) => Math.round(value * 1000) / 1000;
function axesAt(at, z, projection) {
    const origin = iso([at[0], at[1], z], projection);
    const along = (point) => {
        const [px, py] = iso(point, projection);
        return [px - origin[0], py - origin[1]];
    };
    return { origin, x: along([at[0] + 1, at[1], z]), y: along([at[0], at[1] + 1, z]), z: along([at[0], at[1], z + 1]) };
}
const matrixOf = (u, v, origin) => `matrix(${thousandth(u[0])} ${thousandth(u[1])} ${thousandth(v[0])} ${thousandth(v[1])} ${thousandth(origin[0])} ${thousandth(origin[1])})`;
function holdStill(element, [dx, dy]) {
    const text = dx || dy ? `translate(${-dx} ${-dy})` : "";
    element.querySelectorAll("[data-hold]").forEach((node) => (text ? node.setAttribute("transform", text) : node.removeAttribute("transform")));
}
function topMatrix(at, z, projection, along = "x") {
    const { origin, x, y } = axesAt(at, z, projection);
    return along === "x" ? matrixOf(x, y, origin) : matrixOf([-y[0], -y[1]], x, origin);
}
function sideMatrix(at, z, projection, face) {
    const { origin, x, y, z: up } = axesAt(at, z, projection);
    const down = [-up[0], -up[1]];
    return face === "left" ? matrixOf(x, down, origin) : matrixOf([-y[0], -y[1]], down, origin);
}
function screenRise(height, projection) {
    const { cosE, k } = cameraOf(projection);
    return height * cosE * k;
}
function fitProjection(points, width, height, { pad = 24, headroom = 0, azimuth = AZIMUTH, elevation = ELEVATION } = {}) {
    const unit = { origin: [0, 0], scale: 1, azimuth, elevation };
    const flat = points.map((point) => iso(point, unit));
    const xs = flat.map(([x]) => x);
    const ys = flat.map(([, y]) => y);
    const left = Math.min(...xs);
    const right = Math.max(...xs);
    const up = Math.min(...ys);
    const down = Math.max(...ys);
    const scale = Math.min((width - pad * 2) / (right - left), (height - pad * 2 - headroom) / (down - up));
    return {
        origin: [width / 2 - ((left + right) / 2) * scale, headroom + (height - headroom) / 2 - ((up + down) / 2) * scale],
        scale,
        azimuth,
        elevation,
    };
}
const escape = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const attrs = (pairs) => Object.entries(pairs)
    .filter(([, value]) => value !== undefined && value !== false)
    .map(([key, value]) => ` ${key}="${escape(String(value))}"`)
    .join("");
function solidSvg(paths, { tone = "hi", crease = "faint", halo = false, lit = false, flat = false, className, inner = "" } = {}) {
    const parts = [];
    if (halo)
        parts.push(`<path class="iso-halo" d="${paths.fill}"/>`);
    parts.push(`<path class="iso-fill" d="${paths.fill}"/>`);
    if (!flat) {
        paths.shades.forEach((d, shade) => {
            if (d)
                parts.push(`<path class="iso-shade" data-shade="${shade}" d="${d}"/>`);
        });
        parts.push(`<path class="iso-top" d="${paths.top}"/>`);
    }
    if (crease !== "none" && paths.crease)
        parts.push(`<path class="iso-line iso-crease" data-tone="${crease}" d="${paths.crease}"/>`);
    if (crease !== "none" && paths.bevel)
        parts.push(`<path class="iso-line iso-bevel" data-tone="${crease}" d="${paths.bevel}"/>`);
    if (inner)
        parts.push(inner);
    parts.push(`<path class="iso-line iso-edge" data-tone="${tone}" d="${paths.outline}"/>`);
    return `<g${attrs({ class: className ? `iso-solid ${className}` : "iso-solid", "data-lit": lit ? "" : undefined, "data-seamless": paths.seamless ? "" : undefined })}>${parts.join("")}</g>`;
}
function lineSvg(d, { tone = "lo", dotted = false, dashed = false, flow = false, className, free } = {}) {
    if (!d)
        return "";
    return `<path${attrs({
        class: className ? `iso-line ${className}` : "iso-line",
        "data-tone": tone,
        "data-free": free === true ? "" : free || undefined,
        "data-dotted": dotted ? "" : undefined,
        "data-dashed": dashed ? "" : undefined,
        "data-flow": flow ? "" : undefined,
        d,
    })}/>`;
}
function dotsSvg(points, { size = 0.5, tone = "mid", pulse = false } = {}) {
    const dots = points
        .map(([x, y, alpha], index) => `<circle cx="${tenth(x)}" cy="${tenth(y)}" r="${size}"${alpha !== undefined && alpha < 1 ? ` fill-opacity="${Math.max(0, alpha).toFixed(2)}"` : ""}${pulse ? ` data-pulse="" style="--pulse-at:${(index * 137) % 1600}ms"` : ""}/>`)
        .join("");
    return `<g class="iso-dots" data-tone="${tone}">${dots}</g>`;
}
function fadedSvg(list, style = {}) {
    const groups = new Map();
    for (const entry of list) {
        const [d, alpha] = Array.isArray(entry) ? entry : [entry.d, entry.alpha];
        if (!d || !(alpha > 0.02))
            continue;
        const level = Math.min(1, Math.round(alpha * 20) / 20);
        groups.set(level, (groups.get(level) ?? "") + d);
    }
    return [...groups].map(([alpha, d]) => (alpha >= 1 ? lineSvg(d, style) : `<g opacity="${alpha.toFixed(2)}">${lineSvg(d, style)}</g>`)).join("");
}
function fadeLineSvg(points, { fade = [0, 0], tone = "lo", className, stops = 6 } = {}) {
    if (points.length < 2)
        return "";
    const ease = (t) => t * t * (3 - 2 * t);
    const hashOf = (text) => {
        let hash = 2166136261;
        for (let index = 0; index < text.length; index++)
            hash = Math.imul(hash ^ text.charCodeAt(index), 16777619);
        return (hash >>> 0).toString(36);
    };
    const lengths = [0];
    for (let index = 1; index < points.length; index++)
        lengths.push(lengths[index - 1] + Math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]));
    const at = (length) => {
        let index = 1;
        while (index < points.length - 1 && lengths[index] < length)
            index++;
        const span = lengths[index] - lengths[index - 1] || 1;
        const t = Math.max(0, Math.min(1, (length - lengths[index - 1]) / span));
        return [points[index - 1][0] + (points[index][0] - points[index - 1][0]) * t, points[index - 1][1] + (points[index][1] - points[index - 1][1]) * t];
    };
    const pieceOf = (from, to) => [at(from), ...points.filter((_, index) => lengths[index] > from + 1e-6 && lengths[index] < to - 1e-6), at(to)];
    const total = lengths[lengths.length - 1];
    if (total < 1e-6)
        return "";
    let [start, end] = Array.isArray(fade) ? fade : [fade, fade];
    if (start + end > total) {
        const share = total / (start + end);
        start *= share;
        end *= share;
    }
    const paint = `var(--anatomy-${tone})`;
    const ramp = (piece, rising) => {
        const d = pathOf(piece);
        const [x1, y1] = piece[0];
        const [x2, y2] = piece[piece.length - 1];
        const id = `iso-fade-${hashOf(`${d}|${tone}|${rising}`)}`;
        const marks = Array.from({ length: stops + 1 }, (_, index) => {
            const t = index / stops;
            const alpha = ease(rising ? t : 1 - t);
            return `<stop offset="${t.toFixed(3)}" stop-opacity="${alpha.toFixed(3)}" style="stop-color:${paint}"/>`;
        }).join("");
        const gradient = `<linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="${tenth(x1)}" y1="${tenth(y1)}" x2="${tenth(x2)}" y2="${tenth(y2)}">${marks}</linearGradient>`;
        return `<defs>${gradient}</defs><path${attrs({ class: className ? `iso-line ${className}` : "iso-line", "data-tone": tone, "data-free": rising ? "start" : "end", d, style: `stroke:url(#${id})` })}/>`;
    };
    const parts = [];
    if (start > 0.05)
        parts.push(ramp(pieceOf(0, start), true));
    if (total - end - start > 0.05)
        parts.push(lineSvg(pathOf(pieceOf(start, total - end)), { tone, className }));
    if (end > 0.05)
        parts.push(ramp(pieceOf(total - end, total), false));
    return `<g class="iso-fade">${parts.join("")}</g>`;
}
function wireSvg(from, to, { via = [], tone = "lo", dotted = false, dashed = false, flow = false, ends = "both", size = 1.2 } = {}) {
    const d = pathOf([from, ...via, to]);
    const end = ([x, y]) => `<circle class="iso-wire-end" cx="${tenth(x)}" cy="${tenth(y)}" r="${size}"/>`;
    return `<g class="iso-wire" data-tone="${tone}">${lineSvg(d, { tone, dotted, dashed, flow })}${ends === "both" || ends === "from" ? end(from) : ""}${ends === "both" || ends === "to" ? end(to) : ""}</g>`;
}
function faceTextSvg(transform, text, { size = 3, tone = "mid", anchor = "start" } = {}) {
    return `<text class="iso-face-text" data-tone="${tone}" transform="${transform}" font-size="${size}" text-anchor="${anchor}">${escape(text)}</text>`;
}
function groupSvg(children, extra = {}) {
    return `<g${attrs(extra)}>${Array.isArray(children) ? children.join("") : children}</g>`;
}
function figureSvg({ width, height, label, body, standalone = false, className }) {
    const style = standalone ? `<style>${ISO_CSS}</style>` : "";
    return `<svg${attrs({
        xmlns: "http://www.w3.org/2000/svg",
        class: `iso-svg${standalone ? " iso" : ""}${className ? ` ${className}` : ""}`,
        "data-theme": standalone || undefined,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": label,
    })}>${style}${Array.isArray(body) ? body.join("") : body}</svg>`;
}
function swatchSvg(mark) {
    if (mark === "raised" || mark === "flat" || mark === "lit") {
        const body = mark === "flat" ? "M2 6.5 8 3.5 14 6.5 8 9.5Z" : "M2 4.5 8 1.5 14 4.5 14 6 8 9 2 6Z";
        const crease = mark === "flat" ? "" : `<path class="iso-swatch-crease" d="M2 4.5 8 7.5 14 4.5"/>`;
        return `<svg class="iso-swatch" data-mark="${mark}" viewBox="0 0 16 10" aria-hidden="true"><path d="${body}"/>${crease}</svg>`;
    }
    return `<svg class="iso-swatch" data-mark="${mark}" viewBox="0 0 16 10" aria-hidden="true"><path d="M1.5 5h13"/></svg>`;
}
function plateHtml({ fig, title, hint, readout = "", keys = [], caption, body }) {
    const corner = (name, text, extra = "") => `<span class="iso-plate-corner" data-corner="${name}"${extra}>${escape(text)}</span>`;
    const legend = keys.length || caption
        ? `<figcaption class="iso-legend">${keys.length ? `<span class="iso-keys">${keys.map((key) => `<span class="iso-key">${swatchSvg(key.mark)}${escape(key.label)}</span>`).join("")}</span>` : ""}${caption ? `<span class="iso-caption">${escape(caption)}</span>` : ""}</figcaption>`
        : "";
    return `<figure class="iso-figure"><div class="iso-plate"><div class="iso-plate-corners" aria-hidden="true">${corner("fig", fig)}${title ? corner("title", title) : ""}${hint ? corner("hint", hint) : ""}${corner("readout", readout, " data-readout")}</div>${body}</div>${legend}</figure>`;
}
function pageHtml({ title, theme = "dark", body, script = "", width = 650 }) {
    return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escape(title)}</title><style>${ISO_CSS}${PAGE_CSS.replace("__WIDTH__", String(width))}</style></head><body class="iso" data-theme="${theme}"><main class="iso-page">${body}</main>${script ? `<script type="module">${script.replace(/<\/(script)/gi, "<\\/$1")}</script>` : ""}</body></html>`;
}
const PAGE_CSS = `
*{box-sizing:border-box}
html{color-scheme:dark}
body[data-theme="light"]{color-scheme:light}
body{margin:0;background:var(--anatomy-page);color:var(--anatomy-text);font:14px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.iso-page{width:min(__WIDTH__px,100% - 32px);margin:64px auto 96px}
`;
const ISO_CSS = `
.iso{
  --anatomy-page:#0b0b0b;--anatomy-card:#131313;--anatomy-text:#a3a3a3;--anatomy-strong:#e6e6e6;--anatomy-muted:#6e6e6e;
  --anatomy-paper:#131313;--anatomy-face:#1b1b1b;--anatomy-top:#1c1c1c;
  --anatomy-shade-0:#0f0f0f;--anatomy-shade-1:#121212;--anatomy-shade-2:#151515;--anatomy-shade-3:#181818;
  --anatomy-lit-top:#262626;--anatomy-lit-shade:#1d1d1d;
  --anatomy-hi:#d4d4d4;--anatomy-mid:#6a6a6a;--anatomy-lo:#3e3e3e;--anatomy-faint:#2a2a2a;--anatomy-lit:#ffffff;
  --anatomy-halo:#050505;--anatomy-halo-opacity:.9;--anatomy-dot:#e6e6e6;
  --anatomy-red:#ff6b6b;--anatomy-green:#6be0a0;--anatomy-blue:#6b9bff;
  --anatomy-weight:.6px;--anatomy-ease:cubic-bezier(.32,.72,0,1);
  --anatomy-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --anatomy-lift:inset 0 1px 0 0 #ffffff0a,inset 0 0 0 1px #ffffff06,0 0 0 1px #00000026,0 1px 1px -.5px #00000014,0 3px 3px -1.5px #00000014,0 6px 6px -3px #00000014,0 12px 12px -6px #00000014;
}
.iso[data-theme="light"]{
  --anatomy-page:#f2f2ef;--anatomy-card:#fbfbfa;--anatomy-text:#57575a;--anatomy-strong:#18181a;--anatomy-muted:#8c8c90;
  --anatomy-paper:#fbfbfa;--anatomy-face:#ffffff;--anatomy-top:#ffffff;
  --anatomy-shade-0:#dcdcd9;--anatomy-shade-1:#e5e5e2;--anatomy-shade-2:#ededea;--anatomy-shade-3:#f4f4f1;
  --anatomy-lit-top:#ffffff;--anatomy-lit-shade:#eeeeeb;
  --anatomy-hi:#26262a;--anatomy-mid:#8a8a8f;--anatomy-lo:#b4b4b8;--anatomy-faint:#d4d4d6;--anatomy-lit:#000000;
  --anatomy-halo:#8a8a80;--anatomy-halo-opacity:.32;--anatomy-dot:#3a3a3e;
  --anatomy-red:#e5484d;--anatomy-green:#30a46c;--anatomy-blue:#3e63dd;
  --anatomy-lift:inset 0 1px 0 0 #ffffff,0 0 0 1px #0000000f,0 1px 1px -.5px #0000000a,0 3px 3px -1.5px #0000000a,0 6px 6px -3px #0000000a,0 12px 12px -6px #0000000a;
}
.iso-svg{display:block;width:100%;height:auto;overflow:visible}
.iso-svg path,.iso-svg circle{vector-effect:non-scaling-stroke}
.iso-fill{fill:var(--anatomy-paper);stroke:none}
.iso-solid[data-lit]>.iso-fill{fill:var(--anatomy-face)}
.iso-shade,.iso-top{stroke:none}
.iso-shade[data-shade="0"]{fill:var(--anatomy-shade-0)}
.iso-shade[data-shade="1"]{fill:var(--anatomy-shade-1)}
.iso-shade[data-shade="2"]{fill:var(--anatomy-shade-2)}
.iso-shade[data-shade="3"]{fill:var(--anatomy-shade-3)}
.iso-top{fill:var(--anatomy-top)}
.iso-solid[data-lit]>.iso-top{fill:var(--anatomy-lit-top)}
.iso-solid[data-lit]>.iso-shade{fill:var(--anatomy-lit-shade)}
.iso-solid[data-seamless]>.iso-shade,.iso-solid[data-seamless]>.iso-top{stroke-width:.4;stroke-linejoin:round}
.iso-solid[data-seamless]>.iso-shade[data-shade="0"]{stroke:var(--anatomy-shade-0)}
.iso-solid[data-seamless]>.iso-shade[data-shade="1"]{stroke:var(--anatomy-shade-1)}
.iso-solid[data-seamless]>.iso-shade[data-shade="2"]{stroke:var(--anatomy-shade-2)}
.iso-solid[data-seamless]>.iso-shade[data-shade="3"]{stroke:var(--anatomy-shade-3)}
.iso-solid[data-seamless]>.iso-top{stroke:var(--anatomy-top)}
.iso-solid[data-seamless][data-lit]>.iso-top{stroke:var(--anatomy-lit-top)}
.iso-solid[data-seamless][data-lit]>.iso-shade{stroke:var(--anatomy-lit-shade)}
.iso-solid[data-lit]>.iso-edge{stroke:var(--anatomy-lit)}
.iso-halo{fill:var(--anatomy-halo);stroke:none;filter:blur(9px);opacity:var(--anatomy-halo-opacity);transform:translateY(3px)}
.iso-line{fill:none;stroke:var(--anatomy-lo);stroke-width:var(--anatomy-weight);stroke-linecap:round;stroke-linejoin:round}
.iso-line[data-tone="hi"]{stroke:var(--anatomy-hi)}
.iso-line[data-tone="mid"]{stroke:var(--anatomy-mid)}
.iso-line[data-tone="lo"]{stroke:var(--anatomy-lo)}
.iso-line[data-tone="faint"]{stroke:var(--anatomy-faint)}
.iso-line[data-tone="lit"]{stroke:var(--anatomy-lit)}
.iso-line[data-tone="red"]{stroke:var(--anatomy-red)}
.iso-line[data-tone="green"]{stroke:var(--anatomy-green)}
.iso-line[data-tone="blue"]{stroke:var(--anatomy-blue)}
.iso-line[data-dotted]{stroke-dasharray:.01 3.5;stroke-width:1}
.iso-line[data-dashed]{stroke-dasharray:3 3}
.iso-line[data-flow]{stroke-dasharray:2 6}
.iso-wire-end{fill:var(--anatomy-paper);stroke:var(--anatomy-lo);stroke-width:var(--anatomy-weight)}
.iso-wire[data-tone="hi"] .iso-wire-end{stroke:var(--anatomy-hi)}
.iso-wire[data-tone="mid"] .iso-wire-end{stroke:var(--anatomy-mid)}
.iso-wire[data-tone="faint"] .iso-wire-end{stroke:var(--anatomy-faint)}
.iso-wire[data-tone="lit"] .iso-wire-end{stroke:var(--anatomy-lit)}
.iso-dots circle{fill:var(--anatomy-dot);stroke:none;opacity:.32}
.iso-dots[data-tone="hi"] circle{opacity:.85}
.iso-dots[data-tone="mid"] circle{opacity:.45}
.iso-dots[data-tone="lo"] circle{opacity:.22}
.iso-dots[data-tone="faint"] circle{opacity:.12}
.iso-face-text{fill:var(--anatomy-mid);font-family:var(--anatomy-mono);letter-spacing:.04em}
.iso-face-text[data-tone="hi"]{fill:var(--anatomy-hi)}
.iso-face-text[data-tone="lo"]{fill:var(--anatomy-lo)}
.iso-face-text[data-tone="faint"]{fill:var(--anatomy-faint)}
.tb-body{fill:var(--anatomy-shade-2);stroke:none}
.tb-shine{fill:var(--anatomy-top);stroke:none}
.tb-shade{fill:var(--anatomy-shade-0);stroke:none}
.tb-ring{fill:none;stroke:var(--anatomy-lo);stroke-width:.45;stroke-linecap:butt;stroke-linejoin:round}
.tb-ring[data-tone="mid"]{stroke:var(--anatomy-mid)}
.tb-ring[data-tone="faint"]{stroke:var(--anatomy-faint)}
.tb-ring[data-tone="hi"]{stroke:var(--anatomy-hi)}
@keyframes iso-flow{to{stroke-dashoffset:-32}}
@keyframes iso-pulse{0%,100%{opacity:.18}12%{opacity:1}40%{opacity:.18}}
.iso-line[data-flow]{animation:iso-flow 1600ms linear infinite}
.iso-dots circle[data-pulse]{animation:iso-pulse 1600ms steps(1,end) infinite;animation-delay:var(--pulse-at,0ms)}
.iso-figure{margin:32px 0 36px}
.iso-plate{position:relative;padding:52px 24px;border-radius:0;background:var(--anatomy-card);box-shadow:0 0 0 1px var(--anatomy-mid);--anatomy-paper:var(--anatomy-card)}
.iso-plate-corners{position:absolute;inset:18px 20px;pointer-events:none}
.iso-plate-corner{position:absolute;color:var(--anatomy-muted);font-family:var(--anatomy-mono);font-size:10.5px;line-height:1;letter-spacing:.04em;white-space:nowrap}
.iso-plate-corner[data-corner="fig"]{top:0;left:0;color:var(--anatomy-text)}
.iso-plate-corner[data-corner="title"]{top:0;right:0;text-transform:uppercase;letter-spacing:.1em}
.iso-plate-corner[data-corner="hint"]{bottom:0;left:0;text-transform:uppercase;letter-spacing:.1em}
.iso-plate-corner[data-corner="readout"]{right:0;bottom:0;color:var(--anatomy-text);font-variant-numeric:tabular-nums}
.iso-plate .iso-svg{margin:0 auto}
.iso-legend{display:grid;gap:8px;margin-top:12px;color:var(--anatomy-muted);font-size:13px;line-height:1.55}
.iso-keys{display:flex;flex-wrap:wrap;gap:6px 18px}
.iso-key{display:inline-flex;align-items:center;gap:7px}
.iso-swatch{width:16px;height:10px;flex:none;overflow:visible}
.iso-swatch path{fill:none;stroke:var(--anatomy-hi);stroke-width:1;vector-effect:non-scaling-stroke;stroke-linejoin:round;stroke-linecap:round}
.iso-swatch[data-mark="raised"] path,.iso-swatch[data-mark="flat"] path{stroke:var(--anatomy-mid)}
.iso-swatch[data-mark="lit"] path{stroke:var(--anatomy-lit)}
.iso-swatch[data-mark="dim"] path{stroke:var(--anatomy-lo)}
.iso-swatch[data-mark="dotted"] path{stroke-dasharray:.01 3;stroke-width:1.4}
.iso-swatch[data-mark="dashed"] path{stroke-dasharray:3 2.5}
.iso-swatch[data-mark="red"] path{stroke:var(--anatomy-red)}
.iso-swatch[data-mark="green"] path{stroke:var(--anatomy-green)}
.iso-swatch[data-mark="blue"] path{stroke:var(--anatomy-blue)}
.iso-swatch-crease{stroke:var(--anatomy-lo)!important}
@media (max-width:560px){.iso-plate{padding:48px 12px 70px}.iso-plate-corner[data-corner="hint"]{bottom:18px}}
@media (prefers-reduced-motion:reduce){.iso-line[data-flow],.iso-dots circle[data-pulse]{animation:none}}
`;


  var IsoKit = {
    AZIMUTH: AZIMUTH,
    ELEVATION: ELEVATION,
    SHADES: SHADES,
    cameraOf: cameraOf,
    iso: iso,
    towardViewer: towardViewer,
    depthOf: depthOf,
    pathOf: pathOf,
    joinPaths: joinPaths,
    roundedPlan: roundedPlan,
    extrude: extrude,
    ringOf: ringOf,
    slabOf: slabOf,
    circleRing: circleRing,
    cylinder: cylinder,
    insetPlan: insetPlan,
    onTop: onTop,
    planOutline: planOutline,
    lineOnTop: lineOnTop,
    segment: segment,
    rise: rise,
    ring: ring,
    pointsAt: pointsAt,
    dotGrid: dotGrid,
    corners: corners,
    boxCorners: boxCorners,
    knurl: knurl,
    sideArc: sideArc,
    ringSeam: ringSeam,
    sideSeam: sideSeam,
    sideRing: sideRing,
    coil: coil,
    topTicks: topTicks,
    sideTicks: sideTicks,
    radialTicks: radialTicks,
    haloOf: haloOf,
    axisVector: axisVector,
    translateAlong: translateAlong,
    holdStill: holdStill,
    topMatrix: topMatrix,
    sideMatrix: sideMatrix,
    screenRise: screenRise,
    fitProjection: fitProjection,
    solidSvg: solidSvg,
    lineSvg: lineSvg,
    dotsSvg: dotsSvg,
    fadedSvg: fadedSvg,
    fadeLineSvg: fadeLineSvg,
    wireSvg: wireSvg,
    faceTextSvg: faceTextSvg,
    groupSvg: groupSvg,
    figureSvg: figureSvg,
    plateHtml: plateHtml,
    pageHtml: pageHtml,
    ISO_CSS: ISO_CSS,
    ensureStyles: function (doc) {
      var d = doc || (typeof document !== "undefined" ? document : null);
      if (!d || !d.head) return;
      if (d.querySelector("style[data-mcp-iso-css]")) return;
      var style = d.createElement("style");
      style.setAttribute("data-mcp-iso-css", "1");
      style.textContent = ISO_CSS;
      d.head.appendChild(style);
    }
  };

  global.IsoKit = IsoKit;
  if (typeof window !== "undefined") {
    window.IsoKit = IsoKit;
  }

  var _sceneRenderCallback = null;

  function registerSceneRender(fn) {
    if (typeof fn === "function") {
      _sceneRenderCallback = fn;
      global.__sceneRender = fn;
      if (typeof window !== "undefined") {
        window.__sceneRender = fn;
      }
    }
    return fn;
  }

  var _inSetFrameTime = false;
  var _inBridgeSeek = false;

  function createBridgeTimeline() {
    return {
      _isMcpBridge: true,
      _timeSec: 0,
      seek: function (sec) {
        this._timeSec = Math.max(0, Number(sec) || 0);
        if (!_inSetFrameTime && typeof setFrameTime === "function") {
          _inBridgeSeek = true;
          try {
            setFrameTime(this._timeSec * 1000);
          } finally {
            _inBridgeSeek = false;
          }
        }
        return this;
      },
      time: function (sec) {
        if (sec === undefined) {
          return this._timeSec;
        }
        return this.seek(sec);
      },
      duration: function () {
        return (Number(global.__sceneDurationMs) || 10000) / 1000;
      },
      pause: function () {
        return this;
      },
      play: function () {
        return this;
      },
    };
  }

  global.__timelines = global.__timelines || {};
  if (!global.__timelines["main"]) {
    global.__timelines["main"] = createBridgeTimeline();
  }

  function setFrameTime(timeMs) {
    var numeric = Number(timeMs);
    if (!isFinite(numeric)) {
      numeric = 0;
    }
    numeric = Math.max(0, numeric);

    if (_inSetFrameTime) {
      return numeric;
    }

    _inSetFrameTime = true;
    try {
      virtualNow = numeric;
      global.__currentTimeMs = numeric;
      if (typeof window !== "undefined") {
        window.__currentTimeMs = numeric;
      }

      var timeSec = numeric / 1000;
      if (global.__timelines && global.__timelines["main"]) {
        var mainTimeline = global.__timelines["main"];
        if (mainTimeline._isMcpBridge) {
          mainTimeline._timeSec = timeSec;
        } else if (!_inBridgeSeek) {
          if (typeof mainTimeline.seek === "function") {
            mainTimeline.seek(timeSec);
          } else if (typeof mainTimeline.time === "function") {
            mainTimeline.time(timeSec);
          }
        }
      }

      var documentObject = global.document;
      if (documentObject && documentObject.documentElement) {
        var root = documentObject.documentElement;
        root.style.setProperty("--time-ms", String(numeric));
        root.setAttribute("data-frame-time-ms", String(numeric));
        if (documentObject.body) {
          documentObject.body.style.setProperty("--time-ms", String(numeric));
        }
      }

      if (global.IsoKit && typeof global.IsoKit.ensureStyles === "function" && documentObject) {
        if (
          documentObject.querySelector &&
          documentObject.querySelector(".iso, .iso-plate, .iso-svg")
        ) {
          global.IsoKit.ensureStyles(documentObject);
        }
      }

      syncAnimations(numeric);

      if (typeof global.onFrameTime === "function") {
        global.onFrameTime(numeric, MotionKit);
      }

      if (typeof _sceneRenderCallback === "function") {
        _sceneRenderCallback(numeric, MotionKit);
      } else if (typeof global.__sceneRender === "function") {
        global.__sceneRender(numeric, MotionKit);
      }

      if (documentObject && typeof global.CustomEvent === "function") {
        documentObject.dispatchEvent(
          new global.CustomEvent("frametime", { detail: { timeMs: numeric } }),
        );
      }

      return numeric;
    } finally {
      _inSetFrameTime = false;
    }
  }

  global.__currentTimeMs = 0;
  global.MotionKit = MotionKit;
  global.IsoKit = IsoKit;
  global.__setFrameTime = setFrameTime;
  global.__registerSceneRender = registerSceneRender;
  global.__sceneRender = global.__sceneRender || null;
  global.__mcpVideosSyncAnimations = syncAnimations;
  global.__MCP_VIDEOS_HARNESS__ = {
    version: "1.2.0",
    setFrameTime: setFrameTime,
    registerSceneRender: registerSceneRender,
    syncAnimations: syncAnimations,
    MotionKit: MotionKit,
    timelines: global.__timelines,
    IsoKit: IsoKit,
  };

  if (typeof window !== "undefined") {
    window.__currentTimeMs = 0;
    window.MotionKit = MotionKit;
    window.IsoKit = IsoKit;
    window.__setFrameTime = setFrameTime;
    window.__registerSceneRender = registerSceneRender;
    window.__timelines = global.__timelines;
    window.__sceneRender = global.__sceneRender;
  }

  if (global.document && global.document.documentElement) {
    global.document.documentElement.style.setProperty("--time-ms", "0");
    global.document.documentElement.setAttribute("data-frame-time-ms", "0");
  }
})(typeof window !== "undefined" ? window : globalThis);
