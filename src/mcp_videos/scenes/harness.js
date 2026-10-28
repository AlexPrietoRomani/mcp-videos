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

  var MotionKit = {
    clamp: clamp,
    lerp: lerp,
    easeOutCubic: easeOutCubic,
    easeInOutQuad: easeInOutQuad,
    spring: spring,
    progress: progress,
    progressInRange: progress,
    activeWordAt: activeWordAt,
    activeWordIndexAt: activeWordIndexAt,
  };

  function setFrameTime(timeMs) {
    var numeric = Number(timeMs);
    if (!isFinite(numeric)) {
      numeric = 0;
    }
    numeric = Math.max(0, numeric);
    virtualNow = numeric;
    global.__currentTimeMs = numeric;

    var documentObject = global.document;
    if (documentObject && documentObject.documentElement) {
      var root = documentObject.documentElement;
      root.style.setProperty("--time-ms", String(numeric));
      root.setAttribute("data-frame-time-ms", String(numeric));
      if (documentObject.body) {
        documentObject.body.style.setProperty("--time-ms", String(numeric));
      }
    }

    syncAnimations(numeric);

    if (typeof global.onFrameTime === "function") {
      global.onFrameTime(numeric, MotionKit);
    }

    if (documentObject && typeof global.CustomEvent === "function") {
      documentObject.dispatchEvent(
        new global.CustomEvent("frametime", { detail: { timeMs: numeric } }),
      );
    }

    return numeric;
  }

  global.__currentTimeMs = 0;
  global.MotionKit = MotionKit;
  global.__setFrameTime = setFrameTime;
  global.__mcpVideosSyncAnimations = syncAnimations;
  global.__MCP_VIDEOS_HARNESS__ = {
    version: "1.0.0",
    setFrameTime: setFrameTime,
    syncAnimations: syncAnimations,
    MotionKit: MotionKit,
  };

  if (typeof window !== "undefined") {
    window.__currentTimeMs = 0;
    window.MotionKit = MotionKit;
    window.__setFrameTime = setFrameTime;
  }

  if (global.document && global.document.documentElement) {
    global.document.documentElement.style.setProperty("--time-ms", "0");
    global.document.documentElement.setAttribute("data-frame-time-ms", "0");
  }
})(typeof window !== "undefined" ? window : globalThis);
