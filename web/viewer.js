(() => {
  let canvas = document.getElementById("simCanvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "simCanvas";
    const shell = document.querySelector(".viewer-shell") || document.body;
    shell.prepend(canvas);
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const els = {
    playPauseBtn: document.getElementById("playPauseBtn"),
    prevGenBtn: document.getElementById("prevGenBtn"),
    nextGenBtn: document.getElementById("nextGenBtn"),
    generationSlider: document.getElementById("generationSlider"),
    speedSlider: document.getElementById("speedSlider"),
    autoplayToggle: document.getElementById("autoplayToggle"),
    trailToggle: document.getElementById("trailToggle"),
    debugToggle: document.getElementById("debugToggle"),
    showBrainToggle: document.getElementById("showBrainToggle"),
    showManyBirdsToggle: document.getElementById("showManyBirdsToggle"),
    status: document.getElementById("status"),
    modeBadge: document.getElementById("modeBadge"),
    errorMessage: document.getElementById("errorMessage"),
    rankingList: document.getElementById("rankingList"),
    rankSelector: document.getElementById("rankSelector"),
    generationDebugLine: document.getElementById("generationDebugLine"),
    brainPanel: document.getElementById("brainPanel"),
    brainInputs: document.getElementById("brainInputs"),
    brainOutput: document.getElementById("brainOutput"),
    brainDecision: document.getElementById("brainDecision"),
    brainFlapState: document.getElementById("brainFlapState"),
    brainCanvas: document.getElementById("brainCanvas"),
    statGeneration: document.getElementById("statGeneration"),
    statBirdsShown: document.getElementById("statBirdsShown"),
    statAlive: document.getElementById("statAlive"),
    statBestGen: document.getElementById("statBestGen"),
    statBestAll: document.getElementById("statBestAll"),
    statPlayback: document.getElementById("statPlayback"),
  };
  const brainCtx = els.brainCanvas?.getContext("2d");

  const state = {
    replay: null,
    generations: [],
    traces: [],
    tracesFromGenomes: false,
    generationMessage: "",
    generationIndex: 0,
    playT: 0,
    playing: true,
    autoplayEnabled: true,
    simSpeedMultiplier: 1.5,
    showTrails: false,
    showDebug: false,
    showBrain: false,
    showManyBirds: true,
    selectedRank: 1,
    stepAccumulator: 0,
    lastTimestamp: 0,
    birds: [],
    renderEntries: [],
    pipes: [],
    pipeFrames: [],
    pipeFramesLen: 0,
    tracesMaxLen: 0,
    championFramesLen: 0,
    aliveCountAll: 0,
    trailHistory: [],
    logicalWidth: 500,
    logicalHeight: 800,
    dpr: 1,
    skyGradient: null,
    vignetteGradient: null,
    groundGradient: null,
    clouds: [],
  };

  const STORAGE_KEYS = {
    showManyBirds: "showManyBirds",
  };

  const setStatus = (text) => { if (els.status) els.status.textContent = text; };
  const setError = (text) => { if (els.errorMessage) els.errorMessage.textContent = text; };
  if (els.modeBadge) els.modeBadge.textContent = "TRAINING REPLAY";

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function sizeCanvas() {
    const cfg = state.replay?.config || {};
    state.logicalWidth = Number(cfg.world_width || state.logicalWidth || 500);
    state.logicalHeight = Number(cfg.world_height || state.logicalHeight || 800);
    state.dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    canvas.width = Math.round(state.logicalWidth * state.dpr);
    canvas.height = Math.round(state.logicalHeight * state.dpr);
    canvas.style.aspectRatio = `${state.logicalWidth} / ${state.logicalHeight}`;
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    state.skyGradient = null;
    state.vignetteGradient = null;
    state.groundGradient = null;
    initializeClouds();
  }

  function initializeClouds() {
    const cloudCount = Math.max(3, Math.min(7, Math.round(state.logicalWidth / 140)));
    state.clouds = new Array(cloudCount);
    for (let i = 0; i < cloudCount; i += 1) {
      const scale = 0.7 + ((i % 4) * 0.16);
      state.clouds[i] = {
        baseX: (i / cloudCount) * state.logicalWidth,
        y: 80 + ((i * 71) % Math.max(140, state.logicalHeight * 0.35)),
        width: 58 * scale,
        height: 24 * scale,
        speed: 0.14 + (i % 5) * 0.035,
        alpha: 0.11 + (i % 4) * 0.03,
      };
    }
  }

  function normalizeReplayData(raw) {
    const generations = Array.isArray(raw?.generations) ? raw.generations : [];
    return {
      ...raw,
      generations: generations.map((gen, idx) => normalizeGeneration(gen, idx)),
    };
  }

  function normalizeGeneration(generation, idx) {
    const genomes = Array.isArray(generation?.genomes)
      ? generation.genomes
      : (Array.isArray(generation?.birds)
        ? generation.birds
        : []);

    return {
      ...generation,
      generation: generation?.generation ?? idx,
      traces: Array.isArray(generation?.traces) ? generation.traces : [],
      genomes: genomes.map((g, gIdx) => ({
        ...g,
        rank: Number(g?.rank ?? (gIdx + 1)),
        frames: Array.isArray(g?.frames)
          ? g.frames
          : (Array.isArray(g?.trace) ? g.trace : []),
      })),
    };
  }

  function getGeneration() {
    return state.generations?.[state.generationIndex] || null;
  }

  function buildTraces(generation) {
    const oldTraces = Array.isArray(generation?.traces) ? generation.traces : [];
    if (oldTraces.length > 0) {
      state.tracesFromGenomes = false;
      return oldTraces.map((trace, idx) => ({
        rank: Number(trace?.rank ?? (idx + 1)),
        fitness: Number(trace?.fitness ?? 0),
        pipes_passed: Number(trace?.pipes_passed ?? 0),
        steps: Number(trace?.steps ?? 0),
        frames: Array.isArray(trace?.frames)
          ? trace.frames
          : (Array.isArray(trace?.trace) ? trace.trace : []),
      })).sort((a, b) => Number(a.rank || 999) - Number(b.rank || 999));
    }

    state.tracesFromGenomes = true;
    const genomes = Array.isArray(generation?.genomes) ? generation.genomes : [];
    return genomes.map((genome, idx) => ({
      rank: Number(genome?.rank ?? (idx + 1)),
      fitness: Number(genome?.fitness ?? 0),
      pipes_passed: Number(genome?.pipes_passed ?? 0),
      steps: Number(genome?.steps ?? 0),
      frames: Array.isArray(genome?.frames) ? genome.frames : [],
    })).sort((a, b) => Number(a.rank || 999) - Number(b.rank || 999));
  }

  function getTraceByRank(rank) {
    return state.traces.find((t) => Number(t.rank) === Number(rank)) || state.traces[0] || null;
  }

  function getLongestFrameCount() {
    return Math.max(0, ...state.traces.map((t) => (t.frames || []).length));
  }

  function parsePipeFrame(rawPipes, cfg, worldHeight) {
    if (!Array.isArray(rawPipes)) return [];
    return rawPipes.map((pipe) => {
      const gapH = Number(pipe.gap_h || cfg.pipe_gap || 170);
      const halfGap = gapH / 2;
      const gapY = Number(pipe.gap_y || worldHeight / 2);
      return {
        x: Number(pipe.x || 0),
        width: Number(pipe.width || cfg.pipe_width || 70),
        top: gapY - halfGap,
        bottom: gapY + halfGap,
      };
    });
  }

  function buildPipeTimeline() {
    const cfg = state.replay?.config || {};
    const worldHeight = Number(cfg.world_height || state.logicalHeight);
    const champion = getTraceByRank(1) || state.traces[0] || null;
    const frames = champion?.frames || [];
    const timeline = new Array(frames.length);
    let lastKnown = [];
    for (let t = 0; t < frames.length; t += 1) {
      const framePipes = frames[t]?.pipes;
      if (Array.isArray(framePipes)) {
        lastKnown = parsePipeFrame(framePipes, cfg, worldHeight);
      }
      timeline[t] = lastKnown;
    }
    state.pipeFrames = timeline;
    state.pipeFramesLen = timeline.length;
    state.championFramesLen = frames.length;
    state.tracesMaxLen = getLongestFrameCount();
  }

  function getAliveCountAt(playT) {
    let alive = 0;
    for (const trace of state.traces) {
      const frames = trace.frames || [];
      if (playT >= frames.length) continue;
      if (Number(frames[playT]?.alive ?? 0) !== 0) alive += 1;
    }
    return alive;
  }

  function refreshRankSelector() {
    if (!els.rankSelector) return;
    els.rankSelector.innerHTML = "";
    for (const trace of state.traces) {
      const option = document.createElement("option");
      option.value = String(trace.rank);
      option.textContent = `Rank ${trace.rank}`;
      els.rankSelector.appendChild(option);
    }
    const desired = String(state.selectedRank || 1);
    els.rankSelector.value = [...els.rankSelector.options].some((o) => o.value === desired)
      ? desired
      : (els.rankSelector.options[0]?.value || "1");
    state.selectedRank = Number(els.rankSelector.value || 1);
  }

  function applyFrame(playT) {
    const generation = getGeneration();
    if (!generation) return;

    const cfg = state.replay?.config || {};
    const worldHeight = Number(cfg.world_height || state.logicalHeight);
    const worldWidth = Number(cfg.world_width || state.logicalWidth);
    const shown = state.showManyBirds ? state.traces : [getTraceByRank(state.selectedRank)].filter(Boolean);

    const generationLabel = Number(generation?.generation ?? state.generationIndex);
    if (state.traces.length === 0) {
      state.generationMessage = `No frames/traces found in generation ${generationLabel}. Expected generations[g].genomes[].frames[]`;
    } else {
      state.generationMessage = "";
    }

    state.playT = Math.max(0, Math.trunc(playT || 0));
    state.renderEntries = shown.map((trace, idx) => {
      const frames = trace.frames || [];
      if (state.playT >= frames.length) return null;
      const frame = frames[state.playT];
      if (!frame) return null;
      const alive = Boolean(frame.alive ?? 0);
      const x = clamp(Number(frame.x ?? cfg.bird_x ?? 80), 0, worldWidth);
      const y = clamp(Number(frame.y ?? worldHeight / 2), 0, worldHeight);

      if (state.showTrails) {
        const trail = state.trailHistory[idx] || (state.trailHistory[idx] = []);
        if (alive) {
          trail.push({ x, y });
          if (trail.length > 120) trail.shift();
        }
      }

      return {
        rank: Number(trace.rank || 1),
        x,
        y,
        velocity: Number(frame.vy ?? 0),
        alive,
        flap: Boolean(frame.flap ?? 0),
        out: frame.out,
        pipesPassed: Number(frame.pipes_passed ?? 0),
        color: `hsla(${Math.round((idx * 360) / Math.max(1, shown.length))}, 85%, 52%, 0.72)`,
      };
    }).filter(Boolean);

    state.birds = state.renderEntries.filter((bird) => bird.alive);
    state.aliveCountAll = getAliveCountAt(state.playT);

    const pipeIdx = state.pipeFramesLen > 0 ? Math.min(state.playT, state.pipeFramesLen - 1) : 0;
    state.pipes = state.pipeFrames[pipeIdx] || [];

    if (!state.showTrails) {
      state.trailHistory.length = 0;
    }
  }

  function loadGeneration(index) {
    const total = state.generations?.length || 0;
    if (!total) return;
    state.generationIndex = clamp(index, 0, total - 1);
    state.playT = 0;
    state.traces = buildTraces(getGeneration());
    buildPipeTimeline();
    if (els.generationSlider) els.generationSlider.value = String(state.generationIndex);
    refreshRankSelector();
    applyFrame(0);
  }

  function drawBackground() {
    const groundHeight = Math.max(44, state.logicalHeight * 0.085);
    const horizonY = state.logicalHeight - groundHeight;

    if (!state.skyGradient) {
      const sky = ctx.createLinearGradient(0, 0, 0, horizonY);
      sky.addColorStop(0, "#7ecbff");
      sky.addColorStop(0.62, "#b6e7ff");
      sky.addColorStop(1, "#e6f8ff");
      state.skyGradient = sky;
    }
    if (!state.groundGradient) {
      const ground = ctx.createLinearGradient(0, horizonY, 0, state.logicalHeight);
      ground.addColorStop(0, "#d2be6e");
      ground.addColorStop(0.35, "#bca35b");
      ground.addColorStop(1, "#977f42");
      state.groundGradient = ground;
    }
    if (!state.vignetteGradient) {
      const v = ctx.createRadialGradient(
        state.logicalWidth * 0.5,
        state.logicalHeight * 0.35,
        Math.min(state.logicalWidth, state.logicalHeight) * 0.2,
        state.logicalWidth * 0.5,
        state.logicalHeight * 0.5,
        Math.max(state.logicalWidth, state.logicalHeight) * 0.9,
      );
      v.addColorStop(0, "rgba(5, 18, 35, 0)");
      v.addColorStop(1, "rgba(5, 18, 35, 0.24)");
      state.vignetteGradient = v;
    }
    ctx.fillStyle = state.skyGradient;
    ctx.fillRect(0, 0, state.logicalWidth, horizonY);

    const cloudScroll = state.playT;
    for (const cloud of state.clouds) {
      const cloudCycleWidth = state.logicalWidth + cloud.width * 2;
      const x = (cloud.baseX - cloudScroll * cloud.speed) % cloudCycleWidth;
      const drawX = x < -cloud.width * 1.3 ? x + cloudCycleWidth : x;
      ctx.fillStyle = `rgba(255,255,255,${cloud.alpha})`;
      ctx.beginPath();
      ctx.ellipse(drawX, cloud.y, cloud.width * 0.42, cloud.height * 0.52, 0, 0, Math.PI * 2);
      ctx.ellipse(drawX + cloud.width * 0.23, cloud.y - cloud.height * 0.2, cloud.width * 0.34, cloud.height * 0.48, 0, 0, Math.PI * 2);
      ctx.ellipse(drawX - cloud.width * 0.2, cloud.y - cloud.height * 0.15, cloud.width * 0.3, cloud.height * 0.44, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = state.groundGradient;
    ctx.fillRect(0, horizonY, state.logicalWidth, groundHeight);
    ctx.fillStyle = "rgba(255,255,255,0.16)";
    ctx.fillRect(0, horizonY, state.logicalWidth, 4);
    ctx.fillStyle = "rgba(74, 63, 28, 0.22)";
    ctx.fillRect(0, state.logicalHeight - 3, state.logicalWidth, 3);

    ctx.fillStyle = state.vignetteGradient;
    ctx.fillRect(0, 0, state.logicalWidth, state.logicalHeight);
  }

  function drawPipe(pipe) {
    const lipHeight = Math.max(6, pipe.width * 0.14);
    const border = Math.max(2, pipe.width * 0.08);
    const pipeFill = "#57bf48";
    const pipeBorder = "#2f812a";

    ctx.save();
    ctx.shadowColor = "rgba(12, 28, 10, 0.25)";
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 2;
    ctx.fillStyle = pipeFill;
    ctx.fillRect(pipe.x, 0, pipe.width, pipe.top);
    ctx.fillRect(pipe.x, pipe.bottom, pipe.width, state.logicalHeight - pipe.bottom);
    ctx.strokeStyle = pipeBorder;
    ctx.lineWidth = border;
    ctx.strokeRect(pipe.x + border * 0.5, 0, pipe.width - border, pipe.top);
    ctx.strokeRect(pipe.x + border * 0.5, pipe.bottom, pipe.width - border, state.logicalHeight - pipe.bottom);
    ctx.restore();

    ctx.fillStyle = "#71cf63";
    ctx.fillRect(pipe.x - 2, pipe.top - lipHeight, pipe.width + 4, lipHeight);
    ctx.fillRect(pipe.x - 2, pipe.bottom, pipe.width + 4, lipHeight);
    ctx.strokeStyle = pipeBorder;
    ctx.lineWidth = Math.max(1.5, border * 0.8);
    ctx.strokeRect(pipe.x - 2, pipe.top - lipHeight, pipe.width + 4, lipHeight);
    ctx.strokeRect(pipe.x - 2, pipe.bottom, pipe.width + 4, lipHeight);
  }

  function drawBird(bird) {
    const bodyW = 25;
    const bodyH = 17;
    const angle = clamp(bird.velocity * 0.055, -0.65, 0.75);
    const flapWave = bird.flap ? Math.sin((state.playT + bird.rank * 1.7) * 0.5) * 2.3 : -1.2;
    ctx.save();
    ctx.translate(bird.x, bird.y);
    ctx.rotate(angle);
    ctx.shadowColor = "rgba(12, 20, 33, 0.26)";
    ctx.shadowBlur = 6;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 2;

    ctx.beginPath();
    ctx.moveTo(-bodyW * 0.48, -bodyH * 0.5);
    ctx.lineTo(bodyW * 0.58, 0);
    ctx.lineTo(-bodyW * 0.48, bodyH * 0.5);
    ctx.closePath();
    ctx.fillStyle = bird.color;
    ctx.fill();
    ctx.strokeStyle = "rgba(10, 22, 38, 0.32)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.24)";
    ctx.beginPath();
    ctx.moveTo(-bodyW * 0.24, -bodyH * 0.3);
    ctx.lineTo(bodyW * 0.1, -bodyH * 0.07);
    ctx.lineTo(-bodyW * 0.24, bodyH * 0.02);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "rgba(8, 17, 30, 0.42)";
    ctx.beginPath();
    ctx.moveTo(-bodyW * 0.1, 1);
    ctx.lineTo(-bodyW * 0.28, flapWave + 5.2);
    ctx.lineTo(bodyW * 0.08, flapWave + 2.4);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "#ffb347";
    ctx.beginPath();
    ctx.moveTo(bodyW * 0.53, -2);
    ctx.lineTo(bodyW * 0.78, 0.8);
    ctx.lineTo(bodyW * 0.53, 3.4);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "#0f172a";
    ctx.beginPath();
    ctx.arc(1.2, -2.7, 1.6, 0, Math.PI * 2);
    ctx.fill();

    if (bird.rank === 1) {
      ctx.strokeStyle = "rgba(255, 245, 157, 0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(-bodyW * 0.6, -bodyH * 0.58);
      ctx.lineTo(bodyW * 0.7, 0);
      ctx.lineTo(-bodyW * 0.6, bodyH * 0.58);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawRoundedPanel(x, y, w, h, r) {
    const radius = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  }

  function updateStats() {
    const generation = getGeneration();
    const total = state.generations.length;
    const genNum = Number(generation?.generation ?? state.generationIndex);
    const birdsShown = state.birds.length;
    const aliveCount = state.birds.filter((b) => b.alive).length;
    const bestGen = Math.max(0, ...(state.traces.map((trace) => {
      const endFrame = (trace.frames || [])[Math.max(0, (trace.frames || []).length - 1)] || {};
      return Number(endFrame.pipes_passed ?? trace.pipes_passed ?? 0);
    })));
    const champion = getTraceByRank(1);
    const championFrame = (champion?.frames || [])[Math.min(state.playT, Math.max(0, (champion?.frames || []).length - 1))] || {};

    if (els.statGeneration) els.statGeneration.textContent = `${genNum} (idx ${state.generationIndex}, total ${total})`;
    if (els.statBirdsShown) els.statBirdsShown.textContent = String(birdsShown);
    if (els.statAlive) els.statAlive.textContent = String(aliveCount);
    if (els.statBestGen) els.statBestGen.textContent = String(bestGen);
    if (els.statBestAll) els.statBestAll.textContent = String(Number(championFrame.pipes_passed ?? champion?.pipes_passed ?? 0));
    if (els.statPlayback) els.statPlayback.textContent = state.autoplayEnabled ? "ON (sequential)" : "OFF";

    if (els.generationDebugLine) {
      if (state.showDebug) {
        els.generationDebugLine.textContent = [
          `genIdx=${state.generationIndex}`,
          `genNum=${genNum}`,
          `total=${total}`,
          `tracesCount=${state.traces.length}`,
          `playT=${state.playT}`,
          `pipeFramesLen=${state.pipeFramesLen}`,
          `championFramesLen=${state.championFramesLen}`,
          `tracesMaxLen=${state.tracesMaxLen}`,
          `mapping=${state.tracesFromGenomes ? "genomes->traces" : "native-traces"}`,
        ].join(" | ");
      } else {
        els.generationDebugLine.textContent = `genIdx=${state.generationIndex} | genNum=${genNum} | total=${total}`;
      }
    }

    if (els.rankingList) {
      const top = state.traces.slice(0, 5).map((trace) => {
        const endFrame = (trace.frames || [])[Math.max(0, (trace.frames || []).length - 1)] || {};
        const pipesPassed = Number(endFrame.pipes_passed ?? trace.pipes_passed ?? 0);
        return `#${trace.rank}: ${pipesPassed}`;
      }).join(" | ");
      els.rankingList.textContent = `Top pipes: ${top}`;
    }
  }

  function drawBrain() {
    if (!state.showBrain || !els.brainPanel) return;
    const champ = getTraceByRank(1);
    const frame = (champ?.frames || [])[Math.min(state.playT, Math.max(0, (champ?.frames || []).length - 1))] || {};
    const outVal = frame.out;

    if (els.brainInputs) els.brainInputs.textContent = "Recorded frame data only";
    if (outVal === undefined || outVal === null) {
      if (els.brainOutput) els.brainOutput.textContent = "(not recorded)";
      if (els.brainDecision) els.brainDecision.textContent = "(not recorded)";
    } else {
      const num = Number(outVal);
      if (els.brainOutput) els.brainOutput.textContent = Number.isFinite(num) ? num.toFixed(3) : String(outVal);
      if (els.brainDecision) els.brainDecision.textContent = num >= 0.5 ? "flap (>=0.5)" : "no flap (<0.5)";
    }
    if (els.brainFlapState) els.brainFlapState.textContent = frame.flap ? "on" : "off";
    if (brainCtx && els.brainCanvas) {
      brainCtx.clearRect(0, 0, els.brainCanvas.width, els.brainCanvas.height);
      brainCtx.fillStyle = "#94a3b8";
      brainCtx.font = "14px monospace";
      brainCtx.fillText("Brain graph unavailable in replay-only mode", 12, 30);
    }
  }

  function render() {
    if (!state.replay) return;
    drawBackground();
    state.pipes.forEach(drawPipe);
    state.renderEntries.forEach((entry, i) => {
      const trail = state.trailHistory[i] || [];
      if (state.showTrails && trail.length > 1) {
        ctx.strokeStyle = entry.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        trail.forEach((p, idx) => (idx ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.stroke();
      }
    });
    state.birds.forEach(drawBird);

    if (state.generationMessage) {
      ctx.fillStyle = "rgba(17,24,39,0.84)";
      drawRoundedPanel(18, state.logicalHeight - 80, state.logicalWidth - 36, 56, 12);
      ctx.fill();
      ctx.fillStyle = "#fef3c7";
      ctx.font = "14px sans-serif";
      ctx.fillText(state.generationMessage, 28, state.logicalHeight - 46);
    }

    updateStats();
    drawBrain();
  }

  function stepReplay() {
    if (!getGeneration()) return;

    const finishGeneration = () => {
      if (state.playing && state.autoplayEnabled && state.generationIndex < (state.generations.length - 1)) {
        loadGeneration(state.generationIndex + 1);
      } else {
        const lastPipeT = Math.max(0, state.pipeFramesLen - 1);
        applyFrame(lastPipeT);
      }
    };

    if (state.playT >= state.pipeFramesLen || state.aliveCountAll <= 0) {
      finishGeneration();
      return;
    }

    applyFrame(state.playT + 1);
    if (state.playT >= state.pipeFramesLen || state.aliveCountAll <= 0) {
      finishGeneration();
    }
  }

  function animate(ts) {
    if (!state.lastTimestamp) state.lastTimestamp = ts;
    const deltaMs = ts - state.lastTimestamp;
    state.lastTimestamp = ts;

    if (state.replay && state.playing) {
      state.stepAccumulator += ((deltaMs / 1000) * 60) * state.simSpeedMultiplier;
      while (state.stepAccumulator >= 1) {
        state.stepAccumulator -= 1;
        stepReplay();
      }
    }

    render();
    requestAnimationFrame(animate);
  }

  function attachControls() {
    els.playPauseBtn?.addEventListener("click", () => {
      state.playing = !state.playing;
      els.playPauseBtn.textContent = state.playing ? "Pause sim" : "Play sim";
    });
    els.prevGenBtn?.addEventListener("click", () => loadGeneration(state.generationIndex - 1));
    els.nextGenBtn?.addEventListener("click", () => loadGeneration(state.generationIndex + 1));
    els.autoplayToggle?.addEventListener("change", (e) => { state.autoplayEnabled = Boolean(e.target.checked); });
    els.generationSlider?.addEventListener("input", (e) => loadGeneration(Number(e.target.value) || 0));
    els.speedSlider?.addEventListener("input", (e) => { state.simSpeedMultiplier = clamp((Number(e.target.value) || 1500) / 1000, 0.5, 3); });
    els.trailToggle?.addEventListener("change", (e) => { state.showTrails = Boolean(e.target.checked); });
    els.debugToggle?.addEventListener("change", (e) => { state.showDebug = Boolean(e.target.checked); });
    els.showBrainToggle?.addEventListener("change", (e) => {
      state.showBrain = Boolean(e.target.checked);
      if (els.brainPanel) els.brainPanel.hidden = !state.showBrain;
    });
    els.showManyBirdsToggle?.addEventListener("change", (e) => {
      state.showManyBirds = Boolean(e.target.checked);
      window.localStorage.setItem(STORAGE_KEYS.showManyBirds, state.showManyBirds ? "1" : "0");
      state.trailHistory = [];
      applyFrame(state.playT);
    });
    els.rankSelector?.addEventListener("change", (e) => {
      state.selectedRank = Number(e.target.value || 1);
      applyFrame(state.playT);
    });
  }

  async function fetchReplay() {
    const path = `./training_replay.json?v=${Date.now()}`;
    const attemptedUrl = new URL(path, window.location.href).href;
    try {
      const response = await fetch(path, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}\nURL: ${attemptedUrl}`);
      }
      const data = await response.json();
      const normalized = normalizeReplayData(data);
      if (!Array.isArray(normalized.generations) || normalized.generations.length === 0) {
        throw new Error("training_replay.json loaded but has 0 generations.");
      }
      return normalized;
    } catch (error) {
      const command = "python main.py --record-training-replay --replay-top-k 30 --replay-episode 0";
      setStatus("Failed to load training replay.");
      setError([
        String(error.message || error),
        `URL: ${attemptedUrl}`,
        "How to generate:",
        command,
        "Expected file at: web/training_replay.json",
      ].join("\n"));
      throw error;
    }
  }

  async function init() {
    attachControls();
    const showManyBirdsSaved = window.localStorage.getItem(STORAGE_KEYS.showManyBirds);
    state.showManyBirds = showManyBirdsSaved === null ? true : showManyBirdsSaved === "1";
    if (els.showManyBirdsToggle) els.showManyBirdsToggle.checked = state.showManyBirds;
    sizeCanvas();
    window.addEventListener("resize", sizeCanvas);

    try {
      const data = await fetchReplay();
      state.replay = data;
      state.generations = Array.isArray(data.generations) ? data.generations : [];
      sizeCanvas();
      if (els.generationSlider) {
        els.generationSlider.min = "0";
        els.generationSlider.max = String(Math.max(0, state.generations.length - 1));
      }
      setError("");
      setStatus(`Loaded ${state.generations.length} generations from training_replay.json.`);
      loadGeneration(0);
      requestAnimationFrame(animate);
    } catch (error) {
      setStatus("Failed to initialize training replay.");
      setError(String(error?.message || error));
    }
  }

  init();
})();
