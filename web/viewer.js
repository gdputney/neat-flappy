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
    frameIndex: 0,
    playing: true,
    autoplayEnabled: true,
    simSpeedMultiplier: 1.5,
    showTrails: false,
    showDebug: false,
    showBrain: false,
    showManyBirds: false,
    selectedRank: 1,
    stepAccumulator: 0,
    lastTimestamp: 0,
    birds: [],
    pipes: [],
    trailHistory: [],
  };

  const setStatus = (text) => { if (els.status) els.status.textContent = text; };
  const setError = (text) => { if (els.errorMessage) els.errorMessage.textContent = text; };
  if (els.modeBadge) els.modeBadge.textContent = "TRAINING REPLAY";

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function sizeCanvas() {
    const cfg = state.replay?.config || {};
    canvas.width = Number(cfg.world_width || canvas.width || 500);
    canvas.height = Number(cfg.world_height || canvas.height || 800);
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

  function applyFrame(frameIndex) {
    const generation = getGeneration();
    if (!generation) return;

    const cfg = state.replay?.config || {};
    const worldHeight = Number(cfg.world_height || canvas.height);
    const shown = state.showManyBirds ? state.traces : [getTraceByRank(state.selectedRank)].filter(Boolean);

    const generationLabel = Number(generation?.generation ?? state.generationIndex);
    if (state.traces.length === 0) {
      state.generationMessage = `No frames/traces found in generation ${generationLabel}. Expected generations[g].genomes[].frames[]`;
    } else {
      state.generationMessage = "";
    }

    state.frameIndex = frameIndex;
    state.birds = shown.map((trace, idx) => {
      const frames = trace.frames || [];
      const safeIndex = Math.min(frameIndex, Math.max(0, frames.length - 1));
      const frame = frames[safeIndex];
      if (!frame) return null;
      return {
        rank: Number(trace.rank || 1),
        x: clamp(Number(frame.x ?? cfg.bird_x ?? 80), 0, canvas.width),
        y: clamp(Number(frame.y ?? worldHeight / 2), 0, worldHeight),
        velocity: Number(frame.vy ?? 0),
        alive: Boolean(frame.alive ?? 0),
        flap: Boolean(frame.flap ?? 0),
        out: frame.out,
        pipesPassed: Number(frame.pipes_passed ?? 0),
        color: `hsla(${Math.round((idx * 360) / Math.max(1, shown.length))}, 85%, 52%, 0.72)`,
      };
    }).filter(Boolean);

    const champion = getTraceByRank(1) || shown[0] || null;
    const champFrames = champion?.frames || [];
    const champFrame = champFrames[Math.min(frameIndex, Math.max(0, champFrames.length - 1))] || null;
    const pipeSource = Array.isArray(champFrame?.pipes) ? champFrame.pipes : [];

    state.pipes = pipeSource.map((pipe) => {
      const gapH = Number(pipe.gap_h || cfg.pipe_gap || 170);
      const halfGap = gapH / 2;
      return {
        x: Number(pipe.x || 0),
        width: Number(pipe.width || cfg.pipe_width || 70),
        top: Number(pipe.gap_y || worldHeight / 2) - halfGap,
        bottom: Number(pipe.gap_y || worldHeight / 2) + halfGap,
      };
    });

    if (state.showTrails) {
      state.birds.forEach((bird, idx) => {
        state.trailHistory[idx] = state.trailHistory[idx] || [];
        state.trailHistory[idx].push({ x: bird.x, y: bird.y });
        if (state.trailHistory[idx].length > 120) state.trailHistory[idx].shift();
      });
    } else {
      state.trailHistory = state.birds.map(() => []);
    }
  }

  function loadGeneration(index) {
    const total = state.generations?.length || 0;
    if (!total) return;
    state.generationIndex = clamp(index, 0, total - 1);
    state.frameIndex = 0;
    state.traces = buildTraces(getGeneration());
    if (els.generationSlider) els.generationSlider.value = String(state.generationIndex);
    refreshRankSelector();
    applyFrame(0);
  }

  function drawBackground() {
    const sky = ctx.createLinearGradient(0, 0, 0, canvas.height);
    sky.addColorStop(0, "#7bc8ff");
    sky.addColorStop(1, "#c6f0ff");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function drawPipe(pipe) {
    ctx.fillStyle = "#2f6f25";
    ctx.fillRect(pipe.x, 0, pipe.width, pipe.top);
    ctx.fillRect(pipe.x, pipe.bottom, pipe.width, canvas.height - pipe.bottom);
  }

  function drawBird(bird) {
    ctx.save();
    ctx.translate(bird.x, bird.y);
    ctx.rotate(clamp(bird.velocity * 0.11, -0.65, 0.75));
    ctx.fillStyle = bird.color;
    ctx.beginPath();
    ctx.arc(0, 0, 11, 0, Math.PI * 2);
    ctx.fill();
    if (bird.rank === 1) {
      ctx.strokeStyle = "#ffd54f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(0, 0, 13, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  function updateStats() {
    const generation = getGeneration();
    const total = state.generations.length;
    const genNum = Number(generation?.generation ?? state.generationIndex);
    const birdsShown = state.showManyBirds ? state.traces.length : Math.min(1, state.traces.length);
    const aliveCount = state.birds.filter((b) => b.alive).length;
    const bestGen = Math.max(0, ...(state.traces.map((trace) => {
      const endFrame = (trace.frames || [])[Math.max(0, (trace.frames || []).length - 1)] || {};
      return Number(endFrame.pipes_passed ?? trace.pipes_passed ?? 0);
    })));
    const champion = getTraceByRank(1);
    const championFrame = (champion?.frames || [])[Math.min(state.frameIndex, Math.max(0, (champion?.frames || []).length - 1))] || {};

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
          `champFramesLen=${(champion?.frames || []).length}`,
          `currentFrameIndex=${state.frameIndex}`,
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
    const frame = (champ?.frames || [])[Math.min(state.frameIndex, Math.max(0, (champ?.frames || []).length - 1))] || {};
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
    state.birds.forEach((bird, i) => {
      const trail = state.trailHistory[i] || [];
      if (state.showTrails && trail.length > 1) {
        ctx.strokeStyle = bird.color;
        ctx.beginPath();
        trail.forEach((p, idx) => (idx ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.stroke();
      }
      drawBird(bird);
    });

    if (state.generationMessage) {
      ctx.fillStyle = "rgba(17,24,39,0.90)";
      ctx.fillRect(18, canvas.height - 74, canvas.width - 36, 52);
      ctx.fillStyle = "#fef3c7";
      ctx.font = "14px sans-serif";
      ctx.fillText(state.generationMessage, 28, canvas.height - 42);
    }

    updateStats();
    drawBrain();
  }

  function stepReplay() {
    if (!getGeneration()) return;
    const maxFrames = getLongestFrameCount();
    const next = state.frameIndex + 1;
    if (next >= maxFrames) {
      if (state.playing && state.autoplayEnabled && state.generationIndex < (state.generations.length - 1)) {
        loadGeneration(state.generationIndex + 1);
      } else {
        applyFrame(Math.max(0, maxFrames - 1));
      }
      return;
    }
    applyFrame(next);
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
      state.trailHistory = [];
      applyFrame(state.frameIndex);
    });
    els.rankSelector?.addEventListener("change", (e) => {
      state.selectedRank = Number(e.target.value || 1);
      applyFrame(state.frameIndex);
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
