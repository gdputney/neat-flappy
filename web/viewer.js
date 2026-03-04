(() => {
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;

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
    data: null,
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
    renderTimeMs: 0,
    birds: [],
    pipes: [],
    trailHistory: [],
  };

  const setStatus = (text) => { if (els.status) els.status.textContent = text; };
  const setError = (text) => { if (els.errorMessage) els.errorMessage.textContent = text; };
  if (els.modeBadge) els.modeBadge.textContent = "TRAINING REPLAY";

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function getGeneration() {
    return state.data?.generations?.[state.generationIndex] || null;
  }

  function getSortedGenomes(generation) {
    return [...(generation?.genomes || [])].sort((a, b) => Number(a.rank || 999) - Number(b.rank || 999));
  }

  function getGenomeByRank(generation, rank) {
    return getSortedGenomes(generation).find((g) => Number(g.rank) === Number(rank)) || getSortedGenomes(generation)[0] || null;
  }

  function getLongestFrameCount(generation) {
    return Math.max(0, ...getSortedGenomes(generation).map((g) => (g.frames || []).length));
  }

  function refreshRankSelector() {
    const generation = getGeneration();
    if (!generation || !els.rankSelector) return;
    const genomes = getSortedGenomes(generation);
    els.rankSelector.innerHTML = "";
    for (const genome of genomes) {
      const option = document.createElement("option");
      option.value = String(genome.rank);
      option.textContent = `Rank ${genome.rank}`;
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
    const cfg = state.data?.config || {};
    const worldHeight = Number(cfg.world_height || canvas.height);
    const shown = state.showManyBirds ? getSortedGenomes(generation) : [getGenomeByRank(generation, state.selectedRank)].filter(Boolean);

    state.frameIndex = frameIndex;
    state.birds = shown.map((genome, idx) => {
      const frames = genome.frames || [];
      const frame = frames[Math.min(frameIndex, Math.max(0, frames.length - 1))] || {};
      return {
        rank: Number(genome.rank || 1),
        y: clamp(Number(frame.y ?? worldHeight / 2), 0, worldHeight),
        velocity: Number(frame.vy ?? 0),
        alive: Boolean(frame.alive ?? 0),
        flap: Boolean(frame.flap ?? 0),
        out: frame.out,
        pipesPassed: Number(frame.pipes_passed ?? 0),
        color: `hsla(${Math.round((idx * 360) / Math.max(1, shown.length))}, 85%, 52%, 0.72)`,
      };
    });

    const selected = getGenomeByRank(generation, state.selectedRank);
    state.pipes = (selected?.pipes || []).map((pipe) => {
      const halfGap = Number(pipe.gap_h || 0) / 2;
      return {
        x: Number(pipe.x),
        width: Number(cfg.pipe_width || 70),
        top: Number(pipe.gap_y) - halfGap,
        bottom: Number(pipe.gap_y) + halfGap,
      };
    });

    if (state.showTrails) {
      state.birds.forEach((bird, idx) => {
        state.trailHistory[idx] = state.trailHistory[idx] || [];
        state.trailHistory[idx].push({ x: Number(cfg.bird_x || 80), y: bird.y });
        if (state.trailHistory[idx].length > 120) state.trailHistory[idx].shift();
      });
    } else {
      state.trailHistory = state.birds.map(() => []);
    }
  }

  function loadGeneration(index) {
    const total = state.data?.generations?.length || 0;
    if (!total) return;
    state.generationIndex = clamp(index, 0, total - 1);
    state.frameIndex = 0;
    if (els.generationSlider) els.generationSlider.value = String(state.generationIndex);
    refreshRankSelector();
    applyFrame(0);
  }

  function drawBackground() {
    const sky = ctx.createLinearGradient(0, 0, 0, state.worldHeight);
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

  function drawBird(bird, birdIndex) {
    const x = Number(state.data?.config?.bird_x || 80);
    ctx.save();
    ctx.translate(x, bird.y);
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
    if (state.showDebug) {
      ctx.fillStyle = "#111";
      ctx.font = "11px monospace";
      ctx.fillText(`r${bird.rank}: ${bird.pipesPassed}`, 16, -10);
    }
    ctx.restore();
  }

  function updateStats() {
    const generation = getGeneration();
    const total = state.data?.generations?.length || 0;
    const genNum = Number(generation?.generation ?? state.generationIndex);
    const shownCount = state.showManyBirds ? state.birds.length : Math.min(1, state.birds.length);
    const aliveCount = state.birds.filter((b) => b.alive).length;
    const bestGen = Math.max(0, ...((generation?.genomes || []).map((g) => Number(g.pipes_passed || 0))));
    const selected = getGenomeByRank(generation, state.selectedRank);
    const selFrame = (selected?.frames || [])[Math.min(state.frameIndex, Math.max(0, (selected?.frames || []).length - 1))] || {};

    if (els.statGeneration) els.statGeneration.textContent = `${genNum} (idx ${state.generationIndex}, total ${total})`;
    if (els.statBirdsShown) els.statBirdsShown.textContent = String(shownCount);
    if (els.statAlive) els.statAlive.textContent = String(aliveCount);
    if (els.statBestGen) els.statBestGen.textContent = String(bestGen);
    if (els.statBestAll) els.statBestAll.textContent = String(Number(selFrame.pipes_passed ?? 0));
    if (els.statPlayback) els.statPlayback.textContent = state.autoplayEnabled ? "ON (sequential)" : "OFF";

    if (els.generationDebugLine) {
      els.generationDebugLine.textContent = `genIdx=${state.generationIndex} | genNum=${genNum} | total=${total}`;
    }

    if (els.rankingList) {
      const top = getSortedGenomes(generation).slice(0, 5).map((g) => `#${g.rank}: ${g.pipes_passed}`).join(" | ");
      els.rankingList.textContent = `Top pipes: ${top}`;
    }
  }

  function drawBrain() {
    if (!state.showBrain || !els.brainPanel) return;
    const generation = getGeneration();
    const champ = getGenomeByRank(generation, 1);
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
    if (!state.data) return;
    drawBackground();
    state.pipes.forEach(drawPipe);
    state.birds.forEach((bird, i) => {
      if (!bird.alive && !state.showTrails) return;
      const trail = state.trailHistory[i] || [];
      if (state.showTrails && trail.length > 1) {
        ctx.strokeStyle = bird.color;
        ctx.beginPath();
        trail.forEach((p, idx) => (idx ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.stroke();
      }
      drawBird(bird, i);
    });
    updateStats();
    drawBrain();
  }

  function stepReplay() {
    const generation = getGeneration();
    if (!generation) return;
    const maxFrames = getLongestFrameCount(generation);
    const next = state.frameIndex + 1;
    if (next >= maxFrames) {
      if (state.playing && state.autoplayEnabled && state.generationIndex < (state.data.generations.length - 1)) {
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
    state.renderTimeMs += deltaMs;

    if (state.data && state.playing) {
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
      if (!Array.isArray(data.generations) || data.generations.length === 0) {
        throw new Error("training_replay.json loaded but has 0 generations.");
      }
      return data;
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
    try {
      const data = await fetchReplay();
      state.data = data;
      if (els.generationSlider) {
        els.generationSlider.min = "0";
        els.generationSlider.max = String(Math.max(0, data.generations.length - 1));
      }
      setError("");
      setStatus(`Loaded ${data.generations.length} generations from training_replay.json.`);
      loadGeneration(0);
      requestAnimationFrame(animate);
    } catch {
      // Error state rendered in hero message.
    }
  }

  init();
})();
