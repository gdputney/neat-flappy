(() => {
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const prevGenBtn = document.getElementById("prevGenBtn");
  const nextGenBtn = document.getElementById("nextGenBtn");
  const generationSlider = document.getElementById("generationSlider");
  const speedSlider = document.getElementById("speedSlider");
  const autoplayToggle = document.getElementById("autoplayToggle");
  const trailToggle = document.getElementById("trailToggle");
  const championOnlyToggle = document.getElementById("championOnlyToggle");
  const debugToggle = document.getElementById("debugToggle");
  const showBrainToggle = document.getElementById("showBrainToggle");
  const statusEl = document.getElementById("status");
  const rankingList = document.getElementById("rankingList");
  const brainPanel = document.getElementById("brainPanel");
  const brainInputsEl = document.getElementById("brainInputs");
  const brainOutputEl = document.getElementById("brainOutput");
  const brainDecisionEl = document.getElementById("brainDecision");
  const brainFlapStateEl = document.getElementById("brainFlapState");
  const brainCanvas = document.getElementById("brainCanvas");
  const brainCtx = brainCanvas.getContext("2d");

  const statGeneration = document.getElementById("statGeneration");
  const statBirdsShown = document.getElementById("statBirdsShown");
  const statAlive = document.getElementById("statAlive");
  const statBestGen = document.getElementById("statBestGen");
  const statBestAll = document.getElementById("statBestAll");
  const statPlayback = document.getElementById("statPlayback");
  const statSeed = document.getElementById("statSeed");
  const statCurriculumLevel = document.getElementById("statCurriculumLevel");
  const statCurriculumBestEver = document.getElementById("statCurriculumBestEver");
  const statCurriculumGap = document.getElementById("statCurriculumGap");
  const statCurriculumSpeed = document.getElementById("statCurriculumSpeed");
  const statCurriculumSpacing = document.getElementById("statCurriculumSpacing");
  const milestoneBanner = document.getElementById("milestoneBanner");

  const state = {
    data: null,
    generationIndex: 0,
    playing: true,
    autoplayEnabled: true,
    stepAccumulator: 0,
    lastTimestamp: 0,
    maxSteps: 5000,
    generationDone: false,
    pipes: [],
    birds: [],
    nextPipeIndexPerBird: [],
    pipeRngState: 1,
    bestScore: 0,
    showTrails: false,
    showChampionOnly: false,
    showDebug: false,
    showBrain: false,
    trailHistory: [],
    step: 0,
    simSpeedMultiplier: 1.5,
    bestPipesAllTime: 0,
    renderTimeMs: 0,
    runtimeCache: new WeakMap(),
    brainView: {
      championBird: null,
      inputs: null,
      outputs: null,
      activations: null,
      flapDecision: false,
    },
    currentDifficulty: null,
    milestoneBannerText: "",
    milestoneBannerExpiresStep: -1,
    clouds: [
      { x: 30, y: 72, speed: 4, scale: 1.0, alpha: 0.2 },
      { x: 200, y: 120, speed: 7, scale: 1.35, alpha: 0.17 },
      { x: 390, y: 56, speed: 6, scale: 0.9, alpha: 0.22 },
      { x: 140, y: 190, speed: 5, scale: 1.15, alpha: 0.14 },
    ],
  };

  const INPUT_LABELS = [
    "y_norm",
    "velocity_norm",
    "dx_to_next_pipe_norm",
    "gap_error_norm",
    "dy_to_gap_top_norm",
    "dy_to_gap_bottom_norm",
  ];

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function mulberry32(seed) {
    let t = seed >>> 0;
    return () => {
      t += 0x6d2b79f5;
      let x = Math.imul(t ^ (t >>> 15), 1 | t);
      x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function normalizeInputs(bird, pipes, config) {
    const height = config.world_height > 0 ? config.world_height : 1.0;
    const width = config.world_width > 0 ? config.world_width : 1.0;
    const maxAbsVelocity = Math.max(Math.abs(config.velocity_min), Math.abs(config.velocity_max), 1e-6);
    const aheadPipes = pipes.filter((p) => p.x + p.width >= config.bird_x);
    const nextPipe = aheadPipes.length ? aheadPipes.reduce((a, b) => (a.x < b.x ? a : b)) : null;

    const yNorm = clamp((2 * (bird.y / height)) - 1, -1, 1);
    const velocityNorm = clamp(bird.velocity / maxAbsVelocity, -1, 1);

    if (!nextPipe) {
      return [yNorm, velocityNorm, 1.0, 0.0, 0.0, 0.0];
    }

    const gapCenter = (nextPipe.top + nextPipe.bottom) / 2;
    const halfGapHeight = Math.max((nextPipe.bottom - nextPipe.top) / 2, 1e-6);
    const dxNorm = clamp((nextPipe.x - config.bird_x) / width, 0, 1);
    const gapErrorNorm = clamp((bird.y - gapCenter) / halfGapHeight, -1, 1);
    const dyTopNorm = clamp((nextPipe.top - bird.y) / height, -1, 1);
    const dyBottomNorm = clamp((nextPipe.bottom - bird.y) / height, -1, 1);
    return [yNorm, velocityNorm, dxNorm, gapErrorNorm, dyTopNorm, dyBottomNorm];
  }

  function computeTopoOrder(nodeIds, edges) {
    const indegree = new Map(nodeIds.map((id) => [id, 0]));
    const outgoing = new Map(nodeIds.map((id) => [id, []]));
    for (const edge of edges) {
      const inNode = Number(edge.in_node);
      const outNode = Number(edge.out_node);
      indegree.set(outNode, (indegree.get(outNode) || 0) + 1);
      outgoing.get(inNode)?.push(outNode);
    }
    const queue = [...indegree.entries()].filter(([, d]) => d === 0).map(([id]) => id);
    const topo = [];
    while (queue.length) {
      const current = queue.shift();
      topo.push(current);
      for (const outNode of outgoing.get(current) || []) {
        indegree.set(outNode, (indegree.get(outNode) || 0) - 1);
        if ((indegree.get(outNode) || 0) === 0) queue.push(outNode);
      }
    }
    return { topo, cyclic: topo.length !== nodeIds.length };
  }

  function getGenomeRuntime(genomeJson) {
    if (state.runtimeCache.has(genomeJson)) {
      return state.runtimeCache.get(genomeJson);
    }

    const nodes = genomeJson.node_genes || [];
    const nodeIds = nodes.map((n) => Number(n.id));
    const nodeTypes = nodes.map((n) => String(n.type || "hidden"));
    const biases = new Float64Array(nodes.map((n) => Number(n.bias || 0.0)));
    const idToIndex = new Map(nodeIds.map((id, idx) => [id, idx]));
    const inputNodeIndices = [];
    const outputNodeIndices = [];
    const hiddenNodeIndices = [];

    for (let i = 0; i < nodes.length; i += 1) {
      const type = nodeTypes[i];
      if (type === "input") inputNodeIndices.push(i);
      else if (type === "output") outputNodeIndices.push(i);
      else hiddenNodeIndices.push(i);
    }

    const allConnections = (genomeJson.connection_genes || []).map((edge) => ({
      inIndex: idToIndex.get(Number(edge.in_node)),
      outIndex: idToIndex.get(Number(edge.out_node)),
      weight: Number(edge.weight || 0.0),
      enabled: edge.enabled !== false,
    })).filter((edge) => Number.isInteger(edge.inIndex) && Number.isInteger(edge.outIndex));

    const enabledConnections = allConnections.filter((edge) => edge.enabled);
    const incomingEnabled = Array.from({ length: nodes.length }, () => []);
    for (const edge of enabledConnections) {
      incomingEnabled[edge.outIndex].push(edge);
    }

    const { topo, cyclic } = computeTopoOrder(nodeIds, enabledConnections.map((edge) => ({
      in_node: nodeIds[edge.inIndex],
      out_node: nodeIds[edge.outIndex],
    })));
    const evalOrder = cyclic
      ? [...hiddenNodeIndices, ...outputNodeIndices]
      : topo.map((id) => idToIndex.get(id)).filter((idx) => idx !== undefined && nodeTypes[idx] !== "input");

    const layout = computeBrainLayout(nodeTypes, inputNodeIndices, hiddenNodeIndices, outputNodeIndices);
    const runtime = {
      nodeTypes,
      nodeIds,
      inputNodeIndices,
      outputNodeIndices,
      biases,
      incomingEnabled,
      evalOrder,
      cyclic,
      relaxationIterations: 4,
      valuesA: new Float64Array(nodes.length),
      valuesB: new Float64Array(nodes.length),
      outputBuffer: new Float64Array(outputNodeIndices.length),
      allConnections,
      layout,
    };
    state.runtimeCache.set(genomeJson, runtime);
    return runtime;
  }

  function evaluateRuntime(runtime, inputs) {
    const valuesA = runtime.valuesA;
    const valuesB = runtime.valuesB;
    valuesA.fill(0);
    for (let i = 0; i < runtime.inputNodeIndices.length; i += 1) {
      valuesA[runtime.inputNodeIndices[i]] = Number(inputs[i] ?? 0.0);
    }

    if (!runtime.cyclic) {
      for (const nodeIndex of runtime.evalOrder) {
        let weightedSum = runtime.biases[nodeIndex];
        for (const edge of runtime.incomingEnabled[nodeIndex]) {
          weightedSum += edge.weight * valuesA[edge.inIndex];
        }
        valuesA[nodeIndex] = sigmoid(weightedSum);
      }
    } else {
      valuesB.set(valuesA);
      for (let iter = 0; iter < runtime.relaxationIterations; iter += 1) {
        for (const nodeIndex of runtime.evalOrder) {
          let weightedSum = runtime.biases[nodeIndex];
          for (const edge of runtime.incomingEnabled[nodeIndex]) {
            weightedSum += edge.weight * valuesA[edge.inIndex];
          }
          valuesB[nodeIndex] = sigmoid(weightedSum);
        }
        for (const inputIdx of runtime.inputNodeIndices) {
          valuesB[inputIdx] = valuesA[inputIdx];
        }
        valuesA.set(valuesB);
      }
    }

    for (let i = 0; i < runtime.outputNodeIndices.length; i += 1) {
      runtime.outputBuffer[i] = valuesA[runtime.outputNodeIndices[i]];
    }
    return {
      outputs: runtime.outputBuffer,
      activations: valuesA,
    };
  }

  function computeBrainLayout(nodeTypes, inputNodeIndices, hiddenNodeIndices, outputNodeIndices) {
    const width = brainCanvas.width;
    const height = brainCanvas.height;
    const placeColumn = (indices, x) => {
      const positions = new Map();
      if (!indices.length) return positions;
      const step = height / (indices.length + 1);
      for (let i = 0; i < indices.length; i += 1) {
        positions.set(indices[i], { x, y: step * (i + 1) });
      }
      return positions;
    };
    const byInput = placeColumn(inputNodeIndices, 52);
    const byHidden = placeColumn(hiddenNodeIndices, width / 2);
    const byOutput = placeColumn(outputNodeIndices, width - 52);
    const positions = Array.from({ length: nodeTypes.length }, () => ({ x: width / 2, y: height / 2 }));
    for (let i = 0; i < positions.length; i += 1) {
      positions[i] = byInput.get(i) || byHidden.get(i) || byOutput.get(i) || { x: width / 2, y: height / 2 };
    }
    return positions;
  }

  function decideFlap(output, policy, isFlapping, config) {
    if (policy === "hysteresis" || policy === "deterministic") {
      if (output >= config.flap_on_threshold) return true;
      if (output <= config.flap_off_threshold) return false;
      return isFlapping;
    }
    return output >= 0.5;
  }

  function createPipe(rng, x, config) {
    const minCenter = config.pipe_min_margin + (config.pipe_gap_size / 2);
    const maxCenter = config.world_height - config.pipe_min_margin - (config.pipe_gap_size / 2);
    const gapCenter = minCenter + (maxCenter - minCenter) * rng();
    return {
      x,
      width: config.pipe_width,
      top: gapCenter - (config.pipe_gap_size / 2),
      bottom: gapCenter + (config.pipe_gap_size / 2),
    };
  }

  function computeBestPipesAllTime(data) {
    return data.generations.reduce((bestGen, generation) => {
      const genBest = (generation.genomes || []).reduce((bestGenome, genome) => {
        return Math.max(bestGenome, Number(genome.pipes_passed_max || 0));
      }, 0);
      return Math.max(bestGen, genBest);
    }, 0);
  }

  function getChampionPipes(generation) {
    const champ = (generation.genomes || []).find((entry) => Number(entry.rank) === 1) || generation.genomes?.[0];
    return Number(champ?.pipes_passed_max || 0);
  }

  function getGenerationDifficulty(generation) {
    const config = state.data.metadata.config;
    return {
      level: Number(generation.curriculum_level ?? -1),
      bestEver: Number(generation.curriculum_best_pipes_ever ?? 0),
      gap: Number(generation.curriculum_gap ?? config.base_pipe_gap ?? config.pipe_gap_size ?? 180),
      speed: Number(generation.curriculum_pipe_speed ?? config.base_pipe_speed ?? config.pipe_speed ?? 3),
      spacing: Number(generation.curriculum_pipe_spacing ?? config.base_pipe_spacing ?? config.pipe_spacing ?? 220),
    };
  }

  function clearMilestoneBanner() {
    state.milestoneBannerText = "";
    state.milestoneBannerExpiresStep = -1;
    milestoneBanner.textContent = "";
    milestoneBanner.classList.remove("visible");
  }

  function maybeShowMilestoneBanner(prevDifficulty, nextDifficulty) {
    if (!prevDifficulty || !nextDifficulty) return;
    if (!(nextDifficulty.level > prevDifficulty.level)) return;

    const deltaGap = Number((state.data.metadata.config.base_pipe_gap - nextDifficulty.gap).toFixed(2));
    const deltaSpeed = Number((nextDifficulty.speed - state.data.metadata.config.base_pipe_speed).toFixed(2));
    const text = `🏆 Milestone reached: ${nextDifficulty.bestEver} pipes → Gap -${deltaGap}px, Speed +${deltaSpeed.toFixed(2)}`;
    state.milestoneBannerText = text;
    state.milestoneBannerExpiresStep = state.step + 120;
    milestoneBanner.textContent = text;
    milestoneBanner.classList.add("visible");
  }

  function advanceGeneration(delta) {
    if (!state.data) return;
    const total = state.data.generations.length;
    const next = (state.generationIndex + delta + total) % total;
    loadGeneration(next);
  }

  function loadGeneration(generationIdx, options = {}) {
    const generation = state.data.generations[generationIdx];
    const config = state.data.metadata.config;
    const previousDifficulty = state.currentDifficulty;
    const nextDifficulty = getGenerationDifficulty(generation);
    const seed32 = Number(BigInt(generation.pipe_seed) & BigInt(0xffffffff));
    const rng = mulberry32(seed32);

    state.generationIndex = generationIdx;
    state.maxSteps = Number(config.max_steps) || 5000;
    state.step = 0;
    state.generationDone = false;
    state.bestScore = 0;
    state.stepAccumulator = 0;
    state.pipes = [createPipe(rng, config.first_pipe_x, { ...config, pipe_gap_size: nextDifficulty.gap })];
    state.birds = generation.genomes.map((entry, i) => ({
      rank: entry.rank,
      genomeJson: entry.genome_json,
      y: config.bird_start_y,
      velocity: 0,
      flapCooldown: 0,
      isFlapping: false,
      alive: true,
      score: 0,
      color: `hsla(${Math.round((i * 360) / Math.max(1, generation.genomes.length))}, 85%, 52%, 0.72)`,
      runtime: getGenomeRuntime(entry.genome_json),
    }));
    state.nextPipeIndexPerBird = state.birds.map(() => 0);
    state.trailHistory = state.birds.map(() => []);
    state.pipeRngState = rng;
    state.brainView = {
      championBird: state.birds.find((bird) => bird.rank === 1) || state.birds[0] || null,
      inputs: null,
      outputs: null,
      activations: null,
      flapDecision: false,
    };
    state.currentDifficulty = nextDifficulty;
    if (options.clearBanner !== false) {
      clearMilestoneBanner();
    }
    if (options.triggerMilestoneBanner) {
      maybeShowMilestoneBanner(previousDifficulty, nextDifficulty);
    }
    generationSlider.value = String(generationIdx);
    render();
  }

  function formatNumber(value) {
    return Number.isFinite(value) ? value.toFixed(3) : "-";
  }

  function activationToFill(value, type) {
    const normalized = type === "input" ? clamp((value + 1) / 2, 0, 1) : clamp(value, 0, 1);
    const lightness = Math.round(20 + (normalized * 60));
    return `hsl(195 80% ${lightness}%)`;
  }

  function drawBrainOverlay() {
    if (!state.showBrain) return;
    const { championBird, inputs, outputs, activations } = state.brainView;
    if (!championBird || !championBird.runtime || !inputs || !outputs || !activations) return;
    const runtime = championBird.runtime;

    brainInputsEl.innerHTML = INPUT_LABELS.map((label, i) => (
      `<div>${label}: <b>${formatNumber(inputs[i])}</b></div>`
    )).join("");
    const primaryOutput = Number(outputs[0] || 0.0);
    brainOutputEl.textContent = formatNumber(primaryOutput);
    brainDecisionEl.textContent = `${state.brainView.flapDecision ? "flap" : "no flap"} (${primaryOutput >= 0.5 ? ">=0.5" : "<0.5"})`;
    brainFlapStateEl.textContent = `cooldown=${championBird.flapCooldown}, hysteresis=${championBird.isFlapping ? "on" : "off"}`;

    brainCtx.clearRect(0, 0, brainCanvas.width, brainCanvas.height);
    for (const edge of runtime.allConnections) {
      const a = runtime.layout[edge.inIndex];
      const b = runtime.layout[edge.outIndex];
      brainCtx.beginPath();
      brainCtx.moveTo(a.x, a.y);
      brainCtx.lineTo(b.x, b.y);
      const thickness = 0.6 + (Math.min(Math.abs(edge.weight), 2.5) * 0.9);
      brainCtx.lineWidth = edge.enabled ? thickness : 1;
      brainCtx.setLineDash(edge.enabled ? [] : [4, 4]);
      const alpha = edge.enabled ? 0.42 : 0.18;
      brainCtx.strokeStyle = edge.weight >= 0
        ? `rgba(56, 189, 248, ${alpha})`
        : `rgba(248, 113, 113, ${alpha})`;
      brainCtx.stroke();
    }
    brainCtx.setLineDash([]);

    for (let i = 0; i < runtime.layout.length; i += 1) {
      const p = runtime.layout[i];
      const value = activations[i];
      const type = runtime.nodeTypes[i];
      brainCtx.beginPath();
      brainCtx.fillStyle = activationToFill(value, type);
      brainCtx.strokeStyle = type === "output" ? "#fef08a" : "rgba(203, 213, 225, 0.9)";
      brainCtx.lineWidth = type === "output" ? 2 : 1;
      brainCtx.arc(p.x, p.y, 9, 0, Math.PI * 2);
      brainCtx.fill();
      brainCtx.stroke();
    }
  }

  function drawBackground() {
    const sky = ctx.createLinearGradient(0, 0, 0, canvas.height);
    sky.addColorStop(0, "#7bc8ff");
    sky.addColorStop(0.55, "#8dd8ff");
    sky.addColorStop(1, "#c6f0ff");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const timeSeconds = state.renderTimeMs / 1000;
    for (let i = 0; i < state.clouds.length; i += 1) {
      const cloud = state.clouds[i];
      const x = (cloud.x + (timeSeconds * cloud.speed)) % (canvas.width + 140) - 70;
      const y = cloud.y;
      const radius = 16 * cloud.scale;
      ctx.fillStyle = `rgba(255,255,255,${cloud.alpha})`;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.arc(x + radius * 0.85, y - radius * 0.2, radius * 0.8, 0, Math.PI * 2);
      ctx.arc(x + radius * 1.6, y, radius * 0.95, 0, Math.PI * 2);
      ctx.fill();
    }

    const groundHeight = 34;
    const dirt = ctx.createLinearGradient(0, canvas.height - groundHeight, 0, canvas.height);
    dirt.addColorStop(0, "#ab7a36");
    dirt.addColorStop(1, "#7f5522");
    ctx.fillStyle = "#79b84f";
    ctx.fillRect(0, canvas.height - groundHeight - 8, canvas.width, 8);
    ctx.fillStyle = dirt;
    ctx.fillRect(0, canvas.height - groundHeight, canvas.width, groundHeight);
  }

  function drawPipe(pipe, pipeIndex) {
    const border = 4;
    const capHeight = 12;
    const capOverhang = 6;

    const drawSegment = (x, y, width, height) => {
      ctx.fillStyle = "#2f6f25";
      ctx.fillRect(x, y, width, height);
      ctx.fillStyle = "#4ea53f";
      ctx.fillRect(x + border, y + border, Math.max(0, width - (2 * border)), Math.max(0, height - (2 * border)));
      ctx.fillStyle = "rgba(255,255,255,0.10)";
      ctx.fillRect(x + border + 2, y + border, 4, Math.max(0, height - (2 * border)));
    };

    drawSegment(pipe.x, 0, pipe.width, pipe.top);
    drawSegment(pipe.x, pipe.bottom, pipe.width, canvas.height - pipe.bottom);

    ctx.fillStyle = "#2f6f25";
    ctx.fillRect(pipe.x - capOverhang, pipe.top - capHeight, pipe.width + (2 * capOverhang), capHeight);
    ctx.fillRect(pipe.x - capOverhang, pipe.bottom, pipe.width + (2 * capOverhang), capHeight);
    ctx.fillStyle = "#5cbf4c";
    ctx.fillRect(pipe.x - capOverhang + 2, pipe.top - capHeight + 2, pipe.width + (2 * capOverhang) - 4, capHeight - 4);
    ctx.fillRect(pipe.x - capOverhang + 2, pipe.bottom + 2, pipe.width + (2 * capOverhang) - 4, capHeight - 4);

    if (state.showDebug) {
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.font = "12px monospace";
      ctx.fillText(`#${pipeIndex}`, pipe.x + 4, Math.max(14, pipe.top - 4));
    }
  }

  function drawBird(bird, birdIndex, config) {
    const x = config.bird_x;
    const y = bird.y;
    const angle = clamp(bird.velocity * 0.11, -0.65, 0.75);
    const flapPhase = (state.renderTimeMs / 100) + (birdIndex * 0.25);
    const wingLift = bird.isFlapping ? Math.sin(flapPhase) * 4 : 0;

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    ctx.shadowColor = "rgba(0,0,0,0.24)";
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 2;

    ctx.fillStyle = bird.color;
    ctx.beginPath();
    ctx.moveTo(12, 0);
    ctx.quadraticCurveTo(5, -8, -8, -6);
    ctx.quadraticCurveTo(-12, 0, -8, 6);
    ctx.quadraticCurveTo(5, 8, 12, 0);
    ctx.closePath();
    ctx.fill();

    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-2, -1);
    ctx.lineTo(-9, -5 - wingLift);
    ctx.stroke();

    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(5, -2, 1.8, 0, Math.PI * 2);
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

  function stepSimulation() {
    const config = state.data.metadata.config;
    const difficulty = state.currentDifficulty || getGenerationDifficulty(state.data.generations[state.generationIndex]);
    const effectiveConfig = {
      ...config,
      pipe_gap_size: difficulty.gap,
      pipe_speed: difficulty.speed,
      pipe_spacing: difficulty.spacing,
    };
    if (state.generationDone) return;

    if (state.pipes[state.pipes.length - 1].x < effectiveConfig.world_width - effectiveConfig.pipe_spacing) {
      state.pipes.push(createPipe(state.pipeRngState, effectiveConfig.new_pipe_x, effectiveConfig));
    }

    for (let i = 0; i < state.birds.length; i += 1) {
      const bird = state.birds[i];
      if (!bird.alive) continue;

      const inputs = normalizeInputs(bird, state.pipes, effectiveConfig);
      const { outputs, activations } = evaluateRuntime(bird.runtime, inputs);
      const output = Number(outputs[0] || 0);
      const flap = decideFlap(output, state.data.metadata.flap_policy, bird.isFlapping, effectiveConfig);
      bird.isFlapping = flap;

      if (!state.brainView.championBird || bird.rank === 1 || (state.birds.length === 1 && i === 0)) {
        state.brainView = {
          championBird: bird,
          inputs,
          outputs,
          activations,
          flapDecision: flap,
        };
      }

      if (bird.flapCooldown > 0) bird.flapCooldown -= 1;
      if (flap && bird.flapCooldown === 0) {
        bird.velocity = effectiveConfig.jump_strength;
        bird.flapCooldown = effectiveConfig.flap_cooldown_frames;
      }

      bird.velocity = clamp(bird.velocity + effectiveConfig.gravity, effectiveConfig.velocity_min, effectiveConfig.velocity_max);
      bird.y += bird.velocity;

      if (bird.y < 0) {
        bird.y = 0;
        bird.velocity = 0;
      }
      if (bird.y >= effectiveConfig.world_height) {
        bird.alive = false;
      }

      for (const pipe of state.pipes) {
        const withinX = pipe.x <= effectiveConfig.bird_x && effectiveConfig.bird_x <= (pipe.x + pipe.width);
        if (withinX && !(pipe.top <= bird.y && bird.y <= pipe.bottom)) {
          bird.alive = false;
        }
      }

      while (state.nextPipeIndexPerBird[i] < state.pipes.length) {
        const pipe = state.pipes[state.nextPipeIndexPerBird[i]];
        if (effectiveConfig.bird_x <= (pipe.x + pipe.width)) break;
        bird.score += 1;
        state.nextPipeIndexPerBird[i] += 1;
      }

      state.bestScore = Math.max(state.bestScore, bird.score);
      if (state.showTrails) {
        const trail = state.trailHistory[i];
        trail.push({ x: effectiveConfig.bird_x, y: bird.y });
        if (trail.length > 120) trail.shift();
      }
    }

    for (const pipe of state.pipes) {
      pipe.x -= effectiveConfig.pipe_speed;
    }

    const countBefore = state.pipes.length;
    state.pipes = state.pipes.filter((pipe) => (pipe.x + pipe.width) > effectiveConfig.offscreen_pipe_right_threshold);
    const removed = countBefore - state.pipes.length;
    if (removed > 0) {
      state.nextPipeIndexPerBird = state.nextPipeIndexPerBird.map((idx) => Math.max(0, idx - removed));
    }

    state.step += 1;
    if (state.milestoneBannerText && state.step >= state.milestoneBannerExpiresStep) {
      clearMilestoneBanner();
    }
    const aliveCount = state.birds.filter((bird) => bird.alive).length;
    state.generationDone = aliveCount === 0 || state.step >= state.maxSteps;
  }

  function updateStats(generation) {
    const total = state.data.generations.length;
    const visibleBirds = state.showChampionOnly ? Math.min(state.birds.length, 1) : state.birds.length;
    const alive = state.birds.filter((bird) => bird.alive).length;
    statGeneration.textContent = `${state.generationIndex + 1} / ${total}`;
    statBirdsShown.textContent = String(visibleBirds);
    statAlive.textContent = `${alive} / ${state.birds.length}`;
    statBestGen.textContent = String(getChampionPipes(generation));
    statBestAll.textContent = String(state.bestPipesAllTime);
    statPlayback.textContent = state.autoplayEnabled ? "ON (sequential)" : "OFF";
    statSeed.textContent = String(generation.pipe_seed);
    const difficulty = state.currentDifficulty || getGenerationDifficulty(generation);
    statCurriculumLevel.textContent = difficulty.level >= 0 ? String(difficulty.level) : "-";
    statCurriculumBestEver.textContent = String(difficulty.bestEver);
    statCurriculumGap.textContent = Number.isFinite(difficulty.gap) ? difficulty.gap.toFixed(1) : "-";
    statCurriculumSpeed.textContent = Number.isFinite(difficulty.speed) ? difficulty.speed.toFixed(2) : "-";
    statCurriculumSpacing.textContent = Number.isFinite(difficulty.spacing) ? difficulty.spacing.toFixed(1) : "-";
  }

  function render() {
    const generation = state.data?.generations?.[state.generationIndex];
    if (!generation) return;
    const config = state.data.metadata.config;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBackground();
    for (let i = 0; i < state.pipes.length; i += 1) {
      drawPipe(state.pipes[i], i);
    }

    const championBird = state.birds.find((bird) => bird.rank === 1) || state.birds[0];
    for (let i = 0; i < state.birds.length; i += 1) {
      const bird = state.birds[i];
      if (state.showChampionOnly && bird !== championBird) continue;
      if (!bird.alive && !state.showTrails) continue;

      if (state.showTrails) {
        ctx.strokeStyle = bird.color;
        ctx.lineWidth = bird.rank === 1 ? 2.2 : 1;
        ctx.beginPath();
        const points = state.trailHistory[i];
        for (let j = 0; j < points.length; j += 1) {
          const p = points[j];
          if (j === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }

      drawBird(bird, i, config);

      if (state.showDebug) {
        ctx.fillStyle = "#111";
        ctx.font = "11px monospace";
        ctx.fillText(`r${bird.rank}: ${bird.score}`, config.bird_x + 16, bird.y - 10);
      }
    }

    updateStats(generation);

    const topFive = [...state.birds]
      .sort((a, b) => (a.rank - b.rank))
      .slice(0, 5)
      .map((bird) => `#${bird.rank}: ${bird.score}${bird.alive ? "" : " ✖"}`)
      .join(" | ");
    rankingList.textContent = `Top scores: ${topFive}`;
    if (state.milestoneBannerText) {
      milestoneBanner.textContent = state.milestoneBannerText;
      milestoneBanner.classList.add("visible");
    }
    drawBrainOverlay();
  }

  function animate(timestamp) {
    if (!state.lastTimestamp) state.lastTimestamp = timestamp;
    const deltaMs = timestamp - state.lastTimestamp;
    state.lastTimestamp = timestamp;
    state.renderTimeMs += deltaMs;

    if (state.data) {
      if (state.playing) {
        state.stepAccumulator += ((deltaMs / 1000) * 60) * state.simSpeedMultiplier;
        while (state.stepAccumulator >= 1) {
          state.stepAccumulator -= 1;
          stepSimulation();
        }
      }

      if (state.playing && state.autoplayEnabled && state.generationDone) {
        const atLastGeneration = state.generationIndex >= state.data.generations.length - 1;
        if (atLastGeneration) {
          state.autoplayEnabled = false;
          autoplayToggle.checked = false;
        } else {
          loadGeneration(state.generationIndex + 1, { triggerMilestoneBanner: true, clearBanner: false });
        }
      }
    }

    render();
    requestAnimationFrame(animate);
  }

  function attachControls() {
    playPauseBtn.addEventListener("click", () => {
      state.playing = !state.playing;
      playPauseBtn.textContent = state.playing ? "Pause sim" : "Play sim";
    });

    autoplayToggle.addEventListener("change", (event) => {
      state.autoplayEnabled = Boolean(event.target.checked);
    });

    prevGenBtn.addEventListener("click", () => loadGeneration((state.generationIndex - 1 + state.data.generations.length) % state.data.generations.length, { clearBanner: true }));
    nextGenBtn.addEventListener("click", () => loadGeneration((state.generationIndex + 1) % state.data.generations.length, { clearBanner: true }));

    generationSlider.addEventListener("input", (event) => {
      if (!state.data) return;
      loadGeneration(Number(event.target.value) || 0, { clearBanner: true });
    });

    speedSlider.addEventListener("input", (event) => {
      state.simSpeedMultiplier = clamp((Number(event.target.value) || 1500) / 1000, 0.5, 3);
    });

    trailToggle.addEventListener("change", (event) => {
      state.showTrails = Boolean(event.target.checked);
      if (!state.showTrails) {
        state.trailHistory = state.trailHistory.map(() => []);
      }
    });

    championOnlyToggle.addEventListener("change", (event) => {
      state.showChampionOnly = Boolean(event.target.checked);
    });

    debugToggle.addEventListener("change", (event) => {
      state.showDebug = Boolean(event.target.checked);
    });

    showBrainToggle.addEventListener("change", (event) => {
      state.showBrain = Boolean(event.target.checked);
      brainPanel.hidden = !state.showBrain;
    });
  }

  async function loadEvolution() {
    try {
      const response = await fetch("evolution.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!Array.isArray(data.generations) || data.generations.length === 0) {
        throw new Error("No generations in evolution.json");
      }
      state.data = data;
      state.bestPipesAllTime = computeBestPipesAllTime(data);
      generationSlider.min = "0";
      generationSlider.max = String(data.generations.length - 1);
      setStatus(`Loaded ${data.generations.length} generations from evolution.json.`);
      state.simSpeedMultiplier = clamp((Number(speedSlider.value) || 1500) / 1000, 0.5, 3);
      loadGeneration(0, { clearBanner: true });
    } catch (error) {
      setStatus(`Failed to load evolution.json: ${error.message}`);
      console.error(error);
    }
  }

  attachControls();
  loadEvolution();
  requestAnimationFrame(animate);
})();
