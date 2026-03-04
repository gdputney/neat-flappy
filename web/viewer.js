(() => {
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const prevGenBtn = document.getElementById("prevGenBtn");
  const nextGenBtn = document.getElementById("nextGenBtn");
  const generationSlider = document.getElementById("generationSlider");
  const speedSlider = document.getElementById("speedSlider");
  const trailToggle = document.getElementById("trailToggle");
  const championOnlyToggle = document.getElementById("championOnlyToggle");
  const debugToggle = document.getElementById("debugToggle");
  const statusEl = document.getElementById("status");
  const rankingList = document.getElementById("rankingList");
  const speciesLegend = document.getElementById("speciesLegend");

  const statGeneration = document.getElementById("statGeneration");
  const statBirdsShown = document.getElementById("statBirdsShown");
  const statAlive = document.getElementById("statAlive");
  const statBestGen = document.getElementById("statBestGen");
  const statBestAll = document.getElementById("statBestAll");
  const statPlayback = document.getElementById("statPlayback");
  const statSeed = document.getElementById("statSeed");

  const statGeneration = document.getElementById("statGeneration");
  const statBirdsShown = document.getElementById("statBirdsShown");
  const statAlive = document.getElementById("statAlive");
  const statBestGen = document.getElementById("statBestGen");
  const statBestAll = document.getElementById("statBestAll");
  const statPlayback = document.getElementById("statPlayback");
  const statSeed = document.getElementById("statSeed");

  const state = {
    data: null,
    generations: [],
    config: null,
    generationIndex: 0,
    playing: true,
    stepAccumulator: 0,
    lastTimestamp: 0,
    maxSteps: 5000,
    simEnded: false,
    pipes: [],
    birds: [],
    nextPipeIndexPerBird: [],
    pipeRngState: 1,
    bestScore: 0,
    showTrails: false,
    showChampionOnly: false,
    showDebug: false,
    trailHistory: [],
    step: 0,
    autoplayIntervalMs: 1500,
    autoplayElapsedMs: 0,
    bestPipesAllTime: 0,
    renderTimeMs: 0,
    clouds: [
      { x: 30, y: 72, speed: 4, scale: 1.0, alpha: 0.2 },
      { x: 200, y: 120, speed: 7, scale: 1.35, alpha: 0.17 },
      { x: 390, y: 56, speed: 6, scale: 0.9, alpha: 0.22 },
      { x: 140, y: 190, speed: 5, scale: 1.15, alpha: 0.14 },
    ],
  };

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function showError(message) {
    setStatus(`❌ ${message}`);
    rankingList.textContent = "";
    if (speciesLegend) speciesLegend.innerHTML = message;
    if (viewerDebugPanel) {
      viewerDebugPanel.hidden = false;
      viewerDebugPanel.textContent = [
        `URL: ${state.evolutionUrl}`,
        `Top-level keys: ${state.topLevelKeys.join(", ") || "(none)"}`,
        `Total generations: ${state.generations.length}`,
      ].join("\n");

    }
  }

  function getDefined(...values) {
    for (const value of values) {
      if (value !== undefined && value !== null) return value;
    }
    return undefined;
  }

  function getGenerationList(data) {
    return getDefined(
      data?.generations,
      data?.generation_results,
      data?.generations_data,
      data?.run?.generations,
    );
  }

  function getViewerConfig(data) {
    return getDefined(data?.metadata?.config, data?.config);
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

  function activateGenome(genomeJson, inputs) {
    const nodes = genomeJson.node_genes || [];
    const edges = (genomeJson.connection_genes || []).filter((edge) => edge.enabled !== false);
    const inputNodes = nodes.filter((n) => n.type === "input");
    const outputNodes = nodes.filter((n) => n.type === "output");
    const nodeLookup = new Map(nodes.map((n) => [Number(n.id), n]));
    const indegree = new Map(nodes.map((n) => [Number(n.id), 0]));
    const outgoing = new Map(nodes.map((n) => [Number(n.id), []]));
    const incoming = new Map(nodes.map((n) => [Number(n.id), []]));

    for (const edge of edges) {
      const inNode = Number(edge.in_node);
      const outNode = Number(edge.out_node);
      indegree.set(outNode, (indegree.get(outNode) || 0) + 1);
      outgoing.get(inNode)?.push(edge);
      incoming.get(outNode)?.push(edge);
    }

    const queue = [...indegree.entries()].filter(([, d]) => d === 0).map(([id]) => id);
    const topo = [];
    while (queue.length) {
      const current = queue.shift();
      topo.push(current);
      for (const edge of outgoing.get(current) || []) {
        const outNode = Number(edge.out_node);
        indegree.set(outNode, (indegree.get(outNode) || 0) - 1);
        if ((indegree.get(outNode) || 0) === 0) {
          queue.push(outNode);
        }
      }
    }

    for (const node of nodes) {
      const nodeId = Number(node.id);
      if (!topo.includes(nodeId)) {
        topo.push(nodeId);
      }
    }

    const values = new Map(nodes.map((n) => [Number(n.id), 0.0]));
    inputNodes.forEach((node, i) => values.set(Number(node.id), Number(inputs[i] ?? 0.0)));
    for (const nodeId of topo) {
      const node = nodeLookup.get(nodeId);
      if (!node || node.type === "input") {
        continue;
      }
      let weightedSum = Number(node.bias || 0.0);
      for (const edge of incoming.get(nodeId) || []) {
        weightedSum += Number(edge.weight || 0.0) * Number(values.get(Number(edge.in_node)) || 0.0);
      }
      values.set(nodeId, sigmoid(weightedSum));
    }
    return outputNodes.map((node) => Number(values.get(Number(node.id)) || 0.0));
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

  function advanceGeneration(delta) {
    if (!state.data) return;
    const total = state.data.generations.length;
    const next = (state.generationIndex + delta + total) % total;
    loadGeneration(next);
  }

  function loadGeneration(generationIdx) {
    const generation = state.generations[generationIdx];
    const config = state.config;
    const seed32 = Number(BigInt(generation.pipe_seed) & BigInt(0xffffffff));
    const rng = mulberry32(seed32);

    state.generationIndex = generationIdx;
    state.maxSteps = config.max_steps;
    state.step = 0;
    state.simEnded = false;
    state.bestScore = 0;
    state.stepAccumulator = 0;
    state.autoplayElapsedMs = 0;
    state.pipes = [createPipe(rng, config.first_pipe_x, config)];
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
    }));
    state.nextPipeIndexPerBird = state.birds.map(() => 0);
    state.trailHistory = state.birds.map(() => []);
    state.pipeRngState = rng;
    generationSlider.value = String(generationIdx);
    render();
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
    if (state.simEnded) return;

    if (state.pipes[state.pipes.length - 1].x < config.world_width - config.pipe_spacing) {
      state.pipes.push(createPipe(state.pipeRngState, config.new_pipe_x, config));
    }

    for (let i = 0; i < state.birds.length; i += 1) {
      const bird = state.birds[i];
      if (!bird.alive) continue;

      const inputs = normalizeInputs(bird, state.pipes, config);
      const output = activateGenome(bird.genomeJson, inputs)[0] || 0;
      const flap = decideFlap(output, state.data.metadata.flap_policy, bird.isFlapping, config);
      bird.isFlapping = flap;

      if (bird.flapCooldown > 0) bird.flapCooldown -= 1;
      if (flap && bird.flapCooldown === 0) {
        bird.velocity = config.jump_strength;
        bird.flapCooldown = config.flap_cooldown_frames;
      }

      bird.velocity = clamp(bird.velocity + config.gravity, config.velocity_min, config.velocity_max);
      bird.y += bird.velocity;

      if (bird.y < 0) {
        bird.y = 0;
        bird.velocity = 0;
      }
      if (bird.y >= config.world_height) {
        bird.alive = false;
      }

      for (const pipe of state.pipes) {
        const withinX = pipe.x <= config.bird_x && config.bird_x <= (pipe.x + pipe.width);
        if (withinX && !(pipe.top <= bird.y && bird.y <= pipe.bottom)) {
          bird.alive = false;
        }
      }

      while (state.nextPipeIndexPerBird[i] < state.pipes.length) {
        const pipe = state.pipes[state.nextPipeIndexPerBird[i]];
        if (config.bird_x <= (pipe.x + pipe.width)) break;
        bird.score += 1;
        state.nextPipeIndexPerBird[i] += 1;
      }

      state.bestScore = Math.max(state.bestScore, bird.score);
      if (state.showTrails) {
        const trail = state.trailHistory[i];
        trail.push({ x: config.bird_x, y: bird.y });
        if (trail.length > 120) trail.shift();
      }
    }

    for (const pipe of state.pipes) {
      pipe.x -= config.pipe_speed;
    }

    const countBefore = state.pipes.length;
    state.pipes = state.pipes.filter((pipe) => (pipe.x + pipe.width) > config.offscreen_pipe_right_threshold);
    const removed = countBefore - state.pipes.length;
    if (removed > 0) {
      state.nextPipeIndexPerBird = state.nextPipeIndexPerBird.map((idx) => Math.max(0, idx - removed));
    }

    state.step += 1;
    const aliveCount = state.birds.filter((bird) => bird.alive).length;
    if (aliveCount === 0 || state.step >= state.maxSteps) {
      state.simEnded = true;
    }
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
    statPlayback.textContent = `${state.autoplayIntervalMs} ms/gen`;
    statSeed.textContent = String(generation.pipe_seed);
  }

  function render() {
    const generation = state.generations?.[state.generationIndex];
    if (!generation) return;
    const config = state.config;

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
      .map((bird) => `#${bird.rank}: ${bird.score} (fit ${bird.exportedFitness.toFixed(1)}, μpipes ${bird.exportedPipesMean.toFixed(2)})${bird.alive ? "" : " ✖"}`)
      .join(" | ");
    rankingList.textContent = `Top scores: ${topFive}`;
  }

  function animate(timestamp) {
    if (!state.lastTimestamp) state.lastTimestamp = timestamp;
    const deltaMs = timestamp - state.lastTimestamp;
    state.lastTimestamp = timestamp;
    state.renderTimeMs += deltaMs;

    if (state.data) {
      state.stepAccumulator += (deltaMs / 1000) * 60;
      while (state.stepAccumulator >= 1) {
        state.stepAccumulator -= 1;
        stepSimulation();
      }

      if (state.playing) {
        state.autoplayElapsedMs += deltaMs;
        const interval = state.simEnded ? Math.min(450, state.autoplayIntervalMs) : state.autoplayIntervalMs;
        if (state.autoplayElapsedMs >= interval) {
          advanceGeneration(1);
        }
      }
    }

    render();
    requestAnimationFrame(animate);
  }

  function attachControls() {
    playPauseBtn.addEventListener("click", () => {
      state.playing = !state.playing;
      playPauseBtn.textContent = state.playing ? "Pause autoplay" : "Play autoplay";
    });

    prevGenBtn.addEventListener("click", () => advanceGeneration(-1));
    nextGenBtn.addEventListener("click", () => advanceGeneration(1));

    generationSlider.addEventListener("input", (event) => {
      if (!state.generations.length) return;
      loadGeneration(Number(event.target.value) || 0);
    });

    speedSlider.addEventListener("input", (event) => {
      state.autoplayIntervalMs = clamp(Number(event.target.value) || 1500, 500, 3000);
      state.autoplayElapsedMs = 0;
      statPlayback.textContent = `${state.autoplayIntervalMs} ms/gen`;
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
  }

  async function loadEvolution() {
    const url = "./evolution.json";
    state.evolutionUrl = url;
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        showError(`Failed to fetch ${url} (HTTP ${response.status})`);
        return;
      }

      let data;
      try {
        data = await response.json();
      } catch (parseError) {
        showError(`Failed to parse JSON from ${url}: ${parseError.message}`);
        return;
      }

      state.topLevelKeys = Object.keys(data || {});
      const generations = getGenerationList(data);
      if (!Array.isArray(generations) || generations.length === 0) {
        showError(
          `Could not find non-empty generations array. Tried keys: generations, generation_results, generations_data, run.generations. Found top-level keys: ${state.topLevelKeys.join(", ") || "(none)"}`,
        );
        return;
      }

      const viewerConfig = getViewerConfig(data);
      if (!viewerConfig) {
        showError(`Missing viewer config at metadata.config (or config). Top-level keys: ${state.topLevelKeys.join(", ") || "(none)"}`);
        return;
      }

      state.data = data;
      state.bestPipesAllTime = computeBestPipesAllTime(data);
      generationSlider.min = "0";
      generationSlider.max = String(data.generations.length - 1);
      setStatus(`Loaded ${data.generations.length} generations from evolution.json.`);
      loadGeneration(0);
    } catch (error) {
      showError(`Unexpected error while loading ${url}: ${error.message}`);
      console.error(error);
    }
  }

  attachControls();
  loadEvolution();
  requestAnimationFrame(animate);
})();
