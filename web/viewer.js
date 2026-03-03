(() => {
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const generationSlider = document.getElementById("generationSlider");
  const generationValue = document.getElementById("generationValue");
  const generationMax = document.getElementById("generationMax");
  const speedSlider = document.getElementById("speedSlider");
  const speedValue = document.getElementById("speedValue");
  const trailToggle = document.getElementById("trailToggle");
  const statusEl = document.getElementById("status");
  const rankingList = document.getElementById("rankingList");

  const state = {
    data: null,
    generationIndex: 0,
    speed: 1,
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
    trailHistory: [],
    step: 0,
  };

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

  function loadGeneration(generationIdx) {
    const generation = state.data.generations[generationIdx];
    const config = state.data.metadata.config;
    const seed32 = Number(BigInt(generation.pipe_seed) & BigInt(0xffffffff));
    const rng = mulberry32(seed32);

    state.generationIndex = generationIdx;
    state.maxSteps = config.max_steps;
    state.step = 0;
    state.simEnded = false;
    state.bestScore = 0;
    state.stepAccumulator = 0;
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
      color: `hsla(${Math.round((i * 360) / Math.max(1, generation.genomes.length))}, 85%, 52%, 0.65)`,
    }));
    state.nextPipeIndexPerBird = state.birds.map(() => 0);
    state.trailHistory = state.birds.map(() => []);
    state.pipeRngState = rng;
    generationSlider.value = String(generationIdx);
    generationValue.textContent = String(generationIdx);
    render();
  }

  function stepSimulation() {
    const generation = state.data.generations[state.generationIndex];
    const config = state.data.metadata.config;
    if (state.simEnded) {
      return;
    }

    if (state.pipes[state.pipes.length - 1].x < config.world_width - config.pipe_spacing) {
      state.pipes.push(createPipe(state.pipeRngState, config.new_pipe_x, config));
    }

    for (let i = 0; i < state.birds.length; i += 1) {
      const bird = state.birds[i];
      if (!bird.alive) continue;

      const inputs = normalizeInputs(bird, state.pipes, config);
      const output = activateGenome(bird.genomeJson, inputs)[0] || 0;
      let flap = decideFlap(output, state.data.metadata.flap_policy, bird.isFlapping, config);
      bird.isFlapping = flap;

      if (bird.flapCooldown > 0) {
        bird.flapCooldown -= 1;
      }
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
        if (config.bird_x <= (pipe.x + pipe.width)) {
          break;
        }
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
      if (state.playing) {
        const next = (state.generationIndex + 1) % state.data.generations.length;
        loadGeneration(next);
      }
    }
  }

  function render() {
    const generation = state.data?.generations?.[state.generationIndex];
    if (!generation) return;
    const config = state.data.metadata.config;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#2f5f2f";
    for (const pipe of state.pipes) {
      ctx.fillRect(pipe.x, 0, pipe.width, pipe.top);
      ctx.fillRect(pipe.x, pipe.bottom, pipe.width, canvas.height - pipe.bottom);
    }

    state.birds.forEach((bird, index) => {
      if (!bird.alive && !state.showTrails) return;
      if (state.showTrails) {
        ctx.strokeStyle = bird.color;
        ctx.lineWidth = bird.rank === 1 ? 2.2 : 1;
        ctx.beginPath();
        for (let i = 0; i < state.trailHistory[index].length; i += 1) {
          const p = state.trailHistory[index][i];
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }

      ctx.fillStyle = bird.color;
      ctx.beginPath();
      ctx.arc(config.bird_x, bird.y, 9, 0, Math.PI * 2);
      ctx.fill();
      if (bird.rank === 1) {
        ctx.strokeStyle = "#ffd54f";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(config.bird_x, bird.y, 12, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    const aliveCount = state.birds.filter((bird) => bird.alive).length;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(8, 8, 285, 90);
    ctx.fillStyle = "#fff";
    ctx.font = "16px Arial";
    ctx.fillText(`Generation: ${state.generationIndex}`, 16, 30);
    ctx.fillText(`Alive: ${aliveCount}/${state.birds.length}`, 16, 52);
    ctx.fillText(`Best score: ${state.bestScore}`, 16, 74);
    ctx.fillText(`Step: ${state.step}/${state.maxSteps}`, 16, 94);

    const topFive = [...state.birds]
      .sort((a, b) => (a.rank - b.rank))
      .slice(0, 5)
      .map((bird) => `#${bird.rank}: ${bird.score}${bird.alive ? "" : " ✖"}`)
      .join(" | ");
    rankingList.textContent = `Scores: ${topFive}`;
  }

  function animate(timestamp) {
    if (!state.lastTimestamp) state.lastTimestamp = timestamp;
    const deltaMs = timestamp - state.lastTimestamp;
    state.lastTimestamp = timestamp;
    if (state.playing && state.data) {
      state.stepAccumulator += (deltaMs / 1000) * 60 * state.speed;
      while (state.stepAccumulator >= 1) {
        state.stepAccumulator -= 1;
        stepSimulation();
      }
    }
    render();
    requestAnimationFrame(animate);
  }

  function attachControls() {
    playPauseBtn.addEventListener("click", () => {
      state.playing = !state.playing;
      playPauseBtn.textContent = state.playing ? "Pause" : "Play";
    });

    generationSlider.addEventListener("input", (event) => {
      if (!state.data) return;
      loadGeneration(Number(event.target.value) || 0);
    });

    speedSlider.addEventListener("input", (event) => {
      state.speed = Math.max(0.1, Number(event.target.value) || 1);
      speedValue.textContent = `${state.speed.toFixed(2)}x`;
    });

    trailToggle.addEventListener("change", (event) => {
      state.showTrails = Boolean(event.target.checked);
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
      generationSlider.min = "0";
      generationSlider.max = String(data.generations.length - 1);
      generationMax.textContent = String(data.generations.length - 1);
      setStatus(`Loaded ${data.generations.length} generations.`);
      loadGeneration(0);
    } catch (error) {
      setStatus(`Failed to load evolution.json: ${error.message}`);
      console.error(error);
    }
  }

  attachControls();
  loadEvolution();
  requestAnimationFrame(animate);
})();
