(() => {
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const speedSlider = document.getElementById("speedSlider");
  const speedValue = document.getElementById("speedValue");
  const statusEl = document.getElementById("status");

  const state = {
    frames: [],
    fps: 60,
    speed: 1,
    playing: true,
    frameIndex: 0,
    accumulator: 0,
    lastTimestamp: 0,
  };

  function drawFrame(frame) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Pipes
    ctx.fillStyle = "#2f5f2f";
    for (const pipe of frame.pipes || []) {
      const pipeWidth = 80;
      ctx.fillRect(pipe.x, 0, pipeWidth, pipe.top);
      ctx.fillRect(pipe.x, pipe.bottom, pipeWidth, canvas.height - pipe.bottom);
    }

    // Bird
    const birdX = 100;
    const birdY = frame.bird?.y ?? 250;
    ctx.fillStyle = "#ffd54f";
    ctx.beginPath();
    ctx.arc(birdX, birdY, 14, 0, Math.PI * 2);
    ctx.fill();

    // Overlay text
    const generation = frame.generation?.generation ?? "n/a";
    const score = frame.score ?? 0;
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(8, 8, 230, 58);
    ctx.fillStyle = "#fff";
    ctx.font = "18px Arial";
    ctx.fillText(`Score: ${score}`, 16, 32);
    ctx.fillText(`Generation: ${generation}`, 16, 56);
  }

  function animate(timestamp) {
    if (!state.lastTimestamp) {
      state.lastTimestamp = timestamp;
    }

    const deltaMs = timestamp - state.lastTimestamp;
    state.lastTimestamp = timestamp;

    if (state.playing && state.frames.length > 0) {
      const dtFrames = (deltaMs / 1000) * state.fps * state.speed;
      state.accumulator += dtFrames;
      while (state.accumulator >= 1) {
        state.accumulator -= 1;
        state.frameIndex = Math.min(state.frameIndex + 1, state.frames.length - 1);
        if (state.frameIndex >= state.frames.length - 1) {
          state.playing = false;
          playPauseBtn.textContent = "Play";
        }
      }
    }

    if (state.frames.length > 0) {
      drawFrame(state.frames[state.frameIndex]);
    }

    requestAnimationFrame(animate);
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function attachControls() {
    playPauseBtn.addEventListener("click", () => {
      if (!state.frames.length) {
        return;
      }
      if (state.frameIndex >= state.frames.length - 1) {
        state.frameIndex = 0;
      }
      state.playing = !state.playing;
      playPauseBtn.textContent = state.playing ? "Pause" : "Play";
    });

    speedSlider.addEventListener("input", (event) => {
      const raw = Number(event.target.value);
      state.speed = Number.isFinite(raw) && raw > 0 ? raw : 1;
      speedValue.textContent = `${state.speed.toFixed(2)}x`;
    });
  }

  async function loadSimulation() {
    try {
      const response = await fetch("simulation.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      state.frames = data.frames || [];
      state.fps = Math.round(1 / (data.meta?.dt || 1 / 60));
      setStatus(`Loaded ${state.frames.length} frames.`);
      if (state.frames.length > 0) {
        drawFrame(state.frames[0]);
      }
    } catch (error) {
      setStatus(`Failed to load web/simulation.json: ${error}`);
      console.error(error);
    }
  }

  attachControls();
  loadSimulation();
  requestAnimationFrame(animate);
})();
