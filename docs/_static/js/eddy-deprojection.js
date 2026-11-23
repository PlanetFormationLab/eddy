/* eddy · deprojection demo ------------------------------------------------
   <eddy-deprojection> — animated illustration of the annulus method:
   sweep the assumed projected rotation velocity and watch the deprojected
   spectra fall out of / into alignment.

   Synthetic data (a Gaussian line on an annulus of 40 azimuthal samples),
   generated in-browser with a fixed seed — an illustration of what eddy
   does, not a fit to real observations.

   attrs:  vtrue="400"   true projected rotation velocity, m/s
           nspec="40"    spectra around the annulus
           autoplay      start the sweep on load (default on)
   ------------------------------------------------------------------------ */
(function () {
  const C = {
    ink: '#1A1916', fg2: '#585548', fg3: '#7E7B70', hair: '#CFCCC0',
    pt: 'rgba(126,123,112,.42)', truth: '#ABA89B',
    model: '#1B93A2', dust: '#F26B0F', dustSoft: 'rgba(242,107,15,.16)',
    mono: "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace",
    body: 'Inter, system-ui, -apple-system, sans-serif',
  };
  const VMAX = 1000, SIGMA = 165, NOISE = 0.045, NCHAN = 70;

  // deterministic RNG so every reader (and every rebuild) sees the same data
  function rng(seed) {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }
  const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const line = (v) => Math.exp(-0.5 * (v / SIGMA) ** 2);

  class EddyDeprojection extends HTMLElement {
    connectedCallback() {
      if (this._built) { this._start(); return; }
      this._built = true;
      this.vtrue = +(this.getAttribute('vtrue') || 400);
      this.nspec = +(this.getAttribute('nspec') || 40);
      this.v = Math.round(this.vtrue * 0.45);
      this.playing = this.getAttribute('autoplay') !== 'false'
        && !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      this._dir = 1;
      this._build();
      this._data();
      this._curve();
      this._loop = this._loop.bind(this);
      this._draw();          // paint once, synchronously — motion is the only thing the clock owns
      this._start();
    }
    // rAF alone is unreliable inside embedded/offscreen preview frames, so the
    // sweep runs on a timer and rAF is only used to smooth it when available.
    _start() {
      this._stop();
      this._last = performance.now();
      this._timer = setInterval(this._loop, 1000 / 30);
    }
    _stop() { clearInterval(this._timer); this._timer = null; }
    disconnectedCallback() { /* the host may re-parent us; the loop parks itself */ }

    _build() {
      this.style.cssText = 'display:block';
      this.innerHTML = `
        <div class="ed-wrap" style="display:flex;flex-direction:column;gap:12px">
          <canvas class="ed-canvas" style="display:block;width:100%;height:auto"></canvas>
          <div class="ed-controls" style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
            <button class="ed-play" type="button" style="
              flex:none;display:inline-flex;align-items:center;gap:8px;
              font:500 13px ${C.body};color:${C.ink};background:#FFFFFF;
              border:1px solid ${C.hair};border-radius:8px;padding:7px 13px;cursor:pointer">
              <span class="ed-glyph" style="font-size:11px">❚❚</span><span class="ed-label">Pause</span>
            </button>
            <input class="ed-slider" type="range" min="0" max="${Math.round(this.vtrue * 2)}" step="1"
              value="${this.v}" aria-label="Assumed projected rotation velocity"
              style="flex:1 1 200px;min-width:160px;accent-color:${C.dust}">
            <output class="ed-read" style="flex:none;font:12px ${C.mono};color:${C.fg2};min-width:19ch"></output>
          </div>
        </div>`;
      this.canvas = this.querySelector('.ed-canvas');
      this.ctx = this.canvas.getContext('2d');
      this.read = this.querySelector('.ed-read');
      const btn = this.querySelector('.ed-play');
      btn.addEventListener('click', () => { this.playing = !this.playing; this._syncBtn(); });
      const sl = this.querySelector('.ed-slider');
      sl.addEventListener('input', () => { this.v = +sl.value; this.playing = false; this._syncBtn(); this._draw(); });
      this.slider = sl;
      this._syncBtn();
      new ResizeObserver(() => this._draw()).observe(this);
    }
    _syncBtn() {
      this.querySelector('.ed-glyph').textContent = this.playing ? '❚❚' : '▶';
      this.querySelector('.ed-label').textContent = this.playing ? 'Pause' : 'Play';
    }

    // annulus of spectra: each shifted by vtrue·cos(theta)
    _data() {
      const r = rng(20200214);
      this.spectra = [];
      for (let i = 0; i < this.nspec; i++) {
        const th = -Math.PI + (2 * Math.PI * (i + 0.5)) / this.nspec;
        const chan = [];
        for (let k = 0; k < NCHAN; k++) {
          const v = -VMAX + (2 * VMAX * k) / (NCHAN - 1);
          chan.push({ v, I: line(v - this.vtrue * Math.cos(th)) + NOISE * gauss(r) });
        }
        this.spectra.push({ cos: Math.cos(th), chan });
      }
    }
    // scatter about the stacked mean profile, as a function of assumed velocity
    _scatter(vAssumed) {
      const bins = 46, lo = -VMAX, hi = VMAX, w = (hi - lo) / bins;
      const sum = new Float64Array(bins), sq = new Float64Array(bins), n = new Float64Array(bins);
      for (const s of this.spectra) for (const c of s.chan) {
        const v = c.v - vAssumed * s.cos;
        const b = Math.floor((v - lo) / w);
        if (b < 0 || b >= bins) continue;
        sum[b] += c.I; sq[b] += c.I * c.I; n[b]++;
      }
      let acc = 0, m = 0;
      for (let b = 0; b < bins; b++) if (n[b] > 3) { acc += Math.sqrt(Math.max(0, sq[b] / n[b] - (sum[b] / n[b]) ** 2)); m++; }
      return m ? acc / m : 0;
    }
    _curve() {
      const max = Math.round(this.vtrue * 2), step = Math.max(4, Math.round(max / 90));
      this.curveXs = []; this.curveYs = [];
      for (let v = 0; v <= max; v += step) { this.curveXs.push(v); this.curveYs.push(this._scatter(v)); }
      this.curveMax = Math.max(...this.curveYs);
      this.curveMin = Math.min(...this.curveYs);
    }

    _loop() {
      if (!this.isConnected) { this._stop(); return; }
      if (!this.playing) { this._last = performance.now(); return; }
      const now = performance.now();
      const dt = Math.min(4, (now - (this._last || now)) / (1000 / 60));  // in 60 fps frames
      this._last = now;
      const max = +this.slider.max;
      // ease near the truth so the "snap" is legible, then drift back out
      const d = Math.abs(this.v - this.vtrue) / this.vtrue;
      const speed = (1.1 + 5.4 * Math.min(1, d)) * dt;
      this.v += this._dir * speed;
      if (this.v >= max) { this.v = max; this._dir = -1; }
      if (this.v <= 0) { this.v = 0; this._dir = 1; }
      this.slider.value = Math.round(this.v);
      this._draw();
    }

    _draw() {
      const cw = this.clientWidth || 720;
      const W = Math.max(360, cw), H = Math.round(W * 0.58);
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const cv = this.canvas, ctx = this.ctx;
      if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      cv.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const L = 58, R = W - 14, T = 16;
      const gap = 22, hB = 62, B = H - 34 - hB - gap, T2 = B + gap + 34, B2 = T2 + hB;
      const sx = (v) => L + ((v + VMAX) / (2 * VMAX)) * (R - L);
      const sy = (I) => B - ((I + 0.22) / 1.5) * (B - T);

      // ---------- top panel: deprojected spectra ----------
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.25;
      ctx.strokeRect(L, T, R - L, B - T);
      ctx.font = `11px ${C.mono}`; ctx.fillStyle = C.fg3;
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const I of [0, 0.5, 1]) {
        ctx.fillText(I.toFixed(1), L - 8, sy(I));
        ctx.strokeStyle = 'rgba(26,25,22,.10)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(L + 1, sy(I)); ctx.lineTo(R - 1, sy(I)); ctx.stroke();
      }
      ctx.save();
      ctx.translate(16, (T + B) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.fillStyle = C.fg2; ctx.font = `12px ${C.body}`;
      ctx.fillText('Intensity', 0, 0);
      ctx.restore();

      ctx.save();
      ctx.beginPath(); ctx.rect(L + 1, T + 1, R - L - 2, B - T - 2); ctx.clip();
      // deprojected points
      ctx.fillStyle = C.pt;
      for (const s of this.spectra) {
        const sh = this.v * s.cos;
        for (const c of s.chan) {
          const x = sx(c.v - sh);
          ctx.beginPath(); ctx.arc(x, sy(c.I), 1.5, 0, 6.284); ctx.fill();
        }
      }
      // truth
      ctx.strokeStyle = C.truth; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let k = 0; k <= 160; k++) {
        const v = -VMAX + (2 * VMAX * k) / 160;
        k ? ctx.lineTo(sx(v), sy(line(v))) : ctx.moveTo(sx(v), sy(line(v)));
      }
      ctx.stroke();
      // stacked mean of the deprojected points — the profile eddy fits
      const bins = 46, w = (2 * VMAX) / bins;
      const sum = new Float64Array(bins), n = new Float64Array(bins);
      for (const s of this.spectra) for (const c of s.chan) {
        const b = Math.floor((c.v - this.v * s.cos + VMAX) / w);
        if (b >= 0 && b < bins) { sum[b] += c.I; n[b]++; }
      }
      ctx.strokeStyle = C.model; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
      ctx.beginPath();
      let started = false;
      for (let b = 0; b < bins; b++) {
        if (n[b] < 4) continue;
        const x = sx(-VMAX + (b + 0.5) * w), y = sy(sum[b] / n[b]);
        started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
      }
      ctx.stroke();
      ctx.restore();

      // legend
      ctx.font = `11px ${C.mono}`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      const ly = T + 16;
      ctx.fillStyle = C.pt;
      for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.arc(L + 14 + i * 5, ly, 1.6, 0, 6.284); ctx.fill(); }
      ctx.fillStyle = C.fg3; ctx.fillText('deprojected spectra', L + 32, ly);
      ctx.strokeStyle = C.model; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(L + 14, ly + 18); ctx.lineTo(L + 30, ly + 18); ctx.stroke();
      ctx.fillStyle = C.fg3; ctx.fillText('stacked profile', L + 36, ly + 18);
      ctx.strokeStyle = C.truth; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(L + 14, ly + 36); ctx.lineTo(L + 30, ly + 36); ctx.stroke();
      ctx.fillStyle = C.fg3; ctx.fillText('truth', L + 36, ly + 36);

      // aligned / smeared verdict
      const off = Math.abs(this.v - this.vtrue);
      const snapped = off < 26;
      ctx.textAlign = 'right'; ctx.font = `11px ${C.mono}`;
      ctx.fillStyle = snapped ? C.model : C.dust;
      ctx.fillText(snapped ? 'ALIGNED' : 'SMEARED', R - 12, ly);

      // ---------- x axis (shared label under the top panel) ----------
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = C.fg3;
      ctx.font = `11px ${C.mono}`;
      for (const v of [-1000, -500, 0, 500, 1000]) {
        ctx.beginPath(); ctx.moveTo(sx(v), B); ctx.lineTo(sx(v), B + 5); ctx.stroke();
        ctx.fillText(String(v), sx(v), B + 8);
      }
      ctx.fillStyle = C.fg2; ctx.font = `12px ${C.body}`;
      ctx.fillText('Velocity  (m s⁻¹)', (L + R) / 2, B + 24);

      // ---------- bottom panel: scatter vs assumed velocity ----------
      const px = (v) => L + (v / +this.slider.max) * (R - L);
      const lo = this.curveMin - (this.curveMax - this.curveMin) * 0.18;
      const hi = this.curveMax + (this.curveMax - this.curveMin) * 0.12;
      const py = (s) => B2 - ((s - lo) / (hi - lo)) * (B2 - T2);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.25;
      ctx.strokeRect(L, T2, R - L, B2 - T2);
      // truth marker
      ctx.fillStyle = C.dustSoft;
      ctx.fillRect(px(this.vtrue) - 1.5, T2 + 1, 3, B2 - T2 - 2);
      ctx.strokeStyle = C.fg3; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < this.curveXs.length; i++) {
        const x = px(this.curveXs[i]), y = py(this.curveYs[i]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
      // the live position
      const cy = py(this._scatter(this.v));
      ctx.strokeStyle = C.dust; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px(this.v), T2 + 1); ctx.lineTo(px(this.v), B2 - 1); ctx.stroke();
      ctx.fillStyle = C.dust;
      ctx.beginPath(); ctx.arc(px(this.v), cy, 4, 0, 6.284); ctx.fill();
      ctx.font = `11px ${C.mono}`; ctx.fillStyle = C.fg3;
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('scatter about the stacked profile', L + 12, T2 - 6);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillStyle = C.fg2; ctx.font = `12px ${C.body}`;
      ctx.fillText('Assumed projected rotation velocity  (m s⁻¹)', (L + R) / 2, B2 + 7);

      this.read.textContent = `v = ${String(Math.round(this.v)).padStart(4, ' ')} m s⁻¹  Δ ${off < 1 ? '0' : (this.v > this.vtrue ? '+' : '−') + Math.round(off)}`;
    }
  }
  if (!customElements.get('eddy-deprojection')) customElements.define('eddy-deprojection', EddyDeprojection);
})();
