/* PathSnap evaluator UI (Laptop 2).
 *
 * Talks ONLY to the Laptop 1 FastAPI backend:
 *   GET  /health
 *   GET  /trips
 *   POST /predict   {"trip_id": "<id>"}
 *
 * Nothing is hardcoded or cached: every result shown comes from a live request
 * (fetch with cache: "no-store"). The session history only keeps trip_id +
 * accuracy for comparison; clicking it re-selects the trip, it never redraws
 * an old result. Ground truth is never requested or shown.
 *
 * Mock mode (OFF by default, never persisted) reads ui/mock/*.json instead.
 */
'use strict';

(function () {
  // ---------------------------------------------------------------- config
  const DEFAULT_API = 'http://192.168.1.10:8000';
  const LS_API_KEY = 'pathsnap.apiBase';
  const MAX_GPS_DRAWN = 3000;
  const TIMEOUT_MS = { health: 5000, trips: 15000, predict: 60000 };
  const MOCK_FILES = {
    'GET /health': 'mock/health.json',
    'GET /trips': 'mock/trips.json',
    'POST /predict': 'mock/predict_sample.json',
  };
  const COLORS = {
    gps: '#2563eb',
    pred: '#16a34a',
    clf: '#f97316',
    start: '#15803d',
    end: '#dc2626',
  };

  // ---------------------------------------------------------------- dom
  const $ = (id) => document.getElementById(id);
  const els = {
    apiBase: $('apiBase'),
    testBtn: $('testBtn'),
    connStatus: $('connStatus'),
    mockToggle: $('mockToggle'),
    mockBanner: $('mockBanner'),
    fileWarning: $('fileWarning'),
    tripList: $('tripList'),
    reloadTripsBtn: $('reloadTripsBtn'),
    runBtn: $('runBtn'),
    runHint: $('runHint'),
    historyList: $('historyList'),
    historyEmpty: $('historyEmpty'),
    clearHistoryBtn: $('clearHistoryBtn'),
    spinner: $('spinner'),
    spinnerText: $('spinnerText'),
    message: $('message'),
    resultsEmpty: $('resultsEmpty'),
    resultsBody: $('resultsBody'),
  };

  // ---------------------------------------------------------------- state
  const state = {
    mock: false,
    trips: [],
    selected: null,       // trip_id string
    running: false,
    reqSeq: 0,            // increments per request so stale responses are ignored
    tripsSeq: 0,
    history: [],          // [{tripId, hmm, clf, requestId, at, mock}]
    spinnerTimer: null,
  };

  // ---------------------------------------------------------------- helpers
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  const pct = (x) => (isNum(x) ? (x * 100).toFixed(1) + '%' : '—');
  const fixed3 = (x) => (isNum(x) ? x.toFixed(3) : '—');
  const ms = (x) => (isNum(x) ? Math.round(x).toLocaleString() + ' ms' : '—');
  const int = (x) => (isNum(x) ? Math.round(x).toLocaleString() : '—');
  const sleep = (t) => new Promise((r) => setTimeout(r, t));

  function fmtDuration(s) {
    if (!isNum(s) || s < 0) return '—';
    s = Math.round(s);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h} h ${String(m).padStart(2, '0')} min`;
    if (m > 0) return `${m} min ${String(sec).padStart(2, '0')} s`;
    return `${sec} s`;
  }

  function fmtTime(d) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  // ---------------------------------------------------------------- storage (localStorage can throw)
  function loadApiBase() {
    try {
      return localStorage.getItem(LS_API_KEY) || DEFAULT_API;
    } catch (e) {
      return DEFAULT_API;
    }
  }
  function saveApiBase(v) {
    try {
      localStorage.setItem(LS_API_KEY, v);
    } catch (e) {
      /* storage unavailable: keep working with the in-page value */
    }
  }

  // ---------------------------------------------------------------- API layer
  class ApiError extends Error {
    constructor(kind, message, hints, status) {
      super(message);
      this.kind = kind;       // config | network | timeout | http | parse | mock
      this.hints = hints || [];
      this.status = status || null;
    }
  }

  const NETWORK_HINTS = [
    'Is the IP right? On Laptop 1 run ipconfig → “IPv4 Address”.',
    'Both laptops on the same Wi-Fi / phone hotspot (college Wi-Fi often blocks device-to-device traffic).',
    'API started with --host 0.0.0.0 --port 8000 (not 127.0.0.1).',
    'Windows Firewall on Laptop 1 allows inbound TCP 8000.',
    'Open <API URL>/health directly in this browser — it must show JSON.',
  ];

  function normalizeBase(raw) {
    let v = String(raw || '').trim().replace(/\/+$/, '');
    if (!v) {
      throw new ApiError('config', 'Enter the API base URL first, e.g. ' + DEFAULT_API + '.');
    }
    if (!/^https?:\/\//i.test(v)) v = 'http://' + v;
    try {
      const u = new URL(v);
      if (!u.hostname) throw new Error('no host');
    } catch (e) {
      throw new ApiError('config', `“${raw}” is not a valid URL. Use the form ${DEFAULT_API}.`);
    }
    return v;
  }

  function httpError(status, detail, path) {
    const d = detail ? `: ${detail}` : '';
    if (status === 404 && path === '/predict') {
      return new ApiError('http', `Unknown trip (404)${d}`, [
        'The API does not know this trip_id. Click “Reload” on the trips list and pick again.',
      ], status);
    }
    if (status === 404) {
      return new ApiError('http', `Endpoint ${path} not found (404)${d}`, [
        'Is this URL really the PathSnap API (port 8000)? Check the API base URL.',
      ], status);
    }
    if (status === 503) {
      return new ApiError('http', `Model not loaded on Laptop 1 (503)${d}`, [
        'The API is up but the model is not ready. Wait for it to finish loading, or restart the API on Laptop 1 and check its terminal.',
      ], status);
    }
    if (status >= 500) {
      return new ApiError('http', `Server error on Laptop 1 (${status})${d}`, [
        'Something failed inside the API. Check the API terminal on Laptop 1 for the traceback, then try again.',
      ], status);
    }
    return new ApiError('http', `Request failed (HTTP ${status})${d}`, [], status);
  }

  async function mockRequest(method, path) {
    const file = MOCK_FILES[`${method} ${path}`];
    if (!file) throw new ApiError('mock', `No mock file for ${method} ${path}.`);
    let res;
    try {
      res = await fetch(file, { cache: 'no-store' });
    } catch (e) {
      throw new ApiError('mock', `Could not read ${file}.`, [
        'Serve the folder over HTTP: cd ui && python -m http.server 5500',
      ]);
    }
    if (!res.ok) throw new ApiError('mock', `Could not read ${file} (HTTP ${res.status}).`);
    if (path === '/predict') await sleep(600); // let the spinner be visible in mock mode
    try {
      return await res.json();
    } catch (e) {
      throw new ApiError('mock', `${file} is not valid JSON.`);
    }
  }

  async function api(method, path, body, timeoutMs) {
    if (state.mock) return mockRequest(method, path);

    const base = normalizeBase(els.apiBase.value);
    const url = base + path;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    let res;
    let text = '';
    try {
      const opts = { method, cache: 'no-store', signal: ctrl.signal };
      if (body !== undefined) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
      }
      res = await fetch(url, opts);
      text = await res.text();
    } catch (e) {
      if (e && e.name === 'AbortError') {
        throw new ApiError('timeout', `No response from ${base} after ${timeoutMs / 1000} s (${method} ${path}).`, [
          'Laptop 1 may be busy or unreachable. Check the API terminal, then try again.',
          ...NETWORK_HINTS,
        ]);
      }
      throw new ApiError('network', `Cannot reach the API at ${base}.`, NETWORK_HINTS);
    } finally {
      clearTimeout(timer);
    }

    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (e) {
        data = null;
      }
    }
    if (!res.ok) {
      let detail = '';
      if (data && data.detail !== undefined) {
        detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      } else if (text) {
        detail = text.slice(0, 200);
      }
      throw httpError(res.status, detail, path);
    }
    if (data === null || typeof data !== 'object') {
      throw new ApiError('parse', `The API returned something that is not JSON for ${method} ${path}.`, [
        'Is this URL really the PathSnap API?',
      ]);
    }
    return data;
  }

  // ---------------------------------------------------------------- messages
  function showMessage(kind, title, hints) {
    const m = els.message;
    m.replaceChildren();
    m.className = 'message ' + kind;
    m.appendChild(el('div', 'title', title));
    if (hints && hints.length) {
      const ul = el('ul');
      hints.forEach((h) => ul.appendChild(el('li', null, h)));
      m.appendChild(ul);
    }
    m.hidden = false;
  }
  function clearMessage() {
    els.message.hidden = true;
    els.message.replaceChildren();
  }
  function errorTitle(e) {
    return e instanceof ApiError ? e.message : 'Unexpected error: ' + (e && e.message ? e.message : e);
  }

  // ---------------------------------------------------------------- connection test
  function setStatus(kind, text, title) {
    els.connStatus.className = 'status status-' + kind;
    els.connStatus.textContent = text;
    els.connStatus.title = title || text;
  }

  async function testConnection() {
    if (!state.mock) {
      try {
        const base = normalizeBase(els.apiBase.value);
        els.apiBase.value = base;
        saveApiBase(base);
      } catch (e) {
        setStatus('err', e.message);
        return false;
      }
    }
    setStatus('busy', 'Testing…');
    els.testBtn.disabled = true;
    try {
      const h = await api('GET', '/health', undefined, TIMEOUT_MS.health);
      const ok = h.status === 'ok';
      const parts = [];
      parts.push(ok ? 'Connected' : `Status: ${h.status}`);
      if (h.model) parts.push(h.model);
      if (h.hmm === false) parts.push('HMM off');
      if (isNum(h.n_trips)) parts.push(`${h.n_trips} trips`);
      if (h.version) parts.push('v' + h.version);
      const text = (state.mock ? 'MOCK · ' : '') + parts.join(' · ');
      setStatus(state.mock ? 'mock' : ok ? 'ok' : 'err', text);
      return ok;
    } catch (e) {
      setStatus('err', 'Not connected — ' + (e.status ? 'HTTP ' + e.status : e.kind), errorTitle(e));
      if (!state.mock) showMessage('err', errorTitle(e), e.hints);
      return false;
    } finally {
      els.testBtn.disabled = false;
    }
  }

  // ---------------------------------------------------------------- trips
  function listMessage(text, isErr, withRetry) {
    els.tripList.replaceChildren();
    const li = el('li', 'list-msg' + (isErr ? ' err' : ''));
    li.appendChild(el('div', null, text));
    if (withRetry) {
      const b = el('button', null, 'Retry');
      b.type = 'button';
      b.addEventListener('click', loadTrips);
      li.appendChild(b);
    }
    els.tripList.appendChild(li);
  }

  async function loadTrips() {
    const seq = ++state.tripsSeq;
    state.trips = [];
    listMessage('Loading trips…');
    updateRunButton();
    try {
      const data = await api('GET', '/trips', undefined, TIMEOUT_MS.trips);
      if (seq !== state.tripsSeq) return;
      const trips = Array.isArray(data.trips) ? data.trips : null;
      if (!trips) throw new ApiError('parse', 'GET /trips response has no "trips" list.');
      state.trips = trips
        .filter((t) => t && t.trip_id !== undefined && t.trip_id !== null)
        .map((t) => ({ ...t, trip_id: String(t.trip_id) }));
      if (!state.trips.some((t) => t.trip_id === state.selected)) state.selected = null;
      renderTrips();
      if (!state.trips.length) listMessage('The API returned no trips.', true, true);
    } catch (e) {
      if (seq !== state.tripsSeq) return;
      listMessage('Could not load trips — ' + errorTitle(e), true, true);
    }
    updateRunButton();
  }

  function renderTrips() {
    els.tripList.replaceChildren();
    const doneIds = new Set(state.history.map((h) => h.tripId));
    state.trips.forEach((t) => {
      const li = el('li');
      const b = el('button', 'trip');
      b.type = 'button';
      b.dataset.tripId = t.trip_id;
      b.setAttribute('aria-pressed', String(t.trip_id === state.selected));
      const idCell = el('span', 'tid', t.trip_id);
      if (doneIds.has(t.trip_id)) {
        const d = el('span', 'done', '✓');
        d.title = 'Already run this session';
        idCell.appendChild(d);
      }
      b.appendChild(idCell);
      b.appendChild(el('span', 'num', isNum(t.n_points) ? t.n_points.toLocaleString() + ' pts' : '—'));
      b.appendChild(el('span', 'dur', fmtDuration(t.duration_s)));
      b.addEventListener('click', () => selectTrip(t.trip_id));
      b.addEventListener('dblclick', () => {
        selectTrip(t.trip_id);
        runPrediction();
      });
      li.appendChild(b);
      els.tripList.appendChild(li);
    });
  }

  function selectTrip(id) {
    state.selected = id;
    els.tripList.querySelectorAll('.trip').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.tripId === id));
    });
    updateRunButton();
  }

  function updateRunButton() {
    const can = !!state.selected && !state.running;
    els.runBtn.disabled = !can;
    if (state.running) {
      els.runBtn.textContent = 'Running…';
      els.runHint.textContent = 'Waiting for Laptop 1 (usually 1–10 s).';
    } else if (state.selected) {
      els.runBtn.textContent = `Run trip ${state.selected}`;
      els.runHint.textContent = state.mock
        ? 'Mock mode: returns the same sample for every trip.'
        : 'Sends POST /predict to the API.';
    } else {
      els.runBtn.textContent = 'Run';
      els.runHint.textContent = 'Select a trip, then click Run (or double-click a trip).';
    }
  }

  // ---------------------------------------------------------------- spinner
  function setRunning(on, tripId) {
    state.running = on;
    els.spinner.hidden = !on;
    clearInterval(state.spinnerTimer);
    if (on) {
      const t0 = Date.now();
      const label = state.mock ? 'Loading mock sample' : `Running model on Laptop 1 — trip ${tripId}`;
      els.spinnerText.textContent = label + '…';
      state.spinnerTimer = setInterval(() => {
        els.spinnerText.textContent = `${label}… ${((Date.now() - t0) / 1000).toFixed(0)} s`;
      }, 500);
    }
    els.tripList.querySelectorAll('.trip').forEach((b) => (b.disabled = on));
    els.mockToggle.disabled = on;
    els.reloadTripsBtn.disabled = on;
    updateRunButton();
  }

  // ---------------------------------------------------------------- predict
  async function runPrediction() {
    if (!state.selected || state.running) return;
    const tripId = state.selected;
    const seq = ++state.reqSeq;
    clearMessage();
    setRunning(true, tripId);
    const t0 = performance.now();
    try {
      const data = await api('POST', '/predict', { trip_id: tripId }, TIMEOUT_MS.predict);
      const rtt = performance.now() - t0;
      if (seq !== state.reqSeq) return;
      if (!data || typeof data !== 'object' || !data.metrics) {
        throw new ApiError('parse', 'POST /predict response has no "metrics" object.');
      }
      const drawInfo = drawResult(data);
      renderResults(data, rtt);
      addHistory(data, tripId);

      const notes = [];
      if (state.mock && String(data.trip_id) !== tripId) {
        notes.push(`Mock mode always returns the sample for trip ${data.trip_id}, whichever trip you select.`);
      } else if (!state.mock && data.trip_id !== undefined && String(data.trip_id) !== tripId) {
        notes.push(`You asked for trip ${tripId} but the API answered for trip ${data.trip_id}.`);
      }
      if (drawInfo.warnings.length) notes.push(...drawInfo.warnings);
      if (notes.length) showMessage(state.mock ? 'warn' : 'warn', state.mock ? 'Mock data' : 'Note', notes);
    } catch (e) {
      if (seq !== state.reqSeq) return;
      clearMap();
      showEmptyResults();
      showMessage('err', `Trip ${tripId}: ${errorTitle(e)}`, e.hints);
    } finally {
      if (seq === state.reqSeq) setRunning(false);
    }
  }

  // ---------------------------------------------------------------- map
  let map = null;
  const layers = {};

  function initMap() {
    if (typeof window.L === 'undefined') {
      const box = $('map');
      box.replaceChildren(el('div', 'map-fallback',
        'Map library (Leaflet) failed to load from both the CDN and ui/vendor/leaflet/. Check that the vendor folder exists. Metrics still work.'));
      return;
    }
    map = L.map('map', { preferCanvas: true, zoomControl: true }).setView([51.0, 10.0], 5);
    let tileFails = 0;
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    })
      .on('tileerror', () => {
        if (++tileFails === 3) {
          const n = document.getElementById('tileNote');
          if (n) n.hidden = false;
        }
      })
      .on('tileload', () => {
        const n = document.getElementById('tileNote');
        if (n) n.hidden = true;
        tileFails = 0;
      })
      .addTo(map);

    // Panes fix the draw order: GPS dots < predicted route < classifier route (dashed on top) < start/end.
    [['gpsPane', 410], ['predPane', 420], ['clfPane', 430], ['endsPane', 440]].forEach(([name, z]) => {
      map.createPane(name).style.zIndex = z;
    });

    layers.gps = L.layerGroup().addTo(map);
    layers.pred = L.layerGroup().addTo(map);
    layers.clf = L.layerGroup(); // off by default
    layers.ends = L.layerGroup().addTo(map);

    const sw = (cls) => `<span class="sw ${cls}"></span>`;
    L.control.layers(null, {
      [`${sw('sw-gps')}Raw GPS observations`]: layers.gps,
      [`${sw('sw-pred')}Predicted route (classifier + HMM)`]: layers.pred,
      [`${sw('sw-clf')}Classifier-only route`]: layers.clf,
      [`${sw('sw-start')}Start / end`]: layers.ends,
    }, { collapsed: false, position: 'topright' }).addTo(map);

    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function () {
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML =
        '<div class="row"><span class="sw sw-gps"></span>Raw GPS observation</div>' +
        '<div class="row"><span class="sw sw-pred"></span>Predicted route (classifier + HMM)</div>' +
        '<div class="row"><span class="sw sw-clf"></span>Classifier-only route</div>' +
        '<div class="row"><span class="sw sw-start"></span>Start</div>' +
        '<div class="row"><span class="sw sw-end"></span>End</div>' +
        '<div class="note" id="legendNote" hidden></div>' +
        '<div class="note warn" id="tileNote" hidden>Street map tiles can’t load (Laptop 2 has no internet). Routes and metrics are still live from the API.</div>';
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    legend.addTo(map);

    window.addEventListener('resize', () => map.invalidateSize());
  }

  function setLegendNote(text) {
    const n = document.getElementById('legendNote');
    if (!n) return;
    n.textContent = text || '';
    n.hidden = !text;
  }

  function clearMap() {
    if (!map) return;
    Object.values(layers).forEach((g) => g.clearLayers());
    setLegendNote('');
  }

  function validPt(p) {
    return Array.isArray(p) && p.length >= 2 &&
      isNum(p[0]) && isNum(p[1]) && Math.abs(p[0]) <= 90 && Math.abs(p[1]) <= 180;
  }

  // Contract: list of segments, each a list of [lat, lon]. Tolerate a flat list of points too.
  function cleanSegments(segs) {
    if (!Array.isArray(segs) || !segs.length) return [];
    if (validPt(segs[0])) segs = [segs];
    return segs
      .map((s) => (Array.isArray(s) ? s.filter(validPt) : []))
      .filter((s) => s.length >= 2);
  }

  function validBbox(b) {
    return Array.isArray(b) && b.length === 2 && validPt(b[0]) && validPt(b[1]);
  }

  function drawResult(data) {
    const warnings = [];
    const gpsRaw = Array.isArray(data.gps) ? data.gps : [];
    const gps = gpsRaw.filter(validPt);
    const pred = cleanSegments(data.predicted_route);
    const clf = cleanSegments(data.classifier_route);
    if (gpsRaw.length !== gps.length) warnings.push(`${gpsRaw.length - gps.length} invalid GPS coordinates were skipped.`);
    if (!pred.length) warnings.push('The API returned no predicted route segments.');

    if (!map) return { warnings };
    clearMap();

    // Raw GPS: every k-th point when > MAX_GPS_DRAWN, always including the last one.
    const k = gps.length > MAX_GPS_DRAWN ? Math.ceil(gps.length / MAX_GPS_DRAWN) : 1;
    const dotStyle = {
      pane: 'gpsPane', radius: 2, stroke: false, fillColor: COLORS.gps, fillOpacity: 0.75, interactive: false,
    };
    for (let i = 0; i < gps.length; i += k) L.circleMarker(gps[i], dotStyle).addTo(layers.gps);
    if (k > 1 && (gps.length - 1) % k !== 0) L.circleMarker(gps[gps.length - 1], dotStyle).addTo(layers.gps);
    setLegendNote(k > 1
      ? `Showing 1 in every ${k} GPS points (${Math.ceil(gps.length / k).toLocaleString()} of ${gps.length.toLocaleString()}) to keep the map fast.`
      : '');

    pred.forEach((seg) => {
      L.polyline(seg, { pane: 'predPane', color: COLORS.pred, weight: 6, opacity: 0.85, lineJoin: 'round', lineCap: 'round' })
        .bindTooltip('Predicted route (classifier + HMM)', { sticky: true })
        .addTo(layers.pred);
    });
    clf.forEach((seg) => {
      L.polyline(seg, { pane: 'clfPane', color: COLORS.clf, weight: 2.5, opacity: 0.95, dashArray: '6 6' })
        .bindTooltip('Classifier-only route', { sticky: true })
        .addTo(layers.clf);
    });

    if (gps.length) {
      const end = { pane: 'endsPane', radius: 8, color: '#ffffff', weight: 2, fillOpacity: 1 };
      L.circleMarker(gps[0], { ...end, fillColor: COLORS.start })
        .bindTooltip('Start (first GPS point)').addTo(layers.ends);
      L.circleMarker(gps[gps.length - 1], { ...end, fillColor: COLORS.end })
        .bindTooltip('End (last GPS point)').addTo(layers.ends);
    }

    let bounds = null;
    if (validBbox(data.bbox)) {
      bounds = L.latLngBounds(data.bbox);
    } else if (gps.length) {
      bounds = L.latLngBounds(gps);
      warnings.push('Response had no valid bbox; fitted the map to the GPS points instead.');
    }
    if (bounds && bounds.isValid()) map.fitBounds(bounds, { padding: [24, 24] });

    return { warnings };
  }

  // ---------------------------------------------------------------- results panel
  function setText(id, v) {
    $(id).textContent = v;
  }

  function showEmptyResults() {
    els.resultsBody.hidden = true;
    els.resultsEmpty.hidden = false;
  }

  function markBest(idA, idB, a, b) {
    $(idA).classList.remove('best');
    $(idB).classList.remove('best');
    if (!isNum(a) || !isNum(b) || a === b) return;
    $(a > b ? idA : idB).classList.add('best');
  }

  function renderResults(data, rtt) {
    const m = data.metrics || {};
    const h = m.hmm || {};
    const c = m.classifier_only || {};
    const n = isNum(m.n_points) ? m.n_points : data.n_points;
    const t = data.timing_ms || {};

    setText('rTrip', data.trip_id !== undefined ? String(data.trip_id) : state.selected);
    $('rMockTag').hidden = !state.mock;
    setText('rBig', pct(h.top1_accuracy));

    setText('cAcc', pct(c.top1_accuracy));
    setText('hAcc', pct(h.top1_accuracy));
    setText('cCont', fixed3(c.route_continuity));
    setText('hCont', fixed3(h.route_continuity));
    setText('cCorr', `${int(c.n_correct)} / ${int(n)}`);
    setText('hCorr', `${int(h.n_correct)} / ${int(n)}`);
    markBest('cAcc', 'hAcc', c.top1_accuracy, h.top1_accuracy);
    markBest('cCont', 'hCont', c.route_continuity, h.route_continuity);
    markBest('cCorr', 'hCorr', c.n_correct, h.n_correct);
    const hasF1 = isNum(c.route_f1) || isNum(h.route_f1); // optional field added to the contract later
    $('f1Row').hidden = !hasF1;
    if (hasF1) {
      setText('cF1', fixed3(c.route_f1));
      setText('hF1', fixed3(h.route_f1));
      markBest('cF1', 'hF1', c.route_f1, h.route_f1);
    }

    setText('tPre', ms(t.preprocess));
    setText('tClf', ms(t.classifier));
    setText('tHmm', ms(t.hmm));
    setText('tTot', ms(t.total));
    setText('tRtt', ms(rtt));

    setText('iModel', data.model || '—');
    setText('iReq', data.request_id || '—');
    setText('iTime', fmtTime(new Date()) + (state.mock ? ' (mock)' : ''));

    els.resultsEmpty.hidden = true;
    els.resultsBody.hidden = false;
  }

  // ---------------------------------------------------------------- history (trip_id + accuracy only)
  function addHistory(data, tripId) {
    const m = data.metrics || {};
    state.history.unshift({
      tripId: data.trip_id !== undefined ? String(data.trip_id) : tripId,
      hmm: m.hmm ? m.hmm.top1_accuracy : null,
      clf: m.classifier_only ? m.classifier_only.top1_accuracy : null,
      requestId: data.request_id || '',
      at: new Date(),
      mock: state.mock,
    });
    renderHistory();
    renderTrips();
  }

  function renderHistory() {
    els.historyList.replaceChildren();
    els.historyEmpty.hidden = state.history.length > 0;
    state.history.forEach((h) => {
      const li = el('li');
      li.title = 'Click to select this trip (then Run to fetch it again from the API)';
      const left = el('span', 'tid', h.tripId);
      if (h.mock) left.appendChild(el('span', 'mock-tag', 'MOCK'));
      li.appendChild(left);
      li.appendChild(el('span', 'acc', pct(h.hmm)));
      const meta = `HMM ${pct(h.hmm)} · classifier ${pct(h.clf)} · ${fmtTime(h.at)}` +
        (h.requestId ? ` · req ${h.requestId}` : '');
      li.appendChild(el('span', 'meta', meta));
      li.addEventListener('click', () => {
        if (state.running) return;
        if (state.trips.some((t) => t.trip_id === h.tripId)) selectTrip(h.tripId);
      });
      els.historyList.appendChild(li);
    });
  }

  // ---------------------------------------------------------------- mock mode
  function setMock(on) {
    state.mock = on;
    state.reqSeq++; // drop any in-flight result from the other mode
    document.body.classList.toggle('mock', on);
    els.mockBanner.hidden = !on;
    els.apiBase.disabled = on;
    clearMap();
    showEmptyResults();
    clearMessage();
    if (on) {
      setStatus('mock', 'MOCK MODE — not using the API');
    } else {
      setStatus('idle', 'Not tested');
    }
    testConnection();
    loadTrips();
  }

  // ---------------------------------------------------------------- wire up
  function init() {
    if (location.protocol === 'file:') els.fileWarning.hidden = false;

    els.apiBase.value = loadApiBase();
    els.mockToggle.checked = false; // always OFF on load (browsers may restore checkbox state)

    initMap();
    renderHistory();
    updateRunButton();

    els.testBtn.addEventListener('click', async () => {
      clearMessage();
      const ok = await testConnection();
      if (ok && !state.trips.length) loadTrips();
    });
    els.apiBase.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') els.testBtn.click();
    });
    els.apiBase.addEventListener('change', () => {
      try {
        saveApiBase(normalizeBase(els.apiBase.value));
      } catch (e) {
        /* validated again on use */
      }
    });
    els.reloadTripsBtn.addEventListener('click', loadTrips);
    els.runBtn.addEventListener('click', runPrediction);
    els.mockToggle.addEventListener('change', () => setMock(els.mockToggle.checked));
    els.clearHistoryBtn.addEventListener('click', () => {
      state.history = [];
      renderHistory();
      renderTrips();
    });

    testConnection();
    loadTrips();
  }

  init();
})();
