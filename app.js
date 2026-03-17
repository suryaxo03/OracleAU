/* ================================================================
   CONFIG
   ─────
   LOCAL DEV:  Keep the localhost line active, comment out Render
   DEPLOY:     Comment out localhost, uncomment the Render line
================================================================ */

// const API_BASE = 'http://localhost:8000';       // ← LOCAL (active)
const API_BASE = 'https://oracleau.onrender.com'; // ← DEPLOY (comment out above, uncomment this)

/* ================================================================
   STATE
================================================================ */
let priceChart     = null;
let currentPeriod  = '3m';
let historyCache   = {};
let forecastData   = null;
let lastKnownPrice = null;

/* ================================================================
   UTILITIES
================================================================ */
const $ = id => document.getElementById(id);

function fmt(n, dp = 2) {
  if (n == null || isNaN(n)) return '—';
  return Number(n).toFixed(dp);
}

function fmtDate(str) {
  // "2026-03-07" → "07 Mar"
  const d = new Date(str + 'T00:00:00');
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });
}

function flashEl(el) {
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

function setStatus(online) {
  $('status-dot').className = 'status-dot ' + (online ? 'online' : 'offline');
  $('status-text').textContent = online ? 'API Online' : 'API Offline';
}

/* ================================================================
   API CALLS
================================================================ */
async function apiFetch(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* ================================================================
   HEALTH CHECK
================================================================ */
async function checkHealth() {
  try {
    const data = await apiFetch('/health');
    setStatus(data.status === 'ok');
  } catch {
    setStatus(false);
  }
}

/* ================================================================
   PRICE
================================================================ */
async function loadPrice() {
  try {
    const d = await apiFetch('/api/price');
    lastKnownPrice = d.price;

    // Header price
    const priceEl = $('header-price');
    priceEl.textContent = '£' + fmt(d.price, 4);
    flashEl(priceEl);

    // Change badge
    const changeEl = $('header-change');
    const isUp = d.change >= 0;
    changeEl.textContent = (isUp ? '▲' : '▼') + ' £' + fmt(Math.abs(d.change), 4) + ' (' + fmt(d.change_pct, 2) + '%)';
    changeEl.className = 'live-price-change ' + (isUp ? 'up' : 'down');

    // Last updated
    $('last-updated').textContent = 'As of ' + d.last_updated;

    // Price stats grid
    $('price-stats').innerHTML = `
      <div class="stat-tile">
        <div class="stat-label">Open</div>
        <div class="stat-value">£${fmt(d.open, 4)}</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">High / Low</div>
        <div class="stat-value">£${fmt(d.high, 2)}</div>
        <div class="stat-sub">Low: £${fmt(d.low, 2)}</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">52-Week High</div>
        <div class="stat-value">£${fmt(d.week_52_high, 2)}</div>
        <div class="stat-sub">Low: £${fmt(d.week_52_low, 2)}</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Volume</div>
        <div class="stat-value">${d.volume ? d.volume.toLocaleString() : '—'}</div>
        <div class="stat-sub">Prev close: £${fmt(d.prev_close, 4)}</div>
      </div>
    `;

    if (d.stale_data) $('stale-notice').style.display = 'inline';

  } catch(e) {
    $('header-price').textContent = 'Unavailable';
    console.error('Price load failed:', e);
  }
}

/* ================================================================
   FORECAST
================================================================ */
async function loadForecast() {
  try {
    const d = await apiFetch('/api/forecast');
    forecastData = d;

    // Overall signal
    const dirEl = $('overall-direction');
    dirEl.textContent = d.overall_direction;
    dirEl.className   = 'direction-badge ' + d.overall_direction;

    const confEl = $('overall-confidence');
    confEl.textContent = d.overall_confidence + ' Confidence';
    confEl.className   = 'confidence-badge ' + d.overall_confidence;

    $('forecast-meta').innerHTML =
      `Last close: £${fmt(d.last_close, 4)} &nbsp;|&nbsp; ` +
      `Generated: ${new Date(d.generated_at).toLocaleTimeString('en-GB', {hour:'2-digit',minute:'2-digit'})}`;

    if (d.stale_data) $('stale-notice').style.display = 'inline';

    // Table rows
    const tbody = $('forecast-tbody');
    tbody.innerHTML = '';
    d.forecast.forEach((row, i) => {
      const isUp    = row.direction === 'UP';
      const arrow   = row.direction === 'UP' ? '▲' : row.direction === 'DOWN' ? '▼' : '—';
      const arrowCls = isUp ? 'up' : 'down';
      const chgSign  = row.change_pct >= 0 ? '+' : '';

      tbody.innerHTML += `
        <tr class="${i === 0 ? 'day-one' : ''}">
          <td class="td-day">Day ${row.day}</td>
          <td class="td-date" data-day="Day ${row.day}">${fmtDate(row.date)}</td>
          <td class="td-price">£${fmt(row.ensemble, 4)}</td>
          <td class="td-models col-models">
            £${fmt(row.xgboost, 4)}<br>
            £${fmt(row.lstm, 4)}
          </td>
          <td class="td-change ${arrowCls}">
            <span class="dir-arrow ${arrowCls}">${arrow}</span>${chgSign}${fmt(row.change_pct, 3)}%
          </td>
          <td class="td-confidence">
            <span class="conf-pip ${row.confidence}">${row.confidence}</span>
          </td>
        </tr>
      `;
    });

    // Update chart with forecast overlay if chart exists
    if (priceChart) overlayForecast(d);

  } catch(e) {
    $('forecast-tbody').innerHTML =
      `<tr><td colspan="6"><div class="error-state">Forecast unavailable — ${e.message}</div></td></tr>`;
    console.error('Forecast load failed:', e);
  }
}

/* ================================================================
   HISTORY + CHART
================================================================ */
async function loadHistory(period) {
  currentPeriod = period;

  // Update button states
  document.querySelectorAll('.period-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.period === period);
  });

  // Use cache if available
  if (historyCache[period]) {
    renderChart(historyCache[period]);
    return;
  }

  try {
    const d = await apiFetch('/api/history?period=' + period);
    historyCache[period] = d;
    renderChart(d);
  } catch(e) {
    console.error('History load failed:', e);
  }
}

function renderChart(histData) {
  const labels = histData.data.map(r => r.date);
  const prices = histData.data.map(r => r.close);

  // Build forecast extension
  let forecastLabels = [];
  let forecastPrices = [];

  if (forecastData) {
    forecastLabels = forecastData.forecast.map(f => f.date);
    forecastPrices = forecastData.forecast.map(f => f.ensemble);

    // Join the last historical point to the first forecast point
    forecastLabels.unshift(labels[labels.length - 1]);
    forecastPrices.unshift(prices[prices.length - 1]);
  }

  const allLabels = [...labels, ...forecastLabels.slice(1)];

  // Historical dataset
  const histDataset = {
    label: 'SGLN.L Close',
    data: [...prices, ...forecastPrices.slice(1).map(() => null)],
    borderColor: '#c9a84c',
    backgroundColor: 'rgba(201,168,76,0.06)',
    borderWidth: 1.5,
    pointRadius: 0,
    pointHoverRadius: 4,
    fill: true,
    tension: 0.3,
  };

  // Forecast dataset (dashed)
  const forecastDataset = {
    label: '7-Day Forecast',
    data: [...prices.map(() => null), ...forecastPrices],
    borderColor: 'rgba(201,168,76,0.6)',
    borderDash: [5, 4],
    borderWidth: 1.5,
    pointRadius: 3,
    pointBackgroundColor: '#c9a84c',
    pointBorderColor: '#0a0a0b',
    pointBorderWidth: 1.5,
    fill: false,
    tension: 0.3,
  };

  const ctx = $('price-chart').getContext('2d');

  if (priceChart) priceChart.destroy();

  priceChart = new Chart(ctx, {
    type: 'line',
    data: { labels: allLabels, datasets: [histDataset, forecastDataset] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#13131a',
          borderColor: '#1e1e28',
          borderWidth: 1,
          titleColor: '#8888a0',
          bodyColor: '#e8e8f0',
          titleFont: { family: 'JetBrains Mono', size: 10 },
          bodyFont:  { family: 'JetBrains Mono', size: 12 },
          padding: 10,
          callbacks: {
            title: items => items[0].label,
            label: item => {
              if (item.raw == null) return null;
              return (item.datasetIndex === 1 ? '◈ Forecast ' : '◆ Close ') + '£' + fmt(item.raw, 4);
            },
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: '#44445a',
            font: { family: 'JetBrains Mono', size: 10 },
            maxTicksLimit: 8,
            maxRotation: 0,
          },
          grid: { color: 'rgba(30,30,40,0.8)' },
          border: { color: '#1e1e28' },
        },
        y: {
          ticks: {
            color: '#44445a',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: v => '£' + v.toFixed(2),
          },
          grid: { color: 'rgba(30,30,40,0.8)' },
          border: { color: '#1e1e28' },
        }
      }
    }
  });
}

function overlayForecast(d) {
  if (!priceChart || !historyCache[currentPeriod]) return;
  renderChart(historyCache[currentPeriod]);
}

/* ================================================================
   INDICATORS
================================================================ */
async function loadIndicators() {
  try {
    const d = await apiFetch('/api/indicators');

    // RSI
    const rsi = d.rsi.value;
    $('rsi-value').textContent = fmt(rsi, 2);
    $('rsi-signal').textContent = d.rsi.signal || '—';
    $('rsi-signal').className = 'ind-signal ' + (d.rsi.signal || 'NEUTRAL');
    const rsiPct = Math.min(Math.max((rsi / 100) * 100, 0), 100);
    $('rsi-marker').style.left = rsiPct + '%';

    // MACD
    $('macd-value').textContent = fmt(d.macd.macd, 4);
    $('macd-signal-val').textContent = fmt(d.macd.signal, 4);
    $('macd-hist-val').textContent = fmt(d.macd.histogram, 4);
    $('macd-crossover').textContent = d.macd.crossover || '—';
    $('macd-crossover').className = 'ind-signal ' + (d.macd.crossover || 'NEUTRAL');

    // Bollinger Bands
    const pctB = d.bollinger_bands.pct_b;
    $('bb-pct').textContent = fmt(pctB * 100, 1) + '%B';
    $('bb-lower').textContent = fmt(d.bollinger_bands.lower, 2);
    $('bb-upper').textContent = fmt(d.bollinger_bands.upper, 2);
    $('bb-width').textContent = fmt(d.bollinger_bands.width, 4);
    $('bb-marker').style.left = Math.min(Math.max(pctB * 100, 0), 100) + '%';

    // Trend
    const trend  = d.moving_averages.trend || 'NEUTRAL';
    const tLabel = trend.replace(/_/g, ' ');
    $('trend-value').textContent = tLabel;
    $('trend-signal').textContent = tLabel;
    $('trend-signal').className = 'ind-signal ' + trend;
    $('sma50').textContent  = fmt(d.moving_averages.sma_50, 2);
    $('sma200').textContent = fmt(d.moving_averages.sma_200, 2);

    // Volatility
    $('vol-ann').textContent = fmt(d.volatility.volatility_annualised, 2) + '%';
    $('vol-20d').textContent = fmt(d.volatility.volatility_20d, 4) + '%';

    // Crossover signals
    const yesGreen = v => `<span class="ind-signal ${v ? 'BULLISH' : 'NEUTRAL'}">${v ? 'Yes' : 'No'}</span>`;
    const yesRed   = v => `<span class="ind-signal ${v ? 'BEARISH' : 'NEUTRAL'}">${v ? 'Yes' : 'No'}</span>`;
    $('golden-cross').outerHTML = '<span id="golden-cross" class="ind-signal ' + (d.crossover_signals.golden_cross ? 'BULLISH' : 'NEUTRAL') + '">' + (d.crossover_signals.golden_cross ? 'Active' : 'None') + '</span>';
    $('death-cross').outerHTML  = '<span id="death-cross"  class="ind-signal ' + (d.crossover_signals.death_cross  ? 'BEARISH' : 'NEUTRAL') + '">' + (d.crossover_signals.death_cross  ? 'Active' : 'None') + '</span>';
    $('above-sma20').outerHTML  = '<span id="above-sma20"  class="ind-signal ' + (d.moving_averages.above_sma20 ? 'BULLISH' : 'NEUTRAL') + '">' + (d.moving_averages.above_sma20 ? 'Yes' : 'No') + '</span>';
    $('above-sma200').outerHTML = '<span id="above-sma200" class="ind-signal ' + (d.moving_averages.above_sma200 ? 'BULLISH' : 'NEUTRAL') + '">' + (d.moving_averages.above_sma200 ? 'Yes' : 'No') + '</span>';

  } catch(e) {
    console.error('Indicators load failed:', e);
  }
}

/* ================================================================
   ACCURACY TRACKER — weekly batches from /api/accuracy
================================================================ */
async function loadAccuracy() {
  const tbody = $('accuracy-tbody');
  const track = $('acc-track');

  tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:20px">Loading accuracy data...</td></tr>`;
  track.innerHTML = '';

  try {
    const d = await apiFetch('/api/accuracy');

    // ── Summary tiles ─────────────────────────────────────────
    if (d.directional_acc !== null) {
      $('acc-live-da').textContent = d.directional_acc + '%';
      $('acc-live-da').parentElement.querySelector('.acc-tile-sub').textContent =
        `${d.total_evaluated} days · ${d.total_weeks || 0} weeks`;
    }
    if (d.mae !== null) {
      $('acc-mae').textContent = '£' + fmt(d.mae, 4);
    }

    // ── No complete weeks yet ─────────────────────────────────
    if (!d.results || d.results.length === 0) {
      tbody.innerHTML = `
        <tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:28px;line-height:1.8">
          No completed weeks yet.<br>
          <span style="font-size:11px">Results appear the Monday after each 7-day forecast window closes.</span>
        </td></tr>`;
      return;
    }

    // ── Render rows grouped by week ───────────────────────────
    tbody.innerHTML = '';
    track.innerHTML = '';

    // Group results by week_key, preserving order
    const weekGroups = {};
    const weekOrder  = [];
    d.results.forEach(row => {
      const wk = row.week_key || 'unknown';
      if (!weekGroups[wk]) {
        weekGroups[wk] = [];
        weekOrder.push(wk);
      }
      weekGroups[wk].push(row);
    });

    weekOrder.forEach(wk => {
      const rows      = weekGroups[wk];
      const wCorrect  = rows.filter(r => r.correct).length;
      const wTotal    = rows.length;
      const wDA       = Math.round(wCorrect / wTotal * 100);
      const wMAE      = (rows.reduce((s, r) => s + Math.abs(r.error || 0), 0) / wTotal).toFixed(2);

      // ── Week header row ───────────────────────────────────
      tbody.innerHTML += `
        <tr>
          <td colspan="6" style="
            padding: 10px 12px 6px;
            background: rgba(201,168,76,0.04);
            border-bottom: 1px solid var(--panel-border);
            border-top: 1px solid var(--panel-border);
          ">
            <span style="font-family:var(--font-display);font-size:9px;letter-spacing:0.15em;
                         text-transform:uppercase;color:var(--gold-dim)">
              Week of ${fmtDate(rows[0].date)}
            </span>
            <span style="float:right;font-size:10px;color:var(--text-muted)">
              ${wCorrect}/${wTotal} correct &nbsp;·&nbsp;
              <span style="color:${wDA >= 70 ? 'var(--up)' : wDA >= 50 ? 'var(--gold)' : 'var(--down)'}">
                ${wDA}% DA
              </span>
              &nbsp;·&nbsp; MAE £${wMAE}
            </span>
          </td>
        </tr>`;

      // ── Day rows for this week ────────────────────────────
      rows.forEach(row => {
        const err       = row.error || 0;
        const errCls    = err >= 0 ? 'up' : 'down';
        const isCorrect = row.correct;
        const resultCls = isCorrect ? 'correct' : 'miss';
        const actDir    = row.act_direction || '—';

        tbody.innerHTML += `
          <tr>
            <td style="color:var(--text-secondary);padding-left:20px">${fmtDate(row.date)}</td>
            <td style="text-align:right" class="acc-col-predicted">£${fmt(row.predicted, 4)}</td>
            <td style="text-align:right;color:var(--text-primary)">£${fmt(row.actual, 4)}</td>
            <td style="text-align:right" class="td-change acc-col-error ${errCls}">
              ${err >= 0 ? '+' : ''}${fmt(err, 4)}
            </td>
            <td style="text-align:right">
              <span style="color:${actDir === 'UP' ? 'var(--up)' : 'var(--down)'}">
                ${actDir === 'UP' ? '▲' : '▼'} ${actDir}
              </span>
            </td>
            <td style="text-align:right">
              <span class="result-pill ${resultCls}">${isCorrect ? '✓ Correct' : '✗ Miss'}</span>
            </td>
          </tr>`;

        track.innerHTML += `<div class="acc-bar ${resultCls}" title="${fmtDate(row.date)}">${isCorrect ? '✓' : '✗'}</div>`;
      });
    });

  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="6"><div class="error-state">Accuracy data unavailable — ${e.message}</div></td></tr>`;
    console.error('Accuracy load failed:', e);
  }
}

/* ================================================================
   STATS — visit counter from MongoDB via /api/stats
================================================================ */
async function loadStats() {
  try {
    const d = await apiFetch('/api/stats');
    const el = $('footer-visits');
    if (!el) return;

    if (!d.db_connected || d.total_visits === null) {
      el.innerHTML = '';
      return;
    }

    el.innerHTML =
      `<span style="color:var(--gold-dim)">◆</span> ` +
      `<span style="color:var(--text-muted)">` +
      `${d.total_visits.toLocaleString()} visits` +
      (d.visits_today > 0 ? ` · ${d.visits_today} today` : '') +
      `</span>`;

  } catch(e) {
    // Non-critical — silently ignore if stats unavailable
  }
}

/* ================================================================
   STATS — visit counter from MongoDB via /api/stats
================================================================ */
async function loadStats() {
  try {
    const d = await apiFetch('/api/stats');
    if (!d.db_connected) return;

    // Total visits
    if (d.total_visits !== null) {
      const el = $('footer-visits');
      if (el) {
        el.textContent = d.total_visits.toLocaleString() + ' dashboard loads';
        el.style.display = 'inline';
      }
    }

    // Visits today
    if (d.visits_today !== null) {
      const el = $('footer-visits-today');
      if (el) {
        el.textContent = d.visits_today + ' today';
        el.style.display = 'inline';
      }
    }
  } catch(e) {
    // Stats are non-critical — fail silently
    console.debug('Stats load failed (non-critical):', e);
  }
}

/* ================================================================
   PERIOD BUTTON EVENTS
================================================================ */
document.querySelectorAll('.period-btn').forEach(btn => {
  btn.addEventListener('click', () => loadHistory(btn.dataset.period));
});

/* ================================================================
   AUTO-REFRESH — price every 5 min, everything else on load only
================================================================ */
async function init() {
  await checkHealth();
  await Promise.all([loadPrice(), loadForecast(), loadIndicators()]);
  await loadHistory('3m');
  await loadAccuracy();
  loadStats();   // Non-blocking — fire and forget
  loadStats();   // Non-blocking — fire and forget
}

// Refresh price and health silently every 5 minutes
setInterval(async () => {
  await checkHealth();
  await loadPrice();
}, 5 * 60 * 1000);

// Boot
init();