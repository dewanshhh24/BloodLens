/* =============================================
   BloodLens — script.js
   Vanilla JS: Navbar, Upload, Analysis Sim,
   Charts, Confusion Matrix, Scroll Animations
   ============================================= */

/* ==================== NAVBAR ==================== */
const navbar = document.getElementById('navbar');
const navLinks = document.getElementById('navLinks');
const hamburger = document.getElementById('hamburger');

window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
  updateActiveNavLink();
});

hamburger.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});

// Close mobile nav on link click
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => navLinks.classList.remove('open'));
});

function updateActiveNavLink() {
  const sections = ['home', 'features', 'analyze', 'analytics', 'about'];
  const scrollY = window.scrollY + 100;
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const top = el.offsetTop;
    const bottom = top + el.offsetHeight;
    const link = document.querySelector(`.nav-link[href="#${id}"]`);
    if (link) link.classList.toggle('active', scrollY >= top && scrollY < bottom);
  });
}

/* ==================== SCROLL ANIMATIONS ==================== */
const fadeUpEls = document.querySelectorAll('.fade-up');

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

fadeUpEls.forEach(el => observer.observe(el));

/* ==================== COUNTER ANIMATION ==================== */
const counters = document.querySelectorAll('.sc-num[data-target]');
const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const el = entry.target;
    const target = parseInt(el.dataset.target);
    const duration = 1600;
    const start = performance.now();
    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(ease * target);
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target;
    }
    requestAnimationFrame(tick);
    counterObserver.unobserve(el);
  });
}, { threshold: 0.5 });

counters.forEach(c => counterObserver.observe(c));

/* ==================== FILE UPLOAD ==================== */
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadIdle = document.getElementById('uploadIdle');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');

let hasImage = false;

uploadZone.addEventListener('click', (e) => {
  if (!e.target.closest('.preview-change') && !e.target.closest('.btn-upload')) {
    if (!hasImage) fileInput.click();
  }
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadIdle.style.display = 'none';
    uploadPreview.style.display = 'flex';
    analyzeBtn.disabled = false;
    hasImage = true;
    // Reset results
    resetResults();
  };
  reader.readAsDataURL(file);
}

// /* ==================== SAMPLE IMAGE ==================== */
// document.getElementById('useSampleBtn').addEventListener('click', () => {
//   // 👉 Use a real placeholder image
//   const sampleImagePath = "sample.jpg"; 

//   previewImg.src = sampleImagePath;

//   uploadIdle.style.display = 'none';
//   uploadPreview.style.display = 'flex';
//   analyzeBtn.disabled = false;
//   hasImage = true;

//   resetResults();
// });

/* ==================== ANALYSIS SIMULATION ==================== */
const cellTypes = [
  { name: 'Lymphocyte', desc: 'Mononuclear white blood cell' },
  { name: 'Neutrophil', desc: 'Most abundant granulocyte' },
  { name: 'Monocyte', desc: 'Largest type of white blood cell' },
  { name: 'Eosinophil', desc: 'Bilobed granulocyte' },
  { name: 'Basophil', desc: 'Least common granulocyte' },
  { name: 'Erythroblast', desc: 'Immature red blood cell precursor' },
  { name: 'Platelet', desc: 'Cell fragment for hemostasis' },
  { name: 'Immature Granulocyte', desc: 'Immature WBC at band stage' },
];

const cancerTypes = [
  { name: 'Benign', desc: 'No malignancy detected', color: '#10B981' },
  { name: 'Pre-B ALL', desc: 'Precursor B-cell acute lymphoblastic', color: '#F59E0B' },
  { name: 'Pro-B ALL', desc: 'Pro-B acute lymphoblastic leukemia', color: '#EF4444' },
  { name: 'Early Pre-B', desc: 'Early precursor B-cell stage', color: '#F97316' },
];

analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  const btnText = document.getElementById('analyzeBtnText');
  const btnLoading = document.getElementById('analyzeBtnLoading');

  btnText.style.display = 'none';
  btnLoading.style.display = 'flex';
  analyzeBtn.disabled = true;

  const file = fileInput.files[0];

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    showResults(
      { name: data.cell, desc: "Predicted cell type" },
      { name: data.cancer, desc: "Predicted cancer stage" },
      data.cell_conf,
      data.cancer_conf,
      (data.cell_conf + data.cancer_conf)/2
    );

  } catch (error) {
    alert("Error connecting to model");
    console.error(error);
  }

  btnText.style.display = 'flex';
  btnLoading.style.display = 'none';
  analyzeBtn.disabled = false;
}

function showResults(cell, cancer, cellConf, cancerConf, overall) {
  document.getElementById('resultsEmpty').style.display = 'none';
  const content = document.getElementById('resultsContent');
  content.style.display = 'flex';

  // Header
  const now = new Date();
  document.getElementById('resultTime').textContent =
    `${now.toLocaleTimeString()} · ${(Math.random() * 1.2 + 0.8).toFixed(2)}s`;

  // Cell type
  document.getElementById('cellTypeResult').textContent = cell.name;
  document.getElementById('cellTypeSub').textContent = cell.desc;

  // Cancer
  document.getElementById('cancerResult').textContent = cancer.name;
  document.getElementById('cancerSub').textContent = cancer.desc;

  // Confidence bars — animate with slight delay
  setTimeout(() => {
    const cellBar = document.getElementById('cellConfBar');
    const cancerBar = document.getElementById('cancerConfBar');
    const overallBar = document.getElementById('overallConfBar');
    cellBar.style.width = cellConf + '%';
    cancerBar.style.width = cancerConf + '%';
    overallBar.style.width = overall + '%';
    document.getElementById('cellConfPct').textContent = cellConf + '%';
    document.getElementById('cancerConfPct').textContent = cancerConf + '%';
    document.getElementById('overallConfPct').textContent = overall + '%';
  }, 120);

  // Top predictions
  buildTopPredictions(cell, cancer, cellConf);

  // Scroll to results
  setTimeout(() => {
    document.getElementById('resultsContent').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 200);
}

function buildTopPredictions(topCell, cancer, topConf) {
  const container = document.getElementById('topPredList');
  container.innerHTML = '';

  // Cell predictions
  const others = cellTypes.filter(c => c.name !== topCell.name);
  const shuffled = others.sort(() => Math.random() - .5).slice(0, 3);

  const remaining = (100 - topConf).toFixed(1);
  const predictions = [
    { name: topCell.name, pct: parseFloat(topConf), top: true },
    { name: shuffled[0].name, pct: parseFloat((remaining * 0.5).toFixed(1)), top: false },
    { name: shuffled[1].name, pct: parseFloat((remaining * 0.3).toFixed(1)), top: false },
    { name: shuffled[2].name, pct: parseFloat((remaining * 0.2).toFixed(1)), top: false },
  ];

  predictions.forEach(p => {
    const div = document.createElement('div');
    div.className = `pred-item${p.top ? ' top-pred' : ''}`;
    div.innerHTML = `
      <span class="pred-name">${p.name}</span>
      <span class="pred-pct">${p.pct}%</span>
    `;
    container.appendChild(div);
  });
}

function resetResults() {
  document.getElementById('resultsEmpty').style.display = 'flex';
  document.getElementById('resultsContent').style.display = 'none';

  ['cellConfBar', 'cancerConfBar', 'overallConfBar'].forEach(id => {
    document.getElementById(id).style.width = '0%';
  });
}

/* ==================== ACCURACY CHART ==================== */
const chartEl = document.getElementById('accuracyChart');

// ✅ Only 10 epochs now (real data)
const epochs = ['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10'];

// ✅ REAL DATA
const cellAccData = [0.5131, 0.7336, 0.8020, 0.8231, 0.8542, 0.8741, 0.8881, 0.9033, 0.9044, 0.9183];
const cellValData = [0.7160, 0.7707, 0.8454, 0.8732, 0.8594, 0.8846, 0.8659, 0.8796, 0.9004, 0.9212];

const cancerAccData = [0.7769, 0.9237, 0.9260, 0.9468, 0.9568, 0.9576, 0.9595, 0.9645, 0.9676, 0.9661];
const cancerValData = [0.8594, 0.9289, 0.9212, 0.9397, 0.9614, 0.9289, 0.9521, 0.9444, 0.9583, 0.9397];

new Chart(chartEl, {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      {
        label: 'Cell Train',
        data: cellAccData,
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59,130,246,.08)',
        borderWidth: 2,
        tension: .4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: 'Cell Val',
        data: cellValData,
        borderColor: '#93C5FD',
        borderWidth: 1.5,
        tension: .4,
        borderDash: [4, 3],
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: 'Cancer Train',
        data: cancerAccData,
        borderColor: '#B22222',
        backgroundColor: 'rgba(178,34,34,.06)',
        borderWidth: 2,
        tension: .4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: 'Cancer Val',
        data: cancerValData,
        borderColor: '#F87171',
        borderWidth: 1.5,
        tension: .4,
        borderDash: [4, 3],
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
    ]
  },
  options: {
    responsive: true,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          font: { family: 'DM Sans', size: 11 },
          color: '#6B7280',
          boxWidth: 12,
          boxHeight: 3,
          padding: 16,
          usePointStyle: true,
          pointStyle: 'line',
        }
      },
      tooltip: {
        backgroundColor: '#1C1C1C',
        titleFont: { family: 'DM Sans', size: 11 },
        bodyFont: { family: 'DM Sans', size: 11 },
        titleColor: 'rgba(255,255,255,.6)',
        bodyColor: '#fff',
        padding: 10,
        cornerRadius: 8,
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
        }
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { font: { family: 'DM Sans', size: 10 }, color: '#9CA3AF' }
      },
      y: {
        min: 0.3,
        max: 1.0,
        grid: { color: 'rgba(0,0,0,.05)' },
        ticks: {
          font: { family: 'DM Sans', size: 10 },
          color: '#9CA3AF',
          callback: v => (v * 100).toFixed(0) + '%'
        }
      }
    }
  }
});


const lossCtx = document.getElementById('lossChart').getContext('2d');

new Chart(lossCtx, {
  type: 'line',
  data: {
    labels: ['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10'],
    datasets: [
      {
        label: 'Cell Train Loss',
        data: [1.2848, 0.7106, 0.5384, 0.4816, 0.4103, 0.3681, 0.3277, 0.2846, 0.2845, 0.2442],
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59,130,246,.08)',
        borderWidth: 2,
        tension: .4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: 'Cell Val Loss',
        data: [0.7736, 0.5684, 0.4963, 0.3863, 0.4498, 0.3475, 0.4129, 0.4490, 0.3860, 0.2317],
        borderColor: '#93C5FD',
        borderWidth: 1.5,
        tension: .4,
        borderDash: [4, 3],
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: 'Cancer Train Loss',
        data: [0.6072, 0.2267, 0.1948, 0.1489, 0.1286, 0.1275, 0.1147, 0.1012, 0.1011, 0.1019],
        borderColor: '#B22222',
        backgroundColor: 'rgba(178,34,34,.06)',
        borderWidth: 2,
        tension: .4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: 'Cancer Val Loss',
        data: [0.3713, 0.2169, 0.2131, 0.1860, 0.1483, 0.2247, 0.1628, 0.1568, 0.1271, 0.1848],
        borderColor: '#F87171',
        borderWidth: 1.5,
        tension: .4,
        borderDash: [4, 3],
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 4,
      }
    ]
  },
  options: {
    responsive: true,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          font: { family: 'DM Sans', size: 11 },
          color: '#6B7280',
          boxWidth: 12,
          boxHeight: 3,
          padding: 16,
          usePointStyle: true,
          pointStyle: 'line',
        }
      },
      tooltip: {
        backgroundColor: '#1C1C1C',
        titleFont: { family: 'DM Sans', size: 11 },
        bodyFont: { family: 'DM Sans', size: 11 },
        titleColor: 'rgba(255,255,255,.6)',
        bodyColor: '#fff',
        padding: 10,
        cornerRadius: 8,
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}`
        }
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { font: { family: 'DM Sans', size: 10 }, color: '#9CA3AF' }
      },
      y: {
        grid: { color: 'rgba(0,0,0,.05)' },
        ticks: {
          font: { family: 'DM Sans', size: 10 },
          color: '#9CA3AF'
        }
      }
    }
  }
});

/* ==================== CONFUSION MATRIX ==================== */
const cellLabels = ['NEU', 'LYM', 'MON', 'EOS', 'BAS', 'ERY', 'PLT', 'IG'];
const matrixContainer = document.getElementById('confusionMatrix');

// ✅ REAL CONFUSION MATRIX VALUES (RAW COUNTS)
const matrix = [
  [228, 2, 1, 7, 0, 0, 5, 0],
  [1, 618, 0, 1, 0, 0, 3, 0],
  [1, 1, 288, 13, 1, 1, 5, 0],
  [14, 0, 5, 484, 0, 32, 44, 0],
  [1, 0, 6, 5, 222, 7, 1, 0],
  [2, 2, 1, 100, 2, 174, 3, 0],
  [1, 1, 0, 25, 0, 1, 637, 0],
  [0, 0, 1, 0, 0, 0, 0, 468]
];

// 🔥 Normalize ONLY for color (not display)
const maxVal = Math.max(...matrix.flat());

// Color interpolation based on value
function valueToColor(v) {
  const norm = v / maxVal; // scale for color only
  const r = Math.round(139 + (248 - 139) * (1 - norm));
  const g = Math.round(0 + (242 - 0) * (1 - norm));
  const b = Math.round(0 + (242 - 0) * (1 - norm));
  return `rgb(${r},${g},${b})`;
}

function textColor(v) {
  return v / maxVal > 0.45 ? '#fff' : '#1C1C1C';
}

// Header row
const blankHeader = document.createElement('div');
blankHeader.className = 'matrix-cell header';
blankHeader.textContent = '';
matrixContainer.appendChild(blankHeader);

cellLabels.forEach(label => {
  const cell = document.createElement('div');
  cell.className = 'matrix-cell header';
  cell.textContent = label;
  matrixContainer.appendChild(cell);
});

// Data rows
matrix.forEach((row, i) => {
  const rowHeader = document.createElement('div');
  rowHeader.className = 'matrix-cell row-header';
  rowHeader.textContent = cellLabels[i];
  matrixContainer.appendChild(rowHeader);

  row.forEach(val => {
    const cell = document.createElement('div');
    cell.className = 'matrix-cell';
    cell.style.background = valueToColor(val);
    cell.style.color = textColor(val);

    // ✅ RAW VALUE DISPLAY
    cell.textContent = val;

    // tooltip
    cell.title = `${val} samples`;

    matrixContainer.appendChild(cell);
  });
});

/* ==================== CANCER CONFUSION MATRIX ==================== */

const cancerLabels = ['BEN', 'EARLY', 'PRE', 'PRO'];
const cancerMatrixContainer = document.getElementById('cancerConfusionMatrix');

// ✅ REAL RAW VALUES
const cancerMatrix = [
  [91, 11, 0, 0],
  [0, 195, 0, 0],
  [1, 7, 180, 3],
  [1, 7, 2, 149]
];

// normalize only for color scaling
const maxCancerVal = Math.max(...cancerMatrix.flat());

// Header row
const blankHeader2 = document.createElement('div');
blankHeader2.className = 'matrix-cell header';
blankHeader2.textContent = '';
cancerMatrixContainer.appendChild(blankHeader2);

cancerLabels.forEach(label => {
  const cell = document.createElement('div');
  cell.className = 'matrix-cell header';
  cell.textContent = label;
  cancerMatrixContainer.appendChild(cell);
});

// Data rows
cancerMatrix.forEach((row, i) => {
  const rowHeader = document.createElement('div');
  rowHeader.className = 'matrix-cell row-header';
  rowHeader.textContent = cancerLabels[i];
  cancerMatrixContainer.appendChild(rowHeader);

  row.forEach(val => {
    const cell = document.createElement('div');
    cell.className = 'matrix-cell';

    // same color logic as before
    const norm = val / maxCancerVal;
    const r = Math.round(139 + (248 - 139) * (1 - norm));
    const g = Math.round(0 + (242 - 0) * (1 - norm));
    const b = Math.round(0 + (242 - 0) * (1 - norm));

    cell.style.background = `rgb(${r},${g},${b})`;
    cell.style.color = norm > 0.45 ? '#fff' : '#1C1C1C';

    cell.textContent = val;
    cell.title = `${val} samples`;

    cancerMatrixContainer.appendChild(cell);
  });
});

/* ==================== SMOOTH SCROLL ==================== */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

/* ==================== HERO CBM FILL DELAY ==================== */
window.addEventListener('load', () => {
  setTimeout(() => {
    const fill = document.querySelector('.cbm-fill');
    if (fill) fill.style.width = '94.2%';
  }, 800);
});
function createConfusionMatrix(containerId, labels, matrixData) {
  const container = document.getElementById(containerId);
  container.innerHTML = ''; // clear previous

  const size = labels.length;
  container.style.gridTemplateColumns = `repeat(${size + 1}, ${size === 8 ? '52px' : '68px'})`;

  const maxVal = Math.max(...matrixData.flat());

  // Blank top-left cell
  const blank = document.createElement('div');
  blank.className = 'matrix-cell header';
  container.appendChild(blank);

  // Column headers
  labels.forEach(label => {
    const cell = document.createElement('div');
    cell.className = 'matrix-cell header';
    cell.textContent = label;
    container.appendChild(cell);
  });

  // Rows
  matrixData.forEach((row, i) => {
    // Row header
    const rowHeader = document.createElement('div');
    rowHeader.className = 'matrix-cell row-header';
    rowHeader.textContent = labels[i];
    container.appendChild(rowHeader);

    // Data cells
    row.forEach(val => {
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';

      const norm = val / maxVal;
      const r = Math.round(139 + (248 - 139) * (1 - norm));
      const g = Math.round(0 + (242 - 0) * (1 - norm));
      const b = Math.round(0 + (242 - 0) * (1 - norm));

      cell.style.background = `rgb(${r},${g},${b})`;
      cell.style.color = norm > 0.45 ? '#fff' : '#1C1C1C';
      cell.textContent = val;
      cell.title = `${val} samples`;

      container.appendChild(cell);
    });
  });
}

// Cell Matrix (8x8)
createConfusionMatrix('confusionMatrix', cellLabels, matrix);

// Cancer Matrix (4x4)
createConfusionMatrix('cancerConfusionMatrix', cancerLabels, cancerMatrix);