/**
 * Edit Banana Studio V2 — Interactive Vector Artboard Engine
 * - Full 2D Canvas Panning & Zooming (No flex clipping, drag anywhere with cursor)
 * - Transparent Wireframe Overlay (Background image 100% visible and unblocked)
 * - YOLO-Style Multi-Color Bounding Boxes & Badges for all Element Types
 * - Regional Marquee Auto-Scan (光标框选自动识别范围内多层元素)
 * - Photoshop-Style Right-Click Context Menu (颜色、文字排版、图层上下层顺序、删除、替换)
 * - KaTeX Mathematical Formula Live Rendering & Symbol Palette
 * - Solid & Gradient Backgrounds, Marquee Multi-Select, and Native DrawIO XML Export
 */

// Global Studio State
const state = {
  canvasWidth: 1024,
  canvasHeight: 1024,
  zoom: 0.8,
  panX: 40,
  panY: 40,
  elements: [],
  selectedIds: new Set(),
  activeFilter: 'all',
  toolMode: 'select', // 'select' | 'pan'
  viewMode: 'overlay', // 'overlay' | 'vector'
  showYoloTags: true,
  bgImageSrc: null,
  bgOpacity: 1.0,
  
  // Navigation & Drag State
  isPanning: false,
  panStart: { x: 0, y: 0 },
  isSpaceDown: false,
  isMarquee: false,
  marqueeStart: null,
  lastMarqueeBox: null,
  isDraggingElements: false,
  dragStart: null,
  dragElementsInitial: new Map(),
  isResizing: false,
  resizeHandle: null,
  resizeInitialElem: null,
  resizeStartMouse: null,
  
  undoStack: [],
};

// YOLO-Style Color Mapping
const YOLO_COLORS = {
  title: '#ea580c',       // Bright Orange
  module: '#2563eb',      // Electric Blue
  shape: '#2563eb',       // Blue
  content: '#8b5cf6',     // Royal Purple
  text: '#8b5cf6',        // Purple
  formula: '#d97706',     // Golden Amber
  arrow: '#10b981',       // Emerald Green
  symbol: '#06b6d4',      // Cyan / Teal
  image: '#f43f5e',       // Rose / Pink
  image_patch: '#f43f5e', // Rose / Pink
};

// Gradient Presets Dictionary
const GRADIENT_PRESETS = [
  { name: 'Sunset Glow', stops: ['#f43f5e', '#fbbf24'], angle: 135 },
  { name: 'Ocean Breeze', stops: ['#3b82f6', '#06b6d4'], angle: 135 },
  { name: 'Emerald Fresh', stops: ['#10b981', '#a7f3d0'], angle: 135 },
  { name: 'Royal Violet', stops: ['#8b5cf6', '#ec4899'], angle: 135 },
  { name: 'Studio Amber', stops: ['#f59e0b', '#fef08a'], angle: 135 },
  { name: 'Crisp Slate', stops: ['#334155', '#64748b'], angle: 135 },
];

// Helper: Show tactile toast notification
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = 'toast';
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : '🍌';
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px) scale(0.95)';
    toast.style.transition = 'all 0.2s';
    setTimeout(() => toast.remove(), 200);
  }, 2400);
}

// Push state to undo stack
function recordUndo() {
  if (state.undoStack.length > 30) state.undoStack.shift();
  state.undoStack.push(JSON.stringify(state.elements));
}

// Undo action
function handleUndo() {
  if (state.undoStack.length === 0) {
    showToast('已无可撤销的操作');
    return;
  }
  const last = state.undoStack.pop();
  try {
    state.elements = JSON.parse(last);
    renderCanvas();
    renderLayersList();
    renderInspector();
    showToast('已撤销上一步操作');
  } catch (e) {
    console.error('Undo parse error', e);
  }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadInitialWorkflow();
});

function loadInitialWorkflow() {
  loadDemoData();
}

function setupEventListeners() {
  // AI Generation Button
  document.getElementById('btnGenerate')?.addEventListener('click', handleGenerate);

  // Presets Chips
  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      const input = document.getElementById('promptInput');
      if (input && prompt) {
        input.value = prompt;
        input.focus();
        showToast('已填入预置提示词，点击“一键生成图像”开始生成');
      }
    });
  });

  // Local File Upload
  document.getElementById('fileUpload')?.addEventListener('change', handleFileUpload);

  // Demo Load Button
  document.getElementById('btnLoadDemo')?.addEventListener('click', loadDemoData);

  // Export Dropdown
  const btnExportDropdown = document.getElementById('btnExportDropdown');
  const exportMenu = document.getElementById('exportMenu');
  if (btnExportDropdown && exportMenu) {
    btnExportDropdown.addEventListener('click', (e) => {
      e.stopPropagation();
      exportMenu.style.display = exportMenu.style.display === 'block' ? 'none' : 'block';
    });
    document.getElementById('btnExportDrawio')?.addEventListener('click', handleExportDrawIO);
    document.getElementById('btnExportPNG')?.addEventListener('click', () => handleExportRenderedImage('png'));
    document.getElementById('btnExportJPG')?.addEventListener('click', () => handleExportRenderedImage('jpg'));
    document.addEventListener('click', () => { exportMenu.style.display = 'none'; });
  }

  // Image replacement upload input
  document.getElementById('replaceUploadInput')?.addEventListener('change', handleElementImageUpload);
  // Add Element Dropdown Menu
  const btnAdd = document.getElementById('btnAddElementDropdown');
  const addMenu = document.getElementById('addElementMenu');
  if (btnAdd && addMenu) {
    btnAdd.addEventListener('click', (e) => {
      e.stopPropagation();
      addMenu.style.display = addMenu.style.display === 'block' ? 'none' : 'block';
    });
    document.addEventListener('click', () => { addMenu.style.display = 'none'; });
    addMenu.querySelectorAll('.menu-item').forEach(item => {
      item.addEventListener('click', () => {
        const type = item.getAttribute('data-add');
        addNewElement(type);
        addMenu.style.display = 'none';
      });
    });
  }

  // Category Tag Filter Buttons
  document.querySelectorAll('.tag-filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tag-filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeFilter = btn.getAttribute('data-filter') || 'all';
      renderLayersList();
    });
  });

  // Tool Mode (Select vs Pan)
  document.getElementById('btnToolSelect')?.addEventListener('click', (e) => {
    e.stopPropagation();
    setToolMode('select');
  });
  document.getElementById('btnToolPan')?.addEventListener('click', (e) => {
    e.stopPropagation();
    setToolMode(state.toolMode === 'pan' ? 'select' : 'pan');
  });
  // Toggle View Overlay Mode
  const btnToggleOverlay = document.getElementById('btnToggleOverlay');
  btnToggleOverlay?.addEventListener('click', () => {
    state.viewMode = state.viewMode === 'overlay' ? 'vector' : 'overlay';
    btnToggleOverlay.textContent = state.viewMode === 'overlay' ? '👁️ 交互高亮模式' : '🎨 纯矢量画板模式';
    const bgImg = document.getElementById('artboardBgImg');
    if (bgImg) bgImg.style.display = state.viewMode === 'overlay' ? 'block' : 'none';
    renderCanvas();
    showToast(`已切换为: ${state.viewMode === 'overlay' ? '交互高亮模式 (底图100%可见)' : '纯矢量画板模式'}`);
  });

  // Toggle YOLO Detection Tags Overlay
  const btnToggleYoloTags = document.getElementById('btnToggleYoloTags');
  btnToggleYoloTags?.addEventListener('click', () => {
    state.showYoloTags = !state.showYoloTags;
    btnToggleYoloTags.classList.toggle('active', state.showYoloTags);
    renderCanvas();
    showToast(state.showYoloTags ? '已开启 YOLO 风格多色检测标签' : '已隐藏检测标签');
  });

  // Zoom HUD Controls
  document.getElementById('btnZoomIn')?.addEventListener('click', () => adjustZoom(0.15));
  document.getElementById('btnZoomOut')?.addEventListener('click', () => adjustZoom(-0.15));
  document.getElementById('btnZoomFit')?.addEventListener('click', fitZoomToScreen);

  // Background Raster Opacity Slider
  const bgOpacitySlider = document.getElementById('bgOpacitySlider');
  if (bgOpacitySlider) {
    bgOpacitySlider.addEventListener('input', (e) => {
      state.bgOpacity = parseInt(e.target.value, 10) / 100.0;
      const bgImg = document.getElementById('artboardBgImg');
      if (bgImg) bgImg.style.opacity = state.bgOpacity;
    });
  }

  // Floating Regional Scan Action Button on Marquee
  const marqueeActionBtn = document.getElementById('marqueeActionBtn');
  if (marqueeActionBtn) {
    marqueeActionBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (state.lastMarqueeBox) {
        triggerRegionalDeepScan(state.lastMarqueeBox);
        marqueeActionBtn.style.display = 'none';
      }
    });
  }

  // Canvas Stage Event Handlers (Pan, Marquee, Drag, Zoom)
  const container = document.getElementById('stageContainer');
  if (container) {
    container.addEventListener('mousedown', handleStageMouseDown);
    container.addEventListener('wheel', handleStageWheel, { passive: false });
  }

  // Photoshop-Style Right-Click Context Menu
  const artboard = document.getElementById('artboard');
  if (artboard) {
    artboard.addEventListener('contextmenu', handleArtboardContextMenu);
  }
  document.addEventListener('click', () => { hideContextMenu(); });

  window.addEventListener('mousemove', handleWindowMouseMove);
  window.addEventListener('mouseup', handleWindowMouseUp);

  // Resize Handles
  document.querySelectorAll('.resize-handle').forEach(handle => {
    handle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      startResizing(e, handle.getAttribute('data-handle'));
    });
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', handleKeyDown);
  window.addEventListener('keyup', handleKeyUp);
}

// Tool Mode Switching (Select vs Pan)
function setToolMode(mode) {
  state.toolMode = mode;
  const btnSelect = document.getElementById('btnToolSelect');
  const btnPan = document.getElementById('btnToolPan');
  if (btnSelect) btnSelect.classList.toggle('active', mode === 'select');
  if (btnPan) btnPan.classList.toggle('active', mode === 'pan');
  const container = document.getElementById('stageContainer');
  const artboard = document.getElementById('artboard');
  if (container) {
    container.classList.toggle('panning', mode === 'pan');
    container.classList.remove('is-panning');
  }
  if (artboard) {
    artboard.style.cursor = mode === 'pan' ? 'grab' : 'default';
  }
  showToast(mode === 'pan' ? '🖐️ 抓手平移模式 (可鼠标拖动全景画布)' : '🎯 光标选择编辑模式');
}

// Canvas Pan & Zoom Transform
function updateCanvasTransform() {
  const artboard = document.getElementById('artboard');
  if (artboard) {
    artboard.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  }
  const text = document.getElementById('zoomLevelText');
  if (text) text.textContent = `${Math.round(state.zoom * 100)}%`;
  updateTransformBox();
}

function adjustZoom(delta, originX, originY) {
  const oldZoom = state.zoom;
  const newZoom = Math.max(0.15, Math.min(3.0, oldZoom + delta));
  if (newZoom === oldZoom) return;

  const container = document.getElementById('stageContainer');
  if (!container) return;

  const cx = originX !== undefined ? originX : container.clientWidth / 2;
  const cy = originY !== undefined ? originY : container.clientHeight / 2;

  const worldX = (cx - state.panX) / oldZoom;
  const worldY = (cy - state.panY) / oldZoom;

  state.zoom = newZoom;
  state.panX = Math.round(cx - worldX * newZoom);
  state.panY = Math.round(cy - worldY * newZoom);
  updateCanvasTransform();
}

function fitZoomToScreen() {
  const container = document.getElementById('stageContainer');
  if (!container) return;
  const availW = container.clientWidth;
  const availH = container.clientHeight;
  const pad = 60;
  state.zoom = Math.min(1.0, Math.min((availW - pad) / state.canvasWidth, (availH - pad) / state.canvasHeight));
  state.panX = Math.max(10, Math.round((availW - state.canvasWidth * state.zoom) / 2));
  state.panY = Math.max(10, Math.round((availH - state.canvasHeight * state.zoom) / 2));
  updateCanvasTransform();
}

// Stage Mouse Handlers for Pan, Marquee, Drag
function handleStageMouseDown(e) {
  if (e.target.closest('.stage-hud') || e.target.closest('.context-menu') || e.target.closest('.marquee-action-btn') || e.target.closest('.dropdown') || e.target.closest('.inline-canvas-editor')) {
    return;
  }
  hideContextMenu();

  // 1. Pan with Space, Middle Click, or Hand Tool
  if (state.toolMode === 'pan' || state.isSpaceDown || e.button === 1 || e.target.id === 'stageContainer') {
    state.isPanning = true;
    state.panStart = { x: e.clientX, y: e.clientY, initPanX: state.panX, initPanY: state.panY };
    document.getElementById('stageContainer')?.classList.add('is-panning');
    return;
  }

  // 2. Click on Vector Elements or Artboard
  const nodeEl = e.target.closest('.vector-node');
  const stagePoint = getStageCoordinates(e);

  if (nodeEl) {
    const id = nodeEl.getAttribute('data-id');
    const el = state.elements.find(item => item.id === id);
    const wasSelected = state.selectedIds.has(id);

    if (!wasSelected) {
      selectElement(id, e.shiftKey);
    } else if (el && ['title', 'content', 'text', 'label', 'formula'].includes(el.type)) {
      // Second click on already selected text element: activate inline canvas editing!
      activateInlineEdit(el);
      return;
    }
    // Start Element Drag
    state.isDraggingElements = true;
    state.dragStart = stagePoint;
    state.dragElementsInitial.clear();
    state.elements.forEach(el => {
      if (state.selectedIds.has(el.id)) {
        state.dragElementsInitial.set(el.id, {
          x: el.x, y: el.y,
          start: el.arrow_start ? [...el.arrow_start] : null,
          end: el.arrow_end ? [...el.arrow_end] : null,
          wpts: el.vector_points ? el.vector_points.map(p => [...p]) : []
        });
      }
    });
  } else {
    // Click on empty artboard area: start Marquee selection
    if (!e.shiftKey) state.selectedIds.clear();
    state.isMarquee = true;
    state.marqueeStart = stagePoint;
    const marqueeBox = document.getElementById('marqueeBox');
    const marqueeBtn = document.getElementById('marqueeActionBtn');
    if (marqueeBtn) marqueeBtn.style.display = 'none';

    if (marqueeBox) {
      marqueeBox.style.display = 'block';
      marqueeBox.style.left = `${stagePoint.x}px`;
      marqueeBox.style.top = `${stagePoint.y}px`;
      marqueeBox.style.width = '0px';
      marqueeBox.style.height = '0px';
    }
    renderCanvas();
    renderLayersList();
    renderInspector();
  }
}

function handleStageWheel(e) {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) {
    const rect = document.getElementById('stageContainer').getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const delta = e.deltaY < 0 ? 0.08 : -0.08;
    adjustZoom(delta, mouseX, mouseY);
  } else if (e.shiftKey) {
    state.panX -= e.deltaY || e.deltaX;
    updateCanvasTransform();
  } else {
    state.panY -= e.deltaY;
    updateCanvasTransform();
  }
}

function handleWindowMouseMove(e) {
  // 1. Panning Canvas
  if (state.isPanning) {
    const dx = e.clientX - state.panStart.x;
    const dy = e.clientY - state.panStart.y;
    state.panX = state.panStart.initPanX + dx;
    state.panY = state.panStart.initPanY + dy;
    updateCanvasTransform();
    return;
  }

  const stagePoint = getStageCoordinates(e);

  // 2. Marquee Selection
  if (state.isMarquee && state.marqueeStart) {
    const mx = Math.min(state.marqueeStart.x, stagePoint.x);
    const my = Math.min(state.marqueeStart.y, stagePoint.y);
    const mw = Math.abs(stagePoint.x - state.marqueeStart.x);
    const mh = Math.abs(stagePoint.y - state.marqueeStart.y);

    const marqueeBox = document.getElementById('marqueeBox');
    if (marqueeBox) {
      marqueeBox.style.left = `${mx}px`;
      marqueeBox.style.top = `${my}px`;
      marqueeBox.style.width = `${mw}px`;
      marqueeBox.style.height = `${mh}px`;
    }

    state.elements.forEach(el => {
      const elR = el.x + el.width;
      const elB = el.y + el.height;
      const intersect = !(el.x > mx + mw || elR < mx || el.y > my + mh || elB < my);
      if (intersect) state.selectedIds.add(el.id);
    });

    renderCanvas();
    renderLayersList();
  }

  // 3. Dragging Elements (Move)
  if (state.isDraggingElements && state.dragStart) {
    const dx = stagePoint.x - state.dragStart.x;
    const dy = stagePoint.y - state.dragStart.y;

    state.elements.forEach(el => {
      if (state.selectedIds.has(el.id) && state.dragElementsInitial.has(el.id)) {
        const init = state.dragElementsInitial.get(el.id);
        el.x = Math.round(init.x + dx);
        el.y = Math.round(init.y + dy);
        if (init.start && el.arrow_start) {
          el.arrow_start = [Math.round(init.start[0] + dx), Math.round(init.start[1] + dy)];
        }
        if (init.end && el.arrow_end) {
          el.arrow_end = [Math.round(init.end[0] + dx), Math.round(init.end[1] + dy)];
        }
        if (init.wpts && el.vector_points) {
          el.vector_points = init.wpts.map(p => [Math.round(p[0] + dx), Math.round(p[1] + dy)]);
        }
      }
    });

    renderCanvas();
  }

  // 4. Resizing Element
  if (state.isResizing && state.resizeInitialElem && state.resizeStartMouse) {
    const dx = stagePoint.x - state.resizeStartMouse.x;
    const dy = stagePoint.y - state.resizeStartMouse.y;
    const init = state.resizeInitialElem;
    const el = state.elements.find(e => e.id === init.id);
    if (!el) return;

    switch (state.resizeHandle) {
      case 'se':
        el.width = Math.max(30, init.width + dx);
        el.height = Math.max(20, init.height + dy);
        break;
      case 'e':
        el.width = Math.max(30, init.width + dx);
        break;
      case 's':
        el.height = Math.max(20, init.height + dy);
        break;
      case 'nw':
        el.x = init.x + dx;
        el.y = init.y + dy;
        el.width = Math.max(30, init.width - dx);
        el.height = Math.max(20, init.height - dy);
        break;
      case 'ne':
        el.y = init.y + dy;
        el.width = Math.max(30, init.width + dx);
        el.height = Math.max(20, init.height - dy);
        break;
      case 'sw':
        el.x = init.x + dx;
        el.width = Math.max(30, init.width - dx);
        el.height = Math.max(20, init.height + dy);
        break;
    }
    renderCanvas();
  }
}

function handleWindowMouseUp(e) {
  if (state.isPanning) {
    state.isPanning = false;
    document.getElementById('stageContainer')?.classList.remove('is-panning');
  }
  if (state.isMarquee && state.marqueeStart) {
    const stagePoint = getStageCoordinates(e);
    const mx = Math.min(state.marqueeStart.x, stagePoint.x);
    const my = Math.min(state.marqueeStart.y, stagePoint.y);
    const mw = Math.abs(stagePoint.x - state.marqueeStart.x);
    const mh = Math.abs(stagePoint.y - state.marqueeStart.y);

    state.isMarquee = false;
    state.marqueeStart = null;
    const marqueeBox = document.getElementById('marqueeBox');
    if (marqueeBox) marqueeBox.style.display = 'none';

    // If marquee box is drawn over a region, auto-snap to module and trigger deep scan
    if (mw >= 35 && mh >= 35) {
      let snappedBox = [mx, my, mx + mw, my + mh];
      const matchedMod = state.elements.find(el =>
        (el.type === 'module' || el.type === 'shape') &&
        Math.abs(el.x - mx) < 60 && Math.abs(el.y - my) < 60
      );
      if (matchedMod) {
        snappedBox = [matchedMod.x, matchedMod.y, matchedMod.x + matchedMod.width, matchedMod.y + matchedMod.height];
        state.selectedIds.add(matchedMod.id);
      }
      state.lastMarqueeBox = snappedBox;

      // Automatically trigger regional deep scan to extract inner text, images, formulas, and lines!
      triggerRegionalDeepScan(snappedBox);
    }
  }
  if (state.isDraggingElements) {
    state.isDraggingElements = false;
    state.dragStart = null;
    renderInspector();
  }
  if (state.isResizing) {
    state.isResizing = false;
    state.resizeHandle = null;
    state.resizeInitialElem = null;
    state.resizeStartMouse = null;
    renderInspector();
  }
}

function startResizing(e, handleName) {
  const elId = Array.from(state.selectedIds)[0];
  const el = state.elements.find(e => e.id === elId);
  if (!el) return;

  recordUndo();
  state.isResizing = true;
  state.resizeHandle = handleName;
  state.resizeInitialElem = { ...el };
  state.resizeStartMouse = getStageCoordinates(e);
}

// Convert Screen Event to Stage Canvas Coordinates
function getStageCoordinates(e) {
  const artboard = document.getElementById('artboard');
  if (!artboard) return { x: 0, y: 0 };
  const rect = artboard.getBoundingClientRect();
  return {
    x: Math.round((e.clientX - rect.left) / state.zoom),
    y: Math.round((e.clientY - rect.top) / state.zoom),
  };
}

// Set Background Image
function setBgImage(src) {
  const bgImg = document.getElementById('artboardBgImg');
  if (bgImg) {
    bgImg.src = src;
    bgImg.style.display = state.viewMode === 'overlay' ? 'block' : 'none';
    bgImg.style.opacity = state.bgOpacity;
  }
}

// Regional Deep Scan (光标框选自动识别范围内多层元素)
async function triggerRegionalDeepScan(cropBox) {
  showToast(`正在对框选区域 [${cropBox.join(', ')}] 进行多层结构与文本解构...`);
  try {
    const res = await fetch('/api/parse_region', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: 'output/_tmp_parse.png',
        crop_box: cropBox
      })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || '区域解构失败');

    if (data.elements && data.elements.length > 0) {
      recordUndo();
      // Add newly identified sub-elements to canvas
      data.elements.forEach(subEl => {
        state.elements.push(subEl);
        state.selectedIds.add(subEl.id);
      });
      renderCanvas();
      renderLayersList();
      renderInspector();
      showToast(`框选识别成功！自动提取出 ${data.elements.length} 个多层图元 (含文本、图像切片与公式)`, 'success');
    } else {
      showToast('所选区域未检测到细分图元');
    }
  } catch (err) {
    console.error(err);
    showToast(`区域识别失败: ${err.message}`, 'error');
  }
}

// Photoshop-Style Context Menu Handler
function handleArtboardContextMenu(e) {
  e.preventDefault();
  const nodeEl = e.target.closest('.vector-node');
  if (nodeEl) {
    const id = nodeEl.getAttribute('data-id');
    if (!state.selectedIds.has(id)) {
      selectElement(id, false);
    }
  }

  showContextMenuAt(e.clientX, e.clientY);
}

function showContextMenuAt(clientX, clientY) {
  const menu = document.getElementById('contextMenu');
  if (!menu) return;

  menu.style.display = 'block';
  menu.style.left = `${Math.min(window.innerWidth - 220, clientX + 2)}px`;
  menu.style.top = `${Math.min(window.innerHeight - 300, clientY + 2)}px`;

  // Bind context menu actions
  setupContextMenuActions();
}

function hideContextMenu() {
  const menu = document.getElementById('contextMenu');
  if (menu) menu.style.display = 'none';
}

function setupContextMenuActions() {
  const menu = document.getElementById('contextMenu');
  if (!menu) return;

  const getSelected = () => {
    const id = Array.from(state.selectedIds)[0];
    return state.elements.find(e => e.id === id);
  };

  menu.querySelectorAll('[data-action]').forEach(item => {
    item.onclick = (e) => {
      e.stopPropagation();
      const action = item.getAttribute('data-action');
      const el = getSelected();

      if (action === 'delete') {
        if (el) deleteElement(el.id);
      } else if (action === 'duplicate') {
        if (el) duplicateElement(el.id);
      } else if (action === 'replace') {
        if (el) replaceElement(el.id);
      } else if (action === 'upload-replace') {
        if (el) {
          document.getElementById('replaceUploadInput')?.click();
        } else {
          showToast('请先选中要替换的元素');
        }
      } else if (action === 'ai-replace') {
        if (el) {
          generateReplacementForElement(el);
        } else {
          showToast('请先选中要替换的元素');
        }
      } else if (action === 'deep-scan') {
        if (el) {
          triggerRegionalDeepScan([el.x, el.y, el.x + el.width, el.y + el.height]);
        } else if (state.lastMarqueeBox) {
          triggerRegionalDeepScan(state.lastMarqueeBox);
        } else {
          showToast('请先选择一个图元或在画板中框选范围');
        }
      }
      hideContextMenu();
    };
  });

  // Color Submenu Actions
  menu.querySelectorAll('[data-color-action]').forEach(item => {
    item.onclick = (e) => {
      e.stopPropagation();
      const cAction = item.getAttribute('data-color-action');
      const el = getSelected();
      if (!el) return;

      recordUndo();
      if (!el.style) el.style = {};

      if (cAction === 'text-blue') el.style.text_color = '#2563eb';
      else if (cAction === 'text-emerald') el.style.text_color = '#059669';
      else if (cAction === 'text-amber') el.style.text_color = '#d97706';
      else if (cAction === 'text-rose') el.style.text_color = '#e11d48';
      else if (cAction === 'fill-sunset') {
        el.style.fill_mode = 'gradient';
        el.style.gradient = { type: 'linear', angle: 135, stops: ['#f43f5e', '#fbbf24'] };
      } else if (cAction === 'fill-ocean') {
        el.style.fill_mode = 'gradient';
        el.style.gradient = { type: 'linear', angle: 135, stops: ['#3b82f6', '#06b6d4'] };
      } else if (cAction === 'fill-emerald') {
        el.style.fill_mode = 'gradient';
        el.style.gradient = { type: 'linear', angle: 135, stops: ['#10b981', '#a7f3d0'] };
      } else if (cAction === 'fill-wireframe') {
        el.style.fill_mode = 'wireframe';
        el.style.fill = 'transparent';
        el.style.gradient = null;
      }
      renderCanvas();
      renderInspector();
      hideContextMenu();
      showToast('颜色已更新', 'success');
    };
  });

  // Typography Submenu Actions
  menu.querySelectorAll('[data-typo]').forEach(item => {
    item.onclick = (e) => {
      e.stopPropagation();
      const typo = item.getAttribute('data-typo');
      const el = getSelected();
      if (!el) return;

      recordUndo();
      if (!el.style) el.style = {};

      if (typo === 'size-up') el.style.font_size = (el.style.font_size || 16) + 2;
      else if (typo === 'size-down') el.style.font_size = Math.max(10, (el.style.font_size || 16) - 2);
      else if (typo === 'font-inter') el.style.font_family = 'Inter, system-ui, sans-serif';
      else if (typo === 'font-georgia') el.style.font_family = 'Georgia, serif';
      else if (typo === 'font-mono') el.style.font_family = 'Fira Code, monospace';
      else if (typo === 'font-noto') el.style.font_family = 'Noto Sans SC, sans-serif';
      else if (typo === 'convert-latex') {
        el.type = 'formula';
        el.tag = 'formula';
        el.latex = el.label || 'y = f(x)';
      }
      renderCanvas();
      renderInspector();
      hideContextMenu();
      showToast('排版已应用', 'success');
    };
  });

  // Layer Ordering Actions (Photoshop Style)
  menu.querySelectorAll('[data-layer-action]').forEach(item => {
    item.onclick = (e) => {
      e.stopPropagation();
      const lAction = item.getAttribute('data-layer-action');
      const el = getSelected();
      if (!el) return;

      const idx = state.elements.findIndex(item => item.id === el.id);
      if (idx < 0) return;

      recordUndo();
      if (lAction === 'bring-front') {
        const [removed] = state.elements.splice(idx, 1);
        state.elements.push(removed);
      } else if (lAction === 'send-back') {
        const [removed] = state.elements.splice(idx, 1);
        state.elements.unshift(removed);
      } else if (lAction === 'bring-forward') {
        if (idx < state.elements.length - 1) {
          const temp = state.elements[idx];
          state.elements[idx] = state.elements[idx + 1];
          state.elements[idx + 1] = temp;
        }
      } else if (lAction === 'send-backward') {
        if (idx > 0) {
          const temp = state.elements[idx];
          state.elements[idx] = state.elements[idx - 1];
          state.elements[idx - 1] = temp;
        }
      }
      renderCanvas();
      renderLayersList();
      hideContextMenu();
      showToast('图层顺序已调整');
    };
  });
}

// Upload an image and replace only the selected image/module element
function handleElementImageUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  const selectedId = Array.from(state.selectedIds)[0];
  const el = state.elements.find(item => item.id === selectedId);
  if (!el) {
    showToast('请先选中要替换的元素', 'error');
    return;
  }
  const reader = new FileReader();
  reader.onload = (event) => {
    recordUndo();
    el.type = 'image_patch';
    el.tag = 'image';
    el.image_data = event.target.result;
    el.label = `Replaced: ${file.name}`;
    el.style = { ...(el.style || {}), stroke: '#f43f5e', stroke_width: 2, corner_radius: 4 };
    renderCanvas();
    renderLayersList();
    renderInspector();
    showToast(`已上传并替换元素：${file.name}`, 'success');
  };
  reader.readAsDataURL(file);
  e.target.value = '';
}

// Generate a new image using Magpie and replace only the selected element
async function generateReplacementForElement(el) {
  const defaultPrompt = `Clean professional replacement image for diagram element: ${el.label || el.type}. Square vector-style illustration, white background, no text.`;
  const prompt = window.prompt('输入替换该元素的 image_generation prompt：', defaultPrompt);
  if (!prompt) return;

  showToast('正在为当前选中元素生成替换图片...');
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model: document.getElementById('modelSelect')?.value || 'gemini-3.1-flash-image', size: '1024x1024' })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || '元素生图失败');

    recordUndo();
    el.type = 'image_patch';
    el.tag = 'image';
    el.image_data = data.data_url;
    el.label = `AI Generated: ${prompt.slice(0, 28)}`;
    el.style = { ...(el.style || {}), stroke: '#f43f5e', stroke_width: 2, corner_radius: 4 };
    renderCanvas();
    renderLayersList();
    renderInspector();
    showToast('AI 替换图片已生成并应用到该元素', 'success');
  } catch (err) {
    console.error(err);
    showToast(`AI 元素替换失败: ${err.message}`, 'error');
  }
}

// Render direct raster export (PNG/JPG) through backend renderer
async function handleExportRenderedImage(format) {
  const exportMenu = document.getElementById('exportMenu');
  if (exportMenu) exportMenu.style.display = 'none';
  showToast(`正在渲染并导出 ${format.toUpperCase()} 图片...`);
  try {
    const res = await fetch('/api/render_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        elements: state.elements,
        canvas_width: state.canvasWidth,
        canvas_height: state.canvasHeight,
        format,
        render_mode: state.viewMode,
        bg_image_path: 'output/_tmp_parse.png'
      })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || '图片渲染失败');

    const a = document.createElement('a');
    a.href = data.url;
    a.download = data.filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    showToast(`${format.toUpperCase()} 渲染图片导出成功`, 'success');
  } catch (err) {
    console.error(err);
    showToast(`图片导出失败: ${err.message}`, 'error');
  }
}

// Direct-on-Canvas Inline Text Editing
function activateInlineEdit(elem) {
  const artboard = document.getElementById('artboard');
  if (!artboard) return;
  document.getElementById('inlineTextEditor')?.remove();

  const isFormula = elem.type === 'formula';
  const input = document.createElement('textarea');
  input.id = 'inlineTextEditor';
  input.className = 'inline-canvas-editor';
  input.value = elem.latex || elem.label || '';
  input.style.left = `${elem.x}px`;
  input.style.top = `${elem.y}px`;
  input.style.width = `${Math.max(120, elem.width + 10)}px`;
  input.style.height = `${Math.max(34, elem.height + 10)}px`;
  input.style.fontSize = `${elem.style?.font_size || 14}px`;
  input.style.fontFamily = elem.style?.font_family || 'Inter, sans-serif';
  input.style.color = elem.style?.text_color || '#0f172a';

  input.addEventListener('input', (e) => {
    if (isFormula) {
      elem.latex = e.target.value;
      elem.label = e.target.value;
    } else {
      elem.label = e.target.value;
    }
    renderCanvas();
    renderLayersList();
  });

  input.addEventListener('blur', () => {
    input.remove();
    renderCanvas();
    renderInspector();
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' || (e.key === 'Enter' && !e.shiftKey)) {
      input.blur();
    }
  });

  artboard.appendChild(input);
  input.focus();
  input.select();
  showToast('正在行内编辑文本内容 (输入即时生效)...');
}

// API: Handle Image Generation via Magpie Gateway
async function handleGenerate() {
  const promptInput = document.getElementById('promptInput');
  const modelSelect = document.getElementById('modelSelect');
  const btn = document.getElementById('btnGenerate');

  const prompt = promptInput?.value?.trim();
  const model = modelSelect?.value || 'gemini-3.1-flash-image';

  if (!prompt) {
    showToast('请输入架构图或方案图提示词', 'error');
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span>⏳ 正在生成...</span>';
  showToast(`正在调用 Magpie 网关 (${model}) 生成高质量方案图...`);

  try {
    const genRes = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model, size: '1024x1024' })
    });
    const genData = await genRes.json();
    if (!genData.success) throw new Error(genData.detail || '生成失败');

    showToast('生图成功！正在执行多层结构化解构与矢量图元分类...', 'success');
    await parseAndLoadImage(genData.data_url, prompt);
  } catch (err) {
    console.error(err);
    showToast(`生图或解析失败: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>✨ 一键生成图像</span>';
  }
}

// API: Handle File Upload
async function handleFileUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (event) => {
    const dataUrl = event.target.result;
    showToast(`正在上传并多层解构: ${file.name}...`);
    await parseAndLoadImage(dataUrl, file.name);
  };
  reader.readAsDataURL(file);
}

// API: Parse Image & Load to Canvas
async function parseAndLoadImage(dataUrl, promptHints = '') {
  try {
    const parseRes = await fetch('/api/parse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: dataUrl,
        prompt_hints: promptHints
      })
    });
    const data = await parseRes.json();
    if (!data.success) throw new Error(data.detail || '矢量解析失败');

    recordUndo();
    state.canvasWidth = data.canvas_width || 1024;
    state.canvasHeight = data.canvas_height || 1024;
    state.elements = data.elements || [];
    state.selectedIds.clear();
    state.bgImageSrc = data.image_url || dataUrl;

    setBgImage(state.bgImageSrc);

    const artboard = document.getElementById('artboard');
    if (artboard) {
      artboard.style.width = `${state.canvasWidth}px`;
      artboard.style.height = `${state.canvasHeight}px`;
    }
    const svgStage = document.getElementById('svgStage');
    if (svgStage) {
      svgStage.setAttribute('viewBox', `0 0 ${state.canvasWidth} ${state.canvasHeight}`);
    }

    fitZoomToScreen();
    renderCanvas();
    renderLayersList();
    renderInspector();
    showToast(`多层结构化解构完成！共提取 ${state.elements.length} 个结构化图元，底图清晰可见`, 'success');
  } catch (err) {
    console.error(err);
    showToast(`解析失败: ${err.message}`, 'error');
  }
}

// Demo Data Loader (Rich Nature-style Multi-Omics Scheme)
function loadDemoData() {
  recordUndo();
  state.canvasWidth = 1024;
  state.canvasHeight = 1024;
  state.bgImageSrc = '/output/_tmp_parse.png';
  state.bgOpacity = 1.0;
  setBgImage(state.bgImageSrc);

  state.elements = [
    {
      id: 'title_1',
      type: 'title',
      tag: 'title',
      label: 'ViT-DRIVEN MULTI-OMICS FRAMEWORK FOR NAT ESTIMATION',
      x: 35, y: 30, width: 954, height: 60,
      style: {
        fill: 'transparent',
        stroke: '#ea580c',
        stroke_width: 1.5,
        font_family: 'Inter, system-ui, sans-serif',
        font_size: 20,
        font_weight: '700',
        text_color: '#0f172a',
        align: 'center',
      }
    },
    {
      id: 'mod_1',
      type: 'module',
      tag: 'module',
      label: 'Input Data & Multi-layer Integration',
      x: 45, y: 135, width: 290, height: 265,
      style: {
        fill: 'transparent',
        fill_mode: 'wireframe',
        stroke: '#2563eb',
        stroke_width: 2,
        corner_radius: 8,
        font_family: 'Inter, sans-serif',
        font_size: 15,
        font_weight: '600',
        text_color: '#1e40af',
        align: 'center',
      }
    },
    {
      id: 'cnt_1',
      type: 'content',
      tag: 'content',
      label: '(A1) Clinical & Epidemiological Data\n• Patient demographics & History\n(A2) Breast MRI Habitat Analysis\n(A3) Multi-Omics Genomic Profiling',
      x: 60, y: 195, width: 260, height: 180,
      style: {
        fill: 'transparent',
        stroke: '#8b5cf6',
        stroke_width: 1,
        font_family: 'Inter, sans-serif',
        font_size: 12,
        font_weight: 'normal',
        text_color: '#1e293b',
        align: 'left',
      }
    },
    {
      id: 'mod_2',
      type: 'module',
      tag: 'module',
      label: 'ViT-Driven Analytical Feature Extractor',
      x: 375, y: 135, width: 280, height: 265,
      style: {
        fill: 'transparent',
        fill_mode: 'wireframe',
        stroke: '#2563eb',
        stroke_width: 2,
        corner_radius: 8,
        font_family: 'Inter, sans-serif',
        font_size: 15,
        font_weight: '600',
        text_color: '#1e40af',
        align: 'center',
      }
    },
    {
      id: 'formula_1',
      type: 'formula',
      tag: 'formula',
      label: '\\mathcal{L}_{total} = \\alpha \\mathcal{L}_{NAT} + \\beta \\mathcal{L}_{ViT} + \\gamma \\mathcal{L}_{omics}',
      latex: '\\mathcal{L}_{total} = \\alpha \\mathcal{L}_{NAT} + \\beta \\mathcal{L}_{ViT} + \\gamma \\mathcal{L}_{omics}',
      x: 390, y: 280, width: 250, height: 48,
      style: {
        fill: 'rgba(255, 255, 255, 0.95)',
        stroke: '#d97706',
        stroke_width: 2,
        corner_radius: 6,
        font_family: 'KaTeX_Main, serif',
        font_size: 14,
        text_color: '#b45309',
        align: 'center',
      }
    },
    {
      id: 'mod_3',
      type: 'module',
      tag: 'module',
      label: 'NAT Estimation & pCR Prediction',
      x: 695, y: 135, width: 285, height: 265,
      style: {
        fill: 'transparent',
        fill_mode: 'wireframe',
        stroke: '#2563eb',
        stroke_width: 2,
        corner_radius: 8,
        font_family: 'Inter, sans-serif',
        font_size: 15,
        font_weight: '600',
        text_color: '#1e40af',
        align: 'center',
      }
    },
    {
      id: 'sym_1',
      type: 'symbol',
      tag: 'symbol',
      label: 'pCR Biomarker',
      x: 775, y: 280, width: 125, height: 32,
      style: {
        fill: 'rgba(244, 63, 94, 0.15)',
        stroke: '#f43f5e',
        stroke_width: 1.5,
        corner_radius: 16,
        font_family: 'Inter, sans-serif',
        font_size: 12,
        font_weight: '700',
        text_color: '#e11d48',
        align: 'center',
      }
    },
    {
      id: 'arrow_1',
      type: 'arrow',
      tag: 'arrow',
      label: 'Multi-Modal Tokens',
      source_id: 'mod_1',
      target_id: 'mod_2',
      arrow_start: [335, 267],
      arrow_end: [375, 267],
      vector_points: [],
      is_rounded: false,
      style: { stroke: '#10b981', stroke_width: 2, font_size: 12, text_color: '#065f46' }
    },
    {
      id: 'arrow_2',
      type: 'arrow',
      tag: 'arrow',
      label: 'Latent Representation',
      source_id: 'mod_2',
      target_id: 'mod_3',
      arrow_start: [655, 267],
      arrow_end: [695, 267],
      vector_points: [],
      is_rounded: false,
      style: { stroke: '#10b981', stroke_width: 2, font_size: 12, text_color: '#065f46' }
    }
  ];

  state.selectedIds = new Set(['title_1']);
  fitZoomToScreen();
  renderCanvas();
  renderLayersList();
  renderInspector();
}

// Render SVG Vector Artboard
function renderCanvas() {
  const shapesLayer = document.getElementById('svgShapesLayer');
  const arrowsLayer = document.getElementById('svgArrowsLayer');
  const textsLayer = document.getElementById('svgTextsLayer');
  const svgDefs = document.getElementById('svgDefs');
  if (!shapesLayer || !arrowsLayer || !textsLayer) return;

  shapesLayer.innerHTML = '';
  arrowsLayer.innerHTML = '';
  textsLayer.innerHTML = '';

  // Render Defs for Gradients & Markers
  if (svgDefs) {
    let defsHtml = `
      <marker id="arrowHead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#10b981" />
      </marker>
      <marker id="arrowHeadSelected" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
      </marker>
    `;
    state.elements.forEach(el => {
      if (el.style?.gradient?.stops?.length >= 2) {
        const gid = `grad_${el.id}`;
        const g = el.style.gradient;
        const rad = (g.angle || 90) * Math.PI / 180;
        const x1 = Math.round(50 + Math.cos(rad) * 50);
        const y1 = Math.round(50 - Math.sin(rad) * 50);
        const x2 = Math.round(50 - Math.cos(rad) * 50);
        const y2 = Math.round(50 + Math.sin(rad) * 50);
        defsHtml += `
          <linearGradient id="${gid}" x1="${x1}%" y1="${y1}%" x2="${x2}%" y2="${y2}%">
            <stop offset="0%" stop-color="${g.stops[0]}" />
            <stop offset="100%" stop-color="${g.stops[1]}" />
          </linearGradient>
        `;
      }
    });
    svgDefs.innerHTML = defsHtml;
  }

  // 1. Render Arrows (Lines)
  state.elements.filter(el => el.type === 'arrow').forEach(arrow => {
    const isSelected = state.selectedIds.has(arrow.id);
    const start = arrow.arrow_start || [arrow.x, arrow.y];
    const end = arrow.arrow_end || [arrow.x + arrow.width, arrow.y + arrow.height];
    const wpts = arrow.vector_points || [];
    const pts = [start, ...wpts, end];

    let pathD = '';
    if (arrow.is_rounded && pts.length > 2) {
      pathD = buildRoundedPolylineD(pts, 10);
    } else {
      pathD = pts.map((p, idx) => (idx === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`)).join(' ');
    }

    const stroke = isSelected ? '#3b82f6' : (arrow.style?.stroke || YOLO_COLORS.arrow);
    const strokeW = isSelected ? (arrow.style?.stroke_width || 2) + 1 : (arrow.style?.stroke_width || 2);
    const marker = isSelected ? 'url(#arrowHeadSelected)' : 'url(#arrowHead)';

    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `vector-node ${isSelected ? 'selected' : ''}`);
    group.setAttribute('data-id', arrow.id);

    // Wide transparent stroke for reliable clicking
    const hitPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    hitPath.setAttribute('d', pathD);
    hitPath.setAttribute('fill', 'none');
    hitPath.setAttribute('stroke', 'transparent');
    hitPath.setAttribute('stroke-width', Math.max(16, strokeW + 10));
    group.appendChild(hitPath);

    // Visible Arrow Path
    const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathEl.setAttribute('d', pathD);
    pathEl.setAttribute('fill', 'none');
    pathEl.setAttribute('stroke', stroke);
    pathEl.setAttribute('stroke-width', strokeW);
    pathEl.setAttribute('marker-end', marker);
    pathEl.setAttribute('stroke-linejoin', 'round');
    pathEl.setAttribute('stroke-linecap', 'round');
    group.appendChild(pathEl);

    // Label if present
    if (arrow.label) {
      const mid = calculatePathMidpoint(pts);
      const textEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      textEl.setAttribute('x', mid[0]);
      textEl.setAttribute('y', mid[1] - 8);
      textEl.setAttribute('text-anchor', 'middle');
      textEl.setAttribute('font-size', '12');
      textEl.setAttribute('font-weight', '600');
      textEl.setAttribute('fill', arrow.style?.text_color || '#065f46');
      textEl.textContent = arrow.label;
      group.appendChild(textEl);
    }
    group.addEventListener('mousedown', (evt) => {
      evt.stopPropagation();
      selectElement(arrow.id, evt.shiftKey);
    });
    group.addEventListener('contextmenu', (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      selectElement(arrow.id, false);
      showContextMenuAt(evt.clientX, evt.clientY);
    });

    arrowsLayer.appendChild(group);
  });

  // 2. Render Modules & Shapes (Transparent Wireframe by default!)
  state.elements.filter(el => el.type === 'module' || el.type === 'shape' || el.type === 'container').forEach(shape => {
    const isSelected = state.selectedIds.has(shape.id);
    const yoloColor = YOLO_COLORS[shape.type] || YOLO_COLORS.module;
    
    // Fill mode: wireframe vs solid/gradient
    let fill = 'transparent';
    if (state.viewMode === 'vector' || shape.style?.fill_mode === 'solid' || shape.style?.fill_mode === 'gradient') {
      fill = shape.style?.gradient ? `url(#grad_${shape.id})` : (shape.style?.solid_color || shape.style?.fill || '#ffffff');
    } else {
      fill = shape.style?.fill === 'transparent' ? 'rgba(37, 99, 235, 0.04)' : (shape.style?.fill || 'transparent');
    }

    const stroke = isSelected ? '#3b82f6' : (shape.style?.stroke || yoloColor);
    const strokeW = isSelected ? 3 : (shape.style?.stroke_width || 2);
    const radius = shape.style?.corner_radius || 0;

    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `vector-node ${isSelected ? 'selected' : ''}`);
    group.setAttribute('data-id', shape.id);

    // Rect Box
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', shape.x);
    rect.setAttribute('y', shape.y);
    rect.setAttribute('width', shape.width);
    rect.setAttribute('height', shape.height);
    rect.setAttribute('rx', radius);
    rect.setAttribute('ry', radius);
    rect.setAttribute('fill', fill);
    rect.setAttribute('stroke', stroke);
    rect.setAttribute('stroke-width', strokeW);
    if (shape.style?.stroke_style === 'dashed') {
      rect.setAttribute('stroke-dasharray', '5,5');
    }
    group.appendChild(rect);

    // YOLO Detection Tag Badge
    if (state.showYoloTags) {
      const chip = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      chip.setAttribute('transform', `translate(${shape.x + 4}, ${shape.y - 11})`);
      const chipBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      const labelText = shape.label ? (shape.label.length > 20 ? shape.label.slice(0, 18) + '...' : shape.label) : 'Module';
      const badgeW = Math.max(56, labelText.length * 7 + 16);
      chipBg.setAttribute('width', badgeW);
      chipBg.setAttribute('height', '20');
      chipBg.setAttribute('rx', '4');
      chipBg.setAttribute('fill', isSelected ? '#2563eb' : yoloColor);
      const chipText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      chipText.setAttribute('x', '8');
      chipText.setAttribute('y', '14');
      chipText.setAttribute('font-size', '11');
      chipText.setAttribute('font-weight', '700');
      chipText.setAttribute('fill', '#ffffff');
      chipText.textContent = labelText;
      chip.appendChild(chipBg);
      chip.appendChild(chipText);
      group.appendChild(chip);
    }
    group.addEventListener('mousedown', (evt) => {
      evt.stopPropagation();
      selectElement(shape.id, evt.shiftKey);
    });
    group.addEventListener('dblclick', (evt) => {
      evt.stopPropagation();
      activateInlineEdit(shape);
    });
    group.addEventListener('contextmenu', (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      selectElement(shape.id, false);
      showContextMenuAt(evt.clientX, evt.clientY);
    });

    shapesLayer.appendChild(group);
  });

  // 3. Render Image Patches (Crop Patches e.g. MRI, WSI, Heatmaps)
  state.elements.filter(el => el.type === 'image_patch').forEach(patch => {
    const isSelected = state.selectedIds.has(patch.id);
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `vector-node ${isSelected ? 'selected' : ''}`);
    group.setAttribute('data-id', patch.id);

    // Image element inside SVG
    if (patch.image_data) {
      const imgEl = document.createElementNS('http://www.w3.org/2000/svg', 'image');
      imgEl.setAttribute('x', patch.x);
      imgEl.setAttribute('y', patch.y);
      imgEl.setAttribute('width', patch.width);
      imgEl.setAttribute('height', patch.height);
      imgEl.setAttribute('href', patch.image_data);
      imgEl.setAttribute('preserveAspectRatio', 'none');
      group.appendChild(imgEl);
    }

    // Border
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', patch.x);
    rect.setAttribute('y', patch.y);
    rect.setAttribute('width', patch.width);
    rect.setAttribute('height', patch.height);
    rect.setAttribute('fill', 'transparent');
    rect.setAttribute('stroke', isSelected ? '#3b82f6' : YOLO_COLORS.image_patch);
    rect.setAttribute('stroke-width', isSelected ? '3' : '2');
    group.appendChild(rect);

    if (state.showYoloTags) {
      const chip = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      chip.setAttribute('transform', `translate(${patch.x + 4}, ${patch.y - 11})`);
      const chipBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      chipBg.setAttribute('width', '68');
      chipBg.setAttribute('height', '18');
      chipBg.setAttribute('rx', '3');
      chipBg.setAttribute('fill', YOLO_COLORS.image_patch);
      const chipText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      chipText.setAttribute('x', '6');
      chipText.setAttribute('y', '13');
      chipText.setAttribute('font-size', '10');
      chipText.setAttribute('font-weight', '700');
      chipText.setAttribute('fill', '#ffffff');
      chipText.textContent = '🖼️ 图像切片';
      chip.appendChild(chipBg);
      chip.appendChild(chipText);
      group.appendChild(chip);
    }
    group.addEventListener('mousedown', (evt) => {
      evt.stopPropagation();
      selectElement(patch.id, evt.shiftKey);
    });
    group.addEventListener('contextmenu', (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      selectElement(patch.id, false);
      showContextMenuAt(evt.clientX, evt.clientY);
    });

    shapesLayer.appendChild(group);
  });

  // 4. Render Titles, Contents, Formulas, and Symbols
  state.elements.filter(el => ['title', 'content', 'text', 'label', 'formula', 'symbol'].includes(el.type)).forEach(tElem => {
    const isSelected = state.selectedIds.has(tElem.id);
    const yoloColor = YOLO_COLORS[tElem.type] || YOLO_COLORS.text;

    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `vector-node ${isSelected ? 'selected' : ''}`);
    group.setAttribute('data-id', tElem.id);

    // Hit box
    const hitRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    hitRect.setAttribute('x', tElem.x);
    hitRect.setAttribute('y', tElem.y);
    hitRect.setAttribute('width', tElem.width);
    hitRect.setAttribute('height', tElem.height);
    hitRect.setAttribute('rx', tElem.style?.corner_radius || 4);
    hitRect.setAttribute('fill', isSelected ? 'rgba(59, 130, 246, 0.12)' : (tElem.style?.fill || 'transparent'));
    hitRect.setAttribute('stroke', isSelected ? '#3b82f6' : (state.showYoloTags ? yoloColor : (tElem.style?.stroke || 'none')));
    hitRect.setAttribute('stroke-width', isSelected ? '2' : (state.showYoloTags ? '1' : (tElem.style?.stroke_width || '0')));
    if (state.showYoloTags && !isSelected && tElem.type !== 'formula' && tElem.type !== 'symbol') {
      hitRect.setAttribute('stroke-dasharray', '3,3');
    }
    group.appendChild(hitRect);

    // YOLO Tag Pill
    if (state.showYoloTags && (tElem.type === 'title' || tElem.type === 'formula' || tElem.type === 'symbol')) {
      const chip = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      chip.setAttribute('transform', `translate(${tElem.x}, ${tElem.y - 12})`);
      const chipBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      const bText = tElem.type === 'title' ? '📌 标题' : tElem.type === 'formula' ? '📐 LaTeX公式' : '⚛️ 符号';
      chipBg.setAttribute('width', bText.length * 8 + 14);
      chipBg.setAttribute('height', '18');
      chipBg.setAttribute('rx', '3');
      chipBg.setAttribute('fill', yoloColor);
      const chipText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      chipText.setAttribute('x', '6');
      chipText.setAttribute('y', '13');
      chipText.setAttribute('font-size', '10');
      chipText.setAttribute('font-weight', '700');
      chipText.setAttribute('fill', '#ffffff');
      chipText.textContent = bText;
      chip.appendChild(chipBg);
      chip.appendChild(chipText);
      group.appendChild(chip);
    }

    if (tElem.type === 'formula' && tElem.latex && window.katex) {
      // Live KaTeX Math Rendering in SVG foreignObject
      const fo = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject');
      fo.setAttribute('x', tElem.x + 4);
      fo.setAttribute('y', tElem.y + 4);
      fo.setAttribute('width', Math.max(40, tElem.width - 8));
      fo.setAttribute('height', Math.max(20, tElem.height - 8));
      fo.style.pointerEvents = 'none';

      const mathDiv = document.createElement('div');
      mathDiv.style.width = '100%';
      mathDiv.style.height = '100%';
      mathDiv.style.display = 'flex';
      mathDiv.style.alignItems = 'center';
      mathDiv.style.justifyContent = 'center';
      mathDiv.style.color = tElem.style?.text_color || '#b45309';
      mathDiv.style.fontSize = `${tElem.style?.font_size || 14}px`;
      try {
        window.katex.render(tElem.latex, mathDiv, { throwOnError: false, displayMode: true });
      } catch (ke) {
        mathDiv.textContent = tElem.latex;
      }
      fo.appendChild(mathDiv);
      group.appendChild(fo);
    } else {
      // Multiline Text Rendering
      const textLines = (tElem.label || '').split('\n');
      const fontSize = tElem.style?.font_size || (tElem.type === 'title' ? 20 : 13);
      const lineHeight = fontSize * 1.35;
      const startY = tElem.y + fontSize + 4;

      textLines.forEach((line, lidx) => {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        const posX = tElem.style?.align === 'center' ? tElem.x + tElem.width / 2 : tElem.x + 8;
        text.setAttribute('x', posX);
        text.setAttribute('y', startY + lidx * lineHeight);
        text.setAttribute('text-anchor', tElem.style?.align || (tElem.type === 'title' ? 'center' : 'start'));
        text.setAttribute('font-family', tElem.style?.font_family || 'Inter, sans-serif');
        text.setAttribute('font-size', fontSize);
        text.setAttribute('font-weight', tElem.style?.font_weight || (tElem.type === 'title' ? '700' : 'normal'));
        text.setAttribute('fill', isSelected ? '#3b82f6' : (tElem.style?.text_color || '#0f172a'));
        text.setAttribute('pointer-events', 'none');
        text.textContent = line;
        group.appendChild(text);
      });
    }

    // Attach direct element-level click handlers to the image/shape/text group
    group.addEventListener('mousedown', (evt) => {
      evt.stopPropagation();
      selectElement(tElem.id, evt.shiftKey);
    });
    group.addEventListener('dblclick', (evt) => {
      evt.stopPropagation();
      activateInlineEdit(tElem);
    });
    group.addEventListener('contextmenu', (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      selectElement(tElem.id, false);
      showContextMenuAt(evt.clientX, evt.clientY);
    });

    textsLayer.appendChild(group);
  });

  updateTransformBox();
}

// Build Rounded Polyline Path
function buildRoundedPolylineD(pts, radius = 10) {
  if (pts.length <= 2) {
    return pts.map((p, idx) => (idx === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`)).join(' ');
  }
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const pPrev = pts[i - 1];
    const pCurr = pts[i];
    const pNext = pts[i + 1];

    const d1x = pCurr[0] - pPrev[0];
    const d1y = pCurr[1] - pPrev[1];
    const len1 = Math.hypot(d1x, d1y);

    const d2x = pNext[0] - pCurr[0];
    const d2y = pNext[1] - pPrev[1];
    const len2 = Math.hypot(d2x, d2y);

    const r = Math.min(radius, len1 / 2, len2 / 2);
    if (r <= 0.1) {
      d += ` L ${pCurr[0]} ${pCurr[1]}`;
      continue;
    }

    const startX = pCurr[0] - (d1x / len1) * r;
    const startY = pCurr[1] - (d1y / len1) * r;
    const endX = pCurr[0] + (d2x / len2) * r;
    const endY = pCurr[1] + (d2y / len2) * r;

    d += ` L ${startX} ${startY}`;
    d += ` Q ${pCurr[0]} ${pCurr[1]} ${endX} ${endY}`;
  }
  d += ` L ${pts[pts.length - 1][0]} ${pts[pts.length - 1][1]}`;
  return d;
}

function calculatePathMidpoint(pts) {
  if (pts.length === 2) {
    return [(pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2];
  }
  const midIdx = Math.floor(pts.length / 2);
  return pts[midIdx];
}

// Update Transform Box Position
function updateTransformBox() {
  const box = document.getElementById('transformBox');
  if (!box) return;

  if (state.selectedIds.size === 0) {
    box.style.display = 'none';
    return;
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  state.elements.forEach(el => {
    if (state.selectedIds.has(el.id)) {
      minX = Math.min(minX, el.x);
      minY = Math.min(minY, el.y);
      maxX = Math.max(maxX, el.x + el.width);
      maxY = Math.max(maxY, el.y + el.height);
    }
  });

  if (minX === Infinity) {
    box.style.display = 'none';
    return;
  }

  box.style.display = 'block';
  box.style.left = `${minX}px`;
  box.style.top = `${minY}px`;
  box.style.width = `${maxX - minX}px`;
  box.style.height = `${maxY - minY}px`;

  const handles = box.querySelectorAll('.resize-handle');
  const isSingle = state.selectedIds.size === 1;
  handles.forEach(h => {
    h.style.display = isSingle ? 'block' : 'none';
  });
}

// Render Left Sidebar Layers & Filter Counts
function renderLayersList() {
  const list = document.getElementById('layerList');
  if (!list) return;

  const total = state.elements.length;
  const titles = state.elements.filter(e => e.type === 'title').length;
  const modules = state.elements.filter(e => e.type === 'module' || e.type === 'shape').length;
  const contents = state.elements.filter(e => e.type === 'content' || e.type === 'text').length;
  const formulas = state.elements.filter(e => e.type === 'formula').length;
  const arrows = state.elements.filter(e => e.type === 'arrow').length;
  const symbols = state.elements.filter(e => e.type === 'symbol').length;
  const imagePatches = state.elements.filter(e => e.type === 'image_patch').length;
  const groupText = state.elements.filter(e => ['title', 'content', 'text', 'label', 'formula'].includes(e.type)).length;
  const groupGraphic = state.elements.filter(e => ['module', 'shape', 'symbol', 'image_patch'].includes(e.type)).length;
  const groupLine = arrows;

  document.getElementById('layerTotalCount').textContent = `${total} 元素`;
  document.getElementById('countAll').textContent = total;
  document.getElementById('countGroupText').textContent = groupText;
  document.getElementById('countGroupGraphic').textContent = groupGraphic;
  document.getElementById('countGroupLine').textContent = groupLine;
  document.getElementById('countTitle').textContent = titles;
  document.getElementById('countModule').textContent = modules;
  document.getElementById('countFormula').textContent = formulas;
  document.getElementById('countImage').textContent = imagePatches;

  const filtered = state.elements.filter(el => {
    if (state.activeFilter === 'all') return true;
    if (state.activeFilter === 'group_text') return ['title', 'content', 'text', 'label', 'formula'].includes(el.type);
    if (state.activeFilter === 'group_graphic') return ['module', 'shape', 'symbol', 'image_patch'].includes(el.type);
    if (state.activeFilter === 'group_line') return el.type === 'arrow';
    return el.type === state.activeFilter || el.tag === state.activeFilter;
  });

  if (filtered.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <p>当前分类下暂无元素</p>
      </div>
    `;
    return;
  }

  list.innerHTML = '';
  filtered.forEach(el => {
    const isSelected = state.selectedIds.has(el.id);
    const item = document.createElement('div');
    item.className = `layer-item ${isSelected ? 'selected' : ''}`;
    item.setAttribute('data-id', el.id);

    const badgeClass = `badge-${el.tag || el.type || 'module'}`;
    const badgeText = el.type === 'title' ? '标题' : el.type === 'module' ? '模块' : el.type === 'content' ? '内容' : el.type === 'formula' ? '公式' : el.type === 'arrow' ? '连线' : el.type === 'image_patch' ? '图像' : '符号';

    const displayLabel = (el.label || el.latex || `图元 #${el.id}`).split('\n')[0];

    item.innerHTML = `
      <div class="layer-item-main">
        <span class="elem-badge ${badgeClass}">${badgeText}</span>
        <span class="layer-label" title="${displayLabel}">${displayLabel}</span>
      </div>
      <div style="display: flex; gap: 4px;">
        <button class="btn btn-secondary btn-sm" style="padding: 2px 6px;" title="选中定位">🎯</button>
        <button class="btn btn-danger btn-sm" style="padding: 2px 6px;" title="删除" data-action="delete">🗑️</button>
      </div>
    `;

    item.addEventListener('click', (e) => {
      if (e.target.closest('[data-action="delete"]')) {
        deleteElement(el.id);
        return;
      }
      selectElement(el.id, e.shiftKey);
    });

    list.appendChild(item);
  });
}

// Helper: Generate intelligent default prompt for element replacement
function getDefaultPromptForElement(el) {
  const label = (el.label || '').trim() || el.type || 'element';
  if (el.type === 'module') {
    return `Minimalist clean 2D vector icon or module illustration for: ${label.slice(0, 40)}. Flat scientific style, white background, no text.`;
  } else if (el.type === 'image_patch' || el.type === 'symbol') {
    return `Crisp medical/scientific illustration badge representing: ${label.slice(0, 40)}. Square composition, white background, high contrast, no text.`;
  }
  return `Clean professional illustration for diagram component: ${label.slice(0, 40)}. Vector style, white background, no text.`;
}

// Render Right Sidebar Inspector
function renderInspector() {
  const body = document.getElementById('inspectorBody');
  const badge = document.getElementById('selectionBadge');
  if (!body) return;

  if (state.selectedIds.size === 0) {
    if (badge) badge.style.display = 'none';
    body.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">👆</div>
        <p>请点击画板中元素进行点选<br>或在空白处拖拽框选<br><br><span style="font-size:11px;color:var(--text-muted)">💡 提示：右键任意位置呼出高级菜单<br>按住空格可任意平移画布</span></p>
      </div>
    `;
    return;
  }

  if (badge) {
    badge.style.display = 'inline-block';
    badge.textContent = `${state.selectedIds.size} 选中`;
  }

  // Single Selection Inspector
  if (state.selectedIds.size === 1) {
    const elId = Array.from(state.selectedIds)[0];
    const el = state.elements.find(e => e.id === elId);
    if (!el) return;

    const isFormula = el.type === 'formula';
    const isModule = el.type === 'module' || el.type === 'shape';
    const isTextLike = ['title', 'content', 'text', 'label'].includes(el.type);
    const isArrow = el.type === 'arrow';

    body.innerHTML = `
      <!-- Category Tag Section -->
      <div class="prop-section">
        <div class="prop-section-title">
          <span>图元类型分类 (YOLO TAG)</span>
          <span class="elem-badge badge-${el.tag || el.type}">${el.type}</span>
        </div>
        <div class="prop-row">
          <span class="prop-label">切换类型:</span>
          <select id="propTagSelect" class="prop-input" style="flex: 1;">
            <option value="title" ${el.type === 'title' ? 'selected' : ''}>📌 标题 (Title)</option>
            <option value="module" ${el.type === 'module' ? 'selected' : ''}>🔲 模块 (Module)</option>
            <option value="content" ${el.type === 'content' ? 'selected' : ''}>📝 内容说明 (Content)</option>
            <option value="formula" ${el.type === 'formula' ? 'selected' : ''}>📐 数学公式 (Formula/LaTeX)</option>
            <option value="arrow" ${el.type === 'arrow' ? 'selected' : ''}>➡️ 连线箭头 (Arrow)</option>
            <option value="symbol" ${el.type === 'symbol' ? 'selected' : ''}>⚛️ 符号徽标 (Symbol)</option>
          </select>
        </div>
      </div>

      ${isFormula ? `
      <!-- LaTeX Formula Suite -->
      <div class="prop-section">
        <div class="prop-section-title">
          <span>LaTeX 数学公式编辑器 (MATH SUITE)</span>
        </div>
        <textarea id="propFormulaInput" class="prop-textarea" style="font-family: var(--font-mono); font-size: 13px;" placeholder="输入 LaTeX 公式，例如：\\mathcal{L}_{total} = \\alpha \\mathcal{L}_1 + \\beta \\mathcal{L}_2">${el.latex || el.label || ''}</textarea>

        <div style="margin-top: 6px;">
          <span class="prop-label" style="display: block; margin-bottom: 4px;">KaTeX 实时公式预览:</span>
          <div id="katexFormulaPreview" style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; color: #0f172a; text-align: center; min-height: 48px; overflow-x: auto;"></div>
        </div>

        <div style="margin-top: 8px;">
          <span class="prop-label" style="display: block; margin-bottom: 4px;">快速数学符号插入:</span>
          <div style="display: flex; flex-wrap: wrap; gap: 4px;">
            ${['\\alpha', '\\beta', '\\gamma', '\\lambda', '\\sigma', '\\sum', '\\int', '\\frac{a}{b}', '\\sqrt{x}', '\\mathcal{L}', '\\to', '\\partial'].map(sym => `
              <button class="btn btn-secondary btn-sm math-sym-btn" data-sym="${sym}" style="padding: 2px 6px; font-family: var(--font-mono); font-size: 11px;">${sym}</button>
            `).join('')}
          </div>
        </div>
      </div>
      ` : ''}

      ${isTextLike || isModule ? `
      <!-- Text & Label Suite -->
      <div class="prop-section">
        <div class="prop-section-title">
          <span>文本内容与排版 (实时毫秒级响应)</span>
        </div>
        <textarea id="propTextLabel" class="prop-textarea" placeholder="输入或修改文字内容...">${el.label || ''}</textarea>
        
        <div class="prop-row" style="margin-top: 6px;">
          <span class="prop-label">字体选择:</span>
          <select id="propFontFamily" class="prop-input" style="flex: 1;">
            <option value="Inter, system-ui, sans-serif" ${el.style?.font_family?.includes('Inter') ? 'selected' : ''}>Inter / 现代黑体</option>
            <option value="'Roboto', sans-serif" ${el.style?.font_family?.includes('Roboto') ? 'selected' : ''}>Roboto / 无衬线</option>
            <option value="'Georgia', serif" ${el.style?.font_family?.includes('Georgia') ? 'selected' : ''}>Georgia / 优雅衬线</option>
            <option value="'Fira Code', monospace" ${el.style?.font_family?.includes('Fira') ? 'selected' : ''}>Fira Code / 等宽代码</option>
            <option value="'Noto Sans SC', sans-serif" ${el.style?.font_family?.includes('Noto') ? 'selected' : ''}>思源黑体 (Noto Sans)</option>
            <option value="'Songti SC', serif" ${el.style?.font_family?.includes('Songti') ? 'selected' : ''}>宋体 (Songti SC)</option>
          </select>
        </div>

        <div class="prop-row">
          <span class="prop-label">字号大小:</span>
          <div style="display: flex; align-items: center; gap: 6px; flex: 1;">
            <button id="btnFontDec" class="btn btn-secondary btn-sm" style="padding: 4px 8px;">-2</button>
            <input type="range" id="propFontSizeSlider" min="10" max="48" value="${el.style?.font_size || 16}" style="flex: 1; accent-color: var(--accent-banana); cursor: pointer;">
            <input type="number" id="propFontSizeInput" class="prop-input prop-input-sm" value="${el.style?.font_size || 16}">
            <button id="btnFontInc" class="btn btn-secondary btn-sm" style="padding: 4px 8px;">+2</button>
          </div>
        </div>

        <div class="prop-row">
          <span class="prop-label">文字颜色:</span>
          <input type="color" id="propTextColor" value="${rgbToHex(el.style?.text_color || '#0f172a')}" style="cursor: pointer; border: none; width: 40px; height: 28px; background: transparent;">
        </div>
      </div>
      ` : ''}

      ${isModule ? `
      <!-- Module & Shape Styling -->
      <div class="prop-section">
        <div class="prop-section-title">
          <span>模块背景填充 (透明线框 / 渐变色)</span>
        </div>
        
        <div class="prop-row">
          <span class="prop-label">填充模式:</span>
          <div style="display: flex; gap: 8px;">
            <label style="font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px;">
              <input type="radio" name="fillMode" value="wireframe" ${(!el.style?.fill_mode || el.style?.fill_mode === 'wireframe') ? 'checked' : ''}> 👁️ 透明线框 (不挡底图)
            </label>
            <label style="font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px;">
              <input type="radio" name="fillMode" value="solid" ${el.style?.fill_mode === 'solid' ? 'checked' : ''}> 纯色
            </label>
            <label style="font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px;">
              <input type="radio" name="fillMode" value="gradient" ${el.style?.fill_mode === 'gradient' ? 'checked' : ''}> 渐变
            </label>
          </div>
        </div>

        <div id="solidFillContainer" style="display: ${el.style?.fill_mode === 'solid' ? 'flex' : 'none'}; align-items: center; justify-content: space-between;">
          <span class="prop-label">纯色背景:</span>
          <input type="color" id="propFillColor" value="${rgbToHex(el.style?.solid_color || el.style?.fill || '#ffffff')}" style="cursor: pointer; border: none; width: 50px; height: 28px; background: transparent;">
        </div>

        <div id="gradientFillContainer" style="display: ${el.style?.fill_mode === 'gradient' ? 'flex' : 'none'}; flex-direction: column; gap: 8px;">
          <div class="prop-row">
            <span class="prop-label">渐变起始色:</span>
            <input type="color" id="propGradStop1" value="${el.style?.gradient?.stops?.[0] || '#3b82f6'}" style="cursor: pointer; border: none; width: 40px; height: 26px; background: transparent;">
            <span class="prop-label">终止色:</span>
            <input type="color" id="propGradStop2" value="${el.style?.gradient?.stops?.[1] || '#06b6d4'}" style="cursor: pointer; border: none; width: 40px; height: 26px; background: transparent;">
          </div>
          <div class="prop-row">
            <span class="prop-label">渐变角度:</span>
            <input type="range" id="propGradAngle" min="0" max="360" value="${el.style?.gradient?.angle || 135}" style="flex: 1; accent-color: var(--accent-banana);">
            <span id="gradAngleText" style="font-size: 11px; width: 32px; text-align: right;">${el.style?.gradient?.angle || 135}°</span>
          </div>
          <div>
            <span class="prop-label" style="display: block; margin-bottom: 4px;">快速渐变预置:</span>
            <div class="gradient-presets">
              ${GRADIENT_PRESETS.map((p, pidx) => `
                <div class="grad-pill" title="${p.name}" data-idx="${pidx}" style="background: linear-gradient(${p.angle}deg, ${p.stops[0]}, ${p.stops[1]});"></div>
              `).join('')}
            </div>
          </div>
        </div>

        <div class="prop-row" style="margin-top: 6px;">
          <span class="prop-label">边框颜色与粗细:</span>
          <div style="display: flex; align-items: center; gap: 6px;">
            <input type="color" id="propStrokeColor" value="${rgbToHex(el.style?.stroke || '#2563eb')}" style="cursor: pointer; border: none; width: 34px; height: 26px; background: transparent;">
            <input type="number" id="propStrokeWidth" class="prop-input prop-input-sm" min="0" max="10" value="${el.style?.stroke_width || 2}">
          </div>
        </div>
      </div>
      ` : ''}

      <!-- Geometry Position & Size -->
      <div class="prop-section">
        <div class="prop-section-title">
          <span>坐标与几何尺寸</span>
        </div>
        <div class="prop-row">
          <span class="prop-label">宽 / 高:</span>
          <div style="display: flex; gap: 6px;">
            <input type="number" id="propWidth" class="prop-input prop-input-sm" value="${Math.round(el.width)}">
            <input type="number" id="propHeight" class="prop-input prop-input-sm" value="${Math.round(el.height)}">
          </div>
        </div>
        <div class="prop-row">
          <span class="prop-label">X / Y:</span>
          <div style="display: flex; gap: 6px;">
            <input type="number" id="propX" class="prop-input prop-input-sm" value="${Math.round(el.x)}">
            <input type="number" id="propY" class="prop-input prop-input-sm" value="${Math.round(el.y)}">
          </div>
        </div>
      </div>

      <!-- Element Replacement Panel (Image Upload & Image Generation) -->
      <div class="prop-section" style="border-top: 1px dashed var(--border-color); padding-top: 10px; margin-top: 8px;">
        <div class="prop-section-title">
          <span>🖼️ 分元素图片与AI替换 (REPLACEMENT)</span>
        </div>
        
        ${el.image_data ? `
        <div style="margin-bottom: 8px; display: flex; align-items: center; gap: 8px; background: var(--bg-hover); padding: 6px; border-radius: 6px;">
          <img src="${el.image_data}" style="width: 44px; height: 44px; object-fit: contain; border-radius: 4px; border: 1px solid var(--border-color); background: #ffffff;" alt="element preview">
          <div style="flex: 1; font-size: 11px; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            已嵌入独立图片切片<br>
            <span style="color: var(--accent-banana); font-weight: 600;">尺寸: ${Math.round(el.width)}×${Math.round(el.height)} px</span>
          </div>
        </div>
        ` : ''}

        <!-- 1. Upload Local Image -->
        <button id="btnInspectorUploadReplace" class="btn btn-secondary btn-sm" style="width: 100%; justify-content: center; margin-bottom: 8px;">
          <span>📤 上传本地图片替换当前图元</span>
        </button>

        <!-- 2. Re-invoke image_generation for this single element -->
        <div style="background: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 6px; padding: 8px;">
          <span style="font-size: 11px; font-weight: 600; color: #b45309; display: block; margin-bottom: 4px;">✨ 单独重调 AI 生图替换 (image_generation):</span>
          <textarea id="aiElementPromptInput" class="prop-textarea" style="font-size: 12px; min-height: 48px;" placeholder="输入该分元素的专属生图提示词...">${getDefaultPromptForElement(el)}</textarea>
          
          <div class="prop-row" style="margin-top: 6px;">
            <select id="aiElementModelSelect" class="prop-input" style="flex: 1; font-size: 11px;">
              <option value="gemini-3.1-flash-image">Gemini 3.1 Flash Image (极速推荐)</option>
              <option value="gemini-2.5-flash-image">Gemini 2.5 Flash Image</option>
              <option value="flux-schnell">Flux Schnell (高质量写实)</option>
            </select>
          </div>
          <button id="btnInspectorAiReplace" class="btn btn-primary btn-sm" style="width: 100%; justify-content: center; margin-top: 6px;">
            <span>✨ 单独生成并替换当前元素</span>
          </button>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="prop-section" style="margin-top: 8px;">
        <div class="prop-section-title">
          <span>快捷操作 (ACTIONS)</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
          <button id="btnDuplicate" class="btn btn-secondary btn-sm" title="复制当前图元">📄 复制副本</button>
          <button id="btnReplace" class="btn btn-secondary btn-sm" title="替换当前图元形态">🔄 替换图元</button>
          <button id="btnBringForward" class="btn btn-secondary btn-sm" title="移到顶层">⬆️ 置于顶层</button>
          <button id="btnDeleteSelected" class="btn btn-danger btn-sm" title="删除选中的元素">🗑️ 删除元素</button>
        </div>
      </div>
    `;

    bindSingleInspectorHandlers(el);
  } else {
    // Multi-Selection Inspector
    body.innerHTML = `
      <div class="prop-section">
        <div class="prop-section-title">
          <span>批量选中 (${state.selectedIds.size} 元素)</span>
        </div>
        <p style="font-size: 12px; color: var(--text-muted);">已框选多个元素，可在画板中协同拖动或执行批量操作。</p>
        <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">
          <button id="btnBatchDelete" class="btn btn-danger btn-sm">🗑️ 批量删除选中元素</button>
          <button id="btnDeselectAll" class="btn btn-secondary btn-sm">✖️ 清除全部选择</button>
        </div>
      </div>
    `;

    document.getElementById('btnBatchDelete')?.addEventListener('click', () => {
      recordUndo();
      state.elements = state.elements.filter(e => !state.selectedIds.has(e.id));
      state.selectedIds.clear();
      renderCanvas();
      renderLayersList();
      renderInspector();
      showToast('已批量删除所选元素', 'success');
    });

    document.getElementById('btnDeselectAll')?.addEventListener('click', () => {
      state.selectedIds.clear();
      renderCanvas();
      renderLayersList();
      renderInspector();
    });
  }
}

// Bind Single Inspector Handlers with Instant Real-Time Two-Way Binding
function bindSingleInspectorHandlers(el) {
  // Label / Text Input (Real-time update on every keystroke)
  const propTextLabel = document.getElementById('propTextLabel');
  if (propTextLabel) {
    propTextLabel.addEventListener('input', (e) => {
      el.label = e.target.value;
      renderCanvas();
      renderLayersList();
    });
  }

  // Formula LaTeX Input with Real-time KaTeX preview
  const propFormulaInput = document.getElementById('propFormulaInput');
  const previewDiv = document.getElementById('katexFormulaPreview');
  const updateKaTeX = (latex) => {
    el.latex = latex;
    el.label = latex;
    if (previewDiv && window.katex) {
      try {
        window.katex.render(latex, previewDiv, { throwOnError: false, displayMode: true });
      } catch (err) {
        previewDiv.textContent = latex;
      }
    }
    renderCanvas();
    renderLayersList();
  };

  if (propFormulaInput) {
    updateKaTeX(propFormulaInput.value);
    propFormulaInput.addEventListener('input', (e) => updateKaTeX(e.target.value));
  }

  // Math symbol buttons
  document.querySelectorAll('.math-sym-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sym = btn.getAttribute('data-sym');
      if (propFormulaInput) {
        propFormulaInput.value += ' ' + sym;
        updateKaTeX(propFormulaInput.value);
      }
    });
  });

  // Tag Selector
  document.getElementById('propTagSelect')?.addEventListener('change', (e) => {
    recordUndo();
    el.type = e.target.value;
    el.tag = e.target.value;
    renderCanvas();
    renderLayersList();
    renderInspector();
    showToast(`已切换为: ${el.type}`);
  });

  // Font Family
  document.getElementById('propFontFamily')?.addEventListener('change', (e) => {
    recordUndo();
    if (!el.style) el.style = {};
    el.style.font_family = e.target.value;
    renderCanvas();
  });

  // Font Size
  const fontSlider = document.getElementById('propFontSizeSlider');
  const fontInput = document.getElementById('propFontSizeInput');
  const updateFontSize = (val) => {
    if (!el.style) el.style = {};
    el.style.font_size = parseInt(val, 10);
    if (fontSlider) fontSlider.value = val;
    if (fontInput) fontInput.value = val;
    renderCanvas();
  };
  fontSlider?.addEventListener('input', (e) => updateFontSize(e.target.value));
  fontInput?.addEventListener('input', (e) => updateFontSize(e.target.value));
  document.getElementById('btnFontDec')?.addEventListener('click', () => updateFontSize((el.style?.font_size || 16) - 2));
  document.getElementById('btnFontInc')?.addEventListener('click', () => updateFontSize((el.style?.font_size || 16) + 2));

  // Text Color
  document.getElementById('propTextColor')?.addEventListener('input', (e) => {
    if (!el.style) el.style = {};
    el.style.text_color = e.target.value;
    renderCanvas();
  });

  // Fill Mode Switch (Wireframe vs Solid vs Gradient)
  document.querySelectorAll('input[name="fillMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      recordUndo();
      const mode = e.target.value;
      if (!el.style) el.style = {};
      el.style.fill_mode = mode;
      const solidCont = document.getElementById('solidFillContainer');
      const gradCont = document.getElementById('gradientFillContainer');
      if (mode === 'gradient') {
        el.style.gradient = {
          type: 'linear',
          angle: 135,
          stops: [el.style.solid_color || '#3b82f6', '#06b6d4']
        };
        if (solidCont) solidCont.style.display = 'none';
        if (gradCont) gradCont.style.display = 'flex';
      } else if (mode === 'solid') {
        el.style.gradient = null;
        el.style.fill = el.style.solid_color || '#ffffff';
        if (solidCont) solidCont.style.display = 'flex';
        if (gradCont) gradCont.style.display = 'none';
      } else {
        el.style.gradient = null;
        el.style.fill = 'transparent';
        if (solidCont) solidCont.style.display = 'none';
        if (gradCont) gradCont.style.display = 'none';
      }
      renderCanvas();
    });
  });

  // Solid Color Fill
  document.getElementById('propFillColor')?.addEventListener('input', (e) => {
    if (!el.style) el.style = {};
    el.style.solid_color = e.target.value;
    el.style.fill = e.target.value;
    renderCanvas();
  });

  // Gradient Colors & Angle
  document.getElementById('propGradStop1')?.addEventListener('input', (e) => {
    if (!el.style.gradient) el.style.gradient = { stops: ['#3b82f6', '#06b6d4'], angle: 135 };
    el.style.gradient.stops[0] = e.target.value;
    renderCanvas();
  });
  document.getElementById('propGradStop2')?.addEventListener('input', (e) => {
    if (!el.style.gradient) el.style.gradient = { stops: ['#3b82f6', '#06b6d4'], angle: 135 };
    el.style.gradient.stops[1] = e.target.value;
    renderCanvas();
  });
  document.getElementById('propGradAngle')?.addEventListener('input', (e) => {
    if (!el.style.gradient) el.style.gradient = { stops: ['#3b82f6', '#06b6d4'], angle: 135 };
    el.style.gradient.angle = parseInt(e.target.value, 10);
    document.getElementById('gradAngleText').textContent = `${el.style.gradient.angle}°`;
    renderCanvas();
  });

  // Gradient Presets Pills
  document.querySelectorAll('.grad-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      recordUndo();
      const idx = parseInt(pill.getAttribute('data-idx'), 10);
      const preset = GRADIENT_PRESETS[idx];
      if (preset) {
        if (!el.style) el.style = {};
        el.style.fill_mode = 'gradient';
        el.style.gradient = {
          type: 'linear',
          angle: preset.angle,
          stops: [...preset.stops]
        };
        renderCanvas();
        renderInspector();
        showToast(`已应用渐变色: ${preset.name}`, 'success');
      }
    });
  });

  // Stroke Color, Width, and Style
  document.getElementById('propStrokeColor')?.addEventListener('input', (e) => {
    if (!el.style) el.style = {};
    el.style.stroke = e.target.value;
    renderCanvas();
  });
  document.getElementById('propStrokeWidth')?.addEventListener('input', (e) => {
    if (!el.style) el.style = {};
    el.style.stroke_width = parseInt(e.target.value, 10) || 1;
    renderCanvas();
  });

  // Dimensions & Position
  document.getElementById('propWidth')?.addEventListener('input', (e) => {
    el.width = Math.max(20, parseInt(e.target.value, 10) || 50);
    renderCanvas();
  });
  document.getElementById('propHeight')?.addEventListener('input', (e) => {
    el.height = Math.max(20, parseInt(e.target.value, 10) || 30);
    renderCanvas();
  });
  document.getElementById('propX')?.addEventListener('input', (e) => {
    el.x = parseInt(e.target.value, 10) || 0;
    renderCanvas();
  });
  document.getElementById('propY')?.addEventListener('input', (e) => {
    el.y = parseInt(e.target.value, 10) || 0;
    renderCanvas();
  });

  // Action Buttons
  document.getElementById('btnDuplicate')?.addEventListener('click', () => duplicateElement(el.id));
  document.getElementById('btnReplace')?.addEventListener('click', () => replaceElement(el.id));
  document.getElementById('btnBringForward')?.addEventListener('click', () => bringToFront(el.id));
  document.getElementById('btnDeleteSelected')?.addEventListener('click', () => deleteElement(el.id));

  // Inspector Replacement Handlers
  document.getElementById('btnInspectorUploadReplace')?.addEventListener('click', () => {
    document.getElementById('replaceUploadInput')?.click();
  });

  document.getElementById('btnInspectorAiReplace')?.addEventListener('click', async () => {
    const prompt = document.getElementById('aiElementPromptInput')?.value?.trim();
    const model = document.getElementById('aiElementModelSelect')?.value || 'gemini-3.1-flash-image';
    const btn = document.getElementById('btnInspectorAiReplace');
    if (!prompt) {
      showToast('请输入针对该元素的提示词', 'error');
      return;
    }
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span>⏳ 正在生成替换元素...</span>';
    }
    showToast(`正在通过 ${model} 单独为该元素生成替换图片...`);
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model, size: '1024x1024' })
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.detail || '生图失败');

      recordUndo();
      el.type = 'image_patch';
      el.tag = 'image';
      el.image_data = data.data_url;
      el.label = `AI Replaced: ${prompt.slice(0, 24)}`;
      el.style = { ...(el.style || {}), stroke: '#f43f5e', stroke_width: 2, corner_radius: 4 };
      renderCanvas();
      renderLayersList();
      renderInspector();
      showToast('当前分元素已成功由新生成的图片替换！', 'success');
    } catch (err) {
      console.error(err);
      showToast(`替换失败: ${err.message}`, 'error');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span>✨ 单独生成并替换当前元素</span>';
      }
    }
  });
}

// Add New Elements
function addNewElement(type) {
  recordUndo();
  const id = `user_${Date.now() % 100000}`;
  const container = document.getElementById('stageContainer');
  const cx = Math.round((container.clientWidth / 2 - state.panX) / state.zoom);
  const cy = Math.round((container.clientHeight / 2 - state.panY) / state.zoom);

  let newEl = null;
  if (type === 'title') {
    newEl = {
      id, type: 'title', tag: 'title', label: '新主标题 (SCHEME TITLE)',
      x: cx - 200, y: cy - 25, width: 400, height: 50,
      style: { fill: 'transparent', stroke: YOLO_COLORS.title, stroke_width: 1.5, font_size: 20, font_weight: '700', text_color: '#0f172a', align: 'center' }
    };
  } else if (type === 'module') {
    newEl = {
      id, type: 'module', tag: 'module', label: '新模块容器 (New Module)',
      x: cx - 110, y: cy - 60, width: 220, height: 120,
      style: { fill_mode: 'wireframe', fill: 'transparent', stroke: YOLO_COLORS.module, stroke_width: 2, corner_radius: 8, font_size: 15, font_weight: '600', text_color: '#1e40af', align: 'center' }
    };
  } else if (type === 'formula') {
    newEl = {
      id, type: 'formula', tag: 'formula', label: 'y = f(x)', latex: 'y = f(x)',
      x: cx - 80, y: cy - 25, width: 160, height: 45,
      style: { fill: 'rgba(255, 255, 255, 0.95)', stroke: YOLO_COLORS.formula, stroke_width: 1.5, corner_radius: 6, font_size: 14, text_color: '#b45309', align: 'center' }
    };
  } else if (type === 'arrow') {
    newEl = {
      id, type: 'arrow', tag: 'arrow', label: 'Flow',
      x: cx - 80, y: cy, width: 160, height: 10,
      arrow_start: [cx - 80, cy], arrow_end: [cx + 80, cy], vector_points: [], is_rounded: false,
      style: { stroke: YOLO_COLORS.arrow, stroke_width: 2, font_size: 12, text_color: '#065f46' }
    };
  } else if (type === 'symbol') {
    newEl = {
      id, type: 'symbol', tag: 'symbol', label: 'Biomarker',
      x: cx - 50, y: cy - 16, width: 100, height: 32,
      style: { fill: 'rgba(244, 63, 94, 0.15)', stroke: YOLO_COLORS.symbol, stroke_width: 1.5, corner_radius: 16, font_size: 12, font_weight: '700', text_color: '#e11d48', align: 'center' }
    };
  } else {
    // Content
    newEl = {
      id, type: 'content', tag: 'content', label: '新增段落内容说明...',
      x: cx - 120, y: cy - 20, width: 240, height: 40,
      style: { fill: 'transparent', stroke: YOLO_COLORS.content, stroke_width: 1, font_size: 13, font_weight: 'normal', text_color: '#334155', align: 'left' }
    };
  }

  state.elements.push(newEl);
  state.selectedIds.clear();
  state.selectedIds.add(id);
  renderCanvas();
  renderLayersList();
  renderInspector();
  showToast(`已添加新图元: ${newEl.label}`, 'success');
}

// Select Element
function selectElement(id, toggle = false) {
  if (toggle) {
    if (state.selectedIds.has(id)) state.selectedIds.delete(id);
    else state.selectedIds.add(id);
  } else {
    state.selectedIds.clear();
    state.selectedIds.add(id);
  }
  renderCanvas();
  renderLayersList();
  renderInspector();
}

// Element Operations
function deleteElement(id) {
  recordUndo();
  state.elements = state.elements.filter(e => e.id !== id);
  state.selectedIds.delete(id);
  renderCanvas();
  renderLayersList();
  renderInspector();
  showToast('已删除图元');
}

function duplicateElement(id) {
  recordUndo();
  const el = state.elements.find(e => e.id === id);
  if (!el) return;
  const newEl = JSON.parse(JSON.stringify(el));
  newEl.id = `user_${Date.now() % 100000}`;
  newEl.x += 24;
  newEl.y += 24;
  newEl.label = `${el.label || '副本'} (Copy)`;
  state.elements.push(newEl);
  state.selectedIds.clear();
  state.selectedIds.add(newEl.id);
  renderCanvas();
  renderLayersList();
  renderInspector();
  showToast('已复制图元副本', 'success');
}

function replaceElement(id) {
  recordUndo();
  const el = state.elements.find(e => e.id === id);
  if (!el) return;
  if (el.type === 'module' || el.type === 'shape') {
    el.style.stroke_style = el.style.stroke_style === 'dashed' ? 'solid' : 'dashed';
  }
  renderCanvas();
  renderInspector();
  showToast('已切换图元线框样式', 'success');
}

function bringToFront(id) {
  const idx = state.elements.findIndex(e => e.id === id);
  if (idx < 0) return;
  recordUndo();
  const [el] = state.elements.splice(idx, 1);
  state.elements.push(el);
  renderCanvas();
  renderLayersList();
  showToast('已置于顶层');
}

// Keyboard Navigation & Shortcuts
function handleKeyDown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.code === 'Space' && !state.isSpaceDown) {
    state.isSpaceDown = true;
    document.getElementById('stageContainer')?.classList.add('panning');
  }

  if (e.key === 'v' || e.key === 'V') {
    setToolMode('select');
  } else if (e.key === 'h' || e.key === 'H') {
    setToolMode('pan');
  } else if (e.key === 'f' || e.key === 'F') {
    fitZoomToScreen();
  } else if ((e.ctrlKey || e.metaKey) && (e.key === 'd' || e.key === 'D')) {
    e.preventDefault();
    const id = Array.from(state.selectedIds)[0];
    if (id) duplicateElement(id);
  } else if (e.key === ']' && state.selectedIds.size > 0) {
    const id = Array.from(state.selectedIds)[0];
    bringToFront(id);
  } else if (e.key === '[' && state.selectedIds.size > 0) {
    const id = Array.from(state.selectedIds)[0];
    const idx = state.elements.findIndex(e => e.id === id);
    if (idx > 0) {
      recordUndo();
      const [el] = state.elements.splice(idx, 1);
      state.elements.unshift(el);
      renderCanvas();
      renderLayersList();
      showToast('已置于底层');
    }
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (state.selectedIds.size > 0) {
      recordUndo();
      state.elements = state.elements.filter(el => !state.selectedIds.has(el.id));
      state.selectedIds.clear();
      renderCanvas();
      renderLayersList();
      renderInspector();
      showToast('已删除选中元素');
    }
  } else if (e.key === 'Escape') {
    hideContextMenu();
    state.selectedIds.clear();
    renderCanvas();
    renderLayersList();
    renderInspector();
  } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    handleUndo();
  } else if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
    e.preventDefault();
    state.elements.forEach(el => state.selectedIds.add(el.id));
    renderCanvas();
    renderLayersList();
    renderInspector();
  }
}

function handleKeyUp(e) {
  if (e.code === 'Space') {
    state.isSpaceDown = false;
    if (state.toolMode !== 'pan') {
      document.getElementById('stageContainer')?.classList.remove('panning');
    }
  }
}

// API: Export to DrawIO XML
async function handleExportDrawIO() {
  const btn = document.getElementById('btnExportDrawio');
  if (btn) btn.innerHTML = '<span>⏳ 导出中...</span>';

  try {
    const res = await fetch('/api/export_drawio', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        elements: state.elements,
        canvas_width: state.canvasWidth,
        canvas_height: state.canvasHeight
      })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || '导出失败');

    const blob = new Blob([data.xml], { type: 'application/xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `editbanana_${Date.now()}.drawio.xml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('DrawIO XML 导出成功！支持在 diagrams.net 打开拖拽与二次编辑', 'success');
  } catch (err) {
    console.error(err);
    showToast(`导出失败: ${err.message}`, 'error');
  } finally {
    if (btn) btn.innerHTML = '<span>💾 导出 DrawIO</span>';
  }
}

// Utility: RGB to HEX
function rgbToHex(color) {
  if (!color || color === 'transparent') return '#ffffff';
  if (color.startsWith('#')) return color.slice(0, 7);
  const match = color.match(/\d+/g);
  if (!match || match.length < 3) return '#ffffff';
  return '#' + match.slice(0, 3).map(x => parseInt(x, 10).toString(16).padStart(2, '0')).join('');
}
