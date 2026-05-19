// ceangal v2 runtime — clean single-loop architecture
//
// Rules:
//   1. ONE rAF loop (ScrollAnimator only)
//   2. ZERO DOM creation during scroll (transform only)
//   3. State change → rebuild DOM overlay
//   4. Scroll → WASM physics + GPU draw (no tree ops)

import { TTFFont } from "./ttf.js";
import { generateSDFAtlas } from "./sdf.js";

// ── Handle table (GPU objects = opaque ints in WASM) ──

const handles = [null];
const h = (obj) => { handles.push(obj); return handles.length - 1; };
const g = (id) => handles[Number(id)];
const B = (n) => BigInt(n);
const N = (b) => Number(b);

let _device, _context, _format, _wasmMemory;
let _font = null, _atlas = null;

const strings = [];
let strBuf = [];
let _dataChunks = [];
let _dataIsF32 = [];
let _bindingEntries = [];
let SHADERS = [];

// ── Tile-based item dispatch (computed in JS, injected into compute pass) ──
// Ref: https://www.w3.org/TR/webgpu/#dom-gpucomputepassencoder-setbindgroup
const DISPATCH_TILE = 64;
const MAX_ITEMS_PER_TILE = 32;
let _tileCountsBuf = null, _tileIdsBuf = null, _tileBindGroup = null;
let _computePipelineObj = null;
let _scrollY = 0, _vpWidth = 0, _vpHeight = 0;
let _lastItemsF32 = null, _lastItemCount = 0;
let _lastTilePipeline = null;

// ── WASM import namespaces ──

function createDomImports() {
  return {
    begin_str() { strBuf = []; },
    push_byte(b) { strBuf.push(N(b)); },
    commit_str() {
      strings.push(new TextDecoder().decode(new Uint8Array(strBuf)));
      return B(strings.length - 1);
    },
    create_element(tagId) { return B(h(document.createElement(strings[N(tagId)]))); },
    set_text(elId, textId) { g(elId).textContent = strings[N(textId)]; },
    set_attr(elId, nameId, valId) { g(elId).setAttribute(strings[N(nameId)], strings[N(valId)]); },
    set_style(elId, propId, valId) { g(elId).style[strings[N(propId)]] = strings[N(valId)]; },
    append_child(parentId, childId) { g(parentId).appendChild(g(childId)); },
    get_offset_width(elId) { return g(elId).offsetWidth; },
    clear_children(elId) { g(elId).innerHTML = ""; },
    log(strId) { console.log("[ceangal]", strings[N(strId)]); },
  };
}

function createGpuImports(canvas) {
  return {
    get_preferred_format: () => B(h(_format)),
    configure_canvas(deviceId, _fmtId) {
      _context = canvas.getContext("webgpu");
      _context.configure({ device: g(deviceId), format: _format, alphaMode: "premultiplied" });
      return B(h(_context));
    },
    create_shader(deviceId, shaderId, _) {
      return B(h(g(deviceId).createShaderModule({ code: SHADERS[N(shaderId)] || SHADERS[0] })));
    },
    create_buffer(deviceId, size, usage) {
      return B(h(g(deviceId).createBuffer({ size: N(size), usage: N(usage) })));
    },
    write_buffer(deviceId, bufferId, dataPtr, dataLen) {
      g(deviceId).queue.writeBuffer(g(bufferId), 0, new Uint8Array(_wasmMemory.buffer, N(dataPtr), N(dataLen)));
    },
    write_f32_at(deviceId, bufferId, byteOffset, value) {
      g(deviceId).queue.writeBuffer(g(bufferId), N(byteOffset), new Float32Array([value]));
      if (N(byteOffset) === 28) _scrollY = value; // track scroll_y for tile assignment
    },
    write_u32_at(deviceId, bufferId, byteOffset, value) {
      g(deviceId).queue.writeBuffer(g(bufferId), N(byteOffset), new Uint32Array([N(value)]));
    },
    create_compute_pipeline(deviceId, shaderId, _) {
      const p = g(deviceId).createComputePipeline({ layout: "auto", compute: { module: g(shaderId), entryPoint: "fine" } });
      _computePipelineObj = p;
      return B(h(p));
    },
    create_render_pipeline(deviceId, shaderId, _vp, _vl, _fp, _fl, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs_fullscreen" },
        fragment: { module: g(shaderId), entryPoint: "fs_fullscreen", targets: [{ format: _format }] },
        primitive: { topology: "triangle-list" },
      })));
    },
    create_text_pipeline(deviceId, shaderId, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs_main", buffers: [{ arrayStride: 32, attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32x2" },
          { shaderLocation: 2, offset: 16, format: "float32x4" },
        ]}] },
        fragment: { module: g(shaderId), entryPoint: "fs_main", targets: [{ format: _format, blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        }}] },
        primitive: { topology: "triangle-list" },
      })));
    },
    create_image_pipeline(deviceId, shaderId, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs_main", buffers: [{ arrayStride: 16, attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32x2" },
        ]}] },
        fragment: { module: g(shaderId), entryPoint: "fs_main", targets: [{ format: _format, blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        }}] },
        primitive: { topology: "triangle-list" },
      })));
    },
    begin_bindings() { _bindingEntries = []; },
    add_buffer_binding(bufferId) { _bindingEntries.push({ kind: "buffer", obj: g(bufferId) }); },
    add_texture_binding(textureId) { _bindingEntries.push({ kind: "texture", obj: g(textureId) }); },
    add_sampler_binding(samplerId) { _bindingEntries.push({ kind: "sampler", obj: g(samplerId) }); },
    create_bound_group(deviceId, pipelineId, groupIdx) {
      const layout = g(pipelineId).getBindGroupLayout(N(groupIdx));
      const entries = _bindingEntries.map((e, i) => {
        if (e.kind === "buffer") return { binding: i, resource: { buffer: e.obj } };
        if (e.kind === "texture") return { binding: i, resource: e.obj.createView() };
        if (e.kind === "sampler") return { binding: i, resource: e.obj };
      });
      _bindingEntries = [];
      return B(h(g(deviceId).createBindGroup({ layout, entries })));
    },
    set_bind_group(passId, index, bgId) { g(passId).setBindGroup(N(index), g(bgId)); },
    begin_encoder: (deviceId) => B(h(g(deviceId).createCommandEncoder())),
    begin_compute_pass: (encoderId) => B(h(g(encoderId).beginComputePass())),
    dispatch_workgroups(passId, x, y, z) {
      // Lazy init tile dispatch (handles first frame + resize)
      if (_computePipelineObj && _computePipelineObj !== _lastTilePipeline) {
        _setupTileDispatch(_device, canvas.width, canvas.height);
        _lastTilePipeline = _computePipelineObj;
      }
      if (_tileBindGroup) g(passId).setBindGroup(3, _tileBindGroup);
      g(passId).dispatchWorkgroups(N(x), N(y), N(z));
    },
    begin_render_pass(encoderId, r, g_, b, a) {
      return B(h(g(encoderId).beginRenderPass({
        colorAttachments: [{ view: _context.getCurrentTexture().createView(),
          clearValue: { r, g: g_, b, a }, loadOp: "clear", storeOp: "store" }],
      })));
    },
    set_pipeline(passId, pipelineId) { g(passId).setPipeline(g(pipelineId)); },
    draw(passId, n) { g(passId).draw(N(n)); },
    set_vertex_buffer(passId, slot, bufferId) { g(passId).setVertexBuffer(N(slot), g(bufferId)); },
    set_index_buffer(passId, bufferId) { g(passId).setIndexBuffer(g(bufferId), "uint32"); },
    draw_indexed(passId, count) { g(passId).drawIndexed(N(count)); },
    end_pass(passId) { g(passId).end(); },
    finish_and_submit(deviceId, encoderId) { g(deviceId).queue.submit([g(encoderId).finish()]); },
    begin_data() { _dataChunks = []; _dataIsF32 = []; },
    push_f32(v) { _dataChunks.push(v); _dataIsF32.push(true); },
    push_u32(v) { _dataChunks.push(N(v)); _dataIsF32.push(false); },
    flush_to_buffer(deviceId, bufferId) {
      const buf = new ArrayBuffer(_dataChunks.length * 4);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      for (let i = 0; i < _dataChunks.length; i++) {
        if (_dataIsF32[i]) f32[i] = _dataChunks[i]; else u32[i] = _dataChunks[i];
      }
      g(deviceId).queue.writeBuffer(g(bufferId), 0, new Uint8Array(buf));
      // Capture items data for tile assignment (256 items × 12 floats = 3072)
      if (_dataChunks.length === 3072 && _tileCountsBuf) {
        _lastItemsF32 = new Float32Array(f32);
        _lastItemCount = f32[11]; // items[0].meta.w = item count (stored as float)
        _updateTileAssignment(g(deviceId));
      }
      _dataChunks = []; _dataIsF32 = [];
    },
    log_int(v) { console.log("[gpu]", N(v)); },
    log_str(ptr, len) { console.log("[gpu]", new TextDecoder().decode(new Uint8Array(_wasmMemory.buffer, N(ptr), N(len)))); },
  };
}

// ── Tile dispatch: setup + per-frame assignment ──

function _setupTileDispatch(device, width, height) {
  _vpWidth = width; _vpHeight = height;
  const dtx = Math.ceil(width / DISPATCH_TILE);
  const dty = Math.ceil(height / DISPATCH_TILE);
  const tileCount = dtx * dty;

  _tileCountsBuf = device.createBuffer({
    size: Math.max(4, tileCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  _tileIdsBuf = device.createBuffer({
    size: Math.max(4, tileCount * MAX_ITEMS_PER_TILE * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Create bind group for compute group 3 (auto-layout from pipeline)
  const layout = _computePipelineObj.getBindGroupLayout(3);
  _tileBindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: _tileCountsBuf } },
      { binding: 1, resource: { buffer: _tileIdsBuf } },
    ],
  });
}

function _updateTileAssignment(device) {
  if (!_lastItemsF32 || !_tileCountsBuf) return;
  const w = _vpWidth, h = _vpHeight;
  const dtx = Math.ceil(w / DISPATCH_TILE);
  const dty = Math.ceil(h / DISPATCH_TILE);
  const tileCount = dtx * dty;
  const itemCount = _lastItemCount;

  const counts = new Uint32Array(tileCount);
  const ids = new Uint32Array(tileCount * MAX_ITEMS_PER_TILE);

  for (let ri = 0; ri < Math.min(itemCount, 256); ri++) {
    const base = ri * 12;
    const ix = _lastItemsF32[base];
    const iy_base = _lastItemsF32[base + 1];
    const iw = _lastItemsF32[base + 2];
    const ih = _lastItemsF32[base + 3];
    const scrollable = _lastItemsF32[base + 10]; // meta.z = scrollable flag
    const iy = iy_base + (scrollable > 0.5 ? _scrollY : 0);

    const tx0 = Math.max(0, Math.floor(ix / DISPATCH_TILE));
    const ty0 = Math.max(0, Math.floor(iy / DISPATCH_TILE));
    const tx1 = Math.min(dtx - 1, Math.floor((ix + iw) / DISPATCH_TILE));
    const ty1 = Math.min(dty - 1, Math.floor((iy + ih) / DISPATCH_TILE));

    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        const tid = ty * dtx + tx;
        const c = counts[tid];
        if (c < MAX_ITEMS_PER_TILE) {
          ids[tid * MAX_ITEMS_PER_TILE + c] = ri;
          counts[tid] = c + 1;
        }
      }
    }
  }

  device.queue.writeBuffer(_tileCountsBuf, 0, counts);
  device.queue.writeBuffer(_tileIdsBuf, 0, ids);
}

function createFontImports(fontBuffer) {
  const view = new DataView(fontBuffer);
  return {
    len: () => B(fontBuffer.byteLength),
    u8: (offset) => B(view.getUint8(N(offset))),
    u16be: (offset) => B(view.getUint16(N(offset))),
    i16be: (offset) => B(view.getInt16(N(offset))),
    u32be: (offset) => B(view.getUint32(N(offset))),
    i8: (offset) => B(view.getInt8(N(offset))),
  };
}

// ── Scroll animator (single rAF loop) ──

class ScrollAnimator {
  constructor() { this._raf = null; this._lastTime = 0; this._tickFn = null; }
  kick() {
    if (this._raf !== null) return;
    this._lastTime = performance.now();
    const loop = (now) => {
        const dt = Math.max(0, (now - this._lastTime) / 1000);
      this._lastTime = now;
      if (this._tickFn?.(dt)) {
        this._raf = requestAnimationFrame(loop);
      } else {
        this._raf = null;
      }
    };
    this._raf = requestAnimationFrame(loop);
  }
  stop() {
    if (this._raf !== null) { cancelAnimationFrame(this._raf); this._raf = null; }
  }
}

// ══════════════════════════════════════════════════════════════
// Init
// ══════════════════════════════════════════════════════════════

export async function init(wasmUrl, canvas, overlayEl, textareaEl) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  _device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  _format = navigator.gpu.getPreferredCanvasFormat();

  // Load resources
  const [rasterCode, textCode, imageCode, fontBuffer] = await Promise.all([
    fetch("./raster.wgsl?v=" + Date.now()).then(r => r.text()),
    fetch("./text.wgsl?v=" + Date.now()).then(r => r.text()),
    fetch("./image.wgsl?v=" + Date.now()).then(r => r.text()),
    fetch("./font.ttf").then(r => r.arrayBuffer()),
  ]);
  SHADERS = [rasterCode, rasterCode, textCode, imageCode];

  _font = new TTFFont(fontBuffer);
  const chars = []; for (let i = 32; i < 127; i++) chars.push(String.fromCharCode(i));
  _atlas = generateSDFAtlas(_font, chars, 48, 6);

  // Dummy image resources
  const imgTex = _device.createTexture({ size: [1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  _device.queue.writeTexture({ texture: imgTex }, new Uint8Array([0,0,0,0]), { bytesPerRow: 4 }, [1,1]);
  const imgSamp = _device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const imgVtx = _device.createBuffer({ size: 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  const imgIdx = _device.createBuffer({ size: 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });

  // Background texture
  const bgSampObj = _device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  let bgTex;
  {
    // Procedural wallpaper fallback
    const W = 512, H = 512;
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#0a0e1a"; ctx.fillRect(0, 0, W, H);
    for (const b of [
      { x: 0.2, y: 0.3, r: 0.6, c: "rgba(90,20,140,0.7)" },
      { x: 0.8, y: 0.2, r: 0.5, c: "rgba(20,60,160,0.6)" },
      { x: 0.5, y: 0.8, r: 0.7, c: "rgba(10,100,120,0.5)" },
    ]) {
      const grad = ctx.createRadialGradient(b.x*W, b.y*H, 0, b.x*W, b.y*H, b.r*W);
      grad.addColorStop(0, b.c); grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);
    }
    const bmp = await createImageBitmap(c);
    bgTex = _device.createTexture({ size: [W, H], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    _device.queue.copyExternalImageToTexture({ source: bmp }, { texture: bgTex }, [W, H]);
  }

  // WASM
  const wasi = new Proxy({}, { get: () => () => 0 });
  const imports = {
    wasi_snapshot_preview1: wasi,
    dom: createDomImports(),
    gpu: createGpuImports(canvas),
    font_data: createFontImports(fontBuffer),
  };
  const { instance } = await WebAssembly.instantiate(await fetch(wasmUrl).then(r => r.arrayBuffer()), imports);
  _wasmMemory = instance.exports.memory;
  if (instance.exports._start) try { instance.exports._start(); } catch (_) {}

  const ex = instance.exports;
  window._ceangal = ex;
  const container = canvas.parentElement;
  const img = { vtx: h(imgVtx), idx: h(imgIdx), tex: h(imgTex), samp: h(imgSamp) };

  // ══════════════════════════════════════════════════════════
  // Animator: single rAF loop
  // ══════════════════════════════════════════════════════════

  const animator = new ScrollAnimator();
  let _tickCount = 0;
  animator._tickFn = (dt) => {
    const running = ex.scroll_tick ? N(ex.scroll_tick(dt)) === 1 : false;
    updateScrollTransform();
    return running;
  };

  // ══════════════════════════════════════════════════════════
  // Scene lifecycle
  // ══════════════════════════════════════════════════════════

  function prepare() {
    const cw = container.clientWidth, ch = container.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.floor(cw * dpr / 16) * 16;
    const ph = Math.floor(ch * dpr / 16) * 16;
    canvas.width = pw; canvas.height = ph;
    _context = canvas.getContext("webgpu");
    _context.configure({ device: _device, format: _format, alphaMode: "premultiplied" });
    ex.prepare_scene?.(B(h(_device)), B(cw), B(ch), B(pw), B(ph),
      B(img.vtx), B(img.idx), B(0), B(img.tex), B(img.samp),
      B(h(bgTex)), B(h(bgSampObj)));
  }

  prepare();
  ex.todo_init_data?.();

  // ══════════════════════════════════════════════════════════
  // DOM overlay: built ONCE on state change, scroll via transform
  // ══════════════════════════════════════════════════════════

  let _scrollInner = null;

  function buildOverlay() {
    while (overlayEl.firstChild) overlayEl.removeChild(overlayEl.firstChild);

    const listTop = ex.get_list_frame_y ? ex.get_list_frame_y() : 0;
    const listH = ex.get_list_frame_h ? ex.get_list_frame_h() : 9999;

    // Scroll wrapper (clips list items)
    const wrapper = document.createElement("div");
    wrapper.style.cssText = `position:absolute;left:0;top:${listTop}px;width:100%;height:${listH}px;overflow:hidden;pointer-events:none;`;
    const inner = document.createElement("div");
    inner.style.cssText = "position:relative;width:100%;pointer-events:none;";
    wrapper.appendChild(inner);
    overlayEl.appendChild(wrapper);
    _scrollInner = inner;

    const count = ex.get_item_count ? N(ex.get_item_count()) : 0;
    for (let i = 0; i < count; i++) {
      const kind = N(ex.get_item_kind(B(i)));
      if (kind !== 0) continue; // TEXT only

      const x = Number(ex.get_item_x(B(i)));
      const y = Number(ex.get_item_y(B(i)));
      const w = Number(ex.get_item_w(B(i)));
      const itemH = Number(ex.get_item_h(B(i)));
      const scrollable = ex.get_item_scrollable ? N(ex.get_item_scrollable(B(i))) : 0;
      const textId = N(ex.get_item_text(B(i)));
      const text = strings[textId] || "";
      if (!text) continue;

      const fontSize = ex.get_item_font_size ? Number(ex.get_item_font_size(B(i))) : 14;
      const span = document.createElement("span");
      span.textContent = text;
      const selectable = ex.get_item_selectable ? N(ex.get_item_selectable(B(i))) : 0;
      const pe = selectable ? "auto" : "none";
      const us = selectable ? "text" : "none";
      span.style.cssText = `position:absolute;left:${x}px;top:${scrollable ? y - listTop : y}px;width:${w}px;height:${itemH}px;display:flex;align-items:center;font:${fontSize}px sans-serif;color:white;overflow:hidden;pointer-events:${pe};user-select:${us};`;

      if (scrollable) {
        inner.appendChild(span);
      } else {
        overlayEl.appendChild(span);
      }
    }
    updateScrollTransform();
  }

  function updateScrollTransform() {
    if (!_scrollInner) return;
    const scrollY = ex.get_scroll_pos ? ex.get_scroll_pos(B(0), B(0)) : 0;
    _scrollInner.style.transform = `translateY(${scrollY}px)`;
  }

  // Initial overlay build (after all functions defined)
  buildOverlay();

  // ══════════════════════════════════════════════════════════
  // TextField: position textarea over TEXT_FIELD items
  // ══════════════════════════════════════════════════════════

  function positionTextFields() {
    const count = ex.get_item_count ? N(ex.get_item_count()) : 0;
    for (let i = 0; i < count; i++) {
      if (N(ex.get_item_kind(B(i))) !== 5) continue; // TEXT_FIELD = 5
      const x = Number(ex.get_item_x(B(i)));
      const y = Number(ex.get_item_y(B(i)));
      const w = Number(ex.get_item_w(B(i)));
      const h = Number(ex.get_item_h(B(i)));
      const fontSize = Number(ex.get_item_font_size(B(i)));
      const textId = N(ex.get_item_text(B(i)));
      const placeholder = strings[textId] || "";
      textareaEl.style.cssText = `position:absolute;left:${x}px;top:${y}px;width:${w}px;height:${h}px;font:${fontSize}px sans-serif;color:white;background:transparent;border:none;outline:none;padding:12px;caret-color:white;z-index:2;pointer-events:auto;resize:none;`;
      textareaEl.placeholder = placeholder;
      break; // only first field
    }
  }
  positionTextFields();

  textareaEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      const text = textareaEl.value.trim();
      if (text && ex.input_clear) {
        ex.input_clear();
        const bytes = new TextEncoder().encode(text);
        for (let b of bytes) ex.input_push(B(b));
        ex.input_submit();
        textareaEl.value = "";
        buildOverlay();
        positionTextFields();
      }
    }
  });

  // ══════════════════════════════════════════════════════════
  // Events: wheel → physics only, animator handles rendering
  // ══════════════════════════════════════════════════════════

  // Mouse light (write params + fragment-only render if not scrolling)
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    ex.set_mouse?.(e.clientX - rect.left, e.clientY - rect.top);
    if (!_dragging && animator._raf === null) ex.draw_light?.();
  });
  canvas.addEventListener("mouseleave", () => {
    ex.set_mouse?.(0, 0);
    if (animator._raf === null) ex.draw_light?.();
  });

  // Scrollbar drag — use same animator as wheel scroll
  let _dragging = false;
  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    if (e.clientX - rect.left > rect.width - 20) {
      _dragging = true;
      e.preventDefault();
    }
  });
  window.addEventListener("mousemove", (e) => {
    if (!_dragging) return;
    const rect = canvas.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    ex.set_scroll_frac?.(frac);
    animator.kick();
  });
  window.addEventListener("mouseup", () => { _dragging = false; });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const scale = e.deltaMode === 1 ? 20 : 1;
    const dy = -e.deltaY * scale;
    ex.scroll_wheel?.(B(0), 0, dy);
    animator.kick();
  }, { passive: false });

  // ══════════════════════════════════════════════════════════
  // Events: click → state change → rebuild
  // ══════════════════════════════════════════════════════════

  function handleClick(e) {
    if (!ex.handle_click) return;
    const rect = canvas.getBoundingClientRect();
    const result = N(ex.handle_click(e.clientX - rect.left, e.clientY - rect.top));
    if (result === -2) {
      ex.todo_add();
      scheduleOverlay();
    } else if (result >= 0) {
      ex.todo_toggle(B(result));
      scheduleOverlay();
    }
  }
  canvas.addEventListener("click", handleClick);
  overlayEl.addEventListener("click", handleClick);

  let _overlayTimer = 0;
  function scheduleOverlay() {
    clearTimeout(_overlayTimer);
    _overlayTimer = setTimeout(() => buildOverlay(), 16);
  }


  // ══════════════════════════════════════════════════════════
  // Resize
  // ══════════════════════════════════════════════════════════

  let _resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
      animator.stop();
      prepare();
      ex.flush?.();
      buildOverlay();
    }, 150);
  });
}
