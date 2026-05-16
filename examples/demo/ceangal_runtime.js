// ceangal runtime — WASM host for DOM + GPU + font imports
//
// Thin bridge: all scroll physics + scrollbar state live in WASM (Almide).
// JS handles: event dispatch, rAF loop, WebGPU context, resource loading.

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
    },
    create_compute_pipeline(deviceId, shaderId, _) {
      return B(h(g(deviceId).createComputePipeline({ layout: "auto", compute: { module: g(shaderId), entryPoint: "fine" } })));
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
    dispatch_workgroups(passId, x, y, z) { g(passId).dispatchWorkgroups(N(x), N(y), N(z)); },
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
      _dataChunks = []; _dataIsF32 = [];
    },
    log_int(v) { console.log("[gpu]", N(v)); },
    log_str(ptr, len) { console.log("[gpu]", new TextDecoder().decode(new Uint8Array(_wasmMemory.buffer, N(ptr), N(len)))); },
  };
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

// ── Scroll animator (rAF bridge — physics lives in WASM) ──

class ScrollAnimator {
  constructor() { this._raf = null; this._lastTime = 0; this._tickFn = null; }

  kick() {
    if (this._raf !== null) return;
    this._lastTime = performance.now();
    const loop = (now) => {
      const dt = (now - this._lastTime) / 1000;
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

// ── Init ──

export async function init(wasmUrl, canvas, overlayEl, textareaEl) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  _device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  _format = navigator.gpu.getPreferredCanvasFormat();

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

  // Image resources (demo)
  const imgTex = _device.createTexture({ size: [1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  _device.queue.writeTexture({ texture: imgTex }, new Uint8Array([0,0,0,0]), { bytesPerRow: 4 }, [1,1]);
  const imgSamp = _device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const imgVtx = _device.createBuffer({ size: 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  const imgIdx = _device.createBuffer({ size: 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });

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
  const container = canvas.parentElement;
  const img = { vtx: h(imgVtx), idx: h(imgIdx), tex: h(imgTex), samp: h(imgSamp) };
  const animator = new ScrollAnimator();

  animator._tickFn = (dt) => ex.scroll_tick ? N(ex.scroll_tick(dt)) === 1 : false;

  // ── Scene lifecycle ──

  function prepare() {
    const cw = container.clientWidth, ch = container.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.floor(cw * dpr / 16) * 16;
    const ph = Math.floor(ch * dpr / 16) * 16;
    canvas.width = pw; canvas.height = ph;
    _context = canvas.getContext("webgpu");
    _context.configure({ device: _device, format: _format, alphaMode: "premultiplied" });
    ex.prepare_scene?.(B(h(_device)), B(cw), B(ch), B(pw), B(ph),
      B(img.vtx), B(img.idx), B(0), B(img.tex), B(img.samp));
  }

  let _firstRender = true;
  function renderAll() {
    animator.stop();
    prepare();
    if (_firstRender) {
      ex.todo_init_data?.();
      // Re-prepare to include initial data
      prepare();
      _firstRender = false;
    }
    updateViewTextOverlay();
  }

  renderAll();

  // ── Virtual list DOM overlay ──
  const VLIST_ITEM_H = 0;
  const VLIST_COUNT = 0;

  // View tree DOM text overlay
  function updateViewTextOverlay() {
    if (!ex.get_item_count) return;
    // Clear old overlay
    while (overlayEl.firstChild) overlayEl.removeChild(overlayEl.firstChild);

    const count = N(ex.get_item_count());
    for (let i = 0; i < count; i++) {
      const kind = N(ex.get_item_kind(B(i)));
      if (kind !== 0) continue; // only TEXT nodes (kind=0)

      const x = Number(ex.get_item_x(B(i)));
      const y = Number(ex.get_item_y(B(i)));
      const w = Number(ex.get_item_w(B(i)));
      const h = Number(ex.get_item_h(B(i)));
      const textId = N(ex.get_item_text(B(i)));
      const text = strings[textId] || "";

      if (!text) continue;

      const fontSize = ex.get_item_font_size ? Number(ex.get_item_font_size(B(i))) : 14;
      const canSelect = ex.get_item_selectable ? N(ex.get_item_selectable(B(i))) === 1 : true;
      const span = document.createElement("span");
      span.textContent = text;
      const pe = canSelect ? "pointer-events:auto;user-select:text;cursor:text;" : "pointer-events:none;user-select:none;";
      span.style.cssText = `position:absolute;left:${x}px;top:${y}px;width:${w}px;height:${h}px;display:flex;align-items:center;font:${fontSize}px sans-serif;color:white;overflow:hidden;${pe}`;
      overlayEl.appendChild(span);
    }
  }

  updateViewTextOverlay();
  const domPool = []; // recycled <div> elements
  let domVisible = new Map(); // index → element

  function updateVListDOM() {
    if (!ex.scroll_tick) return; // no scroll engine
    const scrollY = ex.get_scroll_pos ? Number(ex.get_scroll_pos(B(0), B(0))) : 0;
    const ch = container.clientHeight;
    const visTop = -scrollY;
    const visBot = visTop + ch;
    const startIdx = Math.max(0, Math.floor(visTop / VLIST_ITEM_H));
    const endIdx = Math.min(VLIST_COUNT, Math.ceil(visBot / VLIST_ITEM_H));

    // Remove out-of-view items
    for (const [idx, el] of domVisible) {
      if (idx < startIdx || idx >= endIdx) {
        el.style.display = "none";
        domPool.push(el);
        domVisible.delete(idx);
      }
    }

    // Add in-view items
    for (let i = startIdx; i < endIdx; i++) {
      if (domVisible.has(i)) {
        // Update position
        const el = domVisible.get(i);
        el.style.top = (i * VLIST_ITEM_H + scrollY + 15) + "px";
      } else {
        // Get or create element
        let el = domPool.pop();
        if (!el) {
          el = document.createElement("div");
          el.style.cssText = "position:absolute;left:5%;pointer-events:none;font:14px sans-serif;color:rgba(255,255,255,0.85);line-height:" + VLIST_ITEM_H + "px;white-space:nowrap;user-select:text;";
          overlayEl.appendChild(el);
        }
        el.textContent = "Item " + i;
        el.style.top = (i * VLIST_ITEM_H + scrollY + 15) + "px";
        el.style.display = "";
        domVisible.set(i, el);
      }
    }
  }

  // Hook into scroll tick
  const origTickFn = animator._tickFn;
  animator._tickFn = (dt) => {
    const r = origTickFn?.(dt) ?? false;
    updateVListDOM();
    return r;
  };

  // Also update on wheel (immediate feedback)
  const origWheel = ex.scroll_wheel;

  // ── Wheel scroll ──

  canvas.style.touchAction = "none";
  canvas.style.overscrollBehavior = "none";
  container.style.overflow = "hidden";

  // Hit test → region ID (cached per pointer position)
  let _activeRegion = 0;

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const scale = e.deltaMode === 1 ? 20 : 1;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = doHitTest(mx, my);
    _activeRegion = hit.region;
    const dy = -e.deltaY * scale;
    const dx = (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY))
      ? -((Math.abs(e.deltaX) > Math.abs(e.deltaY)) ? e.deltaX : e.deltaY) * scale : 0;
    ex.scroll_wheel?.(B(_activeRegion), dx || 0, dy || 0);
    updateVListDOM();
    animator.kick();
  }, { passive: false });

  // ── Touch drag scroll ──

  let _dragging = false, _lastPY = 0;
  const _samples = []; // {t, y} — last 6 samples for velocity estimation

  function estimateVelocity() {
    // Weighted linear regression over recent samples (Flutter VelocityTracker inspired)
    const n = _samples.length;
    if (n < 2) return 0;
    const now = _samples[n - 1].t;
    let sw = 0, swt = 0, swy = 0, swtt = 0, swty = 0;
    for (const s of _samples) {
      const age = (now - s.t) / 1000;
      if (age > 0.15) continue; // ignore samples older than 150ms
      const w = 1 / (1 + age * 10); // recent samples weighted more
      const t = -age;
      sw += w; swt += w * t; swy += w * s.y; swtt += w * t * t; swty += w * t * s.y;
    }
    const det = sw * swtt - swt * swt;
    return det > 1e-9 ? (sw * swty - swt * swy) / det : 0;
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" || e.button !== 0) return;
    _dragging = true; _lastPY = e.clientY;
    _samples.length = 0; _samples.push({ t: performance.now(), y: e.clientY });
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!_dragging) return;
    const dy = e.clientY - _lastPY; _lastPY = e.clientY;
    _samples.push({ t: performance.now(), y: e.clientY });
    if (_samples.length > 8) _samples.shift();
    ex.scroll_drag?.(B(_activeRegion), dy);
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!_dragging) return; _dragging = false;
    ex.scroll_release?.(B(_activeRegion), estimateVelocity());
    animator.kick();
  });
  canvas.addEventListener("pointercancel", () => { _dragging = false; });

  // ── Unified hit test + interaction (all logic in WASM) ──

  const HIT_NONE = 0, HIT_REGION = 1, HIT_THUMB_Y = 2, HIT_THUMB_X = 3, HIT_TRACK_Y = 4, HIT_TRACK_X = 5;
  let _interaction = null; // { type, region, axis }

  function doHitTest(mx, my) {
    if (!ex.hit_test) return { type: HIT_REGION, region: 0 };
    const packed = N(ex.hit_test(mx, my));
    return { type: (packed >> 8) & 0xFF, region: packed & 0xFF };
  }

  // Hover
  canvas.addEventListener("mousemove", (e) => {
    if (_interaction) return;
    const rect = canvas.getBoundingClientRect();
    const hit = doHitTest(e.clientX - rect.left, e.clientY - rect.top);
    const nearBar = hit.type >= HIT_THUMB_Y;
    ex.scrollbar_set_hover_y?.(nearBar && (hit.type === HIT_THUMB_Y || hit.type === HIT_TRACK_Y) ? 1.0 : 0.0);
    ex.scrollbar_set_hover_x?.(nearBar && (hit.type === HIT_THUMB_X || hit.type === HIT_TRACK_X) ? 1.0 : 0.0);
    animator.kick();
  });

  canvas.addEventListener("mouseleave", () => {
    if (!_interaction) {
      ex.scrollbar_set_hover_y?.(0.0);
      ex.scrollbar_set_hover_x?.(0.0);
      animator.kick();
    }
  });

  // Mousedown: dispatch based on hit type
  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = doHitTest(mx, my);

    if (hit.type === HIT_THUMB_Y) {
      _interaction = { type: hit.type, region: hit.region, axis: 0 };
      e.preventDefault();
    } else if (hit.type === HIT_THUMB_X) {
      _interaction = { type: hit.type, region: hit.region, axis: 1 };
      e.preventDefault();
    } else if (hit.type === HIT_TRACK_Y) {
      ex.track_tap_at?.(B(hit.region), B(0), my);
      animator.kick(); e.preventDefault();
    } else if (hit.type === HIT_TRACK_X) {
      ex.track_tap_at?.(B(hit.region), B(1), mx);
      animator.kick(); e.preventDefault();
    }
  });

  // Mousemove: scrollbar drag
  window.addEventListener("mousemove", (e) => {
    if (!_interaction) return;
    const rect = canvas.getBoundingClientRect();
    const pos = _interaction.axis === 0 ? e.clientY - rect.top : e.clientX - rect.left;
    ex.scrollbar_drag_to?.(B(_interaction.region), B(_interaction.axis), pos);
    animator.kick();
  });

  window.addEventListener("mouseup", () => { _interaction = null; });

  // ── Hover ──
  function onHover(e) {
    if (!ex.set_hover) return;
    const rect = canvas.getBoundingClientRect();
    ex.set_hover(e.clientX - rect.left, e.clientY - rect.top);
  }
  container.addEventListener("mousemove", onHover);

  // ── Todo actions ──
  function handleTodoClick(e) {
    if (!ex.handle_click) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const todoId = N(ex.handle_click(mx, my));
    if (todoId >= 0) {
      ex.todo_toggle(B(todoId));
      updateViewTextOverlay();
    }
  }
  canvas.addEventListener("click", handleTodoClick);
  overlayEl.addEventListener("click", handleTodoClick);

  window.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && ex.todo_add) {
      ex.todo_add();
      updateViewTextOverlay();
    }
  });

  // ── Keyboard scroll ──
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "application");
  canvas.setAttribute("aria-label", "Scrollable content");
  canvas.focus();
  const keyMap = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3, PageUp: 4, PageDown: 5, Home: 6, End: 7 };
  canvas.addEventListener("keydown", (e) => {
    const code = keyMap[e.key];
    if (code !== undefined && ex.scroll_key) {
      e.preventDefault();
      ex.scroll_key(B(code));
      animator.kick();
    }
  });

  // ── Resize + visibility ──

  let _resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(renderAll, 150);
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) animator.stop(); });
}
