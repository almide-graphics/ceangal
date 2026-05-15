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
  _device = await adapter.requestDevice();
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
    ex.do_prepare?.(B(h(_device)), B(cw), B(ch), B(pw), B(ph),
      B(img.vtx), B(img.idx), B(0), B(img.tex), B(img.samp));
  }

  function renderAll() {
    animator.stop();
    prepare();
  }

  renderAll();

  // ── Wheel scroll ──

  canvas.style.touchAction = "none";
  canvas.style.overscrollBehavior = "none";

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const scale = e.deltaMode === 1 ? 20 : 1;
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      const dx = -((Math.abs(e.deltaX) > Math.abs(e.deltaY)) ? e.deltaX : e.deltaY) * scale;
      ex.scroll_wheel_x?.(dx);
    } else {
      ex.scroll_wheel?.(-e.deltaY * scale);
    }
    animator.kick();
  }, { passive: false });

  // ── Touch drag scroll ──

  let _dragging = false, _lastPY = 0;
  const _samples = [];

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
    if (_samples.length > 4) _samples.shift();
    ex.scroll_drag?.(dy);
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!_dragging) return; _dragging = false;
    let vel = 0;
    if (_samples.length >= 2) {
      const a = _samples[0], b = _samples[_samples.length - 1];
      const dt = (b.t - a.t) / 1000;
      if (dt > 0.001) vel = (b.y - a.y) / dt;
    }
    ex.scroll_release?.(vel);
    animator.kick();
  });
  canvas.addEventListener("pointercancel", () => { _dragging = false; });

  // ── Scrollbar mouse interaction ──

  let _barAxis = null; // "y" | "x" | null
  let _barStart = 0, _barThumbStart = 0;

  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const cw = container.clientWidth, ch = container.clientHeight;

    // Vertical scrollbar (right 20px)
    if (mx > cw - 20 && ex.scrollbar_info_y) {
      const top = Number(ex.scrollbar_info_y());
      const h = Number(ex.scrollbar_height_y());
      if (h > 1 && my >= top && my <= top + h) {
        _barAxis = "y"; _barStart = my; _barThumbStart = top;
        e.preventDefault(); return;
      }
    }
    // Horizontal scrollbar (bottom 20px)
    if (my > ch - 20 && ex.scrollbar_info_x) {
      const left = Number(ex.scrollbar_info_x());
      const w = Number(ex.scrollbar_width_x());
      if (w > 1 && mx >= left && mx <= left + w) {
        _barAxis = "x"; _barStart = mx; _barThumbStart = left;
        e.preventDefault(); return;
      }
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (!_barAxis) return;
    const rect = canvas.getBoundingClientRect();
    if (_barAxis === "y") {
      const my = e.clientY - rect.top;
      const thumbH = Number(ex.scrollbar_height_y());
      const track = container.clientHeight - thumbH;
      if (track > 0) ex.scrollbar_set_frac_y?.(Math.max(0, Math.min(1, (_barThumbStart + my - _barStart) / track)));
    } else {
      const mx = e.clientX - rect.left;
      const thumbW = Number(ex.scrollbar_width_x());
      const track = container.clientWidth - thumbW;
      if (track > 0) ex.scrollbar_set_frac_x?.(Math.max(0, Math.min(1, (_barThumbStart + mx - _barStart) / track)));
    }
    animator.kick();
  });

  window.addEventListener("mouseup", () => { _barAxis = null; });

  // ── Keyboard scroll ──
  canvas.tabIndex = 0;
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
