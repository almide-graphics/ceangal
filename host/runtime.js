// ceangal's browser host.
//
// Rules:
//   1. ONE rAF loop (ScrollAnimator only)
//   2. ZERO DOM creation during scroll (transform only)
//   3. State change → rebuild DOM overlay
//   4. Scroll → WASM physics + GPU draw (no tree ops)
//
// The `gpu` namespace is NOT implemented here — it belongs to snaidhm, which
// declares it, and arrives as `./gpu.js` from snaidhm's own host package. This
// file owns the UI half: the DOM overlay, fonts, and the frame loop.

import { createGpuHost } from "./gpu.js";
import { TTFFont } from "./ttf.js";
import { generateSDFAtlas } from "./sdf.js";

// DOM elements are their own handle space. GPU handles live in snaidhm's host
// and the two never cross on the wasm side, so they do not share a table.
const domHandles = [null];
const h = (obj) => { domHandles.push(obj); return domHandles.length - 1; };
const g = (id) => domHandles[Number(id)];
const B = (n) => BigInt(n);
const N = (b) => Number(b);

let _device, _context, _format, _wasmMemory;
let _gpu = null;
let _font = null, _atlas = null;

const strings = [];
let strBuf = [];


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

/// Append an app-supplied WGSL module; returns the index `create_shader`
/// resolves it by. Delegates to snaidhm's host, which owns the shader table.
export function registerShader(code) { return _gpu.registerShader(code); }

/// Reset per-frame clear ownership. Call at the top of every frame when the app
/// drives its own render loop.
export function beginFrame() { _gpu?.beginFrame(); }

/// Boot ceangal against a wasm module.
///
/// `hooks` is how an APP extends this without forking the file — which is what
/// a consumer had to do to render its own 3D layer, and how its copy drifted:
///
///   onReady(ctx)        once, after the module is instantiated and the scene
///                       prepared. Load resources, build pipelines.
///   onFrame(ctx, t)     every frame BEFORE ceangal's own draw, so an app pass
///                       renders underneath the UI. `t` is seconds since ready.
///   onResize(ctx, w, h) after the drawing buffer changes size.
///
/// `ctx` carries everything an app needs and nothing it does not:
///   { exports, gpu, device, canvas, registerShader }
/// where `device` is the handle the wasm side uses and `gpu` is snaidhm's host.
///
/// An app that supplies `onFrame` owns the frame: ceangal draws when asked, so
/// the app decides the order its pass and ceangal's compose in.
export async function init(wasmUrl, canvas, overlayEl, textareaEl, hooks = {}) {
  // "wallpaper" (default) paints ceangal's procedural backdrop; "transparent"
  // leaves it clear so an app's own layer shows through the UI.
  const background = hooks.background ?? "wallpaper";
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  _device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  _format = navigator.gpu.getPreferredCanvasFormat();

  // snaidhm owns the `gpu` namespace; everything GPU-side goes through its host.
  _gpu = createGpuHost(canvas);
  _gpu.setFormat(_format);

  // Load resources
  const [rasterCode, textCode, imageCode, fontBuffer] = await Promise.all([
    fetch("./raster.wgsl?v=" + Date.now()).then(r => r.text()),
    fetch("./text.wgsl?v=" + Date.now()).then(r => r.text()),
    fetch("./image.wgsl?v=" + Date.now()).then(r => r.text()),
    fetch("./font.ttf").then(r => r.arrayBuffer()),
  ]);
  // Index order is the contract create_shader resolves against: snaidhm asks
  // for 0 (raster), 2 (text) and 3 (image); 1 is a spare that falls back to
  // raster. An app adds its own module with registerShader().
  for (const code of [rasterCode, rasterCode, textCode, imageCode]) _gpu.registerShader(code);

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
    // Backdrop. An app that renders its own layer underneath the UI (a 3D
    // pass) asks for `transparent` — the coverage the 2D shader reports is
    // taken from this texture's alpha where no item sits, so an opaque
    // wallpaper makes the UI opaque and erases whatever is below it.
    const W = 512, H = 512;
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    if (background === "wallpaper") {
      ctx.fillStyle = "#0a0e1a"; ctx.fillRect(0, 0, W, H);
    }
    for (const b of background === "wallpaper" ? [
      { x: 0.2, y: 0.3, r: 0.6, c: "rgba(90,20,140,0.7)" },
      { x: 0.8, y: 0.2, r: 0.5, c: "rgba(20,60,160,0.6)" },
      { x: 0.5, y: 0.8, r: 0.7, c: "rgba(10,100,120,0.5)" },
    ] : []) {
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
    gpu: _gpu.imports,
    font_data: createFontImports(fontBuffer),
  };
  const { instance } = await WebAssembly.instantiate(await fetch(wasmUrl).then(r => r.arrayBuffer()), imports);
  _wasmMemory = instance.exports.memory;
  _gpu.setMemory(_wasmMemory);
  if (instance.exports._start) try { instance.exports._start(); } catch (_) {}

  const ex = instance.exports;
  window._ceangal = ex;
  const container = canvas.parentElement;
  const img = { vtx: _gpu.register(imgVtx), idx: _gpu.register(imgIdx),
                tex: _gpu.register(imgTex), samp: _gpu.register(imgSamp) };

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
    ex.prepare_scene?.(B(_gpu.register(_device)), B(cw), B(ch), B(pw), B(ph),
      B(img.vtx), B(img.idx), B(0), B(img.tex), B(img.samp),
      B(_gpu.register(bgTex)), B(_gpu.register(bgSampObj)));
  }

  prepare();
  ex.todo_init_data?.();

  // ── App layer ────────────────────────────────────────────────────────────

  const ctx = {
    exports: ex,
    gpu: _gpu,
    // The handle as the WASM side takes it — an i64, so a BigInt. Handing out
    // the raw Number makes every `ctx.exports.f(ctx.device, …)` throw
    // `Cannot convert Number to BigInt` at the boundary.
    device: B(_gpu.register(_device)),
    canvas,
    registerShader: (code) => _gpu.registerShader(code),
    /// Draw ceangal's UI. An app's onFrame calls this where it wants the UI in
    /// its own composition order; skipping it draws no UI that frame.
    drawUI: () => ex.flush?.(),
  };

  if (hooks.onReady) {
    // An app hook that throws must not take the whole boot down silently: the
    // caller's `init(...)` is usually not awaited, so the rejection surfaces as
    // an unhandled promise and the page simply does nothing.
    try {
      await hooks.onReady(ctx);
    } catch (e) {
      console.error("[ceangal] onReady failed — the app layer did not start:", e);
      hooks.onError?.(e);
    }
  }

  if (hooks.onFrame) {
    const t0 = performance.now();
    let reported = false;
    const appFrame = () => {
      try {
        _gpu.beginFrame();
        hooks.onFrame(ctx, (performance.now() - t0) / 1000);
      } catch (e) {
        if (!reported) {
          reported = true;
          console.error("[ceangal] onFrame threw:", e);
          hooks.onError?.(e);
        }
      }
      requestAnimationFrame(appFrame);
    };
    requestAnimationFrame(appFrame);
  }

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
      hooks.onResize?.(ctx, canvas.width, canvas.height);
      ex.flush?.();
      buildOverlay();
    }, 150);
  });
}
