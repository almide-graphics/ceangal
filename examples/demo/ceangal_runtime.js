// ceangal runtime — DOM + GPU + font WASM import implementations
//
// Single WASM binary (ceangal + snaidhm compiled together).
// This runtime provides all three import namespaces: dom, gpu, font.

import { TTFFont } from "./ttf.js";
import { generateSDFAtlas } from "./sdf.js";

const handles = [null];
function h(obj) { handles.push(obj); return handles.length - 1; }
function g(id) { return handles[Number(id)]; }
const B = (n) => BigInt(n);
const N = (b) => Number(b);

let _device, _context, _format, _wasmMemory;
let _font = null, _atlas = null;

// DOM string table
const strings = [];
let strBuf = [];

// GPU streaming data
let _dataChunks = [];
let _dataIsF32 = [];
let _bindingEntries = [];

let SHADERS = [];

// ── DOM imports ──

function createDomImports() {
  return {
    begin_str() { strBuf = []; },
    push_byte(b) { strBuf.push(N(b)); },
    commit_str() {
      const s = new TextDecoder().decode(new Uint8Array(strBuf));
      strings.push(s);
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

// ── GPU imports (from snaidhm) ──

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

// ── Font imports ──

function _glyph(ch) { return _atlas.glyphs.get(String.fromCharCode(N(ch))); }

function createFontImports() {
  return {
    units_per_em: () => B(_font.unitsPerEm),
    atlas_width: () => B(_atlas.atlasWidth),
    atlas_height: () => B(_atlas.atlasHeight),
    glyph_advance(ch) { const g = _glyph(ch); return g ? g.advance : _font.unitsPerEm * 0.3; },
    glyph_has_sdf(ch) { const g = _glyph(ch); return B(g && g.atlasW > 0 ? 1 : 0); },
    glyph_atlas_x(ch) { const g = _glyph(ch); return B(g ? g.atlasX : 0); },
    glyph_atlas_y(ch) { const g = _glyph(ch); return B(g ? g.atlasY : 0); },
    glyph_atlas_w(ch) { const g = _glyph(ch); return B(g ? g.atlasW : 0); },
    glyph_atlas_h(ch) { const g = _glyph(ch); return B(g ? g.atlasH : 0); },
    glyph_sdf_scale(ch) { const g = _glyph(ch); return g ? g.sdfScale : 1.0; },
    glyph_padding(ch) { const g = _glyph(ch); return B(g ? g.padding : 0); },
    glyph_xmin(ch) { const g = _glyph(ch); return g ? g.bounds.xMin : 0.0; },
    glyph_ymin(ch) { const g = _glyph(ch); return g ? g.bounds.yMin : 0.0; },
    glyph_xmax(ch) { const g = _glyph(ch); return g ? g.bounds.xMax : 0.0; },
    glyph_ymax(ch) { const g = _glyph(ch); return g ? g.bounds.yMax : 0.0; },
  };
}

// ── Image resources ──

function generateCardImages() {
  const S = 64;
  const atlas = new Uint8Array(S * 3 * S * 4);
  for (let img = 0; img < 3; img++) {
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = (y * S * 3 + img * S + x) * 4;
        const u = x / S, v = y / S;
        let r, g, b;
        if (img === 0) { const dx=u-0.5,dy=v-0.35,sun=Math.max(0,1-Math.sqrt(dx*dx+dy*dy)*4.5); r=0.92-v*0.35+sun*0.4; g=0.35+sun*0.55-v*0.15; b=0.25+v*0.45; }
        else if (img === 1) { const w=Math.sin(u*12+v*4)*0.04; r=0.1+v*0.15; g=0.35+u*0.2+w; b=0.55+v*0.25+w; }
        else { const bd=Math.sin(v*8+u*3)*0.1; r=0.25+v*0.35+bd; g=0.15+(1-v)*0.4+bd*0.5; b=0.4+v*0.25; }
        atlas[i]=Math.min(255,Math.max(0,r*255)); atlas[i+1]=Math.min(255,Math.max(0,g*255)); atlas[i+2]=Math.min(255,Math.max(0,b*255)); atlas[i+3]=255;
      }
    }
  }
  return { data: atlas, width: S * 3, height: S };
}

function buildImageQuads() {
  const cards = [
    { x: -0.81, y: -0.81, w: 0.40, h: 0.30, uOff: 0 },
    { x: -0.26, y: -0.81, w: 0.40, h: 0.30, uOff: 1/3 },
    { x:  0.29, y: -0.81, w: 0.48, h: 0.30, uOff: 2/3 },
  ];
  const verts = [], idxs = [];
  for (const c of cards) {
    const vi = verts.length / 4;
    const u0 = c.uOff, u1 = c.uOff + 1/3;
    verts.push(c.x,c.y,u0,1, c.x+c.w,c.y,u1,1, c.x+c.w,c.y+c.h,u1,0, c.x,c.y+c.h,u0,0);
    idxs.push(vi,vi+1,vi+2,vi,vi+2,vi+3);
  }
  return { vertices: new Float32Array(verts), indices: new Uint32Array(idxs) };
}

// ── Text input (JS-side) ──

function setupTextInput(overlay, textarea, canvasW) {
  const ndcToX = (nx) => (nx + 1) * 0.5 * canvasW;
  const ndcToY = (ny) => (1 - ny) * 0.5 * canvasW;
  const left = ndcToX(0.29), top = ndcToY(-0.48), width = ndcToX(0.81) - ndcToX(0.29);

  const inputArea = document.createElement("div");
  inputArea.className = "input-area";
  inputArea.setAttribute("role", "textbox");
  inputArea.setAttribute("aria-label", "Text input field");
  Object.assign(inputArea.style, { left: left+"px", top: top+"px", width: width+"px", fontSize: "11px", fontFamily: "sans-serif" });

  const display = document.createElement("div");
  display.className = "input-display";
  inputArea.appendChild(display);
  overlay.appendChild(inputArea);

  Object.assign(textarea.style, { left: left+"px", top: top+"px", width: width+"px", height: "1.4em", fontSize: "11px" });

  let focused = false;
  function renderPh() {
    display.innerHTML = "";
    if (!focused && textarea.value === "") {
      const ph = document.createElement("span"); ph.className = "placeholder"; ph.textContent = "Type here...";
      display.appendChild(ph);
    }
  }
  inputArea.addEventListener("click", () => { textarea.style.pointerEvents = "auto"; textarea.focus(); });
  textarea.addEventListener("focus", () => { focused = true; textarea.style.color = "#333"; textarea.style.caretColor = "#333"; inputArea.style.outline = "1.5px solid rgba(66,133,244,0.6)"; inputArea.style.outlineOffset = "2px"; inputArea.style.borderRadius = "2px"; renderPh(); });
  textarea.addEventListener("blur", () => { focused = false; textarea.style.color = "transparent"; textarea.style.caretColor = "transparent"; textarea.style.pointerEvents = "none"; inputArea.style.outline = "none"; display.innerHTML = ""; if (textarea.value) display.appendChild(document.createTextNode(textarea.value)); else renderPh(); });
  textarea.addEventListener("input", () => renderPh());
  renderPh();
}

// ── Init ──

export async function init(wasmUrl, canvas, overlayEl, textareaEl) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  _device = await adapter.requestDevice();
  _format = navigator.gpu.getPreferredCanvasFormat();

  // Load shaders + font in parallel
  const [rasterCode, textCode, imageCode, fontBuffer] = await Promise.all([
    fetch("./raster.wgsl").then(r => r.text()),
    fetch("./text.wgsl").then(r => r.text()),
    fetch("./image.wgsl").then(r => r.text()),
    fetch("./font.ttf").then(r => r.arrayBuffer()),
  ]);
  SHADERS = [rasterCode, rasterCode, textCode, imageCode];

  // Font + SDF atlas
  _font = new TTFFont(fontBuffer);
  const chars = [];
  for (let i = 32; i < 127; i++) chars.push(String.fromCharCode(i));
  _atlas = generateSDFAtlas(_font, chars, 48, 6);

  // SDF texture
  const sdfTexture = _device.createTexture({ size: [_atlas.atlasWidth, _atlas.atlasHeight], format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const bpr = Math.ceil(_atlas.atlasWidth / 256) * 256;
  const aligned = new Uint8Array(bpr * _atlas.atlasHeight);
  for (let row = 0; row < _atlas.atlasHeight; row++) aligned.set(_atlas.atlasData.subarray(row * _atlas.atlasWidth, (row + 1) * _atlas.atlasWidth), row * bpr);
  _device.queue.writeTexture({ texture: sdfTexture }, aligned, { bytesPerRow: bpr }, [_atlas.atlasWidth, _atlas.atlasHeight]);
  const sdfSampler = _device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // Image resources
  const cardImages = generateCardImages();
  const imageQuads = buildImageQuads();
  const imgTexture = _device.createTexture({ size: [cardImages.width, cardImages.height], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const ibpr = Math.ceil(cardImages.width * 4 / 256) * 256;
  const ialigned = new Uint8Array(ibpr * cardImages.height);
  for (let row = 0; row < cardImages.height; row++) ialigned.set(cardImages.data.subarray(row * cardImages.width * 4, (row + 1) * cardImages.width * 4), row * ibpr);
  _device.queue.writeTexture({ texture: imgTexture }, ialigned, { bytesPerRow: ibpr }, [cardImages.width, cardImages.height]);
  const imgSampler = _device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const imgVertBuf = _device.createBuffer({ size: imageQuads.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  _device.queue.writeBuffer(imgVertBuf, 0, imageQuads.vertices);
  const imgIdxBuf = _device.createBuffer({ size: imageQuads.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  _device.queue.writeBuffer(imgIdxBuf, 0, imageQuads.indices);

  console.log(`SDF atlas: ${_atlas.atlasWidth}x${_atlas.atlasHeight} | Images: ${imageQuads.indices.length / 6}`);

  // WASM
  const wasi = new Proxy({}, { get(_, n) { if (n === "proc_exit") return () => {}; if (n === "fd_prestat_get") return () => 8; return () => 0; }});
  const imports = { wasi_snapshot_preview1: wasi, dom: createDomImports(), gpu: createGpuImports(canvas), font: createFontImports() };
  const { instance } = await WebAssembly.instantiate(await fetch(wasmUrl).then(r => r.arrayBuffer()), imports);
  _wasmMemory = instance.exports.memory;

  if (instance.exports._start) try { instance.exports._start(); } catch (_) {}

  // 1. Render scene via snaidhm (GPU)
  if (instance.exports.do_render) {
    instance.exports.do_render(
      B(h(_device)), B(h(sdfTexture)), B(h(sdfSampler)),
      B(h(imgVertBuf)), B(h(imgIdxBuf)), B(imageQuads.indices.length),
      B(h(imgTexture)), B(h(imgSampler)),
    );
    console.log("ceangal: snaidhm render complete");
  }

  // 2. Create DOM overlay via ceangal (Almide)
  if (instance.exports.init_overlay) {
    try {
      instance.exports.init_overlay(B(h(overlayEl)), B(canvas.clientWidth), B(canvas.clientHeight));
      console.log("ceangal: overlay created");
    } catch (e) {
      console.warn("ceangal: overlay failed (record field codegen issue):", e.message);
    }
  }

  // 3. Text input (JS-side)
  setupTextInput(overlayEl, textareaEl, canvas.clientWidth);
  console.log("ceangal: text input ready");
}
