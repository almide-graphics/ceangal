// ceangal DOM runtime — WASM import implementations for dom.almd
//
// Almide Int = i64 = BigInt in JS. B() wraps returns, N() unwraps args.

const handles = [null];
function h(obj) { handles.push(obj); return handles.length - 1; }
function g(id) { return handles[Number(id)]; }
const B = (n) => BigInt(n);
const N = (b) => Number(b);

// String table: built by begin_str/push_byte/commit_str
const strings = [];
let strBuf = [];

let _wasmMemory;

export function createImports() {
  return { dom: {
    begin_str() { strBuf = []; },
    push_byte(b) { strBuf.push(N(b)); },
    commit_str() {
      const s = new TextDecoder().decode(new Uint8Array(strBuf));
      strings.push(s);
      return B(strings.length - 1);
    },

    create_element(tagId) {
      return B(h(document.createElement(strings[N(tagId)])));
    },

    set_text(elId, textId) {
      g(elId).textContent = strings[N(textId)];
    },

    set_attr(elId, nameId, valId) {
      g(elId).setAttribute(strings[N(nameId)], strings[N(valId)]);
    },

    set_style(elId, propId, valId) {
      g(elId).style[strings[N(propId)]] = strings[N(valId)];
    },

    append_child(parentId, childId) {
      g(parentId).appendChild(g(childId));
    },

    get_offset_width(elId) {
      return g(elId).offsetWidth;
    },

    clear_children(elId) {
      g(elId).innerHTML = "";
    },

    log(strId) {
      console.log("[ceangal]", strings[N(strId)]);
    },
  }};
}

// ── Text input (event-driven, stays in JS for now) ──

export function setupTextInput(overlay, textarea, canvasW) {
  const ndcToX = (nx) => (nx + 1) * 0.5 * canvasW;
  const ndcToY = (ny) => (1 - ny) * 0.5 * canvasW;

  const left = ndcToX(0.29);
  const top = ndcToY(-0.48);
  const width = ndcToX(0.81) - left;

  const inputArea = document.createElement("div");
  inputArea.className = "input-area";
  inputArea.setAttribute("role", "textbox");
  inputArea.setAttribute("aria-label", "Text input field");
  inputArea.style.left = left + "px";
  inputArea.style.top = top + "px";
  inputArea.style.width = width + "px";
  inputArea.style.fontSize = "11px";
  inputArea.style.fontFamily = "sans-serif";

  const display = document.createElement("div");
  display.className = "input-display";
  inputArea.appendChild(display);
  overlay.appendChild(inputArea);

  textarea.style.left = left + "px";
  textarea.style.top = top + "px";
  textarea.style.width = width + "px";
  textarea.style.height = "1.4em";
  textarea.style.fontSize = "11px";

  let focused = false;

  function renderPlaceholder() {
    display.innerHTML = "";
    if (!focused && textarea.value === "") {
      const ph = document.createElement("span");
      ph.className = "placeholder";
      ph.textContent = "Type here...";
      display.appendChild(ph);
    }
  }

  inputArea.addEventListener("click", () => {
    textarea.style.pointerEvents = "auto";
    textarea.focus();
  });

  textarea.addEventListener("focus", () => {
    focused = true;
    textarea.style.color = "#333";
    textarea.style.caretColor = "#333";
    inputArea.style.outline = "1.5px solid rgba(66, 133, 244, 0.6)";
    inputArea.style.outlineOffset = "2px";
    inputArea.style.borderRadius = "2px";
    renderPlaceholder();
  });

  textarea.addEventListener("blur", () => {
    focused = false;
    textarea.style.color = "transparent";
    textarea.style.caretColor = "transparent";
    textarea.style.pointerEvents = "none";
    inputArea.style.outline = "none";
    display.innerHTML = "";
    if (textarea.value) {
      display.appendChild(document.createTextNode(textarea.value));
    } else {
      renderPlaceholder();
    }
  });

  textarea.addEventListener("input", () => renderPlaceholder());
  renderPlaceholder();
}

// ── Init ──

export async function init(wasmUrl, overlayEl, textareaEl, canvasW, canvasH) {
  const wasi = new Proxy({}, { get(_, n) {
    if (n === "proc_exit") return () => {};
    if (n === "fd_prestat_get") return () => 8;
    return () => 0;
  }});

  const imports = { wasi_snapshot_preview1: wasi, ...createImports() };
  const { instance } = await WebAssembly.instantiate(
    await fetch(wasmUrl).then(r => r.arrayBuffer()), imports);
  _wasmMemory = instance.exports.memory;

  if (instance.exports._start) try { instance.exports._start(); } catch (_) {}

  // Register overlay element as handle
  const overlayHandle = h(overlayEl);

  // Create text overlay via Almide
  instance.exports.init_overlay(B(overlayHandle), B(canvasW), B(canvasH));
  console.log("ceangal: overlay created (" + strings.length + " strings committed)");

  // Input handling (JS-side for event callbacks)
  setupTextInput(overlayEl, textareaEl, canvasW);
  console.log("ceangal: text input ready");
}
