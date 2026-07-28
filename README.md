<p align="center">
  <img src="assets/logo.png" alt="Ceangal" width="200">
</p>

<h1 align="center">ceangal</h1>

<p align="center">
  <strong>GPU-native UI framework for Almide</strong><br>
  Layout, widget, and interaction layer on top of <a href="https://github.com/almide-graphics/snaidhm">snaidhm</a>.
</p>

<p align="center">
  <em>ceangal</em> (Irish: /ˈcaŋɡəl/) — binding, bond, connection.<br>
  snaidhm ties the knots; ceangal binds them into a UI.
</p>

---

## Stack

```
ceangal  ← layout, widget, interaction
  └─ snaidhm  ← GPU path renderer, SDF text, images
       └─ lumen  ← vec, mat, color, quat
            └─ almide  ← language, WASM/WGSL codegen
```

## Features

- **Yoga-compatible Flexbox** — full layout engine (flex, gap, wrap, absolute positioning, percentage sizes)
- **GPU compute rendering** — all UI elements rendered via compute shaders, zero Canvas 2D
- **DOM overlay** — text selection, copy, accessibility (ARIA), IME input
- **Virtual list** — O(1) scroll with fixed-height items
- **Declarative views** — `View -> View` pipeline with opaque modifiers
- **Reactive state** — Cell-based dirty tracking, minimal re-render

## Status

Active development. Flexbox layout engine complete (74 Yoga-aligned tests passing). GPU rendering pipeline operational.

## License

MIT

## Host

ceangal's browser host lives in `host/`, not inside an example — a consumer that
had to copy it out of `examples/demo/` is how a downstream copy once diverged
from this repo and a rendering bug got fixed in the copy instead of the source.

`host/` owns the UI half: the frame loop, the DOM overlay, fonts, and ceangal's
own shaders. It does **not** implement the `gpu` namespace — that belongs to
[snaidhm](https://github.com/almide-graphics/snaidhm), which declares it, and
arrives as `gpu.js` from `snaidhm/host/`.

The toolchain resolves `.almd` modules from dependencies but has no equivalent
for web host assets, so a page still copies these files. `tools/assemble-host.mjs`
makes that copy mechanical and checkable:

```sh
# assemble into the directory you serve
node tools/assemble-host.mjs examples/demo host ../snaidhm/host

# CI: fail if a served file has drifted from the package that owns it
node tools/assemble-host.mjs --check examples/demo host ../snaidhm/host
```

### Extending without forking

`init` takes hooks so an app adds its own rendering without copying this file —
which is exactly what a consumer had to do, and how its copy drifted:

```js
await init("app.wasm", canvas, overlay, textarea, {
  async onReady(ctx) {
    const shader = ctx.registerShader(myWgsl);
    ctx.exports.my_init(ctx.device, canvas.width, canvas.height);
  },
  onFrame(ctx, t) {
    ctx.exports.my_frame(ctx.device, t);  // app pass first — it clears
    ctx.drawUI();                          // ceangal composites on top
  },
  onResize(ctx, w, h) { ctx.exports.my_resize(ctx.device, w, h); },
});
```

An app that supplies `onFrame` owns the frame: ceangal draws when asked, so the
app decides the order its pass and the UI compose in.

Each package lists what it contributes in `host/MANIFEST`. The assembler writes
`.provenance` next to the output recording the source commit and hash of every
file, and `--check` fails on drift in either direction — including a served file
that no package claims.
