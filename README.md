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
