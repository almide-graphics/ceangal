// snaidhm Phase 1 — Tiled path renderer
//
// CPU: flatten + coarse tile assignment (segments sorted by path_id within each tile)
// GPU: fine rasterize (per-tile winding number fill)

const TILE_SIZE: u32 = 16u;
const MAX_SEGS_PER_TILE: u32 = 512u;

struct LineSeg {
  p0: vec2<f32>,
  p1: vec2<f32>,
  color: vec4<f32>,
  path_id: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

struct Params {
  width: u32,
  height: u32,
  seg_count: u32,
  tiles_x: u32,
  tiles_y: u32,
  shadow_count: u32,
  num_paths: u32,
  _pad2: u32,
}

// SDF shadow: analytical soft shadow via signed distance field
struct Shadow {
  center: vec2<f32>,
  half_size: vec2<f32>,
  corner_radius: f32,
  offset_x: f32,
  offset_y: f32,
  blur: f32,
  color: vec4<f32>,
}

// Paint descriptor: per-path color/gradient
struct Paint {
  paint_type: u32,  // 0=solid, 1=linear, 2=radial
  _p0: u32, _p1: u32, _p2: u32,
  color0: vec4<f32>,
  color1: vec4<f32>,
  grad_params: vec4<f32>,  // linear: start.xy, end.xy | radial: center.xy, radius, _
}

fn evaluate_paint(paint: Paint, p: vec2<f32>) -> vec4<f32> {
  if paint.paint_type == 1u {
    // Linear gradient
    let start = paint.grad_params.xy;
    let dir = paint.grad_params.zw - start;
    let t = clamp(dot(p - start, dir) / dot(dir, dir), 0.0, 1.0);
    return mix(paint.color0, paint.color1, t);
  } else if paint.paint_type == 2u {
    // Radial gradient
    let center = paint.grad_params.xy;
    let radius = paint.grad_params.z;
    let t = clamp(length(p - center) / radius, 0.0, 1.0);
    return mix(paint.color0, paint.color1, t);
  }
  // Solid
  return paint.color0;
}

// Fine rasterize: per-tile winding fill (command-list driven)
@group(0) @binding(0) var<storage, read>       segments: array<LineSeg>;
@group(0) @binding(1) var<storage, read>       tile_cmd_counts: array<u32>;
@group(0) @binding(2) var<storage, read>       tile_seg_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(4) var<uniform>             params: Params;
@group(0) @binding(5) var<storage, read>       shadows: array<Shadow>;
@group(0) @binding(6) var<storage, read>       paints: array<Paint>;
@group(0) @binding(7) var<storage, read>       tile_cmds: array<u32>;

const MAX_CMDS_PER_TILE: u32 = 32u;

// Analytical area coverage (Vello-style)
// p0, p1 in pixel-local coordinates where pixel occupies [0,1] x [0,1]
// Returns signed area: sum over closed path = ±1 inside, 0 outside
fn seg_area(p0: vec2<f32>, p1: vec2<f32>) -> f32 {
  let y = p0.y;
  let delta = p1 - p0;
  let y0 = clamp(y, 0.0, 1.0);
  let y1 = clamp(y + delta.y, 0.0, 1.0);
  let dy = y0 - y1;
  if abs(dy) < 1e-9 { return 0.0; }
  let inv_dy = 1.0 / delta.y;
  let t0 = (y0 - y) * inv_dy;
  let t1 = (y1 - y) * inv_dy;
  let x0 = p0.x + t0 * delta.x;
  let x1 = p0.x + t1 * delta.x;
  // The -1e-6 epsilon on xmin guarantees xmax - xmin > 0 even for
  // perfectly vertical segments, so no special case is needed.
  let xmin = min(min(x0, x1), 1.0) - 1e-6;
  let xmax = max(x0, x1);
  let b = min(xmax, 1.0);
  let c = max(b, 0.0);
  let d = max(xmin, 0.0);
  let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
  // Flip convention: tile assignment uses rightward ray (segments to the RIGHT
  // of pixel are in tile), but the Vello formula gives a≈1 for segments to the
  // LEFT. (1-a) makes segments to the RIGHT contribute fully.
  return (1.0 - a) * dy;
}

// SDF for rounded rectangle (p relative to center, b = half_size, r = corner radius)
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
  let q = abs(p) - b + vec2<f32>(r, r);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0, 0.0))) - r;
}

fn pack_color(r: f32, g: f32, b: f32, a: f32) -> u32 {
  let ri = u32(clamp(r * 255.0, 0.0, 255.0));
  let gi = u32(clamp(g * 255.0, 0.0, 255.0));
  let bi = u32(clamp(b * 255.0, 0.0, 255.0));
  let ai = u32(clamp(a * 255.0, 0.0, 255.0));
  return ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}

@compute @workgroup_size(16, 16)
fn fine(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= params.width || py >= params.height) { return; }

  let tile_id = wg.y * params.tiles_x + wg.x;
  let cmd_count = min(tile_cmd_counts[tile_id], MAX_CMDS_PER_TILE);
  let tile_base = tile_id * MAX_SEGS_PER_TILE;
  let cmd_base = tile_id * MAX_CMDS_PER_TILE * 4u;

  let p = vec2<f32>(
    f32(px) / f32(params.width) * 2.0 - 1.0,
    1.0 - f32(py) / f32(params.height) * 2.0,
  );

  var color = vec3<f32>(0.08, 0.08, 0.10);  // dark background

  // ── Shadows (SDF-based, before path fills) ──
  for (var si = 0u; si < params.shadow_count; si++) {
    let shadow = shadows[si];
    let sp = p - vec2<f32>(shadow.offset_x, shadow.offset_y);
    let d = sd_rounded_box(sp - shadow.center, shadow.half_size, shadow.corner_radius);
    let shadow_alpha = (1.0 - smoothstep(-shadow.blur * 0.3, shadow.blur, d)) * shadow.color.a;
    color = mix(color, shadow.color.rgb, shadow_alpha);
  }

  // ── Path fills (command-list driven) ──
  let ndc_to_px = 0.5 * f32(params.width);
  let ndc_to_py = 0.5 * f32(params.height);
  let px_f = f32(px);
  let py_f = f32(py);

  for (var ci = 0u; ci < cmd_count; ci++) {
    let cb = cmd_base + ci * 4u;
    let path_id = tile_cmds[cb];
    let backdrop = bitcast<i32>(tile_cmds[cb + 1u]);
    let seg_offset = tile_cmds[cb + 2u];
    let seg_count = tile_cmds[cb + 3u];

    var area = f32(backdrop);
    for (var si = 0u; si < seg_count; si++) {
      let seg_idx = tile_seg_ids[tile_base + seg_offset + si];
      let seg = segments[seg_idx];
      let sp0 = vec2<f32>(
        (seg.p0.x + 1.0) * ndc_to_px - px_f,
        (1.0 - seg.p0.y) * ndc_to_py - py_f,
      );
      let sp1 = vec2<f32>(
        (seg.p1.x + 1.0) * ndc_to_px - px_f,
        (1.0 - seg.p1.y) * ndc_to_py - py_f,
      );
      area += seg_area(sp0, sp1);
    }

    let raw_cov = min(abs(area), 1.0);
    // Attenuate low coverage to suppress all-columns-left horizontal artifacts
    // Scale threshold with resolution (higher res = more artifact segments = higher threshold)
    let cov_thresh = 0.15 * f32(params.width) / 512.0;
    let coverage = raw_cov * smoothstep(0.0, cov_thresh, raw_cov);
    if coverage > 1e-4 {
      let paint_color = evaluate_paint(paints[path_id], p);
      color = mix(color, paint_color.rgb, coverage * paint_color.a);
    }
  }

  pixels[py * params.width + px] = pack_color(color.x, color.y, color.z, 1.0);
}

// ── Fullscreen quad ──

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VSOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),  vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
  out.uv = uvs[idx];
  return out;
}

@group(0) @binding(0) var<storage, read> render_pixels: array<u32>;
@group(0) @binding(1) var<uniform>       render_params: Params;

@fragment
fn fs_fullscreen(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let px = u32(uv.x * f32(render_params.width));
  let py = u32(uv.y * f32(render_params.height));
  let idx = py * render_params.width + px;
  let packed = render_pixels[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  return vec4<f32>(r, g, b, 1.0);
}
