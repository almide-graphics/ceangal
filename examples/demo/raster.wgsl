// snaidhm Phase 1 — Tiled path renderer (content-space 2D scroll + scrollbar)

const TILE_SIZE: u32 = 16u;
const MAX_SEGS_PER_TILE: u32 = 16u;

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
  content_tiles_x: u32,
  content_tiles_y: u32,
  list_clip_top: f32,
  list_clip_bottom: f32,
  scroll_y: f32,
  scroll_x: f32,
  scrollbar_opacity_y: f32,
  scrollbar_opacity_x: f32,
  hover_item_idx: f32,
  scrollbar_hover_y: f32,
  scrollbar_hover_x: f32,
  mouse_x: f32,           // mouse position (physical px) for light effect
  mouse_y: f32,           // mouse position (physical px)
}

struct Shadow {
  center: vec2<f32>,
  half_size: vec2<f32>,
  corner_radius: f32,
  offset_x: f32,
  offset_y: f32,
  blur: f32,
  color: vec4<f32>,
}

struct Paint {
  paint_type: u32,
  _p0: u32, _p1: u32, _p2: u32,
  color0: vec4<f32>,
  color1: vec4<f32>,
  grad_params: vec4<f32>,
}

fn evaluate_paint(paint: Paint, p: vec2<f32>) -> vec4<f32> {
  if paint.paint_type == 1u {
    let start = paint.grad_params.xy;
    let dir = paint.grad_params.zw - start;
    let t = clamp(dot(p - start, dir) / dot(dir, dir), 0.0, 1.0);
    return mix(paint.color0, paint.color1, t);
  } else if paint.paint_type == 2u {
    let center = paint.grad_params.xy;
    let radius = paint.grad_params.z;
    let t = clamp(length(p - center) / radius, 0.0, 1.0);
    return mix(paint.color0, paint.color1, t);
  }
  return paint.color0;
}

@group(0) @binding(0) var<storage, read>       segments: array<LineSeg>;
@group(0) @binding(1) var<storage, read>       tile_cmd_counts: array<u32>;
@group(0) @binding(2) var<storage, read>       tile_seg_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(4) var<uniform>             params: Params;
@group(0) @binding(5) var<storage, read>       shadows: array<Shadow>;
@group(0) @binding(6) var<storage, read>       paints: array<Paint>;
@group(0) @binding(7) var<storage, read>       tile_cmds: array<u32>;
@group(1) @binding(0) var<storage, read>       render_items: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>       scroll_regions: array<vec4<f32>>;
// Item layout: 3 vec4s per item (48 bytes)
//   [i*3+0] = (x, y, w, h) in physical px
//   [i*3+1] = (bg_r, bg_g, bg_b, bg_a)
//   [i*3+2] = (rounded, opacity, scrollable, item_count)
// Region layout in scroll_regions: 3 vec4s per region (48 bytes)
//   [i*3+0] = bounds (x, y, w, h) in physical px
//   [i*3+1] = (scroll_x, scroll_y, content_w, content_h)
//   [i*3+2] = (parent_id_f, region_count_f, 0, 0)

const MAX_CMDS_PER_TILE: u32 = 8u;
const MAX_SCROLL_REGIONS: u32 = 8u;
// Tile-based item dispatch (group 3 — compute only, avoids group 2 conflict with bg_texture)
@group(3) @binding(0) var<storage, read>       tile_item_counts: array<u32>;
@group(3) @binding(1) var<storage, read>       tile_item_ids: array<u32>;
const DISPATCH_TILE: u32 = 64u;
const MAX_ITEMS_PER_TILE: u32 = 32u;

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
  let xmin = min(min(x0, x1), 1.0) - 1e-6;
  let xmax = max(x0, x1);
  let b = min(xmax, 1.0);
  let c = max(b, 0.0);
  let d = max(xmin, 0.0);
  let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
  return (1.0 - a) * dy;
}

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

  // Viewport-sized buffer: items positioned at viewport-relative coords
  if (px >= params.width || py >= params.height) { return; }

  // ── Tile-based item dispatch (tile buffers in group 0 bindings 8-9) ──
  let dispatch_tiles_x = (params.width + DISPATCH_TILE - 1u) / DISPATCH_TILE;
  let dtx = px / DISPATCH_TILE;
  let dty = py / DISPATCH_TILE;
  let tile_id = dty * dispatch_tiles_x + dtx;
  let ri_count = tile_item_counts[tile_id];

  if ri_count > 0u {
    var color = vec3<f32>(0.0);
    var alpha = 0.0;

    for (var i = 0u; i < min(ri_count, MAX_ITEMS_PER_TILE); i++) {
      let ri = tile_item_ids[tile_id * MAX_ITEMS_PER_TILE + i];
      let pos = render_items[ri * 3u];
      let col = render_items[ri * 3u + 1u];
      let item_meta = render_items[ri * 3u + 2u];

      let ix = pos.x;
      let scrollable = item_meta.z;
      let iy = pos.y + params.scroll_y * scrollable;
      let iw = pos.z; let ih = pos.w;

      if iw < 1.0 || ih < 1.0 || col.w < 0.01 { continue; }
      if iy + ih < 0.0 || iy > f32(params.height) { continue; }
      if ix + iw < 0.0 || ix > f32(params.width) { continue; }

      if scrollable > 0.5 {
        if f32(py) < params.list_clip_top || f32(py) >= params.list_clip_bottom { continue; }
      }

      let fpx = f32(px);
      let fpy = f32(py);
      if fpx < ix || fpx > ix + iw || fpy < iy || fpy > iy + ih { continue; }

      let corner_r = item_meta.x;
      let local = vec2<f32>(fpx - ix - iw * 0.5, fpy - iy - ih * 0.5);
      let half = vec2<f32>(iw * 0.5, ih * 0.5);
      let d = sd_rounded_box(local, half, corner_r);

      if d < 1.0 && col.w > 0.01 {
        let aa = 1.0 - smoothstep(-1.0, 0.5, d);
        let a = aa * col.w * item_meta.y;
        color = mix(color, col.xyz, a);
        alpha = alpha + a * (1.0 - alpha);
      }
    }
    pixels[py * params.width + px] = pack_color(color.x, color.y, color.z, alpha);
    return;
  }

  // Empty tile in items mode: write same as old path background (opaque dark)
  let global_item_count = u32(render_items[2].w);
  if global_item_count > 0u {
    pixels[py * params.width + px] = pack_color(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // ── Legacy path rendering (segments) ──
  let max_cpx = f32(params.content_tiles_x * TILE_SIZE);
  let max_cpy = f32(params.content_tiles_y * TILE_SIZE);
  var content_px = f32(px);
  var content_py = f32(py);

  if content_px >= max_cpx || content_py >= max_cpy {
    pixels[py * params.width + px] = pack_color(0.08, 0.08, 0.10, 1.0);
    return;
  }

  let content_tile_x = u32(content_px) / TILE_SIZE;
  let content_tile_y = u32(content_py) / TILE_SIZE;
  let path_tile_id = content_tile_y * params.content_tiles_x + content_tile_x;

  let cmd_count = min(tile_cmd_counts[path_tile_id], MAX_CMDS_PER_TILE);
  let tile_base = path_tile_id * MAX_SEGS_PER_TILE;
  let cmd_base = path_tile_id * MAX_CMDS_PER_TILE * 4u;

  let p = vec2<f32>(
    content_px / f32(params.width) * 2.0 - 1.0,
    1.0 - content_py / f32(params.height) * 2.0,
  );

  var color = vec3<f32>(0.08, 0.08, 0.10);

  for (var si = 0u; si < 0u; si++) {  // shadows disabled (field repurposed for list_clip)
    let shadow = shadows[si];
    let sp = p - vec2<f32>(shadow.offset_x, shadow.offset_y);
    let d = sd_rounded_box(sp - shadow.center, shadow.half_size, shadow.corner_radius);
    let shadow_alpha = (1.0 - smoothstep(-shadow.blur * 0.3, shadow.blur, d)) * shadow.color.a;
    color = mix(color, shadow.color.rgb, shadow_alpha);
  }

  let ndc_to_px = 0.5 * f32(params.width);
  let ndc_to_py = 0.5 * f32(params.height);

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
        (seg.p0.x + 1.0) * ndc_to_px - content_px,
        (1.0 - seg.p0.y) * ndc_to_py - content_py,
      );
      let sp1 = vec2<f32>(
        (seg.p1.x + 1.0) * ndc_to_px - content_px,
        (1.0 - seg.p1.y) * ndc_to_py - content_py,
      );
      area += seg_area(sp0, sp1);
    }

    let raw_cov = min(abs(area), 1.0);
    let cov_thresh = 0.15 * f32(params.width) / 512.0;
    let coverage = raw_cov * smoothstep(0.0, cov_thresh, raw_cov);
    if coverage > 1e-4 {
      let paint_color = evaluate_paint(paints[path_id], p);
      color = mix(color, paint_color.rgb, coverage * paint_color.a);
    }
  }

  pixels[py * params.width + px] = pack_color(color.x, color.y, color.z, 1.0);
}

// ── Fullscreen quad + scrollbar overlay ──

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
@group(2) @binding(0) var bg_texture: texture_2d<f32>;
@group(2) @binding(1) var bg_sampler: sampler;

fn scrollbar_sdf(px_pos: vec2<f32>, thumb_center: vec2<f32>, thumb_half: vec2<f32>, corner_r: f32) -> f32 {
  return sd_rounded_box(px_pos - thumb_center, thumb_half, corner_r);
}

// GPU glass: box blur compositing bg_texture + content items
fn sample_composited(cx: i32, cy: i32, content_w: u32, content_h: u32, vp_w: f32, vp_h: f32, scroll_x: f32, scroll_y: f32) -> vec3<f32> {
  // Convert content coord to viewport UV for bg sampling
  let vp_x = f32(cx) + scroll_x;
  let vp_y = f32(cy) + scroll_y;
  let bg_uv = vec2<f32>(vp_x / vp_w, vp_y / vp_h);
  let bg = textureSampleLevel(bg_texture, bg_sampler, clamp(bg_uv, vec2(0.0), vec2(1.0)), 0.0);
  var color = bg.xyz;

  // Composite content item on top
  let rx = u32(clamp(cx, 0, i32(content_w) - 1));
  let ry = u32(clamp(cy, 0, i32(content_h) - 1));
  if cx >= 0 && cy >= 0 && cx < i32(content_w) && cy < i32(content_h) {
    let p = render_pixels[ry * content_w + rx];
    let ir = f32(p & 0xFFu) / 255.0;
    let ig = f32((p >> 8u) & 0xFFu) / 255.0;
    let ib = f32((p >> 16u) & 0xFFu) / 255.0;
    let ia = f32((p >> 24u) & 0xFFu) / 255.0;
    color = mix(color, vec3(ir, ig, ib), ia);
  }
  return color;
}

fn blur_at_composited(cx: i32, cy: i32, content_w: u32, content_h: u32, vp_w: f32, vp_h: f32, scroll_x: f32, scroll_y: f32, radius: i32) -> vec3<f32> {
  var sum = vec3<f32>(0.0);
  var n = 0.0;
  for (var dy = -radius; dy <= radius; dy += 3) {
    for (var dx = -radius; dx <= radius; dx += 3) {
      sum += sample_composited(cx + dx, cy + dy, content_w, content_h, vp_w, vp_h, scroll_x, scroll_y);
      n += 1.0;
    }
  }
  return sum / n;
}

@fragment
fn fs_fullscreen(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let w = f32(render_params.width);
  let h = f32(render_params.height);
  let w_u = render_params.width;
  let h_u = render_params.height;
  // Content buffer dimensions
  let content_w = max(render_params.width, render_params.content_tiles_x * TILE_SIZE);
  let content_h = max(render_params.height, render_params.content_tiles_y * TILE_SIZE);

  // Viewport pixel position
  let px_u = u32(uv.x * w);
  let py_u = u32(uv.y * h);

  // ── Background: texture sample at viewport UV (fixed, doesn't scroll) ──
  let bg_sample = textureSample(bg_texture, bg_sampler, uv);
  var r = bg_sample.x;
  var g = bg_sample.y;
  var b = bg_sample.z;

  // Read items from viewport-sized pixel buffer (already scroll-positioned)
  let idx = py_u * w_u + px_u;
  let packed = render_pixels[idx];
  let item_r = f32(packed & 0xFFu) / 255.0;
  let item_g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let item_b = f32((packed >> 16u) & 0xFFu) / 255.0;
  let item_a = f32((packed >> 24u) & 0xFFu) / 255.0;
  // Alpha composite: items over background
  r = mix(r, item_r, item_a);
  g = mix(g, item_g, item_a);
  b = mix(b, item_b, item_a);

  // ── Hover highlight (fragment, O(1)) ──
  let hover_ri = i32(render_params.hover_item_idx);
  if hover_ri >= 0 {
    let ri_u = u32(hover_ri);
    let hpos = render_items[ri_u * 3u];
    let hmeta = render_items[ri_u * 3u + 2u];
    let hx = hpos.x; let hy = hpos.y; let hw = hpos.z; let hh = hpos.w;
    let hcr = hmeta.x;
    if f32(px_u) >= hx && f32(px_u) < hx + hw && f32(py_u) >= hy && f32(py_u) < hy + hh {
      let local_h = vec2<f32>(f32(px_u) - hx - hw * 0.5, f32(py_u) - hy - hh * 0.5);
      let half_h = vec2<f32>(hw * 0.5, hh * 0.5);
      let d_h = sd_rounded_box(local_h, half_h, hcr);
      if d_h < 0.5 {
        let aa_h = 1.0 - smoothstep(-1.0, 0.5, d_h);
        r += aa_h * 0.06; g += aa_h * 0.06; b += aa_h * 0.06;
      }
    }
  }

  // ── Mouse light (fragment shader — no compute re-run needed) ──
  let mouse_pos = vec2<f32>(render_params.mouse_x, render_params.mouse_y);
  if mouse_pos.x > 1.0 || mouse_pos.y > 1.0 {
    let dist = length(vec2<f32>(f32(px_u), f32(py_u)) - mouse_pos);
    let glow = exp(-dist * dist / (150.0 * 150.0 * 4.0));
    r += glow * 0.18; g += glow * 0.12; b += glow * 0.28;
  }

  // ── Noise grain ──
  let ns = vec2<f32>(f32(px_u) * 0.7 + 0.1, f32(py_u) * 1.3 + 0.7);
  let grain = fract(sin(dot(ns, vec2<f32>(12.9898, 78.233))) * 43758.5453);
  r += (grain - 0.5) * 0.015; g += (grain - 0.5) * 0.015; b += (grain - 0.5) * 0.015;

  // ── Glass blur: disabled for now (TODO: optimize with mip chain) ──

  let px_x = uv.x * w;
  let px_y = uv.y * h;

  // ── Vertical scrollbar (Cupertino-style: thin → thick on hover) ──
  let scroll_content_h = f32(render_params.content_tiles_y * TILE_SIZE);
  if scroll_content_h > h && render_params.scrollbar_opacity_y > 0.01 {
    let bar_w = mix(6.0, 10.0, render_params.scrollbar_hover_y);
    let margin = mix(3.0, 4.0, render_params.scrollbar_hover_y);
    let thumb_h = max(36.0, h * h / scroll_content_h);
    let scroll_range = scroll_content_h - h;
    let scroll_frac = clamp(-render_params.scroll_y / scroll_range, 0.0, 1.0);
    let thumb_y = scroll_frac * (h - thumb_h);

    let center = vec2<f32>(w - margin - bar_w * 0.5, thumb_y + thumb_h * 0.5);
    let half = vec2<f32>(bar_w * 0.5, thumb_h * 0.5);
    let d = scrollbar_sdf(vec2<f32>(px_x, px_y), center, half, bar_w * 0.5);
    let a = (1.0 - smoothstep(-1.0, 0.5, d)) * 0.6 * render_params.scrollbar_opacity_y;
    r = mix(r, 1.0, a);
    g = mix(g, 1.0, a);
    b = mix(b, 1.0, a);
  }

  // ── Horizontal scrollbar ──
  let scroll_content_w = f32(render_params.content_tiles_x * TILE_SIZE);
  if scroll_content_w > w && render_params.scrollbar_opacity_x > 0.01 {
    let bar_h = mix(6.0, 10.0, render_params.scrollbar_hover_x);
    let margin_x = mix(3.0, 4.0, render_params.scrollbar_hover_x);
    let thumb_w = max(36.0, w * w / scroll_content_w);
    let scroll_range_x = scroll_content_w - w;
    let scroll_frac_x = clamp(-render_params.scroll_x / scroll_range_x, 0.0, 1.0);
    let thumb_x = scroll_frac_x * (w - thumb_w);

    let center_x = vec2<f32>(thumb_x + thumb_w * 0.5, h - margin_x - bar_h * 0.5);
    let half_x = vec2<f32>(thumb_w * 0.5, bar_h * 0.5);
    let d_x = scrollbar_sdf(vec2<f32>(px_x, px_y), center_x, half_x, bar_h * 0.5);
    let a_x = (1.0 - smoothstep(-1.0, 0.5, d_x)) * 0.6 * render_params.scrollbar_opacity_x;
    r = mix(r, 1.0, a_x);
    g = mix(g, 1.0, a_x);
    b = mix(b, 1.0, a_x);
  }

  // ── Inner region scrollbars ──
  // [i*3+0]=bounds, [i*3+1]=(scroll_x, scroll_y, content_w, content_h)
  // [i*3+2]=(parent_id, region_count, bar_opacity_y, bar_opacity_x)
  let rgn_count = u32(scroll_regions[2].y);
  for (var ri = 1u; ri < min(rgn_count, MAX_SCROLL_REGIONS); ri++) {
    let rgn_bounds = scroll_regions[ri * 3u];
    let rgn_scroll = scroll_regions[ri * 3u + 1u];
    let rgn_meta = scroll_regions[ri * 3u + 2u];
    let rgn_bar_oy = rgn_meta.z;
    let rgn_vp_w = rgn_bounds.z;
    let rgn_vp_h = rgn_bounds.w;
    let rgn_content_h = rgn_scroll.w;

    let screen_x = rgn_bounds.x + render_params.scroll_x;
    let screen_y = rgn_bounds.y + render_params.scroll_y;

    // Vertical scrollbar (with fade)
    if rgn_content_h > rgn_vp_h && rgn_bar_oy > 0.01 {
      let inner_range = rgn_content_h - rgn_vp_h;
      let inner_frac = clamp(-rgn_scroll.y / inner_range, 0.0, 1.0);
      let bar_w = 5.0;
      let margin_r = 2.0;
      let thumb_h = max(24.0, rgn_vp_h * rgn_vp_h / rgn_content_h);
      let thumb_y = screen_y + inner_frac * (rgn_vp_h - thumb_h);
      let bar_x = screen_x + rgn_vp_w - margin_r - bar_w;

      if bar_x > 0.0 && bar_x < w && thumb_y < h && thumb_y + thumb_h > 0.0 {
        let center_v = vec2<f32>(bar_x + bar_w * 0.5, thumb_y + thumb_h * 0.5);
        let half_v = vec2<f32>(bar_w * 0.5, thumb_h * 0.5);
        let d_v = scrollbar_sdf(vec2<f32>(px_x, px_y), center_v, half_v, bar_w * 0.5);
        let a_v = (1.0 - smoothstep(-1.0, 0.5, d_v)) * 0.55 * rgn_bar_oy;
        r = mix(r, 1.0, a_v);
        g = mix(g, 1.0, a_v);
        b = mix(b, 1.0, a_v);
      }
    }
  }

  return vec4<f32>(r, g, b, 1.0);
}
