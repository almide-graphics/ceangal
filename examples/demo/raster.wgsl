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
  shadow_count: u32,
  num_paths: u32,
  scroll_y: f32,
  scroll_x: f32,
  scrollbar_opacity_y: f32,
  scrollbar_opacity_x: f32,
  _pad0: u32,
  scrollbar_hover_y: f32,
  scrollbar_hover_x: f32,
  vlist_item_h: f32,      // virtual list: item height in physical px (0 = disabled)
  vlist_item_count: f32,  // virtual list: total item count (as float)
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
@group(1) @binding(0) var<storage, read>       scroll_regions: array<vec4<f32>>;
// Region layout in scroll_regions: 3 vec4s per region (48 bytes)
//   [i*3+0] = bounds (x, y, w, h) in physical px
//   [i*3+1] = (scroll_x, scroll_y, content_w, content_h)
//   [i*3+2] = (parent_id_f, region_count_f, 0, 0)

const MAX_CMDS_PER_TILE: u32 = 8u;
const MAX_SCROLL_REGIONS: u32 = 8u;

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
  if (px >= params.width || py >= params.height) { return; }

  let w = f32(params.width);
  let h = f32(params.height);
  let max_cpx = f32(params.content_tiles_x * TILE_SIZE);
  let max_cpy = f32(params.content_tiles_y * TILE_SIZE);

  // Root scroll (region 0) from Params
  var content_px = f32(px) - params.scroll_x;
  var content_py = f32(py) - params.scroll_y;

  // Apply inner region scroll offsets (innermost match wins)
  // scroll_info: (scroll_x, scroll_y, content_w, content_h)
  let region_count = u32(scroll_regions[2].y); // [0*3+2].y = count
  for (var ri = 1u; ri < min(region_count, MAX_SCROLL_REGIONS); ri++) {
    let bounds = scroll_regions[ri * 3u];
    let scroll_info = scroll_regions[ri * 3u + 1u];
    if content_px >= bounds.x && content_px < bounds.x + bounds.z &&
       content_py >= bounds.y && content_py < bounds.y + bounds.w {
      let local_x = content_px - bounds.x;
      let local_y = content_py - bounds.y;
      // Inner content position (with scroll offset)
      let inner_cx = local_x - scroll_info.x;
      let inner_cy = local_y - scroll_info.y;
      // Clip: only apply if within inner content bounds
      if inner_cx >= 0.0 && inner_cx < scroll_info.z &&
         inner_cy >= 0.0 && inner_cy < scroll_info.w {
        content_px = bounds.x + inner_cx;
        content_py = bounds.y + inner_cy;
      }
      // else: pixel outside inner content → show outer content (no change)
    }
  }

  // ── Overscroll stretch (iOS rubber band visual, skip for virtual list) ──
  let smin_y = -(max_cpy - h);
  if params.vlist_item_h < 0.5 {  // only for non-vlist content
  let ov_top = max(params.scroll_y, 0.0);
  let ov_bot = max(-(params.scroll_y - smin_y), 0.0);
  if ov_top > 0.5 {
    let sf = clamp(ov_top / h * 0.8, 0.0, 0.35);
    let t = f32(py) / h;
    content_py = (t * (1.0 + sf) - t * t * sf) * h;
  } else if ov_bot > 0.5 {
    let sf = clamp(ov_bot / h * 0.8, 0.0, 0.35);
    let t = (h - f32(py)) / h;
    content_py = max_cpy - (t * (1.0 + sf) - t * t * sf) * h;
  }
  let smin_x = -(max_cpx - w);
  let ov_left = max(params.scroll_x, 0.0);
  let ov_right = max(-(params.scroll_x - smin_x), 0.0);
  if ov_left > 0.5 {
    let sf = clamp(ov_left / w * 0.8, 0.0, 0.35);
    let t = f32(px) / w;
    content_px = (t * (1.0 + sf) - t * t * sf) * w;
  } else if ov_right > 0.5 {
    let sf = clamp(ov_right / w * 0.8, 0.0, 0.35);
    let t = (w - f32(px)) / w;
    content_px = max_cpx - (t * (1.0 + sf) - t * t * sf) * w;
  }
  } // end vlist_item_h < 0.5 (overscroll stretch)

  if content_px < 0.0 || content_py < 0.0 {
    pixels[py * params.width + px] = pack_color(0.08, 0.08, 0.10, 1.0);
    return;
  }

  // ── Virtual list fast path: O(1) per pixel, no tiling needed ──
  if params.vlist_item_h > 0.5 {
    let total_h = params.vlist_item_h * params.vlist_item_count;
    if content_py >= total_h {
      pixels[py * params.width + px] = pack_color(0.08, 0.08, 0.10, 1.0);
      return;
    }
    let item_idx = u32(content_py / params.vlist_item_h);
    let item_top = f32(item_idx) * params.vlist_item_h;
    let item_local_y = content_py - item_top;
    let gap = 3.0;
    let margin_x = f32(params.width) * 0.03;
    let corner_r = 8.0;

    // Gap between items
    if item_local_y < gap || content_px < margin_x || content_px > f32(params.width) - margin_x {
      pixels[py * params.width + px] = pack_color(0.08, 0.08, 0.10, 1.0);
      return;
    }

    // Rounded corners via SDF
    let rect_h = params.vlist_item_h - gap;
    let rect_w = f32(params.width) - margin_x * 2.0;
    let local = vec2<f32>(content_px - margin_x - rect_w * 0.5, item_local_y - gap - rect_h * 0.5);
    let d = sd_rounded_box(local, vec2<f32>(rect_w * 0.5, rect_h * 0.5), corner_r);
    if d > 0.5 {
      pixels[py * params.width + px] = pack_color(0.08, 0.08, 0.10, 1.0);
      return;
    }

    // 6-color rotation
    let phase = item_idx % 6u;
    var cr = 0.0; var cg = 0.0; var cb = 0.0;
    if phase == 0u { cr = 0.9; cg = 0.25; cb = 0.25; }
    else if phase == 1u { cr = 0.2; cg = 0.75; cb = 0.3; }
    else if phase == 2u { cr = 0.25; cg = 0.35; cb = 0.9; }
    else if phase == 3u { cr = 0.95; cg = 0.7; cb = 0.1; }
    else if phase == 4u { cr = 0.7; cg = 0.2; cb = 0.8; }
    else { cr = 0.2; cg = 0.7; cb = 0.8; }

    // AA edge
    let aa = 1.0 - smoothstep(-1.0, 0.5, d);
    cr = mix(0.08, cr, aa); cg = mix(0.08, cg, aa); cb = mix(0.10, cb, aa);

    pixels[py * params.width + px] = pack_color(cr, cg, cb, 1.0);
    return;
  }

  if content_px >= max_cpx || content_py >= max_cpy {
    pixels[py * params.width + px] = pack_color(0.08, 0.08, 0.10, 1.0);
    return;
  }

  let content_tile_x = u32(content_px) / TILE_SIZE;
  let content_tile_y = u32(content_py) / TILE_SIZE;
  let tile_id = content_tile_y * params.content_tiles_x + content_tile_x;

  let cmd_count = min(tile_cmd_counts[tile_id], MAX_CMDS_PER_TILE);
  let tile_base = tile_id * MAX_SEGS_PER_TILE;
  let cmd_base = tile_id * MAX_CMDS_PER_TILE * 4u;

  let p = vec2<f32>(
    content_px / f32(params.width) * 2.0 - 1.0,
    1.0 - content_py / f32(params.height) * 2.0,
  );

  var color = vec3<f32>(0.08, 0.08, 0.10);

  for (var si = 0u; si < params.shadow_count; si++) {
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

fn scrollbar_sdf(px_pos: vec2<f32>, thumb_center: vec2<f32>, thumb_half: vec2<f32>, corner_r: f32) -> f32 {
  return sd_rounded_box(px_pos - thumb_center, thumb_half, corner_r);
}

@fragment
fn fs_fullscreen(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let w = f32(render_params.width);
  let h = f32(render_params.height);
  let px_u = u32(uv.x * w);
  let py_u = u32(uv.y * h);
  let idx = py_u * render_params.width + px_u;
  let packed = render_pixels[idx];
  var r = f32(packed & 0xFFu) / 255.0;
  var g = f32((packed >> 8u) & 0xFFu) / 255.0;
  var b = f32((packed >> 16u) & 0xFFu) / 255.0;

  let px_x = uv.x * w;
  let px_y = uv.y * h;

  // ── Vertical scrollbar (Cupertino-style: thin → thick on hover) ──
  let content_h = f32(render_params.content_tiles_y * TILE_SIZE);
  if content_h > h && render_params.scrollbar_opacity_y > 0.01 {
    let bar_w = mix(6.0, 10.0, render_params.scrollbar_hover_y);
    let margin = mix(3.0, 4.0, render_params.scrollbar_hover_y);
    let thumb_h = max(36.0, h * h / content_h);
    let scroll_range = content_h - h;
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
  let content_w = f32(render_params.content_tiles_x * TILE_SIZE);
  if content_w > w && render_params.scrollbar_opacity_x > 0.01 {
    let bar_h = mix(6.0, 10.0, render_params.scrollbar_hover_x);
    let margin_x = mix(3.0, 4.0, render_params.scrollbar_hover_x);
    let thumb_w = max(36.0, w * w / content_w);
    let scroll_range_x = content_w - w;
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
