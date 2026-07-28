// SDF text renderer — sample SDF atlas texture, smoothstep for crisp edges

struct Params {
  atlas_width: f32,
  atlas_height: f32,
  screen_width: f32,
  screen_height: f32,
}

struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var sdf_texture: texture_2d<f32>;
@group(0) @binding(2) var sdf_sampler: sampler;

// Vertex: positioned quad per glyph
// Vertex data: pos(2) + uv(2) + color(4) = 8 floats per vertex
@vertex
fn vs_main(
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  // pos is in NDC [-1, 1]
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.uv = uv;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = textureSample(sdf_texture, sdf_sampler, in.uv).r;

  // Debug: show raw SDF value as grayscale
  // return vec4<f32>(dist, dist, dist, 1.0);

  // SDF: 0.5 = edge (stored as 128/255 ≈ 0.502)
  let edge = 0.502;
  // Adaptive smoothing based on screen-space derivatives
  let dx = dpdx(in.uv.x) * params.atlas_width;
  let dy = dpdy(in.uv.y) * params.atlas_height;
  let spread = clamp(0.5 * length(vec2<f32>(dx, dy)), 0.02, 0.5);
  let alpha = smoothstep(edge - spread, edge + spread, dist);

  if (alpha < 0.01) { discard; }

  return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
