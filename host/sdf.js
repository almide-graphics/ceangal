// SDF generator — compute signed distance field from glyph outlines
//
// For each pixel in the SDF bitmap, compute minimum distance to
// the nearest glyph edge. Inside = positive, outside = negative.
// Stored as uint8: 128 = edge, >128 = inside, <128 = outside.

// Distance from point to line segment
function distToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-10) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
  const cx = ax + t * dx, cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

// Winding number at point (for inside/outside detection)
function windingNumber(px, py, segments) {
  let winding = 0;
  for (const seg of segments) {
    const ax = seg[0], ay = seg[1], bx = seg[2], by = seg[3];
    if (ay <= py) {
      if (by > py) {
        const v = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        if (v > 0) winding++;
      }
    } else {
      if (by <= py) {
        const v = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        if (v < 0) winding--;
      }
    }
  }
  return winding;
}

// Generate SDF bitmap for a set of line segments
// Returns { data: Uint8Array, width, height }
export function generateSDF(segments, bounds, sdfSize, padding) {
  const { xMin, yMin, xMax, yMax } = bounds;
  const glyphW = xMax - xMin || 1;
  const glyphH = yMax - yMin || 1;

  // SDF bitmap dimensions
  const aspect = glyphW / glyphH;
  let w, h;
  if (aspect >= 1) {
    w = sdfSize;
    h = Math.max(1, Math.round(sdfSize / aspect));
  } else {
    h = sdfSize;
    w = Math.max(1, Math.round(sdfSize * aspect));
  }
  w += padding * 2;
  h += padding * 2;

  const data = new Uint8Array(w * h);
  const scale = (sdfSize / Math.max(glyphW, glyphH));
  const radius = padding; // SDF radius in pixels

  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      // Map pixel to glyph coordinates
      const gx = xMin + (px - padding) / scale;
      const gy = yMax - (py - padding) / scale; // Y flipped (TTF Y up, bitmap Y down)

      // Find minimum distance to any segment
      let minDist = Infinity;
      for (const seg of segments) {
        const d = distToSegment(gx, gy, seg[0], seg[1], seg[2], seg[3]);
        minDist = Math.min(minDist, d);
      }

      // Convert to pixel distance
      const pixelDist = minDist * scale;

      // Determine inside/outside
      const inside = Math.abs(windingNumber(gx, gy, segments)) > 0;

      // Signed distance: positive inside, negative outside
      const signedDist = inside ? pixelDist : -pixelDist;

      // Map to 0-255 range: 128 = edge, 128+radius = fully inside
      const value = Math.round(128 + signedDist * (128 / radius));
      data[py * w + px] = Math.max(0, Math.min(255, value));
    }
  }

  return { data, width: w, height: h, scale, padding };
}

// Generate SDF atlas for a set of characters
// Returns { atlasData, atlasWidth, atlasHeight, glyphs: Map<char, glyphInfo> }
export function generateSDFAtlas(font, chars, sdfSize = 48, padding = 6) {
  const glyphSDFs = [];

  for (const ch of chars) {
    const charCode = ch.codePointAt(0);
    const glyphId = font.charToGlyphId(charCode);
    const advance = font.getAdvanceWidth(glyphId);
    const contours = font.getGlyphOutline(glyphId);

    if (!contours || contours.length === 0) {
      // Space or empty glyph
      glyphSDFs.push({
        char: ch, glyphId, advance,
        sdf: null, segments: [],
        bounds: { xMin: 0, yMin: 0, xMax: 0, yMax: 0 },
      });
      continue;
    }

    // Flatten contours to line segments
    const segments = [];
    let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;

    for (const contour of contours) {
      const expanded = expandImpliedOnCurve(contour);
      let i = 0;
      while (i < expanded.length) {
        const curr = expanded[i];
        const next = expanded[(i + 1) % expanded.length];

        let pts;
        if (curr.onCurve && next.onCurve) {
          pts = [[curr.x, curr.y], [next.x, next.y]];
          i++;
        } else if (curr.onCurve && !next.onCurve) {
          const end = expanded[(i + 2) % expanded.length];
          // Subdivide quadratic into line segments
          const steps = 8;
          for (let s = 0; s < steps; s++) {
            const t0 = s / steps, t1 = (s + 1) / steps;
            const p0 = evalQuad(curr, next, end, t0);
            const p1 = evalQuad(curr, next, end, t1);
            segments.push([p0.x, p0.y, p1.x, p1.y]);
            xMin = Math.min(xMin, p0.x, p1.x);
            yMin = Math.min(yMin, p0.y, p1.y);
            xMax = Math.max(xMax, p0.x, p1.x);
            yMax = Math.max(yMax, p0.y, p1.y);
          }
          i += 2;
          continue;
        } else {
          i++;
          continue;
        }

        if (pts) {
          segments.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
          for (const p of pts) {
            xMin = Math.min(xMin, p[0]); yMin = Math.min(yMin, p[1]);
            xMax = Math.max(xMax, p[0]); yMax = Math.max(yMax, p[1]);
          }
        }
      }
    }

    const bounds = { xMin, yMin, xMax, yMax };
    const sdf = segments.length > 0 ? generateSDF(segments, bounds, sdfSize, padding) : null;

    glyphSDFs.push({ char: ch, glyphId, advance, sdf, segments, bounds });
  }

  // Pack into atlas (simple horizontal strip)
  let atlasWidth = 0, atlasHeight = 0;
  for (const g of glyphSDFs) {
    if (g.sdf) {
      atlasWidth += g.sdf.width;
      atlasHeight = Math.max(atlasHeight, g.sdf.height);
    }
  }
  // Pad to power of 2 for GPU
  atlasWidth = nextPow2(Math.max(atlasWidth, 1));
  atlasHeight = nextPow2(Math.max(atlasHeight, 1));

  const atlasData = new Uint8Array(atlasWidth * atlasHeight);
  const glyphs = new Map();
  let x = 0;

  for (const g of glyphSDFs) {
    if (g.sdf) {
      // Copy SDF into atlas
      for (let row = 0; row < g.sdf.height; row++) {
        for (let col = 0; col < g.sdf.width; col++) {
          atlasData[row * atlasWidth + x + col] = g.sdf.data[row * g.sdf.width + col];
        }
      }
      glyphs.set(g.char, {
        atlasX: x, atlasY: 0,
        atlasW: g.sdf.width, atlasH: g.sdf.height,
        advance: g.advance,
        bounds: g.bounds,
        sdfScale: g.sdf.scale,
        padding: g.sdf.padding,
      });
      x += g.sdf.width;
    } else {
      glyphs.set(g.char, {
        atlasX: 0, atlasY: 0, atlasW: 0, atlasH: 0,
        advance: g.advance,
        bounds: { xMin: 0, yMin: 0, xMax: 0, yMax: 0 },
        sdfScale: 1, padding: 0,
      });
    }
  }

  return { atlasData, atlasWidth, atlasHeight, glyphs };
}

function nextPow2(n) {
  let v = 1;
  while (v < n) v <<= 1;
  return v;
}

function expandImpliedOnCurve(contour) {
  const result = [];
  for (let i = 0; i < contour.length; i++) {
    const curr = contour[i];
    const next = contour[(i + 1) % contour.length];
    result.push(curr);
    if (!curr.onCurve && !next.onCurve) {
      result.push({ x: (curr.x + next.x) / 2, y: (curr.y + next.y) / 2, onCurve: true });
    }
  }
  while (result.length > 0 && !result[0].onCurve) result.push(result.shift());
  return result;
}

function evalQuad(p0, p1, p2, t) {
  const u = 1 - t;
  return {
    x: u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
    y: u * u * p0.y + 2 * u * t * p1.y + t * t * p2.y,
  };
}
