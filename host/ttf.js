// Minimal TTF parser — extract glyph outlines as cubic bezier paths
//
// Supports: TrueType outlines (glyf table with quadratic beziers)
// Converts: quadratic beziers → cubic beziers for snaidhm path renderer
//
// NOT supported: CFF/CFF2 (PostScript outlines), variable fonts, hinting

export class TTFFont {
  constructor(buffer) {
    this.data = new DataView(buffer);
    this.tables = {};
    this._parseOffsetTable();
    this._parseCmap();
    this._parseHead();
    this._parseMaxp();
    this._parseLoca();
  }

  // ── Table directory ──

  _parseOffsetTable() {
    const numTables = this.data.getUint16(4);
    for (let i = 0; i < numTables; i++) {
      const offset = 12 + i * 16;
      const tag = String.fromCharCode(
        this.data.getUint8(offset),
        this.data.getUint8(offset + 1),
        this.data.getUint8(offset + 2),
        this.data.getUint8(offset + 3),
      );
      this.tables[tag] = {
        offset: this.data.getUint32(offset + 8),
        length: this.data.getUint32(offset + 12),
      };
    }
  }

  // ── head table ──

  _parseHead() {
    const t = this.tables["head"];
    this.unitsPerEm = this.data.getUint16(t.offset + 18);
    this.indexToLocFormat = this.data.getInt16(t.offset + 50); // 0=short, 1=long
  }

  // ── maxp table ──

  _parseMaxp() {
    const t = this.tables["maxp"];
    this.numGlyphs = this.data.getUint16(t.offset + 4);
  }

  // ── loca table ──

  _parseLoca() {
    const t = this.tables["loca"];
    this.glyphOffsets = [];
    if (this.indexToLocFormat === 0) {
      // Short format: offsets are uint16, multiply by 2
      for (let i = 0; i <= this.numGlyphs; i++) {
        this.glyphOffsets.push(this.data.getUint16(t.offset + i * 2) * 2);
      }
    } else {
      // Long format: offsets are uint32
      for (let i = 0; i <= this.numGlyphs; i++) {
        this.glyphOffsets.push(this.data.getUint32(t.offset + i * 4));
      }
    }
  }

  // ── cmap table (format 4 — BMP Unicode) ──

  _parseCmap() {
    const t = this.tables["cmap"];
    const numSubtables = this.data.getUint16(t.offset + 2);
    let format4Offset = -1;

    for (let i = 0; i < numSubtables; i++) {
      const subtableOffset = t.offset + 4 + i * 8;
      const platformId = this.data.getUint16(subtableOffset);
      const encodingId = this.data.getUint16(subtableOffset + 2);
      const offset = this.data.getUint32(subtableOffset + 4);

      // Unicode BMP (platform 0 or 3, encoding 1)
      if ((platformId === 0 || platformId === 3) && encodingId === 1) {
        format4Offset = t.offset + offset;
        break;
      }
      if (platformId === 0) {
        format4Offset = t.offset + offset;
      }
    }

    this.cmapFormat4 = null;
    if (format4Offset >= 0) {
      const format = this.data.getUint16(format4Offset);
      if (format === 4) {
        this._parseCmapFormat4(format4Offset);
      }
    }
  }

  _parseCmapFormat4(offset) {
    const segCount = this.data.getUint16(offset + 6) / 2;
    const endCodes = [], startCodes = [], idDeltas = [], idRangeOffsets = [];
    const arrayStart = offset + 14;

    for (let i = 0; i < segCount; i++) {
      endCodes.push(this.data.getUint16(arrayStart + i * 2));
    }
    // Skip reservedPad (2 bytes)
    const startOffset = arrayStart + segCount * 2 + 2;
    for (let i = 0; i < segCount; i++) {
      startCodes.push(this.data.getUint16(startOffset + i * 2));
    }
    const deltaOffset = startOffset + segCount * 2;
    for (let i = 0; i < segCount; i++) {
      idDeltas.push(this.data.getInt16(deltaOffset + i * 2));
    }
    const rangeOffset = deltaOffset + segCount * 2;
    for (let i = 0; i < segCount; i++) {
      idRangeOffsets.push(this.data.getUint16(rangeOffset + i * 2));
    }

    this.cmapFormat4 = { segCount, endCodes, startCodes, idDeltas, idRangeOffsets, rangeOffsetBase: rangeOffset };
  }

  // ── Character → Glyph ID ──

  charToGlyphId(charCode) {
    if (!this.cmapFormat4) return 0;
    const { segCount, endCodes, startCodes, idDeltas, idRangeOffsets, rangeOffsetBase } = this.cmapFormat4;

    for (let i = 0; i < segCount; i++) {
      if (endCodes[i] >= charCode && startCodes[i] <= charCode) {
        if (idRangeOffsets[i] === 0) {
          return (charCode + idDeltas[i]) & 0xFFFF;
        } else {
          const glyphIndexOffset = rangeOffsetBase + i * 2 + idRangeOffsets[i] + (charCode - startCodes[i]) * 2;
          const glyphId = this.data.getUint16(glyphIndexOffset);
          return glyphId === 0 ? 0 : (glyphId + idDeltas[i]) & 0xFFFF;
        }
      }
    }
    return 0;
  }

  // ── hmtx table (advance widths) ──

  getAdvanceWidth(glyphId) {
    const t = this.tables["hmtx"];
    if (!t) return this.unitsPerEm;
    const t2 = this.tables["hhea"];
    const numHMetrics = t2 ? this.data.getUint16(t2.offset + 34) : this.numGlyphs;

    if (glyphId < numHMetrics) {
      return this.data.getUint16(t.offset + glyphId * 4);
    }
    // Monospaced fallback: use last entry
    return this.data.getUint16(t.offset + (numHMetrics - 1) * 4);
  }

  // ── Glyph outline extraction ──

  getGlyphOutline(glyphId) {
    const glyfTable = this.tables["glyf"];
    if (!glyfTable) return null;

    const glyphStart = glyfTable.offset + this.glyphOffsets[glyphId];
    const glyphEnd = glyfTable.offset + this.glyphOffsets[glyphId + 1];

    if (glyphStart === glyphEnd) return null; // empty glyph (e.g., space)

    const numberOfContours = this.data.getInt16(glyphStart);

    if (numberOfContours < 0) {
      // Composite glyph — extract first component only (simplified)
      return this._parseCompositeGlyph(glyphStart);
    }

    return this._parseSimpleGlyph(glyphStart, numberOfContours);
  }

  _parseSimpleGlyph(offset, numberOfContours) {
    // Skip: numberOfContours(2) + xMin(2) + yMin(2) + xMax(2) + yMax(2)
    let pos = offset + 10;

    // End points of each contour
    const endPoints = [];
    for (let i = 0; i < numberOfContours; i++) {
      endPoints.push(this.data.getUint16(pos));
      pos += 2;
    }

    const numPoints = endPoints[endPoints.length - 1] + 1;

    // Skip instructions
    const instructionLength = this.data.getUint16(pos);
    pos += 2 + instructionLength;

    // Parse flags
    const flags = [];
    while (flags.length < numPoints) {
      const flag = this.data.getUint8(pos++);
      flags.push(flag);
      if (flag & 0x08) { // repeat
        const repeat = this.data.getUint8(pos++);
        for (let r = 0; r < repeat; r++) flags.push(flag);
      }
    }

    // Parse X coordinates
    const xs = [];
    let x = 0;
    for (let i = 0; i < numPoints; i++) {
      const flag = flags[i];
      if (flag & 0x02) { // x is 1 byte
        const dx = this.data.getUint8(pos++);
        x += (flag & 0x10) ? dx : -dx;
      } else if (!(flag & 0x10)) { // x is 2 bytes (signed)
        x += this.data.getInt16(pos);
        pos += 2;
      }
      // else: same as previous (x unchanged)
      xs.push(x);
    }

    // Parse Y coordinates
    const ys = [];
    let y = 0;
    for (let i = 0; i < numPoints; i++) {
      const flag = flags[i];
      if (flag & 0x04) { // y is 1 byte
        const dy = this.data.getUint8(pos++);
        y += (flag & 0x20) ? dy : -dy;
      } else if (!(flag & 0x20)) { // y is 2 bytes (signed)
        y += this.data.getInt16(pos);
        pos += 2;
      }
      ys.push(y);
    }

    // Build contours: on-curve + off-curve points → cubic beziers
    const contours = [];
    let startIdx = 0;
    for (let c = 0; c < numberOfContours; c++) {
      const endIdx = endPoints[c];
      const points = [];
      for (let i = startIdx; i <= endIdx; i++) {
        points.push({
          x: xs[i],
          y: ys[i],
          onCurve: !!(flags[i] & 0x01),
        });
      }
      contours.push(points);
      startIdx = endIdx + 1;
    }

    return contours;
  }

  _parseCompositeGlyph(offset) {
    // Simplified: extract first component only
    let pos = offset + 10;
    const componentFlags = this.data.getUint16(pos);
    const glyphIndex = this.data.getUint16(pos + 2);
    pos += 4;

    // Read translation
    let dx = 0, dy = 0;
    if (componentFlags & 0x01) { // ARG_1_AND_2_ARE_WORDS
      dx = this.data.getInt16(pos); dy = this.data.getInt16(pos + 2); pos += 4;
    } else {
      dx = this.data.getInt8(pos); dy = this.data.getInt8(pos + 1); pos += 2;
    }

    const contours = this.getGlyphOutline(glyphIndex);
    if (!contours) return null;

    // Apply translation
    return contours.map(contour =>
      contour.map(p => ({ x: p.x + dx, y: p.y + dy, onCurve: p.onCurve }))
    );
  }
}

// ── Convert TrueType contour (quadratic) → cubic bezier paths ──

export function contoursToCubicBeziers(contours, scale, offsetX, offsetY) {
  const beziers = [];

  for (const contour of contours) {
    if (contour.length < 2) continue;

    // TrueType: off-curve points are quadratic control points.
    // Two consecutive off-curve points have an implied on-curve midpoint.
    const expanded = expandImpliedOnCurve(contour);

    let i = 0;
    while (i < expanded.length) {
      const curr = expanded[i];
      const next = expanded[(i + 1) % expanded.length];

      if (curr.onCurve && next.onCurve) {
        // Line segment → degenerate cubic
        beziers.push({
          p0: [curr.x * scale + offsetX, curr.y * scale + offsetY],
          p1: [curr.x * scale + offsetX, curr.y * scale + offsetY],
          p2: [next.x * scale + offsetX, next.y * scale + offsetY],
          p3: [next.x * scale + offsetX, next.y * scale + offsetY],
        });
        i++;
      } else if (curr.onCurve && !next.onCurve) {
        // Quadratic bezier: curr (on) → next (off) → next+1 (on)
        const ctrl = next;
        const end = expanded[(i + 2) % expanded.length];

        // Convert quadratic to cubic:
        // CP1 = P0 + 2/3 * (Q1 - P0)
        // CP2 = P2 + 2/3 * (Q1 - P2)
        const p0 = [curr.x * scale + offsetX, curr.y * scale + offsetY];
        const p3 = [end.x * scale + offsetX, end.y * scale + offsetY];
        const cp1 = [
          p0[0] + (2 / 3) * (ctrl.x * scale + offsetX - p0[0]),
          p0[1] + (2 / 3) * (ctrl.y * scale + offsetY - p0[1]),
        ];
        const cp2 = [
          p3[0] + (2 / 3) * (ctrl.x * scale + offsetX - p3[0]),
          p3[1] + (2 / 3) * (ctrl.y * scale + offsetY - p3[1]),
        ];
        beziers.push({ p0, p1: cp1, p2: cp2, p3 });
        i += 2;
      } else {
        i++; // skip unexpected pattern
      }
    }
  }

  return beziers;
}

function expandImpliedOnCurve(contour) {
  const result = [];
  for (let i = 0; i < contour.length; i++) {
    const curr = contour[i];
    const next = contour[(i + 1) % contour.length];
    result.push(curr);
    // Two consecutive off-curve → insert implied on-curve midpoint
    if (!curr.onCurve && !next.onCurve) {
      result.push({
        x: (curr.x + next.x) / 2,
        y: (curr.y + next.y) / 2,
        onCurve: true,
      });
    }
  }

  // Ensure we start on an on-curve point
  while (result.length > 0 && !result[0].onCurve) {
    result.push(result.shift());
  }

  return result;
}
