/**
 * Lightweight DXF → Room parser
 *
 * Handles:
 *   - LWPOLYLINE (closed flag = 1) as room boundaries
 *   - TEXT / MTEXT as room labels
 *   - $INSUNITS for unit-scale detection
 *
 * No external dependencies — pure string parsing.
 * Outputs rooms in metres, centred around the plan's centroid.
 */

// ─── Public types ────────────────────────────────────────────

export interface DxfRoom {
  /** Label from the nearest TEXT entity inside the polygon, or a fallback. */
  name: string
  /** Width in metres (clamped 1–30). */
  width: number
  /** Depth in metres (clamped 1–30). */
  depth: number
  /** Scene X centre in metres (normalised so plan centroid = 0). */
  centerX: number
  /** Scene Z centre in metres (from DXF Y axis). */
  centerZ: number
  /** Suggested zone fill colour derived from the room name. */
  color: string
  /**
   * Optional explicit polygon outline in scene coordinates ([x, z] pairs,
   * closed — last vertex does NOT repeat first). When provided, the room
   * is built as walls along these edges instead of a width×depth rectangle.
   */
  polygon?: Array<[number, number]>
  /**
   * Outline-only flag: when true, walls are created but no zone fill.
   * Used for building exterior outlines (Floor 1 / Floor 2 perimeters).
   */
  outlineOnly?: boolean
  /**
   * Zone-only flag: when true, the room's zone polygon is created (for the
   * label / fill colour) but no walls. Used for SVG imports where wall geometry
   * already comes from separate wall segments.
   */
  zoneOnly?: boolean
  /**
   * For 2-point single-wall rooms: doors and windows mounted on this wall.
   * `position` is the distance along the wall from start to the attachment's
   * centre, in metres; `width` and `height` are the opening dimensions.
   */
  attachments?: Array<{
    type: 'door' | 'window'
    position: number
    width: number
    height: number
  }>
}

export type DxfParseResult =
  | { ok: true; rooms: DxfRoom[] }
  | { ok: false; message: string }

// ─── DXF tokeniser ──────────────────────────────────────────

interface Pair {
  code: number
  value: string
}

function tokenise(text: string): Pair[] {
  const lines = text.split(/\r?\n/)
  const pairs: Pair[] = []
  for (let i = 0; i + 1 < lines.length; i += 2) {
    const codeLine = lines[i]
    const valueLine = lines[i + 1]
    if (codeLine === undefined || valueLine === undefined) continue
    const code = parseInt(codeLine.trim(), 10)
    if (!isNaN(code)) {
      pairs.push({ code, value: valueLine.trim() })
    }
  }
  return pairs
}

// ─── Unit resolution ($INSUNITS → mm factor) ────────────────

function unitToMmFactor(insunits: number): number {
  switch (insunits) {
    case 1:
      return 25.4 // inches
    case 2:
      return 304.8 // feet
    case 4:
      return 1 // mm ← most common in architecture
    case 5:
      return 10 // cm
    case 6:
      return 1000 // m
    default:
      return 1 // 0 = unitless → assume mm
  }
}

// ─── Raw entity structs ──────────────────────────────────────

interface RawPolyline {
  layer: string
  closed: boolean
  vertices: Array<{ x: number; y: number }>
}

interface RawText {
  x: number
  y: number
  text: string
}

// ─── Entity parser ───────────────────────────────────────────

interface ParsedEntities {
  insunits: number
  polylines: RawPolyline[]
  texts: RawText[]
}

function stripMtextFormatting(raw: string): string {
  return raw
    .replace(/\\[Pp]/g, ' ')
    .replace(/\{[^}]*\}/g, '')
    .replace(/\\[a-zA-Z][^;]*/g, '')
    .trim()
}

function parseEntities(pairs: Pair[]): ParsedEntities {
  const polylines: RawPolyline[] = []
  const texts: RawText[] = []
  let insunits = 4 // default: mm

  type State = 'none' | 'polyline' | 'text'
  let state: State = 'none'
  let pendingX: number | null = null
  let watchInsunits = false

  let poly: RawPolyline = { layer: '', closed: false, vertices: [] }
  let txt: RawText = { x: 0, y: 0, text: '' }

  const flush = () => {
    if (state === 'polyline' && poly.closed && poly.vertices.length >= 3) {
      polylines.push(poly)
    } else if (state === 'text' && txt.text.trim()) {
      texts.push(txt)
    }
  }

  for (const { code, value } of pairs) {
    // $INSUNITS watch
    if (code === 9 && value === '$INSUNITS') {
      watchInsunits = true
      continue
    }
    if (watchInsunits && code === 70) {
      insunits = parseInt(value, 10)
      watchInsunits = false
      continue
    }
    watchInsunits = false

    // Entity type boundary
    if (code === 0) {
      flush()
      pendingX = null
      if (value === 'LWPOLYLINE') {
        poly = { layer: '', closed: false, vertices: [] }
        state = 'polyline'
      } else if (value === 'TEXT' || value === 'MTEXT') {
        txt = { x: 0, y: 0, text: '' }
        state = 'text'
      } else {
        state = 'none'
      }
      continue
    }

    if (state === 'polyline') {
      if (code === 8) poly.layer = value
      else if (code === 70) poly.closed = (parseInt(value, 10) & 1) === 1
      else if (code === 10) pendingX = parseFloat(value)
      else if (code === 20 && pendingX !== null) {
        poly.vertices.push({ x: pendingX, y: parseFloat(value) })
        pendingX = null
      }
    } else if (state === 'text') {
      if (code === 10) txt.x = parseFloat(value)
      else if (code === 20) txt.y = parseFloat(value)
      else if (code === 1) txt.text = stripMtextFormatting(value)
    }
  }
  flush()

  return { insunits, polylines, texts }
}

// ─── Geometry helpers ────────────────────────────────────────

function polygonArea(vertices: Array<{ x: number; y: number }>): number {
  let area = 0
  const n = vertices.length
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    area += vertices[i]!.x * vertices[j]!.y
    area -= vertices[j]!.x * vertices[i]!.y
  }
  return Math.abs(area) / 2
}

function pointInPolygon(
  px: number,
  py: number,
  vertices: Array<{ x: number; y: number }>,
): boolean {
  let inside = false
  const n = vertices.length
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = vertices[i]!.x
    const yi = vertices[i]!.y
    const xj = vertices[j]!.x
    const yj = vertices[j]!.y
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }
  return inside
}

function boundingBox(vertices: Array<{ x: number; y: number }>) {
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity
  for (const { x, y } of vertices) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  return {
    minX,
    maxX,
    minY,
    maxY,
    width: maxX - minX,
    height: maxY - minY,
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
  }
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v))
}

// ─── Room colour hinting ─────────────────────────────────────

function colorForRoom(name: string): string {
  const n = name.toLowerCase()
  if (/living|lounge|客廳|起居/.test(n)) return '#dbeafe'
  if (/master.*(bed|room)|主臥|主人房/.test(n)) return '#d1fae5'
  if (/bed|room|臥|睡/.test(n)) return '#dcfce7'
  if (/kitchen|廚/.test(n)) return '#fef9c3'
  if (/dining|餐/.test(n)) return '#fce7f3'
  if (/bath|toilet|shower|衛|浴|廁/.test(n)) return '#e0f2fe'
  if (/balcony|terrace|陽台|露台/.test(n)) return '#f0fdf4'
  if (/hall|foyer|entry|corridor|走廊|玄關|門廳/.test(n)) return '#f5f5f4'
  if (/study|office|書房|辦公/.test(n)) return '#faf5ff'
  return '#f3f4f6'
}

// ─── Metadata comment parser (fallback) ─────────────────────
//
// Some DXF files (e.g. "Image→CAD Analyzer" output) carry no usable
// room geometry but embed room names + areas in `;` comment lines:
//
//   ; Living Room: 24.5 mm (area) @ West/left zone
//
// When no LWPOLYLINE rooms are found we fall back to this parser.
// The "mm" unit label in these files is a misnomer — the values are
// actually m², which gives realistic room sizes (24.5 m² living room).

interface MetaRoom {
  name: string
  areaM2: number
  location: string
}

/** Typical width:depth aspect ratios for common room types. */
function aspectRatio(name: string): number {
  const n = name.toLowerCase()
  if (/living|lounge/.test(n)) return 1.4
  if (/dining/.test(n)) return 1.3
  if (/kitchen/.test(n)) return 1.5
  if (/bath|toilet|shower/.test(n)) return 1.4
  if (/balcony|terrace/.test(n)) return 2.2
  if (/hall|corridor|foyer/.test(n)) return 2.0
  return 1.2 // bedrooms, studies, etc.
}

function estimateDims(name: string, areaM2: number): [number, number] {
  const ar = aspectRatio(name)
  // area = w × d, w = ar × d  →  d = √(area / ar)
  const depth = Math.sqrt(areaM2 / ar)
  const width = ar * depth
  return [
    Math.round(clamp(width, 1, 30) * 100) / 100,
    Math.round(clamp(depth, 1, 30) * 100) / 100,
  ]
}

/** Map a location hint string to a [col, row] cell in a 3 × 3 grid. */
function locationToCell(loc: string): [number, number] {
  const l = loc.toLowerCase()
  const col = /west|left/.test(l) ? 0 : /east|right/.test(l) ? 2 : 1
  const row = /north/.test(l) ? 0 : /south/.test(l) ? 2 : 1
  return [col, row]
}

/**
 * Auto-layout a list of sized rooms into a grid, centred at (0, 0).
 * Rooms that share a grid cell are stacked along Z with small gaps.
 */
function layoutRooms(
  sized: Array<{ name: string; width: number; depth: number; color: string; col: number; row: number }>,
): DxfRoom[] {
  const SPACING = 1.2 // metres between cells
  const GRID = 3

  // Maximum room dims per column / row
  const colW = [0, 0, 0] as [number, number, number]
  const rowD = [0, 0, 0] as [number, number, number]
  for (const r of sized) {
    if (r.width > (colW[r.col] ?? 0)) colW[r.col] = r.width
    if (r.depth > (rowD[r.row] ?? 0)) rowD[r.row] = r.depth
  }

  const totalW = colW.reduce((s, w) => s + w, 0) + SPACING * (GRID - 1)
  const totalD = rowD.reduce((s, d) => s + d, 0) + SPACING * (GRID - 1)

  // Cell centre X values
  const colX: number[] = []
  let cx = -totalW / 2
  for (let c = 0; c < GRID; c++) {
    colX.push(cx + colW[c]! / 2)
    cx += colW[c]! + SPACING
  }

  // Cell centre Z values
  const rowZ: number[] = []
  let cz = -totalD / 2
  for (let r = 0; r < GRID; r++) {
    rowZ.push(cz + rowD[r]! / 2)
    cz += rowD[r]! + SPACING
  }

  // Stack rooms inside the same cell along Z
  const cellStack = new Map<string, number>()

  return sized.map((r) => {
    const key = `${r.col},${r.row}`
    const stackIdx = cellStack.get(key) ?? 0
    cellStack.set(key, stackIdx + 1)

    const zOffset = stackIdx * (r.depth + 0.6)
    return {
      name: r.name,
      width: r.width,
      depth: r.depth,
      centerX: Math.round((colX[r.col]! + stackIdx * 0.4) * 100) / 100,
      centerZ: Math.round((rowZ[r.row]! + zOffset) * 100) / 100,
      color: r.color,
    }
  })
}

function parseMetadataComments(dxfText: string): DxfRoom[] | null {
  // Match lines like:  ; Living Room: 24.5 mm (area) @ West/left zone
  // Requires "(area)" to skip dimension-only annotations like "(length)".
  const pattern =
    /^[;\s]*([A-Za-z][^:]{2,50}):\s*([\d.]+)\s*(?:mm|m[²2])?\s*\(area\)(?:\s*@\s*([^\n]*))?/gm

  const raws: MetaRoom[] = []
  for (const m of dxfText.matchAll(pattern)) {
    const areaM2 = parseFloat(m[2]!)
    if (!isFinite(areaM2) || areaM2 < 1) continue // skip zero-area and noise
    raws.push({ name: m[1]!.trim(), areaM2, location: m[3]?.trim() ?? '' })
  }

  if (raws.length === 0) return null

  const sized = raws.map((r) => {
    const [w, d] = estimateDims(r.name, r.areaM2)
    const [col, row] = locationToCell(r.location)
    return { name: r.name, width: w, depth: d, color: colorForRoom(r.name), col, row }
  })

  return layoutRooms(sized)
}

// ─── Main export ─────────────────────────────────────────────

/**
 * Parse a DXF string and return detected rooms in scene coordinates (metres).
 *
 * Primary path: closed LWPOLYLINE entities with area ≥ 0.25 m².
 * Fallback path: room metadata embedded in `;` comment lines
 *   (e.g. "Image→CAD Analyzer" skeleton DXFs).
 */
export function parseDxfToRooms(dxfText: string): DxfParseResult {
  let pairs: Pair[]
  try {
    pairs = tokenise(dxfText)
  } catch {
    return { ok: false, message: 'DXF 格式無法解析。' }
  }

  const { insunits, polylines, texts } = parseEntities(pairs)
  const mmFactor = unitToMmFactor(insunits)

  // Scale all coordinates to mm
  const scaledPolylines = polylines.map((p) => ({
    ...p,
    vertices: p.vertices.map((v) => ({ x: v.x * mmFactor, y: v.y * mmFactor })),
  }))
  const scaledTexts = texts.map((t) => ({
    ...t,
    x: t.x * mmFactor,
    y: t.y * mmFactor,
  }))

  // Keep only polylines large enough to be rooms (≥ 0.25 m², min side ≥ 500 mm)
  const MIN_AREA_MM2 = 250_000
  const MIN_DIM_MM = 500

  const roomPolylines = scaledPolylines.filter((p) => {
    const area = polygonArea(p.vertices)
    if (area < MIN_AREA_MM2) return false
    const bb = boundingBox(p.vertices)
    return bb.width >= MIN_DIM_MM && bb.height >= MIN_DIM_MM
  })

  if (roomPolylines.length === 0) {
    // ── Fallback: try parsing room metadata from ; comment lines ──
    const metaRooms = parseMetadataComments(dxfText)
    if (metaRooms && metaRooms.length > 0) {
      return { ok: true, rooms: metaRooms }
    }

    return {
      ok: false,
      message:
        '找不到房間輪廓。DXF 需包含 closed LWPOLYLINE（flag=1）且面積 ≥ 0.25 m² 的多邊形，或在 ; 開頭的 comment 中標注房間面積（area）。',
    }
  }

  // Sort largest first
  roomPolylines.sort((a, b) => polygonArea(b.vertices) - polygonArea(a.vertices))

  const allBBoxes = roomPolylines.map((p) => boundingBox(p.vertices))

  // Centroid of all room centres → use as scene origin
  const avgCX = allBBoxes.reduce((s, b) => s + b.centerX, 0) / allBBoxes.length
  const avgCY = allBBoxes.reduce((s, b) => s + b.centerY, 0) / allBBoxes.length

  const usedTextIndexes = new Set<number>()
  let fallback = 1

  const rooms: DxfRoom[] = roomPolylines.map((poly, i) => {
    const bb = allBBoxes[i]!

    // Find the text closest to the centroid that is inside this polygon
    let bestLabel = ''
    let bestDist = Infinity
    let bestIdx = -1
    for (let j = 0; j < scaledTexts.length; j++) {
      if (usedTextIndexes.has(j)) continue
      const t = scaledTexts[j]!
      if (pointInPolygon(t.x, t.y, poly.vertices)) {
        const dist = Math.hypot(t.x - bb.centerX, t.y - bb.centerY)
        if (dist < bestDist) {
          bestDist = dist
          bestLabel = t.text
          bestIdx = j
        }
      }
    }
    if (bestIdx >= 0) usedTextIndexes.add(bestIdx)

    const name = bestLabel || `Room ${fallback++}`
    const widthM = clamp(bb.width / 1000, 1, 30)
    const depthM = clamp(bb.height / 1000, 1, 30)

    return {
      name,
      width: Math.round(widthM * 100) / 100,
      depth: Math.round(depthM * 100) / 100,
      centerX: Math.round(((bb.centerX - avgCX) / 1000) * 100) / 100,
      centerZ: Math.round(((bb.centerY - avgCY) / 1000) * 100) / 100,
      color: colorForRoom(name),
    }
  })

  return { ok: true, rooms }
}

// ─── SVG floor plan parser ────────────────────────────────────
//
// Handles Matterport FCL-style SVG exports where each room is a
// <text transform="translate(x y)"> element whose text content is:
//
//   ROOM NAME          ← first line (name)
//   24'4" x 18'0"      ← second tspan (dimensions in feet/inches)
//
// Rooms without explicit dimensions are skipped.
//
// Scale is derived from the floor outline polygon:
//   scale = sqrt(sum_of_room_areas_m2 / largest_polygon_area_svg)
// This handles floor plans of any size without a hardcoded constant.

function parseTranslate(transform: string): [number, number] | null {
  const m = transform.match(/translate\(\s*([-\d.]+)[,\s]+([-\d.]+)\s*\)/)
  return m ? [parseFloat(m[1]!), parseFloat(m[2]!)] : null
}

function stripSvgTags(html: string): string {
  return html
    .replace(/<\/?tspan[^>]*>/g, '') // tspan = kerning only, remove without space
    .replace(/<[^>]+>/g, ' ') // other tags → space
    .replace(/\s+/g, ' ')
    .trim()
}

/** Convert a feet-inches string like "24'4"" (ASCII or Unicode curly quotes) to metres. */
function parseFeetInches(str: string): number {
  // Matches both ASCII ' / " and Unicode typographic ' (U+2019) / " (U+201D)
  const m = str.match(/(\d+)['\u2019]\s*(\d*)/)
  if (!m) return 0
  const feet = parseInt(m[1]!, 10)
  const inches = m[2] ? parseInt(m[2], 10) : 0
  return feet * 0.3048 + inches * 0.0254
}

/** Parse a polygon points="..." string into coordinate pairs. */
function parsePolygonPoints(pointsStr: string): [number, number][] {
  const nums = pointsStr.match(/[-\d.]+/g)?.map(Number) ?? []
  const pts: [number, number][] = []
  for (let i = 0; i + 1 < nums.length; i += 2) {
    pts.push([nums[i]!, nums[i + 1]!])
  }
  return pts
}

// ─── SVG wall extraction (C-lite: extract all walls, no room reconstruction) ─
//
// The Matterport FCL SVG doesn't expose a clean wall graph — walls are scattered
// across polygons, paths, and rects with thickness offsets, gaps at doors, and
// multiple stylistic redraws of the same wall. So instead of trying to detect
// rooms from a planar arrangement (which would need ~1 week of robust geometry
// code), we just extract EVERY wall-like line segment and let the editor render
// them as individual walls. The user gets a complete floor plan with all walls
// in correct positions; room labels remain rectangular placeholders.
//
// Wall sources:
//   - polygon edges of cls-1 (the filled floor outlines)
//   - path segments of cls-19 (the stroked floor perimeter)
//   - midlines of thin polygon walls (cls-20/21/22 etc.)
//   - midlines of thin rect walls (any wall-shaped rectangle)

type Seg = [[number, number], [number, number]]

/** Parse SVG path 'd' attribute into a list of straight-line segments.
 *  Handles M/L/H/V/Z (absolute + relative). Curve commands (C/S/Q/T/A)
 *  are silently skipped — for wall extraction we only care about the
 *  straight parts of mixed paths.
 */
function parsePathSegments(d: string): Seg[] {
  const tokens = d.match(/[MLHVZCSQTAmlhvzcsqta]|-?\d+\.?\d*(?:e-?\d+)?/g)
  if (!tokens) return []
  const segs: Seg[] = []
  let cx = 0,
    cy = 0 // current point
  let sx = 0,
    sy = 0 // subpath start
  let cmd: string | null = null
  let i = 0
  const isLetter = (s: string | undefined): boolean =>
    !!s && /^[MLHVZCSQTAmlhvzcsqta]$/.test(s)
  while (i < tokens.length) {
    const t = tokens[i]!
    if (isLetter(t)) {
      cmd = t
      i++
      if (cmd === 'Z' || cmd === 'z') {
        if (cx !== sx || cy !== sy) segs.push([[cx, cy], [sx, sy]])
        cx = sx
        cy = sy
      }
      continue
    }
    // Curve commands: advance position to the end point and skip control points.
    // Each one may repeat implicitly until a new letter command appears.
    if ((cmd === 'C' || cmd === 'c') && !isLetter(tokens[i + 5])) {
      const a = tokens.slice(i, i + 6).map(parseFloat)
      cx = cmd === 'c' ? cx + a[4]! : a[4]!
      cy = cmd === 'c' ? cy + a[5]! : a[5]!
      i += 6
      continue
    }
    if (
      (cmd === 'S' || cmd === 's' || cmd === 'Q' || cmd === 'q') &&
      !isLetter(tokens[i + 3])
    ) {
      const a = tokens.slice(i, i + 4).map(parseFloat)
      cx = cmd === cmd.toLowerCase() ? cx + a[2]! : a[2]!
      cy = cmd === cmd.toLowerCase() ? cy + a[3]! : a[3]!
      i += 4
      continue
    }
    if ((cmd === 'T' || cmd === 't') && !isLetter(tokens[i + 1])) {
      const a = tokens.slice(i, i + 2).map(parseFloat)
      cx = cmd === 't' ? cx + a[0]! : a[0]!
      cy = cmd === 't' ? cy + a[1]! : a[1]!
      i += 2
      continue
    }
    if ((cmd === 'A' || cmd === 'a') && !isLetter(tokens[i + 6])) {
      const a = tokens.slice(i, i + 7).map(parseFloat)
      cx = cmd === 'a' ? cx + a[5]! : a[5]!
      cy = cmd === 'a' ? cy + a[6]! : a[6]!
      i += 7
      continue
    }
    // Straight-line commands
    if (cmd === 'M' || cmd === 'm') {
      const nx = parseFloat(tokens[i]!)
      const ny = parseFloat(tokens[i + 1]!)
      const ex = cmd === 'm' ? cx + nx : nx
      const ey = cmd === 'm' ? cy + ny : ny
      cx = ex
      cy = ey
      sx = ex
      sy = ey
      // After M, implicit subsequent coords are L/l
      cmd = cmd === 'M' ? 'L' : 'l'
      i += 2
      continue
    }
    if (cmd === 'L' || cmd === 'l') {
      const nx = parseFloat(tokens[i]!)
      const ny = parseFloat(tokens[i + 1]!)
      const ex = cmd === 'l' ? cx + nx : nx
      const ey = cmd === 'l' ? cy + ny : ny
      segs.push([[cx, cy], [ex, ey]])
      cx = ex
      cy = ey
      i += 2
      continue
    }
    if (cmd === 'H' || cmd === 'h') {
      const nx = parseFloat(tokens[i]!)
      const ex = cmd === 'h' ? cx + nx : nx
      segs.push([[cx, cy], [ex, cy]])
      cx = ex
      i += 1
      continue
    }
    if (cmd === 'V' || cmd === 'v') {
      const ny = parseFloat(tokens[i]!)
      const ey = cmd === 'v' ? cy + ny : ny
      segs.push([[cx, cy], [cx, ey]])
      cy = ey
      i += 1
      continue
    }
    // Unknown command — skip this token
    i++
  }
  return segs
}

/** Parse a single SVG <rect ... /> tag (with optional rotation transform) into
 *  4 corner points. Only handles rotate() and translate() transforms.
 */
function parseRectCorners(attrs: string): [number, number][] | null {
  const num = (name: string) => {
    const m = attrs.match(new RegExp(`\\b${name}="([-\\d.]+)"`))
    return m ? parseFloat(m[1]!) : null
  }
  const x = num('x') ?? 0
  const y = num('y') ?? 0
  const w = num('width')
  const h = num('height')
  if (w === null || h === null || w <= 0 || h <= 0) return null
  // Local corners (clockwise)
  let corners: [number, number][] = [
    [x, y],
    [x + w, y],
    [x + w, y + h],
    [x, y + h],
  ]
  // Apply transform if present
  const trans = attrs.match(/transform="([^"]+)"/)
  if (trans) {
    const t = trans[1]!
    // translate(tx, ty) or translate(tx ty)
    for (const m of t.matchAll(/translate\(\s*([-\d.]+)[\s,]+([-\d.]+)\s*\)/g)) {
      const tx = parseFloat(m[1]!)
      const ty = parseFloat(m[2]!)
      corners = corners.map(([cx, cy]) => [cx + tx, cy + ty])
    }
    // rotate(deg) or rotate(deg cx cy)
    for (const m of t.matchAll(/rotate\(\s*([-\d.]+)(?:[\s,]+([-\d.]+)[\s,]+([-\d.]+))?\s*\)/g)) {
      const deg = parseFloat(m[1]!)
      const rcx = m[2] ? parseFloat(m[2]) : 0
      const rcy = m[3] ? parseFloat(m[3]) : 0
      const rad = (deg * Math.PI) / 180
      const c = Math.cos(rad)
      const s = Math.sin(rad)
      corners = corners.map(([cx, cy]) => {
        const dx = cx - rcx
        const dy = cy - rcy
        return [rcx + dx * c - dy * s, rcy + dx * s + dy * c]
      })
    }
  }
  return corners
}

/** Centerline of a thin rectangle polygon (4 corners + optional close pt). */
function extractWallMidline(pts: [number, number][]): Seg | null {
  // Use the first 4 distinct points as corners
  const corners = pts.slice(0, 4)
  if (corners.length < 4) return null
  // Lengths of the 4 sides
  const sides: Array<{ a: [number, number]; b: [number, number]; len: number }> = []
  for (let i = 0; i < 4; i++) {
    const a = corners[i]!
    const b = corners[(i + 1) % 4]!
    sides.push({ a, b, len: Math.hypot(b[0] - a[0], b[1] - a[1]) })
  }
  sides.sort((x, y) => x.len - y.len)
  // Two shortest sides = the "short ends" of the rectangle
  const e1 = sides[0]!
  const e2 = sides[1]!
  // Midline = midpoint(e1) → midpoint(e2)
  const m1: [number, number] = [(e1.a[0] + e1.b[0]) / 2, (e1.a[1] + e1.b[1]) / 2]
  const m2: [number, number] = [(e2.a[0] + e2.b[0]) / 2, (e2.a[1] + e2.b[1]) / 2]
  // Degenerate guard
  if (Math.hypot(m2[0] - m1[0], m2[1] - m1[1]) < 0.5) return null
  return [m1, m2]
}

/** Two segments are duplicates if their endpoint sets coincide within epsilon. */
function segsAreDuplicate(s1: Seg, s2: Seg, eps: number): boolean {
  const d = (a: [number, number], b: [number, number]) => Math.hypot(a[0] - b[0], a[1] - b[1])
  return (
    (d(s1[0], s2[0]) < eps && d(s1[1], s2[1]) < eps) ||
    (d(s1[0], s2[1]) < eps && d(s1[1], s2[0]) < eps)
  )
}

// ─── Wall merging & opening attachment ───────────────────────────────────
//
// "C-full" pipeline: take all classified segments → merge axis-aligned
// collinear neighbours into long walls → attach windows and door openings
// at their parametric position along the merged wall.

type SegKind = 'exterior' | 'interior' | 'window'

interface ClassifiedSeg {
  seg: Seg
  kind: SegKind
}

interface OpeningAttachment {
  type: 'door' | 'window'
  /** 0..1 along the wall (start → end) */
  startFraction: number
  /** 0..1 along the wall */
  endFraction: number
}

interface LongWall {
  start: [number, number]
  end: [number, number]
  axis: 'h' | 'v'
  attachments: OpeningAttachment[]
}

/** Classify a thin axis-aligned segment as horizontal or vertical, returning
 *  the normalised form (start = smaller coord on the primary axis). */
function classifyAxis(
  s: Seg,
  tol: number,
): { axis: 'h' | 'v'; perp: number; lo: number; hi: number } | null {
  const dx = s[1][0] - s[0][0]
  const dy = s[1][1] - s[0][1]
  if (Math.abs(dy) <= tol && Math.abs(dx) > tol) {
    // Horizontal: perp = y, primary = x
    const perp = (s[0][1] + s[1][1]) / 2
    const lo = Math.min(s[0][0], s[1][0])
    const hi = Math.max(s[0][0], s[1][0])
    return { axis: 'h', perp, lo, hi }
  }
  if (Math.abs(dx) <= tol && Math.abs(dy) > tol) {
    const perp = (s[0][0] + s[1][0]) / 2
    const lo = Math.min(s[0][1], s[1][1])
    const hi = Math.max(s[0][1], s[1][1])
    return { axis: 'v', perp, lo, hi }
  }
  return null
}

/** Group collinear axis-aligned wall segments and merge each group into the
 *  smallest set of long walls. Adjacent or overlapping intervals are unioned;
 *  intervals separated by more than `bridgeGap` (e.g. wider than a door)
 *  become separate walls.
 */
function mergeIntoLongWalls(
  segs: ClassifiedSeg[],
  options: { axisTol: number; perpBucket: number; bridgeGap: number },
): LongWall[] {
  const { axisTol, perpBucket, bridgeGap } = options

  // Group key: axis + bucketed perpendicular coord
  const groups = new Map<string, Array<{ lo: number; hi: number; kind: SegKind }>>()
  for (const { seg, kind } of segs) {
    if (kind === 'window') continue // windows merge separately; here we want only structural walls
    const c = classifyAxis(seg, axisTol)
    if (!c) continue
    const bucket = Math.round(c.perp / perpBucket)
    const key = `${c.axis}:${bucket}`
    let arr = groups.get(key)
    if (!arr) {
      arr = []
      groups.set(key, arr)
    }
    arr.push({ lo: c.lo, hi: c.hi, kind })
  }

  const walls: LongWall[] = []
  for (const [key, intervals] of groups) {
    const [axis, bucketStr] = key.split(':')
    const perp = parseInt(bucketStr!, 10) * perpBucket
    intervals.sort((a, b) => a.lo - b.lo)

    let curLo = intervals[0]!.lo
    let curHi = intervals[0]!.hi
    for (let i = 1; i < intervals.length; i++) {
      const iv = intervals[i]!
      if (iv.lo <= curHi + bridgeGap) {
        // overlap or adjacent → extend
        if (iv.hi > curHi) curHi = iv.hi
      } else {
        // gap too large → emit current and start new
        walls.push(makeWall(axis as 'h' | 'v', perp, curLo, curHi))
        curLo = iv.lo
        curHi = iv.hi
      }
    }
    walls.push(makeWall(axis as 'h' | 'v', perp, curLo, curHi))
  }
  return walls
}

function makeWall(axis: 'h' | 'v', perp: number, lo: number, hi: number): LongWall {
  if (axis === 'h') {
    return {
      start: [lo, perp],
      end: [hi, perp],
      axis,
      attachments: [],
    }
  }
  return {
    start: [perp, lo],
    end: [perp, hi],
    axis,
    attachments: [],
  }
}

/** Find the long wall best matching this axis-aligned segment (same axis,
 *  perpendicular coord within tol, segment interval inside wall interval).
 */
function findHostWall(
  segSeg: Seg,
  walls: LongWall[],
  perpTol: number,
): { wall: LongWall; startFrac: number; endFrac: number } | null {
  const c = classifyAxis(segSeg, perpTol)
  if (!c) return null
  for (const w of walls) {
    if (w.axis !== c.axis) continue
    const wPerp = w.axis === 'h' ? w.start[1] : w.start[0]
    if (Math.abs(wPerp - c.perp) > perpTol) continue
    const wLo = w.axis === 'h' ? w.start[0] : w.start[1]
    const wHi = w.axis === 'h' ? w.end[0] : w.end[1]
    if (c.lo < wLo - perpTol || c.hi > wHi + perpTol) continue
    const wallLen = wHi - wLo
    if (wallLen <= 0) continue
    const startFrac = Math.max(0, Math.min(1, (c.lo - wLo) / wallLen))
    const endFrac = Math.max(0, Math.min(1, (c.hi - wLo) / wallLen))
    return { wall: w, startFrac, endFrac }
  }
  return null
}

/** Shoelace area of a polygon. */
function svgPolygonArea(pts: [number, number][]): number {
  const n = pts.length
  let a = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    a += pts[i]![0] * pts[j]![1] - pts[j]![0] * pts[i]![1]
  }
  return Math.abs(a) / 2
}

/** Ray-cast point-in-polygon test for [x, y] tuple polygons. */
function svgPointInPoly(px: number, py: number, pts: [number, number][]): boolean {
  const n = pts.length
  let inside = false
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = pts[i]![0]
    const yi = pts[i]![1]
    const xj = pts[j]![0]
    const yj = pts[j]![1]
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-12) + xi) {
      inside = !inside
    }
  }
  return inside
}

/** Centroid of a polygon via the standard shoelace formula. */
function svgPolygonCentroid(pts: [number, number][]): [number, number] {
  const n = pts.length
  let A = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    A += pts[i]![0] * pts[j]![1] - pts[j]![0] * pts[i]![1]
  }
  A /= 2
  if (Math.abs(A) < 1e-9) {
    const cx = pts.reduce((s, p) => s + p[0], 0) / n
    const cy = pts.reduce((s, p) => s + p[1], 0) / n
    return [cx, cy]
  }
  let cx = 0
  let cy = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    const cross = pts[i]![0] * pts[j]![1] - pts[j]![0] * pts[i]![1]
    cx += (pts[i]![0] + pts[j]![0]) * cross
    cy += (pts[i]![1] + pts[j]![1]) * cross
  }
  return [cx / (6 * A), cy / (6 * A)]
}

/**
 * Parse a Matterport (or compatible) SVG floor plan.
 *
 * Extracts room labels whose text contains a `D'D" x D'D"` dimension
 * string. Scale is derived automatically from the largest polygon in the
 * SVG (the floor outline) so it works for floor plans of any size.
 */
export function parseSvgToRooms(svgText: string): DxfParseResult {
  // Match every <text …> … </text> block (single-line SVGs need the s flag)
  const textPattern = /<text([^>]*)>([\s\S]*?)<\/text>/g

  interface Label {
    name: string
    widthM: number
    depthM: number
    svgX: number
    svgY: number
  }

  const labels: Label[] = []

  for (const match of svgText.matchAll(textPattern)) {
    const attrs = match[1] ?? ''
    const inner = match[2] ?? ''

    // Extract SVG position from transform="translate(x y)"
    const transformMatch = attrs.match(/transform="([^"]*)"/)
    if (!transformMatch) continue
    const pos = parseTranslate(transformMatch[1]!)
    if (!pos) continue

    // Collapse tspan markup to plain text
    const text = stripSvgTags(inner)

    // Look for feet-inches dimensions: 24'4" x 18'0"
    // Character classes include both ASCII ('/") and typographic curly quotes
    // (U+2019 RIGHT SINGLE QUOTATION MARK, U+201D RIGHT DOUBLE QUOTATION MARK).
    const dimsMatch = text.match(/(\d+['\u2019]\s*\d*["\u201d])\s*[xX×]\s*(\d+['\u2019]\s*\d*["\u201d])/)
    if (!dimsMatch) continue

    const widthM = parseFeetInches(dimsMatch[1]!)
    const depthM = parseFeetInches(dimsMatch[2]!)

    // Skip objects too small to be rooms (appliances, fixtures)
    if (widthM < 0.8 || depthM < 0.8) continue

    // Room name = everything before the dimension string
    const dimStart = text.indexOf(dimsMatch[0])
    const rawName = text.slice(0, dimStart).trim()
    const name = rawName || 'Room'

    labels.push({
      name,
      widthM: Math.round(widthM * 100) / 100,
      depthM: Math.round(depthM * 100) / 100,
      svgX: pos[0],
      svgY: pos[1],
    })
  }

  if (labels.length === 0) {
    return {
      ok: false,
      message:
        "SVG 中找不到房間資訊。需要包含 D'D\" × D'D\" 格式的尺寸標籤（例如 Matterport FCL 格式）。",
    }
  }

  // ── Scale from floor-outline polygon ───────────────────────────────────
  // Find the largest polygon — this is the floor boundary in Matterport FCL.
  // Derive scale = sqrt(totalRoomAreaM2 / floorPolygonAreaSVG) so the layout
  // is correct for any floor plan size without a hardcoded constant.
  let originX = labels.reduce((s, l) => s + l.svgX, 0) / labels.length
  let originY = labels.reduce((s, l) => s + l.svgY, 0) / labels.length
  let SCALE = 0.009 // fallback

  // Parse every polygon in the SVG so we can (a) derive scale, (b) match
  // labels to room polygons, (c) emit floor outlines as exterior walls,
  // (d) classify wall-shaped polygons by their CSS class (interior wall vs
  // window glass vs other).
  interface ParsedPoly {
    cls: string
    pts: [number, number][]
    area: number
  }
  const allPolys: ParsedPoly[] = []
  for (const m of svgText.matchAll(/<polygon([^>]*)\bpoints="([^"]+)"/g)) {
    const attrs = m[1]!
    const ptsStr = m[2]!
    const clsMatch = attrs.match(/class="([^"]+)"/)
    const cls = clsMatch ? clsMatch[1]! : ''
    const pts = parsePolygonPoints(ptsStr)
    if (pts.length < 3) continue
    allPolys.push({ cls, pts, area: svgPolygonArea(pts) })
  }
  allPolys.sort((a, b) => b.area - a.area)

  // Identify floor outlines: the largest polygon, plus any others within 40%
  // of its area (handles dual-floor SVGs with Floor 1 + Floor 2 outlines).
  const floorOutlines: ParsedPoly[] = []
  if (allPolys.length > 0) {
    const biggest = allPolys[0]!.area
    for (const p of allPolys) {
      if (p.area >= biggest * 0.4) floorOutlines.push(p)
      else break
    }
  }

  if (floorOutlines.length > 0) {
    const totalOutlineArea = floorOutlines.reduce((s, p) => s + p.area, 0)
    const totalRoomAreaM2 = labels.reduce((s, l) => s + l.widthM * l.depthM, 0)
    if (totalRoomAreaM2 > 0) {
      SCALE = Math.sqrt(totalRoomAreaM2 / totalOutlineArea)
    }
    const [cx, cy] = svgPolygonCentroid(floorOutlines[0]!.pts)
    originX = cx
    originY = cy
  }

  // SVG \u2192 scene helper (mm precision)
  const toScene = (svgX: number, svgY: number): [number, number] => [
    Math.round((svgX - originX) * SCALE * 1000) / 1000,
    Math.round((svgY - originY) * SCALE * 1000) / 1000,
  ]

  // Candidate polygons for individual rooms = all polys minus floor outlines,
  // and only those big enough to be a real room (skip tiny fixtures).
  const outlineSet = new Set(floorOutlines)
  const candidatePolys = allPolys.filter((p) => !outlineSet.has(p) && p.area >= 500)

  // For SVG imports we only emit rooms that have a real polygon shape from the
  // SVG (BEDROOM, PRIMARY BATHROOM, \u2026). Rectangle placeholders get dropped \u2014
  // the walls already come from the SVG wall network, so adding placeholder
  // rectangle walls would just create visual clashes. The user can manually
  // label remaining areas in the editor.
  const rooms: DxfRoom[] = []
  for (const l of labels) {
    const containing = candidatePolys
      .filter((p) => svgPointInPoly(l.svgX, l.svgY, p.pts))
      .sort((a, b) => a.area - b.area)
    const match = containing[0]
    if (!match) continue

    const polyAreaM2 = match.area * SCALE * SCALE
    const labelAreaM2 = l.widthM * l.depthM
    if (polyAreaM2 < labelAreaM2 * 0.5 || polyAreaM2 > labelAreaM2 * 2.5) continue

    const [cx, cy] = svgPolygonCentroid(match.pts)
    const c = toScene(cx, cy)
    rooms.push({
      name: l.name,
      width: clamp(l.widthM, 1, 30),
      depth: clamp(l.depthM, 1, 30),
      centerX: c[0],
      centerZ: c[1],
      color: colorForRoom(l.name),
      polygon: match.pts.map(([x, y]) => toScene(x, y)),
      // Zone-only: the label / colour fill, no walls \u2014 wall geometry comes
      // from the SVG wall segments emitted later.
      zoneOnly: true,
    })
  }

  // \u2500\u2500 Classify wall-shaped segments by source \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  const classified: ClassifiedSeg[] = []

  // 1. Floor outline edges \u2192 exterior walls
  for (const outline of floorOutlines) {
    const pts = outline.pts
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i]!
      const b = pts[(i + 1) % pts.length]!
      if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 0.5) {
        classified.push({ seg: [a, b], kind: 'exterior' })
      }
    }
  }

  // 2a. Thin wall polygons \u2192 take midline as interior wall.
  //     cls-20 = filled walls (dark fill, visually solid). cls-21 = stroke-
  //     only thin walls \u2014 required for rooms like PANTRY/OFFICE/BATHROOM
  //     that share the floor's fill colour and have walls drawn only as a
  //     black-stroke thin rectangle. cls-22 is excluded (floor-coloured fill
  //     mostly = decorative outline duplicating cls-21 lines).
  const windowClasses = new Set(['cls-23'])
  const thinWallClasses = new Set(['cls-20', 'cls-21'])
  for (const p of allPolys) {
    if (floorOutlines.includes(p)) continue
    if (p.pts.length < 4 || p.pts.length > 6) continue
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity
    for (const [x, y] of p.pts) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    const long = Math.max(maxX - minX, maxY - minY)
    const short = Math.min(maxX - minX, maxY - minY)
    if (short < 0.5 || long / short < 2.5 || short > 20) continue
    const ml = extractWallMidline(p.pts)
    if (!ml) continue
    if (windowClasses.has(p.cls)) classified.push({ seg: ml, kind: 'window' })
    else if (thinWallClasses.has(p.cls)) classified.push({ seg: ml, kind: 'interior' })
  }

  // 2b. Indoor room polygons (filled with room colour) \u2192 each edge is an
  //     interior wall. Only rooms with INDOOR fills count \u2014 DECK/VERANDA
  //     (cls-4/5/9/14/27, fill #f4f0e5) and pure-stroke decoration are
  //     skipped. cls-3 = filled room (BEDROOM), cls-15 = bathroom fills.
  const indoorRoomClasses = new Set(['cls-3', 'cls-15', 'cls-34', 'cls-38'])
  for (const p of allPolys) {
    if (floorOutlines.includes(p)) continue
    if (p.pts.length < 5) continue
    if (!indoorRoomClasses.has(p.cls)) continue
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity
    for (const [x, y] of p.pts) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    const w = maxX - minX
    const h = maxY - minY
    if (Math.min(w, h) < 50) continue // skip fixtures
    if (w * h < 5000) continue
    const pts = p.pts
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i]!
      const b = pts[(i + 1) % pts.length]!
      if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 5) {
        classified.push({ seg: [a, b], kind: 'interior' })
      }
    }
  }

  // \u2500\u2500 Merge collinear walls into long walls \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // bridgeGap = 90 SVG units (~0.85 m) so the wall spans across door openings
  // (typical door 0.8\u20130.9 m). Wider openings (hallway connections) still
  // break the wall into two. perpBucket = 8 covers half-wall-thickness offset
  // jitter when interior and exterior walls were drawn at slightly different
  // perpendicular coordinates.
  const longWalls = mergeIntoLongWalls(classified, {
    axisTol: 3,
    perpBucket: 8,
    bridgeGap: 90,
  })

  // \u2500\u2500 Attach windows to their host walls \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // For each window midline, find a collinear long wall that contains it,
  // and add an attachment with the window's start/end fraction along the wall.
  const windowSegs = classified.filter((c) => c.kind === 'window').map((c) => c.seg)
  let windowsAttached = 0
  for (const seg of windowSegs) {
    const host = findHostWall(seg, longWalls, 8)
    if (!host) continue
    host.wall.attachments.push({
      type: 'window',
      startFraction: Math.min(host.startFrac, host.endFrac),
      endFraction: Math.max(host.startFrac, host.endFrac),
    })
    windowsAttached++
  }

  // \u2500\u2500 Detect doors from swing-arc paths \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // A door swing path traces a closed shape: the hinge edge lies on a wall,
  // the rest is the arc/panel extending INTO a room. So the wall-aligned
  // door opening is one of the 4 sides of the path's bounding box \u2014 we test
  // all 4 against the long-wall network and pick whichever matches.
  let doorsAttached = 0
  let doorCandidates = 0
  for (const m of svgText.matchAll(/<path[^>]*\bd="([^"]+)"/g)) {
    const d = m[1]!
    // Require an actual arc command (A/a) with separator context so we don't
    // match the letter "a" inside attribute names or numbers.
    if (!/(?:^|[\s,0-9.\-])[Aa](?:[\s,0-9.\-]|$)/.test(d)) continue
    const segs = parsePathSegments(d)
    if (segs.length === 0) continue
    let pminX = Infinity,
      pmaxX = -Infinity,
      pminY = Infinity,
      pmaxY = -Infinity
    for (const s of segs) {
      for (const p of s) {
        if (p[0] < pminX) pminX = p[0]
        if (p[0] > pmaxX) pmaxX = p[0]
        if (p[1] < pminY) pminY = p[1]
        if (p[1] > pmaxY) pmaxY = p[1]
      }
    }
    const dw = pmaxX - pminX
    const dh = pmaxY - pminY
    // Door swing bbox: roughly square (radius ≈ width), door-sized.
    const longSide = Math.max(dw, dh)
    const shortSide = Math.min(dw, dh)
    if (longSide < 55 || longSide > 120) continue
    if (longSide / Math.max(shortSide, 1) > 1.7) continue
    doorCandidates++

    // First traced point of the path = door pivot (hinge on wall).
    const pivot = segs[0]![0]

    // The pivot must lie ON a long wall (within wall thickness ≈ 12 units).
    let pivotHost: LongWall | null = null
    let pivotGap = Infinity
    for (const w of longWalls) {
      const wPerp = w.axis === 'h' ? w.start[1] : w.start[0]
      const wLo = w.axis === 'h' ? w.start[0] : w.start[1]
      const wHi = w.axis === 'h' ? w.end[0] : w.end[1]
      const pivotPerp = w.axis === 'h' ? pivot[1] : pivot[0]
      const pivotParam = w.axis === 'h' ? pivot[0] : pivot[1]
      const gap = Math.abs(wPerp - pivotPerp)
      if (gap > 12) continue
      if (pivotParam < wLo - 5 || pivotParam > wHi + 5) continue
      if (gap < pivotGap) {
        pivotGap = gap
        pivotHost = w
      }
    }
    if (!pivotHost) continue

    // Door extends from pivot toward the bbox far end (parallel to wall).
    const isHostHoriz = pivotHost.axis === 'h'
    const doorWidth = isHostHoriz ? dw : dh
    const wLo = isHostHoriz ? pivotHost.start[0] : pivotHost.start[1]
    const wHi = isHostHoriz ? pivotHost.end[0] : pivotHost.end[1]
    const wallLen = wHi - wLo
    if (wallLen <= 0) continue

    const pivotParam = isHostHoriz ? pivot[0] : pivot[1]
    const bboxLo = isHostHoriz ? pminX : pminY
    const bboxHi = isHostHoriz ? pmaxX : pmaxY
    // Pivot sits at one end of the bbox in the wall-parallel direction;
    // the door spans from pivot toward the far bbox end.
    const distToLo = Math.abs(pivotParam - bboxLo)
    const distToHi = Math.abs(pivotParam - bboxHi)
    let doorStart: number, doorEnd: number
    if (distToLo < distToHi) {
      doorStart = pivotParam
      doorEnd = pivotParam + doorWidth
    } else {
      doorStart = pivotParam - doorWidth
      doorEnd = pivotParam
    }
    const startFrac = Math.max(0, Math.min(1, (doorStart - wLo) / wallLen))
    const endFrac = Math.max(0, Math.min(1, (doorEnd - wLo) / wallLen))
    if (endFrac - startFrac < 0.02) continue

    const centre = (startFrac + endFrac) / 2
    const dup = pivotHost.attachments.some(
      (a) =>
        a.type === 'door' &&
        centre >= a.startFraction - 0.05 &&
        centre <= a.endFraction + 0.05,
    )
    if (dup) continue
    pivotHost.attachments.push({
      type: 'door',
      startFraction: Math.min(startFrac, endFrac),
      endFraction: Math.max(startFrac, endFrac),
    })
    doorsAttached++
  }

  // \u2500\u2500 Append floor outlines as ZONE-ONLY rooms \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // Just the polygon fill + label; the actual walls all come from the merged
  // long-wall list below (so windows/doors attached to exterior walls are
  // preserved instead of being orphaned).
  floorOutlines.forEach((outline, i) => {
    rooms.push({
      name: floorOutlines.length === 1 ? 'Floor Outline' : `Floor ${i + 1} Outline`,
      width: 1,
      depth: 1,
      centerX: 0,
      centerZ: 0,
      color: '#e5e7eb',
      polygon: outline.pts.map(([x, y]) => toScene(x, y)),
      zoneOnly: true,
    })
  })

  // \u2500\u2500 Emit every long wall (with its door/window attachments) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let wallEmitted = 0
  for (const w of longWalls) {
    const start = w.start
    const end = w.end

    // Wall length in metres (for placing attachments)
    const wallLenSvg = Math.hypot(end[0] - start[0], end[1] - start[1])
    const wallLenM = wallLenSvg * SCALE

    // Build attachments in scene units: position = distance from start (m),
    // width = arc width on the wall (m).
    const sceneAttachments = w.attachments.map((a) => {
      const centerFrac = (a.startFraction + a.endFraction) / 2
      const widthFrac = a.endFraction - a.startFraction
      return {
        type: a.type,
        position: centerFrac * wallLenM,
        width: Math.max(0.6, widthFrac * wallLenM), // min sensible width
        height: a.type === 'door' ? 2.1 : 1.2,
      }
    })

    rooms.push({
      name: `Wall ${++wallEmitted}`,
      width: 1,
      depth: 1,
      centerX: 0,
      centerZ: 0,
      color: '#9ca3af',
      polygon: [toScene(start[0], start[1]), toScene(end[0], end[1])],
      outlineOnly: true,
      attachments: sceneAttachments.length > 0 ? sceneAttachments : undefined,
    })
  }

  console.log('[parseSvgToRooms]', {
    polygons: allPolys.length,
    outlines: floorOutlines.length,
    labels: labels.length,
    polygonRooms: rooms.filter((r) => r.zoneOnly).length,
    classifiedSegs: classified.length,
    longWalls: longWalls.length,
    wallsEmitted: wallEmitted,
    windowsAttached,
    doorCandidates,
    doorsAttached,
  })

  return { ok: true, rooms }
}
