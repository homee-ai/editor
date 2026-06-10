import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { NextResponse } from 'next/server'
import { z } from 'zod'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

const DEFAULT_GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
const DEFAULT_GEMINI_MODEL = 'gemini-2.0-flash'
const REASONING_EFFORT_VALUES = new Set(['minimal', 'low', 'medium', 'high'])

const chatMessageSchema = z.object({
  role: z.enum(['system', 'user', 'assistant']),
  content: z.string().min(1).max(20_000),
})

const chatRequestSchema = z.object({
  messages: z.array(chatMessageSchema).min(1).max(50),
  sceneContext: z
    .object({
      selectedNodes: z.array(z.record(z.string(), z.unknown())).max(20).default([]),
      nodes: z.array(z.record(z.string(), z.unknown())).max(200).default([]),
      walls: z.array(z.record(z.string(), z.unknown())).max(200).default([]),
      summary: z.string().max(4000).optional(),
    })
    .optional(),
})

type ChatMessage = z.infer<typeof chatMessageSchema>

const roomActionSchema = z.object({
  type: z.literal('create_room'),
  name: z.string().min(1).max(80).default('Room'),
  width: z.coerce.number().min(1).max(30).default(5),
  depth: z.coerce.number().min(1).max(30).default(5),
  x: z.coerce.number().min(-100).max(100).optional(),
  z: z.coerce.number().min(-100).max(100).optional(),
  color: z
    .string()
    .regex(/^#[0-9a-f]{6}$/i)
    .optional(),
  doors: z.coerce.number().int().min(0).max(8).default(1),
  windows: z.coerce.number().int().min(0).max(16).default(0),
  items: z
    .array(
      z.object({
        assetId: z
          .enum([
            'dining-table-mo9ms5yh',
            'dining-table',
            'livingroom-chair',
            'lounge-chair',
            'sofa',
            'stool',
            'small-indoor-plant',
            'indoor-plant',
            'single-bed',
            'double-bed',
            'bathroom-sink',
            'toilet',
            'kitchen-counter',
            'kitchen',
          ])
          .default('livingroom-chair'),
        count: z.coerce.number().int().min(1).max(80).default(1),
      }),
    )
    .max(20)
    .default([]),
})

const numericTuple2Schema = z.tuple([z.coerce.number(), z.coerce.number()])
const numericTuple3Schema = z.tuple([z.coerce.number(), z.coerce.number(), z.coerce.number()])

const nodePatchSchema = z
  .object({
    name: z.string().min(1).max(120).optional(),
    visible: z.boolean().optional(),
    position: numericTuple3Schema.optional(),
    rotation: numericTuple3Schema.optional(),
    scale: numericTuple3Schema.optional(),
    width: z.coerce.number().min(0.05).max(30).optional(),
    height: z.coerce.number().min(0.05).max(30).optional(),
    thickness: z.coerce.number().min(0.02).max(2).optional(),
    color: z
      .string()
      .regex(/^#[0-9a-f]{6}$/i)
      .optional(),
    start: numericTuple2Schema.optional(),
    end: numericTuple2Schema.optional(),
  })
  .strict()

const updateNodeActionSchema = z.object({
  type: z.literal('update_node'),
  nodeId: z.string().min(1),
  patch: nodePatchSchema,
})

const moveOpeningActionSchema = z.object({
  type: z.literal('move_opening'),
  nodeId: z.string().min(1),
  wallId: z.string().min(1).optional(),
  t: z.coerce.number().min(0).max(1).default(0.5),
  y: z.coerce.number().min(0).max(20).optional(),
})

const deleteNodesActionSchema = z.object({
  type: z.literal('delete_nodes'),
  nodeIds: z.array(z.string().min(1)).min(1).max(80),
})

const createWallActionSchema = z.object({
  type: z.literal('create_wall'),
  name: z.string().min(1).max(80).default('Wall'),
  start: numericTuple2Schema,
  end: numericTuple2Schema,
  thickness: z.coerce.number().min(0.05).max(2).default(0.1),
  height: z.coerce.number().min(0.5).max(10).default(2.8),
})

const extendWallActionSchema = z.object({
  type: z.literal('extend_wall'),
  nodeId: z.string().min(1),
  endpoint: z.enum(['start', 'end']),
  by: z.coerce.number().min(-20).max(20),
})

const duplicateNodeActionSchema = z.object({
  type: z.literal('duplicate_node'),
  nodeId: z.string().min(1),
  offset: numericTuple2Schema.default([1, 0]),
})

const translateWallActionSchema = z.object({
  type: z.literal('translate_wall'),
  nodeId: z.string().min(1),
  delta: numericTuple2Schema,
})

const rotateWallActionSchema = z.object({
  type: z.literal('rotate_wall'),
  nodeId: z.string().min(1),
  angleDeg: z.coerce.number().min(-360).max(360),
  pivot: z.enum(['start', 'end', 'center']).default('center'),
})

const fillGapActionSchema = z.object({
  type: z.literal('fill_gap'),
  wallId1: z.string().min(1),
  wallId2: z.string().min(1),
})

const trimWallActionSchema = z.object({
  type: z.literal('trim_wall'),
  nodeId: z.string().min(1),
  trimToWallId: z.string().min(1),
})

const actionSchema = z.discriminatedUnion('type', [
  roomActionSchema,
  updateNodeActionSchema,
  moveOpeningActionSchema,
  deleteNodesActionSchema,
  createWallActionSchema,
  extendWallActionSchema,
  duplicateNodeActionSchema,
  translateWallActionSchema,
  rotateWallActionSchema,
  fillGapActionSchema,
  trimWallActionSchema,
])

const plannerResponseSchema = z.object({
  reply: z.string().min(1).max(20_000),
  actions: z.array(actionSchema).max(80).default([]),
})

export async function POST(request: Request) {
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json(
      { error: 'invalid_request', message: 'Body must be JSON.' },
      { status: 400 },
    )
  }

  const parsed = chatRequestSchema.safeParse(body)
  if (!parsed.success) {
    return NextResponse.json(
      { error: 'invalid_request', details: parsed.error.issues },
      { status: 400 },
    )
  }

  const provider = getLocalEnv('EDITOR_LLM_PROVIDER') ?? 'gemini'
  const baseUrl =
    getLocalEnv('EDITOR_LLM_BASE_URL') ??
    (provider === 'gemini' ? DEFAULT_GEMINI_BASE_URL : 'http://localhost:11434/v1')
  const model =
    getLocalEnv('EDITOR_LLM_MODEL') ?? (provider === 'gemini' ? DEFAULT_GEMINI_MODEL : 'llama3')
  const apiKey = resolveApiKey(
    getLocalEnv('EDITOR_LLM_API_KEY') ??
      getLocalEnv('GEMINI_API_KEY') ??
      getLocalEnv('GOOGLE_API_KEY') ??
      '',
  )

  if (!apiKey && !isLocalBaseUrl(baseUrl)) {
    return NextResponse.json(
      {
        error: 'missing_api_key',
        message: 'Set EDITOR_LLM_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY in .env.local.',
      },
      { status: 400 },
    )
  }

  const messages = withSystemPrompt(parsed.data.messages, parsed.data.sceneContext)
  const reasoningEffort = getReasoningEffort()
  const latestUserMessage = [...parsed.data.messages]
    .reverse()
    .find((message) => message.role === 'user')?.content

  try {
    const upstream = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      },
      body: JSON.stringify({
        model,
        messages,
        ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
        temperature: Number(getLocalEnv('EDITOR_LLM_TEMPERATURE') ?? 0.2),
        max_tokens: Math.max(Number(getLocalEnv('EDITOR_LLM_MAX_TOKENS') ?? 4096), 4096),
        stream: false,
      }),
    })

    const text = await upstream.text()
    let data: unknown
    try {
      data = text ? JSON.parse(text) : null
    } catch {
      data = text
    }

    if (!upstream.ok) {
      console.error('[ai-chat] upstream failure', upstream.status, data)
      const errObj = Array.isArray(data) ? data[0] : data
      const upstreamMessage =
        (errObj && typeof errObj === 'object' && 'error' in errObj && (errObj as any).error?.message) ||
        (typeof data === 'string' ? data.slice(0, 500) : null)
      return NextResponse.json(
        {
          error: 'llm_request_failed',
          status: upstream.status,
          message: upstreamMessage
            ? `LLM ${upstream.status}: ${upstreamMessage}`
            : `LLM 回傳 ${upstream.status}`,
          details: data,
        },
        { status: 502 },
      )
    }

    const content = extractReply(data)
    if (!content) {
      return NextResponse.json(
        { error: 'llm_response_unparseable', details: data },
        { status: 502 },
      )
    }

    const jsonText = extractJsonObject(content)
    if (!jsonText) {
      return NextResponse.json({ reply: content, actions: [], model, provider })
    }

    try {
      const planned = plannerResponseSchema.parse(JSON.parse(jsonText))
      const actions = planned.actions.filter(
        (action) => action.type !== 'create_room' || allowsCreateRooms(latestUserMessage),
      )
      if (actions.length !== planned.actions.length && actions.length === 0) {
        return NextResponse.json({
          ...planned,
          reply:
            '目前這句看起來是編輯既有物件，我沒有新增房間。請先選取要修改的物件，再描述要怎麼調整。',
          actions: [],
          model,
          provider,
        })
      }
      return NextResponse.json({ ...planned, actions, model, provider })
    } catch {
      return NextResponse.json({ reply: content, actions: [], model, provider })
    }
  } catch (error) {
    return NextResponse.json(
      {
        error: 'llm_request_failed',
        message: error instanceof Error ? error.message : 'Unexpected LLM request failure.',
      },
      { status: 502 },
    )
  }
}

function extractJsonObject(text: string): string | null {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fenced?.[1]) return fenced[1].trim()

  const start = text.indexOf('{')
  const end = text.lastIndexOf('}')
  if (start === -1 || end <= start) return null

  return text.slice(start, end + 1)
}

function extractReply(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null
  const choices = (data as { choices?: unknown }).choices
  if (!Array.isArray(choices) || choices.length === 0) return null
  const first = choices[0]
  if (!first || typeof first !== 'object') return null
  const message = (first as { message?: unknown }).message
  if (!message || typeof message !== 'object') return null
  const content = (message as { content?: unknown }).content

  return typeof content === 'string' && content.trim() ? content : null
}

function withSystemPrompt(
  messages: ChatMessage[],
  sceneContext?: z.infer<typeof chatRequestSchema>['sceneContext'],
): ChatMessage[] {
  if (messages.some((message) => message.role === 'system')) return messages

  return [
    {
      role: 'system',
      content: [
        'You are Pascal Editor AI, a concise architectural design assistant inside a local 3D building editor.',
        'Reply in the same language as the user.',
        'Return JSON only.',
        'Only use create_room when the user explicitly asks to create, add, generate, build, design, plan, or make new rooms/areas.',
        'For edits to existing things such as move, resize, widen, delete, reduce, recolor, adjust, or change, use update_node, move_opening, or delete_nodes. Do not use create_room for edits.',
        'Prefer selectedNodes as targets. If no selected node or exact target is available, return actions: [] and ask the user to select the target.',
        'When the user asks to create or design a room, return create_room actions.',
        'You may return multiple create_room actions to form a full floor plan or venue.',
        'Keep reply under 280 characters. Do not include markdown in reply.',
        'Supported action schema:',
        '{"reply":"short explanation","actions":[{"type":"create_room","name":"Living Room","width":5,"depth":5,"x":0,"z":0,"color":"#dbeafe","doors":1,"windows":2,"items":[{"assetId":"sofa","count":1}]},{"type":"update_node","nodeId":"item_x","patch":{"position":[1,0,2],"rotation":[0,1.57,0],"width":1.8,"height":1.2,"color":"#dbeafe","name":"New name"}},{"type":"update_node","nodeId":"wall_x","patch":{"start":[-5,0],"end":[5,0]}},{"type":"move_opening","nodeId":"door_x","wallId":"wall_x","t":0.5},{"type":"delete_nodes","nodeIds":["item_x"]}]}',
        'Wall actions: {"type":"create_wall","name":"Wall","start":[-5,0],"end":[5,0],"thickness":0.1,"height":2.8} | {"type":"extend_wall","nodeId":"wall_x","endpoint":"end","by":2} (by>0=extend,by<0=shorten) | {"type":"duplicate_node","nodeId":"wall_x","offset":[0,3]} | {"type":"translate_wall","nodeId":"wall_x","delta":[1,0]} | {"type":"rotate_wall","nodeId":"wall_x","angleDeg":90,"pivot":"center"} | {"type":"fill_gap","wallId1":"wall_a","wallId2":"wall_b"} | {"type":"trim_wall","nodeId":"wall_x","trimToWallId":"wall_y"}',
        'Verb-to-action mapping — CRITICAL, do NOT confuse these:',
        '  "延伸 / 伸長 / 加長 / 拉長 / extend / lengthen" → extend_wall (one endpoint moves along the wall\'s own axis; the other endpoint stays fixed; this is NOT a translation)',
        '  "縮短 / 截短 / shorten"               → extend_wall with negative `by`',
        '  "移動 / 平移 / 搬 / 挪 / move / shift / translate" → translate_wall (whole wall shifts; BOTH endpoints move by the same delta)',
        '  "旋轉 / 轉 / rotate / turn"           → rotate_wall',
        '  "複製 / 拷貝 / duplicate / copy"      → duplicate_node',
        '  "新增牆 / 加一道牆 / 加牆 / add wall / create wall / new wall" → create_wall',
        '  "補齊 / 連接 / 接起來 / fill gap / connect" → fill_gap (requires two wall ids)',
        '  "修剪 / 裁切 / 切掉 / trim / clip"    → trim_wall',
        '⚠ NEVER use translate_wall when the user says "延伸/伸長/extend". Extend changes ONE endpoint; translate moves BOTH. They produce visually different results.',
        'Scene coordinate convention (the 2D floorplan view rotates the raw scene by 90°, so compass directions do NOT map directly to raw axes):',
        '  Compass North = scene -X (top of screen)',
        '  Compass South = scene +X (bottom of screen)',
        '  Compass East  = scene -Z (right of screen)',
        '  Compass West  = scene +Z (left of screen)',
        'Cardinal-direction vectors (use EXACTLY when user names a direction): North = [-d, 0], South = [+d, 0], East = [0, -d], West = [0, +d].',
        'A wall whose endpoints vary in Z runs east-west on screen. A wall whose endpoints vary in X runs north-south on screen.',
        'Wall context fields (READ THESE — do not recompute from start/end):',
        '  `orientation`: "east-west" | "north-south" | "diagonal" — tells you which axis the wall runs along.',
        '  `westEndpoint` / `eastEndpoint` / `northEndpoint` / `southEndpoint`: each is the literal string "start" or "end". Use this VALUE directly as the extend_wall `endpoint` parameter.',
        'For "延伸" (extend) / "縮短" (shorten) with a cardinal direction:',
        '  Semantic rule (IMPORTANT — applies to BOTH extend and shorten):',
        '    "向 [direction]" indicates the direction the wall is GROWING (extend) or RETRACTING (shorten). The endpoint that MOVES is always the one on the named side, but it moves in opposite directions for extend vs shorten.',
        '    "向西延伸 X" → the west endpoint moves further west (wall grows westward). endpoint = wall.westEndpoint, by = +X.',
        '    "向西縮短 X" → the wall retracts westward, i.e. the EAST endpoint moves westward (wall shrinks from the east side, remaining wall positioned more to the west). endpoint = wall.eastEndpoint, by = -X.',
        '    Same pattern for east/north/south: extend moves the same-side endpoint outward; shorten moves the OPPOSITE-side endpoint inward.',
        '  Case A — direction matches wall axis (user says west/east AND wall.orientation="east-west", OR user says north/south AND wall.orientation="north-south"):',
        '    Emit extend_wall using the rules above.',
        '  Case B — direction is perpendicular to wall axis (user says west/east on a north-south wall, or user says north/south on an east-west wall):',
        '    For "延伸": user means "push the wall outward in that direction" — emit translate_wall instead. Set delta = [0, +d] for west, [0, -d] for east, [-d, 0] for north, [+d, 0] for south. Mention in reply that you translated.',
        '    For "縮短": user means "pull the wall inward in that direction" — emit translate_wall with the OPPOSITE delta (delta = [0, -d] for "向西縮短", etc). Mention in reply.',
        '  Case C — wall is diagonal: reply actions:[] and ask for clarification.',
        'For create_wall with a cardinal direction (e.g. "在北邊新增一道牆"), place the wall in the requested region of the floor plan using current wall bounding box as reference.',
        'Prefer extend_wall over update_node for natural-language extend/shorten requests.',
        'Wall thickness: use update_node with patch.thickness for "厚度/變厚/變薄/thickness". Range 0.02–2 meters. "增加 X 厚度" means add X to current; the current value is in wall.thickness so compute new = current + X. Default thickness is 0.1 if unset.',
        'For doors/windows on walls, position[0] is distance from the wall start. Prefer move_opening with t=0.5 for "middle/center of wall".',
        'Use meters. width/depth must be between 1 and 30. x/z are optional room-center coordinates.',
        'Supported assetId values: dining-table-mo9ms5yh, dining-table, livingroom-chair, lounge-chair, sofa, stool, small-indoor-plant, indoor-plant, single-bed, double-bed, bathroom-sink, toilet, kitchen-counter, kitchen.',
        'For exhibition halls, salons, restaurants, classrooms, or offices, split the program into multiple rooms/areas and add repeated furniture counts.',
        'Use coordinates to lay rooms beside each other with about 1 meter spacing. Keep total dimensions reasonable.',
        'If the user is only asking a question, return {"reply":"...","actions":[]}.',
        sceneContext?.summary ? `Current scene summary: ${sceneContext.summary}` : '',
        sceneContext?.selectedNodes?.length
          ? `Selected nodes: ${JSON.stringify(sceneContext.selectedNodes)}`
          : 'Selected nodes: none.',
        sceneContext?.walls?.length ? `Available walls: ${JSON.stringify(sceneContext.walls)}` : '',
        sceneContext?.nodes?.length ? `Scene nodes: ${JSON.stringify(sceneContext.nodes)}` : '',
      ].join('\n'),
    },
    ...messages,
  ]
}

function allowsCreateRooms(message?: string): boolean {
  if (!message) return false
  const normalized = message.toLowerCase()
  const createTerms = [
    'create',
    'add',
    'generate',
    'build',
    'design',
    'plan',
    'make',
    '新增',
    '加入',
    '增加',
    '建立',
    '建',
    '生成',
    '產生',
    '設計',
    '規劃',
  ]
  const editTerms = [
    'move',
    'resize',
    'widen',
    'delete',
    'remove',
    'reduce',
    'recolor',
    'adjust',
    'change',
    '移',
    '搬',
    '調整',
    '修改',
    '改',
    '加寬',
    '變寬',
    '刪',
    '移除',
    '減少',
  ]

  return (
    createTerms.some((term) => normalized.includes(term)) &&
    !editTerms.some((term) => normalized.includes(term))
  )
}

function getLocalEnv(name: string): string | undefined {
  return process.env[name] || readEnvFileValue(name)
}

function readEnvFileValue(name: string): string | undefined {
  for (const file of envFileCandidates()) {
    if (!existsSync(file)) continue

    const value = parseEnvValue(readFileSync(file, 'utf8'), name)
    if (value) return value
  }

  return undefined
}

function envFileCandidates(): string[] {
  const cwd = process.cwd()
  return [
    join(cwd, '.env.local'),
    resolve(cwd, '../../.env.local'),
    join(cwd, '.env'),
    resolve(cwd, '../../.env'),
  ]
}

function parseEnvValue(contents: string, name: string): string | undefined {
  for (const line of contents.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue

    const eq = trimmed.indexOf('=')
    if (eq <= 0) continue
    if (trimmed.slice(0, eq).trim() !== name) continue

    return unquoteEnvValue(trimmed.slice(eq + 1).trim())
  }

  return undefined
}

function unquoteEnvValue(value: string): string {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1)
  }

  return value
}

function resolveApiKey(raw: string): string {
  const value = raw.trim()
  if (!value) return ''

  if (value.startsWith('file:')) {
    try {
      return readFileSync(expandPath(value.slice(5).trim()), 'utf8').trim()
    } catch {
      return ''
    }
  }

  return value
}

function getReasoningEffort(): string | undefined {
  const value = getLocalEnv('EDITOR_LLM_REASONING_EFFORT')?.trim().toLowerCase()
  if (!value) return undefined
  return REASONING_EFFORT_VALUES.has(value) ? value : undefined
}

function expandPath(value: string): string {
  return value
    .replace(/^~(?=$|\/|\\)/, homedir())
    .replace(/\$(\w+)|%(\w+)%/g, (_match, posixName: string, windowsName: string) => {
      const name = posixName || windowsName
      return process.env[name] ?? ''
    })
}

function isLocalBaseUrl(value: string): boolean {
  try {
    const url = new URL(value)
    return url.hostname === 'localhost' || url.hostname === '127.0.0.1' || url.hostname === '::1'
  } catch {
    return false
  }
}
