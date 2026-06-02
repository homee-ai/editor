import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { NextResponse } from 'next/server'
import { z } from 'zod'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

const DEFAULT_GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
const DEFAULT_GEMINI_MODEL = 'gemini-2.0-flash'

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
        'Use create_wall for "add a wall" requests. Use extend_wall for "extend/shorten". Use translate_wall for "move wall". Use rotate_wall for "rotate wall". Use fill_gap to connect two wall endpoints. Use trim_wall to clip a wall at another wall\'s intersection.',
        'To resize or extend a wall, prefer extend_wall over update_node for natural language requests.',
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
