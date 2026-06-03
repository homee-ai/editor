'use client'

import { DoorNode, useScene, WindowNode } from '@pascal-app/core'
import { applySceneGraphToEditor, CATALOG_ITEMS } from '@pascal-app/editor'
import { convertIfcToPascal } from '@pascal-app/ifc-converter'
import { useViewer } from '@pascal-app/viewer'
import { AlertCircle, Loader2, Paperclip, Plus, Send, Sparkles } from 'lucide-react'
import type { FormEvent } from 'react'
import { useEffect, useRef, useState } from 'react'
import { parseDxfToRooms, parseSvgToRooms } from '@/lib/dxf-to-rooms'
import { cn } from '@/lib/utils'

type ChatRole = 'user' | 'assistant'

type CreateRoomAction = {
  type: 'create_room'
  name: string
  width: number
  depth: number
  x?: number
  z?: number
  color?: string
  doors?: number
  windows?: number
  /**
   * Optional explicit polygon outline ([x, z] vertex pairs in scene metres,
   * closed — last vertex does not repeat first). When provided, width/depth/
   * x/z are ignored and walls are built along these edges.
   */
  polygon?: Array<[number, number]>
  /** Outline only: build walls but skip the zone fill, doors, windows, items. */
  outlineOnly?: boolean
  /** Zone only: build the zone polygon (for labelling / colour) but no walls. */
  zoneOnly?: boolean
  /**
   * For 2-point single-wall create_room calls: doors/windows to mount on
   * the created wall. `position` is metres from wall start.
   */
  attachments?: Array<{
    type: 'door' | 'window'
    position: number
    width: number
    height: number
  }>
  items?: Array<{
    assetId: string
    count: number
  }>
}

type UpdateNodeAction = {
  type: 'update_node'
  nodeId: string
  patch: {
    name?: string
    visible?: boolean
    position?: [number, number, number]
    rotation?: [number, number, number]
    scale?: [number, number, number]
    width?: number
    height?: number
    thickness?: number
    color?: string
    start?: [number, number]
    end?: [number, number]
  }
}

type MoveOpeningAction = {
  type: 'move_opening'
  nodeId: string
  wallId?: string
  t?: number
  y?: number
}

type DeleteNodesAction = {
  type: 'delete_nodes'
  nodeIds: string[]
}

type CreateWallAction = {
  type: 'create_wall'
  name: string
  start: [number, number]
  end: [number, number]
  thickness: number
  height: number
}

type ExtendWallAction = {
  type: 'extend_wall'
  nodeId: string
  endpoint: 'start' | 'end'
  by: number
}

type DuplicateNodeAction = {
  type: 'duplicate_node'
  nodeId: string
  offset: [number, number]
}

type TranslateWallAction = {
  type: 'translate_wall'
  nodeId: string
  delta: [number, number]
}

type RotateWallAction = {
  type: 'rotate_wall'
  nodeId: string
  angleDeg: number
  pivot: 'start' | 'end' | 'center'
}

type FillGapAction = {
  type: 'fill_gap'
  wallId1: string
  wallId2: string
}

type TrimWallAction = {
  type: 'trim_wall'
  nodeId: string
  trimToWallId: string
}

type SceneAction =
  | CreateRoomAction
  | UpdateNodeAction
  | MoveOpeningAction
  | DeleteNodesAction
  | CreateWallAction
  | ExtendWallAction
  | DuplicateNodeAction
  | TranslateWallAction
  | RotateWallAction
  | FillGapAction
  | TrimWallAction

type ChatMessage = {
  id: string
  role: ChatRole
  content: string
  actions?: SceneAction[]
  applied?: boolean
}

type ChatResponse = {
  reply?: string
  actions?: SceneAction[]
  error?: string
  message?: string
}

const SUGGESTIONS = [
  'Add a master bedroom',
  'Design a kitchen',
  'Create a living room',
  'Build a bathroom',
]

function createMessage(role: ChatRole, content: string): ChatMessage {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    role,
    content,
  }
}

type SceneSnapshot = { nodes: Record<string, unknown>; rootNodeIds: string[] }

const MAX_HISTORY = 10

export function AiChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [history, setHistory] = useState<SceneSnapshot[]>([])
  const [dxfMode, setDxfMode] = useState(false)
  const [dxfText, setDxfText] = useState('')
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const dxfTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const ifcFileInputRef = useRef<HTMLInputElement | null>(null)
  const [isImportingIfc, setIsImportingIfc] = useState(false)
  const [ifcProgress, setIfcProgress] = useState('')

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  })

  const handleIfcUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    event.target.value = ''

    setIsImportingIfc(true)
    setIfcProgress('讀取檔案中...')
    setError(null)

    try {
      const buffer = await file.arrayBuffer()
      const data = new Uint8Array(buffer)
      const scene = await convertIfcToPascal(data, (message) => {
        setIfcProgress(message)
      })

      const { nodes, rootNodeIds } = useScene.getState()
      const snapshot: SceneSnapshot = {
        nodes: structuredClone(nodes) as Record<string, unknown>,
        rootNodeIds: [...rootNodeIds],
      }
      applySceneGraphToEditor(scene)
      setHistory((current) => [...current.slice(-MAX_HISTORY + 1), snapshot])

      const nodeCount = Object.keys(scene.nodes).length
      setMessages((current) => [
        ...current,
        createMessage('user', `📦 匯入 IFC：${file.name}`),
        createMessage(
          'assistant',
          `IFC 匯入完成！共載入 ${nodeCount} 個節點。\n\n你可以用自然語言描述想要的修改，例如：「把客廳的牆壁顏色改成白色」或「刪除所有門」。`,
        ),
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'IFC 匯入失敗')
    } finally {
      setIsImportingIfc(false)
      setIfcProgress('')
    }
  }

  const handleDxfUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    // Reset so the same file can be re-uploaded
    event.target.value = ''

    const reader = new FileReader()
    reader.onload = () => {
      const isSvgFile = file.name.toLowerCase().endsWith('.svg')
      applyDxfText(reader.result as string, `📐 上傳 ${isSvgFile ? 'SVG' : 'DXF'}：${file.name}`)
    }
    reader.onerror = () => setError('無法讀取檔案。')
    reader.readAsText(file)
  }

  const applyDxfText = (rawText: string, label: string) => {
    const trimmed = rawText.trimStart()
    const isSvg =
      trimmed.startsWith('<svg') ||
      trimmed.startsWith('<SVG') ||
      (trimmed.startsWith('<?xml') && trimmed.includes('<svg'))
    const result = isSvg ? parseSvgToRooms(rawText) : parseDxfToRooms(rawText)
    if (!result.ok) {
      setError(`解析失敗：${result.message}`)
      return false
    }
    const { rooms } = result
    const actions: CreateRoomAction[] = rooms.map((room) => ({
      type: 'create_room',
      name: room.name,
      width: room.width,
      depth: room.depth,
      x: room.centerX,
      z: room.centerZ,
      color: room.color,
      doors: room.outlineOnly || room.zoneOnly ? 0 : 1,
      windows: 0,
      polygon: room.polygon,
      outlineOnly: room.outlineOnly,
      zoneOnly: room.zoneOnly,
      attachments: room.attachments,
    }))
    const isFloorOutline = (r: { name: string }) => /Floor.*Outline|Floor Outline/i.test(r.name)
    const labeledRooms = rooms.filter((r) => !r.outlineOnly && !isFloorOutline(r))
    const outlineRooms = rooms.filter((r) => isFloorOutline(r))
    const wallSegments = rooms.filter((r) => r.outlineOnly && r.polygon?.length === 2)
    const totalDoors = wallSegments.reduce(
      (s, w) => s + (w.attachments?.filter((a) => a.type === 'door').length ?? 0),
      0,
    )
    const totalWindows = wallSegments.reduce(
      (s, w) => s + (w.attachments?.filter((a) => a.type === 'window').length ?? 0),
      0,
    )
    const summaryLines = labeledRooms.map((r) => `• ${r.name}（${r.width} × ${r.depth} m）`)
    const extras: string[] = []
    if (outlineRooms.length > 0) extras.push(`${outlineRooms.length} 個樓層外輪廓`)
    if (wallSegments.length > 0) extras.push(`${wallSegments.length} 道牆`)
    if (totalWindows > 0) extras.push(`${totalWindows} 扇窗`)
    if (totalDoors > 0) extras.push(`${totalDoors} 扇門`)
    const summary = summaryLines.length > 0
      ? `偵測到 ${labeledRooms.length} 個有實際形狀的房間：\n${summaryLines.join('\n')}\n• 含 ${extras.join('、')}`
      : `從 SVG 抽出 ${extras.join('、')}`
    setMessages((current) => [
      ...current,
      createMessage('user', label),
      {
        ...createMessage('assistant', `${summary}\n\n點擊 Apply 加入場景。`),
        actions,
      },
    ])
    setError(null)
    return true
  }

  const handleDxfBuild = () => {
    const trimmed = dxfText.trim()
    if (!trimmed) return
    const ok = applyDxfText(trimmed, '📐 貼上 DXF 內容')
    if (ok) {
      setDxfMode(false)
      setDxfText('')
    }
  }

  const handleUndo = () => {
    if (history.length === 0) return
    const previous = history[history.length - 1]!
    applySceneGraphToEditor(previous)
    setHistory((current) => current.slice(0, -1))
    return true
  }

  const sendMessage = async (content: string) => {
    const trimmed = content.trim()
    if (!trimmed || isSending) return

    if (/^(復原|undo|撤銷|回復|上一步)/i.test(trimmed)) {
      const didUndo = handleUndo()
      const reply = didUndo
        ? '已復原上一步操作。'
        : '沒有可以復原的操作。'
      setMessages((current) => [
        ...current,
        createMessage('user', trimmed),
        createMessage('assistant', reply),
      ])
      return
    }

    const nextMessages = [...messages, createMessage('user', trimmed)]
    setMessages(nextMessages)
    setInput('')
    setError(null)
    setIsSending(true)

    try {
      const response = await fetch('/api/ai-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: nextMessages.map(({ role, content }) => ({ role, content })),
          sceneContext: buildSceneContext(),
        }),
      })
      const data = (await response.json()) as ChatResponse

      if (!response.ok || !data.reply) {
        throw new Error(data.message ?? data.error ?? `Request failed (${response.status})`)
      }

      setMessages((current) => [
        ...current,
        {
          ...createMessage('assistant', data.reply ?? ''),
          actions: data.actions?.length ? data.actions : undefined,
        },
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'AI request failed')
    } finally {
      setIsSending(false)
      textareaRef.current?.focus()
    }
  }

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    void sendMessage(input)
  }

  const handleApplyActions = (messageId: string, actions: SceneAction[]) => {
    try {
      const { nodes, rootNodeIds } = useScene.getState()
      const snapshot: SceneSnapshot = {
        nodes: structuredClone(nodes) as Record<string, unknown>,
        rootNodeIds: [...rootNodeIds],
      }
      const scene = buildSceneWithActions(actions)
      applySceneGraphToEditor(scene)
      setHistory((current) => [...current.slice(-MAX_HISTORY + 1), snapshot])
      setMessages((current) =>
        current.map((message) =>
          message.id === messageId ? { ...message, applied: true } : message,
        ),
      )
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not apply scene changes')
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-sidebar text-sidebar-foreground">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-3 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col justify-center gap-5">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="flex h-11 w-11 items-center justify-center rounded-full bg-accent text-foreground">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <h2 className="font-semibold text-sm">Ask AI</h2>
                <p className="mt-1 text-muted-foreground text-xs">Gemini local bridge</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {messages.map((message) => (
              <div
                className={cn(
                  'rounded-lg border border-border/60 px-3 py-2 text-sm leading-6',
                  message.role === 'user'
                    ? 'bg-accent text-foreground'
                    : 'bg-background/60 text-sidebar-foreground',
                )}
                key={message.id}
              >
                <div className="mb-1 font-medium text-muted-foreground text-xs">
                  {message.role === 'user' ? 'You' : 'AI'}
                </div>
                <div className="whitespace-pre-wrap break-words">{message.content}</div>
                {message.role === 'assistant' && message.actions && message.actions.length > 0 && (
                  <button
                    className="mt-3 w-full rounded-md border border-border/70 bg-accent px-3 py-2 font-medium text-xs text-foreground transition-colors hover:bg-accent/80 disabled:cursor-default disabled:opacity-60"
                    disabled={message.applied}
                    onClick={() => handleApplyActions(message.id, message.actions ?? [])}
                    type="button"
                  >
                    {message.applied
                      ? 'Applied'
                      : `Apply ${message.actions.length} change${
                          message.actions.length === 1 ? '' : 's'
                        }`}
                  </button>
                )}
              </div>
            ))}
            {isSending && (
              <div className="flex items-center gap-2 rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-muted-foreground text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                Thinking
              </div>
            )}
          </div>
        )}
      </div>

      <div className="border-border/60 border-t p-3">
        {history.length > 0 && (
          <button
            className="mb-3 w-full rounded-md border border-border/60 px-2 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            onClick={handleUndo}
            type="button"
          >
            ↩ Undo last change ({history.length})
          </button>
        )}
        {!dxfMode && (
          <div className="mb-3">
            <div className="mb-2 text-muted-foreground text-xs">Suggestions</div>
            <div className="grid grid-cols-2 gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  className="truncate rounded-md border border-border/60 px-2 py-1.5 text-left text-xs text-sidebar-foreground transition-colors hover:bg-accent"
                  disabled={isSending}
                  key={suggestion}
                  onClick={() => void sendMessage(suggestion)}
                  type="button"
                >
                  {suggestion}
                </button>
              ))}
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <button
                className="flex items-center justify-center gap-1.5 rounded-md border border-dashed border-border/70 px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border hover:bg-accent hover:text-foreground"
                onClick={() => fileInputRef.current?.click()}
                type="button"
              >
                <Paperclip className="h-3.5 w-3.5 shrink-0" />
                上傳 DXF / SVG
              </button>
              <button
                className="flex items-center justify-center gap-1.5 rounded-md border border-dashed border-border/70 px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isImportingIfc}
                onClick={() => ifcFileInputRef.current?.click()}
                type="button"
              >
                {isImportingIfc ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                    {ifcProgress || '轉換中...'}
                  </>
                ) : (
                  '匯入 IFC'
                )}
              </button>
              <button
                className="col-span-2 flex items-center justify-center gap-1.5 rounded-md border border-dashed border-border/70 px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border hover:bg-accent hover:text-foreground"
                onClick={() => {
                  setDxfMode(true)
                  setError(null)
                  setTimeout(() => dxfTextareaRef.current?.focus(), 50)
                }}
                type="button"
              >
                📐 貼上文字建立
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="mb-3 flex gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-destructive text-xs">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span className="break-words">{error}</span>
          </div>
        )}

        {dxfMode ? (
          <div className="rounded-xl border border-border bg-background/80 p-2 shadow-sm">
            <div className="mb-1 px-2 pt-1 text-muted-foreground text-xs">
              貼上 DXF 或 SVG 文字內容
            </div>
            <textarea
              className="max-h-56 min-h-36 w-full resize-none bg-transparent px-2 py-2 font-mono text-xs outline-none placeholder:text-muted-foreground"
              onChange={(e) => setDxfText(e.target.value)}
              placeholder={'0\nSECTION\n  2\nENTITIES\n  0\nLWPOLYLINE\n...'}
              ref={dxfTextareaRef}
              value={dxfText}
            />
            <div className="flex items-center justify-between pt-1">
              <button
                className="px-2 py-1 text-muted-foreground text-xs transition-colors hover:text-foreground"
                onClick={() => {
                  setDxfMode(false)
                  setDxfText('')
                  setError(null)
                }}
                type="button"
              >
                取消
              </button>
              <button
                className="rounded-lg bg-primary px-4 py-1.5 font-medium text-primary-foreground text-xs transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!dxfText.trim()}
                onClick={handleDxfBuild}
                type="button"
              >
                建立場景
              </button>
            </div>
          </div>
        ) : (
          <form
            className="rounded-xl border border-border bg-background/80 p-2 shadow-sm"
            onSubmit={handleSubmit}
          >
            <textarea
              className="max-h-40 min-h-24 w-full resize-none bg-transparent px-2 py-2 text-sm outline-none placeholder:text-muted-foreground"
              disabled={isSending}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  handleSubmit(event)
                }
              }}
              placeholder="Ask anything..."
              ref={textareaRef}
              value={input}
            />
            <input
              accept=".dxf,.svg"
              className="hidden"
              onChange={handleDxfUpload}
              ref={fileInputRef}
              type="file"
            />
            <input
              accept=".ifc"
              className="hidden"
              onChange={(e) => void handleIfcUpload(e)}
              ref={ifcFileInputRef}
              type="file"
            />
            <div className="flex items-center justify-end">
              <button
                aria-label="Send"
                className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isSending || !input.trim()}
                type="submit"
              >
                {isSending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}

function buildSceneWithActions(actions: SceneAction[]) {
  const { nodes, rootNodeIds } = useScene.getState()
  const nextNodes = structuredClone(nodes) as Record<string, any>
  const nextRootNodeIds = [...rootNodeIds] as string[]
  const levelId = findFirstLevelId(nextNodes)

  if (!levelId) {
    throw new Error('No level found to apply AI changes.')
  }

  const level = nextNodes[levelId]
  const children = Array.isArray(level.children) ? [...level.children] : []
  let fallbackCenterX = findWallMaxX(nextNodes)

  for (const action of actions) {
    if (action.type === 'create_room') {
      // Determine the wall ring. Two cases:
      //   1. Explicit polygon (from SVG import — may be L-shaped / notched)
      //   2. Width × depth rectangle (the default AI-chat path)
      let points: Array<[number, number]>
      let width: number
      let depth: number
      let centerX: number
      let centerZ: number

      if (action.polygon && action.polygon.length >= 2) {
        points = action.polygon.map(([x, z]) => [x, z] as [number, number])
        // Compute bbox + centroid for openings/items placement
        let minX = Infinity,
          maxX = -Infinity,
          minZ = Infinity,
          maxZ = -Infinity
        for (const [x, z] of points) {
          if (x < minX) minX = x
          if (x > maxX) maxX = x
          if (z < minZ) minZ = z
          if (z > maxZ) maxZ = z
        }
        width = maxX - minX
        depth = maxZ - minZ
        centerX = (minX + maxX) / 2
        centerZ = (minZ + maxZ) / 2
      } else {
        width = clamp(action.width, 1, 30)
        depth = clamp(action.depth, 1, 30)
        centerX =
          typeof action.x === 'number'
            ? action.x
            : fallbackCenterX === null
              ? 0
              : fallbackCenterX + width / 2 + 1.5
        centerZ = typeof action.z === 'number' ? action.z : 0
        const halfW = width / 2
        const halfD = depth / 2
        points = [
          [centerX - halfW, centerZ - halfD],
          [centerX + halfW, centerZ - halfD],
          [centerX + halfW, centerZ + halfD],
          [centerX - halfW, centerZ + halfD],
        ]
      }

      // 2-point polygon = a single open wall segment; ≥3 points = closed loop.
      const isSingleWall = points.length === 2
      const wallCount = action.zoneOnly ? 0 : isSingleWall ? 1 : points.length
      const wallIds = Array.from({ length: wallCount }, () => createId('wall'))

      for (let index = 0; index < wallCount; index++) {
        const start = points[index]!
        const end = isSingleWall
          ? points[1]!
          : points[(index + 1) % points.length]!
        const wallId = wallIds[index]!
        nextNodes[wallId] = {
          object: 'node',
          id: wallId,
          type: 'wall',
          name: isSingleWall ? action.name : `${action.name} Wall ${index + 1}`,
          parentId: levelId,
          visible: true,
          metadata: { createdBy: 'ai-chat' },
          children: [],
          start,
          end,
          thickness: 0.1,
          height: 2.8,
          frontSide: 'unknown',
          backSide: 'unknown',
        }
        children.push(wallId)
      }

      // For single-wall imports with attached doors/windows, build the
      // DoorNode / WindowNode children of that wall.
      if (isSingleWall && action.attachments && action.attachments.length > 0) {
        const singleWallId = wallIds[0]!
        const wallStart = points[0]!
        const wallEnd = points[1]!
        const wallLen = Math.hypot(wallEnd[0] - wallStart[0], wallEnd[1] - wallStart[1])
        for (const a of action.attachments) {
          const clampedPos = Math.max(0, Math.min(wallLen, a.position))
          if (a.type === 'door') {
            const doorId = createId('door')
            nextNodes[doorId] = DoorNode.parse({
              id: doorId,
              name: `Door`,
              parentId: singleWallId,
              metadata: { createdBy: 'ai-chat' },
              position: [clampedPos, 1.05, 0],
              rotation: [0, 0, 0],
              side: 'front',
              wallId: singleWallId,
              width: Math.max(0.6, a.width),
              height: a.height,
            })
            const wallNode = nextNodes[singleWallId]
            if (wallNode && Array.isArray(wallNode.children)) {
              wallNode.children.push(doorId)
            }
          } else {
            const windowId = createId('window')
            nextNodes[windowId] = WindowNode.parse({
              id: windowId,
              name: `Window`,
              parentId: singleWallId,
              metadata: { createdBy: 'ai-chat' },
              position: [clampedPos, 1.35, 0],
              rotation: [0, 0, 0],
              side: 'front',
              wallId: singleWallId,
              width: Math.max(0.4, a.width),
              height: a.height,
            })
            const wallNode = nextNodes[singleWallId]
            if (wallNode && Array.isArray(wallNode.children)) {
              wallNode.children.push(windowId)
            }
          }
        }
      }

      // Outline-only rooms (e.g. floor exterior) skip zone fill, openings, items.
      if (!action.outlineOnly) {
        const zoneId = createId('zone')
        nextNodes[zoneId] = {
          object: 'node',
          id: zoneId,
          type: 'zone',
          name: action.name,
          parentId: levelId,
          visible: true,
          metadata: { createdBy: 'ai-chat' },
          polygon: points,
          color: action.color ?? '#3b82f6',
        }
        children.push(zoneId)

        // Openings only make geometric sense on rectangular rooms — placing a
        // door at "wall index 0, fraction 0.5" on an L-shaped polygon is
        // ambiguous, so we skip them for polygon imports too.
        if (!action.polygon) {
          addRoomOpenings({
            children,
            depth,
            levelId,
            nextNodes,
            wallIds,
            width,
            doors: action.doors ?? 1,
            windows: action.windows ?? 0,
          })
        }
        addRoomItems({
          centerX,
          centerZ,
          children,
          depth,
          items: action.items ?? [],
          levelId,
          nextNodes,
          width,
        })
      }

      fallbackCenterX = centerX + width / 2
      continue
    }

    if (action.type === 'update_node') {
      updateNodeFromAction(nextNodes, action)
      continue
    }

    if (action.type === 'move_opening') {
      moveOpeningFromAction(nextNodes, action)
      continue
    }

    if (action.type === 'delete_nodes') {
      deleteNodesFromAction(nextNodes, nextRootNodeIds, action.nodeIds)
      continue
    }

    if (action.type === 'create_wall') {
      const wallId = createId('wall')
      nextNodes[wallId] = {
        object: 'node',
        id: wallId,
        type: 'wall',
        name: action.name,
        parentId: levelId,
        visible: true,
        metadata: { createdBy: 'ai-chat' },
        children: [],
        start: action.start,
        end: action.end,
        thickness: action.thickness,
        height: action.height,
        frontSide: 'unknown',
        backSide: 'unknown',
      }
      children.push(wallId)
      continue
    }

    if (action.type === 'extend_wall') {
      extendWallEndpoint(nextNodes, action)
      continue
    }

    if (action.type === 'duplicate_node') {
      duplicateSceneNode(nextNodes, action, levelId, children)
      continue
    }

    if (action.type === 'translate_wall') {
      translateWall(nextNodes, action)
      continue
    }

    if (action.type === 'rotate_wall') {
      rotateWall(nextNodes, action)
      continue
    }

    if (action.type === 'fill_gap') {
      fillGapBetweenWalls(nextNodes, action, levelId, children)
      continue
    }

    if (action.type === 'trim_wall') {
      trimWallAtIntersection(nextNodes, action)
    }
  }

  nextNodes[levelId] = { ...level, children }

  return {
    nodes: nextNodes,
    rootNodeIds: nextRootNodeIds,
  }
}

function buildSceneContext() {
  const { nodes } = useScene.getState()
  const nodeMap = nodes as Record<string, any>
  const selectedIds = useViewer.getState().selection.selectedIds
  const selectedNodes = selectedIds
    .map((id) => nodeMap[id])
    .filter(Boolean)
    .slice(0, 20)
    .map((node) => (node?.type === 'wall' ? summarizeWall(node) : summarizeNode(node, nodeMap)))
  const counts = Object.values(nodes).reduce<Record<string, number>>((acc, node) => {
    if (!node?.type) return acc
    acc[node.type] = (acc[node.type] ?? 0) + 1
    return acc
  }, {})

  const compactNodes = Object.values(nodeMap)
    .filter((node) => ['door', 'window', 'item', 'zone'].includes(node?.type))
    .slice(0, 150)
    .map((node) => summarizeNode(node, nodeMap))
  const walls = Object.values(nodeMap)
    .filter((node) => node?.type === 'wall')
    .slice(0, 150)
    .map((node) => summarizeWall(node))

  return {
    selectedNodes,
    nodes: compactNodes,
    walls,
    summary: Object.entries(counts)
      .map(([type, count]) => `${type}:${count}`)
      .join(', '),
  }
}

function summarizeNode(node: any, nodes: Record<string, any>) {
  const parentWall =
    typeof node.parentId === 'string' && nodes[node.parentId]?.type === 'wall'
      ? summarizeWall(nodes[node.parentId])
      : undefined

  return {
    id: node.id,
    type: node.type,
    name: node.name,
    parentId: node.parentId,
    position: node.position,
    rotation: node.rotation,
    width: node.width,
    height: node.height,
    start: node.start,
    end: node.end,
    assetId: node.asset?.id,
    assetName: node.asset?.name,
    parentWall,
  }
}

function summarizeWall(wall: any) {
  const start = Array.isArray(wall.start) ? wall.start : [0, 0]
  const end = Array.isArray(wall.end) ? wall.end : [0, 0]
  const dx = Number(end[0]) - Number(start[0])
  const dz = Number(end[1]) - Number(start[1])
  const absDx = Math.abs(dx)
  const absDz = Math.abs(dz)
  // 2D floorplan rotates the scene by FLOORPLAN_VIEW_ROTATION_DEG (90°).
  // Result: scene +X = compass South, scene -X = North; scene +Z = West, -Z = East.
  // So a wall varying in Z runs east-west in compass space; varying in X runs north-south.
  let orientation: 'east-west' | 'north-south' | 'diagonal'
  if (absDz >= absDx * 4) orientation = 'east-west'
  else if (absDx >= absDz * 4) orientation = 'north-south'
  else orientation = 'diagonal'
  // +Z = West, so the endpoint with the LARGER z is the west endpoint.
  const westEndpoint = start[1] >= end[1] ? 'start' : 'end'
  // +X = South, so the endpoint with the LARGER x is the south endpoint.
  const southEndpoint = start[0] >= end[0] ? 'start' : 'end'
  return {
    id: wall.id,
    type: wall.type,
    name: wall.name,
    parentId: wall.parentId,
    start: wall.start,
    end: wall.end,
    length: getWallLength(wall),
    thickness: wall.thickness,
    height: wall.height,
    orientation,
    westEndpoint,
    eastEndpoint: westEndpoint === 'start' ? 'end' : 'start',
    southEndpoint,
    northEndpoint: southEndpoint === 'start' ? 'end' : 'start',
  }
}

function addRoomOpenings({
  children,
  depth,
  doors,
  levelId,
  nextNodes,
  wallIds,
  width,
  windows,
}: {
  children: string[]
  depth: number
  doors: number
  levelId: string
  nextNodes: Record<string, any>
  wallIds: string[]
  width: number
  windows: number
}) {
  const bottomWallId = wallIds[0]
  const rightWallId = wallIds[1]
  const topWallId = wallIds[2]
  if (bottomWallId) {
    const count = Math.min(doors, 4)
    for (let index = 0; index < count; index++) {
      const doorId = createId('door')
      nextNodes[doorId] = DoorNode.parse({
        id: doorId,
        name: `Door ${index + 1}`,
        parentId: bottomWallId,
        metadata: { createdBy: 'ai-chat' },
        position: [distributedDistance(index, count, width), 1.05, 0],
        rotation: [0, 0, 0],
        side: 'front',
        wallId: bottomWallId,
        width: 0.9,
        height: 2.1,
      })
      appendChild(nextNodes, bottomWallId, doorId)
    }
  }

  const windowTargets = [topWallId, rightWallId].filter(Boolean)
  for (let index = 0; index < Math.min(windows, 12); index++) {
    const wallId = windowTargets[index % windowTargets.length]
    if (!wallId) continue
    const wallLength = wallId === rightWallId ? depth : width
    const windowsOnWall = Math.ceil(windows / windowTargets.length)
    const positionIndex = Math.floor(index / windowTargets.length)
    const windowId = createId('window')
    nextNodes[windowId] = WindowNode.parse({
      id: windowId,
      name: `Window ${index + 1}`,
      parentId: wallId,
      metadata: { createdBy: 'ai-chat' },
      position: [distributedDistance(positionIndex, windowsOnWall, wallLength), 1.35, 0],
      rotation: [0, 0, 0],
      side: 'front',
      wallId,
      width: 1.4,
      height: 1.2,
    })
    appendChild(nextNodes, wallId, windowId)
  }

  void children
  void levelId
}

function updateNodeFromAction(nodes: Record<string, any>, action: UpdateNodeAction) {
  const node = nodes[action.nodeId]
  if (!node) return

  const patch = sanitizeNodePatch(node, action.patch, nodes)
  if (Object.keys(patch).length === 0) return

  nodes[action.nodeId] = {
    ...node,
    ...patch,
    metadata: {
      ...(node.metadata ?? {}),
      updatedBy: 'ai-chat',
    },
  }
}

function moveOpeningFromAction(nodes: Record<string, any>, action: MoveOpeningAction) {
  const node = nodes[action.nodeId]
  if (!node || (node.type !== 'door' && node.type !== 'window')) return

  const nextWallId = action.wallId ?? node.wallId ?? node.parentId
  const wall = typeof nextWallId === 'string' ? nodes[nextWallId] : null
  if (!wall || wall.type !== 'wall') return

  const oldParentId = typeof node.parentId === 'string' ? node.parentId : null
  if (oldParentId && oldParentId !== wall.id) {
    removeChild(nodes, oldParentId, node.id)
    appendChild(nodes, wall.id, node.id)
  }

  const wallLength = getWallLength(wall)
  const currentPosition = normalizeTuple3(node.position, [
    0,
    node.type === 'window' ? 1.35 : 1.05,
    0,
  ])
  const width = typeof node.width === 'number' ? node.width : node.type === 'window' ? 1.4 : 0.9
  const margin = Math.min(Math.max(width / 2 + 0.2, 0.35), wallLength / 2)
  const distance = clamp(wallLength * clamp(action.t ?? 0.5, 0, 1), margin, wallLength - margin)

  nodes[action.nodeId] = {
    ...node,
    parentId: wall.id,
    wallId: wall.id,
    position: [distance, action.y ?? currentPosition[1], 0],
    metadata: {
      ...(node.metadata ?? {}),
      updatedBy: 'ai-chat',
    },
  }
}

function deleteNodesFromAction(
  nodes: Record<string, any>,
  rootNodeIds: string[],
  nodeIds: string[],
) {
  const queue = [...new Set(nodeIds)]
  const deleted = new Set<string>()

  while (queue.length > 0) {
    const nodeId = queue.pop()
    if (!nodeId || deleted.has(nodeId)) continue
    const node = nodes[nodeId]
    if (!node) continue

    if (typeof node.parentId === 'string') {
      removeChild(nodes, node.parentId, nodeId)
    }
    const rootIndex = rootNodeIds.indexOf(nodeId)
    if (rootIndex >= 0) rootNodeIds.splice(rootIndex, 1)

    if (Array.isArray(node.children)) {
      queue.push(
        ...node.children.filter(
          (childId: unknown): childId is string => typeof childId === 'string',
        ),
      )
    }

    delete nodes[nodeId]
    deleted.add(nodeId)
  }
}

function sanitizeNodePatch(
  node: any,
  patch: UpdateNodeAction['patch'],
  nodes: Record<string, any>,
) {
  const next: Record<string, unknown> = {}

  if (typeof patch.name === 'string') next.name = patch.name
  if (typeof patch.visible === 'boolean') next.visible = patch.visible
  if (patch.rotation) next.rotation = normalizeTuple3(patch.rotation, node.rotation ?? [0, 0, 0])
  if (patch.scale && node.type === 'item')
    next.scale = normalizeTuple3(patch.scale, node.scale ?? [1, 1, 1])
  if (typeof patch.color === 'string' && node.type === 'zone') next.color = patch.color
  if (typeof patch.width === 'number' && ['door', 'window'].includes(node.type)) {
    next.width = clamp(patch.width, 0.1, 12)
  }
  if (typeof patch.height === 'number' && ['wall', 'door', 'window'].includes(node.type)) {
    next.height = clamp(patch.height, 0.1, 12)
  }
  if (typeof patch.thickness === 'number' && node.type === 'wall') {
    next.thickness = clamp(patch.thickness, 0.02, 2)
  }
  if (patch.position && ['item', 'door', 'window'].includes(node.type)) {
    next.position =
      node.type === 'door' || node.type === 'window'
        ? clampOpeningPosition(node, patch.position, nodes)
        : normalizeTuple3(patch.position, node.position ?? [0, 0, 0])
  }
  if (node.type === 'wall') {
    if (patch.start) next.start = normalizeTuple2(patch.start, node.start ?? [0, 0])
    if (patch.end) next.end = normalizeTuple2(patch.end, node.end ?? [0, 0])
  }

  return next
}

function clampOpeningPosition(
  node: any,
  position: [number, number, number],
  nodes: Record<string, any>,
) {
  const wallId = node.wallId ?? node.parentId
  const wall = typeof wallId === 'string' ? nodes[wallId] : null
  if (!wall || wall.type !== 'wall') return normalizeTuple3(position, node.position ?? [0, 1, 0])

  const wallLength = getWallLength(wall)
  const width = typeof node.width === 'number' ? node.width : node.type === 'window' ? 1.4 : 0.9
  const margin = Math.min(Math.max(width / 2 + 0.2, 0.35), wallLength / 2)
  const normalized = normalizeTuple3(position, node.position ?? [0, 1, 0])
  return [clamp(normalized[0], margin, wallLength - margin), normalized[1], 0]
}

function addRoomItems({
  centerX,
  centerZ,
  children,
  depth,
  items,
  levelId,
  nextNodes,
  width,
}: {
  centerX: number
  centerZ: number
  children: string[]
  depth: number
  items: NonNullable<CreateRoomAction['items']>
  levelId: string
  nextNodes: Record<string, any>
  width: number
}) {
  const placements = expandItems(items).slice(0, 120)
  if (placements.length === 0) return

  const columns = Math.max(1, Math.ceil(Math.sqrt(placements.length * (width / depth))))
  const rows = Math.max(1, Math.ceil(placements.length / columns))
  const stepX = width / (columns + 1)
  const stepZ = depth / (rows + 1)

  placements.forEach((assetId, index) => {
    const asset = CATALOG_ITEMS.find((item) => item.id === assetId)
    if (!asset) return

    const column = index % columns
    const row = Math.floor(index / columns)
    const itemId = createId('item')
    nextNodes[itemId] = {
      object: 'node',
      id: itemId,
      type: 'item',
      name: asset.name,
      parentId: levelId,
      visible: true,
      metadata: { createdBy: 'ai-chat' },
      position: [
        centerX - width / 2 + stepX * (column + 1),
        0,
        centerZ - depth / 2 + stepZ * (row + 1),
      ],
      rotation: [0, rotationForAsset(assetId), 0],
      scale: [1, 1, 1],
      children: [],
      asset,
    }
    children.push(itemId)
  })
}

function expandItems(items: NonNullable<CreateRoomAction['items']>): string[] {
  return items.flatMap((item) =>
    Array.from({ length: clamp(item.count, 1, 80) }, () => item.assetId),
  )
}

function distributedDistance(index: number, count: number, length: number): number {
  if (count <= 1) return length / 2
  const margin = Math.min(1.2, Math.max(0.4, length * 0.15))
  const usable = Math.max(0.1, length - margin * 2)
  return margin + (usable * index) / (count - 1)
}

function appendChild(nodes: Record<string, any>, parentId: string, childId: string) {
  const parent = nodes[parentId]
  if (!parent) return
  const children = Array.isArray(parent.children) ? parent.children : []
  if (children.includes(childId)) return
  nodes[parentId] = { ...parent, children: [...children, childId] }
}

function removeChild(nodes: Record<string, any>, parentId: string, childId: string) {
  const parent = nodes[parentId]
  if (!parent || !Array.isArray(parent.children)) return
  nodes[parentId] = {
    ...parent,
    children: parent.children.filter((id: unknown) => id !== childId),
  }
}

function getWallLength(wall: any): number {
  if (!Array.isArray(wall?.start) || !Array.isArray(wall?.end)) return 0
  const dx = Number(wall.end[0]) - Number(wall.start[0])
  const dz = Number(wall.end[1]) - Number(wall.start[1])
  const length = Math.sqrt(dx * dx + dz * dz)
  return Number.isFinite(length) ? length : 0
}

function normalizeTuple2(value: unknown, fallback: [number, number]): [number, number] {
  if (!Array.isArray(value) || value.length < 2) return fallback
  const tuple = value.slice(0, 2).map((item) => Number(item))
  if (tuple.some((item) => !Number.isFinite(item))) return fallback
  return tuple as [number, number]
}

function normalizeTuple3(
  value: unknown,
  fallback: [number, number, number],
): [number, number, number] {
  if (!Array.isArray(value) || value.length < 3) return fallback
  const tuple = value.slice(0, 3).map((item) => Number(item))
  if (tuple.some((item) => !Number.isFinite(item))) return fallback
  return tuple as [number, number, number]
}

function rotationForAsset(assetId: string): number {
  if (assetId.includes('table') || assetId === 'kitchen-counter') return Math.PI / 2
  return 0
}

function findFirstLevelId(nodes: Record<string, any>): string | null {
  const level = Object.values(nodes).find((node) => node?.type === 'level')
  return typeof level?.id === 'string' ? level.id : null
}

function findWallMaxX(nodes: Record<string, any>): number | null {
  let maxX: number | null = null
  for (const node of Object.values(nodes)) {
    if (node?.type !== 'wall') continue
    for (const point of [node.start, node.end]) {
      const x = Array.isArray(point) && typeof point[0] === 'number' ? point[0] : null
      if (x === null) continue
      maxX = maxX === null ? x : Math.max(maxX, x)
    }
  }

  return maxX
}

function createId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replaceAll('-', '').slice(0, 16)}`
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function extendWallEndpoint(nodes: Record<string, any>, action: ExtendWallAction) {
  const wall = nodes[action.nodeId]
  if (!wall || wall.type !== 'wall') return
  const start = wall.start as [number, number]
  const end = wall.end as [number, number]
  const dx = end[0] - start[0]
  const dz = end[1] - start[1]
  const len = Math.sqrt(dx * dx + dz * dz)
  if (len < 1e-6) return
  const ux = dx / len
  const uz = dz / len
  if (action.endpoint === 'end') {
    nodes[action.nodeId] = {
      ...wall,
      end: [end[0] + ux * action.by, end[1] + uz * action.by],
      metadata: { ...(wall.metadata ?? {}), updatedBy: 'ai-chat' },
    }
  } else {
    nodes[action.nodeId] = {
      ...wall,
      start: [start[0] - ux * action.by, start[1] - uz * action.by],
      metadata: { ...(wall.metadata ?? {}), updatedBy: 'ai-chat' },
    }
  }
}

function duplicateSceneNode(
  nodes: Record<string, any>,
  action: DuplicateNodeAction,
  levelId: string,
  children: string[],
) {
  const node = nodes[action.nodeId]
  if (!node) return
  const [dx, dz] = action.offset
  const newId = createId(node.type as string)
  const cloned = structuredClone(node) as Record<string, any>
  cloned.id = newId
  cloned.parentId = levelId
  cloned.children = []
  cloned.metadata = { ...(cloned.metadata ?? {}), createdBy: 'ai-chat' }
  if (node.type === 'wall') {
    cloned.start = [cloned.start[0] + dx, cloned.start[1] + dz]
    cloned.end = [cloned.end[0] + dx, cloned.end[1] + dz]
  } else if (node.type === 'item' && Array.isArray(cloned.position)) {
    cloned.position = [cloned.position[0] + dx, cloned.position[1], cloned.position[2] + dz]
  }
  nodes[newId] = cloned
  children.push(newId)
}

function translateWall(nodes: Record<string, any>, action: TranslateWallAction) {
  const wall = nodes[action.nodeId]
  if (!wall || wall.type !== 'wall') return
  const [dx, dz] = action.delta
  nodes[action.nodeId] = {
    ...wall,
    start: [wall.start[0] + dx, wall.start[1] + dz],
    end: [wall.end[0] + dx, wall.end[1] + dz],
    metadata: { ...(wall.metadata ?? {}), updatedBy: 'ai-chat' },
  }
}

function rotateWall(nodes: Record<string, any>, action: RotateWallAction) {
  const wall = nodes[action.nodeId]
  if (!wall || wall.type !== 'wall') return
  const angle = (action.angleDeg * Math.PI) / 180
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  let pivotX: number
  let pivotZ: number
  if (action.pivot === 'start') {
    ;[pivotX, pivotZ] = wall.start as [number, number]
  } else if (action.pivot === 'end') {
    ;[pivotX, pivotZ] = wall.end as [number, number]
  } else {
    pivotX = ((wall.start[0] as number) + (wall.end[0] as number)) / 2
    pivotZ = ((wall.start[1] as number) + (wall.end[1] as number)) / 2
  }
  const rotatePoint = ([x, z]: [number, number]): [number, number] => {
    const rx = x - pivotX
    const rz = z - pivotZ
    return [pivotX + rx * cos - rz * sin, pivotZ + rx * sin + rz * cos]
  }
  nodes[action.nodeId] = {
    ...wall,
    start: rotatePoint(wall.start as [number, number]),
    end: rotatePoint(wall.end as [number, number]),
    metadata: { ...(wall.metadata ?? {}), updatedBy: 'ai-chat' },
  }
}

function fillGapBetweenWalls(
  nodes: Record<string, any>,
  action: FillGapAction,
  levelId: string,
  children: string[],
) {
  const wall1 = nodes[action.wallId1]
  const wall2 = nodes[action.wallId2]
  if (!wall1 || wall1.type !== 'wall' || !wall2 || wall2.type !== 'wall') return
  const endpoints1: Array<[number, number]> = [wall1.start, wall1.end]
  const endpoints2: Array<[number, number]> = [wall2.start, wall2.end]
  let bestDist = Infinity
  let bestP1 = endpoints1[0]!
  let bestP2 = endpoints2[0]!
  for (const p1 of endpoints1) {
    for (const p2 of endpoints2) {
      const d = Math.hypot(p2[0] - p1[0], p2[1] - p1[1])
      if (d < bestDist) {
        bestDist = d
        bestP1 = p1
        bestP2 = p2
      }
    }
  }
  if (bestDist < 0.01) return
  const gapWallId = createId('wall')
  nodes[gapWallId] = {
    object: 'node',
    id: gapWallId,
    type: 'wall',
    name: 'Gap Wall',
    parentId: levelId,
    visible: true,
    metadata: { createdBy: 'ai-chat' },
    children: [],
    start: bestP1,
    end: bestP2,
    thickness: (wall1.thickness as number | undefined) ?? 0.1,
    height: (wall1.height as number | undefined) ?? 2.8,
    frontSide: 'unknown',
    backSide: 'unknown',
  }
  children.push(gapWallId)
}

function trimWallAtIntersection(nodes: Record<string, any>, action: TrimWallAction) {
  const target = nodes[action.nodeId]
  const trimWall = nodes[action.trimToWallId]
  if (!target || target.type !== 'wall' || !trimWall || trimWall.type !== 'wall') return
  const ip = lineIntersectionPoint(
    target.start as [number, number],
    target.end as [number, number],
    trimWall.start as [number, number],
    trimWall.end as [number, number],
  )
  if (!ip) return
  const dx = (target.end[0] as number) - (target.start[0] as number)
  const dz = (target.end[1] as number) - (target.start[1] as number)
  const len2 = dx * dx + dz * dz
  if (len2 < 1e-10) return
  const t =
    ((ip[0] - (target.start[0] as number)) * dx + (ip[1] - (target.start[1] as number)) * dz) /
    len2
  if (t > 0.5) {
    nodes[action.nodeId] = {
      ...target,
      end: ip,
      metadata: { ...(target.metadata ?? {}), updatedBy: 'ai-chat' },
    }
  } else {
    nodes[action.nodeId] = {
      ...target,
      start: ip,
      metadata: { ...(target.metadata ?? {}), updatedBy: 'ai-chat' },
    }
  }
}

function lineIntersectionPoint(
  p1: [number, number],
  p2: [number, number],
  p3: [number, number],
  p4: [number, number],
): [number, number] | null {
  const d1x = p2[0] - p1[0]
  const d1z = p2[1] - p1[1]
  const d2x = p4[0] - p3[0]
  const d2z = p4[1] - p3[1]
  const cross = d1x * d2z - d1z * d2x
  if (Math.abs(cross) < 1e-10) return null
  const dx = p3[0] - p1[0]
  const dz = p3[1] - p1[1]
  const t = (dx * d2z - dz * d2x) / cross
  return [p1[0] + t * d1x, p1[1] + t * d1z]
}
