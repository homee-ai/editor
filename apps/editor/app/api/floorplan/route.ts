import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import type { NextRequest } from 'next/server'
import { getSceneOperations } from '@/lib/scene-store-server'
import { guardSceneApiRequest, sceneApiJson, sceneApiPreflight } from '@/lib/scene-api-security'

// CubiCasa + two VLM calls take minutes; run on Node with a generous budget.
export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const maxDuration = 600

const REPO_ROOT = resolve(process.cwd(), '../..')
const ORCHESTRATOR = join(REPO_ROOT, 'tools/floor2ifc/floorplan_to_scene.py')

function pythonExe(): string {
  if (process.env.FLOORPLAN_PYTHON) return process.env.FLOORPLAN_PYTHON
  const conda = join(homedir(), 'miniforge3/envs/floor2ifc/bin/python')
  return existsSync(conda) ? conda : 'python3'
}

function runOrchestrator(png: string, workRoot: string, pxPerMeter: number, wallHeight: number): Promise<void> {
  return new Promise((resolveRun, reject) => {
    const args = [ORCHESTRATOR, png, '--work-root', workRoot,
      '--px-per-meter', String(pxPerMeter), '--wall-height', String(wallHeight)]
    const proc = spawn(pythonExe(), args, { cwd: join(REPO_ROOT, 'tools/floor2ifc') })
    let stderr = ''
    proc.stdout.on('data', (d) => process.stdout.write(d))
    proc.stderr.on('data', (d) => { stderr += d.toString(); process.stderr.write(d) })
    proc.on('error', reject)
    proc.on('close', (code) =>
      code === 0 ? resolveRun() : reject(new Error(`pipeline exit ${code}\n${stderr.slice(-2000)}`)))
  })
}

export function OPTIONS(request: NextRequest) {
  return sceneApiPreflight(request)
}

export async function POST(request: NextRequest) {
  const guard = guardSceneApiRequest(request)
  if (guard) return guard

  let form: FormData
  try {
    form = await request.formData()
  } catch {
    return sceneApiJson(request, { error: 'invalid_request', details: 'expected multipart form-data' }, { status: 400 })
  }
  const file = form.get('file')
  if (!(file instanceof File)) {
    return sceneApiJson(request, { error: 'invalid_request', details: 'missing file field' }, { status: 400 })
  }
  const pxPerMeter = Number(form.get('pxPerMeter') ?? 60) || 60
  const wallHeight = Number(form.get('wallHeight') ?? 2.5) || 2.5
  const stem = (file.name.replace(/\.[^.]+$/, '') || 'floorplan').replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 40) || 'floorplan'

  const work = await mkdtemp(join(tmpdir(), 'floorplan-'))
  const pngPath = join(work, `${stem}.png`)
  await writeFile(pngPath, Buffer.from(await file.arrayBuffer()))

  try {
    await runOrchestrator(pngPath, work, pxPerMeter, wallHeight)
    const scenePath = join(work, stem, 'vlm', stem, 'scene.json')
    const graph = JSON.parse(await readFile(scenePath, 'utf-8'))
    const operations = await getSceneOperations()
    const meta = await operations.saveScene({
      name: stem, projectId: null, graph: graph as never, thumbnailUrl: null,
    })
    return sceneApiJson(request, meta, { status: 201, headers: { Location: `/scene/${meta.id}` } })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'pipeline_failed'
    return sceneApiJson(request, { error: 'pipeline_failed', message }, { status: 500 })
  }
}
