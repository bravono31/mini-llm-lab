import type { Vec } from '../../engine/alloc'
import type { ParamSpec } from '../../engine/params'
import { Heatmap } from '../viz/Heatmap'
import type { ValueKind } from '../viz/Tag'

/** Draw one parameter tensor (1-D as a single row) as a heat-map. */
export function SpecHeat({ data, spec, title, maxAbs, cell, legend = true, tag }: { data: Vec; spec: ParamSpec; title?: string; maxAbs?: number; cell?: number; legend?: boolean; tag?: ValueKind }) {
  const rows = spec.shape.length === 2 ? spec.shape[0] : 1
  const cols = spec.shape.length === 2 ? spec.shape[1] : spec.shape[0]
  const c = cell ?? Math.max(4, Math.min(14, Math.floor(560 / cols)))
  return <Heatmap values={data.subarray(spec.offset, spec.offset + spec.size)} rows={rows} cols={cols} cell={c} gap={c >= 8 ? 1 : 0} maxAbs={maxAbs} title={title ?? `${spec.name}  [${spec.shape.join(' × ')}]`} legend={legend} rowName="i" colName="j" tag={tag} />
}

export function stats(v: ArrayLike<number>): { mean: number; std: number; absMax: number; norm: number } {
  let s = 0
  let s2 = 0
  let am = 0
  for (let i = 0; i < v.length; i++) {
    s += v[i]
    s2 += v[i] * v[i]
    if (Math.abs(v[i]) > am) am = Math.abs(v[i])
  }
  const mean = s / v.length
  return { mean, std: Math.sqrt(Math.max(0, s2 / v.length - mean * mean)), absMax: am, norm: Math.sqrt(s2) }
}
