import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { ML_DATA } from '../../data/ml-labels'
import { buildDataset, featurize, segment, SoftmaxRegression, type Dataset, type Word } from '../../engine/classifier'
import { useLab } from './LabProvider'

/** State of the machine-learning part: the labelled dataset, the classifier and its training history, for the current language. */

export const ML_LR = 0.5
/** the classifier the chapters open with has already been trained this many steps (as the LLM part opens on pretrained weights) */
export const ML_PRETRAIN_STEPS = 300

export interface MlLastStep {
  /** weights before the step */
  W: Float64Array
  dW: Float64Array
}

export interface MlSession {
  model: SoftmaxRegression
  /** mean loss on the training / test examples, one entry per step taken (index 0 = before any step) */
  trainLoss: number[]
  testLoss: number[]
}

export interface Ml {
  ds: Dataset
  session: MlSession
  model: SoftmaxRegression
  steps: number
  /** bumps whenever the model changes */
  version: number
  last: MlLastStep | null
  train: (n: number) => void
  resetToZero: () => void
  resetToTrained: () => void
  /** the top-bar input, as the classifier sees it */
  words: Word[]
  x: Float64Array
  probs: Float64Array
  pred: number
}

const Ctx = createContext<Ml | null>(null)

function runSteps(s: MlSession, ds: Dataset, n: number): MlLastStep | null {
  let last: MlLastStep | null = null
  for (let i = 0; i < n; i++) {
    const W = s.model.W.slice()
    const { dW } = s.model.gradient(ds.X, ds.y, ds.train)
    s.model.step(ds.X, ds.y, ds.train, ML_LR)
    s.trainLoss.push(s.model.loss(ds.X, ds.y, ds.train))
    s.testLoss.push(s.model.loss(ds.X, ds.y, ds.test))
    last = { W, dW }
  }
  return last
}

function makeSession(ds: Dataset, steps: number): MlSession {
  const model = new SoftmaxRegression(ds.classes.length, ds.vocab.length)
  const s = { model, trainLoss: [model.loss(ds.X, ds.y, ds.train)], testLoss: [model.loss(ds.X, ds.y, ds.test)] }
  runSteps(s, ds, steps)
  return s
}

export function MlProvider({ children }: { children: ReactNode }) {
  const lab = useLab()
  const ds = useMemo(() => buildDataset(ML_DATA[lab.lang]), [lab.lang])
  const [state, setState] = useState(() => ({ ds, session: makeSession(ds, ML_PRETRAIN_STEPS), last: null as MlLastStep | null }))
  const [version, setVersion] = useState(0)
  // language switch: start again from the trained classifier of the new dataset
  const current = state.ds === ds ? state : { ds, session: makeSession(ds, ML_PRETRAIN_STEPS), last: null }
  if (current !== state) setState(current)
  const { session, last } = current

  const train = useCallback(
    (n: number) => {
      const l = runSteps(session, ds, n)
      setState((s) => ({ ...s, last: l }))
      setVersion((v) => v + 1)
    },
    [session, ds],
  )
  const resetToZero = useCallback(() => {
    setState({ ds, session: makeSession(ds, 0), last: null })
    setVersion((v) => v + 1)
  }, [ds])
  const resetToTrained = useCallback(() => {
    setState({ ds, session: makeSession(ds, ML_PRETRAIN_STEPS), last: null })
    setVersion((v) => v + 1)
  }, [ds])

  const words = useMemo(() => segment(lab.input, ds.data), [lab.input, ds])
  const x = useMemo(() => featurize(lab.input, ds.data, ds.vocab), [lab.input, ds])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const probs = useMemo(() => session.model.probs(x), [session, x, version])
  const pred = probs.indexOf(Math.max(...probs))

  const value: Ml = useMemo(
    () => ({ ds, session, model: session.model, steps: session.trainLoss.length - 1, version, last, train, resetToZero, resetToTrained, words, x, probs, pred }),
    [ds, session, version, last, train, resetToZero, resetToTrained, words, x, probs, pred],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useMl(): Ml {
  const v = useContext(Ctx)
  if (!v) throw new Error('MlProvider missing')
  return v
}
