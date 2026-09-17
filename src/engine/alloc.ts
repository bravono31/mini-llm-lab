// Numeric buffer type shared by the whole engine.
// Float32 in the app; tests switch to Float64 for precise finite-difference checks.
export type Vec = Float32Array | Float64Array

let ctor: Float32ArrayConstructor | Float64ArrayConstructor = Float32Array

export function alloc(n: number): Vec {
  return new ctor(n)
}

export function usePrecision(p: 'f32' | 'f64'): void {
  ctor = p === 'f64' ? Float64Array : Float32Array
}
