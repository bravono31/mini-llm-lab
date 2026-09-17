import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/mini-llm-lab/',
  plugins: [react()],
  test: {
    include: ['tests/**/*.test.ts'],
  },
})
