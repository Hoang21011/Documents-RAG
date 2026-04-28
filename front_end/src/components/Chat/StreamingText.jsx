import { useEffect, useRef } from 'react'
import styles from './StreamingText.module.css'

/**
 * Animated blinking cursor while streaming.
 */
export function StreamingCursor() {
  return <span className={styles.cursor} aria-hidden="true">▋</span>
}

/**
 * StepIndicator – show pipeline steps with animated dots.
 */
export function StepIndicator({ steps }) {
  if (!steps?.length) return null
  const current = steps[steps.length - 1]
  return (
    <div className={styles.stepRow}>
      <span className={styles.stepDot} />
      <span className={styles.stepText}>{current}</span>
    </div>
  )
}
