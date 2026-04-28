import { useState, useRef, useEffect } from 'react'
import { parseCitations, getChunkById, formatMetadata } from '../../utils/citationUtils'
import styles from './CitationPopup.module.css'

/**
 * CitationTag – renders [1] as a clickable/hoverable hyperlink.
 * Shows a popup with the original chunk text on hover.
 */
export function CitationTag({ id, chunks }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const chunk = getChunkById(id, chunks)

  // Close popup when clicking outside
  useEffect(() => {
    function handler(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  if (!chunk) return <span className={styles.tag}>[{id}]</span>

  return (
    <span className={styles.wrapper} ref={ref}>
      <button
        className={`${styles.tag} ${chunk.low_relevance ? styles.warn : ''}`}
        onClick={() => setOpen((v) => !v)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        aria-label={`Xem tài liệu ${id}`}
      >
        [{id}]
      </button>

      {open && (
        <div className={styles.popup} role="tooltip">
          <div className={styles.popupHeader}>
            <span className={styles.popupId}>Tài liệu [{id}]</span>
            {chunk.low_relevance && (
              <span className={styles.popupWarn}>⚠ Độ liên quan thấp</span>
            )}
            <span className={styles.popupScore}>
              Điểm: {(chunk.score * 100).toFixed(0)}%
            </span>
          </div>
          <p className={styles.popupMeta}>{formatMetadata(chunk.metadata)}</p>
          <p className={styles.popupContent}>{chunk.content}</p>
        </div>
      )}
    </span>
  )
}

/**
 * CitationText – renders a full text string, replacing [n] with CitationTag
 */
export function CitationText({ text, chunks }) {
  const parts = parseCitations(text, chunks)
  return (
    <span>
      {parts.map((part, i) =>
        part.type === 'citation'
          ? <CitationTag key={i} id={part.id} chunks={chunks} />
          : <span key={i}>{part.content}</span>
      )}
    </span>
  )
}
