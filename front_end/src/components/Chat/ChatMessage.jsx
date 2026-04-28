import { useState } from 'react'
import { CitationText } from './CitationPopup'
import { StreamingCursor, StepIndicator } from './StreamingText'
import { formatMetadata } from '../../utils/citationUtils'
import styles from './ChatMessage.module.css'

/**
 * SourceCard – renders one source document with an expandable full-text viewer.
 * "Xem tài liệu gốc" now shows the full chunk content inline, not a broken URL.
 */
function SourceCard({ chunk }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className={`${styles.sourceCard} ${chunk.low_relevance ? styles.lowRelevance : ''}`}>
      <div className={styles.sourceHeader}>
        <span className={styles.sourceIndex}>[{chunk.id}]</span>
        {chunk.low_relevance && (
          <span className={styles.warnBadge}>⚠ Độ liên quan thấp</span>
        )}
        <span className={styles.sourceScore}>{(chunk.score * 100).toFixed(0)}% liên quan</span>
      </div>

      <p className={styles.sourceMeta}>{formatMetadata(chunk.metadata)}</p>

      {/* Snippet preview */}
      <p className={styles.sourceSnippet}>
        {expanded ? chunk.content : chunk.content.slice(0, 200) + (chunk.content.length > 200 ? '…' : '')}
      </p>

      <div className={styles.sourceActions}>
        {/* Toggle full content inline */}
        <button
          className={styles.sourceLink}
          onClick={() => setExpanded(v => !v)}
        >
          {expanded ? '↑ Thu gọn' : '📄 Xem tài liệu gốc'}
        </button>

        {/* Copy snippet */}
        <button
          className={styles.sourceCopy}
          onClick={() => navigator.clipboard.writeText(chunk.content)}
          title="Sao chép nội dung"
        >
          📋
        </button>
      </div>

      {/* Additional metadata when expanded */}
      {expanded && chunk.metadata && (
        <div className={styles.sourceMeta} style={{ marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          {chunk.metadata.source && <div>📁 <strong>File:</strong> {chunk.metadata.source}</div>}
          {chunk.metadata.ngay_ban_hanh && <div>📅 <strong>Ngày ban hành:</strong> {chunk.metadata.ngay_ban_hanh}</div>}
          {chunk.metadata.ngay_co_hieu_luc && <div>✅ <strong>Hiệu lực:</strong> {chunk.metadata.ngay_co_hieu_luc}</div>}
          {chunk.metadata.pham_vi && <div>🌐 <strong>Phạm vi:</strong> {chunk.metadata.pham_vi}</div>}
        </div>
      )}
    </div>
  )
}

/**
 * ChatMessage – renders one message bubble (user or assistant).
 */
export default function ChatMessage({ message }) {
  const { role, content, chunks = [], steps = [], done, error } = message

  if (role === 'user') {
    return (
      <div className={styles.row + ' ' + styles.userRow}>
        <div className={styles.userBubble}>{content}</div>
      </div>
    )
  }

  // ── Assistant bubble ──────────────────────────────────────────────
  const hasContent = content && content.trim().length > 0
  const lines = hasContent ? content.split('\n') : []

  return (
    <div className={styles.row + ' ' + styles.assistantRow}>
      <div className={styles.avatar}>⚖️</div>

      <div className={styles.assistantBody}>
        {/* Pipeline steps */}
        {!done && <StepIndicator steps={steps} />}

        {/* Error state */}
        {error && (
          <div className={styles.errorBox}>❌ {error}</div>
        )}

        {/* Answer content with inline citations */}
        {hasContent && (
          <div className={styles.answerText}>
            {lines.map((line, i) => (
              <span key={i}>
                <CitationText text={line} chunks={chunks} />
                {i < lines.length - 1 && <br />}
              </span>
            ))}
            {!done && <StreamingCursor />}
          </div>
        )}

        {/* Sources reference list – only when done */}
        {done && chunks.length > 0 && (
          <div className={styles.sourcesSection}>
            <h4 className={styles.sourcesTitle}>📚 Tài liệu tham khảo</h4>
            <div className={styles.sourcesList}>
              {chunks.map((chunk) => (
                <SourceCard key={chunk.id} chunk={chunk} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
