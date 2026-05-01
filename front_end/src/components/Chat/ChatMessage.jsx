import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
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
  const [metaExpanded, setMetaExpanded] = useState(false)

  // Giới hạn snippet ban đầu khoảng 250 ký tự hoặc 3 dòng
  const snippet = chunk.content.length > 250 && !expanded 
    ? chunk.content.slice(0, 250) + "..." 
    : chunk.content

  return (
    <div className={`${styles.sourceCard} ${chunk.low_relevance ? styles.lowRelevance : ''}`}>
      <div className={styles.sourceHeader}>
        <div className={styles.sourceInfo}>
          <span className={styles.sourceIndex}>[{chunk.id}]</span>
          <span className={styles.sourceScore}>{(chunk.score * 100).toFixed(0)}% liên quan</span>
          {chunk.low_relevance && (
            <span className={styles.warnBadge}>⚠ Thấp</span>
          )}
        </div>
        <button 
          className={styles.sourceCopy}
          onClick={() => navigator.clipboard.writeText(chunk.content)}
          title="Sao chép"
        >
          📋
        </button>
      </div>

      <p className={styles.sourceMeta}>{formatMetadata(chunk.metadata)}</p>

      <p className={styles.sourceSnippet}>
        {snippet}
        {chunk.content.length > 250 && (
          <button 
            className={styles.expandBtn} 
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Thu gọn" : "Xem thêm"}
          </button>
        )}
      </p>

      <div className={styles.sourceActions}>
        <button
          className={styles.sourceLink}
          onClick={() => setMetaExpanded(!metaExpanded)}
        >
          {metaExpanded ? '↑ Ẩn chi tiết' : '📄 Chi tiết nguồn'}
        </button>
      </div>

      {metaExpanded && chunk.metadata && (
        <div className={styles.metaDetails}>
          {chunk.metadata.source && <div>📁 <strong>File:</strong> {chunk.metadata.source}</div>}
          {chunk.metadata.ngay_ban_hanh && <div>📅 <strong>Ngày ban hành:</strong> {chunk.metadata.ngay_ban_hanh}</div>}
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
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({node, ...props}) => <CitationText text={props.children} chunks={chunks} />,
                li: ({node, ...props}) => <li><CitationText text={props.children} chunks={chunks} /></li>,
              }}
            >
              {content}
            </ReactMarkdown>
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
