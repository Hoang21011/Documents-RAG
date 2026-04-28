import styles from './Sidebar.module.css'

const SUGGESTED = [
  'Quy định về liêm chính học thuật là gì?',
  'Sinh viên vi phạm đạo văn bị xử lý như thế nào?',
  'Điều kiện nhận học bổng tại trường NEU?',
  'Quy trình xét tốt nghiệp cho sinh viên chính quy?',
]

export default function Sidebar({ onSelect, onClear }) {
  return (
    <aside className={styles.sidebar} id="sidebar">
      <div className={styles.brand}>
        <span className={styles.brandIcon}>⚖️</span>
        <div>
          <h1 className={styles.brandName}>LegalRAG</h1>
          <p className={styles.brandSub}>Trợ lý Pháp luật AI</p>
        </div>
      </div>

      <nav className={styles.nav}>
        <p className={styles.sectionLabel}>Câu hỏi gợi ý</p>
        {SUGGESTED.map((q) => (
          <button
            key={q}
            className={styles.suggestion}
            onClick={() => onSelect(q)}
            id={`suggest-${q.slice(0, 20).replace(/\s/g, '-')}`}
          >
            <span className={styles.suggestionIcon}>💬</span>
            <span>{q}</span>
          </button>
        ))}
      </nav>

      <div className={styles.bottom}>
        <button className={styles.clearBtn} onClick={onClear} id="clear-chat-btn">
          🗑 Cuộc trò chuyện mới
        </button>
        <p className={styles.footer}>
          Powered by Qwen2.5-7B · Milvus · Redis
        </p>
      </div>
    </aside>
  )
}
