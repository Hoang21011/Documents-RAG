import { useRef, useEffect } from 'react'
import { useChat } from './hooks/useChat'
import ChatMessage from './components/Chat/ChatMessage'
import ChatInput from './components/Chat/ChatInput'
import Sidebar from './components/Sidebar/Sidebar'
import styles from './App.module.css'

export default function App() {
  const { messages, isLoading, sendMessage, clearChat } = useChat()
  const bottomRef = useRef(null)

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className={styles.layout}>
      <Sidebar
        onSelect={(q) => sendMessage(q)}
        onClear={clearChat}
      />

      <main className={styles.main}>
        <div className={styles.messagesArea} id="messages-area">
          {messages.length === 0 && (
            <div className={styles.empty}>
              <p className={styles.emptyIcon}>🎓</p>
              <h2 className={styles.emptyTitle}>Chatbot Tra cứu Quy Chế</h2>
              <p className={styles.emptySub}>
                Hỏi bất kỳ quy định, quy chế, nội quy của Trường ĐH Kinh tế Quốc dân —<br />
                hệ thống sẽ tra cứu và tổng hợp kết quả kèm nguồn trích dẫn chính xác.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          <div ref={bottomRef} />
        </div>

        <div className={styles.inputArea}>
          <ChatInput onSend={sendMessage} disabled={isLoading} />
        </div>
      </main>
    </div>
  )
}
