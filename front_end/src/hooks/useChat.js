import { useState, useRef, useCallback } from 'react'
import { streamChat } from '../services/chatService'

export function useChat() {
  const [messages, setMessages]   = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const sessionIdRef              = useRef(`session_${Date.now()}`)

  const addMessage = (msg) =>
    setMessages((prev) => [...prev, msg])

  const updateLastAssistant = (updater) =>
    setMessages((prev) => {
      const copy = [...prev]
      const last = copy[copy.length - 1]
      if (last?.role === 'assistant') {
        copy[copy.length - 1] = { ...last, ...updater(last) }
      }
      return copy
    })

  const sendMessage = useCallback(async (query, filters = null) => {
    if (!query.trim() || isLoading) return

    // Append user message
    addMessage({ role: 'user', content: query, id: Date.now() })

    // Placeholder assistant message
    const assistantId = Date.now() + 1
    addMessage({
      role: 'assistant',
      id: assistantId,
      content: '',
      chunks: [],
      steps: [],
      done: false,
      error: null,
    })

    setIsLoading(true)

    await streamChat(query, sessionIdRef.current, filters, {
      onStep: (msg) =>
        updateLastAssistant((prev) => ({
          steps: [...(prev.steps || []), msg],
        })),

      onSources: (chunks) =>
        updateLastAssistant(() => ({ chunks })),

      onToken: (token) =>
        updateLastAssistant((prev) => ({
          content: (prev.content || '') + token,
        })),

      onDone: (event) =>
        updateLastAssistant(() => ({
          content: event.answer,
          chunks: event.chunks,
          done: true,
        })),

      onError: (msg) =>
        updateLastAssistant(() => ({
          error: msg,
          done: true,
        })),
    })

    setIsLoading(false)
  }, [isLoading])

  const clearChat = useCallback(() => {
    setMessages([])
    sessionIdRef.current = `session_${Date.now()}`
  }, [])

  return { messages, isLoading, sendMessage, clearChat, sessionId: sessionIdRef.current }
}
