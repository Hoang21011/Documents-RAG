const BASE_URL = '/api'  // proxied via Vite to http://localhost:8000

/**
 * streamChat – gọi SSE endpoint và xử lý từng event.
 * @param {string} query
 * @param {string} sessionId
 * @param {object} filters
 * @param {{ onStep, onSources, onToken, onDone, onError }} handlers
 */
export async function streamChat(query, sessionId, filters, handlers) {
  const { onStep, onSources, onToken, onDone, onError } = handlers

  const response = await fetch(`${BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId, filters }),
  })

  if (!response.ok) {
    onError?.(`HTTP ${response.status}: ${response.statusText}`)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete last line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue

      try {
        const event = JSON.parse(raw)
        switch (event.type) {
          case 'step':    onStep?.(event.message);  break
          case 'sources': onSources?.(event.chunks); break
          case 'token':   onToken?.(event.content);  break
          case 'done':    onDone?.(event);           break
          case 'error':   onError?.(event.message);  break
        }
      } catch (e) {
        console.warn('Failed to parse SSE event:', raw)
      }
    }
  }
}
