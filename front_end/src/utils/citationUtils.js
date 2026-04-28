/**
 * parseCitations(text, chunks)
 * Tách nội dung text thành mảng các segment:
 *   { type: 'text', content: string }
 *   { type: 'citation', id: number }
 */
export function parseCitations(text, chunks = []) {
  const parts = []
  // Regex: tìm tất cả [số]
  const regex = /\[(\d+)\]/g
  let lastIndex = 0
  let match

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: text.slice(lastIndex, match.index) })
    }
    const id = parseInt(match[1])
    parts.push({ type: 'citation', id })
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.slice(lastIndex) })
  }
  return parts
}

/**
 * getChunkById(id, chunks)
 * Tìm chunk theo citation index (1-based)
 */
export function getChunkById(id, chunks) {
  return chunks.find((c) => c.id === id) || null
}

/**
 * formatMetadata(metadata)
 * Trả về chuỗi mô tả ngắn gọn từ metadata của chunk
 */
export function formatMetadata(metadata = {}) {
  const parts = []
  if (metadata.title) parts.push(metadata.title)
  if (metadata.co_quan_ban_hanh) parts.push(metadata.co_quan_ban_hanh)
  if (metadata.ngay_ban_hanh) parts.push(`Ngày ${metadata.ngay_ban_hanh}`)
  return parts.join(' • ') || 'Tài liệu không rõ nguồn'
}
