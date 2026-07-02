import { useState, useRef, useEffect } from "react"
import axios from "axios"

const SUGGESTIONS = [
    "नेपालको राजधानी के हो?",
    "सगरमाथाको उचाइ कति छ?",
    "NepBPE के हो?",
    "नेपालको राष्ट्रिय फूल के हो?",
    "दशैं के हो?",
]

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "नमस्ते! म NepBPE हुँ। नेपालको बारेमा केही सोध्नुहोस्।"
    }
  ])
  const [input, setInput]     = useState("")
  const [loading, setLoading] = useState(false)
  const bottomRef             = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const send = async (text) => {
    const msg = text || input.trim()
    if (!msg || loading) return

    setInput("")
    setMessages(prev => [...prev, { role: "user", text: msg }])
    setLoading(true)

    try {
      const res = await axios.post("http://localhost:8000/chat", {
        message: msg,
        temperature: 0.8,
        max_tokens: 100,
      })
      setMessages(prev => [...prev, {
        role: "assistant",
        text: res.data.response,
        tokens: res.data.tokens,
      }])
    } catch {
      setMessages(prev => [...prev, {
        role: "assistant",
        text: "माफ गर्नुहोस्, केही त्रुटि भयो।",
      }])
    }
    setLoading(false)
  }

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div style={{ maxWidth: 680, margin: "0 auto" }}>
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6, letterSpacing: "-0.3px" }}>
        Chat
      </h1>
      <p style={{ color: "#888", fontSize: 14, marginBottom: 24 }}>
        Chat with NepBPE-184M in Nepali. Fine-tuned on 300+ Nepali Q&A pairs.
      </p>

      {/* Chat window */}
      <div style={{
        border: "1px solid #e5e5e5",
        borderRadius: 12,
        height: 420,
        overflowY: "auto",
        padding: "20px 20px",
        marginBottom: 12,
        background: "#fafafa",
      }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            display: "flex",
            justifyContent: m.role === "user" ? "flex-end" : "flex-start",
            marginBottom: 16,
          }}>
            {m.role === "assistant" && (
              <div style={{
                width: 28, height: 28,
                borderRadius: "50%",
                background: "#0d0d0d",
                color: "#fff",
                fontSize: 11,
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginRight: 10,
                flexShrink: 0,
                marginTop: 2,
              }}>
                N
              </div>
            )}
            <div style={{
              maxWidth: "75%",
              background: m.role === "user" ? "#0d0d0d" : "#fff",
              color: m.role === "user" ? "#fff" : "#0d0d0d",
              border: m.role === "user" ? "none" : "1px solid #e5e5e5",
              borderRadius: m.role === "user"
                ? "18px 18px 4px 18px"
                : "18px 18px 18px 4px",
              padding: "10px 14px",
              fontSize: 14,
              lineHeight: 1.6,
            }}>
              {m.text}
              {m.tokens && (
                <div style={{ fontSize: 10, color: "#888", marginTop: 4 }}>
                  {m.tokens} tokens
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 28, height: 28, borderRadius: "50%",
              background: "#0d0d0d", color: "#fff",
              fontSize: 11, fontWeight: 600,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              N
            </div>
            <div style={{
              background: "#fff",
              border: "1px solid #e5e5e5",
              borderRadius: "18px 18px 18px 4px",
              padding: "10px 16px",
              fontSize: 14, color: "#888",
            }}>
              <span style={{ letterSpacing: 2 }}>···</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions */}
      {messages.length <= 1 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
          {SUGGESTIONS.map(s => (
            <button key={s} onClick={() => send(s)} style={{
              background: "none",
              border: "1px solid #e5e5e5",
              borderRadius: 20,
              padding: "4px 12px",
              fontSize: 12,
              cursor: "pointer",
              color: "#555",
            }}>
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{
        display: "flex",
        gap: 8,
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        overflow: "hidden",
        background: "#fff",
      }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="नेपालीमा प्रश्न सोध्नुहोस्..."
          disabled={loading}
          style={{
            flex: 1,
            padding: "14px 16px",
            border: "none",
            outline: "none",
            fontSize: 15,
            background: "transparent",
            color: "#0d0d0d",
          }}
        />
        <button
          onClick={() => send()}
          disabled={loading || !input.trim()}
          style={{
            padding: "0 20px",
            background: input.trim() ? "#0d0d0d" : "#f5f5f5",
            color: input.trim() ? "#fff" : "#aaa",
            border: "none",
            cursor: input.trim() ? "pointer" : "not-allowed",
            fontSize: 16,
            transition: "all 0.15s",
          }}
        >
          →
        </button>
      </div>
      <p style={{ fontSize: 11, color: "#aaa", marginTop: 8, textAlign: "center" }}>
        Press Enter to send · NepBPE-184M fine-tuned on Nepali Q&A
      </p>
    </div>
  )
}