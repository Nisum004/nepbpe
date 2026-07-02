import { useState, useCallback } from "react"
import axios from "axios"

// Soft pastel colors like OpenAI tokenizer
const COLORS = [
  { bg: "#fde8d8", border: "#f4a571" },
  { bg: "#d8f0fd", border: "#71bbf4" },
  { bg: "#d8fde8", border: "#71f4aa" },
  { bg: "#fdf0d8", border: "#f4d071" },
  { bg: "#ead8fd", border: "#b571f4" },
  { bg: "#fdd8ea", border: "#f471b5" },
  { bg: "#d8fdfa", border: "#71f0e8" },
  { bg: "#f0fdd8", border: "#c4f471" },
]

function Token({ token, id, colorIndex }) {
  const [hovered, setHovered] = useState(false)
  const c = COLORS[colorIndex % COLORS.length]

  return (
    <span
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "inline-block",
        background: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: 3,
        padding: "1px 5px",
        margin: "2px 2px",
        fontSize: 14,
        cursor: "default",
        position: "relative",
        whiteSpace: "pre",
        fontFamily: "'Inter', monospace",
        lineHeight: 1.6,
      }}
    >
      {token.replace(/^▁/, " ")}
      {hovered && (
        <span style={{
          position: "absolute",
          bottom: "calc(100% + 4px)",
          left: "50%",
          transform: "translateX(-50%)",
          background: "#0d0d0d",
          color: "#fff",
          fontSize: 11,
          padding: "3px 7px",
          borderRadius: 4,
          whiteSpace: "nowrap",
          zIndex: 10,
          pointerEvents: "none",
        }}>
          {id}
        </span>
      )}
    </span>
  )
}

function TokenRow({ label, data, tag }) {
  if (!data) return null
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        marginBottom: 10,
      }}>
        <span style={{ fontSize: 13, fontWeight: 500, color: "#0d0d0d" }}>
          {label}
        </span>
        <span style={{
          fontSize: 12,
          color: "#888",
          background: "#f5f5f5",
          padding: "2px 8px",
          borderRadius: 20,
        }}>
          {data.count} tokens
        </span>
        {tag && (
          <span style={{
            fontSize: 12,
            color: tag.color,
            background: tag.bg,
            padding: "2px 8px",
            borderRadius: 20,
            fontWeight: 500,
          }}>
            {tag.text}
          </span>
        )}
      </div>
      <div style={{
        background: "#fafafa",
        border: "1px solid #e5e5e5",
        borderRadius: 8,
        padding: "12px 14px",
        lineHeight: 2.2,
        minHeight: 52,
      }}>
        {data.tokens.map((tok, i) => (
          <Token key={i} token={tok} id={data.ids[i]} colorIndex={i} />
        ))}
      </div>
    </div>
  )
}

const EXAMPLES = [
  "नेपाली भाषा धेरै सुन्दर छ",
  "बिहान सबेरै उठेर तातो पानी पिएपछि, म सधैं आफ्नो कार्यालयको काम सुरु गर्नु अघि दैनिक समाचार पत्र पढ्छु।",
  "जब गर्मी मौसम सकिएर वर्षाको समय सुरु हुन्छ, तब नेपालका ग्रामीण क्षेत्रका किसानहरू आफ्नो खेतमा धान रोप्नका लागि हिलो टेक्दै काम गर्न व्यस्त हुन्छन्।",
  "मलाई हिजो साँझ मेरो साथीले एउटा महत्त्वपूर्ण कुरा भनेको थियो, तर मलाई त्यो कुरा अहिले राम्रोसँग याद भइरहेको छैन।",
]

export default function Tokenizer() {
  const [text, setText] = useState("नेपाली भाषा धेरै सुन्दर छ")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const tokenize = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await axios.post("https://nisum04-nepbpe-api.hf.space/tokenize", { text })
      setResult(res.data)
    } catch {
      alert("Could not reach backend at https://nisum04-nepbpe-api.hf.space")
    }
    setLoading(false)
  }, [text])

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) tokenize()
  }

  return (
    <div>
      {/* Header */}
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6, letterSpacing: "-0.3px" }}>
        Tokenizer
      </h1>
      <p style={{ color: "#888", fontSize: 14, marginBottom: 28 }}>
        Compare how NepBPE, GPT-4o, and Llama3 tokenize Nepali text.
        Hover over any token to see its ID.
      </p>

      {/* Input area */}
      <div style={{
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        overflow: "hidden",
        marginBottom: 16,
      }}>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Type Nepali text here..."
          style={{
            width: "100%",
            padding: "16px 18px",
            border: "none",
            outline: "none",
            fontSize: 16,
            lineHeight: 1.6,
            resize: "vertical",
            minHeight: 100,
            background: "#fff",
            color: "#0d0d0d",
          }}
        />
        <div style={{
          borderTop: "1px solid #e5e5e5",
          padding: "10px 14px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#fafafa",
        }}>
          <span style={{ fontSize: 12, color: "#aaa" }}>⌘ + Enter to tokenize</span>
          <button onClick={tokenize} disabled={loading} style={{
            background: "#0d0d0d",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            padding: "7px 16px",
            fontSize: 13,
            fontWeight: 500,
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.5 : 1,
          }}>
            {loading ? "Loading..." : "Tokenize"}
          </button>
        </div>
      </div>

      {/* Example pills */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 32 }}>
        <span style={{ fontSize: 12, color: "#aaa", paddingTop: 4 }}>Examples:</span>
        {EXAMPLES.map(ex => (
          <button key={ex} onClick={() => setText(ex)} style={{
            background: "none",
            border: "1px solid #e5e5e5",
            borderRadius: 20,
            padding: "3px 12px",
            fontSize: 12,
            cursor: "pointer",
            color: "#555",
            transition: "all 0.1s",
          }}>
            {ex.length > 20 ? ex.slice(0, 20) + "…" : ex}
          </button>
        ))}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Stats row */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 12,
            marginBottom: 28,
          }}>
            {[
              { label: "NepBPE tokens", value: result.nepbpe.count, highlight: true },
              { label: "GPT-4o tokens", value: result.gpt4o.count },
              { label: "Llama3 tokens", value: result.llama3.count },
              { label: "Saved vs GPT-4o", value: `${result.savings.vs_gpt4o}%` },
            ].map((s, i) => (
              <div key={i} style={{
                border: `1px solid ${s.highlight ? "#0d0d0d" : "#e5e5e5"}`,
                borderRadius: 8,
                padding: "14px 16px",
              }}>
                <div style={{
                  fontSize: 22,
                  fontWeight: 600,
                  letterSpacing: "-0.5px",
                  color: s.highlight ? "#0d0d0d" : "#333",
                }}>
                  {s.value}
                </div>
                <div style={{ fontSize: 12, color: "#888", marginTop: 3 }}>
                  {s.label}
                </div>
              </div>
            ))}
          </div>

          {/* Divider + label */}
          <div style={{ borderTop: "1px solid #e5e5e5", marginBottom: 24 }} />

          <TokenRow
            label="NepBPE"
            data={result.nepbpe}
            tag={{ text: "Our model", color: "#166534", bg: "#dcfce7" }}
          />
          <TokenRow
            label="GPT-4o"
            data={result.gpt4o}
            tag={result.savings.vs_gpt4o > 0
              ? { text: `${result.savings.vs_gpt4o}% more`, color: "#991b1b", bg: "#fee2e2" }
              : null}
          />
          <TokenRow
            label="Llama 3"
            data={result.llama3}
            tag={result.savings.vs_llama3 > 0
              ? { text: `${result.savings.vs_llama3}% more`, color: "#991b1b", bg: "#fee2e2" }
              : null}
          />
        </>
      )}
    </div>
  )
}