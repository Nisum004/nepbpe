import { useState } from "react"
import axios from "axios"

const PROMPTS = ["नेपाल", "काठमाडौं", "हिमालयको", "नेपाली भाषा", "नेपाल सरकारले"]

export default function Generate() {
  const [prompt, setPrompt]   = useState("नेपाल")
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [temp, setTemp]       = useState(0.9)
  const [topK, setTopK]       = useState(50)
  const [maxTok, setMaxTok]   = useState(80)

  const generate = async () => {
    setLoading(true)
    setResult(null)
    try {
      const res = await axios.post("http://localhost:8000/generate", {
        prompt,
        temperature: temp,
        top_k: topK,
        max_tokens: maxTok,
        rep_penalty: 1.3,
      })
      setResult(res.data)
    } catch {
      alert("Backend error!")
    }
    setLoading(false)
  }

  return (
    <div>
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6, letterSpacing: "-0.3px" }}>
        Generate
      </h1>
      <p style={{ color: "#888", fontSize: 14, marginBottom: 28 }}>
        Generate Nepali text using NepBPE-184M.
      </p>

      {/* Prompt input */}
      <div style={{
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        overflow: "hidden",
        marginBottom: 16,
      }}>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Enter a Nepali prompt..."
          style={{
            width: "100%",
            padding: "16px 18px",
            border: "none",
            outline: "none",
            fontSize: 16,
            lineHeight: 1.6,
            resize: "vertical",
            minHeight: 80,
            background: "#fff",
            color: "#0d0d0d",
          }}
        />
        <div style={{
          borderTop: "1px solid #e5e5e5",
          padding: "10px 14px",
          display: "flex",
          justifyContent: "flex-end",
          background: "#fafafa",
        }}>
          <button onClick={generate} disabled={loading} style={{
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
            {loading ? "Generating..." : "Generate →"}
          </button>
        </div>
      </div>

      {/* Example prompts */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 28 }}>
        <span style={{ fontSize: 12, color: "#aaa", paddingTop: 4 }}>Try:</span>
        {PROMPTS.map(p => (
          <button key={p} onClick={() => setPrompt(p)} style={{
            background: "none",
            border: "1px solid #e5e5e5",
            borderRadius: 20,
            padding: "3px 12px",
            fontSize: 12,
            cursor: "pointer",
            color: "#555",
          }}>
            {p}
          </button>
        ))}
      </div>

      {/* Parameters */}
      <div style={{
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        padding: 20,
        marginBottom: 24,
      }}>
        <p style={{ fontSize: 13, fontWeight: 500, marginBottom: 16, color: "#888" }}>
          Parameters
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 24 }}>
          {[
            { label: "Temperature", value: temp, min: 0.1, max: 1.5, step: 0.1, set: setTemp },
            { label: "Top-K",       value: topK, min: 1,   max: 100, step: 1,   set: setTopK },
            { label: "Max tokens",  value: maxTok,min: 10, max: 200, step: 10,  set: setMaxTok },
          ].map((s, i) => (
            <div key={i}>
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 12,
                color: "#888",
                marginBottom: 8,
              }}>
                <span>{s.label}</span>
                <span style={{ color: "#0d0d0d", fontWeight: 500 }}>{s.value}</span>
              </div>
              <input
                type="range"
                min={s.min} max={s.max} step={s.step}
                value={s.value}
                onChange={e => s.set(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#0d0d0d" }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Output */}
      {result && (
        <div style={{
          border: "1px solid #e5e5e5",
          borderRadius: 10,
          padding: 24,
        }}>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 14,
          }}>
            <span style={{ fontSize: 13, fontWeight: 500 }}>Output</span>
            <span style={{
              fontSize: 12,
              color: "#888",
              background: "#f5f5f5",
              padding: "2px 8px",
              borderRadius: 20,
            }}>
              {result.tokens} tokens
            </span>
          </div>
          <p style={{ fontSize: 16, lineHeight: 1.8, color: "#0d0d0d" }}>
            {result.generated}
          </p>
        </div>
      )}
    </div>
  )
}