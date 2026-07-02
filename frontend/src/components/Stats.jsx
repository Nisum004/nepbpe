import { useEffect, useState } from "react"
import axios from "axios"

export default function Stats() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    axios.get("https://nisum04-nepbpe-api.hf.space/stats").then(r => setStats(r.data))
  }, [])

  if (!stats) return <p style={{ color: "#888" }}>Loading...</p>

  const rows = [
    { label: "Model",          value: stats.model },
    { label: "Parameters",     value: stats.parameters },
    { label: "Architecture",   value: stats.architecture },
    { label: "Vocabulary",     value: stats.vocab_size.toLocaleString() + " tokens" },
    { label: "Tokenizer",      value: stats.tokenizer },
    { label: "Corpus",         value: stats.corpus },
    { label: "vs GPT-4o",      value: stats.savings.vs_gpt4o + " fewer tokens" },
    { label: "vs Llama3",      value: stats.savings.vs_llama3 + " fewer tokens" },
  ]

  return (
    <div>
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6, letterSpacing: "-0.3px" }}>
        About
      </h1>
      <p style={{ color: "#888", fontSize: 14, marginBottom: 28 }}>
        NepBPE is a custom Nepali tokenizer and language model trained from scratch
        on Nepali Wikipedia. 
        Built as a final year project for CSIT 8th Semester by Nisum Yonghang and Prasamsha Pudasaini from St.Xavier's College.
      </p>

      {/* Stats table */}
      <div style={{
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        overflow: "hidden",
        marginBottom: 20,
      }}>
        {rows.map((r, i) => (
          <div key={i} style={{
            display: "flex",
            borderBottom: i < rows.length - 1 ? "1px solid #f0f0f0" : "none",
            padding: "13px 20px",
            background: "#fff",
          }}>
            <span style={{ width: 180, fontSize: 13, color: "#888", flexShrink: 0 }}>
              {r.label}
            </span>
            <span style={{ fontSize: 13, fontWeight: 500, color: "#0d0d0d" }}>
              {r.value}
            </span>
          </div>
        ))}
      </div>

      {/* HuggingFace */}
      <div style={{
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        padding: 20,
      }}>
        <p style={{ fontSize: 13, fontWeight: 500, marginBottom: 14, color: "#888" }}>
          Open Source
        </p>
        {Object.entries(stats.huggingface).map(([k, v]) => (
          <div key={k} style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            marginBottom: 10,
          }}>
            <span style={{ fontSize: 13, color: "#888", width: 80 }}>{k}</span>
            <a href={v} target="_blank" rel="noreferrer" style={{
              fontSize: 13,
              color: "#0d0d0d",
              textDecoration: "none",
              borderBottom: "1px solid #e5e5e5",
            }}>
              {v}
            </a>
          </div>
        ))}
      </div>
    </div>
  )
}