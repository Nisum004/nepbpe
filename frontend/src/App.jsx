import { useState } from "react"
import Tokenizer from "./components/Tokenizer"
import Benchmark from "./components/Benchmark"
import Generate from "./components/Generate"
import Stats from "./components/Stats"
import Chat from "./components/Chat"

const TABS = ["Tokenizer", "Benchmark", "Generate", "Chat", "About"]

export default function App() {
  const [tab, setTab] = useState(0)

  return (
    <div style={{ minHeight: "100vh", background: "#fff" }}>

      <header style={{
        borderBottom: "1px solid #e5e5e5",
        padding: "0 40px",
        display: "flex",
        alignItems: "center",
        height: 56,
        position: "sticky",
        top: 0,
        background: "#fff",
        zIndex: 100,
      }}>

        <span style={{ fontWeight: 600, fontSize: 15, letterSpacing: "-0.2px" }}>
          NepBPE
        </span>

        <span style={{
          marginLeft: 10,
          background: "#f0f0f0",
          color: "#555",
          fontSize: 11,
          fontWeight: 500,
          padding: "2px 8px",
          borderRadius: 20,
        }}>
          184M
        </span>

        <nav style={{ display: "flex", gap: 4, marginLeft: 40 }}>
          {TABS.map((t, i) => (
            <button
              key={i}
              onClick={() => setTab(i)}
              style={{
                background: tab === i ? "#f5f5f5" : "none",
                border: "none",
                cursor: "pointer",
                padding: "6px 12px",
                fontSize: 14,
                fontWeight: tab === i ? 500 : 400,
                color: tab === i ? "#0d0d0d" : "#888",
                borderRadius: 6,
                transition: "all 0.1s",
              }}
            >
              {t}
            </button>
          ))}
        </nav>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ fontSize: 13, color: "#888" }}>
            83.9% fewer tokens than GPT-4o
          </span>
          <a
            href="https://huggingface.co/nisum04/nepbpe-184m"
            target="_blank"
            rel="noreferrer"
            style={{
              fontSize: 13,
              color: "#0d0d0d",
              textDecoration: "none",
              border: "1px solid #e5e5e5",
              padding: "5px 12px",
              borderRadius: 6,
              fontWeight: 500,
            }}
          >
            HuggingFace ↗
          </a>
        </div>

      </header>

      <main style={{ maxWidth: 900, margin: "0 auto", padding: "40px 24px" }}>
        {tab === 0 && <Tokenizer />}
        {tab === 1 && <Benchmark />}
        {tab === 2 && <Generate />}
        {tab === 3 && <Chat />}
        {tab === 4 && <Stats />}
      </main>

    </div>
  )
}