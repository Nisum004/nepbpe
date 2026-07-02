import { useEffect, useState } from "react"
import axios from "axios"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from "recharts"

export default function Benchmark() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get("https://nisum04-nepbpe-api.hf.space/benchmark")
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
  }, [])

  if (error) return <p style={{ color: "red", padding: 24 }}>Error: {error}</p>
  if (!data) return <p style={{ color: "#888", padding: 24 }}>Loading...</p>

  const chartData = Object.entries(data.details).map(([cat, sentences]) => {
    const avg = arr => Math.round(arr.reduce((a, b) => a + b, 0) / arr.length)
    return {
      category: cat,
      NepBPE:   avg(sentences.map(s => s.nepbpe)),
      "GPT-4o": avg(sentences.map(s => s.gpt4)),
      Llama3:   avg(sentences.map(s => s.llama3)),
    }
  })

  return (
    <div>
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6, letterSpacing: "-0.3px" }}>
        Benchmark
      </h1>
      <p style={{ color: "#888", fontSize: 14, marginBottom: 28 }}>
        Token efficiency comparison across Nepali text categories.
      </p>

      {/* Summary cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 32 }}>
        {[
          { label: "Fewer tokens vs GPT-4o", value: data.savings_percent.vs_gpt4o.toFixed(1) + "%" },
          { label: "Fewer tokens vs Llama3",  value: data.savings_percent.vs_llama3.toFixed(1) + "%" },
          { label: "Cost per 1M words",        value: "$0.81", sub: "vs $5.00 (GPT-4o)" },
        ].map((s, i) => (
          <div key={i} style={{
            border: "1px solid #e5e5e5",
            borderRadius: 8,
            padding: "18px 20px",
          }}>
            <div style={{ fontSize: 28, fontWeight: 600, letterSpacing: "-0.5px" }}>
              {s.value}
            </div>
            <div style={{ fontSize: 12, color: "#888", marginTop: 4 }}>{s.label}</div>
            {s.sub && <div style={{ fontSize: 11, color: "#aaa", marginTop: 2 }}>{s.sub}</div>}
          </div>
        ))}
      </div>

      {/* Chart */}
      <div style={{
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        padding: 24,
        marginBottom: 24,
      }}>
        <p style={{ fontWeight: 500, marginBottom: 20, fontSize: 14 }}>
          Average tokens per sentence by category
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} barCategoryGap="30%">
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
            <XAxis dataKey="category" stroke="#aaa" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis stroke="#aaa" fontSize={12} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{
                border: "1px solid #e5e5e5",
                borderRadius: 8,
                fontSize: 13,
                boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
              }}
              cursor={{ fill: "#f5f5f5" }}
            />
            <Legend iconType="square" iconSize={8} />
            <Bar dataKey="NepBPE"  fill="#0d0d0d" radius={[3,3,0,0]} />
            <Bar dataKey="GPT-4o"  fill="#d1d5db" radius={[3,3,0,0]} />
            <Bar dataKey="Llama3"  fill="#e5e7eb" radius={[3,3,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Table */}
      <div style={{ border: "1px solid #e5e5e5", borderRadius: 10, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#fafafa", borderBottom: "1px solid #e5e5e5" }}>
              {["Sentence", "NepBPE", "GPT-4o", "Llama3", "Savings"].map(h => (
                <th key={h} style={{
                  padding: "10px 16px",
                  textAlign: "left",
                  fontWeight: 500,
                  color: "#888",
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.values(data.details).flat().map((row, i) => {
              const saving = (((row.gpt4 - row.nepbpe) / row.gpt4) * 100).toFixed(1)
              return (
                <tr key={i} style={{
                  borderBottom: "1px solid #f0f0f0",
                  background: "#fff",
                }}>
                  <td style={{ padding: "10px 16px", maxWidth: 320, color: "#333" }}>
                    {row.sentence}
                  </td>
                  <td style={{ padding: "10px 16px", fontWeight: 600 }}>{row.nepbpe}</td>
                  <td style={{ padding: "10px 16px", color: "#888" }}>{row.gpt4}</td>
                  <td style={{ padding: "10px 16px", color: "#888" }}>{row.llama3}</td>
                  <td style={{ padding: "10px 16px", color: "#166534", fontWeight: 500 }}>
                    {saving}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}