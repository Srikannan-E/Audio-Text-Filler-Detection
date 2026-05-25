import { useState, useRef, useCallback, useEffect } from 'react'
import { create } from 'zustand'

// ─── Config ────────────────────────────────────────────────────────────────
const API_URL      = import.meta.env.VITE_API_URL      ?? 'http://localhost:8001'
const SARVAM_KEY   = import.meta.env.VITE_SARVAM_API_KEY ?? ''
const CHUNK_MS     = 25000   // 25 s per audio chunk — under Sarvam free-tier 28 s cap
const MAX_HIST     = 40      // max points on charts

// ─── Zustand store ─────────────────────────────────────────────────────────
const useStore = create((set, get) => ({
  isListening:    false,
  error:          null,
  segments:       [],          // [{text, isFiller}]
  fillerCounts:   {},          // {word: count}
  totalFillers:   0,
  totalWords:     0,
  sessionSecs:    0,
  wpm:            0,
  fluencyScore:   null,
  wpmHistory:     [],          // [{t, v}]
  fillerHistory:  [],          // [{t, v}]
  startTime:      null,

  setListening: (v) => set(s => ({
    isListening: v,
    startTime:   v && !s.startTime ? Date.now() : s.startTime,
  })),
  setError: (e) => set({ error: e }),

  addChunk: (transcript, segments, fillerCounts) => {
    const s = get()
    const now     = Date.now()
    const secs    = s.startTime ? (now - s.startTime) / 1000 : 1

    // Merge segments
    const newSegs = s.segments.length
      ? [...s.segments, { text: ' ', isFiller: false }, ...segments]
      : segments

    // Merge filler counts
    const merged = { ...s.fillerCounts }
    for (const [w, c] of Object.entries(fillerCounts)) {
      merged[w] = (merged[w] ?? 0) + c
    }
    const totalFillers = Object.values(merged).reduce((a, b) => a + b, 0)

    // Word count from segments
    const fullText  = newSegs.map(seg => seg.text).join('')
    const totalWords = fullText.trim().split(/\s+/).filter(Boolean).length

    // WPM
    const wpm = secs > 5 ? Math.round((totalWords / secs) * 60) : 0

    // Fluency score 0–100: penalise fillers & low/high WPM
    const fillerRate   = totalWords > 0 ? totalFillers / totalWords : 0
    const wpmScore     = wpm >= 100 && wpm <= 160 ? 100 : wpm < 100 ? wpm : Math.max(0, 200 - wpm)
    const fillerPenalty = Math.min(100, fillerRate * 500)
    const fluencyScore  = Math.round(Math.max(0, wpmScore - fillerPenalty))

    const t = Math.round(secs)
    set({
      segments:     newSegs,
      fillerCounts: merged,
      totalFillers,
      totalWords,
      sessionSecs:  secs,
      wpm,
      fluencyScore,
      wpmHistory:    [...s.wpmHistory,    { t, v: wpm        }].slice(-MAX_HIST),
      fillerHistory: [...s.fillerHistory, { t, v: totalFillers }].slice(-MAX_HIST),
    })
  },

  tick: () => {
    const s = get()
    if (!s.isListening || !s.startTime) return
    const secs = (Date.now() - s.startTime) / 1000
    const wpm  = secs > 5
      ? Math.round((s.totalWords / secs) * 60)
      : s.wpm
    set({
      sessionSecs: secs,
      wpm,
      wpmHistory: [...s.wpmHistory, { t: Math.round(secs), v: wpm }].slice(-MAX_HIST),
    })
  },

  reset: () => set({
    isListening: false, error: null,
    segments: [], fillerCounts: {}, totalFillers: 0,
    totalWords: 0, sessionSecs: 0, wpm: 0, fluencyScore: null,
    wpmHistory: [], fillerHistory: [], startTime: null,
  }),
}))

// ─── useAudioCapture hook ───────────────────────────────────────────────────
function useAudioCapture() {
  const recorderRef      = useRef(null)
  const chunksRef        = useRef([])
  const flushTimerRef    = useRef(null)
  const tickTimerRef     = useRef(null)
  const streamRef        = useRef(null)
  const isRunningRef     = useRef(false)
  const chunkStartRef    = useRef(null)   // for countdown ring

  const { setListening, setError, addChunk, tick, reset } = useStore()

  const postChunk = useCallback(async (blob) => {
    if (!blob || blob.size < 500) return
    const form = new FormData()
    form.append('file',    blob, 'chunk.webm')
    form.append('api_key', SARVAM_KEY)
    try {
      const res  = await fetch(`${API_URL}/transcribe`, { method: 'POST', body: form })
      const data = await res.json()
      if (data.error) { setError(data.error); return }
      if (data.transcript) addChunk(data.transcript, data.segments ?? [], data.fillerCounts ?? {})
    } catch (e) {
      setError(`Network error: ${e.message}`)
    }
  }, [addChunk, setError])

  const startRecorder = useCallback(() => {
    if (!streamRef.current) return
    const rec = new MediaRecorder(streamRef.current, { mimeType: 'audio/webm;codecs=opus' })
    chunksRef.current = []
    rec.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      chunksRef.current = []
      postChunk(blob)
      chunkStartRef.current = Date.now()
      if (isRunningRef.current) startRecorder()
    }
    rec.start()
    recorderRef.current  = rec
    chunkStartRef.current = Date.now()
  }, [postChunk])

  const start = useCallback(async () => {
    if (isRunningRef.current) return
    if (!SARVAM_KEY) { setError('VITE_SARVAM_API_KEY missing in .env.local'); return }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current    = stream
      isRunningRef.current = true
      setListening(true)
      setError(null)
      startRecorder()
      flushTimerRef.current = setInterval(() => {
        if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
      }, CHUNK_MS)
      tickTimerRef.current = setInterval(tick, 1000)
    } catch (e) {
      setError(`Mic error: ${e.message}`)
    }
  }, [startRecorder, setListening, setError, tick])

  const stop = useCallback(() => {
    isRunningRef.current = false
    clearInterval(flushTimerRef.current)
    clearInterval(tickTimerRef.current)
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setListening(false)
  }, [setListening])

  useEffect(() => () => { isRunningRef.current = false; stop() }, [stop])

  return { start, stop, chunkStartRef }
}

// ─── Helpers ────────────────────────────────────────────────────────────────
function fmt(s) {
  s = Math.floor(s)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

function scoreColor(score) {
  if (score === null) return '#94a3b8'
  if (score >= 75) return '#22c55e'
  if (score >= 50) return '#f59e0b'
  return '#ef4444'
}

// ─── CountdownRing ──────────────────────────────────────────────────────────
function CountdownRing({ chunkStartRef }) {
  const [left, setLeft] = useState(CHUNK_MS / 1000)
  useEffect(() => {
    const id = setInterval(() => {
      if (!chunkStartRef.current) return
      const elapsed = (Date.now() - chunkStartRef.current) / 1000
      setLeft(Math.max(0, Math.round(CHUNK_MS / 1000 - elapsed)))
    }, 500)
    return () => clearInterval(id)
  }, [chunkStartRef])

  const total  = CHUNK_MS / 1000
  const r      = 16
  const circ   = 2 * Math.PI * r
  const offset = circ * (1 - left / total)
  const color  = left > 10 ? '#22c55e' : left > 5 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
      <svg width={40} height={40} viewBox="0 0 40 40">
        <circle cx={20} cy={20} r={r} fill="none" stroke="#e2e8f0" strokeWidth={3}/>
        <circle cx={20} cy={20} r={r} fill="none" stroke={color} strokeWidth={3}
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          transform="rotate(-90 20 20)"
          style={{ transition: 'stroke-dashoffset 0.5s linear, stroke 0.5s' }}/>
        <text x={20} y={24} textAnchor="middle" fontSize={11} fontWeight={700} fill={color}>{left}</text>
      </svg>
      <span style={{ fontSize: 10, color: '#94a3b8', letterSpacing: '0.05em' }}>
        {left <= 3 ? 'SENDING' : 'CHUNK'}
      </span>
    </div>
  )
}

// ─── Controls ───────────────────────────────────────────────────────────────
function Controls() {
  const { isListening, totalWords, sessionSecs, reset } = useStore()
  const { start, stop, chunkStartRef } = useAudioCapture()

  const handleToggle = () => isListening ? stop() : start()
  const handleReset  = () => { stop(); setTimeout(reset, 100) }

  return (
    <div style={card}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>

        {/* Timer */}
        <div style={{ textAlign: 'center', minWidth: 64 }}>
          <div style={{ fontSize: 26, fontWeight: 700, color: '#0f172a', fontVariantNumeric: 'tabular-nums' }}>
            {fmt(sessionSecs)}
          </div>
          <div style={label}>TIME</div>
        </div>

        {/* Words */}
        <div style={{ textAlign: 'center', minWidth: 48 }}>
          <div style={{ fontSize: 26, fontWeight: 700, color: '#0f172a' }}>{totalWords}</div>
          <div style={label}>WORDS</div>
        </div>

        {/* Countdown ring — only while listening */}
        {isListening && <CountdownRing chunkStartRef={chunkStartRef} />}

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={handleToggle} style={{
            ...btn,
            background: isListening ? '#fee2e2' : '#dbeafe',
            color:      isListening ? '#dc2626' : '#2563eb',
          }}>
            <span style={{
              display: 'inline-block', width: 8, height: 8,
              borderRadius: isListening ? 2 : '50%',
              background: isListening ? '#dc2626' : '#22c55e',
              marginRight: 7,
            }}/>
            {isListening ? 'Stop' : 'Start'}
          </button>
          <button onClick={handleReset} style={{ ...btn, background: '#f1f5f9', color: '#475569' }}>
            Reset
          </button>
        </div>
      </div>

      {/* Error banner */}
      <ErrorBanner />
    </div>
  )
}

// ─── Error banner ────────────────────────────────────────────────────────────
function ErrorBanner() {
  const error = useStore(s => s.error)
  if (!error) return null
  return (
    <div style={{
      marginTop: 12, padding: '10px 14px',
      background: '#fef2f2', border: '1px solid #fecaca',
      borderRadius: 8, color: '#dc2626', fontSize: 13,
    }}>
      {error}
    </div>
  )
}

// ─── Transcript ──────────────────────────────────────────────────────────────
function Transcript() {
  const { segments, isListening } = useStore()
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [segments])

  const hasContent = segments.length > 0

  return (
    <div style={{ ...card, minHeight: 160 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 600, fontSize: 14, color: '#0f172a' }}>Transcript</span>
          {isListening && (
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: '#ef4444', display: 'inline-block',
              animation: 'pulse 1.5s ease-in-out infinite',
            }}/>
          )}
        </div>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>
          <span style={{ color: '#ef4444', marginRight: 4 }}>■</span>filler words
        </span>
      </div>

      <div style={{ fontSize: 15, lineHeight: 1.75, color: '#1e293b', minHeight: 80 }}>
        {!hasContent ? (
          <div style={{ color: '#cbd5e1', fontSize: 14, paddingTop: 20, textAlign: 'center' }}>
            {isListening ? '🎙 Listening… start speaking' : 'Press Start to begin'}
          </div>
        ) : (
          segments.map((seg, i) =>
            seg.isFiller ? (
              <mark key={i} style={{
                background: '#fee2e2', color: '#dc2626',
                padding: '1px 3px', borderRadius: 3,
                fontWeight: 600,
              }}>
                {seg.text}
              </mark>
            ) : (
              <span key={i}>{seg.text}</span>
            )
          )
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

// ─── Shared styles ────────────────────────────────────────────────────────────
const card = {
  background: '#fff',
  border: '1px solid #e2e8f0',
  borderRadius: 12,
  padding: '18px 20px',
  boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
}
const btn = {
  padding: '9px 18px',
  fontSize: 14, fontWeight: 600,
  border: 'none', borderRadius: 8,
  cursor: 'pointer',
  display: 'flex', alignItems: 'center',
  transition: 'opacity 0.15s',
}
const label  = { fontSize: 10, color: '#94a3b8', letterSpacing: '0.08em', marginTop: 3 }
const label2 = { fontSize: 10, color: '#94a3b8', letterSpacing: '0.08em', marginTop: 5 }

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#f8fafc',
      fontFamily: '"DM Sans", system-ui, sans-serif',
      padding: '32px 16px',
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
        button:hover { opacity: 0.85; }
      `}</style>

      <div style={{ maxWidth: 720, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', letterSpacing: '-0.02em' }}>
            Filler Awareness
          </h1>
          <p style={{ fontSize: 13, color: '#94a3b8', marginTop: 3 }}>
            Speak naturally — filler words are highlighted in real time
          </p>
        </div>

        {/* Stack */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Controls />
          <Transcript />
        </div>

      </div>
    </div>
  )
}