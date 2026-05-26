import { useState, useRef, useCallback, useEffect } from 'react'
import { create } from 'zustand'

// ─── Config ────────────────────────────────────────────────────────────────
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const SARVAM_KEY = import.meta.env.VITE_SARVAM_API_KEY ?? ''
const MAX_HIST = 40 // max points on charts

// ─── Zustand store ────────────────────────────────────────────────────────
const useStore = create((set, get) => ({
  isListening: false,
  error: null,
  segments: [], // [{text, isFiller}]
  fillerCounts: {}, // {word: count}
  totalFillers: 0,
  totalWords: 0,
  sessionSecs: 0,
  wpm: 0,
  fluencyScore: null,
  wpmHistory: [], // [{t, v}]
  fillerHistory: [], // [{t, v}]
  startTime: null,

  setListening: (v) =>
    set((s) => ({
      isListening: v,
      startTime: v && !s.startTime ? Date.now() : s.startTime,
    })),
  setError: (e) => set({ error: e }),

  addChunk: (transcript, segments, fillerCounts) => {
    const s = get()
    const now = Date.now()
    const secs = s.startTime ? (now - s.startTime) / 1000 : 1

    // Merge segments
    const newSegs =
      s.segments.length
        ? [...s.segments, { text: ' ', isFiller: false }, ...segments]
        : segments

    // Merge filler counts
    const merged = { ...s.fillerCounts }
    for (const [w, c] of Object.entries(fillerCounts)) {
      merged[w] = (merged[w] ?? 0) + c
    }
    const totalFillers = Object.values(merged).reduce((a, b) => a + b, 0)

    // Word count from segments
    const fullText = newSegs.map((seg) => seg.text).join('')
    const totalWords = fullText.trim().split(/\s+/).filter(Boolean).length

    // WPM
    const wpm = secs > 5 ? Math.round((totalWords / secs) * 60) : 0

    // Fluency score 0–100: penalise fillers & low/high WPM
    const fillerRate = totalWords > 0 ? totalFillers / totalWords : 0
    const wpmScore =
      wpm >= 100 && wpm <= 160 ? 100 : wpm < 100 ? wpm : Math.max(0, 200 - wpm)
    const fillerPenalty = Math.min(100, fillerRate * 500)
    const fluencyScore = Math.round(Math.max(0, wpmScore - fillerPenalty))

    const t = Math.round(secs)
    set({
      segments: newSegs,
      fillerCounts: merged,
      totalFillers,
      totalWords,
      sessionSecs: secs,
      wpm,
      fluencyScore,
      wpmHistory: [...s.wpmHistory, { t, v: wpm }].slice(-MAX_HIST),
      fillerHistory: [...s.fillerHistory, { t, v: totalFillers }].slice(
        -MAX_HIST
      ),
    })
  },

  tick: () => {
    const s = get()
    if (!s.isListening || !s.startTime) return
    const secs = (Date.now() - s.startTime) / 1000
    const wpm =
      secs > 5
        ? Math.round((s.totalWords / secs) * 60)
        : s.wpm
    set({
      sessionSecs: secs,
      wpm,
      wpmHistory: [...s.wpmHistory, { t: Math.round(secs), v: wpm }].slice(
        -MAX_HIST
      ),
    })
  },

  reset: () =>
    set({
      isListening: false,
      error: null,
      segments: [],
      fillerCounts: {},
      totalFillers: 0,
      totalWords: 0,
      sessionSecs: 0,
      wpm: 0,
      fluencyScore: null,
      wpmHistory: [],
      fillerHistory: [],
      startTime: null,
    }),
}))

// ─── useAudioCapture hook (Live Recording) ─────────────────────────────────
function useAudioCapture() {
  const recorderRef = useRef(null)
  const allBlobsRef = useRef([])
  const tickTimerRef = useRef(null)
  const streamRef = useRef(null)
  const isRunningRef = useRef(false)
  const [isPosting, setIsPosting] = useState(false)

  const { setListening, setError, addChunk, tick } = useStore()

  const postFull = useCallback(
    async (blobs) => {
      if (!blobs.length) return
      setIsPosting(true)
      const blob = new Blob(blobs, { type: 'audio/webm' })
      const form = new FormData()
      form.append('file', blob, 'recording.webm')
      form.append('api_key', SARVAM_KEY)
      try {
        const res = await fetch(`${API_URL}/transcribe`, {
          method: 'POST',
          body: form,
        })
        const data = await res.json()
        if (data.error) {
          setError(data.error)
          return
        }
        if (data.transcript)
          addChunk(
            data.transcript,
            data.segments ?? [],
            data.fillerCounts ?? {}
          )
      } catch (e) {
        setError(`Network error: ${e.message}`)
      } finally {
        setIsPosting(false)
      }
    },
    [addChunk, setError]
  )

  const start = useCallback(async () => {
    if (isRunningRef.current) return
    if (!SARVAM_KEY) {
      setError('VITE_SARVAM_API_KEY missing in .env.local')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      isRunningRef.current = true
      allBlobsRef.current = []

      const rec = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      })

      rec.ondataavailable = (e) => {
        if (e.data.size > 0) allBlobsRef.current.push(e.data)
      }

      rec.onstop = () => postFull(allBlobsRef.current)

      rec.start(1000)
      recorderRef.current = rec

      setListening(true)
      setError(null)
      tickTimerRef.current = setInterval(tick, 1000)
    } catch (e) {
      setError(`Mic error: ${e.message}`)
    }
  }, [postFull, setListening, setError, tick])

  const stop = useCallback(() => {
    isRunningRef.current = false
    clearInterval(tickTimerRef.current)
    if (recorderRef.current?.state === 'recording')
      recorderRef.current.stop()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setListening(false)
  }, [setListening])

  useEffect(() => () => {
    isRunningRef.current = false
    stop()
  }, [stop])

  return { start, stop, isPosting }
}

// ─── useFileUpload hook ───────────────────────────────────────────────────
function useFileUpload() {
  const fileInputRef = useRef(null)
  const [isPosting, setIsPosting] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  const { setError, addChunk, setListening } = useStore()

  const uploadFile = useCallback(
    async (file) => {
      if (!file) return
      if (!SARVAM_KEY) {
        setError('VITE_SARVAM_API_KEY missing in .env.local')
        return
      }

      // Validate file type
      const validTypes = [
        'audio/mpeg',
        'audio/mp3',
        'audio/wav',
        'audio/webm',
        'audio/ogg',
      ]
      if (!validTypes.includes(file.type)) {
        setError(`Unsupported file type: ${file.type}. Use MP3, WAV, WebM, or OGG.`)
        return
      }

      // Validate file size (max 25MB)
      if (file.size > 25 * 1024 * 1024) {
        setError('File too large. Max 25MB.')
        return
      }

      setIsPosting(true)
      setUploadProgress(0)
      setError(null)

      const form = new FormData()
      form.append('file', file)
      form.append('api_key', SARVAM_KEY)

      try {
        const res = await fetch(`${API_URL}/transcribe`, {
          method: 'POST',
          body: form,
        })
        const data = await res.json()
        if (data.error) {
          setError(data.error)
          return
        }
        if (data.transcript) {
          addChunk(
            data.transcript,
            data.segments ?? [],
            data.fillerCounts ?? {}
          )
          setListening(false)
        }
      } catch (e) {
        setError(`Upload error: ${e.message}`)
      } finally {
        setIsPosting(false)
        setUploadProgress(0)
      }
    },
    [addChunk, setError, setListening]
  )

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) uploadFile(file)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const triggerFileInput = () => fileInputRef.current?.click()

  return { triggerFileInput, handleFileChange, isPosting, uploadProgress, fileInputRef }
}

// ─── Helpers ──────────────────────────────────────────────────────────────
function fmt(s) {
  s = Math.floor(s)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(
    s % 60
  ).padStart(2, '0')}`
}

function scoreColor(score) {
  if (score === null) return '#94a3b8'
  if (score >= 75) return '#22c55e'
  if (score >= 50) return '#f59e0b'
  return '#ef4444'
}

// ─── Tab Switcher ─────────────────────────────────────────────────────────
function TabSwitcher({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'live', label: '🎙 Live', icon: '🎤' },
    { id: 'upload', label: '📤 Upload', icon: '📁' },
  ]

  return (
    <div
      style={{
        display: 'flex',
        gap: 8,
        marginBottom: 20,
        borderBottom: '1px solid #e2e8f0',
      }}
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          style={{
            padding: '12px 16px',
            fontSize: 14,
            fontWeight: 600,
            border: 'none',
            background: 'transparent',
            color: activeTab === tab.id ? '#2563eb' : '#94a3b8',
            borderBottom:
              activeTab === tab.id ? '2px solid #2563eb' : 'none',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

// ─── Live Recording Controls ──────────────────────────────────────────────
function LiveControls() {
  const { isListening, totalWords, sessionSecs, reset } = useStore()
  const { start, stop, isPosting } = useAudioCapture()

  const handleToggle = () => (isListening ? stop() : start())
  const handleReset = () => {
    stop()
    setTimeout(reset, 100)
  }

  return (
    <div style={card}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 24,
          flexWrap: 'wrap',
        }}
      >
        {/* Timer */}
        <div style={{ textAlign: 'center', minWidth: 64 }}>
          <div
            style={{
              fontSize: 26,
              fontWeight: 700,
              color: '#0f172a',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {fmt(sessionSecs)}
          </div>
          <div style={label}>TIME</div>
        </div>

        {/* Words */}
        <div style={{ textAlign: 'center', minWidth: 48 }}>
          <div style={{ fontSize: 26, fontWeight: 700, color: '#0f172a' }}>
            {totalWords}
          </div>
          <div style={label}>WORDS</div>
        </div>

        {/* Analysing indicator */}
        {isPosting && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 7,
              color: '#f59e0b',
              fontSize: 13,
              fontWeight: 500,
            }}
          >
            <span
              style={{
                animation: 'spin 1s linear infinite',
                display: 'inline-block',
              }}
            >
              ⏳
            </span>
            Analysing…
          </div>
        )}

        <div style={{ flex: 1 }} />

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={handleToggle}
            disabled={isPosting}
            style={{
              ...btn,
              background: isListening ? '#fee2e2' : '#dbeafe',
              color: isListening ? '#dc2626' : '#2563eb',
              opacity: isPosting ? 0.5 : 1,
              cursor: isPosting ? 'not-allowed' : 'pointer',
            }}
          >
            <span
              style={{
                display: 'inline-block',
                width: 8,
                height: 8,
                borderRadius: isListening ? 2 : '50%',
                background: isListening ? '#dc2626' : '#22c55e',
                marginRight: 7,
              }}
            />
            {isListening ? 'Stop' : 'Start'}
          </button>
          <button
            onClick={handleReset}
            disabled={isPosting}
            style={{
              ...btn,
              background: '#f1f5f9',
              color: '#475569',
              opacity: isPosting ? 0.5 : 1,
              cursor: isPosting ? 'not-allowed' : 'pointer',
            }}
          >
            Reset
          </button>
        </div>
      </div>

      <ErrorBanner />
    </div>
  )
}

// ─── File Upload Controls ─────────────────────────────────────────────────
function FileUploadControls() {
  const { reset, totalWords, totalFillers } = useStore()
  const { triggerFileInput, handleFileChange, isPosting, fileInputRef } =
    useFileUpload()

  const handleReset = () => {
    reset()
  }

  return (
    <div style={card}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 24,
          flexWrap: 'wrap',
        }}
      >
        {/* File Stats */}
        <div style={{ textAlign: 'center', minWidth: 64 }}>
          <div
            style={{
              fontSize: 26,
              fontWeight: 700,
              color: '#0f172a',
            }}
          >
            {totalWords}
          </div>
          <div style={label}>WORDS</div>
        </div>

        <div style={{ textAlign: 'center', minWidth: 64 }}>
          <div
            style={{
              fontSize: 26,
              fontWeight: 700,
              color: '#dc2626',
            }}
          >
            {totalFillers}
          </div>
          <div style={label}>FILLERS</div>
        </div>

        {/* Uploading indicator */}
        {isPosting && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 7,
              color: '#f59e0b',
              fontSize: 13,
              fontWeight: 500,
            }}
          >
            <span
              style={{
                animation: 'spin 1s linear infinite',
                display: 'inline-block',
              }}
            >
              ⏳
            </span>
            Processing…
          </div>
        )}

        <div style={{ flex: 1 }} />

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={triggerFileInput}
            disabled={isPosting}
            style={{
              ...btn,
              background: '#dbeafe',
              color: '#2563eb',
              opacity: isPosting ? 0.5 : 1,
              cursor: isPosting ? 'not-allowed' : 'pointer',
            }}
          >
            📁 Choose File
          </button>
          <button
            onClick={handleReset}
            disabled={isPosting}
            style={{
              ...btn,
              background: '#f1f5f9',
              color: '#475569',
              opacity: isPosting ? 0.5 : 1,
              cursor: isPosting ? 'not-allowed' : 'pointer',
            }}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      <ErrorBanner />
    </div>
  )
}

// ─── Transcript ────────────────────────────────────────────────────────────
function Transcript() {
  const { segments, isListening } = useStore()
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [segments])

  const hasContent = segments.length > 0

  return (
    <div style={{ ...card, minHeight: 160 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 600, fontSize: 14, color: '#0f172a' }}>
            Transcript
          </span>
          {isListening && (
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: '#ef4444',
                display: 'inline-block',
                animation: 'pulse 1.5s ease-in-out infinite',
              }}
            />
          )}
        </div>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>
          <span style={{ color: '#ef4444', marginRight: 4 }}>■</span>filler
          words
        </span>
      </div>

      <div
        style={{
          fontSize: 15,
          lineHeight: 1.75,
          color: '#1e293b',
          minHeight: 80,
        }}
      >
        {!hasContent ? (
          <div
            style={{
              color: '#cbd5e1',
              fontSize: 14,
              paddingTop: 20,
              textAlign: 'center',
            }}
          >
            {isListening ? '🎙 Listening… start speaking' : '⏳ Waiting for content'}
          </div>
        ) : (
          segments.map((seg, i) =>
            seg.isFiller ? (
              <mark
                key={i}
                style={{
                  background: '#fee2e2',
                  color: '#dc2626',
                  padding: '1px 3px',
                  borderRadius: 3,
                  fontWeight: 600,
                }}
              >
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

// ─── Filler Stats ──────────────────────────────────────────────────────────
function FillerStats() {
  const { fillerCounts, totalFillers, totalWords } = useStore()

  if (!fillerCounts || Object.keys(fillerCounts).length === 0) {
    return null
  }

  const entries = Object.entries(fillerCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)

  return (
    <div style={card}>
      <div style={{ fontWeight: 600, fontSize: 14, color: '#0f172a', marginBottom: 14 }}>
        Top Fillers
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {entries.map(([word, count]) => (
          <div
            key={word}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 12px',
              background: '#f1f5f9',
              borderRadius: 8,
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 500, color: '#1e293b' }}>
              {word}
            </span>
            <span
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: '#dc2626',
                background: '#fee2e2',
                padding: '2px 8px',
                borderRadius: 4,
              }}
            >
              {count}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Error banner ──────────────────────────────────────────────────────────
function ErrorBanner() {
  const error = useStore((s) => s.error)
  if (!error) return null
  return (
    <div
      style={{
        marginTop: 12,
        padding: '10px 14px',
        background: '#fef2f2',
        border: '1px solid #fecaca',
        borderRadius: 8,
        color: '#dc2626',
        fontSize: 13,
      }}
    >
      {error}
    </div>
  )
}

// ─── Shared styles ────────────────────────────────────────────────────────
const card = {
  background: '#fff',
  border: '1px solid #e2e8f0',
  borderRadius: 12,
  padding: '18px 20px',
  boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
}
const btn = {
  padding: '9px 18px',
  fontSize: 14,
  fontWeight: 600,
  border: 'none',
  borderRadius: 8,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  transition: 'opacity 0.15s',
}
const label = {
  fontSize: 10,
  color: '#94a3b8',
  letterSpacing: '0.08em',
  marginTop: 3,
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('live')

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#f8fafc',
        fontFamily: '"DM Sans", system-ui, sans-serif',
        padding: '32px 16px',
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        button:hover:not(:disabled) { opacity: 0.85; }
      `}</style>

      <div style={{ maxWidth: 720, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <h1
            style={{
              fontSize: 28,
              fontWeight: 700,
              color: '#0f172a',
              letterSpacing: '-0.02em',
            }}
          >
            Filler Awareness
          </h1>
          <p style={{ fontSize: 13, color: '#94a3b8', marginTop: 3 }}>
            Analyze speech fluency — upload files or record live
          </p>
        </div>

        {/* Tab Switcher */}
        <TabSwitcher activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Stack */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {activeTab === 'live' ? (
            <>
              <LiveControls />
              <Transcript />
            </>
          ) : (
            <>
              <FileUploadControls />
              <Transcript />
              <FillerStats />
            </>
          )}
        </div>
      </div>
    </div>
  )
}