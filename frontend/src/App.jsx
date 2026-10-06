import { useState } from 'react'
import { Activity, ArrowUpRight, CheckCircle2, CircleAlert, FileVideo, RotateCcw, Upload } from 'lucide-react'

const API = import.meta.env.VITE_API_URL || ''

function FaultCard({ fault }) {
  return <article className={`fault fault-${fault.severity}`}>
    <div className="fault-head">
      <span className="fault-joint">{fault.joint}</span>
      <span className="severity">{fault.severity}</span>
    </div>
    <p>{fault.description}</p>
    <small>{fault.phase} phase · frames {fault.frame_range?.[0]}–{fault.frame_range?.[1]}</small>
  </article>
}

function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function analyze() {
    if (!file) return
    setBusy(true)
    setError('')
    setResult(null)
    const form = new FormData()
    form.append('video', file)
    try {
      const response = await fetch(`${API}/analyze`, { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Analysis failed')
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  function reset() {
    setFile(null)
    setResult(null)
    setError('')
  }

  return <main>
    <header className="topbar">
      <div className="brand"><span className="brand-mark"><Activity size={19} /></span><span>FORM LAB</span></div>
      <span className="status"><span className="status-dot" /> analysis system online</span>
    </header>

    <section className="hero">
      <div className="eyebrow">SHOOTING BIOMECHANICS / 01</div>
      <h1>Make every release<br /><em>repeatable.</em></h1>
      <p className="lede">Upload a side-view shot to map your movement, compare it against your reference form, and see where the mechanics drift.</p>
    </section>

    {!result && <section className="workspace">
      <label className={`dropzone ${file ? 'has-file' : ''}`}>
        <input type="file" accept="video/*" onChange={e => setFile(e.target.files?.[0] || null)} />
        {file ? <><CheckCircle2 className="drop-icon ready" size={32} /><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(1)} MB · ready to analyze</span></> : <><Upload className="drop-icon" size={32} /><strong>Drop your shooting video here</strong><span>MP4, MOV, AVI or MKV · side view recommended</span></>}
      </label>
      {error && <div className="error"><CircleAlert size={18} />{error}</div>}
      <button className="primary" disabled={!file || busy} onClick={analyze}>{busy ? <><span className="spinner" /> Processing motion...</> : <><FileVideo size={18} /> Analyze shot <ArrowUpRight size={18} /></>}</button>
      {busy && <div className="progress"><div className="progress-bar" /><span>Extracting landmarks · measuring phases · rendering feedback</span></div>}
    </section>}

    {result && <section className="results">
      <div className="result-toolbar"><div><div className="eyebrow">ANALYSIS COMPLETE</div><h2>Your shot, decoded.</h2></div><button className="ghost" onClick={reset}><RotateCcw size={17} /> New analysis</button></div>
      <div className="stat-row"><div><span>FAULTS FOUND</span><strong>{result.faults.length}</strong></div><div><span>LOAD FRAME</span><strong>{result.phases.load}</strong></div><div><span>RELEASE FRAME</span><strong>{result.phases.release}</strong></div><div><span>FOLLOW-THROUGH</span><strong>{result.features.followthrough_frames}f</strong></div></div>
      <div className="video-grid"><div className="video-panel"><div className="panel-label">OVERLAY / FAULT MAP</div><video controls src={`${API}${result.artifacts.overlay_video}`} /></div><div className="video-panel"><div className="panel-label">CORRECTION / TARGET FORM</div><video controls src={`${API}${result.artifacts.correction_video}`} /></div></div>
      <div className="lower-grid"><section><div className="section-title"><span>01</span><h3>Movement notes</h3></div><div className="fault-list">{result.faults.length ? result.faults.map((fault, i) => <FaultCard fault={fault} key={i} />) : <div className="success"><CheckCircle2 /> No significant faults detected.</div>}</div></section><section className="feedback"><div className="section-title"><span>02</span><h3>Coach's read</h3></div><blockquote>{result.feedback}</blockquote></section></div>
      <section className="plots"><div className="section-title"><span>03</span><h3>Angle trajectories</h3></div><div className="plot-grid">{Object.entries(result.artifacts.plots).map(([name, path]) => <img key={name} src={`${API}${path}`} alt={`${name} angle trajectory`} />)}</div></section>
    </section>}
    <footer><span>FORM LAB / PRIVATE TRAINING ANALYTICS</span><span>BUILT FOR THE NEXT REP</span></footer>
  </main>
}

export default App
