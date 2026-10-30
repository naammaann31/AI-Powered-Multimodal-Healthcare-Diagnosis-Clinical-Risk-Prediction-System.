import { useState } from 'react'
import axios from 'axios'
import './App.css'
import Auth from './Auth'
import glsLogo from './assets/gls-logo.jpg'
import {
  LogOut, User, Image as ImageIcon, Droplet, Wind, Brain, Microscope,
  Search, Lock, FileText, Flame, Loader2, AlertTriangle, CheckCircle2, Circle,
  AlertOctagon, AlertCircle,
} from 'lucide-react'

const MODULE_ICON = { xray: Wind, brain: Brain, skin: Microscope }

const API = 'http://127.0.0.1:8000'

// Single source of truth for the vitals form's default shape (21 BRFSS fields) — used
// both for initial state and on reset, so they can never drift apart again.
const DEFAULT_VITALS = {
  high_bp: '0', high_chol: '0', chol_check: '1', bmi: '', smoker: '0', stroke: '0',
  heart_disease_or_attack: '0', phys_activity: '1', fruits: '1', veggies: '1',
  hvy_alcohol_consump: '0', any_healthcare: '1', no_doc_bc_cost: '0', gen_hlth: '3',
  ment_hlth: '0', phys_hlth: '0', diff_walk: '0', sex: '1', age: '', education: '4', income: '5'
}

export default function App() {
  // Persist the logged-in user across page refreshes — a refresh shouldn't
  // silently sign the user out, since the backend session itself has no expiry.
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('mediai_user')
      return saved ? JSON.parse(saved) : null
    } catch {
      return null
    }
  })

  const login = (u) => {
    setUser(u)
    try { localStorage.setItem('mediai_user', JSON.stringify(u)) } catch { /* storage unavailable */ }
  }

  const logout = () => {
    setUser(null)
    try { localStorage.removeItem('mediai_user') } catch { /* storage unavailable */ }
  }

  const [module, setModule] = useState('xray')
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)

  const [patient, setPatient] = useState({ name: '', age: '', gender: 'Male', id: '' })
  const [vitals, setVitals] = useState(DEFAULT_VITALS)

  const [xrayResult, setXrayResult] = useState(null)
  const [vitalsResult, setVitalsResult] = useState(null)
  const [brainResult, setBrainResult] = useState(null)
  const [skinResult, setSkinResult] = useState(null)
  const [guidance, setGuidance] = useState("")
  const [xrayHeatmap, setXrayHeatmap] = useState(null)
  const [brainHeatmap, setBrainHeatmap] = useState(null)
  const [skinHeatmap, setSkinHeatmap] = useState(null)

  const [loading, setLoading] = useState(false)
  const [gradcamLoading, setGradcamLoading] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [error, setError] = useState(null)
  const [savedMsg, setSavedMsg] = useState(null)

  // ── Screen routing ───────────────────────────────────────────────
  if (!user) return (
    <Auth onSuccess={(u) => login(u)} />
  )

  // ── Helpers ──────────────────────────────────────────────────────
  const reset = () => {
    setImage(null); setPreview(null); setError(null); setSavedMsg(null)
    setXrayResult(null); setVitalsResult(null); setBrainResult(null); setSkinResult(null)
    setXrayHeatmap(null); setBrainHeatmap(null); setSkinHeatmap(null)
    setVitals(DEFAULT_VITALS)
    setPatient({ name: '', age: '', gender: 'Male', id: '' })
  }

  const switchModule = (m) => { setModule(m); reset() }

  // ── ✅ NEW: Patient info validation ──────────────────────────────
  const isPatientInfoComplete =
    patient.name.trim() !== '' &&
    patient.age.trim() !== '' &&
    patient.id.trim() !== ''

  const handleImage = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setImage(file); setPreview(URL.createObjectURL(file)); setError(null)
  }

  const savePrediction = async (data) => {
    try {
      await axios.post(`${API}/save-prediction`, data)
      setSavedMsg('Patient record saved to database.')
      setTimeout(() => setSavedMsg(null), 4000)
    } catch { console.error('Failed to save prediction') }
  }

  const analyze = async () => {
    setError(null); setSavedMsg(null)
    if (!isPatientInfoComplete) {
      setError('Please complete all patient information (Name, Patient ID, Age) before running the diagnosis.')
      return
    }
    if (!image) { setError('Please upload an image before running the diagnosis.'); return }
    if (module === 'xray' && (!vitals.bmi || !vitals.age)) {
      setError('Please fill in required vitals: BMI and Age Group.'); return
    }
    setLoading(true)
    setXrayResult(null); setVitalsResult(null); setBrainResult(null); setSkinResult(null)
    setXrayHeatmap(null); setBrainHeatmap(null); setSkinHeatmap(null)
    try {
      const imgForm = new FormData()
      imgForm.append('image', image)
      if (module === 'xray') {
        const vForm = new FormData()
        Object.keys(vitals).forEach(k => vForm.append(k, vitals[k] || 0))
        const [xr, vr] = await Promise.all([
          axios.post(`${API}/predict-xray`, imgForm),
          axios.post(`${API}/predict-vitals`, vForm)
        ])
        if (xr.data.status === 'error' || xr.data.error === true) {
          setError(xr.data.message)
        } else {
          setXrayResult(xr.data); setVitalsResult(vr.data)
          await savePrediction({
            patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown',
            patient_age: patient.age || 'N/A', patient_gender: patient.gender || 'Male',
            module: 'xray', diagnosis: xr.data.diagnosis, risk_score: xr.data.risk_score,
            probabilities: xr.data.probabilities, vitals_diagnosis: vr.data.diagnosis,
            vitals_risk_score: vr.data.risk_score, vitals_probabilities: vr.data.probabilities,
            saved_by: user?.username || 'Unknown'
          })
        }
      } else if (module === 'brain') {
        const br = await axios.post(`${API}/predict-brain`, imgForm)
        if (br.data.status === 'error') setError(br.data.message)
        else {
          setBrainResult(br.data)
          setGuidance(br.data.guidance)
          await savePrediction({
            patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown',
            patient_age: patient.age || 'N/A', patient_gender: patient.gender || 'Male',
            module: 'brain', diagnosis: br.data.diagnosis, risk_score: br.data.risk_score,
            probabilities: br.data.probabilities, saved_by: user?.username || 'Unknown'
          })
        }
      } else {
        const sk = await axios.post(`${API}/predict-skin`, imgForm)
        if (sk.data.status === 'error') setError(sk.data.message)
        else {
          setSkinResult(sk.data)
          setGuidance(sk.data.guidance)
          await savePrediction({
            patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown',
            patient_age: patient.age || 'N/A', patient_gender: patient.gender || 'Male',
            module: 'skin', diagnosis: sk.data.diagnosis, risk_score: sk.data.risk_score,
            probabilities: sk.data.probabilities, saved_by: user?.username || 'Unknown'
          })
        }
      }
    } catch { setError('Cannot connect to backend. Please make sure the server is running.') }
    setLoading(false)
  }

  const generateHeatmaps = async () => {
    if (!image) return
    setGradcamLoading(true)
    try {
      const imgForm = new FormData()
      imgForm.append('image', image)
      if (module === 'xray') {
        const res = await axios.post(`${API}/gradcam-xray`, imgForm)
        if (res.data.status === 'error') setError(res.data.message)
        else setXrayHeatmap(res.data.heatmap)
      } else if (module === 'brain') {
        const res = await axios.post(`${API}/gradcam-brain`, imgForm)
        if (res.data.status === 'error' || res.data.error === true) setError(res.data.message)
        else setBrainHeatmap(res.data.heatmap)
      } else {
        const res = await axios.post(`${API}/gradcam-skin`, imgForm)
        if (res.data.status === 'error' || res.data.error === true) setError(res.data.message)
        else setSkinHeatmap(res.data.heatmap)
      }
    } catch { setError('Failed to generate heatmap!') }
    setGradcamLoading(false)
  }

  const downloadReport = async () => {
    if (!image) return
    setReportLoading(true)
    try {
      const form = new FormData()
      form.append('image', image)
      form.append('module', module)
      form.append('patient_name', patient.name || '')
      form.append('patient_id', patient.id || '')
      form.append('patient_age', patient.age || '')
      form.append('patient_gender', patient.gender || 'Male')
      if (module === 'xray' && xrayResult && vitalsResult) {
        form.append('xray_diagnosis', xrayResult.diagnosis)
        form.append('xray_risk_score', xrayResult.risk_score)
        form.append('xray_prob_normal', xrayResult.probabilities['Normal'])
        form.append('xray_prob_pneumonia', xrayResult.probabilities['Pneumonia'])
        form.append('vitals_diagnosis', vitalsResult.diagnosis)
        form.append('vitals_risk_score', vitalsResult.risk_score)
        form.append('vitals_prob_no_diabetes', vitalsResult.probabilities['No Diabetes'])
        form.append('vitals_prob_diabetes', vitalsResult.probabilities['Diabetes'])
        form.append('heatmap', xrayHeatmap || '')
      }
      if (module === 'brain' && brainResult) {
        form.append('brain_diagnosis', brainResult.diagnosis)
        form.append('brain_risk_score', brainResult.risk_score)
        form.append('brain_prob_glioma', brainResult.probabilities['Glioma'])
        form.append('brain_prob_meningioma', brainResult.probabilities['Meningioma'])
        form.append('brain_prob_no_tumor', brainResult.probabilities['No Tumor'])
        form.append('brain_prob_pituitary', brainResult.probabilities['Pituitary'])
        form.append('heatmap', brainHeatmap || '')
      }
      if (module === 'skin' && skinResult) {
        form.append('skin_diagnosis', skinResult.diagnosis)
        form.append('skin_risk_score', skinResult.risk_score)
        form.append('skin_probabilities', JSON.stringify(skinResult.probabilities))
        form.append('heatmap', skinHeatmap || '')
      }
      const res = await axios.post(`${API}/generate-report`, form, { responseType: 'blob' })
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `MediAI_Report_${patient.name || 'Patient'}.pdf`)
      document.body.appendChild(link); link.click(); link.remove()
    } catch { setError('Failed to generate PDF report!') }
    setReportLoading(false)
  }

  // ── Risk helpers ─────────────────────────────────────────────────
  const riskColor = s => s >= 70 ? '#dc2626' : s >= 40 ? '#d97706' : '#059669'
  const riskBg = s => s >= 70 ? '#fef2f2' : s >= 40 ? '#fffbeb' : '#ecfdf5'
  const riskLabel = s => s >= 70 ? 'High Risk' : s >= 40 ? 'Moderate Risk' : 'Low Risk'
  const RiskIcon = ({ score, size = 14 }) => {
    const color = riskColor(score)
    const Icon = score >= 70 ? AlertOctagon : score >= 40 ? AlertTriangle : CheckCircle2
    return <Icon size={size} color={color} strokeWidth={2.2} />
  }

  // ── Static data ──────────────────────────────────────────────────
  const HERO_STATS = {
    xray: [['X-Ray Accuracy', '98%'], ['Diabetes Acc.', '78%'], ['X-Ray Model', 'ResNet-50'], ['Explainability', 'Grad-CAM'], ['Reports', 'PDF Export']],
    brain: [['MRI Accuracy', '94.75%'], ['Tumour Classes', '4 Types'], ['MRI Model', 'EfficientNet-B3'], ['Explainability', 'Grad-CAM'], ['Reports', 'PDF Export']],
    skin: [['Skin Accuracy', '74.7%'], ['Lesion Classes', '7 Types'], ['Skin Model', 'EfficientNet-B3'], ['Explainability', 'Grad-CAM'], ['Reports', 'PDF Export']]
  }
  // Field schema for the BRFSS-based diabetes risk model (21 real health/lifestyle
  // indicators, replacing the old 8-field Pima-dataset form). 'options' arrays are
  // [value, label] pairs; backend Form field names must match these keys exactly
  // (see backend/main.py predict_vitals_endpoint).
  const YES_NO = [['0', 'No'], ['1', 'Yes']]
  const VITALS_FIELDS = [
    { group: 'Health Conditions', fields: [
      ['high_bp', 'High Blood Pressure', 'toggle', YES_NO],
      ['high_chol', 'High Cholesterol', 'toggle', YES_NO],
      ['chol_check', 'Cholesterol Checked (5 yrs)', 'toggle', YES_NO],
      ['stroke', 'History of Stroke', 'toggle', YES_NO],
      ['heart_disease_or_attack', 'Heart Disease/Attack', 'toggle', YES_NO],
      ['diff_walk', 'Difficulty Walking', 'toggle', YES_NO],
    ]},
    { group: 'Lifestyle', fields: [
      ['smoker', 'Smoked 100+ Cigarettes', 'toggle', YES_NO],
      ['phys_activity', 'Physical Activity', 'toggle', YES_NO],
      ['fruits', 'Eats Fruit ≥1/day', 'toggle', YES_NO],
      ['veggies', 'Eats Vegetables ≥1/day', 'toggle', YES_NO],
      ['hvy_alcohol_consump', 'Heavy Alcohol Use', 'toggle', YES_NO],
    ]},
    { group: 'Healthcare Access', fields: [
      ['any_healthcare', 'Has Health Coverage', 'toggle', YES_NO],
      ['no_doc_bc_cost', 'Skipped Doctor (Cost)', 'toggle', YES_NO],
    ]},
    { group: 'General Health', fields: [
      ['bmi', 'BMI (kg/m²)', 'number', '25.0', true],
      ['gen_hlth', 'General Health', 'select', [['1','Excellent'],['2','Very Good'],['3','Good'],['4','Fair'],['5','Poor']]],
      ['ment_hlth', 'Poor Mental Health Days (past 30)', 'number', '0'],
      ['phys_hlth', 'Poor Physical Health Days (past 30)', 'number', '0'],
    ]},
    { group: 'Demographics', fields: [
      ['sex', 'Sex', 'select', [['0','Female'],['1','Male']]],
      ['age', 'Age Group', 'select', [
        ['1','18-24'],['2','25-29'],['3','30-34'],['4','35-39'],['5','40-44'],['6','45-49'],
        ['7','50-54'],['8','55-59'],['9','60-64'],['10','65-69'],['11','70-74'],['12','75-79'],['13','80+']
      ], true],
      ['education', 'Education Level', 'select', [
        ['1','Never attended / Kindergarten'],['2','Elementary (1-8)'],['3','Some high school'],
        ['4','High school graduate'],['5','Some college / technical'],['6','College graduate']
      ]],
      ['income', 'Household Income', 'select', [
        ['1','< $10,000'],['2','$10,000-$14,999'],['3','$15,000-$19,999'],['4','$20,000-$24,999'],
        ['5','$25,000-$34,999'],['6','$35,000-$49,999'],['7','$50,000-$74,999'],['8','$75,000+']
      ]],
    ]},
  ]
  const BRAIN_CONDITIONS = [
    ['#dc2626', '#fef2f2', '#dc262620', 'Glioma', 'Most common malignant brain tumour; arises from glial cells.'],
    ['#ea580c', '#fff7ed', '#ea580c20', 'Meningioma', 'Usually benign; originates from the meninges.'],
    ['#7c3aed', '#faf5ff', '#7c3aed20', 'Pituitary', 'Affects the pituitary gland; often benign.'],
    ['#059669', '#ecfdf5', '#05966920', 'No Tumor', 'No abnormality detected in the scan.'],
  ]
  const SKIN_CONDITIONS = [
    ['#dc2626', '#fef2f2', '#dc262620', 'Melanoma', 'Most dangerous skin cancer; can spread if untreated.'],
    ['#dc2626', '#fef2f2', '#dc262620', 'Basal Cell Carcinoma', 'Most common skin cancer; slow-growing, highly treatable.'],
    ['#ea580c', '#fff7ed', '#ea580c20', 'Actinic Keratosis', 'Precancerous patch caused by sun damage.'],
    ['#d97706', '#fffbeb', '#d9770620', 'Benign Keratosis', 'Harmless, common with age.'],
    ['#d97706', '#fffbeb', '#d9770620', 'Dermatofibroma', 'Harmless firm bump, usually on the legs.'],
    ['#059669', '#ecfdf5', '#05966920', 'Melanocytic Nevus', 'A common mole; usually benign.'],
    ['#059669', '#ecfdf5', '#05966920', 'Vascular Lesion', 'Benign blood-vessel-related growth.'],
  ]

  // ── Sub-components ───────────────────────────────────────────────
  const ResultCard = ({ icon, title, result, color }) => {
    if (!result || !result.probabilities) return null
    return (
    <div className="result-card" style={{ borderTopColor: color }}>
      <div className="result-card-header">
        <span className="result-card-icon">{icon}</span>
        <h3 className="result-card-title">{title}</h3>
      </div>
      <div className="dx-box" style={{ background: riskBg(result.risk_score), borderColor: riskColor(result.risk_score) + '30' }}>
        <p className="dx-label">Primary Diagnosis</p>
        <p className="dx-diagnosis">{result.diagnosis}</p>
        {result.low_confidence ? (
          <div className="dx-risk-row">
            <Circle size={14} color="#6b7280" strokeWidth={2.2} />
            <span className="dx-risk-label" style={{ color: '#6b7280' }}>Low Confidence</span>
            <span className="dx-risk-score">— {result.note}</span>
          </div>
        ) : (
          <div className={`dx-risk-row ${result.risk_score >= 70 ? 'risk-pulse' : ''}`}>
            <RiskIcon score={result.risk_score} />
            <span className="dx-risk-label" style={{ color: riskColor(result.risk_score) }}>{riskLabel(result.risk_score)}</span>
            <span className="dx-risk-score">— Score: {result.risk_score}/100</span>
          </div>
        )}
      </div>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width: `${result.risk_score}%`, background: `linear-gradient(90deg,${color},${riskColor(result.risk_score)})` }} />
      </div>
      <p className="prob-section-label">Probability Breakdown</p>
      {Object.entries(result.probabilities).map(([d, p]) => (
        <div key={d} className="prob-row">
          <div className="prob-row-header">
            <span className="prob-class">{d}</span>
            <span className="prob-pct">{p}%</span>
          </div>
          <div className="prob-track">
            <div className="prob-fill" style={{ width: `${p}%`, background: color }} />
          </div>
        </div>
      ))}
    </div>
    )
  }

  const PatientBadge = () => patient.name ? (
    <div className="patient-badge">
      <p className="patient-badge-label">🧑‍⚕️ Patient</p>
      <p className="patient-badge-name">{patient.name}</p>
      <p className="patient-badge-meta">
        {patient.id && `ID: ${patient.id} · `}
        {patient.age && `Age: ${patient.age} · `}
        {patient.gender}
      </p>
    </div>
  ) : null

  const HeatmapSection = ({ heatmap, color, borderColor }) => (
    <div className="heatmap-card" style={{ borderColor }}>
      <div className="heatmap-card-header">
        <div>
          <p className="heatmap-title"><Microscope size={16} /> AI Attention Heatmap — Grad-CAM</p>
          <p className="heatmap-sub">Visualise which image regions influenced the AI diagnosis</p>
        </div>
        <button
          onClick={generateHeatmaps} disabled={gradcamLoading}
          className={`action-btn ${gradcamLoading ? '' : 'action-btn-heat'}`}
          style={gradcamLoading ? {} : { borderColor: color, color }}
        >
          {gradcamLoading ? <><Loader2 size={15} className="spin" /> Generating…</> : <><Flame size={15} /> Generate Heatmap</>}
        </button>
      </div>

      {!heatmap && !gradcamLoading && (
        <div className="heatmap-empty">
          <span className="heatmap-empty-icon"><Microscope size={30} /></span>
          <p className="heatmap-empty-text">Click "Generate Heatmap" to visualise AI attention</p>
          <p className="heatmap-empty-sub">Highlights the regions the model focused on</p>
        </div>
      )}
      {gradcamLoading && (
        <div className="heatmap-empty">
          <span className="heatmap-empty-icon"><Loader2 size={30} className="spin" /></span>
          <p className="heatmap-empty-text">Generating heatmap, please wait…</p>
        </div>
      )}
      {heatmap && (
        <>
          <div className="heatmap-grid">
            <div className="heatmap-img-block">
              <p className="heatmap-img-label">Original Image</p>
              <img src={preview} alt="original" className="heatmap-img" />
            </div>
            <div className="heatmap-img-block">
              <p className="heatmap-img-label">AI Attention Heatmap</p>
              <img src={`data:image/jpeg;base64,${heatmap}`} alt="heatmap" className="heatmap-img" style={{ background: 'transparent' }} />
            </div>
          </div>
          <div className="heatmap-legend">
            <strong>How to read:</strong>&nbsp; <strong style={{ color: '#dc2626' }}>Red/Yellow</strong> — areas of highest AI attention (likely abnormal). &nbsp;<strong style={{ color: '#2563eb' }}>Blue</strong> — low relevance to the diagnosis.
          </div>
        </>
      )}
    </div>
  )

  // ── Render ───────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* ── Navbar ──────────────────────────────────────────────── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <img src={glsLogo} alt="GLS University" className="navbar-logo" />
          <div>
            <div className="navbar-title">MediAI Diagnostics</div>
            <div className="navbar-sub">AI-Powered Clinical Decision Support</div>
          </div>
        </div>

        <div className="navbar-center">
          {[['xray', Wind, 'Chest X-Ray'], ['brain', Brain, 'Brain MRI'], ['skin', Microscope, 'Skin Lesion']].map(([m, Icon, label]) => (
            <button
              key={m} onClick={() => switchModule(m)}
              className={`tab-btn ${module === m ? (m === 'brain' ? 'tab-active-purple' : m === 'skin' ? 'tab-active-amber' : 'tab-active-pink') : ''}`}
            >
              <Icon size={15} /> {label}
            </button>
          ))}
        </div>

        <div className="navbar-right">
          {user && (
            <div className="user-pill">
              <span className="user-pill-dot" />
              <span>{user.username}</span>
            </div>
          )}
          <div className="nav-sep" />
          <button className="logout-btn" onClick={() => { reset(); logout() }}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </nav>

      {/* ── Hero Banner ─────────────────────────────────────────── */}
      <div className={`hero ${module}`}>
        <div className="hero-inner">
          <div className="hero-main">
            <div className="hero-icon">
              {(() => { const Icon = MODULE_ICON[module]; return <Icon size={22} strokeWidth={2} /> })()}
            </div>
            <div style={{ maxWidth: '420px' }}>
              <h2 className="hero-title">
                {module === 'xray' ? 'Chest X-Ray + Diabetes Analysis' : module === 'brain' ? 'Brain MRI Tumour Detection' : 'Skin Lesion Classification'}
              </h2>
              <p className="hero-sub">
                {module === 'xray'
                  ? 'Upload a chest X-ray and enter patient vitals for AI-powered pneumonia detection and diabetes risk assessment.'
                  : module === 'brain'
                  ? 'Upload a brain MRI scan for AI-powered tumour classification and comprehensive risk assessment.'
                  : 'Upload a dermatoscopic photo of a skin lesion for AI-powered classification across 7 lesion types.'}
              </p>
            </div>
          </div>
          <div className="hero-stats">
            {HERO_STATS[module].map(([label, value]) => (
              <div key={label} className="hero-stat">
                <p className="hero-stat-label">{label}</p>
                <p className="hero-stat-value">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Page Body ───────────────────────────────────────────── */}
      <div className="page-body">

        {error && <div className="alert alert-error"><AlertTriangle size={16} style={{ flexShrink: 0 }} /> {error}</div>}
        {savedMsg && <div className="alert alert-success"><CheckCircle2 size={16} style={{ flexShrink: 0 }} /> {savedMsg}</div>}

        {/* ── Input Grid ────────────────────────────────────────── */}
        <div className="main-grid">

          {/* ── LEFT COLUMN ─────────────────────────────────────── */}
          <div>
            {/* Patient Info */}
            <div className="card">
              <div className="card-header">
                <div className={`card-icon ${module === 'brain' ? 'card-icon-purple' : module === 'skin' ? 'card-icon-amber' : 'card-icon-pink'}`}><User size={18} /></div>
                <div>
                  <p className="card-title">Patient Information</p>
                  <p className="card-sub">Enter patient demographic details</p>
                </div>
              </div>
              <div className="form-grid-2">
                <div className="form-group span-2">
                  <label className="form-label">
                    Full Name <span style={{ color: 'var(--red)' }}>*</span>
                  </label>
                  <input
                    className="form-input"
                    placeholder="Enter patient full name"
                    value={patient.name}
                    onChange={e => setPatient({ ...patient, name: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">
                    Patient ID <span style={{ color: 'var(--red)' }}>*</span>
                  </label>
                  <input
                    className="form-input"
                    placeholder="PT-001"
                    value={patient.id}
                    onChange={e => setPatient({ ...patient, id: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">
                    Age <span style={{ color: 'var(--red)' }}>*</span>
                  </label>
                  <input
                    className="form-input"
                    type="number"
                    placeholder="Years"
                    value={patient.age}
                    onChange={e => setPatient({ ...patient, age: e.target.value })}
                  />
                </div>
                <div className="form-group span-2">
                  <label className="form-label">Gender</label>
                  <select className="form-input" value={patient.gender}
                    onChange={e => setPatient({ ...patient, gender: e.target.value })}>
                    <option>Male</option><option>Female</option><option>Other</option>
                  </select>
                </div>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-4)', marginTop: '6px' }}>
                <span style={{ color: 'var(--red)' }}>*</span> Required fields — must be filled to unlock image upload
              </p>
            </div>

            {/* ── ✅ Image Upload — locked until patient info is complete ── */}
            <div className="card">
              <div className="card-header">
                <div className={`card-icon ${module === 'brain' ? 'card-icon-purple' : module === 'skin' ? 'card-icon-amber' : 'card-icon-pink'}`}>
                  {(() => { const Icon = module === 'xray' ? ImageIcon : module === 'brain' ? Brain : Microscope; return <Icon size={18} /> })()}
                </div>
                <div>
                  <p className="card-title">{module === 'xray' ? 'Chest X-Ray Image' : module === 'brain' ? 'Brain MRI Scan' : 'Skin Lesion Photo'}</p>
                  <p className="card-sub">
                    {isPatientInfoComplete
                      ? 'Upload your medical imaging scan'
                      : 'Complete patient info above to enable upload'}
                  </p>
                </div>
              </div>

              {/* Upload zone — conditionally locked */}
              <label
                className={`upload-zone ${!isPatientInfoComplete ? 'upload-zone-disabled' : ''}`}
                onClick={e => { if (!isPatientInfoComplete) e.preventDefault() }}
                title={!isPatientInfoComplete ? 'Fill in Name, Patient ID, and Age first' : ''}
              >
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImage}
                  style={{ display: 'none' }}
                  disabled={!isPatientInfoComplete}
                />
                {!isPatientInfoComplete ? (
                  <>
                    <span className="upload-icon"><Lock size={30} /></span>
                    <p className="upload-hint-main">Image upload locked</p>
                    <p className="upload-hint-sub">Fill Name, Patient ID &amp; Age first</p>
                  </>
                ) : preview ? (
                  <img src={preview} alt="preview" className="upload-preview" />
                ) : (
                  <>
                    <span className="upload-icon"><ImageIcon size={30} /></span>
                    <p className="upload-hint-main">Click to upload scan image</p>
                    <p className="upload-hint-sub">PNG, JPG, JPEG — Max 10 MB</p>
                  </>
                )}
              </label>

              {module === 'brain' && (
                <div className="module-warning"><AlertTriangle size={14} /> Only brain MRI scans are accepted for this module.</div>
              )}
              {module === 'skin' && (
                <div className="module-warning"><AlertTriangle size={14} /> Only close-up skin lesion photos are accepted for this module.</div>
              )}
            </div>
          </div>

          {/* ── RIGHT COLUMN ────────────────────────────────────── */}
          <div>
            {/* Vitals (X-Ray mode) */}
            {module === 'xray' && (
              <div className="card">
                <div className="card-header">
                  <div className="card-icon card-icon-green"><Droplet size={18} /></div>
                  <div>
                    <p className="card-title">Patient Vitals</p>
                    <p className="card-sub">Health & lifestyle indicators for diabetes risk assessment</p>
                  </div>
                </div>
                {VITALS_FIELDS.map(({ group, fields }) => {
                  const allToggle = fields.every(f => f[2] === 'toggle')
                  return (
                  <div key={group} style={{ marginBottom: '10px' }}>
                    <p style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '5px' }}>
                      {group}
                    </p>
                    <div className={allToggle ? 'vitals-grid-4' : 'vitals-grid'}>
                      {fields.map(([key, label, type, opt, req]) => (
                        <div key={key} className="form-group">
                          <label className="form-label">
                            {label}{req && <span className="req">*</span>}
                          </label>
                          {type === 'toggle' ? (
                            <div className="toggle-group">
                              {opt.map(([v, l]) => (
                                <button type="button" key={v}
                                  className={`toggle-btn ${vitals[key] === v ? 'toggle-btn-active' : ''}`}
                                  onClick={() => setVitals({ ...vitals, [key]: v })}>
                                  {l}
                                </button>
                              ))}
                            </div>
                          ) : type === 'select' ? (
                            <select className="form-input" value={vitals[key]}
                              onChange={e => setVitals({ ...vitals, [key]: e.target.value })}>
                              {opt.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                            </select>
                          ) : (
                            <input type="number" className="form-input" placeholder={opt}
                              value={vitals[key]} onChange={e => setVitals({ ...vitals, [key]: e.target.value })} />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                  )
                })}
                <p style={{ fontSize: '11px', color: 'var(--text-4)', marginTop: '6px' }}>
                  <span style={{ color: 'var(--red)' }}>*</span> Required fields
                </p>
              </div>
            )}

            {/* Detectable conditions (Brain mode) */}
            {module === 'brain' && (
              <div className="card">
                <div className="card-header">
                  <div className="card-icon card-icon-purple"><Search size={18} /></div>
                  <div>
                    <p className="card-title">Detectable Conditions</p>
                    <p className="card-sub">4 tumour types supported by this model</p>
                  </div>
                </div>
                {BRAIN_CONDITIONS.map(([dot, bg, border, name, desc]) => (
                  <div key={name} className="condition-item" style={{ background: bg, borderColor: border }}>
                    <span className="condition-dot" style={{ background: dot }} />
                    <div>
                      <p className="condition-name">{name}</p>
                      <p className="condition-desc">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Detectable conditions (Skin mode) */}
            {module === 'skin' && (
              <div className="card">
                <div className="card-header">
                  <div className="card-icon card-icon-amber"><Search size={18} /></div>
                  <div>
                    <p className="card-title">Detectable Conditions</p>
                    <p className="card-sub">7 lesion types supported by this model</p>
                  </div>
                </div>
                {SKIN_CONDITIONS.map(([dot, bg, border, name, desc]) => (
                  <div key={name} className="condition-item" style={{ background: bg, borderColor: border }}>
                    <span className="condition-dot" style={{ background: dot }} />
                    <div>
                      <p className="condition-name">{name}</p>
                      <p className="condition-desc">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Analyze button */}
            <button
              onClick={analyze} disabled={loading}
              className={`analyze-btn ${module === 'brain' ? 'purple' : module === 'skin' ? 'amber' : 'pink'}`}
            >
              {loading
                ? <><Loader2 size={17} className="spin" /> Analysing Patient Data…</>
                : <><Search size={17} /> Run AI Diagnosis</>
              }
            </button>
          </div>
        </div>

        {/* ── X-Ray Results ─────────────────────────────────────── */}
        {xrayResult && vitalsResult && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <h2 className="results-title">📊 Diagnosis Results</h2>
                <p className="results-date">
                  {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              <PatientBadge />
            </div>

            <div className="result-row">
              <ResultCard icon={<Wind size={19} color="#5c8f3e" />} title="Chest X-Ray Analysis" result={xrayResult} color="#5c8f3e" />
              <ResultCard icon={<Droplet size={19} color="#059669" />} title="Diabetes Risk Analysis" result={vitalsResult} color="#059669" />
            </div>

            <div className="action-row">
              <button onClick={downloadReport} disabled={reportLoading}
                className={`action-btn ${reportLoading ? '' : 'action-btn-pdf'}`}>
                {reportLoading ? <><Loader2 size={15} className="spin" /> Generating PDF…</> : <><FileText size={15} /> Download PDF Report</>}
              </button>
            </div>

            <div className="disclaimer">
              <AlertTriangle size={16} />
              <p><strong>Medical Disclaimer:</strong> This AI-assisted result is for clinical decision support only. Always consult a qualified medical professional before making any clinical decisions.</p>
            </div>

            <HeatmapSection heatmap={xrayHeatmap} color="#5c8f3e" borderColor="#c3e0ab" />
          </section>
        )}

        {/* ── Brain Results ──────────────────────────────────────── */}
        {brainResult && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <h2 className="results-title">📊 Brain MRI Results</h2>
                <p className="results-date">
                  {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              <PatientBadge />
            </div>

            <ResultCard icon={<Brain size={19} color="#7c3aed" />} title="Brain Tumour Analysis" result={brainResult} color="#7c3aed" />

            <div className="action-row">
              <button onClick={downloadReport} disabled={reportLoading}
                className={`action-btn ${reportLoading ? '' : 'action-btn-pdf'}`}>
                {reportLoading ? <><Loader2 size={15} className="spin" /> Generating PDF…</> : <><FileText size={15} /> Download PDF Report</>}
              </button>
            </div>

            <div className="disclaimer">
              <AlertTriangle size={16} />
              <p><strong>Medical Disclaimer:</strong> This AI-assisted result is for clinical decision support only. Always consult a qualified medical professional before making any clinical decisions.</p>
            </div>

            <HeatmapSection heatmap={brainHeatmap} color="#7c3aed" borderColor="#ddd6fe" />
          </section>
        )}

        {/* ── Skin Results ──────────────────────────────────────── */}
        {skinResult && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <h2 className="results-title">📊 Skin Lesion Results</h2>
                <p className="results-date">
                  {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              <PatientBadge />
            </div>

            <ResultCard icon={<Microscope size={19} color="#d97706" />} title="Skin Lesion Analysis" result={skinResult} color="#d97706" />

            {guidance && (
              <div className="module-warning" style={{ marginTop: '12px', background: 'var(--blue-light)', border: '1px solid #bfdbfe', color: 'var(--blue-dark)' }}>
                <AlertCircle size={14} /> {guidance}
              </div>
            )}

            <div className="action-row">
              <button onClick={downloadReport} disabled={reportLoading}
                className={`action-btn ${reportLoading ? '' : 'action-btn-pdf'}`}>
                {reportLoading ? <><Loader2 size={15} className="spin" /> Generating PDF…</> : <><FileText size={15} /> Download PDF Report</>}
              </button>
            </div>

            <div className="disclaimer">
              <AlertTriangle size={16} />
              <p><strong>Medical Disclaimer:</strong> This AI-assisted result is for clinical decision support only. Always consult a qualified medical professional before making any clinical decisions.</p>
            </div>

            <HeatmapSection heatmap={skinHeatmap} color="#d97706" borderColor="#fde68a" />
          </section>
        )}

        {/* Footer */}
        <div className="page-footer">
          <p>🏥 MediAI Diagnostics · GLS University Capstone Project 2025–26 · Integrated MSc(IT) Programme</p>
        </div>
      </div>
    </div>
  )
}