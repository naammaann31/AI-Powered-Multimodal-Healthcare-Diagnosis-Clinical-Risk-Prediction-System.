import { useState, useEffect } from 'react'
import axios from 'axios'
import { Eye, EyeOff, ArrowRight, AlertCircle, ShieldCheck, Wind, Brain, Microscope, Droplet } from 'lucide-react'
import glsLogo from './assets/gls-logo.jpg'

const API = 'http://127.0.0.1:8000'

// Chrome (and most Chromium browsers) deliberately ignore autocomplete="off" on
// login-shaped forms — it's an intentional browser policy, not a bug we can fix
// with attributes alone. These are the standard, battle-tested countermeasures:
// hidden decoy fields (with the "normal" names) to absorb the browser's autofill
// match, opaque names on the real fields, and read-only-until-focused so nothing
// can be silently populated before the user interacts with the field.
const decoyStyle = { position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none', border: 0, padding: 0 }

const FEATURES = [
  [Wind, 'Chest X-Ray', 'Pneumonia detection with Grad-CAM heatmaps'],
  [Brain, 'Brain MRI', '4-class tumour classification with explainable AI'],
  [Microscope, 'Skin Lesion', '7-class lesion screening with Grad-CAM heatmaps'],
  [Droplet, 'Diabetes Risk', 'Risk prediction from 21 clinical health indicators'],
]

export default function Auth({ onSuccess }) {
  const [form, setForm] = useState({ username: '', email: '', password: '', confirmPassword: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isLogin, setIsLogin] = useState(true)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [locked, setLocked] = useState({ username: true, email: true, password: true, confirmPassword: true })

  useEffect(() => { setTimeout(() => setLoaded(true), 60) }, [])
  useEffect(() => { setForm({ username: '', email: '', password: '', confirmPassword: '' }) }, [])

  const fadeIn = (delay = 0) => ({
    opacity: loaded ? 1 : 0,
    transform: loaded ? 'translateY(0)' : 'translateY(8px)',
    transition: `opacity 0.4s ease ${delay}s, transform 0.4s ease ${delay}s`
  })

  const unlock = field => setLocked(l => ({ ...l, [field]: false }))

  const handle = async (e) => {
    if (e) e.preventDefault()
    setError(null)
    if (!form.email || !form.password) { setError('Please fill in all required fields.'); return }
    if (!isLogin && !form.username) { setError('Please enter your full name.'); return }
    if (!isLogin && form.password !== form.confirmPassword) { setError('Passwords do not match.'); return }
    setLoading(true)
    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register'
      const payload = isLogin
        ? { email: form.email, password: form.password }
        : { username: form.username, email: form.email, password: form.password }
      const res = await axios.post(`${API}${endpoint}`, payload)
      if (!isLogin) {
        setIsLogin(true)
        setForm({ ...form, password: '', confirmPassword: '' })
        alert('Account created successfully! Please sign in.')
      } else {
        onSuccess({ username: res.data.username, email: res.data.email })
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'Something went wrong. Please try again.')
    }
    setLoading(false)
  }

  const inp = {
    width: '100%', padding: '12px 14px',
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    background: '#f9fafb',
    color: '#111827', fontSize: '15px',
    boxSizing: 'border-box', outline: 'none',
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif",
    transition: 'border-color 0.15s, background 0.15s, box-shadow 0.15s'
  }

  const lbl = {
    fontSize: '13px', fontWeight: '600',
    color: '#374151',
    display: 'block', marginBottom: '7px'
  }

  const focusInput = (e, field) => {
    if (field) unlock(field)
    e.target.style.borderColor = '#2563eb'; e.target.style.background = '#ffffff'; e.target.style.boxShadow = '0 0 0 3px rgba(37,99,235,0.1)'
  }
  const blurInput = e => { e.target.style.borderColor = '#e5e7eb'; e.target.style.background = '#f9fafb'; e.target.style.boxShadow = 'none' }

  const bluePanel = (
    <div key="blue" style={{
      flex: '0 0 46%', minHeight: '100vh',
      background: 'linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #3b82f6 100%)',
      position: 'relative', overflow: 'hidden',
      display: 'flex', flexDirection: 'column', justifyContent: 'center',
      padding: '48px 56px', color: 'white',
      animation: 'fadeSlide 0.4s ease'
    }}>
      <div style={{ position: 'absolute', top: '-20%', right: '-10%', width: '420px', height: '420px', borderRadius: '50%', background: 'rgba(255,255,255,0.07)' }} />
      <div style={{ position: 'absolute', bottom: '-15%', left: '-8%', width: '300px', height: '300px', borderRadius: '50%', background: 'rgba(255,255,255,0.05)' }} />

      <div style={{ position: 'relative', zIndex: 1, maxWidth: '460px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '32px' }}>
          <img src={glsLogo} alt="GLS University" style={{ height: '50px', width: 'auto', borderRadius: '7px', boxShadow: '0 4px 14px rgba(0,0,0,0.2)' }} />
          <div>
            <p style={{ fontSize: '19px', fontWeight: '800', letterSpacing: '-0.2px' }}>MediAI Diagnostics</p>
            <p style={{ fontSize: '13px', color: 'rgba(255,255,255,0.72)' }}>Clinical Decision Support</p>
          </div>
        </div>

        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '7px', padding: '7px 16px', borderRadius: '999px', background: 'rgba(255,255,255,0.14)', border: '1px solid rgba(255,255,255,0.25)', fontSize: '13px', fontWeight: '600', marginBottom: '22px' }}>
          <ShieldCheck size={15} strokeWidth={2} />
          🎓 GLS University Capstone 2026–27
        </span>

        <h1 style={{ fontSize: '36px', fontWeight: '800', lineHeight: 1.22, letterSpacing: '-0.6px', margin: '0 0 16px' }}>
          AI-Powered Multi-Modal Clinical Diagnostics
        </h1>
        <p style={{ fontSize: '16px', color: 'rgba(255,255,255,0.82)', lineHeight: 1.65, margin: '0 0 36px' }}>
          Deep-learning analysis for pneumonia detection, brain tumour classification,
          skin lesion screening, and diabetes risk prediction — with explainable AI and
          automated clinical reports.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {FEATURES.map(([Icon, title, desc]) => (
            <div key={title} style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(255,255,255,0.14)', border: '1px solid rgba(255,255,255,0.22)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <Icon size={19} strokeWidth={2} />
              </div>
              <div>
                <p style={{ fontSize: '15px', fontWeight: '700' }}>{title}</p>
                <p style={{ fontSize: '13px', color: 'rgba(255,255,255,0.68)' }}>{desc}</p>
              </div>
            </div>
          ))}
        </div>

        <p style={{ fontSize: '11.5px', color: 'rgba(255,255,255,0.45)', marginTop: '38px', letterSpacing: '0.3px' }}>
          🏥 MediAI Diagnostics · GLS University · Integrated MSc(IT) · Capstone 2026–27
        </p>
      </div>
    </div>
  )

  const formPanel = (
    <div key="form" style={{
      flex: '1', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'radial-gradient(900px 560px at 85% 10%, rgba(37,99,235,0.07), transparent 60%), radial-gradient(700px 480px at 10% 90%, rgba(124,58,237,0.05), transparent 55%), #eef1f6',
      padding: '24px', animation: 'fadeSlide 0.4s ease'
    }}>
      <div style={{
        width: '100%', maxWidth: '420px', background: '#ffffff', borderRadius: '16px',
        border: '1px solid #e5e7eb', boxShadow: '0 20px 48px rgba(17,24,39,0.09), 0 4px 12px rgba(17,24,39,0.05)',
        padding: '40px 36px', ...fadeIn(0)
      }}>
        <h2 style={{ fontSize: '30px', fontWeight: '800', color: '#111827', letterSpacing: '-0.3px', marginBottom: '10px' }}>
          {isLogin ? 'Welcome back' : 'Create your account'}
        </h2>
        <p style={{ fontSize: '15px', color: '#6b7280', fontWeight: '400', marginBottom: '28px' }}>
          {isLogin ? 'Sign in to access your clinical dashboard' : 'Join MediAI Diagnostics to get started'}
        </p>

        {error && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', padding: '11px 14px', marginBottom: '18px', color: '#b91c1c', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircle size={15} style={{ flexShrink: 0 }} /> {error}
          </div>
        )}

        <form autoComplete="off" onSubmit={handle}>
          {/* Decoy fields: invisible, unreachable by tab, but carry the "normal"
              names/types — absorbs the browser's autofill match so it leaves
              the real fields (below, with opaque names) alone. */}
          <input type="text" name="username" autoComplete="username" tabIndex={-1} aria-hidden="true" style={decoyStyle} defaultValue="" />
          <input type="email" name="email" autoComplete="email" tabIndex={-1} aria-hidden="true" style={decoyStyle} defaultValue="" />
          <input type="password" name="password" autoComplete="current-password" tabIndex={-1} aria-hidden="true" style={decoyStyle} defaultValue="" />

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginBottom: '20px' }}>
            {!isLogin && (
              <div>
                <label style={lbl}>Full Name</label>
                <input className="mediai-field" style={inp} placeholder="Dr. Jane Smith" value={form.username}
                  autoComplete="off" name="mdc-fullname" id="mdc-fullname" readOnly={locked.username}
                  onFocus={e => focusInput(e, 'username')} onBlur={blurInput}
                  onChange={e => setForm({ ...form, username: e.target.value })} />
              </div>
            )}
            <div>
              <label style={lbl}>Email Address</label>
              <input className="mediai-field" style={inp} type="text" inputMode="email" placeholder="name@hospital.com" value={form.email}
                autoComplete="off" name="mdc-emailaddr" id="mdc-emailaddr" readOnly={locked.email}
                onFocus={e => focusInput(e, 'email')} onBlur={blurInput}
                onChange={e => setForm({ ...form, email: e.target.value })} />
            </div>
            <div>
              <label style={lbl}>Password</label>
              <div style={{ position: 'relative' }}>
                <input className="mediai-field" style={{ ...inp, paddingRight: '44px' }}
                  type={showPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  autoComplete="off" name="mdc-secret" id="mdc-secret" readOnly={locked.password}
                  value={form.password}
                  onFocus={e => focusInput(e, 'password')} onBlur={blurInput}
                  onChange={e => setForm({ ...form, password: e.target.value })} />
                <button type="button" onClick={() => setShowPassword(!showPassword)}
                  style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#9ca3af', padding: 0, display: 'flex', alignItems: 'center' }}>
                  {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                </button>
              </div>
            </div>
            {!isLogin && (
              <div>
                <label style={lbl}>Confirm Password</label>
                <div style={{ position: 'relative' }}>
                  <input className="mediai-field" style={{ ...inp, paddingRight: '44px' }}
                    type={showConfirm ? 'text' : 'password'}
                    placeholder="••••••••"
                    autoComplete="off" name="mdc-secret2" id="mdc-secret2" readOnly={locked.confirmPassword}
                    value={form.confirmPassword}
                    onFocus={e => focusInput(e, 'confirmPassword')} onBlur={blurInput}
                    onChange={e => setForm({ ...form, confirmPassword: e.target.value })} />
                  <button type="button" onClick={() => setShowConfirm(!showConfirm)}
                    style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#9ca3af', padding: 0, display: 'flex', alignItems: 'center' }}>
                    {showConfirm ? <EyeOff size={17} /> : <Eye size={17} />}
                  </button>
                </div>
              </div>
            )}
          </div>

          <button type="submit" disabled={loading}
            onMouseEnter={e => { if (!loading) { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 10px 24px rgba(37,99,235,0.4)' } }}
            onMouseLeave={e => { if (!loading) { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 14px rgba(37,99,235,0.3)' } }}
            style={{ width: '100%', padding: '13px 22px', background: loading ? '#e5e7eb' : 'linear-gradient(135deg,#2563eb,#1d4ed8)', border: 'none', borderRadius: '9px', color: loading ? '#9ca3af' : 'white', fontSize: '15.5px', fontWeight: '700', fontFamily: "'Inter', sans-serif", cursor: loading ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', transition: 'all 0.18s', boxShadow: loading ? 'none' : '0 4px 14px rgba(37,99,235,0.3)' }}>
            {loading ? 'Please wait…' : (<>{isLogin ? 'Sign In' : 'Create Account'} <ArrowRight size={16} /></>)}
          </button>
        </form>

        <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '14px', color: '#6b7280' }}>
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <span onClick={() => { setIsLogin(!isLogin); setError(null) }}
            style={{ color: '#2563eb', fontWeight: '600', cursor: 'pointer' }}>
            {isLogin ? 'Create one' : 'Sign in'}
          </span>
        </p>
      </div>
    </div>
  )

  return (
    <div className="auth-split" style={{ minHeight: '100vh', width: '100%', display: 'flex', fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}>
      {/* Force any field the browser still manages to autofill back to our own
          theme instead of the jarring native blue/yellow autofill background. */}
      <style>{`
        input.mediai-field:-webkit-autofill,
        input.mediai-field:-webkit-autofill:hover,
        input.mediai-field:-webkit-autofill:focus {
          -webkit-text-fill-color: #111827;
          caret-color: #111827;
          transition: background-color 9999s ease-in-out 0s;
          box-shadow: 0 0 0 1000px #f9fafb inset;
        }
        @keyframes fadeSlide {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @media (max-width: 860px) {
          .auth-split { flex-direction: column; }
        }
      `}</style>
      {isLogin ? [bluePanel, formPanel] : [formPanel, bluePanel]}
    </div>
  )
}
