"""
Generates the 5 required UML diagrams (Use Case, Class, Activity/Swimlane,
Sequence, State Chart) for the Capstone-II documentation, as PNG images,
from the actual MediAI Diagnostics architecture (backend/*.py, frontend/src/*.jsx).
Run once; output goes to documentation/diagrams/.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import os

OUT = os.path.join(os.path.dirname(__file__), 'diagrams')
os.makedirs(OUT, exist_ok=True)

NAVY = '#1d4ed8'
PURPLE = '#7c3aed'
AMBER = '#d97706'
GREEN = '#059669'
GRAY = '#64748b'
DARK = '#0f172a'
LIGHT = '#f8fafc'


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, text, fc='white', ec=DARK, fontsize=10, bold=True, lines=None):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(r)
    if lines is None:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                 fontsize=fontsize, fontweight='bold' if bold else 'normal', zorder=3, wrap=True)
    return r


def class_box(ax, x, y_top, w, title, attrs, methods, ec=NAVY, line_h=0.26, fontsize=8.3):
    """Draws a UML class box anchored at its TOP-LEFT corner (x, y_top), with
    height auto-computed from the attribute/method counts so text never overflows
    the border. Returns (x, y_bottom, w, h)."""
    title_h = 0.5
    pad = 0.22
    attrs_h = line_h * max(len(attrs), 1) + 0.1
    methods_h = line_h * max(len(methods), 1) + 0.1
    h = title_h + attrs_h + methods_h + pad
    y = y_top - h

    r = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", linewidth=1.6, edgecolor=ec, facecolor='white', zorder=2)
    ax.add_patch(r)
    ax.add_patch(Rectangle((x, y_top - title_h), w, title_h, facecolor=ec, edgecolor=ec, zorder=3))
    ax.text(x + w/2, y_top - title_h/2, title, ha='center', va='center', color='white', fontsize=10.5, fontweight='bold', zorder=4)

    attrs_top = y_top - title_h
    ax.plot([x, x + w], [attrs_top - attrs_h, attrs_top - attrs_h], color=ec, linewidth=1.2, zorder=4)
    ay = attrs_top - 0.08 - line_h/2
    for a in attrs:
        ax.text(x + 0.12, ay, f"- {a}", ha='left', va='center', fontsize=fontsize, zorder=4)
        ay -= line_h

    methods_top = attrs_top - attrs_h
    my = methods_top - 0.08 - line_h/2
    for m in methods:
        ax.text(x + 0.12, my, f"+ {m}", ha='left', va='center', fontsize=fontsize, zorder=4, fontstyle='italic')
        my -= line_h
    return (x, y, w, h)


def arrow(ax, p1, p2, style='-|>', color=DARK, lw=1.3, ls='-', label=None, connectionstyle='arc3,rad=0.0', fontsize=8):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, linewidth=lw,
                         color=color, linestyle=ls, connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my + 0.12, label, ha='center', va='bottom', fontsize=fontsize, color=color)


def stick_actor(ax, x, y, label):
    ax.plot(x, y + 0.75, 'o', ms=14, mfc='white', mec=DARK, mew=1.6, zorder=3)
    ax.plot([x, x], [y + 0.58, y], color=DARK, lw=1.6, zorder=3)
    ax.plot([x - 0.3, x + 0.3], [y + 0.42, y + 0.42], color=DARK, lw=1.6, zorder=3)
    ax.plot([x, x - 0.28], [y, y - 0.35], color=DARK, lw=1.6, zorder=3)
    ax.plot([x, x + 0.28], [y, y - 0.35], color=DARK, lw=1.6, zorder=3)
    ax.text(x, y - 0.55, label, ha='center', va='top', fontsize=9, fontweight='bold')


# ════════════════════════════════════════════════════════════════════
# 1. USE CASE DIAGRAM
# ════════════════════════════════════════════════════════════════════
def use_case_diagram():
    fig, ax = new_fig(13, 10)
    ax.text(6.7, 9.6, 'Use Case Diagram — MediAI Diagnostics', ha='center', fontsize=14, fontweight='bold')

    boundary = FancyBboxPatch((2.6, 0.5), 10, 8.6, boxstyle="square,pad=0", linewidth=1.6, edgecolor=DARK, facecolor='#f8fafc', zorder=0)
    ax.add_patch(boundary)
    ax.text(7.6, 8.85, 'MediAI Diagnostics System', ha='center', fontsize=10.5, fontweight='bold', color=GRAY)

    stick_actor(ax, 1.1, 4.6, "Healthcare\nProfessional")

    uses = [
        ("Register Account", 4.3, 8.0),
        ("Login", 4.3, 7.1),
        ("Logout", 4.3, 6.2),
        ("Enter Patient\nInformation", 4.3, 5.2),
        ("View Patient\nHistory", 4.3, 4.0),
        ("Upload Chest\nX-ray Image", 9.6, 7.6),
        ("Upload Brain\nMRI Scan", 9.6, 6.6),
        ("Upload Skin\nLesion Photo", 9.6, 5.6),
        ("Enter Vitals /\nHealth Indicators", 9.6, 4.6),
        ("Generate Grad-CAM\nHeatmap", 9.6, 3.4),
        ("Download PDF\nReport", 9.6, 2.3),
    ]
    centers = {}
    for label, x, y in uses:
        e = Ellipse((x, y), 2.5, 1.0, facecolor='white', edgecolor=NAVY, linewidth=1.4, zorder=2)
        ax.add_patch(e)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.8, zorder=3)
        centers[label.split('\n')[0]] = (x, y)

    run_dx = (7.0, 1.3)
    e = Ellipse(run_dx, 2.9, 1.0, facecolor='#fef3c7', edgecolor=AMBER, linewidth=1.8, zorder=2)
    ax.add_patch(e)
    ax.text(run_dx[0], run_dx[1], 'Run AI Diagnosis', ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

    for label in ["Login", "Register Account", "Logout", "Enter Patient\nInformation", "View Patient\nHistory"]:
        x, y = centers[label.split('\n')[0]]
        ax.plot([1.45, x - 1.2], [4.6, y], color=DARK, lw=1.1, zorder=1)

    for label in ["Upload Chest\nX-ray Image", "Upload Brain\nMRI Scan", "Upload Skin\nLesion Photo", "Enter Vitals /\nHealth Indicators"]:
        x, y = centers[label.split('\n')[0]]
        arrow(ax, (x - 1.25, y - 0.1), (run_dx[0] + 0.6, run_dx[1] + 0.45), style='-|>', ls=(0, (4, 3)), color=GRAY,
              connectionstyle=f'arc3,rad={0.15 if y>run_dx[1] else -0.15}')
    ax.text(5.4, 2.9, '<<include>>', fontsize=7.5, color=GRAY, style='italic')

    sub = [("Detect Pneumonia", 5.6, -0.0), ("Classify Brain Tumor", 7.4, -0.0)]
    # place sub-diagnoses below boundary isn't ideal; instead annotate inside run_dx ellipse
    ax.text(run_dx[0], run_dx[1]-0.38, '(Pneumonia · Brain Tumor · Skin Lesion · Diabetes Risk)', ha='center', fontsize=6.6, color=GRAY)

    for label in ["Generate Grad-CAM\nHeatmap", "Download PDF\nReport"]:
        x, y = centers[label.split('\n')[0]]
        arrow(ax, (run_dx[0]+0.9, run_dx[1]+0.1), (x-1.25, y), style='-|>', ls=(0, (4, 3)), color=GRAY,
              connectionstyle='arc3,rad=-0.1')
    ax.text(8.4, 1.9, '<<extend>>', fontsize=7.5, color=GRAY, style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '1_use_case_diagram.png'), dpi=180)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# 2. CLASS DIAGRAM
# ════════════════════════════════════════════════════════════════════
def class_diagram():
    fig, ax = new_fig(15, 11.5)
    ax.text(7.5, 11.1, 'Class Diagram — MediAI Diagnostics', ha='center', fontsize=14, fontweight='bold')

    top = 10.6
    ux, uy, uw, uh = class_box(ax, 0.4, top, 3.0, 'User',
              ['username: String', 'email: String', 'password: String', 'created_at: DateTime'],
              ['register()', 'login()'], ec=NAVY)

    px, py, pw, ph = class_box(ax, 4.0, top, 4.0, 'PredictionRecord',
              ['patient_id: String', 'patient_name: String', 'patient_age: String', 'patient_gender: String',
               'module: String', 'diagnosis: String', 'risk_score: Int', 'probabilities: Dict', 'timestamp: DateTime'],
              ['save()'], ec=NAVY)

    rx, ry, rw, rh = class_box(ax, 8.6, top, 3.2, 'ReportGenerator',
              ['patient: Dict', 'module: String'],
              ['generate_report() : PDF'], ec=GREEN)

    gx, gy, gw, gh = class_box(ax, 12.3, top, 2.3, 'GradCAMGenerator',
              ['model', 'target_layer'],
              ['generate(image)'], ec=AMBER)

    dtop = 6.6
    dx, dy, dw, dh = class_box(ax, 5.6, dtop, 3.8, 'DiagnosisModel\n«abstract»',
              ['device: String', 'model_path: String'],
              ['predict(input) : Result'], ec=PURPLE)

    subclasses = [
        ('XrayModel', 'ResNet-50 · 2 classes', 0.3),
        ('BrainTumorModel', 'EfficientNet-B3 · 4 classes', 3.9),
        ('SkinLesionModel', 'EfficientNet-B3 · 7 classes', 7.5),
        ('VitalsModel', 'MLP (BatchNorm) · 2 classes', 11.1),
    ]
    sub_top = 2.8
    for name, note, x in subclasses:
        sx, sy, sw, sh = class_box(ax, x, sub_top, 3.2, name, [note], ['predict(input) : Result'], ec=PURPLE)
        arrow(ax, (sx + sw/2, sy + sh), (dx + dw/2, dy), style='-|>', color=PURPLE, connectionstyle='arc3,rad=0.0')

    arrow(ax, (ux + uw, uy), (px, py), style='-|>', color=NAVY, connectionstyle='arc3,rad=-0.3')
    ax.text((ux + uw + px) / 2, min(uy, py) - 0.55, '1   creates   *', fontsize=7.5, color=GRAY, ha='center')

    arrow(ax, (px + 0.6, py), (dx + dw - 0.3, dy + dh), style='-|>', color=NAVY, connectionstyle='arc3,rad=-0.15')
    ax.text(px + 0.0, py - 0.35, 'produced by', fontsize=7.5, color=GRAY)

    arrow(ax, (rx + 0.3, ry), (dx + dw/2 + 0.6, dy + dh), style='-|>', color=GREEN, connectionstyle='arc3,rad=0.15')
    ax.text(rx - 0.4, ry - 0.35, 'uses', fontsize=7.5, color=GRAY)

    arrow(ax, (gx + gw/2, gy), (dx + dw - 0.1, dy + dh + 0.3), style='-|>', color=AMBER, connectionstyle='arc3,rad=0.25')
    ax.text(gx - 0.3, gy - 0.35, 'visualizes', fontsize=7.5, color=GRAY)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '2_class_diagram.png'), dpi=180)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# 3. ACTIVITY DIAGRAM (SWIMLANE)
# ════════════════════════════════════════════════════════════════════
def activity_diagram():
    fig, ax = new_fig(13, 15)
    ax.text(6.5, 14.6, 'Activity Diagram (Swimlane) — Run AI Diagnosis', ha='center', fontsize=14, fontweight='bold')

    lanes = ['User', 'Frontend (React)', 'Backend (FastAPI)', 'AI Model / MongoDB']
    lane_w = 3.0
    colors = [NAVY, PURPLE, AMBER, GREEN]
    top_y, bot_y = 14.0, 0.3
    for i, (lane, c) in enumerate(zip(lanes, colors)):
        x0 = 0.5 + i * lane_w
        ax.add_patch(Rectangle((x0, bot_y), lane_w, top_y - bot_y, facecolor='white', edgecolor=GRAY, linewidth=1.0, zorder=0))
        ax.add_patch(Rectangle((x0, top_y - 0.1), lane_w, 0.4, facecolor=c, edgecolor=c, zorder=1))
        ax.text(x0 + lane_w/2, top_y + 0.1, lane, ha='center', va='center', color='white', fontsize=9.5, fontweight='bold', zorder=2)
    lane_cx = [0.5 + i * lane_w + lane_w/2 for i in range(4)]

    def act(lane_i, y, text, w=2.6, h=0.7, diamond=False, fc=None, ec=None):
        x = lane_cx[lane_i]
        ec = ec or colors[lane_i]
        if diamond:
            pts = [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)]
            ax.add_patch(plt.Polygon(pts, closed=True, facecolor='#fef3c7', edgecolor=AMBER, linewidth=1.3, zorder=2))
            ax.text(x, y, text, ha='center', va='center', fontsize=7.0, zorder=3)
        else:
            fc = fc or ('#eff6ff' if lane_i < 2 else '#fff7ed' if lane_i == 2 else '#ecfdf5')
            box(ax, x - w/2, y - h/2, w, h, text, fc=fc, ec=ec, fontsize=7.8, bold=False)
        return (x, y)

    # Explicit, generously-spaced row centers (top to bottom) — chosen so that every
    # box/diamond's vertical extent (±h/2) never overlaps its neighbour's.
    R = {'login': 13.1, 'patient': 12.1, 'upload': 11.1, 'click': 10.1,
         'validimg': 8.7, 'infer': 7.3, 'conf': 5.9,
         'diag': 4.9, 'uncertain': 3.9, 'reject': 3.4,
         'save': 2.3, 'display': 1.3}

    act(0, R['login'], 'Login')
    act(0, R['patient'], 'Enter Patient Info\n& Select Module')
    act(0, R['upload'], 'Upload Image /\nEnter Vitals')
    act(0, R['click'], 'Click "Run AI\nDiagnosis"')

    act(1, R['click'], 'Validate Form,\nPOST /predict-*')
    act(2, R['click'], 'Receive Request')

    act(2, R['validimg'], 'Valid\nImage?', diamond=True, w=2.0, h=0.9)
    act(2, R['infer'], 'Preprocess &\nRun Inference')
    act(3, R['infer'], 'Model Forward Pass\n(PyTorch) + softmax')
    act(2, R['conf'], 'Confidence ≥\nthreshold?', diamond=True, w=2.0, h=0.9)

    act(3, R['diag'], 'Build Diagnosis Result', w=2.7, h=0.55, fc='#ecfdf5', ec=GREEN)
    act(3, R['uncertain'], 'Build "Uncertain" Result', w=2.7, h=0.55, fc='#fff7ed', ec=AMBER)
    reject = act(2, R['reject'], 'Build Error /\nReject Response', w=2.7, h=0.6, fc='#fee2e2', ec='#dc2626')

    act(3, R['save'], 'Save Prediction (MongoDB)', w=2.7, h=0.6)
    act(1, R['display'], 'Receive & Display\nResult Card')
    act(0, R['display'], 'View Result → Heatmap /\nPDF (optional)')

    # ── User column, top to bottom ──
    arrow(ax, (lane_cx[0], R['login']-0.35), (lane_cx[0], R['patient']+0.35), color=DARK)
    arrow(ax, (lane_cx[0], R['patient']-0.35), (lane_cx[0], R['upload']+0.35), color=DARK)
    arrow(ax, (lane_cx[0], R['upload']-0.35), (lane_cx[0], R['click']+0.35), color=DARK)

    # ── User -> Frontend -> Backend (same row) ──
    arrow(ax, (lane_cx[0]+1.3, R['click']), (lane_cx[1]-1.3, R['click']), color=DARK)
    arrow(ax, (lane_cx[1]+1.3, R['click']), (lane_cx[2]-1.0, R['click']), color=DARK)

    # ── Backend column down to the two decision diamonds ──
    arrow(ax, (lane_cx[2], R['click']-0.35), (lane_cx[2], R['validimg']+0.45), color=DARK)
    arrow(ax, (lane_cx[2], R['validimg']-0.45), (lane_cx[2], R['infer']+0.35), color=NAVY)
    ax.text(lane_cx[2]+0.3, (R['validimg']+R['infer'])/2, 'yes', fontsize=6.8, color=NAVY)
    arrow(ax, (lane_cx[2]-0.9, R['validimg']-0.15), (reject[0]-0.3, reject[1]+0.3), color='#dc2626', connectionstyle='arc3,rad=0.3')
    ax.text(lane_cx[2]-1.7, R['validimg']-0.7, 'no', fontsize=6.8, color='#dc2626')

    arrow(ax, (lane_cx[2], R['infer']-0.35), (lane_cx[2], R['conf']+0.45), color=DARK)

    # ── Backend <-> AI Model (inference), same row ──
    arrow(ax, (lane_cx[2]+1.0, R['infer']), (lane_cx[3]-1.35, R['infer']), color=DARK)

    # ── confidence diamond -> two stacked result boxes (AI Model lane) ──
    arrow(ax, (lane_cx[2]+1.0, R['conf']+0.1), (lane_cx[3]-1.35, R['diag']), color=GREEN, connectionstyle='arc3,rad=-0.15')
    ax.text(lane_cx[2]+1.5, R['conf']+0.45, 'yes', fontsize=6.8, color=GREEN)
    arrow(ax, (lane_cx[2]+1.0, R['conf']-0.1), (lane_cx[3]-1.35, R['uncertain']), color=AMBER, connectionstyle='arc3,rad=0.15')
    ax.text(lane_cx[2]+1.5, R['conf']-0.45, 'no', fontsize=6.8, color=AMBER)

    # ── both result boxes -> Save Prediction ──
    arrow(ax, (lane_cx[3], R['diag']-0.3), (lane_cx[3], R['save']+0.3), color=DARK, connectionstyle='arc3,rad=-0.12')
    arrow(ax, (lane_cx[3], R['uncertain']-0.3), (lane_cx[3], R['save']+0.3), color=DARK, connectionstyle='arc3,rad=0.12')

    # ── Save Prediction -> Frontend display ; reject -> Frontend display ──
    arrow(ax, (lane_cx[3]-1.35, R['save']), (lane_cx[1]+1.3, R['display']), color=DARK, connectionstyle='arc3,rad=0.22')
    arrow(ax, (reject[0]-1.35, reject[1]-0.1), (lane_cx[1]+1.3, R['display']+0.05), color='#dc2626', connectionstyle='arc3,rad=-0.12')

    # ── Frontend -> User (final) ──
    arrow(ax, (lane_cx[1]-1.3, R['display']), (lane_cx[0]+1.3, R['display']), color=DARK)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '3_activity_diagram_swimlane.png'), dpi=180)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# 4. SEQUENCE DIAGRAM
# ════════════════════════════════════════════════════════════════════
def sequence_diagram():
    fig, ax = new_fig(13, 10)
    ax.text(6.5, 9.6, 'Sequence Diagram — Run AI Diagnosis (Chest X-ray + Diabetes)', ha='center', fontsize=13, fontweight='bold')

    objs = [('User', 1.0), ('ReactFrontend', 3.4), ('FastAPIBackend', 6.1), ('PredictModule', 8.4), ('PyTorchModel', 10.3), ('MongoDB', 12.1)]
    top = 8.9
    for name, x in objs:
        box(ax, x - 0.85, top, 1.7, 0.5, ':' + name, fc='#eff6ff', ec=NAVY, fontsize=8.3)
        ax.plot([x, x], [top, 0.6], color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1)

    xs = {n: x for n, x in objs}

    def msg(frm, to, y, text, dashed=False, fontsize=7.6):
        arrow(ax, (xs[frm], y), (xs[to], y), style='-|>', color=DARK, ls=(0, (4, 3)) if dashed else '-')
        mx = (xs[frm] + xs[to]) / 2
        ax.text(mx, y + 0.1, text, ha='center', va='bottom', fontsize=fontsize)

    y = 8.3
    msg('User', 'ReactFrontend', y, '1: click "Run AI Diagnosis"'); y -= 0.55
    msg('ReactFrontend', 'FastAPIBackend', y, '2: POST /predict-xray(image)'); y -= 0.55
    msg('FastAPIBackend', 'PredictModule', y, '3: predict_xray(bytes)'); y -= 0.55
    msg('PredictModule', 'PredictModule', y, '4: _open_image() / _check_xray_like()', fontsize=6.8); y -= 0.5
    msg('PredictModule', 'PyTorchModel', y, '5: forward(tensor)'); y -= 0.55
    msg('PyTorchModel', 'PredictModule', y, '6: logits', dashed=True); y -= 0.55
    msg('PredictModule', 'PredictModule', y, '7: softmax + confidence check', fontsize=6.8); y -= 0.5
    msg('PredictModule', 'FastAPIBackend', y, '8: diagnosis, risk_score, probabilities', dashed=True); y -= 0.55
    msg('FastAPIBackend', 'ReactFrontend', y, '9: 200 OK (JSON result)', dashed=True); y -= 0.55
    msg('ReactFrontend', 'FastAPIBackend', y, '10: POST /save-prediction'); y -= 0.55
    msg('FastAPIBackend', 'MongoDB', y, '11: insert_one(prediction)'); y -= 0.55
    msg('MongoDB', 'FastAPIBackend', y, '12: inserted_id', dashed=True); y -= 0.55
    msg('FastAPIBackend', 'ReactFrontend', y, '13: 200 OK', dashed=True); y -= 0.55
    msg('ReactFrontend', 'User', y, '14: display Result Card', dashed=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '4_sequence_diagram.png'), dpi=180)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# 5. STATE CHART DIAGRAM
# ════════════════════════════════════════════════════════════════════
def state_chart_diagram():
    fig, ax = new_fig(13, 9)
    ax.text(6.5, 8.6, 'State Chart Diagram — Diagnosis Request Lifecycle', ha='center', fontsize=14, fontweight='bold')

    def state(x, y, text, w=2.3, h=0.8, fc='#eff6ff', ec=NAVY):
        box(ax, x - w/2, y - h/2, w, h, text, fc=fc, ec=ec, fontsize=8.3, bold=False)
        return (x, y)

    s_start = (0.6, 7.5)
    ax.add_patch(plt.Circle(s_start, 0.14, facecolor=DARK, zorder=3))
    idle = state(2.0, 7.5, 'Idle')
    arrow(ax, (s_start[0]+0.14, s_start[1]), (idle[0]-1.15, idle[1]))

    info = state(4.6, 7.5, 'PatientInfo\nEntered')
    arrow(ax, (idle[0]+1.15, idle[1]), (info[0]-1.15, info[1]), label='enter patient info', fontsize=6.8)

    uploaded = state(7.3, 7.5, 'ImageUploaded /\nVitalsEntered')
    arrow(ax, (info[0]+1.15, info[1]), (uploaded[0]-1.15, uploaded[1]), label='upload / fill form', fontsize=6.8)

    analyzing = state(10.2, 7.5, 'Analyzing', fc='#fef3c7', ec=AMBER)
    arrow(ax, (uploaded[0]+1.15, uploaded[1]), (analyzing[0]-1.15, analyzing[1]), label='click Run Diagnosis', fontsize=6.8)

    invalid = state(10.2, 5.7, 'Invalid', fc='#fee2e2', ec='#dc2626')
    arrow(ax, (analyzing[0], analyzing[1]-0.4), (invalid[0], invalid[1]+0.4), label='validation failed', fontsize=6.8)
    arrow(ax, (invalid[0]-1.0, invalid[1]+0.1), (uploaded[0]+0.3, uploaded[1]-0.9), label='re-upload', fontsize=6.6,
          connectionstyle='arc3,rad=0.3')

    result = state(7.3, 5.7, 'ResultReady', fc='#ecfdf5', ec=GREEN)
    arrow(ax, (analyzing[0]-1.15, analyzing[1]-0.2), (result[0]+1.15, result[1]+0.2), label='confidence ≥ threshold', fontsize=6.6)

    uncertain = state(4.6, 5.7, 'Uncertain', fc='#fff7ed', ec=AMBER)
    arrow(ax, (uploaded[0]-0.3, uploaded[1]-0.9), (uncertain[0]+0.3, uncertain[1]+0.6), label='confidence < threshold', fontsize=6.6,
          connectionstyle='arc3,rad=-0.1')
    arrow(ax, (analyzing[0]-1.1, analyzing[1]-0.35), (uncertain[0]+1.2, uncertain[1]+0.35), label='', connectionstyle='arc3,rad=0.35')

    heatmap = state(9.0, 3.9, 'HeatmapGenerated', fc='#ede9fe', ec=PURPLE)
    arrow(ax, (result[0]+0.6, result[1]-0.4), (heatmap[0]-0.3, heatmap[1]+0.5), label='generate heatmap\n(optional)', fontsize=6.4)

    report = state(5.6, 3.9, 'ReportDownloaded', fc='#fff7ed', ec=AMBER)
    arrow(ax, (result[0]-0.4, result[1]-0.4), (report[0]+0.5, report[1]+0.5), label='download PDF\n(optional)', fontsize=6.4)
    arrow(ax, (uncertain[0]-0.1, uncertain[1]-0.4), (report[0]-0.3, report[1]+0.5), label='download PDF\n(optional)', fontsize=6.4,
          connectionstyle='arc3,rad=-0.2')

    saved = state(7.3, 2.1, 'Saved', fc='#dbeafe', ec=NAVY)
    arrow(ax, (result[0], result[1]-0.4), (saved[0]+0.5, saved[1]+1.0), label='auto-save to DB', fontsize=6.4, connectionstyle='arc3,rad=0.25')
    arrow(ax, (heatmap[0]-0.3, heatmap[1]-0.4), (saved[0]+0.9, saved[1]+0.5), connectionstyle='arc3,rad=0.2')
    arrow(ax, (report[0]+0.3, report[1]-0.4), (saved[0]-0.9, saved[1]+0.5), connectionstyle='arc3,rad=-0.2')

    s_end = (7.3, 0.6)
    ax.add_patch(plt.Circle(s_end, 0.16, facecolor='white', edgecolor=DARK, linewidth=1.4, zorder=3))
    ax.add_patch(plt.Circle(s_end, 0.08, facecolor=DARK, zorder=4))
    arrow(ax, (saved[0], saved[1]-0.4), (s_end[0], s_end[1]+0.18))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '5_state_chart_diagram.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    use_case_diagram()
    class_diagram()
    activity_diagram()
    sequence_diagram()
    state_chart_diagram()
    print("All 5 UML diagrams generated in:", OUT)
