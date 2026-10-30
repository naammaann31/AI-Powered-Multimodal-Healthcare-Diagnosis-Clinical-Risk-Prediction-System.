from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import base64
from datetime import datetime
from PIL import Image as PILImage

def generate_report(patient, module, xray_result=None, vitals_result=None,
                    brain_result=None, skin_result=None, heatmap_b64=None, original_image_bytes=None):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    elements = []

    # ── Colors ───────────────────────────────────────────
    PISTACHIO      = colors.HexColor('#5c8f3e')
    LIGHTPISTACHIO = colors.HexColor('#eef7e6')
    PURPLE     = colors.HexColor('#7c3aed')
    LIGHTPURP  = colors.HexColor('#f5f3ff')
    GREEN      = colors.HexColor('#059669')
    RED        = colors.HexColor('#dc2626')
    ORANGE     = colors.HexColor('#d97706')
    LIGHTORANGE = colors.HexColor('#fff7ed')
    GRAY       = colors.HexColor('#64748b')
    LIGHTGRAY  = colors.HexColor('#f8fafc')
    BORDER     = colors.HexColor('#e2e8f0')
    BLACK      = colors.HexColor('#0f172a')

    accent = PURPLE if module == 'brain' else ORANGE if module == 'skin' else PISTACHIO
    lightaccent = LIGHTPURP if module == 'brain' else LIGHTORANGE if module == 'skin' else LIGHTPISTACHIO

    # ── Custom Styles ─────────────────────────────────────
    title_style = ParagraphStyle('title', fontSize=22, textColor=colors.white,
                                  fontName='Helvetica-Bold', alignment=TA_LEFT,
                                  leading=28)
    sub_style = ParagraphStyle('sub', fontSize=10, textColor=colors.white,
                                fontName='Helvetica', alignment=TA_LEFT, leading=14)
    section_style = ParagraphStyle('section', fontSize=12, textColor=accent,
                                    fontName='Helvetica-Bold', leading=16)
    normal_style = ParagraphStyle('normal', fontSize=10, textColor=BLACK,
                                   fontName='Helvetica', leading=14)
    small_style = ParagraphStyle('small', fontSize=8, textColor=GRAY,
                                  fontName='Helvetica', leading=12)
    label_style = ParagraphStyle('label', fontSize=8, textColor=GRAY,
                                  fontName='Helvetica-Bold', leading=10,
                                  spaceAfter=2)
    value_style = ParagraphStyle('value', fontSize=11, textColor=BLACK,
                                  fontName='Helvetica-Bold', leading=14)
    disclaimer_style = ParagraphStyle('disclaimer', fontSize=8, textColor=ORANGE,
                                       fontName='Helvetica', leading=12,
                                       alignment=TA_CENTER)

    # ── HEADER ────────────────────────────────────────────
    header_data = [[
        Paragraph('🏥 MediAI Diagnostics', title_style),
        Paragraph(f'AI-Powered Clinical Diagnosis Report<br/>'
                  f'<font size="9">GLS University Capstone Project 2025-26</font>', sub_style)
    ]]
    header_table = Table(header_data, colWidths=[90*mm, 80*mm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), accent),
        ('PADDING', (0,0), (-1,-1), 14),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [8,8,8,8]),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 6*mm))

    # ── Report Meta ───────────────────────────────────────
    now = datetime.now()
    meta_data = [
        [Paragraph('REPORT DATE', label_style),
         Paragraph('REPORT ID', label_style),
         Paragraph('MODULE', label_style),
         Paragraph('STATUS', label_style)],
        [Paragraph(now.strftime('%d %B %Y'), value_style),
         Paragraph(f'RPT-{now.strftime("%Y%m%d%H%M")}', value_style),
         Paragraph('Brain MRI' if module == 'brain' else 'Skin Lesion' if module == 'skin' else 'Chest X-Ray + Diabetes', value_style),
         Paragraph('✓ Completed', ParagraphStyle('ok', fontSize=11,
                   textColor=GREEN, fontName='Helvetica-Bold', leading=14))]
    ]
    meta_table = Table(meta_data, colWidths=[42*mm, 45*mm, 52*mm, 31*mm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHTGRAY),
        ('BOX', (0,0), (-1,-1), 1, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 5*mm))

    # ── Patient Information ───────────────────────────────
    elements.append(Paragraph('Patient Information', section_style))
    elements.append(Spacer(1, 2*mm))

    patient_data = [
        [Paragraph('FULL NAME', label_style),
         Paragraph('PATIENT ID', label_style),
         Paragraph('AGE', label_style),
         Paragraph('GENDER', label_style)],
        [Paragraph(patient.get('name') or 'N/A', value_style),
         Paragraph(patient.get('id') or 'N/A', value_style),
         Paragraph(str(patient.get('age') or 'N/A'), value_style),
         Paragraph(patient.get('gender') or 'N/A', value_style)]
    ]
    patient_table = Table(patient_data, colWidths=[42*mm, 45*mm, 32*mm, 51*mm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), lightaccent),
        ('BOX', (0,0), (-1,-1), 1.5, accent),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 5*mm))

    # ── Helper: Risk color ────────────────────────────────
    def get_risk_color(score):
        if score >= 70: return RED
        if score >= 40: return ORANGE
        return GREEN

    def get_risk_label(score):
        if score >= 70: return 'HIGH RISK'
        if score >= 40: return 'MEDIUM RISK'
        return 'LOW RISK'

    # ── Diagnosis Results ─────────────────────────────────
    elements.append(Paragraph('AI Diagnosis Results', section_style))
    elements.append(Spacer(1, 2*mm))

    if module == 'xray' and xray_result and vitals_result:
        for result, title in [(xray_result, '🫁 Chest X-Ray Analysis'),
                               (vitals_result, '🩸 Diabetes Risk Analysis')]:
            risk_color = get_risk_color(result['risk_score'])
            risk_label = get_risk_label(result['risk_score'])

            diag_data = [
                [Paragraph(title, ParagraphStyle('t', fontSize=11, textColor=accent,
                           fontName='Helvetica-Bold', leading=14)),
                 Paragraph('', normal_style)],
                [Paragraph('DIAGNOSIS', label_style),
                 Paragraph('RISK SCORE', label_style)],
                [Paragraph(result['diagnosis'], ParagraphStyle('diag', fontSize=14,
                           textColor=BLACK, fontName='Helvetica-Bold', leading=18)),
                 Paragraph(f"{result['risk_score']}/100 — {risk_label}",
                           ParagraphStyle('risk', fontSize=12, textColor=risk_color,
                                          fontName='Helvetica-Bold', leading=16))]
            ]
            diag_table = Table(diag_data, colWidths=[85*mm, 85*mm])
            diag_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), lightaccent),
                ('BACKGROUND', (0,1), (-1,-1), colors.white),
                ('BOX', (0,0), (-1,-1), 1, BORDER),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                ('PADDING', (0,0), (-1,-1), 8),
                ('SPAN', (0,0), (-1,0)),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(diag_table)
            elements.append(Spacer(1, 3*mm))

            # Probabilities
            prob_rows = [[Paragraph('CONDITION', label_style),
                          Paragraph('PROBABILITY', label_style)]]
            for disease, prob in result['probabilities'].items():
                prob_rows.append([
                    Paragraph(disease, normal_style),
                    Paragraph(f'{prob}%', ParagraphStyle('prob', fontSize=10,
                              textColor=accent, fontName='Helvetica-Bold', leading=14))
                ])
            prob_table = Table(prob_rows, colWidths=[120*mm, 50*mm])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), LIGHTGRAY),
                ('BOX', (0,0), (-1,-1), 1, BORDER),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                ('PADDING', (0,0), (-1,-1), 7),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHTGRAY]),
            ]))
            elements.append(prob_table)
            elements.append(Spacer(1, 4*mm))

    elif module == 'brain' and brain_result:
        risk_color = get_risk_color(brain_result['risk_score'])
        risk_label = get_risk_label(brain_result['risk_score'])

        diag_data = [
            [Paragraph('🧠 Brain MRI Tumor Analysis', ParagraphStyle('t', fontSize=11,
                       textColor=accent, fontName='Helvetica-Bold', leading=14)),
             Paragraph('', normal_style)],
            [Paragraph('DIAGNOSIS', label_style),
             Paragraph('RISK SCORE', label_style)],
            [Paragraph(brain_result['diagnosis'], ParagraphStyle('diag', fontSize=14,
                       textColor=BLACK, fontName='Helvetica-Bold', leading=18)),
             Paragraph(f"{brain_result['risk_score']}/100 — {risk_label}",
                       ParagraphStyle('risk', fontSize=12, textColor=risk_color,
                                      fontName='Helvetica-Bold', leading=16))]
        ]
        diag_table = Table(diag_data, colWidths=[85*mm, 85*mm])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), lightaccent),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('BOX', (0,0), (-1,-1), 1, BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
            ('PADDING', (0,0), (-1,-1), 8),
            ('SPAN', (0,0), (-1,0)),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(diag_table)
        elements.append(Spacer(1, 3*mm))

        prob_rows = [[Paragraph('TUMOR TYPE', label_style),
                      Paragraph('PROBABILITY', label_style)]]
        for disease, prob in brain_result['probabilities'].items():
            prob_rows.append([
                Paragraph(disease, normal_style),
                Paragraph(f'{prob}%', ParagraphStyle('prob', fontSize=10,
                          textColor=accent, fontName='Helvetica-Bold', leading=14))
            ])
        prob_table = Table(prob_rows, colWidths=[120*mm, 50*mm])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), LIGHTGRAY),
            ('BOX', (0,0), (-1,-1), 1, BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
            ('PADDING', (0,0), (-1,-1), 7),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHTGRAY]),
        ]))
        elements.append(prob_table)
        elements.append(Spacer(1, 4*mm))

    elif module == 'skin' and skin_result:
        risk_color = get_risk_color(skin_result['risk_score'])
        risk_label = get_risk_label(skin_result['risk_score'])

        diag_data = [
            [Paragraph('🩹 Skin Lesion Analysis', ParagraphStyle('t', fontSize=11,
                       textColor=accent, fontName='Helvetica-Bold', leading=14)),
             Paragraph('', normal_style)],
            [Paragraph('DIAGNOSIS', label_style),
             Paragraph('RISK SCORE', label_style)],
            [Paragraph(skin_result['diagnosis'], ParagraphStyle('diag', fontSize=14,
                       textColor=BLACK, fontName='Helvetica-Bold', leading=18)),
             Paragraph(f"{skin_result['risk_score']}/100 — {risk_label}",
                       ParagraphStyle('risk', fontSize=12, textColor=risk_color,
                                      fontName='Helvetica-Bold', leading=16))]
        ]
        diag_table = Table(diag_data, colWidths=[85*mm, 85*mm])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), lightaccent),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('BOX', (0,0), (-1,-1), 1, BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
            ('PADDING', (0,0), (-1,-1), 8),
            ('SPAN', (0,0), (-1,0)),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(diag_table)
        elements.append(Spacer(1, 3*mm))

        prob_rows = [[Paragraph('LESION TYPE', label_style),
                      Paragraph('PROBABILITY', label_style)]]
        for disease, prob in skin_result['probabilities'].items():
            prob_rows.append([
                Paragraph(disease, normal_style),
                Paragraph(f'{prob}%', ParagraphStyle('prob', fontSize=10,
                          textColor=accent, fontName='Helvetica-Bold', leading=14))
            ])
        prob_table = Table(prob_rows, colWidths=[120*mm, 50*mm])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), LIGHTGRAY),
            ('BOX', (0,0), (-1,-1), 1, BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
            ('PADDING', (0,0), (-1,-1), 7),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHTGRAY]),
        ]))
        elements.append(prob_table)
        elements.append(Spacer(1, 4*mm))

    # ── Images: Original + Heatmap ────────────────────────
    if original_image_bytes or heatmap_b64:
        elements.append(Paragraph('Medical Imaging', section_style))
        elements.append(Spacer(1, 2*mm))
        img_cells = []

        if original_image_bytes:
            orig_buf = io.BytesIO(original_image_bytes)
            pil_img = PILImage.open(orig_buf).convert('RGB')
            pil_img.thumbnail((300, 300))
            img_buf = io.BytesIO()
            pil_img.save(img_buf, format='JPEG')
            img_buf.seek(0)
            rl_img = RLImage(img_buf, width=75*mm, height=75*mm)
            img_cells.append([Paragraph('ORIGINAL SCAN', label_style), rl_img])

        if heatmap_b64:
            heatmap_bytes = base64.b64decode(heatmap_b64)
            heatmap_buf = io.BytesIO(heatmap_bytes)
            rl_heatmap = RLImage(heatmap_buf, width=75*mm, height=75*mm)
            img_cells.append([Paragraph('AI ATTENTION HEATMAP (Grad-CAM)', label_style), rl_heatmap])

        if img_cells:
            img_row_labels = [cell[0] for cell in img_cells]
            img_row_images = [cell[1] for cell in img_cells]
            col_w = [85*mm] * len(img_cells)

            img_table = Table(
                [img_row_labels, img_row_images],
                colWidths=col_w
            )
            img_table.setStyle(TableStyle([
                ('BOX', (0,0), (-1,-1), 1, BORDER),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                ('PADDING', (0,0), (-1,-1), 8),
                ('BACKGROUND', (0,0), (-1,0), LIGHTGRAY),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(img_table)
            elements.append(Spacer(1, 4*mm))

            elements.append(Paragraph(
                '🔴 Red/Yellow = High AI attention (abnormality detected)   '
                '🔵 Blue = Low attention (normal regions)',
                small_style))
            elements.append(Spacer(1, 4*mm))

    # ── Disclaimer ────────────────────────────────────────
    elements.append(HRFlowable(width='100%', thickness=1, color=BORDER))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(
        'MEDICAL DISCLAIMER: This report is generated by an AI-assisted diagnostic system '
        'and is intended for clinical decision support purposes only. It should NOT be used as '
        'a substitute for professional medical advice, diagnosis, or treatment. Always consult '
        'a qualified healthcare professional for medical decisions.',
        disclaimer_style))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        f'Generated on {now.strftime("%d %B %Y at %H:%M")} | '
        f'MediAI Diagnostics | GLS University Capstone 2025-26',
        ParagraphStyle('footer', fontSize=8, textColor=GRAY,
                       fontName='Helvetica', alignment=TA_CENTER)))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()