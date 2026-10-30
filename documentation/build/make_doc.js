const fs = require('fs');
const path = require('path');
const sizeOf = (() => {
  // Minimal PNG/JPEG dimension reader (no extra deps) — reused from file bytes.
  return function getImageSize(filePath) {
    const buf = fs.readFileSync(filePath);
    if (buf.readUInt32BE(0) === 0x89504e47 || buf.slice(0, 8).toString('hex') === '89504e470d0a1a0a') {
      // PNG: width/height at bytes 16-24
      const width = buf.readUInt32BE(16);
      const height = buf.readUInt32BE(20);
      return { width, height };
    }
    // Fallback JPEG parser
    let offset = 2;
    while (offset < buf.length) {
      if (buf[offset] !== 0xff) break;
      const marker = buf[offset + 1];
      if (marker === 0xc0 || marker === 0xc2) {
        const height = buf.readUInt16BE(offset + 5);
        const width = buf.readUInt16BE(offset + 7);
        return { width, height };
      }
      offset += 2 + buf.readUInt16BE(offset + 2);
    }
    throw new Error('Unsupported image format: ' + filePath);
  };
})();

const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, PageBreak, TableOfContents, ShadingType, VerticalAlign,
  Header, Footer, PageNumber, ExternalHyperlink, TabStopType, TabStopPosition, Tab
} = require('docx');

const ROOT = 'C:\\Users\\ADMIN\\Desktop\\capstone-healthcare-ai';
const SS = path.join(ROOT, 'screenshots');
const DG = path.join(ROOT, 'documentation', 'diagrams');

const FONT = 'Times New Roman';
const NAVY = '1d4ed8';
const DARKGRAY = '444444';

// ── Helpers ──────────────────────────────────────────────────────────────
function fitImage(w, h, maxW = 600, maxH = 720) {
  const scale = Math.min(maxW / w, maxH / h, 1);
  return { width: Math.round(w * scale), height: Math.round(h * scale) };
}

function imgParagraph(filePath, opts = {}) {
  const { width, height } = sizeOf(filePath);
  const dims = fitImage(width, height, opts.maxW || 600, opts.maxH || 700);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [new ImageRun({ data: fs.readFileSync(filePath), transformation: dims, type: path.extname(filePath).slice(1) === 'png' ? 'png' : 'jpg' })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text, italics: true, size: 20, font: FONT, color: DARKGRAY })],
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, spacing: { after: 200 }, children: [new TextRun({ text, font: FONT })] });
}
function h1NoBreak(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 200 }, children: [new TextRun({ text, font: FONT })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 120 }, children: [new TextRun({ text, font: FONT })] });
}
function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 }, children: [new TextRun({ text, font: FONT })] });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: 24, bold: opts.bold || false })],
  });
}
function bullet(text) {
  return new Paragraph({
    spacing: { after: 80 },
    bullet: { level: 0 },
    children: [new TextRun({ text, font: FONT, size: 24 })],
  });
}
function centerBig(text, size, bold = true, after = 0) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after }, children: [new TextRun({ text, font: FONT, size, bold })] });
}
function blank(n = 1) {
  return new Paragraph({ spacing: { after: n * 120 }, children: [] });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: 'DCE6F1' } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, font: FONT, size: 20, bold: !!opts.header })] })],
  });
}

function dataTable(headers, rows, widths) {
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((hd, i) => cell(hd, { header: true, width: widths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, { width: widths[i] })) })),
    ],
  });
}

// ── Screenshot catalog (from visual review) ────────────────────────────────
const CLIENT_SHOTS = [
  ['image.png', 'Landing / Welcome page with project overview and accuracy highlights'],
  ['image copy.png', 'Login screen ("Welcome back")'],
  ['image copy 2.png', 'Chest X-Ray + Diabetes module — patient information and vitals input form'],
  ['image copy 3.png', 'Chest X-Ray + Diabetes — diagnosis result cards with probability breakdown'],
  ['image copy 4.png', 'Brain MRI module — patient info, detectable conditions panel, and uploaded scan preview'],
  ['image copy 5.png', 'Brain MRI — diagnosis result with risk score and probability breakdown'],
  ['image copy 6.png', 'Brain MRI — Grad-CAM AI attention heatmap (original vs. heatmap overlay)'],
  ['image copy 8.png', 'Skin Lesion module — patient info, 7 detectable lesion types, and uploaded photo'],
  ['image copy 9.png', 'Skin Lesion — diagnosis result with 7-class probability breakdown'],
  ['image copy 10.png', 'Skin Lesion — Grad-CAM AI attention heatmap (original vs. heatmap overlay)'],
];
const REPORT_SHOTS = [
  ['image copy 7.png', 'Generated PDF diagnostic report — Brain MRI module'],
  ['image copy 11.png', 'Generated PDF diagnostic report — Skin Lesion module'],
];

// ── Document ────────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 27, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT, italics: true }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: 'MediAI Diagnostics — Capstone Project II', font: FONT, size: 18, color: '888888' })] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: 'Page ', font: FONT, size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }),
    },
    children: [
      // ═══════════════════════ 1. COVER PAGE ═══════════════════════
      blank(2),
      centerBig('GLS UNIVERSITY', 36, true, 80),
      centerBig('Faculty of Computer Applications & Information Technology', 26, true, 240),
      centerBig('iMSc(IT) / iMCA — Semester IX', 24, false, 40),
      centerBig('221601907 — Capstone Project – II', 24, false, 360),
      blank(1),
      centerBig('AI-Powered Multimodal Healthcare Diagnosis &', 28, true, 0),
      centerBig('Clinical Risk Prediction System', 28, true, 40),
      centerBig('("MediAI Diagnostics")', 22, false, 400),
      blank(2),
      centerBig('Submitted By', 22, true, 120),
      centerBig('Shah Khush Prakashbhai   (Enrollment No: 202201619010082, A-84)', 22, false, 40),
      centerBig('Trivedi Naman Ravikumar   (Enrollment No: 202201619010107, A-110)', 22, false, 360),
      blank(1),
      centerBig('Guided By', 22, true, 120),
      centerBig('____________________________', 22, false, 360),
      blank(2),
      centerBig('Academic Year 2026–27', 22, true, 0),

      // ═══════════════════════ 2. COLLEGE CERTIFICATE ═══════════════════════
      new Paragraph({ pageBreakBefore: true, children: [] }),
      centerBig('CERTIFICATE', 30, true, 360),
      p('This is to certify that Mr. Shah Khush Prakashbhai (Enrollment No: 202201619010082) and ' +
        'Mr. Trivedi Naman Ravikumar (Enrollment No: 202201619010107), students of iMSc(IT) / iMCA ' +
        'Semester IX, Faculty of Computer Applications & Information Technology, GLS University, have ' +
        'satisfactorily completed the Capstone Project – II entitled "AI-Powered Multimodal Healthcare ' +
        'Diagnosis & Clinical Risk Prediction System" during the academic year 2026–27, in partial ' +
        'fulfilment of the requirements of the degree, under our guidance and supervision.'),
      p('The work presented in this project is original and has been carried out by the above-named ' +
        'students themselves.'),
      blank(3),
      new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: 9000 }],
        children: [new TextRun({ text: 'Internal Guide', font: FONT, size: 24 }), new TextRun({ text: '\tHead of Department', font: FONT, size: 24 })] }),
      blank(2),
      new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: 9000 }],
        children: [new TextRun({ text: 'External Examiner', font: FONT, size: 24 }), new TextRun({ text: '\tDate: ___________', font: FONT, size: 24 })] }),
      blank(2),
      p('Place: ________________', { align: AlignmentType.LEFT }),

      // ═══════════════════════ 3. INDEX ═══════════════════════
      new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: 'Index', font: FONT })] }),
      new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-3' }),
      p('(Right-click the table above in Microsoft Word and choose "Update Field" to populate page numbers after opening this document.)', { align: AlignmentType.CENTER }),

      // ═══════════════════════ 4. PROBLEM STATEMENT ═══════════════════════
      h1('Problem Statement'),
      p('Early and accurate diagnosis is critical for common conditions such as pneumonia, brain ' +
        'tumors, skin cancer, and diabetes, yet several practical barriers limit timely clinical ' +
        'decision-making in real-world healthcare settings:'),
      bullet('Specialist radiologists, dermatologists, and endocrinologists are not uniformly available, ' +
        'especially in smaller clinics and under-resourced regions, which delays diagnosis.'),
      bullet('Manual interpretation of X-rays, MRI scans, and dermatoscopic images is time-consuming and ' +
        'subject to inter-observer variability, even among experienced clinicians.'),
      bullet('Many existing AI-assisted diagnostic prototypes — including this project’s own earlier ' +
        'Capstone-I baseline — are trained on narrow, single-source datasets and present predictions with ' +
        'no indication of confidence, risking misleadingly confident results on unfamiliar, real-world images.'),
      bullet('Patient records and diagnostic history are often fragmented across paper records or ' +
        'disconnected systems, making longitudinal tracking difficult.'),
      p('This project addresses these problems by building a single, consistent clinical decision-support ' +
        'system that screens four conditions using properly trained deep-learning models, visually explains ' +
        'its reasoning (Grad-CAM), flags low-confidence predictions instead of forcing a guess, generates a ' +
        'standardized clinical report, and keeps a structured, searchable history of every patient’s predictions.'),
      h2('Objectives'),
      bullet('Build and train four independent diagnostic models from properly sourced, larger public ' +
        'datasets using sound machine-learning methodology.'),
      bullet('Provide visual explainability for every image-based prediction via Grad-CAM.'),
      bullet('Add reliability safeguards — out-of-distribution rejection and confidence thresholds — so the ' +
        'system degrades gracefully instead of confidently guessing on invalid or unfamiliar inputs.'),
      bullet('Generate automated, standardized PDF clinical reports for each diagnosis.'),
      bullet('Persist patient and prediction history in a structured, queryable database.'),

      // ═══════════════════════ 5. BRIEF OF SYSTEM ═══════════════════════
      h1('Brief of System'),
      p('MediAI Diagnostics is an AI-powered, multimodal clinical decision-support web application that ' +
        'assists healthcare professionals in screening four conditions from patient data: pneumonia from ' +
        'chest X-rays, brain tumors from MRI scans, skin lesions (including melanoma) from dermatoscopic ' +
        'photographs, and diabetes risk from clinical health indicators. The system is built as a ' +
        'capstone extension of an earlier prototype (Capstone-I), which covered only chest X-ray and ' +
        'brain MRI analysis with a basic web interface.'),
      p('In this phase (Capstone-II), the system has been substantially re-engineered: all four diagnostic ' +
        'models were retrained from scratch on larger, better-sourced public medical datasets using proper ' +
        'machine-learning methodology (stratified train/validation/test splits, class-weighted loss to ' +
        'handle imbalanced disease classes, transfer learning from ImageNet-pretrained backbones, early ' +
        'stopping, and mixed-precision GPU training). A new Skin Lesion Detection module was added, ' +
        'Explainable AI (Grad-CAM) visual heatmaps were implemented for every image-based module, ' +
        'automated PDF diagnostic report generation was built, and a reliability layer was added so that ' +
        'the system can reject invalid/out-of-distribution images and flag low-confidence predictions as ' +
        '"Uncertain" instead of forcing a guess.'),
      p('The application is a full-stack system: a React + Vite single-page frontend communicates with a ' +
        'Python FastAPI backend over a REST API; the backend runs four independent PyTorch deep-learning ' +
        'models for inference, generates Grad-CAM explainability heatmaps with OpenCV, produces downloadable ' +
        'PDF reports with ReportLab, and persists user accounts and every prediction record in MongoDB.'),
      h2('Key Modules'),
      bullet('User Management — secure registration/login for healthcare professionals'),
      bullet('Chest X-ray Analysis — pneumonia detection (ResNet-50)'),
      bullet('Brain MRI Analysis — 4-class brain tumor classification (EfficientNet-B3)'),
      bullet('Skin Lesion Detection — 7-class dermatoscopic lesion classification (EfficientNet-B3)'),
      bullet('Clinical Vitals Module — diabetes risk prediction from 21 health indicators (MLP)'),
      bullet('Explainable AI — Grad-CAM heatmaps for every image-based diagnosis'),
      bullet('Automated PDF Report Generation — downloadable clinical report per diagnosis'),
      bullet('Patient History & Database — MongoDB-backed prediction history per patient'),
      h2('Reliability Features'),
      bullet('Out-of-distribution input rejection — heuristic checks reject images that don\u2019t statistically resemble the expected scan type'),
      bullet('Confidence-threshold gating — predictions below a 65% confidence threshold are reported as "Uncertain" rather than a forced diagnosis'),
      bullet('Checkpointed, resumable model training — resilient to interruptions during multi-hour GPU training runs'),

      // ═══════════════════════ 5. UML DIAGRAMS ═══════════════════════
      h1('UML Diagrams'),
      h2('1. Use Case Diagram'),
      p('Shows the Healthcare Professional actor interacting with the system\u2019s core use cases — account ' +
        'management, patient data entry, uploading scans/photos for each of the four diagnostic modules, and ' +
        'the supporting "Run AI Diagnosis" use case that includes the four module-specific detections and ' +
        'extends to heatmap generation and PDF report download.'),
      imgParagraph(path.join(DG, '1_use_case_diagram.png'), { maxH: 620 }),
      caption('Fig. 5.1 — Use Case Diagram'),

      h2('2. Class Diagram'),
      p('Models the system\u2019s core domain classes: User and PredictionRecord (persisted in MongoDB), the ' +
        'abstract DiagnosisModel with its four concrete PyTorch model subclasses, and the supporting ' +
        'GradCAMGenerator and ReportGenerator service classes.'),
      imgParagraph(path.join(DG, '2_class_diagram.png'), { maxH: 680 }),
      caption('Fig. 5.2 — Class Diagram'),

      h2('3. Activity Diagram (Swimlane)'),
      p('Traces the end-to-end "Run AI Diagnosis" flow across four responsibility lanes — User, Frontend ' +
        '(React), Backend (FastAPI), and AI Model/MongoDB — including both decision points (image validity ' +
        'and confidence-threshold checks) and their alternate paths.'),
      imgParagraph(path.join(DG, '3_activity_diagram_swimlane.png'), { maxH: 700 }),
      caption('Fig. 5.3 — Activity Diagram (Swimlane)'),

      h2('4. Sequence Diagram'),
      p('Details the object-level message flow for the Chest X-ray + Diabetes diagnosis flow, from the ' +
        'user\u2019s click through the frontend, backend, prediction module, PyTorch model, and the final ' +
        'MongoDB save.'),
      imgParagraph(path.join(DG, '4_sequence_diagram.png'), { maxH: 650 }),
      caption('Fig. 5.4 — Sequence Diagram'),

      h2('5. State Chart Diagram'),
      p('Models the lifecycle of a single diagnosis request, from Idle through patient-info entry, image ' +
        'upload, analysis, and its terminal states (ResultReady / Uncertain / Invalid), including the ' +
        'optional heatmap-generation and report-download transitions before the record is saved.'),
      imgParagraph(path.join(DG, '5_state_chart_diagram.png'), { maxH: 620 }),
      caption('Fig. 5.5 — State Chart Diagram'),

      // ═══════════════════════ 6. DATA DICTIONARY ═══════════════════════
      h1('Data Dictionary'),
      p('The system uses MongoDB (a NoSQL document database) with two primary collections.'),
      h2('Collection: users'),
      dataTable(
        ['Field', 'Type', 'Description'],
        [
          ['_id', 'ObjectId', 'MongoDB-generated unique document identifier'],
          ['username', 'String', 'Full name of the registered healthcare professional'],
          ['email', 'String', 'Login identifier; must be unique'],
          ['password', 'String', 'Account password (plain string field in the current schema)'],
          ['created_at', 'DateTime', 'Account creation timestamp (UTC, auto-generated)'],
        ],
        [2000, 1600, 5400]
      ),
      blank(1),
      h2('Collection: predictions'),
      dataTable(
        ['Field', 'Type', 'Description'],
        [
          ['_id', 'ObjectId', 'MongoDB-generated unique document identifier'],
          ['patient_id', 'String', 'Clinician-entered patient identifier'],
          ['patient_name', 'String', 'Patient full name'],
          ['patient_age', 'String', 'Patient age'],
          ['patient_gender', 'String', 'Patient gender (Male / Female / Other)'],
          ['module', 'String', 'Diagnostic module used: "xray", "brain", "skin"'],
          ['diagnosis', 'String', 'Predicted class label (or "Uncertain" if below confidence threshold)'],
          ['risk_score', 'Integer', 'Computed risk score, 0–100'],
          ['probabilities', 'Object (Dict)', 'Per-class predicted probability (%), one key per class'],
          ['vitals_diagnosis', 'String', 'Diabetes prediction label (only present for module = "xray")'],
          ['vitals_risk_score', 'Integer', 'Diabetes risk score, 0–100 (only present for module = "xray")'],
          ['vitals_probabilities', 'Object (Dict)', 'Diabetes class probabilities (only present for module = "xray")'],
          ['saved_by', 'String', 'Username of the healthcare professional who ran the diagnosis'],
          ['timestamp', 'DateTime', 'Record creation timestamp (UTC, auto-generated)'],
        ],
        [2200, 1800, 5000]
      ),

      // ═══════════════════════ 7. SCREENSHOTS ═══════════════════════
      h1('All Screenshots with URL'),
      h2('7.1 Admin Side'),
      p('MediAI Diagnostics has a single authenticated user role (Healthcare Professional) and no separate ' +
        'administrator interface in this release — all screens below fall under "Client Side".'),
      h2('7.2 Client Side'),
      p('Application URL: http://localhost:5173/ (React/Vite development server) — all client-side screens ' +
        'below are views within this single-page application.'),
      ...CLIENT_SHOTS.flatMap(([file, cap], i) => [
        imgParagraph(path.join(SS, file), { maxH: 420 }),
        caption(`Fig. 7.${i + 1} — ${cap}   (URL: http://localhost:5173/)`),
      ]),
      h2('7.3 Reports'),
      p('Reports are generated on demand via POST http://127.0.0.1:8000/generate-report and downloaded as a PDF file by the browser.'),
      ...REPORT_SHOTS.flatMap(([file, cap], i) => [
        imgParagraph(path.join(SS, file), { maxH: 620 }),
        caption(`Fig. 7.${CLIENT_SHOTS.length + i + 1} — ${cap}   (Source: POST http://127.0.0.1:8000/generate-report)`),
      ]),

      // ═══════════════════════ 8. REFERENCES ═══════════════════════
      h1('References'),
      h2('Datasets'),
      bullet('Brain Tumor MRI Dataset (deduplicated, 4-class: glioma / meningioma / pituitary / no tumor, 6,507 images) — sourced via Kaggle'),
      bullet('RSNA Pneumonia Detection Challenge Dataset — https://www.kaggle.com/c/rsna-pneumonia-detection-challenge'),
      bullet('CDC BRFSS 2015 Diabetes Health Indicators Dataset (Alex Teboul) — https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset'),
      bullet('HAM10000 ("Human Against Machine with 10000 training images") — https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000'),
      h2('Frameworks & Libraries'),
      bullet('FastAPI — https://fastapi.tiangolo.com/'),
      bullet('PyTorch & Torchvision — https://pytorch.org/'),
      bullet('React — https://react.dev/'),
      bullet('Vite — https://vite.dev/'),
      bullet('MongoDB — https://www.mongodb.com/'),
      bullet('ReportLab (PDF generation) — https://www.reportlab.com/'),
      bullet('scikit-learn — https://scikit-learn.org/'),
      bullet('OpenCV — https://opencv.org/'),

      // ═══════════════════════ 9. CONCLUSION ═══════════════════════
      h1('Conclusion'),
      p('MediAI Diagnostics successfully demonstrates an end-to-end, multimodal AI-assisted clinical ' +
        'decision-support application covering four distinct diagnostic tasks on a single, consistent ' +
        'platform. All four models were trained using sound machine-learning methodology — stratified ' +
        'data splits, class-weighted loss for imbalanced classes, transfer learning, early stopping, and ' +
        'mixed-precision GPU training — achieving strong held-out test performance (98% chest X-ray ' +
        'accuracy, 94.75% brain MRI accuracy, 74.7% skin lesion accuracy, and 78% diabetes risk accuracy). ' +
        'Beyond raw accuracy, the system was deliberately built with reliability in mind: Grad-CAM makes ' +
        'every image-based prediction visually explainable, out-of-distribution and confidence-threshold ' +
        'checks prevent the system from forcing a confident-sounding guess on invalid or unfamiliar inputs, ' +
        'and every diagnosis can be exported as a standardized, shareable PDF clinical report.'),
      p('Testing the system against real-world images sourced outside the training datasets also surfaced ' +
        'an important, honest limitation: a single 2D-slice classifier trained on one curated dataset has a ' +
        'narrower generalization ceiling than a true multi-institutional clinical system, and occasionally ' +
        'misclassifies images that differ in acquisition style, framing, or scanner from its training ' +
        'source. This is a known, well-documented challenge in medical imaging AI rather than an ' +
        'implementation defect, and the project’s confidence-threshold and out-of-distribution safeguards ' +
        'exist specifically to make that limitation visible to the user instead of hiding it behind a ' +
        'falsely confident diagnosis.'),
      p('Future work identified during this project includes: training on multi-source, multi-institutional ' +
        'datasets to further close the real-world generalization gap; adding test-time augmentation and ' +
        'model ensembling; expanding the Skin Lesion module with additional lesion classes such as ' +
        'squamous cell carcinoma; and integrating an LLM-based medical assistant to explain diagnostic ' +
        'reports in plain language and answer clinician questions, as outlined in the original project ' +
        'proposal. Overall, the project meets its stated objectives and delivers a genuinely usable, ' +
        'reliability-conscious foundation for AI-assisted multimodal clinical screening.'),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = path.join(ROOT, 'documentation', 'Capstone_Project_II_Documentation.docx');
  fs.writeFileSync(outPath, buffer);
  console.log('Document written to:', outPath);
});
