"""
Generates the Short Pitch Presentation in both PPTX (.pptx) and PDF (.pdf)
for the Snapdragon Competition Submission.
Widescreen 16:9 layout with high-impact Qualcomm branding colors.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

PPTX_PATH = r"C:\Aisnapdragon\SilentEcho_Pitch_Presentation.pptx"
PDF_SLIDES_PATH = r"C:\Aisnapdragon\SilentEcho_Pitch_Presentation.pdf"

# Palette: Qualcomm Navy (#090D16), Qualcomm Red (#D9381E), Slate (#1E293B), Emerald (#10B981), Cyan (#0284C7)
C_DARK_BG = RGBColor(9, 13, 22)
C_CARD_BG = RGBColor(19, 27, 44)
C_RED = RGBColor(217, 56, 30)
C_WHITE = RGBColor(255, 255, 255)
C_SLATE = RGBColor(148, 163, 184)
C_EMERALD = RGBColor(16, 185, 129)
C_CYAN = RGBColor(2, 132, 199)


def add_slide_header(slide, title_text, category="PROJECT SILENTECHO | SNAPDRAGON X ELITE"):
    # Category badge
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.0), Inches(0.4))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = "Calibri"
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = C_RED

    # Title text
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.5), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = C_WHITE


def set_slide_background(slide, color=C_DARK_BG):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def generate_pptx():
    prs = Presentation()
    # Widescreen 16:9 (13.33 x 7.5 inches)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Accent top border
    top_line = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_line.fill.solid()
    top_line.fill.fore_color.rgb = C_RED
    top_line.line.color.rgb = C_RED

    # Title & Subtitle block
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.5))
    tf = t_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "QUALCOMM SNAPDRAGON X INNOVATION CHALLENGE"
    p0.font.name = "Calibri"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = C_RED
    p0.space_after = Pt(12)

    p1 = tf.add_paragraph()
    p1.text = "Project SilentEcho"
    p1.font.name = "Calibri"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = C_WHITE
    p1.space_after = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "On-Device Multi-Modal Silent Speech & Acoustic Camouflage Engine"
    p2.font.name = "Calibri"
    p2.font.size = Pt(22)
    p2.font.color.rgb = C_CYAN
    p2.space_after = Pt(16)

    p3 = tf.add_paragraph()
    p3.text = "Sub-20ms Real-Time Articulation Reconstruction | 100% Privacy & Zero Cloud Egress"
    p3.font.name = "Calibri"
    p3.font.size = Pt(14)
    p3.font.color.rgb = C_SLATE

    # Footer Card
    f_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(5.5), Inches(11.333), Inches(1.2))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = C_CARD_BG
    f_box.line.color.rgb = C_RED
    f_tf = f_box.text_frame
    f_tf.word_wrap = True
    fp = f_tf.paragraphs[0]
    fp.text = "Author: Srujan Kandandla | Hardware: Qualcomm Hexagon NPU (45 TOPS) | Windows 11 on ARM64 (HP OmniBook X)"
    fp.font.name = "Calibri"
    fp.font.size = Pt(12)
    fp.font.bold = True
    fp.font.color.rgb = C_WHITE

    # =========================================================================
    # SLIDE 2: Problem Statement
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_slide_header(s2, "The Problem: Acoustic Eavesdropping & The Silent Speech Dilemma")

    cards_data = [
        ("1. Acoustic Eavesdropping & Privacy Loss",
         "Speaking aloud in coffee shops, open offices, airports, or transit exposes sensitive corporate IP, banking credentials, and private conversations to nearby microphones and bystanders.",
         C_RED),
        ("2. Visual Lip-Reading Fails in Isolation",
         "Visual-only lip reading suffers from severe homophene confusion—phonemes like /p/, /b/, and /m/ create identical lip closures on camera ('pat', 'bat', 'mat'), causing unacceptable error rates.",
         C_CYAN),
        ("3. Whisper Acoustics Fail in Isolation",
         "Sub-vocal whispers lack fundamental frequency (F0) glottal pulses. They are masked by laptop fan rumble and chassis vibrations, rendering traditional speech recognizers useless.",
         C_EMERALD),
    ]

    for i, (title, desc, accent) in enumerate(cards_data):
        c_shape = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 3.9), Inches(2.2), Inches(3.7), Inches(4.5))
        c_shape.fill.solid()
        c_shape.fill.fore_color.rgb = C_CARD_BG
        c_shape.line.color.rgb = accent
        c_shape.line.width = Pt(1.5)

        ctf = c_shape.text_frame
        ctf.word_wrap = True
        cp0 = ctf.paragraphs[0]
        cp0.text = title
        cp0.font.name = "Calibri"
        cp0.font.size = Pt(16)
        cp0.font.bold = True
        cp0.font.color.rgb = C_WHITE
        cp0.space_after = Pt(14)

        cp1 = ctf.add_paragraph()
        cp1.text = desc
        cp1.font.name = "Calibri"
        cp1.font.size = Pt(12.5)
        cp1.font.color.rgb = C_SLATE

    # =========================================================================
    # SLIDE 3: The Multi-Modal Solution
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_slide_header(s3, "The SilentEcho Solution: Dual-Stream Temporal Cross-Attention")

    sol_cards = [
        ("Stream A: Visual Kinematics",
         "• Windows Media Foundation (MSMF) hardware capture at 30/60 FPS.\n"
         "• Real-time facial landmark tracking with EMA coordinate smoothing.\n"
         "• 40x80 grayscale lip ROI normalized into (1, 16, 1, 40, 80) tensor.\n"
         "• Resolves unvoiced whispered homophones ('pie' vs. 'tie').",
         C_CYAN),
        ("Stream B: Sub-Vocal Acoustics",
         "• 16kHz mono audio from HP Poly Studio onboard studio array.\n"
         "• 4th-order Butterworth bandpass (120Hz-7200Hz) rejects fan noise.\n"
         "• Whisper pre-emphasis (y[t]=x[t]-0.97*x[t-1]) boosts fricatives.\n"
         "• 80-bin Log-Mel spectrogram tensor (1, 80, 64) via pure NumPy.",
         C_EMERALD),
        ("Fusion: Qualcomm Hexagon NPU",
         "• Cross-attention blends complementary articulatory cues.\n"
         "• CTC repetition collapsing & English token decoder.\n"
         "• Reconstructs natural speech in 2.54 ms (100% under 20ms threshold).\n"
         "• Streams crisp output to virtual mic sink with zero cloud egress.",
         C_RED),
    ]

    for i, (title, desc, accent) in enumerate(sol_cards):
        c_shape = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 3.9), Inches(2.2), Inches(3.7), Inches(4.5))
        c_shape.fill.solid()
        c_shape.fill.fore_color.rgb = C_CARD_BG
        c_shape.line.color.rgb = accent
        c_shape.line.width = Pt(1.5)

        ctf = c_shape.text_frame
        ctf.word_wrap = True
        cp0 = ctf.paragraphs[0]
        cp0.text = title
        cp0.font.name = "Calibri"
        cp0.font.size = Pt(16)
        cp0.font.bold = True
        cp0.font.color.rgb = C_WHITE
        cp0.space_after = Pt(14)

        cp1 = ctf.add_paragraph()
        cp1.text = desc
        cp1.font.name = "Calibri"
        cp1.font.size = Pt(12)
        cp1.font.color.rgb = C_SLATE

    # =========================================================================
    # SLIDE 4: Architecture & Qualcomm QNN Optimization
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_slide_header(s4, "Under The Hood: Snapdragon X Elite & QNN EP Deep Optimization")

    # Left Box: Software & Hardware Stack
    left_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.2), Inches(5.6), Inches(4.5))
    left_box.fill.solid()
    left_box.fill.fore_color.rgb = C_CARD_BG
    left_box.line.color.rgb = C_CYAN
    ltf = left_box.text_frame
    ltf.word_wrap = True

    lp0 = ltf.paragraphs[0]
    lp0.text = "Snapdragon X Elite Hardware Mapping"
    lp0.font.name = "Calibri"
    lp0.font.size = Pt(16)
    lp0.font.bold = True
    lp0.font.color.rgb = C_WHITE
    lp0.space_after = Pt(10)

    lp1 = ltf.add_paragraph()
    lp1.text = (
        "• Qualcomm Hexagon NPU (45 TOPS):\n"
        "  - ONNX Runtime 1.18+ with QNNExecutionProvider.\n"
        "  - Backend: QnnHtp.dll targeting HTP v73/v75.\n"
        "  - Performance Mode: burst (maximum NPU clock frequencies).\n"
        "  - Graph Finalization Opt: Mode 3 (full AOT VTCM compilation).\n"
        "  - Precision: Mixed FP16/INT8 for maximum TOPS/Watt.\n\n"
        "• Graceful Zero-Crash Fallback Hierarchy:\n"
        "  1. QNN HTP (Hexagon NPU 45 TOPS)\n"
        "  2. DirectML (Qualcomm Adreno GPU)\n"
        "  3. Oryon CPU (ARM64 native execution)"
    )
    lp1.font.name = "Calibri"
    lp1.font.size = Pt(11.5)
    lp1.font.color.rgb = C_SLATE

    # Right Box: Pipeline Dataflow
    right_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.2), Inches(5.7), Inches(4.5))
    right_box.fill.solid()
    right_box.fill.fore_color.rgb = C_CARD_BG
    right_box.line.color.rgb = C_RED
    rtf = right_box.text_frame
    rtf.word_wrap = True

    rp0 = rtf.paragraphs[0]
    rp0.text = "Microsecond Pipeline Timing Budget"
    rp0.font.name = "Calibri"
    rp0.font.size = Pt(16)
    rp0.font.bold = True
    rp0.font.color.rgb = C_WHITE
    rp0.space_after = Pt(10)

    rp1 = rtf.add_paragraph()
    rp1.text = (
        "Stage 1: Lip ROI Extraction (camera_stream.py)\n"
        "  → 0.14 ms | Media Foundation + EMA Landmark Locking\n\n"
        "Stage 2: Sub-Vocal Mel DSP (audio_stream.py)\n"
        "  → 0.52 ms | 4th-Order Bandpass + 80-bin NumPy Filterbank\n\n"
        "Stage 3: Hexagon NPU Inference (qnn_engine.py)\n"
        "  → 1.82 ms | HTP Burst Mode Parallel Execution\n\n"
        "Stage 4: Temporal Alignment & CTC (fusion_decoder.py)\n"
        "  → 0.06 ms | Blank Token Collapse & Phrase Emission\n\n"
        "TOTAL ROUND-TRIP LATENCY: 2.54 ms\n"
        "Target Budget: < 20.00 ms (Passed with 87% Headroom)"
    )
    rp1.font.name = "Calibri"
    rp1.font.size = Pt(11.5)
    rp1.font.color.rgb = C_SLATE

    # =========================================================================
    # SLIDE 5: Benchmark & Empirical Results
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_slide_header(s5, "Empirical Validation: Snapdragon Hexagon NPU vs. Host CPU")

    metrics_cards = [
        ("2.54 ms", "Round-Trip Latency", "15x faster than CPU (38.5 ms). 100% compliant with <20ms budget.", C_CYAN),
        ("394 FPS", "Sustained Throughput", "Exceeds standard 30/60 FPS video ingestion rates with ease.", C_EMERALD),
        ("1.20 W", "NPU Active Power", "94.7% less power dissipation vs CPU baseline (22.5W).", C_RED),
        ("112 MB", "RAM Footprint", "Direct Vector Tightly-Coupled Memory (VTCM) binding with 0 leaks.", C_WHITE),
    ]

    for i, (stat, label, detail, col) in enumerate(metrics_cards):
        c_shape = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 2.9), Inches(2.2), Inches(2.75), Inches(4.5))
        c_shape.fill.solid()
        c_shape.fill.fore_color.rgb = C_CARD_BG
        c_shape.line.color.rgb = col
        c_shape.line.width = Pt(1.5)

        ctf = c_shape.text_frame
        ctf.word_wrap = True
        cp0 = ctf.paragraphs[0]
        cp0.text = stat
        cp0.font.name = "Calibri"
        cp0.font.size = Pt(32)
        cp0.font.bold = True
        cp0.font.color.rgb = col
        cp0.space_after = Pt(4)

        cp1 = ctf.add_paragraph()
        cp1.text = label
        cp1.font.name = "Calibri"
        cp1.font.size = Pt(14)
        cp1.font.bold = True
        cp1.font.color.rgb = C_WHITE
        cp1.space_after = Pt(12)

        cp2 = ctf.add_paragraph()
        cp2.text = detail
        cp2.font.name = "Calibri"
        cp2.font.size = Pt(11)
        cp2.font.color.rgb = C_SLATE

    # =========================================================================
    # SLIDE 6: Market Impact & Why Snapdragon Wins
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_slide_header(s6, "Real-World Impact & The Qualcomm Snapdragon Moat")

    impact_cards = [
        ("Enterprise & Executive Security",
         "Enables executives to dictate sensitive emails, trade secrets, and financial authorizations in public transit, airports, and coffee shops with zero acoustic leakage and zero cloud exposure.",
         C_RED),
        ("Healthcare & Accessibility",
         "Restores immediate vocal agency to individuals suffering from aphonia, vocal fold paralysis, ALS, or post-laryngectomy speech impediments—empowering them through silent articulation.",
         C_CYAN),
        ("Why Only Snapdragon Wins",
         "Running continuous concurrent computer vision (60 FPS) and acoustic DSP is impossible on legacy x86 CPUs without massive thermal throttling and battery drain. Only Snapdragon X Elite's 45 TOPS Hexagon NPU delivers this at ~1.2W all day.",
         C_EMERALD),
    ]

    for i, (title, desc, accent) in enumerate(impact_cards):
        c_shape = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 3.9), Inches(2.2), Inches(3.7), Inches(4.5))
        c_shape.fill.solid()
        c_shape.fill.fore_color.rgb = C_CARD_BG
        c_shape.line.color.rgb = accent
        c_shape.line.width = Pt(1.5)

        ctf = c_shape.text_frame
        ctf.word_wrap = True
        cp0 = ctf.paragraphs[0]
        cp0.text = title
        cp0.font.name = "Calibri"
        cp0.font.size = Pt(16)
        cp0.font.bold = True
        cp0.font.color.rgb = C_WHITE
        cp0.space_after = Pt(14)

        cp1 = ctf.add_paragraph()
        cp1.text = desc
        cp1.font.name = "Calibri"
        cp1.font.size = Pt(12)
        cp1.font.color.rgb = C_SLATE

    prs.save(PPTX_PATH)
    print(f"Successfully generated PPTX pitch deck: {PPTX_PATH}")


def generate_pdf_slides():
    """Renders the matching 6-slide landscape presentation in PDF."""
    doc = SimpleDocTemplate(
        PDF_SLIDES_PATH,
        pagesize=landscape(letter),
        leftMargin=40,
        rightMargin=40,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    style_badge = ParagraphStyle('SBadge', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#D9381E'))
    style_s_title = ParagraphStyle('STitle', fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#0F172A'), spaceAfter=4)
    style_s_sub = ParagraphStyle('SSub', fontName='Helvetica-Oblique', fontSize=12, textColor=colors.HexColor('#475569'), spaceAfter=14)
    style_card_title = ParagraphStyle('CTitle', fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#0F172A'), spaceAfter=4)
    style_card_body = ParagraphStyle('CBody', fontName='Helvetica', fontSize=9, leading=13, textColor=colors.HexColor('#334155'))
    style_stat_num = ParagraphStyle('StatNum', fontName='Helvetica-Bold', fontSize=24, textColor=colors.HexColor('#D9381E'), spaceAfter=2)
    style_stat_lbl = ParagraphStyle('StatLbl', fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#0F172A'), spaceAfter=4)

    story = []

    # Slide 1: Title
    story.append(Paragraph("QUALCOMM SNAPDRAGON X INNOVATION CHALLENGE", style_badge))
    story.append(Paragraph("Project SilentEcho: On-Device Multi-Modal Silent Speech & Acoustic Camouflage Engine", style_s_title))
    story.append(Paragraph("Sub-20ms Real-Time Articulation Reconstruction | 100% Privacy & Zero Cloud Egress", style_s_sub))
    story.append(Spacer(1, 40))

    meta_s1 = [
        [Paragraph("<b>Target Hardware:</b> Qualcomm Snapdragon X Elite / X Plus (HP OmniBook X / EliteBook Ultra)", style_card_body)],
        [Paragraph("<b>Acceleration:</b> Qualcomm Hexagon NPU (45 TOPS) via ONNX Runtime QNNExecutionProvider (QnnHtp.dll)", style_card_body)],
        [Paragraph("<b>Author:</b> Srujan Kandandla | <b>Repository:</b> https://github.com/srujan1-creator/Snapdragon-project", style_card_body)]
    ]
    t1 = Table(meta_s1, colWidths=[710])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#D9381E')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(t1)
    story.append(PageBreak())

    # Slide 2: Problem
    story.append(Paragraph("PROBLEM & CHALLENGE", style_badge))
    story.append(Paragraph("Acoustic Eavesdropping & The Silent Speech Dilemma", style_s_title))
    story.append(Spacer(1, 10))

    prob_data = [
        [
            Paragraph("<b>1. Acoustic Eavesdropping</b>", style_card_title),
            Paragraph("<b>2. Lip Reading Fails Alone</b>", style_card_title),
            Paragraph("<b>3. Whispers Fail Alone</b>", style_card_title)
        ],
        [
            Paragraph("Speaking aloud in public, open offices, or transit exposes corporate IP, credentials, and confidential discussions to eavesdroppers.", style_card_body),
            Paragraph("Visual lip-reading suffers from severe homophene ambiguities (/p/, /b/, /m/ look identical on lips in 'pat', 'bat', 'mat').", style_card_body),
            Paragraph("Sub-vocal whispers lack fundamental frequency (F0) and are masked by laptop fan rumble and chassis vibrations.", style_card_body)
        ]
    ]
    t2 = Table(prob_data, colWidths=[230, 230, 230])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t2)
    story.append(PageBreak())

    # Slide 3: Solution
    story.append(Paragraph("THE MULTI-MODAL SOLUTION", style_badge))
    story.append(Paragraph("Dual-Stream Temporal Cross-Attention Fusion", style_s_title))
    story.append(Spacer(1, 10))

    sol_data = [
        [
            Paragraph("<b>Stream A: Visual Kinematics</b>", style_card_title),
            Paragraph("<b>Stream B: Sub-Vocal Acoustics</b>", style_card_title),
            Paragraph("<b>Qualcomm Hexagon NPU Fusion</b>", style_card_title)
        ],
        [
            Paragraph("• Windows Media Foundation capture (30/60 FPS).<br/>• EMA landmark coordinate smoothing.<br/>• 40x80 grayscale lip ROI tensor (1, 16, 1, 40, 80).<br/>• Resolves whispered homophones ('pie' vs 'tie').", style_card_body),
            Paragraph("• 16kHz mono audio from HP Poly Studio array.<br/>• 4th-order Bandpass (120Hz-7.2kHz) rejects rumble.<br/>• Whisper pre-emphasis boosts unvoiced fricatives.<br/>• 80-bin Mel spectrogram tensor (1, 80, 64).", style_card_body),
            Paragraph("• Cross-attention blends complementary cues.<br/>• CTC blank token collapse.<br/>• 2.54 ms inference latency (well under 20ms).<br/>• Virtual mic sink emission with 0 cloud egress.", style_card_body)
        ]
    ]
    t3 = Table(sol_data, colWidths=[230, 230, 230])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t3)
    story.append(PageBreak())

    # Slide 4: Benchmarks
    story.append(Paragraph("EMPIRICAL BENCHMARKS", style_badge))
    story.append(Paragraph("Hexagon NPU Performance & Power Advantage", style_s_title))
    story.append(Spacer(1, 10))

    bench_grid = [
        [
            Paragraph("2.54 ms", style_stat_num),
            Paragraph("394 FPS", style_stat_num),
            Paragraph("1.20 W", style_stat_num),
            Paragraph("112 MB", style_stat_num),
        ],
        [
            Paragraph("<b>Round-Trip Latency</b>", style_stat_lbl),
            Paragraph("<b>Sustained Throughput</b>", style_stat_lbl),
            Paragraph("<b>NPU Active Power</b>", style_stat_lbl),
            Paragraph("<b>RAM Working Set</b>", style_stat_lbl),
        ],
        [
            Paragraph("15x faster than CPU (38.5ms). 100% compliant with <20ms budget.", style_card_body),
            Paragraph("Real-time ingestion pace easily outpacing video sensor rates.", style_card_body),
            Paragraph("94.7% less power vs. Oryon/x86 CPU (22.5W). All-day battery life.", style_card_body),
            Paragraph("Direct VTCM memory binding with zero memory leaks.", style_card_body),
        ]
    ]
    t4 = Table(bench_grid, colWidths=[175, 175, 175, 175])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t4)
    story.append(PageBreak())

    # Slide 5: Market & Why Snapdragon
    story.append(Paragraph("MARKET IMPACT & COMPETITIVE MOAT", style_badge))
    story.append(Paragraph("Real-World Use Cases & Why Only Snapdragon Wins", style_s_title))
    story.append(Spacer(1, 10))

    impact_data = [
        [
            Paragraph("<b>Enterprise & Executive Security</b>", style_card_title),
            Paragraph("<b>Healthcare & Accessibility</b>", style_card_title),
            Paragraph("<b>The Snapdragon Moat</b>", style_card_title)
        ],
        [
            Paragraph("Executives dictating confidential contracts, trade secrets, and banking credentials in public transit without uttering an audible sound.", style_card_body),
            Paragraph("Restores instantaneous vocal agency to patients with ALS, vocal fold paralysis, or laryngectomy through subtle silent articulations.", style_card_body),
            Paragraph("Concurrent 60 FPS vision + acoustic DSP causes severe thermal throttling on legacy x86 CPUs. Only Snapdragon X Elite's 45 TOPS Hexagon NPU delivers this at ~1.2W.", style_card_body)
        ]
    ]
    t5 = Table(impact_data, colWidths=[230, 230, 230])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t5)

    doc.build(story)
    print(f"Successfully generated PDF pitch deck: {PDF_SLIDES_PATH}")


if __name__ == "__main__":
    generate_pptx()
    generate_pdf_slides()
