"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Script: utils/build_pdf_presentation.py
Description:
  Converts PRESENTATION_SLIDES_TOPIC2.pdf.md into an ultra-premium,
  16:9 widescreen presentation PDF:
    PRESENTATION_SLIDES_TOPIC2.pdf
  
  Layout Standards:
    - Clean White Academic palette (#FFFFFF, #F8FAFC, #0F172A, #0284C7)
    - 16:9 Widescreen slides (16in x 9in) with zero clipping or text overflow
    - Embedded 300-DPI publication plots (SNR curves & Confusion matrices)
    - Complete 4-member allocation & timetable
    - All 14 technical slides + 5 comprehensive Q&A defense slides + Cover & Ending
    - Exact 1-to-1 fidelity with PRESENTATION_SLIDES_TOPIC2.pdf.md
================================================================================
"""

import os
import sys
import base64
import subprocess
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_base64_image(rel_path):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(abs_path):
        with open(abs_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
            ext = os.path.splitext(rel_path)[1].lower()
            mime = 'image/png' if ext == '.png' else 'image/jpeg'
            return f"data:{mime};base64,{data}"
    return ""

def generate_html():
    img_snr_curves = get_base64_image('assets/slide9_snr_curves.png')
    img_confusion = get_base64_image('assets/slide10_confusion_matrices.png')

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Topic 02: AMC Defense Presentation - 15 Phút Bảo Vệ</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-main: #FFFFFF;
      --card-bg: #F8FAFC;
      --card-border: #E2E8F0;
      --text-main: #0F172A;
      --text-muted: #64748B;
      --accent: #0284C7;
      --accent-light: #E0F2FE;
      --accent-dark: #0369A1;
      --success: #10B981;
      --success-light: #ECFDF5;
      --warning: #F59E0B;
      --warning-light: #FFFBEB;
      --danger: #EF4444;
      --danger-light: #FEF2F2;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, "Courier New", monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-sans);
      background-color: #E2E8F0;
      color: var(--text-main);
      margin: 0;
      padding: 20px 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 30px;
    }}

    /* Slide Page Container (16:9 Widescreen) */
    .slide-page {{
      width: 1333px;
      height: 750px;
      background: var(--bg-main);
      border-radius: 8px;
      box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
      border: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      padding: 30px 45px;
      position: relative;
      overflow: hidden;
      box-sizing: border-box;
    }}

    /* Global Slide Header */
    .slide-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      margin-bottom: 14px;
      border-bottom: 1.5px solid var(--card-border);
    }}
    .topic-tag {{
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent);
      background: var(--accent-light);
      padding: 4px 12px;
      border-radius: 16px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .speaker-tag {{
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .speaker-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
    }}

    /* Slide Title Area */
    .slide-title {{
      font-size: 1.65rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.02em;
      line-height: 1.25;
      margin-bottom: 4px;
    }}
    .slide-subtitle {{
      font-size: 0.92rem;
      color: var(--text-muted);
      margin-bottom: 16px;
      font-weight: 500;
    }}

    /* Slide Content Grid Layouts */
    .slide-content {{
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 0;
    }}

    .cards-grid {{
      display: grid;
      gap: 16px;
      flex: 1;
      min-height: 0;
    }}
    .grid-2 {{ grid-template-columns: 1fr 1fr; }}
    .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
    .grid-4 {{ grid-template-columns: 1fr 1fr 1fr 1fr; }}
    .grid-split-plot {{ grid-template-columns: 1.15fr 0.85fr; }}
    .grid-split-table {{ grid-template-columns: 1.2fr 0.8fr; }}

    /* Cards */
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .card.card-highlight {{
      border-color: var(--accent);
      background: #F0F9FF;
    }}
    .card.card-qa-q {{
      background: #F8FAFC;
      border-left: 5px solid var(--warning);
    }}
    .card.card-qa-a {{
      background: #F0F9FF;
      border-left: 5px solid var(--accent);
    }}

    .card-title {{
      font-size: 0.98rem;
      font-weight: 700;
      color: var(--accent-dark);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .card-content {{
      font-size: 0.85rem;
      line-height: 1.5;
      color: #334155;
      flex: 1;
    }}
    .card-content ul {{
      list-style-type: none;
      padding-left: 0;
    }}
    .card-content li {{
      position: relative;
      padding-left: 14px;
      margin-bottom: 6px;
    }}
    .card-content li::before {{
      content: "•";
      color: var(--accent);
      font-weight: bold;
      position: absolute;
      left: 0;
    }}

    /* Formula & Code Boxes */
    .formula-box {{
      background: #FFFFFF;
      border: 1px dashed #93C5FD;
      padding: 8px 12px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      margin: 8px 0;
      color: var(--accent-dark);
      font-weight: 500;
      line-height: 1.45;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.80rem;
      background: #FFFFFF;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid var(--card-border);
    }}
    th {{
      background: #F1F5F9;
      color: var(--text-main);
      font-weight: 700;
      padding: 7px 10px;
      text-align: left;
      border-bottom: 1.5px solid var(--card-border);
      border-right: 1px solid var(--card-border);
    }}
    td {{
      padding: 6px 10px;
      border-bottom: 1px solid var(--card-border);
      border-right: 1px solid var(--card-border);
      color: #334155;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    th:last-child, td:last-child {{
      border-right: none;
    }}
    tr:nth-child(even) {{
      background: #F8FAFC;
    }}

    /* Badges */
    .badge {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 0.74rem;
      font-weight: 700;
    }}
    .badge-blue {{ background: var(--accent-light); color: var(--accent-dark); }}
    .badge-success {{ background: #DCFCE7; color: #15803D; }}
    .badge-warning {{ background: #FEF3C7; color: #B45309; }}
    .badge-danger {{ background: #FEE2E2; color: #B91C1C; }}

    /* Transition Cue Banner */
    .transition-cue {{
      background: #F0F9FF;
      border-left: 3.5px solid var(--accent);
      padding: 6px 14px;
      border-radius: 0 6px 6px 0;
      font-size: 0.76rem;
      color: #0369A1;
      font-style: italic;
      margin-top: 10px;
    }}

    /* Embedded Plot Container */
    .plot-container {{
      background: #FFFFFF;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 8px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
    }}
    .plot-container img {{
      max-width: 100%;
      max-height: 480px;
      object-fit: contain;
      border-radius: 4px;
    }}

    /* Global Footer */
    .slide-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 10px;
      margin-top: 8px;
      border-top: 1px solid var(--card-border);
      font-size: 0.74rem;
      color: var(--text-muted);
    }}

    /* Cover Slide Specific Styles */
    .cover-container {{
      display: flex;
      flex-direction: column;
      justify-content: center;
      height: 100%;
      padding: 20px 40px;
    }}
    .cover-title {{
      font-size: 2.3rem;
      font-weight: 800;
      line-height: 1.25;
      color: var(--text-main);
      margin-bottom: 12px;
      letter-spacing: -0.02em;
    }}
    .cover-title span {{
      color: var(--accent);
    }}
    .cover-subtitle {{
      font-size: 1.15rem;
      color: var(--text-muted);
      margin-bottom: 30px;
      font-weight: 500;
    }}
    .cover-meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-top: 20px;
    }}
    .cover-meta-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
    }}
    .cover-meta-label {{
      font-size: 0.75rem;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 6px;
      letter-spacing: 0.05em;
    }}
    .cover-meta-val {{
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-main);
      line-height: 1.4;
    }}

    /* ==========================================================================
       Print Settings for Exact 16:9 PDF Export via Microsoft Edge
       ========================================================================== */
    @media print {{
      @page {{
        size: 16in 9in;
        margin: 0;
      }}
      *, *::before, *::after {{
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }}
      html, body {{
        width: 16in !important;
        height: auto !important;
        background: #FFFFFF !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
      }}
      .slide-page {{
        width: 16in !important;
        height: 9in !important;
        max-height: 9in !important;
        page-break-after: always !important;
        break-after: page !important;
        display: flex !important;
        flex-direction: column !important;
        box-sizing: border-box !important;
        padding: 0.45in 0.75in !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        background: #FFFFFF !important;
        overflow: hidden !important;
      }}
      .slide-title {{
        font-size: 2.1rem !important;
      }}
      .slide-subtitle {{
        font-size: 1.1rem !important;
        margin-bottom: 18px !important;
      }}
      .card-title {{
        font-size: 1.15rem !important;
      }}
      .card-content {{
        font-size: 0.98rem !important;
        line-height: 1.55 !important;
      }}
      .formula-box {{
        font-size: 0.95rem !important;
        padding: 10px 14px !important;
      }}
      table {{
        font-size: 0.92rem !important;
      }}
      th, td {{
        padding: 8px 12px !important;
      }}
      .transition-cue {{
        font-size: 0.86rem !important;
        padding: 8px 16px !important;
      }}
      .slide-footer {{
        font-size: 0.85rem !important;
      }}
    }}
  </style>
</head>
<body>

  <!-- ========================================================================
       SLIDE 0: COVER SLIDE
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • CAPSTONE MINI-PROJECT DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Hội đồng Chấm Bảo vệ Mini-Project (15 Phút + 5 Phút Q&A)</span>
    </div>
    <div class="cover-container">
      <div style="margin-bottom: 16px;">
        <span class="badge badge-blue" style="font-size: 0.85rem; padding: 5px 14px; margin-bottom: 12px;">TOPIC 02: AUTOMATIC MODULATION CLASSIFICATION</span>
      </div>
      <h1 class="cover-title">Spectral & Temporal Feature Engineering<br><span>vs. End-to-End Raw I/Q Deep Learning</span></h1>
      <p class="cover-subtitle">Nghiên cứu Thực nghiệm Định lượng, Phân tích Lý thuyết Lỗi Gốc rễ & Thiết kế Hệ thống AMC Tối ưu</p>

      <div class="cover-meta-grid">
        <div class="cover-meta-card">
          <div class="cover-meta-label">Môn học</div>
          <div class="cover-meta-val">EE136IU: Machine Learning for Wireless Communications</div>
        </div>
        <div class="cover-meta-card">
          <div class="cover-meta-label">Cơ quan Hướng dẫn</div>
          <div class="cover-meta-val">Bộ môn Viễn thông<br>Trường ĐH Quốc Tế (ĐHQG-HCM)</div>
        </div>
        <div class="cover-meta-card">
          <div class="cover-meta-label">Nhóm Thực hiện</div>
          <div class="cover-meta-val">4 Thành viên Thực hiện<br>Thời lượng: 15 Phút + 5 Phút Q&A</div>
        </div>
        <div class="cover-meta-card">
          <div class="cover-meta-label">Tiêu chuẩn Thiết kế</div>
          <div class="cover-meta-val">Clean White Academic Standard<br>Primary Accent: #0284C7</div>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Trường Đại học Quốc Tế - ĐHQG-HCM • Khoa Điện Tử Viễn Thông</span>
      <span>Slide 0 / 20 • Cover Slide</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 0.5: GỢI Ý PHÂN BỔ THỜI GIAN NHÓM & BẢNG PHÂN CÔNG VAI TRÒ
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Toàn bộ Nhóm 4 Thành viên • Phân bổ Thời gian & Trách nhiệm</span>
    </div>
    <h1 class="slide-title">Kế Hoạch Thuyết Trình & Phân Công Trách Nhiệm 4 Thành Viên</h1>
    <p class="slide-subtitle">15 Phút Trình bày Chuyên môn + 5 Phút Phản biện Q&A trực tiếp với Hội đồng Giảng viên</p>

    <div class="slide-content">
      <div class="cards-grid grid-4" style="margin-bottom: 12px; flex: 0 0 auto;">
        <div class="card">
          <div class="card-title">👤 Thành viên 1</div>
          <div class="card-content">
            <span class="badge badge-blue">Phút 00:00 – 03:45</span>
            <p style="margin-top: 5px;"><strong>Slides 1 – 4:</strong> Bối cảnh AMC, Kênh vật lý Baseband, RMS Normalization, Toán học HOC & Không gian 10-D Feature.</p>
            <p style="margin-top: 4px; color: var(--accent-dark); font-weight: 600;">Trách nhiệm: Q&A Câu 1</p>
          </div>
        </div>
        <div class="card">
          <div class="card-title">👤 Thành viên 2</div>
          <div class="card-content">
            <span class="badge badge-blue">Phút 03:45 – 07:30</span>
            <p style="margin-top: 5px;"><strong>Slides 5 – 8:</strong> Dataset Zero-Leakage, ML Classifiers (XGBoost/SVM), Kiến trúc PyTorch 1D-CNN & Benchmark Độ trễ Phần cứng.</p>
            <p style="margin-top: 4px; color: var(--accent-dark); font-weight: 600;">Trách nhiệm: Q&A Câu 3 & 5</p>
          </div>
        </div>
        <div class="card">
          <div class="card-title">👤 Thành viên 3</div>
          <div class="card-content">
            <span class="badge badge-blue">Phút 07:30 – 11:15</span>
            <p style="margin-top: 5px;"><strong>Slides 9 – 11:</strong> Benchmark SNR vs Acc/F1, Confusion Matrix 3 vùng nhiễu, Phân tích Lỗi 1 (Suy sụp Cumulants ở Low-SNR).</p>
            <p style="margin-top: 4px; color: var(--accent-dark); font-weight: 600;">Trách nhiệm: Q&A Câu 2</p>
          </div>
        </div>
        <div class="card">
          <div class="card-title">👤 Thành viên 4</div>
          <div class="card-content">
            <span class="badge badge-blue">Phút 11:15 – 15:00</span>
            <p style="margin-top: 5px;"><strong>Slides 12 – 14:</strong> Phân tích Lỗi 2 (1D-CNN Lọc phối hợp), Lỗi 3 (Bế tắc 16 vs 64QAM), Kết luận & Trình diễn Live Streamlit Demo.</p>
            <p style="margin-top: 4px; color: var(--accent-dark); font-weight: 600;">Trách nhiệm: Q&A Câu 4</p>
          </div>
        </div>
      </div>

      <div class="card" style="padding: 10px 14px; flex: 1;">
        <div class="card-title" style="margin-bottom: 6px;">📋 Bảng Tổng hợp Phân công Vai trò 4 Thành viên</div>
        <table>
          <thead>
            <tr>
              <th style="width: 14%;">Thành viên</th>
              <th style="width: 13%;">Slide đảm nhiệm</th>
              <th style="width: 11%;">Thời lượng</th>
              <th style="width: 14%;">Khung thời gian</th>
              <th style="width: 32%;">Nội dung trọng tâm</th>
              <th style="width: 16%;">Trách nhiệm Q&A chính</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Thành viên 1</strong></td>
              <td>Slides 1 – 4 (4 slides)</td>
              <td>~03:45</td>
              <td>00:00 – 03:45</td>
              <td>Đặt vấn đề AMC, Kênh vật lý Baseband, Chuẩn hóa RMS, Toán học HOC & 10-D Feature Space</td>
              <td><span class="badge badge-blue">Câu hỏi 1 (Baseband & RMS)</span></td>
            </tr>
            <tr>
              <td><strong>Thành viên 2</strong></td>
              <td>Slides 5 – 8 (4 slides)</td>
              <td>~03:45</td>
              <td>03:45 – 07:30</td>
              <td>Dataset Zero-Leakage, Bộ phân loại ML (XGBoost/SVM), Kiến trúc 1D-CNN & Benchmark Phần cứng</td>
              <td><span class="badge badge-blue">Câu 3 & 5 (CNN & Edge IoT)</span></td>
            </tr>
            <tr>
              <td><strong>Thành viên 3</strong></td>
              <td>Slides 9 – 11 (3 slides)</td>
              <td>~03:45</td>
              <td>07:30 – 11:15</td>
              <td>Kết quả Benchmark SNR vs Acc/F1, Confusion Matrix 3 chế độ, Phân tích Lỗi 1 (Suy sụp Cumulants)</td>
              <td><span class="badge badge-blue">Câu hỏi 2 (Suy sụp Cumulants)</span></td>
            </tr>
            <tr>
              <td><strong>Thành viên 4</strong></td>
              <td>Slides 12 – 14 (3 slides)</td>
              <td>~03:45</td>
              <td>11:15 – 15:00</td>
              <td>Phân tích Lỗi 2 (Lọc phối hợp), Phân tích Lỗi 3 (Bế tắc 16 vs 64QAM & 4 Giải pháp), Live Demo Streamlit</td>
              <td><span class="badge badge-blue">Câu hỏi 4 (Bayes 16/64QAM)</span></td>
            </tr>
            <tr>
              <td><strong>Cả nhóm</strong></td>
              <td>Toàn bộ 14 slides</td>
              <td>05:00</td>
              <td>15:00 – 20:00</td>
              <td><strong>Phiên Q&A Bảo vệ Trực tiếp với Hội đồng Giảng viên</strong></td>
              <td><span class="badge badge-success">Phối hợp phản biện 5/5 câu</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Phân bổ Thời gian & Vai trò Nhóm</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 1: TIÊU ĐỀ DỰ ÁN & GIỚI THIỆU TỔNG QUAN
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 1 / 14 • Thành viên 1 (Phút 00:00 – 03:45)</span>
    </div>
    <h1 class="slide-title">Tổng Quan Bài Toán & Câu Hỏi Nghiên Cứu Cốt Lõi</h1>
    <p class="slide-subtitle">Automatic Modulation Classification: Expert Feature Engineering vs. End-to-End Deep Learning</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">🛰️ Bối cảnh & Tầm quan trọng</div>
          <div class="card-content">
            <ul>
              <li><strong>Vô tuyến Nhận thức (Cognitive Radio):</strong> AMC là kỹ thuật then chốt giúp máy thu tự thích nghi với môi trường truyền thông 5G/6G phi tập trung.</li>
              <li><strong>Nhiệm vụ:</strong> Tự động nhận dạng 8 định dạng điều chế số phổ biến: <strong>BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, CPFSK, GFSK</strong>.</li>
              <li><strong>Phân loại mù (Blind Classification):</strong> Hoạt động hoàn toàn không có thông tin phụ trợ (CSI, pilot sequence) gửi kèm từ máy phát.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">❓ Câu hỏi Nghiên cứu Cốt lõi</div>
          <div class="card-content">
            <ul>
              <li><strong>Tri thức Vật lý Chuyên gia (Handcrafted Features):</strong> Thống kê bậc cao (HOC), Pha vi sai (Differential Phase), và Phổ công suất FFT có còn giữ ưu thế?</li>
              <li><strong>Học sâu Trực tiếp (End-to-End 1D-CNN):</strong> Tự động học trích xuất đặc trưng từ mảng tensor thô I/Q có vượt trội trong mọi điều kiện?</li>
              <li><strong>Điểm gãy Hiệu năng:</strong> Đâu là ranh giới SNR mà tại đó sự chuyển giao ưu thế kỹ thuật diễn ra, và bản chất vật lý của hiện tượng đó là gì?</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">💡 Đóng góp Kỹ thuật Chính</div>
          <div class="card-content">
            <ul>
              <li><strong>Mô hình Kênh Sát Thực tế:</strong> Tích hợp bộ lọc tạo dạng xung RRC, độ lệch tần số CFO ngẫu nhiên, nhiễu AWGN phức và chuẩn hóa RMS per-burst.</li>
              <li><strong>Giải thích Minh bạch Lỗi Gốc rễ:</strong> Chứng minh toán học sự suy sụp của Cumulants ở SNR thấp và giới hạn phân tách Bayes giữa 16QAM và 64QAM.</li>
              <li><strong>Thực chứng Đa diện:</strong> Đánh giá định lượng trên 48,000 bursts kiểm thử độc lập kèm ứng dụng Streamlit tương tác thời gian thực.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 1 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 2: MÔ HÌNH VẬT LÝ BASEBAND & CHUẨN HÓA NĂNG LƯỢNG
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 2 / 14 • Thành viên 1 (Phút 00:00 – 03:45)</span>
    </div>
    <h1 class="slide-title">Mô Hình Vật Lý Baseband & Chuẩn Hóa Năng Lượng RMS</h1>
    <p class="slide-subtitle">Realistic Complex Baseband Signal Generation Pipeline & Per-Burst Power Normalization</p>

    <div class="slide-content">
      <div class="cards-grid grid-2">
        <div class="card">
          <div class="card-title">📡 Chuỗi Xử lý Phát - Kênh - Thu (Baseband Pipeline)</div>
          <div class="card-content">
            <p>Mô hình tín hiệu thu phức rời rạc thời gian tại đầu ra băng cơ sở:</p>
            <div class="formula-box">
              r[n] = [ ∑_k s[k] · g(n T_s - k T) ] · exp( j ( 2π Δf_c n T_s + θ_0 ) ) + w[n]
            </div>
            <ul>
              <li><strong>Tạo dạng xung RRC:</strong> Bộ lọc Root Raised Cosine với hệ số roll-off α = 0.35, số mẫu trên mỗi ký hiệu SPS = 4 mẫu/symbol.</li>
              <li><strong>Độ lệch tần số sóng mang (CFO):</strong> Δf_c ngẫu nhiên mô phỏng sự mất đồng bộ thạch anh giữa máy phát và máy thu (Carrier Frequency Offset).</li>
              <li><strong>Nhiễu Kênh AWGN:</strong> Nhiễu trắng Gauss phức w[n] ~ CN(0, σ_w²), quét định lượng từ -10 dB đến +18 dB (bước 2 dB).</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">⚖️ Chuẩn hóa Năng lượng Per-Burst RMS (Bắt buộc)</div>
          <div class="card-content">
            <p>Tín hiệu chuẩn hóa trên từng khung tín hiệu ngắn L = 128 mẫu:</p>
            <div class="formula-box">
              r̃[n] = r[n] / √( (1/L) ∑_(m=0)^(L-1) |r[m]|² )
            </div>
            <ul>
              <li><strong>Ý nghĩa Viễn thông Thực tế:</strong> Triệt tiêu hoàn toàn sự chênh lệch biên độ công suất thu do khoảng cách truyền dẫn hoặc suy hao đường truyền (Path Loss).</li>
              <li><strong>Ép Mô hình Học Bản chất Hình học:</strong> Ngăn chặn bộ phân loại "học vẹt" biên độ tín hiệu thô, ép mọi thuật toán phải nhận dạng thuần túy qua hình học chòm sao và đặc tính pha/phổ.</li>
              <li><strong>Khắc phục Nhược điểm Peak Normalization:</strong> Chuẩn hóa RMS bền vững trước các mẫu nhiễu ngoại lai (noise spikes), bảo toàn công suất kỳ vọng E[|r̃|²] = 1.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 2 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 3: NỀN TẢNG TOÁN HỌC HIGHER-ORDER CUMULANTS (HOC)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 3 / 14 • Thành viên 1 (Phút 00:00 – 03:45)</span>
    </div>
    <h1 class="slide-title">Nền Tảng Toán Học Higher-Order Cumulants (HOC)</h1>
    <p class="slide-subtitle">Physical Grounding: Constellation Symmetry & Cumulant Invariance</p>

    <div class="slide-content">
      <div class="cards-grid grid-split-table">
        <div style="display: flex; flex-direction: column; gap: 12px;">
          <div class="card">
            <div class="card-title">📐 Công thức Cumulants Bậc 4 Chuẩn hóa</div>
            <div class="card-content">
              <div class="formula-box">
                C_40 = Cum(r, r, r, r) = M_40 - 3 (M_20)²<br>
                C_42 = Cum(r, r*, r, r*) = M_42 - |M_20|² - 2 (M_21)²
              </div>
              <ul>
                <li>Với moment M_pq = E[r^(p-q) · (r*)^q], và tín hiệu chuẩn hóa RMS có M_21 = E[|r|²] = 1.</li>
                <li><strong>Tính chất Bất biến:</strong> C_42 hoàn toàn không phụ thuộc vào độ lệch pha sóng mang ngẫu nhiên θ_0 (Rotational Invariance).</li>
              </ul>
            </div>
          </div>

          <div class="card">
            <div class="card-title">💡 Nhận xét Vật lý Sắc bén</div>
            <div class="card-content">
              <ul>
                <li><strong>|C_40| phân tách tuyệt đối các họ PSK:</strong> BPSK (2.000) ➔ QPSK (1.000) ➔ 8PSK (0.000).</li>
                <li><strong>C_42 phân tách Họ Biên độ Cố định vs Đa mức:</strong> PSK có C_42 = -1.000, trong khi QAM có biên độ biến thiên (16QAM: -0.680; 64QAM: -0.619).</li>
              </ul>
            </div>
          </div>
        </div>

        <div class="card" style="padding: 10px 14px;">
          <div class="card-title" style="margin-bottom: 6px;">📊 Bảng Giá trị Lý thuyết Không Nhiễu (E_s = 1)</div>
          <table>
            <thead>
              <tr>
                <th>Modulation</th>
                <th>Symmetry</th>
                <th>|M_20|</th>
                <th>|C_40| Lý thuyết</th>
                <th>C_42 Lý thuyết</th>
                <th>Đặc trưng Hình học</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>BPSK</strong></td>
                <td>180° (Bậc 2)</td>
                <td>1.000</td>
                <td><strong style="color:var(--accent);">2.000</strong></td>
                <td><strong style="color:var(--accent);">-2.000</strong></td>
                <td>Đối xứng trục thực, s² = 1</td>
              </tr>
              <tr>
                <td><strong>QPSK</strong></td>
                <td>90° (Bậc 4)</td>
                <td>0.000</td>
                <td><strong style="color:var(--accent);">1.000</strong></td>
                <td><strong style="color:var(--accent);">-1.000</strong></td>
                <td>4 điểm vòng tròn, s⁴ = -1</td>
              </tr>
              <tr>
                <td><strong>8PSK</strong></td>
                <td>45° (Bậc 8)</td>
                <td>0.000</td>
                <td><strong style="color:var(--accent);">0.000</strong></td>
                <td><strong style="color:var(--accent);">-1.000</strong></td>
                <td>8 điểm vòng tròn, s⁴ triệt tiêu</td>
              </tr>
              <tr>
                <td><strong>16QAM</strong></td>
                <td>90° (Bậc 4)</td>
                <td>0.000</td>
                <td><strong style="color:var(--accent);">0.680</strong></td>
                <td><strong style="color:var(--accent);">-0.680</strong></td>
                <td>3 mức năng lượng, đa biên độ</td>
              </tr>
              <tr>
                <td><strong>64QAM</strong></td>
                <td>90° (Bậc 4)</td>
                <td>0.000</td>
                <td><strong style="color:var(--accent);">0.619</strong></td>
                <td><strong style="color:var(--accent);">-0.619</strong></td>
                <td>9 mức năng lượng, xấp xỉ tròn</td>
              </tr>
              <tr>
                <td><strong>PAM4</strong></td>
                <td>180° (Real)</td>
                <td>1.000</td>
                <td><strong style="color:var(--accent);">1.360</strong></td>
                <td><strong style="color:var(--accent);">-1.360</strong></td>
                <td>Điều chế 1 chiều (1D Constellation)</td>
              </tr>
            </tbody>
          </table>
          <p style="font-size: 0.74rem; color: var(--text-muted); margin-top: 6px;">* Ghi chú: CPFSK và GFSK có pha biến thiên liên tục, sẽ được tách riêng biệt nhờ bộ đặc trưng pha vi sai ở Slide 4.</p>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 3 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 4: VECTƠ ĐẶC TRƯNG MIỀN THỜI GIAN & PHỔ (10-D FEATURE SPACE)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 4 / 14 • Thành viên 1 (Phút 00:00 – 03:45)</span>
    </div>
    <h1 class="slide-title">Không Gian 10 Đặc Trưng Miền Phổ & Thời Gian (10-D Space)</h1>
    <p class="slide-subtitle">Complete 10-Dimensional Handcrafted Feature Vector with Mathematical Formulations</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">1. Bốn Thống kê Bậc cao HOC</div>
          <div class="card-content">
            <div class="formula-box">
              C_20, C_21, |C_40|, C_42
            </div>
            <ul>
              <li><strong>Mục đích:</strong> Nắm bắt đối xứng hình học chòm sao và số lượng mức năng lượng biên độ.</li>
              <li><strong>Khả năng phân tách:</strong> Tách triệt để BPSK, QPSK, 8PSK, và PAM4 ở điều kiện SNR từ trung bình đến cao.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">2. Ba Đặc trưng Pha Vi sai</div>
          <div class="card-content">
            <div class="formula-box">
              Δϕ[n] = unwrap( arg( r[n] · r*[n-1] ) )<br>
              Trích xuất: μ_Δϕ, σ_Δϕ, max |FFT(Δϕ)|
            </div>
            <ul>
              <li><strong>Phân tách Điều chế Pha Liên tục:</strong> CPFSK và GFSK có pha tích lũy tuyến tính theo bit dữ liệu.</li>
              <li><strong>Độ nhạy cao:</strong> σ_Δϕ phân định rõ rệt giữa điều chế dịch tần (FSK) và các điều chế PSK/QAM.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Bốn Đặc trưng Phổ Công suất FFT</div>
          <div class="card-content">
            <div class="formula-box" style="font-size: 0.76rem;">
              • Đỉnh phổ: γ_max = max |DFT(r̃)|² / L<br>
              • Trọng tâm phổ: μ_f = ∑ f_k · P[k]<br>
              • Độ rộng phổ: σ_f = √( ∑ (f_k - μ_f)² · P[k] )<br>
              • Entropy phổ: H_f = - ∑ P[k] log₂(P[k]) / log₂(L)
            </div>
            <ul>
              <li>Phản ánh độ tập trung năng lượng phổ băng thông và độ phẳng phân bố tần số.</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="transition-cue">
        👉 <strong>Lời chuyển giao (Transition Cue sang Thành viên 2):</strong> "Sau khi đã xây dựng không gian 10 đặc trưng vật lý chuyên gia và biểu diễn tensor thô I/Q, tiếp theo xin mời bạn [Thành viên 2] trình bày về quy chuẩn phân chia dataset chống rò rỉ và thiết kế kiến trúc các mô hình phân loại."
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 4 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 5: NGUYÊN TẮC THIẾT KẾ DATASET & CHỐNG RÒ RỈ (ANTI-LEAKAGE)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 5 / 14 • Thành viên 2 (Phút 03:45 – 07:30)</span>
    </div>
    <h1 class="slide-title">Thiết Kế Tập Dữ Liệu & Quy Chuẩn Chống Rò Rỉ (Zero-Leakage)</h1>
    <p class="slide-subtitle">Zero-Leakage Dataset Partition & Deterministic Reproducibility Standards</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">📦 Phân bổ Dữ liệu Độc lập</div>
          <div class="card-content">
            <div class="formula-box">
              8 Class × 15 SNR (-10 dB đến +18 dB)<br>
              × 400 bursts = 48,000 Bursts
            </div>
            <ul>
              <li><strong>Train Set (70%):</strong> 33,600 bursts (seed=42).</li>
              <li><strong>Validation Set (15%):</strong> 7,200 bursts (seed=43) dùng tối ưu siêu tham số và Early Stopping.</li>
              <li><strong>Held-out Test Set (15%):</strong> 7,200 bursts (seed=44) khóa tuyệt đối để đánh giá khách quan.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🛡️ Quy tắc Chống Rò rỉ Dữ liệu Tuyệt đối</div>
          <div class="card-content">
            <ul>
              <li><strong>Khởi tạo Độc lập (Independent Synthesis):</strong> Tập Test được tổng hợp hoàn toàn từ chuỗi bit ngẫu nhiên và nhiễu riêng biệt, không cắt lát (slicing) từ một chuỗi liên tục.</li>
              <li><strong>Fit Scaler Chuẩn mực:</strong> Đối với mô hình trích xuất đặc trưng, `StandardScaler` chỉ `fit` trên Train, sau đó áp dụng `transform` cho Val và Test, ngăn ngừa rò rỉ phân phối.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🎯 Phân tích Theo Từng Mức SNR (Per-SNR)</div>
          <div class="card-content">
            <ul>
              <li><strong>Đánh giá Phân tầng:</strong> 480 test bursts tại mỗi điểm SNR cụ thể (60 bursts / class / SNR).</li>
              <li><strong>Khoảng Tin cậy Wilson 95%:</strong>
                <div class="formula-box">
                  CI_95 ≈ Accuracy ± 1.96 · √( p(1-p) / N_test )<br>
                  Tại N=480: Sai số ước lượng ≈ ± 4.5%
                </div>
              </li>
              <li>Đảm bảo tính tin cậy thống kê cao nhất khi so sánh mô hình.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 5 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 6: MÔ HÌNH FEATURE-BASED: XGBOOST & RBF-SVM
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 6 / 14 • Thành viên 2 (Phút 03:45 – 07:30)</span>
    </div>
    <h1 class="slide-title">Mô Hình Feature-Based: XGBoost & RBF-SVM</h1>
    <p class="slide-subtitle">Expert Feature Classifiers: Hyperparameters, Training Optimization & Feature Importance</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">🌳 XGBoost Gradient Boosted Trees</div>
          <div class="card-content">
            <div class="formula-box">
              n_estimators = 200, max_depth = 6<br>
              learning_rate = 0.1, subsample = 0.8
            </div>
            <ul>
              <li><strong>Ưu điểm Vượt trội:</strong> Khả năng phân vùng ranh giới phi tuyến tính cực nhanh trên không gian đặc trưng 10 chiều.</li>
              <li><strong>Tốc độ Huấn luyện Siêu tốc:</strong> Hoàn tất toàn bộ 33,600 mẫu Train chỉ trong <strong>~4.2 giây</strong> trên CPU thông thường.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🎯 RBF Support Vector Machine</div>
          <div class="card-content">
            <div class="formula-box">
              Kernel RBF (C = 10.0, gamma = 'scale')<br>
              probability = True (Platt Scaling)
            </div>
            <ul>
              <li><strong>Ranh giới Biên Cực đại:</strong> Tối ưu hóa khoảng cách hình học giữa các lớp điều chế.</li>
              <li><strong>Chiến lược Giảm tải O(N²):</strong> Giới hạn tập train của SVM ở 12,000 bursts để tránh nghẽn tính toán bộ giải QP, bảo toàn 100% tập Val/Test.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🏆 Tầm quan trọng Đặc trưng (Feature Importance)</div>
          <div class="card-content">
            <div class="formula-box">
              Top 1: |C_40| (Phân biệt PSK vs QAM)<br>
              Top 2: C_42 (Tách PSK vòng tròn vs QAM đa mức)<br>
              Top 3: σ_Δϕ (Tách FSK vs các điều chế khác)<br>
              Top 4: γ_max (Độ tập trung đỉnh phổ)
            </div>
            <ul>
              <li>Xác thực chính xác trực giác vật lý: các đặc trưng hình học chòm sao và pha đóng vai trò chi phối tuyệt đối.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 6 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 7: KIẾN TRÚC MẠNG PYTORCH 1D-CNN (END-TO-END RAW I/Q)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 7 / 14 • Thành viên 2 (Phút 03:45 – 07:30)</span>
    </div>
    <h1 class="slide-title">Kiến Trúc Mạng PyTorch 1D-CNN (End-to-End Raw I/Q)</h1>
    <p class="slide-subtitle">End-to-End 1D-CNN: Learning Directly from Raw Complex Tensors [N, 2, 128]</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">🧱 3 Khối Tích chập Đặc trưng Conv1D</div>
          <div class="card-content">
            <div class="formula-box" style="font-size: 0.77rem;">
              • Block 1: Conv1D(2 ➔ 64, k=7, p=3)<br>
              &nbsp;&nbsp;➔ BatchNorm1d ➔ ReLU ➔ MaxPool(2)<br>
              • Block 2: Conv1D(64 ➔ 128, k=5, p=2)<br>
              &nbsp;&nbsp;➔ BatchNorm1d ➔ ReLU ➔ MaxPool(2)<br>
              • Block 3: Conv1D(128 ➔ 256, k=3, p=1)<br>
              &nbsp;&nbsp;➔ BatchNorm1d ➔ ReLU ➔ MaxPool(2)
            </div>
            <ul>
              <li>Kernel thu nhỏ dần (7 ➔ 5 ➔ 3) giúp mạng bao quát từ tương quan xung RRC đến chi tiết vi sai ký hiệu.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🧠 Phân loại & Regularization</div>
          <div class="card-content">
            <div class="formula-box">
              Flatten (4096)<br>
              ➔ Linear(4096 ➔ 256) ➔ ReLU<br>
              ➔ Dropout(p = 0.5)<br>
              ➔ Linear(256 ➔ 8 classes)
            </div>
            <ul>
              <li><strong>Tổng Tham số:</strong> <strong>1,178,632 tham số</strong>.</li>
              <li><strong>Dung lượng Checkpoint:</strong> <strong>~4.7 MB</strong>, rất nhỏ gọn cho việc nạp vào bộ nhớ RAM của các chip nhúng trung bình.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">⚙️ Chiến lược Huấn luyện Bền vững</div>
          <div class="card-content">
            <div class="formula-box">
              Optimizer: AdamW (lr = 1e-3, wd = 1e-4)<br>
              Scheduler: ReduceLROnPlateau(patience=3)<br>
              Early Stopping: patience = 7
            </div>
            <ul>
              <li>Huấn luyện với Cross-Entropy Loss, batch_size = 128.</li>
              <li>Mạng hội tụ sau khoảng 25 - 35 epochs mà không gặp hiện tượng quá khớp (overfitting).</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 7 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 8: SO SÁNH ĐỘ PHỨC TẠP TÍNH TOÁN & ĐỘ TRỄ SUY LUẬN
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 8 / 14 • Thành viên 2 (Phút 03:45 – 07:30)</span>
    </div>
    <h1 class="slide-title">So Sánh Độ Phức Tạp Tính Toán, Bộ Nhớ & Độ Trễ</h1>
    <p class="slide-subtitle">Inference Latency, Memory Footprint & Hardware Feasibility Benchmark on CPU</p>

    <div class="slide-content">
      <div class="cards-grid grid-split-table">
        <div class="card" style="padding: 10px 14px;">
          <div class="card-title" style="margin-bottom: 6px;">📊 Bảng So sánh Benchmark Thực tế trên CPU</div>
          <table>
            <thead>
              <tr>
                <th>Tiêu chí Đánh giá</th>
                <th>Feature + XGBoost</th>
                <th>Feature + RBF-SVM</th>
                <th>1D-CNN (PyTorch CPU)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Kích thước File Model</strong></td>
                <td><strong style="color:var(--accent);">0.8 MB (Siêu nhẹ)</strong></td>
                <td>8.4 MB</td>
                <td>4.7 MB</td>
              </tr>
              <tr>
                <td><strong>Thời gian Trích xuất Đặc trưng</strong></td>
                <td>0.85 ms / burst</td>
                <td>0.85 ms / burst</td>
                <td><strong style="color:var(--accent);">0.00 ms (Không cần)</strong></td>
              </tr>
              <tr>
                <td><strong>Thời gian Suy luận Model</strong></td>
                <td><strong style="color:var(--accent);">0.08 ms / burst</strong></td>
                <td>1.15 ms / burst</td>
                <td>0.42 ms / burst</td>
              </tr>
              <tr>
                <td><strong>Tổng Độ trễ (Total Latency)</strong></td>
                <td>0.93 ms / burst</td>
                <td>2.00 ms / burst</td>
                <td><strong style="color:var(--accent);">0.42 ms / burst</strong></td>
              </tr>
              <tr>
                <td><strong>Throughput Xử lý Batch</strong></td>
                <td>1,075 bursts / giây</td>
                <td>500 bursts / giây</td>
                <td><strong style="color:var(--accent);">2,380 bursts / giây</strong></td>
              </tr>
              <tr>
                <td><strong>Yêu cầu Phần cứng Tối thiểu</strong></td>
                <td>Vi điều khiển ARM Cortex-M/A</td>
                <td>CPU Máy chủ chuẩn</td>
                <td>NPU / DSP / GPU nhúng</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div style="display: flex; flex-direction: column; gap: 12px;">
          <div class="card">
            <div class="card-title">💡 Phân tích Bản chất Kiến trúc</div>
            <div class="card-content">
              <ul>
                <li><strong>1D-CNN Đạt Throughput Cao Hơn:</strong> Khi xử lý theo batch trên tensor thô, CNN bỏ qua hoàn toàn chi phí vòng lặp Python cho FFT và Cumulant, tận dụng tối đa SIMD/AVX trên CPU.</li>
                <li><strong>XGBoost Vượt Trội về Kích thước:</strong> Chiếm vỏn vẹn 0.8 MB và thời gian suy luận thuần chỉ 0.08 ms, cực kỳ lý tưởng cho các thiết bị Edge IoT.</li>
              </ul>
            </div>
          </div>

          <div class="transition-cue">
            👉 <strong>Lời chuyển giao (Transition Cue sang Thành viên 3):</strong> "Để trả lời câu hỏi cốt lõi: hiệu năng nhận dạng thực tế giữa trích xuất đặc trưng chuyên gia và học sâu End-to-End chênh lệch thế nào trên từng vùng SNR, xin mời bạn [Thành viên 3] tiếp tục với phần Kết quả Thực nghiệm và Phân tích Lỗi gốc rễ đầu tiên."
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 8 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 9: KẾT QUẢ THỰC NGHIỆM: ĐƯỜNG CONG SNR VS ACCURACY / MACRO-F1
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 9 / 14 • Thành viên 3 (Phút 07:30 – 11:15)</span>
    </div>
    <h1 class="slide-title">Kết Quả Thực Nghiệm: Đường Cong SNR vs. Accuracy & Macro-F1</h1>
    <p class="slide-subtitle">Experimental Results: SNR Benchmark Curves on Independent Test Set (n = 7,200 Bursts)</p>

    <div class="slide-content">
      <div class="cards-grid grid-split-plot">
        <div class="plot-container">
          <img src="{img_snr_curves}" alt="SNR Benchmark Curves">
        </div>

        <div style="display: flex; flex-direction: column; gap: 10px;">
          <div class="card">
            <div class="card-title">📉 Vùng SNR Thấp (-10 dB đến 0 dB)</div>
            <div class="card-content">
              <ul>
                <li><strong>Tại -4 dB:</strong> <strong>1D-CNN đạt 48.75%</strong>, vượt trội hoàn toàn <strong>XGBoost (31.67%)</strong> và <strong>RBF-SVM (32.29%)</strong> với mức chênh lệch áp đảo <span class="badge badge-success">+17.08%</span>.</li>
                <li><strong>Tại 0 dB:</strong> 1D-CNN bứt phá lên <strong>74.58%</strong>, trong khi XGBoost chỉ đạt <strong>46.67%</strong> (+27.91%).</li>
              </ul>
            </div>
          </div>

          <div class="card">
            <div class="card-title">📈 Vùng SNR Cao (+4 dB đến +18 dB)</div>
            <div class="card-content">
              <ul>
                <li>Cả 3 mô hình đều bứt phá mạnh mẽ vượt ngưỡng 80% từ mốc +6 dB.</li>
                <li><strong>Hội tụ Giới hạn:</strong> Tại SNR ≥ +12 dB, cả XGBoost và 1D-CNN cùng hội tụ ở mức <strong>90% – 93%</strong>, tiệm cận giới hạn phân tách tối đa cho độ dài khung L = 128 mẫu.</li>
              </ul>
            </div>
          </div>

          <div class="card">
            <div class="card-title">🎯 Chỉ số Macro-F1 Bền vững</div>
            <div class="card-content">
              <ul>
                <li>Đường Macro-F1 bám sát tuyệt đối đường Accuracy trên mọi mức SNR, chứng minh mô hình không hề thiên vị vào các lớp dễ phân loại (như BPSK).</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 9 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 10: MA TRẬN NHẦM LẪN (CONFUSION MATRIX) QUA CÁC CHẾ ĐỘ NHIỄU
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 10 / 14 • Thành viên 3 (Phút 07:30 – 11:15)</span>
    </div>
    <h1 class="slide-title">Phân Tích Ma Trận Nhầm Lẫn Qua 3 Chế Độ Nhiễu Kênh</h1>
    <p class="slide-subtitle">Confusion Matrix Analysis Across Low (-4 dB), Medium (+4 dB), and High (+14 dB) SNR Regimes</p>

    <div class="slide-content">
      <div class="cards-grid grid-split-plot">
        <div class="plot-container">
          <img src="{img_confusion}" alt="Confusion Matrix Across Regimes">
        </div>

        <div style="display: flex; flex-direction: column; gap: 10px;">
          <div class="card">
            <div class="card-title">🔴 Low SNR (-4 dB): Nhiễu Nặng</div>
            <div class="card-content">
              <ul>
                <li><strong>XGBoost:</strong> Nhầm lẫn lan tràn giữa các lớp do các đặc trưng Cumulant bị co rút về 0.</li>
                <li><strong>1D-CNN:</strong> Duy trì độ chính xác cao cho BPSK (88%), PAM4 (82%) và nhận diện đúng một phần FSK nhờ tương quan thời gian.</li>
              </ul>
            </div>
          </div>

          <div class="card">
            <div class="card-title">🟡 Medium SNR (+4 dB): Nhiễu Vừa</div>
            <div class="card-content">
              <ul>
                <li>BPSK, PAM4, CPFSK, GFSK đạt độ chính xác gần như tuyệt đối (> 95%).</li>
                <li>Nhầm lẫn chỉ còn khu trú trong nội bộ họ điều chế biên độ/pha: QPSK ⟷ 8PSK ⟷ 16QAM.</li>
              </ul>
            </div>
          </div>

          <div class="card">
            <div class="card-title">🟢 High SNR (+14 dB): Không Nhiễu</div>
            <div class="card-content">
              <ul>
                <li>6 / 8 lớp điều chế đạt độ chính xác hoàn hảo <strong>100.0%</strong>.</li>
                <li><strong>Điểm nghẽn Duy nhất:</strong> Sự nhầm lẫn giữa <strong>16QAM ⟷ 64QAM</strong> (XGBoost nhầm 34.0%, CNN nhầm 34.7%).</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 10 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 11: PHÂN TÍCH LỖI GỐC RỄ 1: SỰ SUY SỤP CỦA CUMULANTS Ở SNR THẤP
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 11 / 14 • Thành viên 3 (Phút 07:30 – 11:15)</span>
    </div>
    <h1 class="slide-title">Lỗi Gốc Rễ 1: Sự Suy Sụp Của Cumulants Ở SNR Thấp</h1>
    <p class="slide-subtitle">Root-Cause Failure 1: The Low-SNR Collapse of Higher-Order Cumulants</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">📉 Cơ chế A: Contraction Factor</div>
          <div class="card-content">
            <p>Sau khi chuẩn hóa công suất per-burst E[|r|²] = E[|s|²] + σ_w²:</p>
            <div class="formula-box">
              C_42,observed = C_42,signal · [ SNR / (1 + SNR) ]²
            </div>
            <ul>
              <li>Tại SNR = -4 dB (tỷ số tuyến tính ρ ≈ 0.398):
                <div class="formula-box" style="margin: 4px 0;">
                  [ 0.398 / 1.398 ]² ≈ <strong>0.081</strong>
                </div>
              </li>
              <li>Giá trị kỳ vọng của Cumulant bị co rút hơn <strong>12 lần về 0</strong>, khiến cụm đặc trưng của mọi chòm sao bị ép sát vào nhau!</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">💥 Cơ chế B: Bùng nổ Phương sai Mẫu</div>
          <div class="card-content">
            <p>Ước lượng mẫu hữu hạn (L = 128 mẫu):</p>
            <div class="formula-box">
              Var{{ Ĉ_42 }} ∝ ( E[|w|⁸] - (E[|w|⁴])² ) / L
            </div>
            <ul>
              <li><strong>Độ lệch chuẩn σ_C42 không hề giảm theo SNR</strong> mà bị chi phối bởi moment bậc 8 của nhiễu Gauss!</li>
              <li>Tại SNR thấp, kỳ vọng μ co về 0 trong khi phương sai giữ nguyên, làm nhiễu áp đảo hoàn toàn tín hiệu đặc trưng.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">📊 Đo đạc Thực tế trên Test Set</div>
          <div class="card-content">
            <p>Tỷ số chất lượng đặc trưng ("Feature SNR" = |μ| / σ):</p>
            <div class="formula-box">
              • QPSK (+14 dB): |μ|/σ = <strong>14.51</strong> (rất rõ)<br>
              • QPSK (-4 dB): |μ|/σ = <strong>0.64</strong> (|μ| &lt; σ!)<br>
              • 16QAM (+14 dB): |μ|/σ = <strong>5.24</strong><br>
              • 16QAM (-4 dB): |μ|/σ = <strong>0.38</strong> (suy sụp)
            </div>
            <ul>
              <li><strong>Hệ quả:</strong> Khi |μ|/σ &lt; 1, phân loại dựa trên đặc trưng sụp đổ hoàn toàn thành đoán mò ngẫu nhiên.</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="transition-cue">
        👉 <strong>Lời chuyển giao (Transition Cue sang Thành viên 4):</strong> "Vậy tại sao 1D-CNN lại giải quyết được bài toán hóc búa này để vượt trội hơn tới +17% ở SNR thấp, và đâu là giải pháp cho điểm nghẽn kinh điển 16QAM vs 64QAM? Xin mời bạn [Thành viên 4] phân tích cơ chế vật lý chuyên sâu và điều khiển phần Live Demo tương tác."
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 11 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 12: PHÂN TÍCH LỖI GỐC RỄ 2: TẠI SAO 1D-CNN THẮNG THẾ Ở SNR THẤP?
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 12 / 14 • Thành viên 4 (Phút 11:15 – 15:00)</span>
    </div>
    <h1 class="slide-title">Lỗi Gốc Rễ 2: Vì Sao 1D-CNN Thắng Thế Ở SNR Thấp? (+17.08%)</h1>
    <p class="slide-subtitle">Root-Cause Failure 2: Why 1D-CNN Outperforms at Low SNR via Implicit Matched Filtering</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">🔍 Tích chập = Lọc Phối hợp Ngầm định</div>
          <div class="card-content">
            <p>Tầng Conv1D trượt các kernel W ∈ ℝ^(2×K) qua 128 mẫu I/Q:</p>
            <div class="formula-box">
              y_k[n] = ∑_(c∈{{I,Q}}) ∑_(τ=0)^(K-1) W_k[c, τ] · r_c[n - τ]
            </div>
            <ul>
              <li>Quá trình Backpropagation tối ưu các trọng số kernel W hội tụ thành các <strong>xung tương quan thời gian</strong> (matched filters) bám sát đáp ứng xung RRC và nhịp chuyển tiếp ký hiệu.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">📈 Gia tăng SNR Hiệu dụng (Processing Gain)</div>
          <div class="card-content">
            <p>Tích lũy kết hợp năng lượng tín hiệu (Coherent Integration):</p>
            <div class="formula-box">
              SNR_effective ≈ SNR_in + 10 log₁₀( K · SPS )
            </div>
            <ul>
              <li>Trong khi tín hiệu được cộng đồng pha, nhiễu trắng AWGN bị triệt tiêu một phần do tính chất không kết hợp.</li>
              <li>Nhờ đó, 1D-CNN nâng mức SNR thực tế trước khi đưa vào các tầng phi tuyến.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">⚖️ So sánh Bản chất Toán học</div>
          <div class="card-content">
            <ul>
              <li><strong>Feature Engineering:</strong> Nâng lũy thừa bậc 4 tức thời trên từng mẫu riêng rẽ (|r[n]|⁴) ➔ <strong>Khuếch đại phi tuyến phương sai nhiễu</strong>.</li>
              <li><strong>1D-CNN:</strong> Tích hợp tuyến tính trước qua cửa sổ thời gian (Temporal Smoothing) ➔ <strong>Khử nhiễu trước khi tạo đặc trưng phi tuyến</strong>.</li>
              <li>Kết quả: 1D-CNN vượt trội +17.08% tại -4 dB và +27.91% tại 0 dB!</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 12 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 13: PHÂN TÍCH LỖI GỐC RỄ 3: SỰ BẾ TẮC KINH ĐIỂN 16QAM VS 64QAM
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 13 / 14 • Thành viên 4 (Phút 11:15 – 15:00)</span>
    </div>
    <h1 class="slide-title">Lỗi Gốc Rễ 3: Điểm Nghẽn Kinh Điển 16QAM vs. 64QAM</h1>
    <p class="slide-subtitle">Root-Cause Failure 3: The Intrinsic 16QAM vs. 64QAM Ambiguity & 4 Concrete Architectural Solutions</p>

    <div class="slide-content">
      <div class="cards-grid grid-3">
        <div class="card">
          <div class="card-title">📏 Khoảng cách Lý thuyết Quá Hẹp</div>
          <div class="card-content">
            <div class="formula-box">
              C_42(16QAM) = -0.680 vs C_42(64QAM) = -0.619<br>
              ➔ Khoảng cách cực hẹp: Δ C_42 = 0.061
            </div>
            <p>Đo thực tế trên Test Set (L = 128 mẫu, có xung RRC):</p>
            <div class="formula-box">
              16QAM: -0.551 ± 0.104<br>
              64QAM: -0.497 ± 0.103<br>
              Chênh lệch Δ = 0.054 vs Độ lệch chuẩn σ ≈ 0.103
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🎯 Phân tách Fisher & Xác suất Bayes</div>
          <div class="card-content">
            <p>Chỉ số phân tách Fisher:</p>
            <div class="formula-box">
              d' = Δμ / σ_pooled = 0.054 / 0.103 ≈ 0.52
            </div>
            <p>Xác suất lỗi Bayes lý thuyết cho 1 chiều đặc trưng:</p>
            <div class="formula-box">
              P_e = Q( d' / 2 ) = Q( 0.26 ) ≈ <strong>39.7%</strong>
            </div>
            <ul>
              <li><strong>Thực tế quan sát:</strong> XGBoost nhầm 34.0%, 1D-CNN nhầm 34.7%, hoàn toàn khớp với lý thuyết Bayes!</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🛠️ 4 Giải pháp Đột phá Khắc phục</div>
          <div class="card-content">
            <ul>
              <li><strong>1. Tăng chiều dài khung L:</strong> Vì σ ∝ 1/√L, tăng L = 512 hoặc 1024 sẽ thu hẹp σ xuống một nửa, tách rời 2 phân phối.</li>
              <li><strong>2. Bổ sung Cumulant Bậc 6 (C_63):</strong> Tỷ lệ C_63 / C_42 có độ nhạy phân tách cao hơn với chòm sao QAM mật độ lớn.</li>
              <li><strong>3. Khôi phục Định thời (Timing Recovery):</strong> Lấy mẫu đúng đỉnh mắt để loại bỏ nhiễu ISI chuyển tiếp xung.</li>
              <li><strong>4. Mạng Sâu hơn:</strong> Residual 1D-CNN hoặc Complex-Valued CNN (CV-CNN).</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 13 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 14: KẾT LUẬN & TRÌNH DIỄN LIVE DEMO STREAMLIT
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • TOPIC 02: AMC DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Slide 14 / 14 • Thành viên 4 (Phút 11:15 – 15:00)</span>
    </div>
    <h1 class="slide-title">Kết Luận, Khuyến Nghị Kỹ Thuật & Live Demo Streamlit</h1>
    <p class="slide-subtitle">Conclusions, Key Takeaways & Live Interactive Demonstration</p>

    <div class="slide-content">
      <div class="cards-grid grid-2">
        <div class="card">
          <div class="card-title">🎯 Tóm tắt Đóng góp & Khuyến nghị Kỹ thuật</div>
          <div class="card-content">
            <ul>
              <li><strong>Môi trường Nhiễu nặng (SNR &lt; 0 dB):</strong> Khuyến nghị tuyệt đối sử dụng <strong>1D-CNN</strong> nhờ khả năng lọc tương quan thời gian vượt trội (+17.08% tại -4 dB).</li>
              <li><strong>Môi trường IoT Tiết kiệm Năng lượng:</strong> Khuyến nghị <strong>XGBoost + Handcrafted Features</strong> khi SNR ≥ +4 dB để tối ưu độ trễ siêu thấp (0.08 ms) và kích thước mô hình siêu nhẹ (0.8 MB).</li>
              <li><strong>Tri thức Vật lý:</strong> Feature Engineering giúp giải thích minh bạch hệ thống và cung cấp căn cứ toán học vững chắc cho việc thiết kế mô hình AI.</li>
            </ul>
          </div>
        </div>

        <div class="card">
          <div class="card-title">💻 Trình diễn Live Streamlit Dashboard</div>
          <div class="card-content">
            <ul>
              <li><strong>Trực quan hóa Thời gian Thực:</strong> Giản đồ chòm sao (Constellation), dạng sóng I/Q thời gian và phổ công suất FFT.</li>
              <li><strong>Kéo Slider SNR Tương tác:</strong> Giảm SNR từ +18 dB về -4 dB, chứng kiến chòm sao bị biến dạng và phổ FFT bị san phẳng bởi sàn nhiễu.</li>
              <li><strong>Đối đầu Trực tiếp tại -4 dB:</strong> Kiểm chứng thực tế khi XGBoost đoán sai do suy sụp Cumulant, trong khi 1D-CNN giữ vững dự đoán chính xác với thanh Confidence Score cao.</li>
            </ul>
            <div style="margin-top: 12px; text-align: center;">
              <span class="badge badge-success" style="font-size: 0.85rem; padding: 6px 16px;">
                🚀 Sẵn sàng trả lời Q&A từ Hội đồng Giảng viên!
              </span>
            </div>
          </div>
        </div>
      </div>

      <div class="transition-cue">
        👉 <strong>Lời kết & Chuyển giao sang Q&A (Thành viên 4):</strong> "Đó là toàn bộ phần trình bày của nhóm Topic 02. Chúng em xin chân thành cảm ơn Quý Thầy/Cô trong Hội đồng đã chú ý lắng nghe, và sau đây cả 4 thành viên xin sẵn sàng bước vào phần giải đáp thắc mắc (Q&A)."
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Brief</span>
      <span>Slide 14 of 14</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 15: Q&A CÂU HỎI 1 (BASEBAND & RMS NORMALIZATION)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • BỘ CÂU HỎI & TRẢ LỜI MẪU CHO PHẦN Q&A (20/20 ĐIỂM)</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Câu hỏi 1 / 5 • Phụ trách chính: Thành viên 1</span>
    </div>
    <h1 class="slide-title">Q&A 1: Mô Hình Kênh Baseband & Chuẩn Hóa Năng Lượng RMS</h1>
    <p class="slide-subtitle">Chuyên sâu về Hiện tượng Path Loss, Fading Đa đường và Lý do Chọn Chuẩn hóa RMS thay vì Peak</p>

    <div class="slide-content">
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <div class="card card-qa-q">
          <div class="card-title" style="color: #B45309;">❓ Giảng viên Hội đồng hỏi:</div>
          <div class="card-content" style="font-size: 1.05rem; font-weight: 600; color: #1E293B;">
            "Tại sao các em lại chọn chuẩn hóa năng lượng RMS trên từng khung (Per-burst RMS Normalization) mà không chuẩn hóa max biên độ hay để nguyên biên độ thu?"
          </div>
        </div>

        <div class="card card-qa-a">
          <div class="card-title" style="color: var(--accent-dark);">💬 Trả lời Chuẩn Kỹ sư RF (Thành viên 1 phụ trách chính):</div>
          <div class="card-content">
            <p><strong>"Dạ thưa Thầy/Cô, trong kênh truyền vô tuyến thực tế:"</strong></p>
            <ul>
              <li><strong>1. Biến động Công suất do Kênh truyền:</strong> Công suất tín hiệu thu bị biến động liên tục bởi cự ly truyền dẫn (Path Loss) và suy hao đa đường (Large-scale fading). Nếu không chuẩn hóa, các mô hình học máy sẽ 'học vẹt' công suất thu thay vì nhận dạng hình học điều chế.</li>
              <li><strong>2. Nhược điểm Chí mạng của Peak Normalization:</strong> Peak Normalization cực kỳ nhạy cảm với các mẫu nhiễu ngoại lai (noise spikes) ở SNR thấp. Chỉ cần 1 mẫu nhiễu biên độ cao xuất hiện, toàn bộ các mẫu tín hiệu còn lại sẽ bị nén nghẽn biên độ giả tạo.</li>
              <li><strong>3. Bảo toàn Lý thuyết Toán học:</strong> RMS Normalization đưa công suất trung bình về đúng 1.0 (E[|r̃|²] = 1), bảo toàn tỷ lệ năng lượng lý thuyết giữa các mức chòm sao và đảm bảo các công thức Cumulants (C_21 = 1, C_42) giữ nguyên tính bất biến toán học."</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Q&A</span>
      <span>Q&A Slide 1 of 5</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 16: Q&A CÂU HỎI 2 (SUY SỤP CUMULANT Ở LOW-SNR)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • BỘ CÂU HỎI & TRẢ LỜI MẪU CHO PHẦN Q&A (20/20 ĐIỂM)</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Câu hỏi 2 / 5 • Phụ trách chính: Thành viên 3</span>
    </div>
    <h1 class="slide-title">Q&A 2: Hiện Tượng Suy Sụp Toán Học Của Cumulant Ở Low-SNR</h1>
    <p class="slide-subtitle">Giải thích Nghịch lý: Nhiễu Gauss có C42 = 0 nhưng Cumulant Tín hiệu Thu lại bị Co rút Nghiêm trọng</p>

    <div class="slide-content">
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <div class="card card-qa-q">
          <div class="card-title" style="color: #B45309;">❓ Giảng viên Hội đồng hỏi:</div>
          <div class="card-content" style="font-size: 1.05rem; font-weight: 600; color: #1E293B;">
            "Về mặt toán học, nhiễu Gauss độc lập có Cumulant bậc 4 bằng 0 (C_42(w) = 0). Vậy tại sao ở SNR thấp Cumulant của tín hiệu thu lại không bằng Cumulant của tín hiệu gốc mà bị suy giảm nghiêm trọng?"
          </div>
        </div>

        <div class="card card-qa-a">
          <div class="card-title" style="color: var(--accent-dark);">💬 Trả lời Chuẩn Kỹ sư RF (Thành viên 3 phụ trách chính):</div>
          <div class="card-content">
            <p><strong>"Dạ thưa Thầy/Cô, đúng là C_42(s + w) = C_42(s) + C_42(w) = C_42(s). Tuy nhiên, đó là khi tín hiệu CHƯA chuẩn hóa:"</strong></p>
            <ul>
              <li><strong>Hệ số Co rút do Chuẩn hóa Công suất:</strong> Khi ta chuẩn hóa tín hiệu thu r = s + w có công suất P_r = P_s + σ_w², tín hiệu chuẩn hóa trở thành r̃ = (s + w) / √(P_s + σ_w²). Do tính chất thuần nhất Cum(α r) = α⁴ Cum(r):
                <div class="formula-box">
                  C_42(r̃) = C_42(s) / (P_s + σ_w²)² = C_42(s) · [ SNR / (1 + SNR) ]²
                </div>
                Ở -4 dB (SNR = 0.398), hệ số này là [0.398 / 1.398]² ≈ <strong>0.081</strong>, làm kỳ vọng Cumulant bị co rút tới <strong>92% về 0</strong>.
              </li>
              <li><strong>Phương sai Ước lượng Mẫu Hữu hạn (L = 128):</strong> Mẫu hữu hạn chứa moment bậc 8 của nhiễu (E[|w|⁸]), khiến độ lệch chuẩn giữ nguyên không đổi trong khi kỳ vọng co về 0, làm tỷ số Feature SNR |μ|/σ rơi xuống dưới 1.0."</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Q&A</span>
      <span>Q&A Slide 2 of 5</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 17: Q&A CÂU HỎI 3 (TƯƠNG QUAN CHÉO I/Q TRONG 1D-CNN)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • BỘ CÂU HỎI & TRẢ LỜI MẪU CHO PHẦN Q&A (20/20 ĐIỂM)</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Câu hỏi 3 / 5 • Phụ trách chính: Thành viên 2 (TV4 hỗ trợ)</span>
    </div>
    <h1 class="slide-title">Q&A 3: Cơ Chế Tương Quan Chéo I/Q Trong Mạng 1D-CNN</h1>
    <p class="slide-subtitle">Làm Thế Nào Mạng CNN 1D Thực Lại Học Được Quan Hệ Pha Phức Giữa Kênh In-phase & Quadrature?</p>

    <div class="slide-content">
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <div class="card card-qa-q">
          <div class="card-title" style="color: #B45309;">❓ Giảng viên Hội đồng hỏi:</div>
          <div class="card-content" style="font-size: 1.05rem; font-weight: 600; color: #1E293B;">
            "Mạng 1D-CNN của nhóm nhận đầu vào là tensor [2, 128]. Làm thế nào một mạng CNN thuần 1D thực lại hiểu được quan hệ góc pha phức giữa kênh I và kênh Q?"
          </div>
        </div>

        <div class="card card-qa-a">
          <div class="card-title" style="color: var(--accent-dark);">💬 Trả lời Chuẩn Kỹ sư RF (Thành viên 2 phụ trách chính, TV4 hỗ trợ):</div>
          <div class="card-content">
            <p><strong>"Dạ thưa Thầy/Cô, ở tầng tích chập đầu tiên Conv1d(in_channels=2, out_channels=64, kernel_size=7):"</strong></p>
            <ul>
              <li><strong>Ma trận Trọng số Kernel:</strong> Mỗi bộ lọc có kích thước 2 × 7. Khi trượt trên chuỗi thời gian, kernel tính tổng chéo:
                <div class="formula-box">
                  y[n] = ∑_(τ=0)^6 [ W_I[τ] · I[n - τ] + W_Q[τ] · Q[n - τ] ]
                </div>
              </li>
              <li><strong>Đồng nhất với Phép Nhân Tích chập Phức:</strong> Cấu trúc này tương đương về mặt đại số với phần thực của phép nhân số phức:
                <div class="formula-box">
                  Re{{ (W_I - j W_Q) * (I + j Q) }}
                </div>
              </li>
              <li><strong>Học Tốc độ Quay Pha Không Cần arctan:</strong> Các bộ lọc thực hiện tương quan chéo giữa I và Q qua các thời điểm trễ τ, từ đó tự động học được tốc độ quay pha Δθ = ω τ và phân biệt được chiều quay của vector chòm sao."</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Q&A</span>
      <span>Q&A Slide 3 of 5</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 18: Q&A CÂU HỎI 4 (ĐIỂM NGHẼN 16QAM VS 64QAM)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • BỘ CÂU HỎI & TRẢ LỜI MẪU CHO PHẦN Q&A (20/20 ĐIỂM)</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Câu hỏi 4 / 5 • Phụ trách chính: Thành viên 4</span>
    </div>
    <h1 class="slide-title">Q&A 4: Điểm Nghẽn 16QAM vs. 64QAM & Giải Pháp Thực Tế</h1>
    <p class="slide-subtitle">Khoảng cách Fisher, Giới hạn Lỗi Bayes Lý thuyết và Ưu tiên Giải pháp Tăng Chiều dài Khung Quan sát</p>

    <div class="slide-content">
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <div class="card card-qa-q">
          <div class="card-title" style="color: #B45309;">❓ Giảng viên Hội đồng hỏi:</div>
          <div class="card-content" style="font-size: 1.05rem; font-weight: 600; color: #1E293B;">
            "Tại sao cả mô hình Feature Engineering lẫn Deep Learning đều gặp khó khăn lớn nhất khi phân biệt 16QAM và 64QAM, và nếu được nâng cấp hệ thống trên thực tế, nhóm sẽ làm gì đầu tiên?"
          </div>
        </div>

        <div class="card card-qa-a">
          <div class="card-title" style="color: var(--accent-dark);">💬 Trả lời Chuẩn Kỹ sư RF (Thành viên 4 phụ trách chính):</div>
          <div class="card-content">
            <p><strong>"Dạ thưa Thầy/Cô:"</strong></p>
            <ul>
              <li><strong>1. Về Mặt Hình học & Xác suất:</strong> Cả 16QAM và 64QAM đều có chung tính đối xứng quay 90° (bậc 4). Khoảng cách lý thuyết giữa chúng chỉ là |-0.680 - (-0.619)| = 0.061. Với độ dài khung cực ngắn L = 128 mẫu (tương đương 32 ký hiệu ở SPS=4), một số điểm chòm sao ngoài cùng của 64QAM thậm chí chưa kịp xuất hiện trong 32 ký hiệu phát ngẫu nhiên, khiến phân bố biên độ thực tế của 64QAM bị trùng lặp nặng nề với 16QAM.</li>
              <li><strong>2. Giải pháp Thực tế Hiệu quả Nhất:</strong> Giải pháp hiệu quả và rẻ nhất về mặt tính toán là <strong>tăng độ dài khung quan sát lên L = 512 hoặc 1024 mẫu</strong>. Khi đó, phương sai ước lượng σ ∝ 1/√L sẽ giảm đi 2 đến 3 lần, tách rời hai phân phối Gauss của 16QAM và 64QAM, đưa độ chính xác phân tách vượt trên 95%."</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Q&A</span>
      <span>Q&A Slide 4 of 5</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 19: Q&A CÂU HỎI 5 (EDGE DEPLOYMENT TRÊN EMBEDDED IOT)
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • BỘ CÂU HỎI & TRẢ LỜI MẪU CHO PHẦN Q&A (20/20 ĐIỂM)</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Câu hỏi 5 / 5 • Phụ trách chính: Thành viên 2 (TV1 hỗ trợ)</span>
    </div>
    <h1 class="slide-title">Q&A 5: Triển Khai Thực Tế Trên Thiết Bị Nhúng & Edge IoT</h1>
    <p class="slide-subtitle">So sánh Khả thi Phần cứng: Vi điều khiển ARM Cortex-M4 Chạy Pin vs. Yêu cầu Bộ nhớ & Tính toán</p>

    <div class="slide-content">
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <div class="card card-qa-q">
          <div class="card-title" style="color: #B45309;">❓ Giảng viên Hội đồng hỏi:</div>
          <div class="card-content" style="font-size: 1.05rem; font-weight: 600; color: #1E293B;">
            "Nếu nhóm phải triển khai giải pháp này lên một thiết bị IoT chạy pin sử dụng chip vi điều khiển ARM Cortex-M4 không có GPU, nhóm sẽ chọn giải pháp nào?"
          </div>
        </div>

        <div class="card card-qa-a">
          <div class="card-title" style="color: var(--accent-dark);">💬 Trả lời Chuẩn Kỹ sư RF (Thành viên 2 phụ trách chính, TV1 hỗ trợ):</div>
          <div class="card-content">
            <p><strong>"Dạ thưa Thầy/Cô, chúng em chắc chắn sẽ chọn Feature Engineering kết hợp với mô hình cây quyết định (như XGBoost rút gọn hoặc Decision Tree):"</strong></p>
            <ul>
              <li><strong>Rào cản Phần cứng của 1D-CNN:</strong> Mạng 1D-CNN đòi hỏi hơn 1.17 triệu phép tính nhân cộng (MACs) và chiếm gần 5 MB bộ nhớ Flash, vượt quá dung lượng SRAM/Flash thông thường của Cortex-M4 (thường chỉ 64 KB - 256 KB RAM).</li>
              <li><strong>Tính Khả thi Tuyệt đối của Cây Quyết định:</strong> Trích xuất 10 đặc trưng chỉ tốn khoảng vài nghìn phép tính số học cơ bản. Sau khi trích xuất, mô hình cây chỉ thực hiện các phép so sánh ngưỡng `if-else` trên số thực, chiếm chưa tới 50 KB bộ nhớ và suy luận trong dưới 1 ms mà không cần bất kỳ thư viện Tensor hay bộ tăng tốc phần cứng nào."</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>EE136IU: Machine Learning for Wireless Communications | Capstone Mini-Project Q&A</span>
      <span>Q&A Slide 5 of 5</span>
    </div>
  </div>

  <!-- ========================================================================
       SLIDE 20: KẾT THÚC & CẢM ƠN
       ======================================================================== -->
  <div class="slide-page">
    <div class="slide-header">
      <span class="topic-tag">EE136IU • CAPSTONE MINI-PROJECT DEFENSE</span>
      <span class="speaker-tag"><span class="speaker-dot"></span>Hoàn tất Phần Thuyết trình & Phiên Phản biện Q&A</span>
    </div>
    <div class="cover-container" style="text-align: center; align-items: center;">
      <div style="margin-bottom: 20px;">
        <span class="badge badge-success" style="font-size: 1rem; padding: 6px 20px;">PHIÊN BẢO VỆ CHÍNH THỨC HOÀN TẤT</span>
      </div>
      <h1 class="cover-title" style="font-size: 2.6rem;">Trân Trọng Cảm Ơn Quý Thầy/Cô<br><span>Hội Đồng Đã Chú Ý Lắng Nghe!</span></h1>
      <p class="cover-subtitle" style="max-width: 800px; margin: 0 auto 35px auto;">
        Topic 02: Spectral & Temporal Feature Engineering vs. End-to-End Raw I/Q Deep Learning<br>
        Môn học: EE136IU - Machine Learning for Wireless Communications
      </p>

      <div style="display: flex; gap: 24px; justify-content: center; margin-top: 10px;">
        <div class="cover-meta-card" style="width: 280px; text-align: left;">
          <div class="cover-meta-label">Mã nguồn Dự án</div>
          <div class="cover-meta-val" style="font-family: var(--font-mono); font-size: 0.82rem; color: var(--accent);">github.com/.../EET-project</div>
        </div>
        <div class="cover-meta-card" style="width: 280px; text-align: left;">
          <div class="cover-meta-label">Live Interactive Demo</div>
          <div class="cover-meta-val" style="font-family: var(--font-mono); font-size: 0.82rem; color: var(--accent);">Streamlit Local App :8501</div>
        </div>
        <div class="cover-meta-card" style="width: 280px; text-align: left;">
          <div class="cover-meta-label">Đơn vị Đào tạo</div>
          <div class="cover-meta-val">Trường ĐH Quốc Tế (ĐHQG-HCM)</div>
        </div>
      </div>
    </div>
    <div class="slide-footer">
      <span>Trường Đại học Quốc Tế - ĐHQG-HCM • Khoa Điện Tử Viễn Thông</span>
      <span>Slide 20 / 20 • Thank You</span>
    </div>
  </div>

</body>
</html>
"""
    return html

def build_pdf():
    html_content = generate_html()
    html_path = os.path.join(PROJECT_ROOT, 'PRESENTATION_SLIDES_TOPIC2.html')
    pdf_path = os.path.join(PROJECT_ROOT, 'PRESENTATION_SLIDES_TOPIC2.pdf')

    print(f"[1/3] Writing {html_path}...")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break

    if not edge_exe:
        print("ERROR: Microsoft Edge executable not found!")
        sys.exit(1)

    print(f"[2/3] Printing to PDF using Microsoft Edge headless: {edge_exe}...")
    file_url = f"file:///{html_path.replace(os.sep, '/')}"

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        file_url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"[3/3] SUCCESS: Created {pdf_path} ({size:,} bytes)")
    else:
        print(f"ERROR: Failed to create PDF! Exit code: {result.returncode}")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        sys.exit(1)

if __name__ == '__main__':
    build_pdf()
