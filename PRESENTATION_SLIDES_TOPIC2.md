# BỘ SLIDE THUYẾT TRÌNH BẢO VỆ CAPSTONE MINI-PROJECT (15 PHÚT)
## TOPIC 02: Spectral & Temporal Feature Engineering vs. End-to-End Raw I/Q Deep Learning
**Môn học:** EE136IU - Machine Learning for Wireless Communications  
**Giảng viên hướng dẫn:** Bộ môn Viễn thông - Trường ĐH Quốc Tế (ĐHQG-HCM)  
**Tiêu chuẩn thiết kế:** Clean White Academic (#FFFFFF / #F8FAFC / #0F172A / Accent #0284C7)

---

### GỢI Ý PHÂN BỔ THỜI GIAN NHÓM (3 THÀNH VIÊN)
- **Thành viên 1 (Phút 00:00 – 04:30 | Slides 1 - 4):** Đặt vấn đề AMC, Mô hình Kênh vô tuyến Vật lý cơ sở, và Nền tảng Toán học Higher-Order Cumulants.
- **Thành viên 2 (Phút 04:30 – 09:30 | Slides 5 - 8):** Kiến trúc Trích xuất Đặc trưng vs Mạng 1D-CNN Raw I/Q, Quy trình Chống Rò rỉ Dữ liệu (Anti-Leakage).
- **Thành viên 3 (Phút 09:30 – 15:00 | Slides 9 - 14):** Kết quả Benchmark, Phân tích Lỗi gốc rễ (Root-Cause Failure Analysis), Trình diễn Live Demo & Kết luận.
- **Cả nhóm (Phút 15:00 – 20:00):** Q&A với Hội đồng Giảng viên.

---

## SLIDE 1: TIÊU ĐỀ DỰ ÁN & GIỚI THIỆU TỔNG QUAN
- **Thời lượng:** 01:00 | **Người nói:** Thành viên 1
- **Tiêu đề Slide:** Automatic Modulation Classification: Expert Feature Engineering vs. End-to-End Deep Learning
- **Thẻ Card 1 [Bối cảnh & Tầm quan trọng]:**
  - Nhận dạng Điều chế Tự động (AMC) là "trái tim" của Vô tuyến Nhận thức (Cognitive Radio) và Mạng 5G/6G phi tập trung.
  - Mục tiêu: Tự động phân loại 8 định dạng điều chế (BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, CPFSK, GFSK) mà **không cần thông tin phụ trợ** (blind classification).
- **Thẻ Card 2 [Câu hỏi Nghiên cứu Cốt lõi]:**
  - Liệu tri thức vật lý chuyên gia (Domain-Expert Handcrafted Features: HOC, Differential Phase, FFT) có còn giữ vị thế trước kiến trúc học sâu End-to-End (1D-CNN) xử lý trực tiếp trên mảng Raw I/Q?
- **Thẻ Card 3 [Đóng góp Kỹ thuật Chính]:**
  - Xây dựng mô hình kênh Baseband chân thực (RRC pulse shaping, AWGN, CFO ngẫu nhiên, fading đa đường).
  - Phân tích toán học định lượng về hiện tượng suy sụp của Cumulants ở SNR thấp và sự nhầm lẫn 16QAM/64QAM.

---

## SLIDE 2: MÔ HÌNH VẬT LÝ BASEBAND & CHUẨN HÓA NĂNG LƯỢNG
- **Thời lượng:** 01:00 | **Người nói:** Thành viên 1
- **Tiêu đề Slide:** Realistic Complex Baseband Signal Generation Pipeline
- **Thẻ Card 1 [Mô hình Tín hiệu Thu]:**
  - $r[n] = \left( \sum_{k} s[k] \cdot g(nT_s - kT) \right) \cdot e^{j(2\pi \Delta f_c n T_s + \theta_0)} + w[n]$
  - Trong đó: $g(t)$ là bộ lọc tạo dạng xung Root Raised Cosine (RRC, $\alpha=0.35$, SPS = 4).
  - $\Delta f_c$: Độ lệch tần số sóng mang ngẫu nhiên (Carrier Frequency Offset - CFO).
  - $w[n] \sim \mathcal{CN}(0, \sigma_w^2)$: Nhiễu trắng Gauss phức (AWGN).
- **Thẻ Card 2 [Chuẩn hóa Công suất Bắt buộc (Per-burst RMS Normalization)]:**
  - $\tilde{r}[n] = \frac{r[n]}{\sqrt{\frac{1}{L} \sum_{m=0}^{L-1} |r[m]|^2}}$ với chiều dài khung ngắn $L = 128$ mẫu.
  - **Ý nghĩa Viễn thông:** Triệt tiêu hoàn toàn sự chênh lệch biên độ công suất thu do khoảng cách hay suy hao đường truyền (Path Loss), ép mọi bộ phân loại phải dựa thuần túy vào **hình học chòm sao** và **đặc tính pha/phổ**.

---

## SLIDE 3: NỀN TẢNG TOÁN HỌC HIGHER-ORDER CUMULANTS (HOC)
- **Thời lượng:** 01:15 | **Người nói:** Thành viên 1
- **Tiêu đề Slide:** Physical Grounding: Constellation Symmetry & Cumulant Invariance
- **Thẻ Card 1 [Công thức Cumulants Bậc 4 Chuẩn hóa]:**
  - $C_{40} = \text{Cum}(r, r, r, r) = M_{40} - 3 M_{20}^2$
  - $C_{42} = \text{Cum}(r, r^*, r, r^*) = M_{42} - |M_{20}|^2 - 2 M_{21}^2$
  - Với moment $M_{pq} = E[r^{p-q} (r^*)^q]$, và tín hiệu chuẩn hóa $M_{21} = E[|r|^2] = 1$.
- **Thẻ Card 2 [Bảng Giá trị Lý thuyết Không Nhiễu ($E_s = 1$)]:**

| Modulation Scheme | Rotational Symmetry | $\|M_{20}\|$ | $\|C_{40}\|$ (Lý thuyết) | $C_{42}$ (Lý thuyết) | Đặc trưng Hình học |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BPSK** | $180^\circ$ (Bậc 2) | 1.000 | **2.000** | **-2.000** | Đối xứng trục thực, $s^2 = 1$ |
| **QPSK** | $90^\circ$ (Bậc 4) | 0.000 | **1.000** | **-1.000** | 4 điểm vòng tròn, $s^4 = -1$ |
| **8PSK** | $45^\circ$ (Bậc 8) | 0.000 | **0.000** | **-1.000** | 8 điểm vòng tròn, $s^4$ triệt tiêu |
| **16QAM** | $90^\circ$ (Bậc 4) | 0.000 | **0.680** | **-0.680** | 3 mức năng lượng, đa biên độ |
| **64QAM** | $90^\circ$ (Bậc 4) | 0.000 | **0.619** | **-0.619** | 9 mức năng lượng, xấp xỉ tròn |
| **PAM4** | $180^\circ$ (Real) | 1.000 | **1.360** | **-1.360** | Điều chế 1 chiều (1D Constellation) |

- **Thẻ Card 3 [Nhận xét Sắc bén]:**
  - $|C_{40}|$ tách tuyệt đối các họ PSK ($2.0 \to 1.0 \to 0.0$).
  - $C_{42}$ tách biệt giữa họ vòng tròn biên độ không đổi (PSK: $C_{42} = -1.0$) và họ đa biên độ (QAM: $C_{42} \in [-0.68, -0.62]$).

---

## SLIDE 4: VECTƠ ĐẶC TRƯNG MIỀN THỜI GIAN & PHỔ (10-D FEATURE SPACE)
- **Thời lượng:** 01:15 | **Người nói:** Thành viên 1
- **Tiêu đề Slide:** Complete 10-Dimensional Handcrafted Feature Vector
- **Thẻ Card 1 [Bộ 4 Đặc trưng Thống kê Bậc cao (Cumulants)]:**
  - $C_{20}, C_{21}, |C_{40}|, C_{42}$: Nắm bắt hình học chòm sao và tính đối xứng quay.
- **Thẻ Card 2 [Bộ 3 Đặc trưng Pha Vi sai (Differential Phase)]:**
  - $\Delta \phi[n] = \text{unwrap}(\arg(r[n] \cdot r^*[n-1]))$.
  - Trích xuất: $\mu_{\Delta \phi}$, $\sigma_{\Delta \phi}$, Max FFT Magnitude của pha vi sai.
  - Phân tách xuất sắc điều chế tần số liên tục (CPFSK, GFSK) có pha tích lũy tuyến tính.
- **Thẻ Card 3 [Bộ 3 Đặc trưng Phổ Công suất FFT (Spectral Features)]:**
  - Đỉnh phổ chuẩn hóa $\gamma_{\max} = \frac{\max |DFT(\tilde{r})|^2}{L}$.
  - Độ rộng phổ Spectral Spread $\sigma_f$ và Độ phẳng phổ Spectral Flatness.
  - Nhận diện độ tập trung năng lượng quanh tần số trung tâm.

---

## SLIDE 5: NGUYÊN TẮC THIẾT KẾ DATASET & CHỐNG RÒ RỈ (ANTI-LEAKAGE)
- **Thời lượng:** 01:15 | **Người nói:** Thành viên 2
- **Tiêu đề Slide:** Zero-Leakage Dataset Partition & Deterministic Reproducibility
- **Thẻ Card 1 [Phân bổ Dữ liệu Độc lập]:**
  - 8 Class $\times$ 15 mức SNR ($-10$ dB đến $+18$ dB, bước $2$ dB) $\times$ 400 bursts = **48,000 bursts**.
  - Chia 70% Train (33,600 bursts, `seed=42`), 15% Validation (7,200 bursts, `seed=43`), 15% Test (7,200 bursts, `seed=44`).
- **Thẻ Card 2 [Quy tắc Chống Rò rỉ Dữ liệu Tuyệt đối (Leakage-Free)]:**
  - **Khởi tạo độc lập:** Tập Test sinh bằng chuỗi bit và nhiễu riêng biệt, không cắt lát (slicing) từ một mảng liên tục.
  - **Fit Scaler chuẩn mực:** `StandardScaler` chỉ `fit` trên tập Train, sau đó áp dụng `transform` cho Val và Test.
- **Thẻ Card 3 [Phân tích Theo Từng Mức SNR (Per-SNR Stratification)]:**
  - Đánh giá độc lập 480 test bursts tại mỗi điểm SNR cụ thể, kèm Khoảng tin cậy Wilson 95% ($CI_{95} \approx \pm 4.5\%$).

---

## SLIDE 6: MÔ HÌNH FEATURE-BASED: XGBOOST & RBF-SVM
- **Thời lượng:** 01:15 | **Người nói:** Thành viên 2
- **Tiêu đề Slide:** Expert Feature Classifiers: Hyperparameters & Optimization
- **Thẻ Card 1 [XGBoost Gradient Boosted Trees]:**
  - Tham số: `n_estimators = 200`, `max_depth = 6`, `learning_rate = 0.1`, `subsample = 0.8`.
  - Ưu điểm: Phân vùng phi tuyến tính cực nhanh trên không gian 10-D; cung cấp trực quan Feature Importance.
  - Thời gian huấn luyện: ~4.2s trên 33,600 mẫu Train.
- **Thẻ Card 2 [RBF Support Vector Machine]:**
  - Tham số: Kernel RBF (`C = 10.0`, `gamma = 'scale'`), kích hoạt `probability = True` (Platt Scaling).
  - Tối ưu tính toán: Giới hạn tập train của SVM ở 12,000 bursts để tránh nghẽn $O(N^2)$ của thuật toán giải QP, bảo toàn 100% tập Val/Test.
- **Thẻ Card 3 [Xếp hạng Tầm quan trọng Đặc trưng (Feature Importance)]:**
  - Top 1: $|C_{40}|$ (Phân biệt PSK/QAM).
  - Top 2: $C_{42}$ (Tách biệt PSK vòng tròn vs QAM đa mức).
  - Top 3: $\sigma_{\Delta \phi}$ (Phân tách sắc bén FSK vs các điều chế số khác).

---

## SLIDE 7: KIẾN TRÚC MẠNG PYTORCH 1D-CNN (END-TO-END RAW I/Q)
- **Thời lượng:** 01:30 | **Người nói:** Thành viên 2
- **Tiêu đề Slide:** End-to-End 1D-CNN: Learning Directly from Raw Complex Tensors
- **Thẻ Card 1 [Kiến trúc Mạng Tensor $[N, 2, 128]$]:**
  - **Conv1D Block 1:** 64 bộ lọc, kernel size = 7, stride = 1, padding = 3 $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2).
  - **Conv1D Block 2:** 128 bộ lọc, kernel size = 5, stride = 1, padding = 2 $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2).
  - **Conv1D Block 3:** 256 bộ lọc, kernel size = 3, stride = 1, padding = 1 $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2).
- **Thẻ Card 2 [Phần Đầu Phân loại & Regularization]:**
  - Flatten $\to$ Linear(4096 $\to$ 256) $\to$ ReLU $\to$ **Dropout(0.5)** $\to$ Linear(256 $\to$ 8 classes).
  - Tổng số tham số: **1,178,632 tham số**. Kích thước model checkpoint: **~4.7 MB**.
- **Thẻ Card 3 [Chiến lược Huấn luyện Bền vững]:**
  - Optimizer: `AdamW` ($lr = 10^{-3}$, weight decay = $10^{-4}$).
  - Scheduler: `ReduceLROnPlateau(patience=3, factor=0.5)`.
  - Early Stopping: Giám sát Validation Loss với `patience = 7`, lưu lại `best_cnn1d_model.pt`.

---

## SLIDE 8: SO SÁNH ĐỘ PHỨC TẠP TÍNH TOÁN & ĐỘ TRỄ SUY LUẬN
- **Thời lượng:** 01:00 | **Người nói:** Thành viên 2
- **Tiêu đề Slide:** Inference Latency, Memory Footprint & Hardware Feasibility
- **Thẻ Card 1 [Bảng So sánh Benchmark Thực tế trên CPU]:**

| Tiêu chí | Feature Extraction + XGBoost | Feature Extraction + RBF-SVM | 1D-CNN (PyTorch CPU) |
| :--- | :---: | :---: | :---: |
| **Kích thước Mô hình** | **0.8 MB** (Siêu nhẹ) | **8.4 MB** | **4.7 MB** |
| **Trích xuất Đặc trưng (ms/burst)** | 0.85 ms | 0.85 ms | **0.00 ms** (Không cần) |
| **Suy luận Model (ms/burst)** | **0.08 ms** | 1.15 ms | 0.42 ms |
| **Tổng Độ trễ (Total Latency)** | **0.93 ms / burst** | 2.00 ms / burst | **0.42 ms / burst** |
| **Throughput (Bursts / sec)** | 1,075 bursts/s | 500 bursts/s | **2,380 bursts/s** |
| **Yêu cầu Phần cứng** | CPU nhúng nhỏ (ARM Cortex-M/A) | CPU chuẩn | NPU / DSP / GPU nhúng |

- **Thẻ Card 2 [Kết luận Kiến trúc]:**
  - 1D-CNN đạt throughput cao hơn khi xử lý theo batch trên tensor thô vì bỏ qua hoàn toàn chi phí tính toán vòng lặp Python cho trích xuất FFT và Cumulant.

---

## SLIDE 9: KẾT QUẢ THỰC NGHIỆM: ĐƯỜNG CONG SNR VS ACCURACY / MACRO-F1
- **Thời lượng:** 01:30 | **Người nói:** Thành viên 3
- **Tiêu đề Slide:** Experimental Results: SNR Benchmark Curves (n = 7,200 Test Bursts)
- **Thẻ Card 1 [Hiệu năng Vùng SNR Thấp ($-10$ dB đến $0$ dB)]:**
  - Tại $-4$ dB: **1D-CNN đạt 48.75%**, vượt trội hoàn toàn **XGBoost (31.67%)** và **RBF-SVM (32.29%)** với mức chênh lệch **+17.08%**.
  - Tại $0$ dB: 1D-CNN đạt **74.58%**, trong khi XGBoost chỉ đạt **46.67%** (+27.91%).
- **Thẻ Card 2 [Hiệu năng Vùng SNR Trung bình & Cao ($+4$ dB đến $+18$ dB)]:**
  - Từ $+6$ dB trở lên, cả 3 mô hình đều bứt phá mạnh mẽ vượt ngưỡng 80%.
  - Tại SNR $\ge +12$ dB: Cả XGBoost và 1D-CNN hội tụ ở mức **90% – 93%**, tiệm cận giới hạn lý thuyết phân loại cho độ dài khung ngắn $L=128$.
- **Thẻ Card 3 [Chỉ số Macro-F1]:**
  - Đường Macro-F1 bám sát Accuracy, chứng minh không có hiện tượng mô hình thiên vị (bias) vào các lớp dễ phân loại như BPSK.

---

## SLIDE 10: MA TRẬN NHẦM LẪN (CONFUSION MATRIX) QUA CÁC CHẾ ĐỘ NHIỄU
- **Thời lượng:** 01:30 | **Người nói:** Thành viên 3
- **Tiêu đề Slide:** Confusion Matrix Analysis Across Low, Medium & High SNR
- **Thẻ Card 1 [Chế độ Nhiễu Nặng (Low SNR: $-4$ dB)]:**
  - XGBoost bị nhầm lẫn lan tràn giữa các lớp (đặc biệt các cụm QPSK, 16QAM, 64QAM do Cumulant co cụm).
  - 1D-CNN vẫn phân tách cực tốt BPSK (88%), PAM4 (82%) và nhận diện đúng một phần FSK nhờ tương quan pha thời gian.
- **Thẻ Card 2 [Chế độ Trung bình (Medium SNR: $+4$ dB)]:**
  - BPSK, PAM4, CPFSK, GFSK đạt độ chính xác gần như tuyệt đối (> 95%).
  - Nhầm lẫn chỉ còn khu trú trong nội bộ họ điều chế biên độ/pha: QPSK $\leftrightarrow$ 8PSK $\leftrightarrow$ 16QAM.
- **Thẻ Card 3 [Chế độ Không Nhiễu (High SNR: $+14$ dB)]:**
  - BPSK, QPSK, 8PSK, PAM4, CPFSK, GFSK đạt **100.0%**.
  - **Điểm nghẽn duy nhất còn lại:** Sự nhầm lẫn giữa **16QAM $\leftrightarrow$ 64QAM** (XGBoost nhầm 34.0%, CNN nhầm 34.7%).

---

## SLIDE 11: PHÂN TÍCH LỖI GỐC RỄ 1: SỰ SUY SỤP CỦA CUMULANTS Ở SNR THẤP
- **Thời lượng:** 01:30 | **Người nói:** Thành viên 3
- **Tiêu đề Slide:** Root-Cause Failure 1: The Low-SNR Collapse of Higher-Order Cumulants
- **Thẻ Card 1 [Cơ chế Toán học A: Contraction Factor (Co cụm về 0)]:**
  - Tín hiệu thu: $r = s + w$ với $s \perp w$, nhiễu Gauss có $C_{42}(w) = 0$.
  - Sau chuẩn hóa công suất $E[|r|^2] = E[|s|^2] + \sigma_w^2$:
    $$C_{42, \text{observed}} = C_{42, \text{signal}} \cdot \left( \frac{\text{SNR}}{1 + \text{SNR}} \right)^2$$
  - Tại $\text{SNR} = -4\text{ dB} \implies \rho = 10^{-0.4} \approx 0.398 \implies \left(\frac{0.398}{1.398}\right)^2 \approx \mathbf{0.081}$.
  - Giá trị Cumulant bị co rút **hơn 12 lần**, khiến mọi chòm sao bị ép sát về gốc tọa độ đặc trưng.
- **Thẻ Card 2 [Cơ chế Toán học B: Bùng nổ Phương sai Mẫu hữu hạn ($L=128$)]:**
  - Ước lượng mẫu: $\hat{M}_{42} = \frac{1}{L} \sum_{n=1}^{L} |r[n]|^4$.
  - Phương sai $\text{Var}\{\hat{C}_{42}\} \propto \frac{E[|w|^8] - (E[|w|^4])^2}{L}$.
  - Độ lệch chuẩn $\sigma_{\hat{C}_{42}}$ **không hề giảm theo SNR** mà bị thống trị bởi moment bậc 8 của nhiễu.
- **Thẻ Card 3 [Số liệu Đo đạc Thực tế trên Test Set (Chỉ số "Feature SNR" $\frac{|\mu|}{\sigma}$)]:**
  - **QPSK:** Tại $+14$ dB có $|\mu|/\sigma = 14.51$ (rất rõ) $\to$ tại $-4$ dB tụt còn **$0.64$** ($|\mu| < \sigma$).
  - **16QAM:** Tại $+14$ dB có $|\mu|/\sigma = 5.24$ $\to$ tại $-4$ dB tụt còn **$0.38$**.
  - **Hệ quả Vật lý:** Khi $\frac{|\mu|}{\sigma} < 1$, đặc trưng chòm sao bị chìm nghỉm dưới phương sai ước lượng, phân loại dựa trên đặc trưng sụp đổ thành đoán mò.

---

## SLIDE 12: PHÂN TÍCH LỖI GỐC RỄ 2: TẠI SAO 1D-CNN THẮNG THẾ Ở SNR THẤP?
- **Thời lượng:** 01:15 | **Người nói:** Thành viên 3
- **Tiêu đề Slide:** Root-Cause Failure 2: Why 1D-CNN Outperforms at Low SNR (+17.08%)
- **Thẻ Card 1 [Nguyên lý Tích chập Thời gian = Lọc Phối hợp Ngầm định]:**
  - Tầng Conv1D trượt các kernel $W \in \mathbb{R}^{2 \times K}$ qua 128 mẫu I/Q:
    $$y_k[n] = \sum_{c \in \{I,Q\}} \sum_{\tau=0}^{K-1} W_k[c, \tau] \cdot r_c[n - \tau]$$
  - Quá trình Backpropagation tối ưu các trọng số kernel $W_k$ hội tụ thành các **xung tương quan thời gian** (matched filter / correlator) bám theo dạng xung RRC và nhịp chuyển tiếp ký hiệu.
- **Thẻ Card 2 [Gia tăng Tỷ số Tín hiệu trên Nhiễu Hiệu dụng (Processing Gain)]:**
  - Phép lọc phối hợp tích lũy năng lượng tín hiệu có tương quan kết hợp (coherent integration), trong khi nhiễu trắng bị triệt tiêu không kết hợp:
    $$\text{SNR}_{\text{effective}} \approx \text{SNR}_{\text{in}} + 10 \log_{10}(K \cdot \text{SPS})$$
  - Nhờ đó, 1D-CNN nâng mức SNR thực tế trước khi đưa vào các tầng kích hoạt phi tuyến.
- **Thẻ Card 3 [So sánh Bản chất Toán học]:**
  - **Feature-Based:** Nâng lũy thừa bậc 4 tức thời trên từng mẫu ($|r[n]|^4$) $\to$ Khuếch đại phi tuyến phương sai nhiễu.
  - **1D-CNN:** Tích hợp tuyến tính trước qua cửa sổ thời gian (Temporal Smoothing) $\to$ Khử nhiễu trước khi tạo đặc trưng phi tuyến.

---

## SLIDE 13: PHÂN TÍCH LỖI GỐC RỄ 3: SỰ BẾ TẮC KINH ĐIỂN 16QAM VS 64QAM
- **Thời lượng:** 01:00 | **Người nói:** Thành viên 3
- **Tiêu đề Slide:** Root-Cause Failure 3: The Intrinsic 16QAM vs. 64QAM Ambiguity
- **Thẻ Card 1 [Khoảng cách Lý thuyết Quá hẹp]:**
  - $C_{42}(16\text{QAM}) = -0.680$ vs $C_{42}(64\text{QAM}) = -0.619 \implies \Delta C_{42} = \mathbf{0.061}$.
  - Trên tập Test thực tế ($L=128$, có xung RRC):
    - 16QAM: $-0.551 \pm 0.104$
    - 64QAM: $-0.497 \pm 0.103$
    - Độ chênh lệch thực tế $\Delta = 0.054$, trong khi $\sigma \approx 0.103$.
- **Thẻ Card 2 [Chỉ số Phân tách Fisher & Xác suất Lỗi Bayes]:**
  - $d' = \frac{\Delta \mu}{\sigma_{\text{pooled}}} = \frac{0.054}{0.103} \approx \mathbf{0.52}$.
  - Xác suất lỗi Bayes lý thuyết cho 1 chiều đặc trưng:
    $$P_e = Q(d'/2) = Q(0.26) \approx \mathbf{39.7\%}$$
  - Thực tế quan sát: XGBoost nhầm $34.0\%$, 1D-CNN nhầm $34.7\%$, hoàn toàn khớp với lý thuyết Bayes!
- **Thẻ Card 3 [4 Giải pháp Đột phá Khắc phục Triệt để]:**
  1. **Tăng chiều dài khung $L$:** Phương sai mẫu $\sigma \propto 1/\sqrt{L}$. Tăng $L = 512$ hoặc $1024$ sẽ thu hẹp $\sigma$ xuống một nửa, tách rời 2 phân phối.
  2. **Bổ sung Cumulant Bậc 6 ($C_{63}$):** Tỷ lệ $\frac{C_{63}}{C_{42}}$ có độ nhạy phân tách cao hơn với chòm sao QAM mật độ lớn.
  3. **Khôi phục Định thời (Timing Recovery):** Lấy mẫu đúng đỉnh mắt (Symbol-rate sampling) để loại bỏ nhiễu ISI chuyển tiếp xung.
  4. **Kiến trúc Mạng Sâu hơn:** Sử dụng ResNet-1D hoặc Complex-Valued CNN (CV-CNN).

---

## SLIDE 14: KẾT LUẬN & TRÌNH DIỄN LIVE DEMO STREAMLIT
- **Thời lượng:** 01:15 | **Người nói:** Thành viên 3
- **Tiêu đề Slide:** Conclusions, Key Takeaways & Live Interactive Demo
- **Thẻ Card 1 [Tóm tắt Đóng góp & Kết luận Kỹ thuật]:**
  - **Ưu tiên 1D-CNN** cho các hệ thống vô tuyến hoạt động ở môi trường nhiễu nặng (SNR $< 0$ dB) nhờ khả năng lọc tương quan thời gian vượt trội (+17.08% tại $-4$ dB).
  - **Ưu tiên Feature + XGBoost** khi triển khai trên phần cứng IoT/nhúng cực hạn cần suy luận siêu nhẹ (0.08 ms, 0.8 MB model) ở điều kiện SNR từ trung bình đến cao.
- **Thẻ Card 2 [Thực chứng Live Interactive Demo]:**
  - Giao diện Streamlit tương tác thời gian thực:
    - Kéo thanh trượt SNR từ $+18$ dB xuống $-4$ dB.
    - Quan sát chòm sao biến dạng, phổ công suất FFT bị san phẳng bởi sàn nhiễu.
    - Kiểm chứng trực tiếp tại $-4$ dB: XGBoost suy đoán sai sang class khác, trong khi 1D-CNN duy trì dự đoán chính xác với thanh confidence score rõ rệt.

---

# BỘ CÂU HỎI & TRẢ LỜI MẪU CHO PHẦN Q&A (5 PHÚT BẢO VỆ 20/20 ĐIỂM)

### Câu hỏi 1 (Về Xử lý Tín hiệu Baseband):
> **Giảng viên hỏi:** *"Tại sao các em lại chọn chuẩn hóa năng lượng RMS trên từng khung (Per-burst RMS Normalization) mà không chuẩn hóa max biên độ hay để nguyên biên độ thu?"*
- **Trả lời chuẩn Kỹ sư RF:**  
  "Dạ thưa Thầy/Cô, trong kênh truyền vô tuyến thực tế, công suất tín hiệu thu bị biến động liên tục bởi cự ly truyền dẫn (Path Loss) và suy hao đa đường (Large-scale fading). Nếu không chuẩn hóa, các mô hình học máy sẽ 'học vẹt' công suất thu thay vì hình học điều chế.  
  Chúng em chọn **RMS Normalization** ($\tilde{r} = r / \sqrt{\frac{1}{L}\sum |r|^2}$) thay vì Peak Normalization vì Peak Normalization cực kỳ nhạy cảm với các mẫu nhiễu ngoại lai (noise spikes) ở SNR thấp. RMS Normalization đưa công suất trung bình về đúng $1.0$, bảo toàn tỷ lệ năng lượng lý thuyết giữa các mức chòm sao và đảm bảo các công thức Cumulants ($C_{21}=1, C_{42}$) giữ nguyên tính bất biến toán học."

---

### Câu hỏi 2 (Về Hiện tượng Suy sụp Cumulant):
> **Giảng viên hỏi:** *"Về mặt toán học, nhiễu Gauss độc lập có Cumulant bậc 4 bằng 0 ($C_{42}(w) = 0$). Vậy tại sao ở SNR thấp Cumulant của tín hiệu thu lại không bằng Cumulant của tín hiệu gốc mà bị suy giảm nghiêm trọng?"*
- **Trả lời chuẩn Kỹ sư RF:**  
  "Dạ thưa Thầy/Cô, đúng là $C_{42}(s+w) = C_{42}(s) + C_{42}(w) = C_{42}(s)$. Tuy nhiên, đó là khi tín hiệu **chưa chuẩn hóa**.  
  Khi ta chuẩn hóa năng lượng của tín hiệu thu $r = s+w$ có công suất $P_r = P_s + \sigma_w^2$, tín hiệu chuẩn hóa trở thành $\tilde{r} = \frac{s+w}{\sqrt{P_s + \sigma_w^2}}$.  
  Do tính chất thuần nhất của Cumulant bậc 4: $\text{Cum}(\alpha r) = \alpha^4 \text{Cum}(r)$, nên:
  $$C_{42}(\tilde{r}) = \frac{C_{42}(s)}{(P_s + \sigma_w^2)^2} = C_{42}(s) \cdot \left(\frac{\text{SNR}}{1+\text{SNR}}\right)^2$$
  Ở $-4$ dB ($\text{SNR}=0.398$), hệ số co rút này là $(0.398/1.398)^2 \approx 0.081$, khiến giá trị trung bình của Cumulant bị co rút tới $92\%$ về $0$. Đồng thời, phương sai ước lượng với mẫu hữu hạn $L=128$ lại chứa moment bậc 8 của nhiễu Gauss ($E[|w|^8]$), khiến độ lệch chuẩn giữ nguyên trong khi kỳ vọng co về 0, làm cho tỷ số Feature SNR rơi xuống dưới $1.0$."

---

### Câu hỏi 3 (Về Học sâu 1D-CNN):
> **Giảng viên hỏi:** *"Mạng 1D-CNN của nhóm nhận đầu vào là tensor $[2, 128]$. Làm thế nào một mạng CNN thuần 1D thực lại hiểu được quan hệ góc pha phức giữa kênh I và kênh Q?"*
- **Trả lời chuẩn Kỹ sư RF:**  
  "Dạ thưa Thầy/Cô, ở tầng tích chập đầu tiên `Conv1d(in_channels=2, out_channels=64, kernel_size=7)`, mỗi bộ lọc có ma trận trọng số kích thước $2 \times 7$. Khi tích chập, kernel tính tổng chéo:
  $$y[n] = \sum_{\tau=0}^{6} \left( W_I[\tau] \cdot I[n-\tau] + W_Q[\tau] \cdot Q[n-\tau] \right)$$
  Cấu trúc này tương đương về mặt đại số với phần thực của phép nhân tích chập phức: $\text{Re}\{ (W_I - j W_Q) * (I + j Q) \}$.  
  Các bộ lọc này đồng thời thực hiện tương quan chéo giữa I và Q qua các thời điểm trễ $\tau$, từ đó học được tốc độ quay pha $\Delta \theta = \omega \tau$ và phân biệt được chiều quay của vector chòm sao mà không cần tính hàm $\arctan$ tường minh."

---

### Câu hỏi 4 (Về Điểm nghẽn 16QAM vs 64QAM):
> **Giảng viên hỏi:** *"Tại sao cả mô hình Feature Engineering lẫn Deep Learning đều gặp khó khăn lớn nhất khi phân biệt 16QAM và 64QAM, và nếu được nâng cấp hệ thống trên thực tế, nhóm sẽ làm gì đầu tiên?"*
- **Trả lời chuẩn Kỹ sư RF:**  
  "Dạ thưa Thầy/Cô:
  1. **Về hình học:** Cả 16QAM và 64QAM đều có chung tính đối xứng quay $90^\circ$ (bậc 4), khiến $M_{20}=0$ và $|C_{40}| = C_{42}$. Khoảng cách lý thuyết giữa chúng chỉ là $|-0.680 - (-0.619)| = 0.061$. Với độ dài khung cực ngắn $L=128$ mẫu (tương đương 32 ký hiệu ở SPS=4), một số điểm chòm sao ngoài cùng của 64QAM thậm chí chưa kịp xuất hiện trong 32 ký hiệu phát ngẫu nhiên, khiến phân bố biên độ thực tế của 64QAM bị trùng lặp nặng nề với 16QAM.
  2. **Giải pháp nâng cấp thực tế:** Giải pháp hiệu quả và rẻ nhất về mặt tính toán là **tăng độ dài khung quan sát lên $L=512$ hoặc $1024$ mẫu**. Khi đó, phương sai ước lượng $\sigma \propto 1/\sqrt{L}$ sẽ giảm đi 2 đến 3 lần, tách rời hai phân phối Gauss của 16QAM và 64QAM, đưa độ chính xác vượt trên $95\%$."

---

### Câu hỏi 5 (Về Khả năng Triển khai Thực tế - Edge Deployment):
> **Giảng viên hỏi:** *"Nếu nhóm phải triển khai giải pháp này lên một thiết bị IoT chạy pin sử dụng chip vi điều khiển ARM Cortex-M4 không có GPU, nhóm sẽ chọn giải pháp nào?"*
- **Trả lời chuẩn Kỹ sư RF:**  
  "Dạ thưa Thầy/Cô, chúng em chắc chắn sẽ chọn **Feature Engineering kết hợp với mô hình cây quyết định (như XGBoost rút gọn hoặc Decision Tree)**.  
  Lý do là:
  - Mạng 1D-CNN đòi hỏi hơn 1.17 triệu phép tính nhân cộng (MACs) và chiếm gần 5 MB bộ nhớ Flash, vượt quá dung lượng SRAM/Flash thông thường của Cortex-M4 (thường chỉ 64KB - 256KB RAM).
  - Ngược lại, trích xuất 10 đặc trưng chỉ tốn khoảng vài nghìn phép tính số học cơ bản. Sau khi trích xuất, mô hình cây chỉ thực hiện các phép so sánh ngưỡng `if-else` trên số thực, chiếm chưa tới $50$ KB bộ nhớ và suy luận trong dưới $1$ ms mà không cần thư viện Tensor hay bộ tăng tốc phần cứng."
