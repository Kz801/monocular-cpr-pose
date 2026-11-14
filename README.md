# CPR-Vision-Assessment

Monocular Vision-Based Real-Time CPR Training System with Deep Learning Pose Assessment Frameworks.

This repository provides an open-source implementation of a low-cost, sensor-free cardiopulmonary resuscitation (CPR) skill assessment pipeline. Using an ordinary monocular camera, the system tracks upper-body skeletal landmarks in real time to evaluate compression depth, rate, chest recoil completeness, and rescuer posture.

---

## Key Features

1. **Monocular Real-Time Pose Tracking**:
   - Primary: YOLOv26s-pose (single-stage coordinate regression, 640x640 resolution).
   - Fallback: MediaPipe BlazePose for lightweight CPU execution.
   - Continuous landmark tracking: bilateral shoulders, elbows, wrists, and hands.

2. **Dual-Modal Spatial Calibration**:
   - **Manual Ruler Calibration**: Pixel-to-cm conversion factor $\gamma = H_0 / |y_1 - y_2|$ using a standard 16 cm metric ruler or smartphone reference.
   - **Automated Segmentation Calibration**: Automated manikin silhouette bounding extraction via YOLO-seg ($\gamma = H / h$, with manikin height $H = 160$ cm).

3. **Biomechanical & Clinical Assessment Metrics**:
   - **Compression Depth**: Clinical target of 5.0–6.0 cm.
   - **Compression Rate**: Target 100–120 compressions/min, stabilized via exponential moving average:
     $$R_{\text{smooth}, n} = 0.1 \cdot R_{\text{smooth}, n-1} + 0.9 \cdot R_n$$
   - **Chest Recoil Completeness**: Residual displacement $< 0.5$ cm at stroke top.
   - **Elbow Extension**: Forearm-to-upper-arm angle $\theta \in [165^\circ, 195^\circ]$ ($180^\circ \pm 15^\circ$).
   - **Upper-Limb Verticality**: Angle $\beta \in [80^\circ, 100^\circ]$ ($90^\circ \pm 10^\circ$).
   - **Posture Compliance**: Sequence classified as compliant if $\ge 60\%$ of valid frames meet tolerance bounds.

4. **Composite CPR Quality Index (C-CQI) & Dual-Criterion Certification**:
   - Multi-domain composite score:
     $$\text{C-CQI} = 0.40 S_{\text{depth}} + 0.30 S_{\text{rate}} + 0.10 S_{\text{recoil}} + 0.10 S_{\text{elbow}} + 0.10 S_{\text{vert}}$$
   - Dual-criterion pass rule:
     $$\text{Certification} = (\text{C-CQI} \ge 75) \land (S_{\text{depth}} \ge 60) \land (S_{\text{rate}} \ge 60)$$

5. **Repeated-Measures Statistical Validation**:
   - Linear Mixed-Effects Models (LMM) with participant random intercepts.
   - Generalized Estimating Equations (GEE) with exchangeable working correlation.
   - Intraclass Correlation Coefficient (ICC), Bland-Altman agreement, Cohen's kappa, PABAK, and DeLong ROC test.

---

## Repository Structure

```
cpr-vision-assessment/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── weights/
│   └── yolo26s-pose.pt             # Pre-trained pose estimation weights
├── data/
│   ├── spring60_cycles.csv         # Calibrated coil-spring benchmark (n=448, 60 fps)
│   ├── manikin_cycles.csv          # Raw electronic manikin compression cycles (n=2256)
│   ├── manikin_cycles_qc.csv       # Quality-controlled manikin cycles (n=2118)
│   ├── trial_level_qc.csv          # Aggregated trial metrics (n=74 trials)
│   ├── student_scores.csv          # 15 student validation trials vs expert ratings
│   └── qc_audit.csv                # Quality control filtering audit log
├── figures/
│   ├── Fig11_Models.png            # Model comparison & coil-spring precision plot
│   ├── Fig11_Models.tiff           # High-resolution TIFF
│   ├── Fig12_C-CQI.png             # Five-domain compliance & C-CQI distribution plot
│   └── Fig12_C-CQI.tiff            # High-resolution TIFF
├── cpr_core/
│   ├── __init__.py                 # Core package exports
│   ├── pose.py                     # YOLO and MediaPipe pose estimation interface
│   ├── calibration.py              # Manual ruler and segmentation calibration
│   ├── kinematics.py               # Real-time state machine, depth, rate, recoil & angles
│   ├── metrics.py                  # C-CQI, domain scores and certification evaluation
│   └── stats.py                    # LMM, GEE, ICC(2,1), kappa, PABAK, DeLong ROC
├── run_inference.py                # Real-time video/webcam assessment demo
├── run_qc_filter.py                # Quality control filtering and dataset aggregation
├── run_statistical_analysis.py     # Full reproduction of manuscript statistical tables
├── run_make_figures.py             # Publication figure generation script
└── run_benchmark.py                # Hardware latency and computational complexity profiler
```

---

## Installation

1. Clone or download this repository:
   ```bash
   git clone https://github.com/your-username/CPR-Vision-Assessment.git
   cd CPR-Vision-Assessment
   ```

2. Create a clean Python environment (Python 3.10+ recommended):
   ```bash
   conda create -n cpr_vision python=3.10 -y
   conda activate cpr_vision
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quickstart

### 1. Real-Time Tracking & Assessment Demo

Run the end-to-end evaluation pipeline with real-time HUD overlays:

- **Built-in Simulation Demo** (runs without external video or camera):
  ```bash
  python run_inference.py --headless --max-frames 200
  ```

- **Live Webcam Execution**:
  ```bash
  python run_inference.py --source 0
  ```

- **Offline Video Analysis & Export**:
  ```bash
  python run_inference.py --source path/to/cpr_video.mp4 --save-video output_assessed.mp4
  ```

---

### 2. Quality Control & Dataset Cleaning

Reproduce the pre-specified QC filtering rules (removing physically implausible displacements $>10.0$ cm and unstable trials $<10$ cycles):

```bash
python run_qc_filter.py
```

Outputs:
- `data/qc_audit.csv`: Attrition audit table.
- `data/manikin_cycles_qc.csv`: 2,118 validated compression cycles.
- `data/trial_level_qc.csv`: 74 trial-level composite records.

---

### 3. Statistical Analysis & Manuscript Tables

Reproduce all inferential statistics, repeated-measures models, and agreement metrics:

```bash
python run_statistical_analysis.py
```

Results generated:
- **Phase 1 Spring Benchmark**:
  - Naive one-sample t-test: Mean = 6.275 cm (95% CI: 6.224–6.327), $t = 10.46, P < 0.001$.
  - LMM (participant random intercept): Adjusted Mean = 6.272 cm (95% CI: 6.077–6.467), $P = 0.006$.
  - GEE (exchangeable): Adjusted Mean = 6.272 cm (95% CI: 6.084–6.460), $P = 0.005$.
  - Clustering parameters: $\text{ICC} = 0.442$, $\text{Design Effect} = 13.75$, $N_{\text{eff}} = 32.6$ (vs naive 448).
- **Phase 2 Manikin Depth Classification**:
  - Pre-QC accuracy: 78.50% ($n=2256$).
  - Post-QC accuracy: 79.70% ($n=2118, \kappa = 0.646$).
- **Multilevel Regression Contrasts**:
  - Post-QC LMM: Insufficient depth $-1.18$ cm (SE 0.06), Excessive depth $+1.00$ cm (SE 0.05).
  - Residual SD reduction from 2.30 cm to 0.68 cm, with $\text{ICC} = 0.586$.
- **Chest Recoil Agreement**:
  - Agreement rate: 95.37%, $\text{PABAK} = 0.907$.
  - Cluster-robust GEE logit odds ratio: $\text{OR} = 87.21$ (95% CI: 28.60–265.92, $P < 0.001$).
- **Compression Rate & Posture**:
  - Rate exact matches: 54/75 (72.0%), MAE = 0.72 cpm, overall accuracy 97.28%.
  - Elbow extension agreement: 65/75 (86.7%).
  - Upper-limb verticality agreement: 67/75 (89.3%).
- **C-CQI & Student Certification**:
  - Scoring reliability: $\text{ICC}(2,1) = 0.940$ ($P < 0.001$).
  - Bland-Altman mean bias: $-1.47$ points (95% LoA: $-6.95$ to $+4.02$).
  - Certification concordance at 75 points: 86.7% (13/15, $\kappa = 0.667$, $100\%$ specificity, zero false positives).
  - DeLong ROC test: Multi-domain C-CQI ($\text{AUC} = 1.000$) significantly outperforming single-domain depth ($\text{AUC} = 0.722$): $Z = 2.123, P = 0.0337$.

---

### 4. Publication Figure Generation

Generate Figure 11 and Figure 12 in vector/high-resolution raster formats:

```bash
python run_make_figures.py
```

Saved to `figures/`:
- `Fig11_Models.png` & `Fig11_Models.tiff`: Multilevel vs naive depth contrasts and spring precision benchmark.
- `Fig12_C-CQI.png` & `Fig12_C-CQI.tiff`: Five-domain compliance and C-CQI distribution.

---

### 5. Computational Complexity Profiling

Benchmark model parameters, theoretical compute (GFLOPs), and hardware inference latency:

```bash
python run_benchmark.py
```

Reports:
- Weights file size on disk: 23.03 MB.
- Parameter count: 11.81 M (YOLOv26s-pose).
- Theoretical FLOPs: ~29.2 GFLOPs.
- GPU latency and CPU fallback throughput.

---

## Data Availability Statement

To respect participant privacy in compliance with institutional review board (IRB) ethics protocols, raw video recordings depicting human rescuers are withheld from public repositories. However, all extracted de-identified keypoint trajectories, cycle-level kinematic measurements, electronic sensor ground truth, expert examiner evaluations, and statistical datasets are fully provided in the `data/` folder to enable complete computational reproducibility.

---

## Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{zheng2026monocular,
  title={Monocular Vision-Based Real-Time CPR Training System with Deep Learning Pose Assessment Frameworks},
  author={Zheng, Wei and Zhang, Kai and Gan, Linling and Zhang, Xiaojuan and Zhou, Jing and Tang, Dandan and Qin, Qin and Tang, Hui and Hu, Ting and Xie, Can and Tan, Li and Li, Xinke},
  journal={PLOS ONE / PLOS Digital Health},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
