import os
import sys
import time
import argparse
import cv2
import numpy as np

from cpr_core.pose import PoseTracker
from cpr_core.calibration import SpatialCalibration
from cpr_core.kinematics import CPRKinematics
from cpr_core.metrics import CPRMetrics

def draw_hud(frame, res, metrics_summary, wave_history, n_cycles):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 260), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    depth_val = res["displacement"]
    rate_val = res["smooth_rate"]
    theta = res["theta"]
    beta = res["beta"]
    elbow_ok = res["elbow_valid"]
    vert_ok = res["vert_valid"]

    depth_ok = 5.0 <= depth_val <= 6.0
    rate_ok = 100.0 <= rate_val <= 120.0

    c_depth = (0, 255, 0) if depth_ok else (0, 0, 255)
    c_rate = (0, 255, 0) if rate_ok else (0, 0, 255)
    c_elbow = (0, 255, 0) if elbow_ok else (0, 0, 255)
    c_vert = (0, 255, 0) if vert_ok else (0, 0, 255)

    cv2.putText(frame, "CPR REAL-TIME FEEDBACK", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Cycles Count: {n_cycles}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"Depth:  {depth_val:.2f} cm (5-6 cm)", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_depth, 2)
    cv2.putText(frame, f"Rate:   {rate_val:.1f} cpm (100-120)", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_rate, 2)
    cv2.putText(frame, f"Elbow:  {theta:.1f} deg (180+/-15)", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_elbow, 1)
    cv2.putText(frame, f"Vert:   {beta:.1f} deg (90+/-10)", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_vert, 1)
    
    cqi = metrics_summary["c_cqi"]
    cert = metrics_summary["certified"]
    c_cert = (0, 255, 0) if cert else (0, 140, 255)
    cert_txt = "STATUS: CERTIFIED" if cert else "STATUS: TRAINING"

    cv2.putText(frame, f"C-CQI Score: {cqi:.1f} / 100", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, cert_txt, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_cert, 2)

    wave_h, wave_w = 90, 280
    wave_x, wave_y = 20, h - 110
    cv2.rectangle(frame, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (30, 30, 30), -1)
    cv2.rectangle(frame, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (80, 80, 80), 1)
    
    t_target_top = wave_y + int(wave_h * (1.0 - 6.0 / 8.0))
    t_target_bot = wave_y + int(wave_h * (1.0 - 5.0 / 8.0))
    cv2.rectangle(frame, (wave_x, t_target_top), (wave_x + wave_w, t_target_bot), (0, 60, 0), -1)
    cv2.line(frame, (wave_x, t_target_top), (wave_x + wave_w, t_target_top), (0, 180, 0), 1)
    cv2.line(frame, (wave_x, t_target_bot), (wave_x + wave_w, t_target_bot), (0, 180, 0), 1)

    if len(wave_history) > 1:
        points = []
        for i, val in enumerate(wave_history[-wave_w:]):
            px = wave_x + i
            norm_val = np.clip(val / 8.0, 0.0, 1.0)
            py = wave_y + int(wave_h * (1.0 - norm_val))
            points.append((px, py))
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2)

def generate_mock_frame(step, w=640, h=480):
    img = np.ones((h, w, 3), dtype=np.uint8) * 45
    t = step * 0.05
    dy = int(np.sin(t * 3.5) * 45 + 50)
    
    cv2.rectangle(img, (200, 380), (440, 440), (120, 120, 120), -1)
    cv2.putText(img, "MANIKIN SIMULATION", (220, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)
    
    head = (320, 120 + dy // 3)
    shoulder = (320, 180 + dy // 2)
    elbow = (320, 260 + dy * 2 // 3)
    wrist = (320, 360 + dy)

    cv2.circle(img, head, 30, (200, 200, 200), -1)
    cv2.line(img, head, shoulder, (0, 220, 255), 4)
    cv2.line(img, shoulder, elbow, (0, 220, 255), 4)
    cv2.line(img, elbow, wrist, (0, 220, 255), 4)
    for pt in [head, shoulder, elbow, wrist]:
        cv2.circle(img, pt, 6, (0, 0, 255), -1)

    return img, {
        "shoulder": np.array(shoulder, dtype=float),
        "elbow": np.array(elbow, dtype=float),
        "wrist": np.array(wrist, dtype=float)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="Video path or camera index")
    parser.add_argument("--weights", type=str, default="", help="Path to yolo pose weights")
    parser.add_argument("--save-video", type=str, default="", help="Path to save output video")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to process")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_weights = os.path.join(base_dir, "weights", "yolo26s-pose.pt")
    weights_path = args.weights if args.weights else (default_weights if os.path.exists(default_weights) else None)

    tracker = PoseTracker(weights_path=weights_path)
    calib = SpatialCalibration(factor=0.06)
    kinematics = CPRKinematics(gamma=calib.gamma)

    cap = None
    use_mock = True
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if cap.isOpened():
            use_mock = False
    elif os.path.isfile(args.source):
        cap = cv2.VideoCapture(args.source)
        if cap.isOpened():
            use_mock = False

    writer = None
    wave_history = []
    frame_idx = 0
    t0 = time.time()

    print(f"Starting CPR Vision Assessment (Backend: {tracker.backend}, Mock: {use_mock})")

    while frame_idx < args.max_frames:
        if not use_mock and cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
            kpts = tracker.predict(frame)
        else:
            frame, kpts = generate_mock_frame(frame_idx)

        h, w = frame.shape[:2]
        if writer is None and args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (w, h))

        if kinematics.baseline_y is None:
            if kpts is not None and "wrist" in kpts:
                kinematics.set_baseline(kpts["wrist"][1])

        current_time = frame_idx * (1.0 / 30.0)
        res = kinematics.update(kpts, current_time)

        if res is not None:
            wave_history.append(res["displacement"])
            scores = CPRMetrics.compute_scores(kinematics.cycles, kinematics.frame_postures)
            draw_hud(frame, res, scores, wave_history, len(kinematics.cycles))

        if writer is not None:
            writer.write(frame)

        if not args.headless:
            cv2.imshow("Monocular CPR Quality Assessment", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    final_scores = CPRMetrics.compute_scores(kinematics.cycles, kinematics.frame_postures)
    print("-" * 50)
    print("Execution Summary:")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Cycles Tracked:   {len(kinematics.cycles)}")
    print(f"Final C-CQI Score:      {final_scores['c_cqi']:.2f} / 100")
    print(f"Certification Outcome:  {'PASSED' if final_scores['certified'] else 'UNSATISFACTORY'}")

if __name__ == "__main__":
    main()
