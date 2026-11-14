import numpy as np

class PoseTracker:
    def __init__(self, weights_path=None, conf_thresh=0.25, iou_thresh=0.70, device=None):
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.device = device
        self.model = None
        self.backend = "yolo"
        if weights_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights_path)
            except Exception:
                self.model = None
        if self.model is None:
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.backend = "mediapipe"
            except Exception:
                self.backend = "none"

    def predict(self, frame):
        h, w = frame.shape[:2]
        if self.backend == "yolo" and self.model is not None:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=640,
                verbose=False,
                device=self.device
            )
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                ls, rs = kpts[5], kpts[6]
                le, re = kpts[7], kpts[8]
                lw, rw = kpts[9], kpts[10]
                shoulder = (ls + rs) / 2.0 if (ls[0] > 0 and rs[0] > 0) else (ls if ls[0] > 0 else rs)
                elbow = (le + re) / 2.0 if (le[0] > 0 and re[0] > 0) else (le if le[0] > 0 else re)
                wrist = (lw + rw) / 2.0 if (lw[0] > 0 and rw[0] > 0) else (lw if lw[0] > 0 else rw)
                return {
                    "shoulder": shoulder,
                    "elbow": elbow,
                    "wrist": wrist,
                    "left_shoulder": ls,
                    "right_shoulder": rs,
                    "left_elbow": le,
                    "right_elbow": re,
                    "left_wrist": lw,
                    "right_wrist": rw
                }
        elif self.backend == "mediapipe":
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                ls = np.array([lm[11].x * w, lm[11].y * h])
                rs = np.array([lm[12].x * w, lm[12].y * h])
                le = np.array([lm[13].x * w, lm[13].y * h])
                re = np.array([lm[14].x * w, lm[14].y * h])
                lw = np.array([lm[15].x * w, lm[15].y * h])
                rw = np.array([lm[16].x * w, lm[16].y * h])
                shoulder = (ls + rs) / 2.0
                elbow = (le + re) / 2.0
                wrist = (lw + rw) / 2.0
                return {
                    "shoulder": shoulder,
                    "elbow": elbow,
                    "wrist": wrist,
                    "left_shoulder": ls,
                    "right_shoulder": rs,
                    "left_elbow": le,
                    "right_elbow": re,
                    "left_wrist": lw,
                    "right_wrist": rw
                }
        return None
