import numpy as np

class CPRKinematics:
    def __init__(self, gamma=None, state_threshold_cm=2.0, recoil_threshold_cm=0.5):
        self.gamma = gamma
        self.state_threshold = state_threshold_cm
        self.recoil_threshold = recoil_threshold_cm
        self.baseline_y = None
        self.state = "idle"
        self.current_max_depth = 0.0
        self.last_compression_time = None
        self.smooth_rate = 110.0
        self.cycles = []
        self.frame_postures = []

    def set_baseline(self, y0):
        self.baseline_y = float(y0)

    def calculate_elbow_angle(self, shoulder, elbow, wrist):
        v_es = shoulder - elbow
        v_ew = wrist - elbow
        norm_es = np.linalg.norm(v_es)
        norm_ew = np.linalg.norm(v_ew)
        if norm_es == 0 or norm_ew == 0:
            return 180.0
        cosine = np.dot(v_es, v_ew) / (norm_es * norm_ew)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def calculate_verticality_angle(self, shoulder, wrist):
        dx = abs(shoulder[0] - wrist[0])
        dy = abs(shoulder[1] - wrist[1])
        if dx == 0:
            return 90.0
        return float(np.degrees(np.arctan(dy / dx)))

    def update(self, keypoints, timestamp):
        if keypoints is None or "wrist" not in keypoints or self.baseline_y is None or self.gamma is None:
            return None
        wrist_y = keypoints["wrist"][1]
        displacement = max(0.0, (wrist_y - self.baseline_y) * self.gamma)
        
        shoulder = keypoints["shoulder"]
        elbow = keypoints["elbow"]
        wrist = keypoints["wrist"]
        
        theta = self.calculate_elbow_angle(shoulder, elbow, wrist)
        beta = self.calculate_verticality_angle(shoulder, wrist)
        
        elbow_valid = abs(180.0 - theta) <= 15.0
        vert_valid = 80.0 <= beta <= 100.0
        self.frame_postures.append({
            "elbow_valid": elbow_valid,
            "vert_valid": vert_valid,
            "theta": theta,
            "beta": beta
        })
        
        new_cycle = None
        if self.state == "idle":
            if displacement >= self.state_threshold:
                self.state = "compressing"
                self.current_max_depth = displacement
        elif self.state == "compressing":
            if displacement > self.current_max_depth:
                self.current_max_depth = displacement
            elif displacement < self.current_max_depth - 0.5:
                self.state = "releasing"
        elif self.state == "releasing":
            if displacement <= self.state_threshold:
                recoil_complete = 1 if displacement < self.recoil_threshold else 0
                rate = self.smooth_rate
                if self.last_compression_time is not None:
                    dt = timestamp - self.last_compression_time
                    if dt > 0.2:
                        raw_rate = 60.0 / dt
                        self.smooth_rate = 0.1 * self.smooth_rate + 0.9 * raw_rate
                        rate = self.smooth_rate
                self.last_compression_time = timestamp
                
                cycle_data = {
                    "cycle_id": len(self.cycles) + 1,
                    "depth": self.current_max_depth,
                    "rate": rate,
                    "recoil_complete": recoil_complete,
                    "in_depth_target": 1 if 5.0 <= self.current_max_depth <= 6.0 else 0,
                    "in_rate_target": 1 if 100.0 <= rate <= 120.0 else 0
                }
                self.cycles.append(cycle_data)
                new_cycle = cycle_data
                self.current_max_depth = 0.0
                self.state = "idle"
        
        return {
            "displacement": displacement,
            "state": self.state,
            "smooth_rate": self.smooth_rate,
            "theta": theta,
            "beta": beta,
            "elbow_valid": elbow_valid,
            "vert_valid": vert_valid,
            "new_cycle": new_cycle
        }
