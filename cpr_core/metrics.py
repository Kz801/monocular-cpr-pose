import numpy as np

class CPRMetrics:
    W_DEPTH = 0.40
    W_RATE = 0.30
    W_RECOIL = 0.10
    W_ELBOW = 0.10
    W_VERT = 0.10

    @staticmethod
    def compute_scores(cycles, frame_postures):
        n_c = len(cycles)
        n_f = len(frame_postures)
        s_depth = 100.0 * float(np.mean([c["in_depth_target"] for c in cycles])) if n_c > 0 else 0.0
        s_rate = 100.0 * float(np.mean([c["in_rate_target"] for c in cycles])) if n_c > 0 else 0.0
        s_recoil = 100.0 * float(np.mean([c["recoil_complete"] for c in cycles])) if n_c > 0 else 0.0
        s_elbow = 100.0 * float(np.mean([f["elbow_valid"] for f in frame_postures])) if n_f > 0 else 0.0
        s_vert = 100.0 * float(np.mean([f["vert_valid"] for f in frame_postures])) if n_f > 0 else 0.0
        
        c_cqi = (
            CPRMetrics.W_DEPTH * s_depth +
            CPRMetrics.W_RATE * s_rate +
            CPRMetrics.W_RECOIL * s_recoil +
            CPRMetrics.W_ELBOW * s_elbow +
            CPRMetrics.W_VERT * s_vert
        )
        
        posture_pass_elbow = 1 if (s_elbow >= 60.0) else 2
        posture_pass_vert = 1 if (s_vert >= 60.0) else 2
        
        certified = 1 if (c_cqi >= 75.0 and s_depth >= 60.0 and s_rate >= 60.0) else 0
        
        return {
            "s_depth": s_depth,
            "s_rate": s_rate,
            "s_recoil": s_recoil,
            "s_elbow": s_elbow,
            "s_vert": s_vert,
            "c_cqi": c_cqi,
            "certified": certified,
            "posture_elbow_class": posture_pass_elbow,
            "posture_vert_class": posture_pass_vert
        }

    @staticmethod
    def compute_subindex(depth_compliance, recoil_compliance):
        w_d, w_r = 0.80, 0.20
        subindex = 100.0 * (w_d * depth_compliance + w_r * recoil_compliance)
        passed = 1 if (subindex >= 75.0 and depth_compliance >= 0.60) else 0
        return subindex, passed
