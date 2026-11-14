import os
import numpy as np
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    man_path = os.path.join(data_dir, "manikin_cycles.csv")
    man = pd.read_csv(man_path)

    implausible = man["vision_depth"] > 10.0
    trial_counts = man.groupby("trial_id").size()
    unstable_trials = trial_counts[trial_counts < 10].index.tolist()
    unstable = man["trial_id"].isin(unstable_trials)

    qc = man.loc[~implausible].copy()
    qc = qc.loc[~qc["trial_id"].isin(unstable_trials)].copy()

    audit = pd.DataFrame([
        {
            "rule": "Raw cycle-level manikin data",
            "cycles_flagged": 0,
            "unique_additional_removed": 0,
            "cycles_remaining": len(man)
        },
        {
            "rule": "Rule 1: vision depth > 10.0 cm",
            "cycles_flagged": int(implausible.sum()),
            "unique_additional_removed": int(implausible.sum()),
            "cycles_remaining": len(man) - int(implausible.sum())
        },
        {
            "rule": "Rule 2: trial with < 10 cycles",
            "cycles_flagged": int(unstable.sum()),
            "unique_additional_removed": int((unstable & ~implausible).sum()),
            "cycles_remaining": len(man) - int(implausible.sum()) - int((unstable & ~implausible).sum())
        },
        {
            "rule": "Final quality-controlled dataset",
            "cycles_flagged": len(man) - len(qc),
            "unique_additional_removed": len(man) - len(qc),
            "cycles_remaining": len(qc)
        }
    ])
    audit.to_csv(os.path.join(data_dir, "qc_audit.csv"), index=False)
    qc.to_csv(os.path.join(data_dir, "manikin_cycles_qc.csv"), index=False)

    w_d, w_r = 0.40, 0.10
    g = qc.groupby(["subject", "set", "trial_id"]).agg(
        n_cycles=("vision_depth", "size"),
        mean_depth=("vision_depth", "mean"),
        sd_depth=("vision_depth", "std"),
        depth_compliance=("in_range", "mean"),
        recoil_compliance=("recoil_complete", "mean"),
        sensor_depth_compliance=("sensor_class", lambda x: float((x == 2).mean())),
        sensor_recoil_compliance=("sensor_recoil_complete", "mean"),
        class_accuracy=("correct_class", "mean")
    ).reset_index()

    g["subindex_cv"] = 100.0 * (w_d * g["depth_compliance"] + w_r * g["recoil_compliance"]) / (w_d + w_r)
    g["subindex_sensor"] = 100.0 * (w_d * g["sensor_depth_compliance"] + w_r * g["sensor_recoil_compliance"]) / (w_d + w_r)
    g["pass_cv"] = ((g["subindex_cv"] >= 75.0) & (g["depth_compliance"] >= 0.60)).astype(int)
    g["pass_sensor"] = ((g["subindex_sensor"] >= 75.0) & (g["sensor_depth_compliance"] >= 0.60)).astype(int)
    g.to_csv(os.path.join(data_dir, "trial_level_qc.csv"), index=False)

    print("Quality Control Pipeline Complete")
    print("-" * 50)
    print(f"Raw cycles:        {len(man)}")
    print(f"Flagged (>10cm):   {int(implausible.sum())}")
    print(f"Post-QC cycles:    {len(qc)}")
    print(f"Post-QC trials:    {len(g)}")
    print(f"Trial Pass (CV):   {int(g['pass_cv'].sum())} / {len(g)}")
    print(f"Trial Pass (Ref):  {int(g['pass_sensor'].sum())} / {len(g)}")

if __name__ == "__main__":
    main()
