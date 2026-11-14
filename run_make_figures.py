import os
import numpy as np
import pandas as pd
from scipy import stats as _st
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

def make_figures():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    qc = pd.read_csv(os.path.join(data_dir, "manikin_cycles_qc.csv"))
    tri = pd.read_csv(os.path.join(data_dir, "trial_level_qc.csv"))
    spr = pd.read_csv(os.path.join(data_dir, "spring60_cycles.csv"))

    qc["sensor_f"] = qc["sensor_class"].astype("category")
    qc["set_f"] = qc["set"].astype("category")

    form = "vision_depth ~ C(sensor_f, Treatment(2)) + C(set_f)"
    c1 = "C(sensor_f, Treatment(2))[T.1]"
    c3 = "C(sensor_f, Treatment(2))[T.3]"

    def est(res, term):
        b = float(res.params[term])
        lo, hi = res.conf_int().loc[term].tolist()
        return b, float(res.bse[term]), lo, hi

    qc_ols = smf.ols(form, data=qc).fit()
    qc_lmm = smf.mixedlm(form, data=qc, groups=qc["subject"]).fit(reml=True, method="powell", maxiter=3000)
    qc_gee = sm.GEE.from_formula(form, groups="subject", data=qc, family=Gaussian(), cov_struct=Exchangeable()).fit(maxiter=300)

    ols1, ols3 = est(qc_ols, c1), est(qc_ols, c3)
    lmm1, lmm3 = est(qc_lmm, c1), est(qc_lmm, c3)
    gee1, gee3 = est(qc_gee, c1), est(qc_gee, c3)

    depth = qc["in_range"].mean() * 100.0
    depth_ref = (qc["sensor_class"] == 2).mean() * 100.0
    recoil = qc["recoil_complete"].mean() * 100.0
    recoil_ref = qc["sensor_recoil_complete"].mean() * 100.0
    elbow, vert = 65.0 / 75.0 * 100.0, 68.0 / 75.0 * 100.0
    elbow_ref, vert_ref = 67.0 / 75.0 * 100.0, 70.0 / 75.0 * 100.0
    n_trials = len(tri)
    trial_pass_cv = int(tri["pass_cv"].sum())
    trial_pass_ref = int(tri["pass_sensor"].sum())

    naive_mean = spr["vision_depth"].mean()
    naive_ci = _st.t.interval(0.95, len(spr) - 1, loc=naive_mean, scale=_st.sem(spr["vision_depth"]))
    ms = smf.mixedlm("vision_depth ~ 1", data=spr, groups="subject").fit(reml=True, method="powell", maxiter=2000)
    lmm_mean = float(ms.fe_params["Intercept"])
    lmm_ci = ms.conf_int().loc["Intercept"].tolist()
    gs = sm.GEE.from_formula("vision_depth ~ 1", groups="subject", data=spr, family=Gaussian(), cov_struct=Exchangeable()).fit(maxiter=200)
    gee_mean = float(gs.params["Intercept"])
    gee_ci = gs.conf_int().loc["Intercept"].tolist()

    spr_icc = float(ms.cov_re.iloc[0, 0]) / (float(ms.cov_re.iloc[0, 0]) + float(ms.scale))
    n_subj = spr["subject"].nunique()
    design_effect = 1.0 + (len(spr) / n_subj - 1.0) * spr_icc
    effective_n = len(spr) / design_effect

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })

    BLUE = "#0072B2"
    ORANGE = "#E69F00"
    GREEN = "#009E73"
    VERM = "#D55E00"
    GREY = "#555555"
    LIGHT = "#ECECEC"

    def save_plot(fig, name):
        png_path = os.path.join(fig_dir, name + ".png")
        tiff_path = os.path.join(fig_dir, name + ".tiff")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(tiff_path, bbox_inches="tight", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
        print(f"Generated {name} (.png, .tiff)")

    fig11 = plt.figure(figsize=(7.4, 3.5))
    gs11 = fig11.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)

    ax11a = fig11.add_subplot(gs11[0, 0])
    mcols = [GREY, BLUE, GREEN]
    mshapes = ["o", "s", "D"]
    y_top = [4.4, 3.6, 2.8]
    y_bot = [1.4, 0.6, -0.2]
    all_ys = y_top + y_bot
    all_ests = [ols1, lmm1, gee1, ols3, lmm3, gee3]
    all_cols = mcols * 2
    all_shapes = mshapes * 2

    ax11a.axvline(0, color="#666666", ls="--", lw=0.9, zorder=1)
    ax11a.text(0.0, 5.25, "Reference (5-6 cm)", ha="center", va="bottom", fontsize=6.8, color="#555555", fontstyle="italic")

    for yv, (b, se, lo, hi), col, mk in zip(all_ys, all_ests, all_cols, all_shapes):
        ax11a.errorbar(b, yv, xerr=[[b - lo], [hi - b]], fmt=mk, ms=4.5, color=col, capsize=3.0, lw=1.2, zorder=3)
        lbl = f"{b:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        ax11a.text(b, yv + 0.22, lbl, va="bottom", ha="center", fontsize=6.2, color=col, fontweight="semibold")

    ax11a.text(-2.50, 5.0, "Insufficient depth (<5 cm)", fontsize=7.4, fontweight="bold", color=BLUE, ha="left")
    ax11a.text(-2.50, 2.0, "Excessive depth (>6 cm)", fontsize=7.4, fontweight="bold", color=VERM, ha="left")

    ax11a.set_yticks(all_ys)
    ax11a.set_yticklabels(["OLS", "LMM", "GEE", "OLS", "LMM", "GEE"], fontsize=7.4)
    ax11a.tick_params(axis="y", length=2)
    ax11a.set_xlim(-2.60, 2.60)
    ax11a.set_ylim(-0.7, 5.7)
    ax11a.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax11a.set_xlabel("Adjusted depth difference vs 5-6 cm (cm)")

    legend_elements = [
        Line2D([0], [0], marker="o", color=GREY, label="OLS", ls="", ms=4.5),
        Line2D([0], [0], marker="s", color=BLUE, label="LMM", ls="", ms=4.5),
        Line2D([0], [0], marker="D", color=GREEN, label="GEE", ls="", ms=4.5),
    ]
    ax11a.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.99), frameon=True, facecolor="white", edgecolor="#E0E0E0", framealpha=0.9, ncol=3, handlelength=0.8, columnspacing=0.8, borderaxespad=0.2, fontsize=6.8)
    ax11a.set_title("A  Multilevel vs naive depth contrasts (post-QC)", loc="left", fontweight="bold", pad=8, fontsize=9.2)

    ax11b = fig11.add_subplot(gs11[0, 1])
    rows = [
        ("Naive t-test", naive_mean, naive_ci, GREY, "o"),
        ("Multilevel LMM", lmm_mean, lmm_ci, BLUE, "s"),
        ("Population GEE", gee_mean, gee_ci, GREEN, "D"),
    ]
    y_pos = [2.2, 1.2, 0.2]
    ax11b.axvline(6.00, color=VERM, ls="--", lw=1.0, zorder=1)
    ax11b.text(6.00, 3.0, "Mechanical stop (6.00 cm)", color=VERM, fontsize=6.8, ha="center", va="bottom", fontweight="medium")

    for (name, m, (lo, hi), col, mk), y in zip(rows, y_pos):
        ax11b.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=mk, ms=4.8, color=col, capsize=3.2, lw=1.3, zorder=3)
        lbl = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
        ax11b.text(m, y + 0.22, lbl, va="bottom", ha="center", fontsize=6.2, color=col, fontweight="semibold")

    ax11b.set_yticks(y_pos)
    ax11b.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax11b.tick_params(axis="y", length=2)
    ax11b.set_xlim(5.90, 6.60)
    ax11b.set_ylim(-0.5, 3.4)
    ax11b.set_xlabel("Measured coil-spring depth (cm)")
    ax11b.text(0.97, 0.08, f"Clustering: ICC = {spr_icc:.3f}\nDesign Effect = {design_effect:.1f}\nN_eff = {effective_n:.1f} (vs naive {len(spr)})", transform=ax11b.transAxes, ha="right", va="bottom", fontsize=6.4, bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.2", facecolor="#FBFBFB", edgecolor="#CCCCCC", linewidth=0.6))
    ax11b.set_title("B  Coil-spring precision benchmark (60 fps)", loc="left", fontweight="bold", pad=8, fontsize=9.2)

    save_plot(fig11, "Fig11_Models")

    fig12 = plt.figure(figsize=(7.2, 3.3))
    gs12 = fig12.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.38)

    ax12a = fig12.add_subplot(gs12[0, 0])
    labels = ["Depth (5-6 cm)", "Rate (100-120/min)", "Complete recoil", "Elbow (180 +/- 15 deg)", "Verticality (80-100 deg)"]
    sys_vals = np.array([depth, np.nan, recoil, elbow, vert])
    ref_vals = np.array([depth_ref, np.nan, recoil_ref, elbow_ref, vert_ref])
    yy = np.arange(len(labels))[::-1].astype(float)
    h_bar = 0.32

    ax12a.barh(yy + h_bar/2 + 0.01, np.nan_to_num(sys_vals), height=h_bar, color=BLUE, label="Vision system", zorder=3)
    ax12a.barh(yy - h_bar/2 - 0.01, np.nan_to_num(ref_vals), height=h_bar, color=ORANGE, label="Sensor reference", zorder=3)
    ax12a.barh(yy[1], 15.0, height=h_bar*1.5, color=LIGHT, hatch="////", edgecolor="#C0C0C0", linewidth=0.5, zorder=2)
    ax12a.text(17.0, yy[1], "Not archived", va="center", ha="left", fontsize=7.2, color="#666666", fontstyle="italic")

    for i, (s, r) in enumerate(zip(sys_vals, ref_vals)):
        for val, yv, col in [(s, yy[i] + h_bar/2 + 0.01, BLUE), (r, yy[i] - h_bar/2 - 0.01, ORANGE)]:
            if np.isnan(val):
                continue
            if val >= 75:
                ax12a.text(val - 1.8, yv, f"{val:.1f}%", va="center", ha="right", fontsize=7.0, color="white", fontweight="bold")
            else:
                ax12a.text(val + 1.8, yv, f"{val:.1f}%", va="center", ha="left", fontsize=7.0, color=col, fontweight="bold")

    ax12a.set_yticks(yy)
    ax12a.set_yticklabels(labels, fontsize=7.8)
    ax12a.set_xlim(0, 105)
    ax12a.set_ylim(-0.75, len(labels) - 0.2)
    ax12a.set_xlabel("Compliance (%)")
    ax12a.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax12a.axvline(75, color="#777777", ls="--", lw=0.8, zorder=1)
    ax12a.text(75.0, -0.52, "75% guideline threshold", fontsize=6.8, color="#555555", va="top", ha="center")
    ax12a.legend(loc="upper right", bbox_to_anchor=(0.98, 0.96), frameon=True, facecolor="white", edgecolor="#E0E0E0", framealpha=0.9, ncol=1, handlelength=1.1, borderaxespad=0.2, fontsize=7.0, labelspacing=0.35)
    ax12a.set_title("A  Five-domain compliance (post-QC)", loc="left", fontweight="bold", pad=8, fontsize=9.2)

    ax12b = fig12.add_subplot(gs12[0, 1])
    xs = np.linspace(15, 100, 400)
    dens = {}
    for data, col, lab in [(tri["subindex_cv"], BLUE, "Vision system"), (tri["subindex_sensor"], ORANGE, "Sensor reference")]:
        kde = gaussian_kde(data, bw_method=0.35)
        dv = kde(xs)
        dens[lab] = dv
        ax12b.plot(xs, dv, color=col, lw=1.6, label=lab)
        ax12b.fill_between(xs, 0, dv, color=col, alpha=0.14, linewidth=0)

    ax12b.axvline(75, color=VERM, ls="--", lw=1.0)
    ymax = max(d.max() for d in dens.values())
    ax12b.set_xlim(15, 100)
    ax12b.set_ylim(0, ymax * 1.55)
    ax12b.set_xlabel("Trial-level C-CQI (depth-recoil sub-index)")
    ax12b.set_ylabel("Density")
    ax12b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax12b.text(73.8, ymax * 0.70, "75-point certification threshold", color=VERM, fontsize=6.8, rotation=90, va="center", ha="right", fontweight="medium")
    ax12b.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2, fontsize=7.2)

    info = f"Pass at >= 75 points\nSystem:    {trial_pass_cv}/{n_trials} ({trial_pass_cv/n_trials*100:.1f}%)\nReference: {trial_pass_ref}/{n_trials} ({trial_pass_ref/n_trials*100:.1f}%)"
    ax12b.text(0.98, 0.96, info, transform=ax12b.transAxes, ha="right", va="top", fontsize=6.8, linespacing=1.35, bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.2", facecolor="#FBFBFB", edgecolor="#CCCCCC", linewidth=0.6))
    ax12b.set_title(f"B  C-CQI distribution (n = {n_trials} trials)", loc="left", fontweight="bold", pad=8, fontsize=9.2)

    save_plot(fig12, "Fig12_C-CQI")

if __name__ == "__main__":
    make_figures()
