"""
Plot per-joint activation signals from single_leg_activations.csv.
Produces one PNG per joint (joint_0_hip.png, joint_1_thigh.png, joint_2_calf.png)
plus cost.png, all saved to the same directory as the CSV.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

JOINT_NAMES = ["hip", "thigh", "calf"]
NUM_JOINTS  = 3
NUM_MUSCLES = 6   # 2 per joint: agonist = 2j, antagonist = 2j+1

def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

def plot_joint(df: pd.DataFrame, j: int, out_dir: str):
    name   = JOINT_NAMES[j]
    ag     = 2 * j
    ant    = 2 * j + 1
    t      = df["call"].to_numpy()

    cmd_ag   = df[f"cmd_m{ag}"].to_numpy()
    cmd_ant  = df[f"cmd_m{ant}"].to_numpy()
    act_ag   = df[f"act_m{ag}"].to_numpy()
    act_ant  = df[f"act_m{ant}"].to_numpy()
    tau      = df[f"tau_j{j}"].to_numpy()
    mean_ag  = df[f"samp_mean_m{ag}"].to_numpy()
    std_ag   = df[f"samp_std_m{ag}"].to_numpy()
    mean_ant = df[f"samp_mean_m{ant}"].to_numpy()
    std_ant  = df[f"samp_std_m{ant}"].to_numpy()

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Joint {j} — {name}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    # ── Agonist ──────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.fill_between(t, mean_ag - std_ag, mean_ag + std_ag,
                     alpha=0.25, color="steelblue", label="sample mean±std")
    ax0.plot(t, mean_ag,  color="steelblue", linewidth=0.8, linestyle="--")
    ax0.plot(t, cmd_ag,   color="royalblue", linewidth=1.2, label="committed cmd")
    ax0.plot(t, act_ag,   color="navy",      linewidth=1.2, label="filtered act", linestyle=":")
    ax0.set_ylabel("activation")
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_title("Agonist", fontsize=10)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_xticklabels([])

    # ── Antagonist ───────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.fill_between(t, mean_ant - std_ant, mean_ant + std_ant,
                     alpha=0.25, color="tomato", label="sample mean±std")
    ax1.plot(t, mean_ant, color="tomato",     linewidth=0.8, linestyle="--")
    ax1.plot(t, cmd_ant,  color="firebrick",  linewidth=1.2, label="committed cmd")
    ax1.plot(t, act_ant,  color="darkred",    linewidth=1.2, label="filtered act", linestyle=":")
    ax1.set_ylabel("activation")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Antagonist", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xticklabels([])

    # ── Torque ───────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(t, tau, color="seagreen", linewidth=1.2)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("torque (Nm)")
    ax2.set_title("Net torque (agonist − antagonist)", fontsize=10)
    ax2.set_xticklabels([])

    # ── Best cost ────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.plot(t, df["best_cost"].to_numpy(), color="dimgray", linewidth=1.0)
    ax3.set_ylabel("cost")
    ax3.set_xlabel("update() call")
    ax3.set_title("Best rollout cost", fontsize=10)

    out_path = os.path.join(out_dir, f"joint_{j}_{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
               os.path.join(os.path.dirname(__file__), "single_leg_activations.csv")
    out_dir  = os.path.dirname(os.path.abspath(csv_path))

    df = load(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    for j in range(NUM_JOINTS):
        plot_joint(df, j, out_dir)

    print("Done.")

if __name__ == "__main__":
    main()
