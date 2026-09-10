import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import RESULTS_PATH, MONTHLY_MAE_CSV_PATH, MONTHLY_MAE_PLOT


# ==========================================================
# 1) Load 2025 prediction results (from train_temperature_model.py)
# ==========================================================
print("Loading 2025 prediction results...")

if not RESULTS_PATH.exists():
    print(f"Prediction results not found: {RESULTS_PATH}")
    print("Run src/train_temperature_model.py first.")
    raise SystemExit(1)

results = pd.read_csv(RESULTS_PATH)
results["target_date"] = pd.to_datetime(results["target_date"])


# ==========================================================
# 2) Absolute error and monthly grouping
# ==========================================================
results["absolute_error"] = (
    results["predicted_tavg"] - results["actual_tavg"]
).abs()

results["month"] = results["target_date"].dt.month

overall_mae = results["absolute_error"].mean()

monthly_mae = (
    results.groupby("month")["absolute_error"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "monthly_mae", "count": "n_days"})
    .reset_index()
)


# ==========================================================
# 3) Save monthly table
# ==========================================================
monthly_mae.to_csv(MONTHLY_MAE_CSV_PATH, index=False)


# ==========================================================
# 4) Print summary
# ==========================================================
print("\n=== Monthly MAE — Ridge Model, Test Year 2025 ===")
print(f"Overall MAE: {overall_mae:.3f} °C")

print(
    monthly_mae.to_string(
        index=False,
        formatters={
            "monthly_mae": "{:.3f}".format,
        },
    )
)

best_month = monthly_mae.loc[monthly_mae["monthly_mae"].idxmin()]
worst_month = monthly_mae.loc[monthly_mae["monthly_mae"].idxmax()]

print(
    f"\nBest month:  {int(best_month['month'])} "
    f"(MAE = {best_month['monthly_mae']:.3f} °C)"
)
print(
    f"Worst month: {int(worst_month['month'])} "
    f"(MAE = {worst_month['monthly_mae']:.3f} °C)"
)


# ==========================================================
# 5) Plot monthly MAE
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 4.2))

bars = ax.bar(
    monthly_mae["month"],
    monthly_mae["monthly_mae"],
    color="steelblue",
    edgecolor="none",
)

ax.axhline(
    overall_mae,
    color="darkorange",
    linestyle="--",
    linewidth=1.4,
    label=f"Overall MAE = {overall_mae:.3f}",
)

for bar, value in zip(bars, monthly_mae["monthly_mae"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha="center",
        fontsize=9,
    )

ax.set_xticks(monthly_mae["month"])
ax.set_xticklabels([str(m) for m in monthly_mae["month"]])

ax.set_xlabel("Month (2025)")
ax.set_ylabel("MAE (°C)")
ax.set_title("Monthly Mean Absolute Error — Ridge Model, Test Year 2025")
ax.set_ylim(0, 1.8)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(MONTHLY_MAE_PLOT, dpi=160, bbox_inches="tight")
plt.close(fig)


# ==========================================================
# 6) Show output files
# ==========================================================
print("\n=== Created Files ===")
print(f"Monthly MAE table: {MONTHLY_MAE_CSV_PATH}")
print(f"Monthly MAE plot:  {MONTHLY_MAE_PLOT}")
