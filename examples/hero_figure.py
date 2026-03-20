"""Generate the hero figure for the README."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from kernelboost.utilities import RankTransformer

from kernelmoments import Plotter, PlotStyle

data = yf.download(["KO", "PEP", "^GSPC"], start="2019-01-01", end="2025-12-31")
returns = data["Close"].pct_change().dropna() * 100
returns.columns = ["KO", "PEP", "SP500"]

style = PlotStyle(
    scatter_alpha=0.25,
    scatter_size=6,
    scatter_color="C0",
    line_color="C1",
    line_width=2.5,
    band_line_width=1.2,
    x_quantile_trim=0.02,
)

p = Plotter(returns, x_scaler=RankTransformer(), style=style)
p.fit(x="SP500", y="KO", z="PEP")

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

p.plot(x="SP500", y="KO", moment="mean", bands=True, ax=axes[0, 0])
p.plot(x="SP500", y="KO", moment="variance", ax=axes[0, 1])
p.plot(x="SP500", y="KO", z="PEP", moment="covariance", ax=axes[1, 0])
p.plot(x="SP500", y="KO", z="PEP", moment="correlation", ax=axes[1, 1])

# labels
for ax, label in zip(
    axes.flat,
    [
        "E[KO | SP500]",
        "Var[KO | SP500]",
        "Cov[KO, PEP | SP500]",
        "Corr[KO, PEP | SP500]",
    ],
):
    ax.set_title(label, fontsize=11, fontweight="bold")

fig.tight_layout()
fig.savefig("assets/hero.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved assets/hero.png")
