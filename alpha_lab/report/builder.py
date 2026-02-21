from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from alpha_lab.types import EvaluationResult, GuardReport

matplotlib.use("Agg")


class ReportBuilder:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def _plot_series(
        self,
        series: pd.Series,
        path: Path,
        title: str,
        ylabel: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=140)
        if not series.empty:
            series.sort_index().plot(ax=ax, lw=1.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _plot_quantiles(self, quantile_returns: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=140)
        if not quantile_returns.empty:
            cumulative = (1 + quantile_returns.fillna(0.0)).cumprod() - 1.0
            for col in cumulative.columns:
                cumulative[col].plot(ax=ax, lw=1.1, label=f"Q{int(col) + 1}")
            ax.legend(ncol=min(5, len(cumulative.columns)), fontsize=8)
        ax.set_title("Quantile Cumulative Returns")
        ax.set_ylabel("Cumulative Return")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def build(
        self,
        task_name: str,
        evaluations: dict[str, EvaluationResult],
        guard_report: GuardReport,
    ) -> Path:
        report_root = self.output_dir / task_name
        report_root.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, float | str]] = []
        sections: list[str] = []

        for factor_name, result in evaluations.items():
            factor_dir = report_root / factor_name
            factor_dir.mkdir(parents=True, exist_ok=True)

            ic_plot = factor_dir / "ic.png"
            rank_ic_plot = factor_dir / "rank_ic.png"
            ls_plot = factor_dir / "long_short.png"
            turnover_plot = factor_dir / "turnover.png"
            quantile_plot = factor_dir / "quantiles.png"

            self._plot_series(result.ic_series, ic_plot, f"{factor_name} IC", "IC")
            self._plot_series(
                result.rank_ic_series,
                rank_ic_plot,
                f"{factor_name} RankIC",
                "RankIC",
            )
            self._plot_series(
                result.long_short_returns,
                ls_plot,
                f"{factor_name} Long-Short Return",
                "Return",
            )
            self._plot_series(
                result.turnover,
                turnover_plot,
                f"{factor_name} Turnover",
                "Turnover",
            )
            self._plot_quantiles(result.quantile_returns, quantile_plot)

            metrics_df = pd.DataFrame([result.metrics]).T.reset_index()
            metrics_df.columns = ["metric", "value"]
            metrics_df.to_csv(factor_dir / "metrics.csv", index=False)
            result.stability.to_csv(factor_dir / "stability.csv", index=False)

            summary_rows.append(
                {
                    "factor": factor_name,
                    "ic_mean": result.metrics.get("ic_mean", float("nan")),
                    "rank_ic_mean": result.metrics.get("rank_ic_mean", float("nan")),
                    "ls_sharpe": result.metrics.get("ls_sharpe", float("nan")),
                    "ls_net_mean": result.metrics.get("ls_net_mean", float("nan")),
                    "ls_net_sharpe": result.metrics.get("ls_net_sharpe", float("nan")),
                    "turnover_mean": result.metrics.get("turnover_mean", float("nan")),
                }
            )

            sections.append(
                "<section>"
                f"<h2>{factor_name}</h2>"
                "<h3>Metrics</h3>"
                f"{metrics_df.to_html(index=False, float_format=lambda x: f'{x:.6f}')}"
                "<h3>Stability</h3>"
                f"{result.stability.to_html(index=False, float_format=lambda x: f'{x:.6f}')}"
                "<div class='grid'>"
                f"<img src='{factor_name}/ic.png' alt='IC' />"
                f"<img src='{factor_name}/rank_ic.png' alt='RankIC' />"
                f"<img src='{factor_name}/long_short.png' alt='Long Short' />"
                f"<img src='{factor_name}/turnover.png' alt='Turnover' />"
                f"<img src='{factor_name}/quantiles.png' alt='Quantiles' />"
                "</div></section>"
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(report_root / "summary.csv", index=False)

        guard_df = pd.DataFrame(guard_report.as_rows())
        guard_html = (
            guard_df.to_html(index=False) if not guard_df.empty else "<p>No guard issues.</p>"
        )
        summary_html = (
            summary_df.to_html(index=False, float_format=lambda x: f"{x:.6f}")
            if not summary_df.empty
            else "<p>No factors evaluated.</p>"
        )

        html = (
            "<!doctype html><html><head><meta charset='utf-8' />"
            f"<title>{task_name} - Alpha Lab Report</title>"
            "<style>"
            "body { font-family: Arial, sans-serif; margin: 24px; color: #1d2330; }"
            "h1, h2 { color: #1b3a57; }"
            "table { border-collapse: collapse; width: 100%; margin-bottom: 12px; }"
            "td, th { border: 1px solid #d8dee9; padding: 6px 8px; font-size: 12px; }"
            "th { background: #eef3fb; }"
            ".grid { display: grid; "
            "grid-template-columns: repeat(2, minmax(300px, 1fr)); gap: 10px; }"
            "img { width: 100%; border: 1px solid #e5e9f0; border-radius: 6px; }"
            "section { margin-top: 24px; padding-top: 8px; border-top: 1px solid #e5e9f0; }"
            "</style></head><body>"
            f"<h1>{task_name} Factor Evaluation Report</h1>"
            "<h2>Summary</h2>"
            f"{summary_html}"
            "<h2>Guard Checks</h2>"
            f"{guard_html}"
            f"{''.join(sections)}"
            "</body></html>"
        )

        report_path = report_root / "report.html"
        report_path.write_text(html, encoding="utf-8")
        return report_path
