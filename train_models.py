"""Run once to reproduce all three saved models and measured reports."""

from src.ml import train_model


if __name__ == "__main__":
    for kind in ("crop", "fertilizer", "yield"):
        bundle = train_model(kind)
        metrics = bundle["metrics"]
        summary = f"accuracy={metrics['accuracy']:.3%}" if kind != "yield" else (
            f"MAE={metrics['mae']:.3f} tons/ha, R2={metrics['r2']:.3f}"
        )
        print(f"{kind}: {metrics['selected_model']}; {summary}", flush=True)
