import mlflow
import pandas as pd
import os
import matplotlib.pyplot as plt


def _set_tracking_uri():
    """Internal helper to set MLflow tracking URI to project_root/mlruns."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracking_uri = f"file:///{project_root}/mlruns".replace("\\", "/")
    print("Tracking URI:", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)


def get_last_runs(experiment_name: str, n: int = 9):
    _set_tracking_uri()

    # Get experiment by name
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Fetch all runs
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f"Runs found: {len(df)}")

    # Sort newest first and return last n
    df = df.sort_values(by="start_time", ascending=False).head(n)
    return df


def get_runs_by_ids(run_ids: list[str]):
    """
    Retrieve runs directly by their run_id list.
    Returns a single DataFrame containing only those runs.
    """
    _set_tracking_uri()

    # Query ALL runs across ALL experiments (MLflow limitation)
    df_all = mlflow.search_runs()

    # Filter down to only the requested run IDs
    df_filtered = df_all[df_all["run_id"].isin(run_ids)]

    if df_filtered.empty:
        raise ValueError(f"No runs found for provided IDs: {run_ids}")

    print(f"Found {len(df_filtered)} runs matching provided IDs.")
    return df_filtered


def save_runs_to_excel(df: pd.DataFrame, output_path: str):
    # Remove timezone info for Excel compatibility
    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

    df.to_excel(output_path, index=False)
    print(f"Saved to {output_path}")


def melt_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format MLflow output to Tableau-friendly long format."""
    metric_cols = [c for c in df.columns if c.startswith("metrics.")]

    df_long = df.melt(
        id_vars=["run_id", "experiment_id", "status", "artifact_uri", "start_time", "end_time"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value"
    )

    return df_long

def plot_data(df, output_path: str):
    print("Plotting data...")

    df = df.sort_values("tags.mlflow.runName")

    # Metrics to plot
    metrics = [
        "metrics.cv_mean_accuracy",
        "metrics.cv_mean_recall",
        "metrics.cv_mean_f1_score",
        "metrics.cv_mean_precision"
    ]

    # Extract metric values and convert to percentage
    metric_values = df[metrics].copy() * 100

    # Apply tiny offsets for overlapping metrics (adjust visually as needed)
    offsets = {
        "metrics.cv_mean_accuracy": 0.0,
        "metrics.cv_mean_recall": 0.1,     # +0.1%
        "metrics.cv_mean_f1_score": 0.0,
        "metrics.cv_mean_precision": 0.0
    }

    for m in metrics:
        metric_values[m] += offsets.get(m, 0)

    # Compute min/max for dynamic y-axis
    ymin = metric_values.min().min() - 0.2  # 0.2% padding
    ymax = metric_values.max().max() + 0.2

    plt.figure(figsize=(12, 6))

    # Plot each metric
    for m in metrics:
        plt.plot(
            df["tags.mlflow.runName"],
            metric_values[m],
            marker='o',
            label=m.replace("metrics.cv_mean_", "").capitalize()
        )

    plt.xlabel("Experiment Run Name")
    plt.ylabel("Metric Value (%)")
    plt.title("Comparison of CV Mean Metrics Across Experiments")
    plt.xticks(rotation=45)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_data_old(df: pd.DataFrame, output_path: str):
    import matplotlib.pyplot as plt
    print("Plotting data...")
    df = df.sort_values("tags.mlflow.runName")

    metrics = [
        "metrics.cv_mean_accuracy",
        "metrics.cv_mean_recall",
        "metrics.cv_mean_f1_score",
        "metrics.cv_mean_precision"
    ]

    metric_values = df[metrics].values.flatten()

    # Determine min/max with small padding
    ymin = metric_values.min() - 0.005  # subtract 0.5%
    ymax = metric_values.max() + 0.005  # add 0.5%


    #print(df.columns.tolist())
    # Set figure size
    plt.figure(figsize=(12, 6))

    for m in metrics:
        plt.plot(df["tags.mlflow.runName"], df[m], marker='o', label=m.replace("metrics.", ""))

    plt.xlabel("Experiment Run Name")
    plt.ylabel("Metric Value")
    plt.title("Comparison of CV Mean Metrics Across Experiments")
    plt.xticks(rotation=45)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# ======================================================
# Example usage
# ======================================================

# Option A — get last 9 runs
# df = get_last_runs("Default", 9)

# Option B — get specific runs by ID
df = get_runs_by_ids([
    # individual
    '6ad5eb2a745a4c8382e72fc6f6a220b0',
    '99d72077c98f4472b19a46918305d82a',
    '5c38c47c314b400088259e81b45c2888',
    'add63f443ff44a84aa778378f7a7b8c0',
    'dfc90e71f50049878929d68524c63dfa',
    '3636caee57d048eb9231b57528286787',
    'd4a1480211ef40609bfd311163ff6666', # baseline
    'f77543ff914d467ca78077328e7f60ca',
    'b04a6c319ed8416b8900e34cd880b76a'

    # consolidated
    'b9c7506367d2442a8aa6e24f8f1a6596',
    'b996987b18e54ebc86cb221fb7edfa9e',
    '1a38024763854ba1af26ee9a592ac4c6',
    '082d1d3a70e644cfb5e8fcbcb515d448',
    '00fba5f8a7ff466aa6fdec0ad73fdc33',
    'bc413906cf7a4da7ba21c35d7e6258a0',
    'd800a4bb6cd845898bed13cbcd6047a5',
    'fe0067b1fb174b3a8d41a3df93f5b981',
   # '7ad193bc7e364278be28e08069ef0097' # baseline
    # individual

])

#df = melt_data(df)
plot_data(df, "naive_bayes_experiments_consolidated.png")
save_runs_to_excel(df, "naive_bayes_experiments_consolidated.xlsx")
