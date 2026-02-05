import csv
import os
import matplotlib.pyplot as plt
import pandas as pd


class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.episode_log_path = os.path.join(output_dir, "training_log.csv")
        self.step_log_path = os.path.join(output_dir, "detailed_log.csv")

        # Initialize logs with headers if they don't exist
        self._ensure_header(
            self.episode_log_path, ["episode", "reward", "steps", "lr", "entropy"]
        )
        self._ensure_header(
            self.step_log_path,
            ["episode", "step", "reward", "correction_l2", "dist_goal"],
        )

        self.episode_buffer = []
        self.step_buffer = []

    def _ensure_header(self, path, expected_cols):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(expected_cols)
            return

        # Check if existing file needs migration
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)

            if header and len(header) < len(expected_cols):
                print(
                    f"Migrating log file {path}: adding missing columns {expected_cols[len(header):]}"
                )
                # Read all data
                with open(path, "r", newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                # Update header and pad rows
                rows[0] = expected_cols
                for i in range(1, len(rows)):
                    while len(rows[i]) < len(expected_cols):
                        rows[i].append("")

                # Write back
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
        except Exception as e:
            print(f"Warning: Could not check/migrate log file {path}: {e}")

    def truncate(self, last_episode):
        """Remove all entries after last_episode to resume from a specific checkpoint."""
        for path in [self.episode_log_path, self.step_log_path]:
            if not os.path.exists(path):
                continue

            try:
                with open(path, "r", newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                header = rows[0]
                truncated_rows = [header]
                for row in rows[1:]:
                    if not row:
                        continue
                    try:
                        # First column is always 'episode' in both files
                        ep = int(row[0])
                        if ep <= last_episode:
                            truncated_rows.append(row)
                    except (ValueError, IndexError):
                        continue

                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(truncated_rows)

                print(f"Truncated {path} to episode {last_episode}")
            except Exception as e:
                print(f"Error truncating {path}: {e}")

    def log_step(self, episode, step, reward, correction, dist):
        self.step_buffer.append([episode, step, reward, correction, dist])

    def log_episode(self, episode, reward, steps, lr, entropy):
        self.episode_buffer.append([episode, reward, steps, lr, entropy])
        self.flush()

    def flush(self):
        if self.episode_buffer:
            with open(self.episode_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.episode_buffer)
            self.episode_buffer = []

        if self.step_buffer:
            with open(self.step_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.step_buffer)
            self.step_buffer = []

    def plot(self, show=False):
        try:
            df = pd.read_csv(self.episode_log_path)
            if df.empty:
                print("No data to plot.")
                return

            plt.figure(figsize=(10, 6))
            plt.plot(df["episode"], df["reward"], label="Episode Reward")
            # Rolling average
            if len(df) > 10:
                plt.plot(
                    df["episode"],
                    df["reward"].rolling(window=10).mean(),
                    label="10-Ep Moving Avg",
                )

            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Training Progress")
            plt.legend()
            plt.grid(True)

            save_path = os.path.join(self.output_dir, "training_plot.png")
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

            if show:
                plt.show()  # Note: blocks execution
            else:
                plt.close()

        except Exception as e:
            print(f"Error plotting: {e}")
