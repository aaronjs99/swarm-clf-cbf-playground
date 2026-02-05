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
        if not os.path.exists(self.episode_log_path):
            with open(self.episode_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "steps"])

        if not os.path.exists(self.step_log_path):
            with open(self.step_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["episode", "step", "reward", "correction_l2", "dist_goal"]
                )

        self.episode_buffer = []
        self.step_buffer = []

    def log_step(self, episode, step, reward, correction, dist):
        self.step_buffer.append([episode, step, reward, correction, dist])

    def log_episode(self, episode, reward, steps):
        self.episode_buffer.append([episode, reward, steps])
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
