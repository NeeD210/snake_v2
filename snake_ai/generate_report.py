import csv
import statistics
import os

def generate_report(input_file="report.csv", output_file="report_analysis.md"):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    games = []
    scores = []
    losses = []

    try:
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                games.append(int(row['Game']))
                scores.append(int(row['Score']))
                losses.append(float(row['Loss']))
    except ValueError as e:
        print(f"Error parsing CSV: {e}")
        return

    if not games:
        print("No data found in report.")
        return

    total_games = len(games)
    avg_score = statistics.mean(scores)
    max_score = max(scores)
    min_score = min(scores)
    avg_loss = statistics.mean(losses)
    
    # "Win" here defined as finding food at least once (Score > 0)
    wins = sum(1 for s in scores if s > 0)
    win_rate = (wins / total_games) * 100

    # Split into first and second half for trend
    half = total_games // 2
    first_half_avg = statistics.mean(scores[:half]) if half > 0 else 0
    second_half_avg = statistics.mean(scores[half:]) if half > 0 else 0
    improvement = second_half_avg - first_half_avg

    markdown_report = f"""# Snake AI Training Analysis

## Summary
*   **Total Games**: {total_games}
*   **Average Score**: {avg_score:.2f}
*   **Max Score**: {max_score}
*   **Food Discovery Rate (Score > 0)**: {win_rate:.1f}%
*   **Average Loss**: {avg_loss:.4f}

## Progress Trend
*   **First Half Avg Score**: {first_half_avg:.2f}
*   **Second Half Avg Score**: {second_half_avg:.2f}
*   **Improvement**: {improvement:+.2f}

## Conclusion
The agent is successfully finding food in {win_rate:.1f}% of games.
{"The agent is improving over time." if improvement > 0 else "Performance is stable or stagnating."}
"""

    print(markdown_report)

    with open(output_file, 'w') as f:
        f.write(markdown_report)
    
    print(f"Analysis saved to {output_file}")

if __name__ == "__main__":
    generate_report()
