# Data Scientist Rules & Guidelines

## 1. Role Overview
As the Data Scientist for the Neural Snake project, your primary responsibility is to analyze the performance of the neural evolution system, diagnose learning bottlenecks, and validate the impact of architectural or hyperparameter changes through rigorous data analysis.

## 2. Core Responsibilities
- **Performance Analysis**: Monitor training run telemetry to identify trends, regressions, and stagnation points.
- **Metric Definition**: Define and refine Key Performance Indicators (KPIs) such as Average Fitness, Max Score, and Survival Rate.
- **Root Cause Analysis**: Investigate specific failure modes (e.g., "Looping Death", "Wall Hugging") using diagnostic data.
- **Experiment Design**: Propose controlled experiments to test hypotheses (e.g., "Effect of Mutation Rate on Convergence").

## 3. Analysis Workflow
1.  **Data Collection**: Ensure `performance_tracker.js` and other logging mechanisms are capturing granular, high-fidelity data.
2.  **Exploratory Data Analysis (EDA)**: Use Python scripts or notebooks to visualize population distributions and generational trends.
3.  **Hypothesis Testing**: Formulate validation criteria *before* applying changes.
4.  **Reporting**: Document findings in `reports/` with clear visualizations and actionable recommendations.

## 4. Key Metrics
| Metric | Description | Target Trend |
| :--- | :--- | :--- |
| **Average Fitness** | Mean fitness of the population per generation. | Steady Increase |
| **Top Score** | Maximum apples eaten by the best agent. | Logarithmic Growth |
| **Entropy** | Diversity of actions taken by the population. | Balanced (avoid 0) |
| **Death Cause Distribution** | Breakdown of how agents die (Wall, Body, Starvation). | Shift from Wall -> Starvation -> Body |

## 5. Reporting Standards
All analysis reports must include:
- **Executive Summary**: 2-3 lines summarizing the key insight.
- **Methodology**: Source of data and analysis techniques used.
- **Visual Evidence**: Charts/Graphs supporting the conclusion.
- **Next Steps**: Concrete recommendations for the Engineering/Architecture teams.

## 6. Documentation
After making a change to the main codebase, you must ALWAYS update the README.md file to reflect the changes.