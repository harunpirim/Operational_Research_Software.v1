"""
Chart Generator for OR Assistant Visualizations

This module provides specialized chart generation for different types of
optimization problems, including network flow diagrams for transportation
problems, sensitivity analysis charts, and Gantt charts for scheduling.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List, Tuple
import re
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class ChartGenerator:
    """Generate visualizations for optimization problem solutions."""

    def __init__(self):
        """Initialize the chart generator with default theme settings."""
        self.theme = {
            "background": "#1E1E2E",
            "paper_bgcolor": "#1E1E2E",
            "plot_bgcolor": "#1E1E2E",
            "font_color": "#FFFFFF",
            "grid_color": "#444444",
            "source_color": "#3498db",  # Blue
            "destination_color": "#2ecc71",  # Green
            "edge_color": "#95a5a6",  # Gray
            "binding_color": "#e74c3c",  # Red
            "nonbinding_color": "#7f8c8d",  # Dark gray
        }

    def generate_network_chart(
        self, solution: Dict[str, Any], problem_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Generate a network flow diagram for transportation problems.

        Args:
            solution: Solution dictionary with 'variables' containing flow values
            problem_data: Problem data with supply/demand information

        Returns:
            Plotly Figure object with network visualization
        """
        # Extract flows from solution variables
        flows = self._extract_flows(solution.get("variables", {}))

        # Get supply and demand from problem_data
        supply = problem_data.get("parameters", {}).get("supply", [])
        demand = problem_data.get("parameters", {}).get("demand", [])

        # Extract source and destination names
        source_names = problem_data.get("parameters", {}).get(
            "source_names", [f"Source_{i}" for i in range(len(supply))]
        )
        dest_names = problem_data.get("parameters", {}).get(
            "dest_names", [f"Dest_{j}" for j in range(len(demand))]
        )

        # Create figure
        fig = go.Figure()

        # Calculate node positions
        n_sources = len(source_names)
        n_dests = len(dest_names)

        source_y = [
            i / (n_sources - 1) if n_sources > 1 else 0.5 for i in range(n_sources)
        ]
        dest_y = [i / (n_dests - 1) if n_dests > 1 else 0.5 for i in range(n_dests)]

        # Add source nodes
        fig.add_trace(
            go.Scatter(
                x=[0] * n_sources,
                y=source_y,
                mode="markers+text",
                marker=dict(size=30, color=self.theme["source_color"]),
                text=[
                    f"{name}<br>Supply: {supply[i] if i < len(supply) else 0}"
                    for i, name in enumerate(source_names)
                ],
                textposition="middle left",
                hovertemplate="%{text}<extra></extra>",
                name="Sources",
                showlegend=False,
            )
        )

        # Add destination nodes
        fig.add_trace(
            go.Scatter(
                x=[1] * n_dests,
                y=dest_y,
                mode="markers+text",
                marker=dict(size=30, color=self.theme["destination_color"]),
                text=[
                    f"{name}<br>Demand: {demand[i] if i < len(demand) else 0}"
                    for i, name in enumerate(dest_names)
                ],
                textposition="middle right",
                hovertemplate="%{text}<extra></extra>",
                name="Destinations",
                showlegend=False,
            )
        )

        # Add edges for flows > 0.001
        max_flow = max([flow for _, _, flow in flows], default=1)

        for i, j, flow in flows:
            if flow > 0.001:
                # Calculate edge width proportional to flow
                width = max(1, 10 * flow / max_flow)

                # Get source and destination positions
                try:
                    source_idx = int(i)
                except ValueError:
                    # Handle letter indices (A, B, C) or named indices
                    if len(i) == 1 and i.isalpha():
                        source_idx = ord(i.upper()) - ord("A")
                    else:
                        # Try to find the index in source_names
                        try:
                            source_idx = source_names.index(i)
                        except ValueError:
                            continue

                try:
                    dest_idx = int(j)
                except ValueError:
                    # Handle letter indices (A, B, C) or named indices
                    if len(j) == 1 and j.isalpha():
                        dest_idx = ord(j.upper()) - ord("A")
                    else:
                        # Try to find the index in dest_names
                        try:
                            dest_idx = dest_names.index(j)
                        except ValueError:
                            continue

                if source_idx < n_sources and dest_idx < n_dests:
                    # Add edge line
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[source_y[source_idx], dest_y[dest_idx]],
                            mode="lines+text",
                            line=dict(color=self.theme["edge_color"], width=width),
                            text=[f"{flow:.0f} units"],
                            textposition="middle center",
                            hovertemplate=(
                                f"From {source_names[source_idx]} to"
                                f" {dest_names[dest_idx]}: {flow:.0f}"
                                " units<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

        # Update layout
        fig.update_layout(
            title="Transportation Network Flow",
            xaxis=dict(showgrid=False, showticklabels=False, range=[-0.2, 1.2]),
            yaxis=dict(showgrid=False, showticklabels=False, range=[-0.1, 1.1]),
            paper_bgcolor=self.theme["paper_bgcolor"],
            plot_bgcolor=self.theme["plot_bgcolor"],
            font=dict(color=self.theme["font_color"]),
            hovermode="closest",
            margin=dict(l=100, r=100, t=50, b=50),
        )

        return fig

    def generate_sensitivity_chart(self, solution: Dict[str, Any]) -> go.Figure:
        """
        Generate a bar chart showing shadow prices for LP sensitivity analysis.

        Args:
            solution: Solution dictionary potentially containing shadow prices

        Returns:
            Plotly Figure object with sensitivity analysis
        """
        # Check if shadow prices are available
        shadow_prices = solution.get("shadow_prices", {})

        if not shadow_prices:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No sensitivity analysis data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color=self.theme["font_color"]),
            )
        else:
            # Prepare data for bar chart
            constraints = list(shadow_prices.keys())
            prices = list(shadow_prices.values())

            # Determine colors based on binding status (shadow price > 0)
            colors = [
                (
                    self.theme["binding_color"]
                    if abs(p) > 0.001
                    else self.theme["nonbinding_color"]
                )
                for p in prices
            ]

            # Create bar chart
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=constraints,
                        y=prices,
                        marker_color=colors,
                        hovertemplate="%{x}<br>Shadow Price: %{y:.4f}<extra></extra>",
                    )
                ]
            )

            # Add horizontal line at y=0
            fig.add_hline(y=0, line_color=self.theme["grid_color"], line_width=1)

            fig.update_layout(
                title="Sensitivity Analysis - Shadow Prices",
                xaxis_title="Constraint",
                yaxis_title="Shadow Price",
                showlegend=False,
            )

        # Apply theme
        fig.update_layout(
            paper_bgcolor=self.theme["paper_bgcolor"],
            plot_bgcolor=self.theme["plot_bgcolor"],
            font=dict(color=self.theme["font_color"]),
            xaxis=dict(gridcolor=self.theme["grid_color"]),
            yaxis=dict(gridcolor=self.theme["grid_color"]),
        )

        return fig

    def generate_gantt_chart(
        self, solution: Dict[str, Any], problem_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Generate a Gantt chart for scheduling problems.

        Args:
            solution: Solution dictionary with scheduling variables
            problem_data: Problem data with job information

        Returns:
            Plotly Figure object with Gantt chart
        """
        # Extract schedule from solution
        schedule_data = self._extract_schedule(
            solution.get("variables", {}), problem_data
        )

        if not schedule_data:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No scheduling data found in solution",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color=self.theme["font_color"]),
            )
        else:
            # Convert to DataFrame for plotly express
            df = pd.DataFrame(schedule_data)

            # Create color mapping for jobs
            unique_jobs = df["Job"].unique()
            colors = px.colors.qualitative.Set3[: len(unique_jobs)]
            color_map = dict(zip(unique_jobs, colors))

            # Create Gantt chart
            fig = px.timeline(
                df,
                x_start="Start",
                x_end="Finish",
                y="Machine",
                color="Job",
                color_discrete_map=color_map,
                hover_data=["Duration"],
                title="Job Schedule",
            )

            # Update layout for better visibility
            fig.update_yaxes(autorange="reversed")  # Machines from top to bottom

        # Apply dark theme
        fig.update_layout(
            paper_bgcolor=self.theme["paper_bgcolor"],
            plot_bgcolor=self.theme["plot_bgcolor"],
            font=dict(color=self.theme["font_color"]),
            xaxis=dict(gridcolor=self.theme["grid_color"], title="Time"),
            yaxis=dict(gridcolor=self.theme["grid_color"], title="Machine/Resource"),
        )

        return fig

    def _extract_flows(
        self, variables: Dict[str, float]
    ) -> List[Tuple[str, str, float]]:
        """
        Extract flow values from solution variables.

        Handles patterns like:
        - ship_(i,j)
        - ship_i_j
        - x_i_j
        - flow_source_dest
        """
        flows = []

        # Common patterns for transportation variables
        patterns = [
            r"ship_\((\w+),(\w+)\)",  # ship_(i,j)
            r"ship_(\w+)_(\w+)",  # ship_i_j
            r"x_(\w+)_(\w+)",  # x_i_j
            r"flow_(\w+)_(\w+)",  # flow_source_dest
            r"transport_(\w+)_(\w+)",  # transport_i_j
        ]

        for var_name, value in variables.items():
            if value and value > 0.001:  # Only consider positive flows
                for pattern in patterns:
                    match = re.match(pattern, var_name)
                    if match:
                        source, dest = match.groups()
                        flows.append((source, dest, value))
                        break

        return flows

    def _extract_schedule(
        self, variables: Dict[str, float], problem_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract scheduling information from solution variables.

        Returns a list of dictionaries suitable for Gantt chart creation.
        """
        schedule_data = []

        # Get base date for timeline
        base_date = datetime(2024, 1, 1)

        # Look for start time variables
        start_times = {}
        processing_times = problem_data.get("parameters", {}).get(
            "processing_times", {}
        )

        # Pattern matching for scheduling variables
        for var_name, value in variables.items():
            # Match patterns like start_j_m (job j on machine m)
            match = re.match(r"start_(\w+)_(\w+)", var_name)
            if match and value is not None:
                job, machine = match.groups()
                start_times[(job, machine)] = value

        # Build schedule data
        for (job, machine), start_time in start_times.items():
            # Get processing time (default to 1 if not found)
            duration = 1
            if isinstance(processing_times, dict):
                duration = processing_times.get(job, {}).get(machine, 1)
            elif isinstance(processing_times, list) and job.isdigit():
                # Handle list format
                job_idx = int(job)
                if job_idx < len(processing_times):
                    duration = processing_times[job_idx]

            schedule_data.append(
                {
                    "Job": f"Job {job}",
                    "Machine": f"Machine {machine}",
                    "Start": base_date + timedelta(hours=start_time),
                    "Finish": base_date + timedelta(hours=start_time + duration),
                    "Duration": duration,
                }
            )

        return schedule_data

    def generate_efficient_frontier(
        self, solution: Dict[str, Any], problem_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Generate efficient frontier chart for portfolio optimization problems.

        Args:
            solution: Solution dictionary with optimal portfolio weights
            problem_data: Problem data with returns and covariance matrix

        Returns:
            Plotly Figure object with efficient frontier visualization
        """
        # Check if this is a portfolio optimization problem
        if problem_data.get("problem_type") != "portfolio_optimization":
            fig = go.Figure()
            fig.add_annotation(
                text=(
                    "Efficient frontier is only available for portfolio optimization"
                    " problems"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=self.theme["font_color"]),
            )
            fig.update_layout(
                paper_bgcolor=self.theme["paper_bgcolor"],
                plot_bgcolor=self.theme["plot_bgcolor"],
            )
            return fig

        # Extract returns and covariance from parameters
        params = problem_data.get("parameters", {})
        returns = params.get("returns", params.get("expected_returns", []))
        covariance = params.get("covariance", params.get("covariance_matrix", []))

        if not returns or not covariance:
            fig = go.Figure()
            fig.add_annotation(
                text="Missing returns or covariance data for efficient frontier",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=self.theme["font_color"]),
            )
            fig.update_layout(
                paper_bgcolor=self.theme["paper_bgcolor"],
                plot_bgcolor=self.theme["plot_bgcolor"],
            )
            return fig

        # Convert to numpy arrays
        returns = np.array(returns)
        covariance = np.array(covariance)
        n_assets = len(returns)

        # Generate efficient frontier points
        frontier_returns = []
        frontier_volatilities = []
        sharpe_ratios = []

        # Risk-free rate assumption (2% annual)
        risk_free_rate = 0.02

        # Generate 100 portfolio points with different risk tolerances
        risk_levels = np.linspace(0, 1, 100)

        for risk_level in risk_levels:
            # For demonstration, we'll use a simple equal-weight adjusted by risk level
            # In practice, this would involve solving an optimization problem
            weights = np.ones(n_assets) / n_assets

            # Adjust weights based on risk level (simplified approach)
            if risk_level > 0:
                # Tilt towards higher return assets
                return_ranks = np.argsort(returns)[::-1]
                for i, rank in enumerate(return_ranks):
                    weights[rank] *= 1 + risk_level * (n_assets - i) / n_assets
                weights /= weights.sum()

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            frontier_returns.append(portfolio_return)
            frontier_volatilities.append(portfolio_volatility)

            # Calculate Sharpe ratio
            if portfolio_volatility > 0:
                sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

        # Extract optimal portfolio from solution
        optimal_weights = []
        for i in range(n_assets):
            weight_var_names = [f"weight_{i}", f"w_{i}", f"x_{i}", f"allocation_{i}"]
            for var_name in weight_var_names:
                if var_name in solution.get("variables", {}):
                    optimal_weights.append(solution["variables"][var_name])
                    break
            else:
                optimal_weights.append(0)

        if sum(optimal_weights) > 0:
            optimal_weights = np.array(optimal_weights)
            optimal_return = np.dot(optimal_weights, returns)
            optimal_variance = np.dot(
                optimal_weights, np.dot(covariance, optimal_weights)
            )
            optimal_volatility = np.sqrt(optimal_variance)
        else:
            optimal_return = None
            optimal_volatility = None

        # Calculate equal-weight benchmark
        equal_weights = np.ones(n_assets) / n_assets
        equal_return = np.dot(equal_weights, returns)
        equal_variance = np.dot(equal_weights, np.dot(covariance, equal_weights))
        equal_volatility = np.sqrt(equal_variance)

        # Create the figure
        fig = go.Figure()

        # Add efficient frontier scatter plot
        fig.add_trace(
            go.Scatter(
                x=frontier_volatilities,
                y=frontier_returns,
                mode="markers",
                marker=dict(
                    size=8,
                    color=sharpe_ratios,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                ),
                text=[
                    f"Return: {r:.2%}<br>Risk: {v:.2%}<br>Sharpe: {s:.2f}"
                    for r, v, s in zip(
                        frontier_returns, frontier_volatilities, sharpe_ratios
                    )
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Efficient Frontier",
            )
        )

        # Add optimal portfolio marker
        if optimal_return is not None:
            fig.add_trace(
                go.Scatter(
                    x=[optimal_volatility],
                    y=[optimal_return],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="gold",
                        symbol="star",
                        line=dict(color="white", width=2),
                    ),
                    text=(
                        f"Optimal Portfolio<br>Return: {optimal_return:.2%}<br>Risk:"
                        f" {optimal_volatility:.2%}"
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    name="Optimal Portfolio",
                    showlegend=True,
                )
            )

        # Add equal-weight benchmark
        fig.add_trace(
            go.Scatter(
                x=[equal_volatility],
                y=[equal_return],
                mode="markers",
                marker=dict(
                    size=15,
                    color="red",
                    symbol="square",
                    line=dict(color="white", width=1),
                ),
                text=(
                    f"Equal-Weight<br>Return: {equal_return:.2%}<br>Risk:"
                    f" {equal_volatility:.2%}"
                ),
                hovertemplate="%{text}<extra></extra>",
                name="Equal-Weight",
                showlegend=True,
            )
        )

        # Update layout
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Portfolio Volatility (Std Dev)",
            yaxis_title="Expected Annual Return",
            paper_bgcolor=self.theme["paper_bgcolor"],
            plot_bgcolor=self.theme["plot_bgcolor"],
            font=dict(color=self.theme["font_color"]),
            xaxis=dict(gridcolor=self.theme["grid_color"], tickformat=".1%"),
            yaxis=dict(gridcolor=self.theme["grid_color"], tickformat=".1%"),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"
            ),
        )

        return fig

    def generate_variable_distribution(self, solution: Dict[str, Any]) -> go.Figure:
        """
        Generate a pie chart showing the proportion of objective contributed by each variable.

        Works as a generic fallback chart for any problem type.

        Args:
            solution: Solution dictionary with variable values

        Returns:
            Plotly Figure object with variable distribution
        """
        variables = solution.get("variables", {})

        # Calculate contribution of each variable to objective
        # For simplicity, we'll use absolute values
        contributions = []

        for var_name, value in variables.items():
            if value and abs(value) > 0.001:
                contributions.append(
                    {
                        "variable": var_name,
                        "value": abs(value),
                        "contribution": abs(
                            value
                        ),  # Simplified - in reality would need objective coefficients
                    }
                )

        if not contributions:
            fig = go.Figure()
            fig.add_annotation(
                text="No non-zero variables found in solution",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=self.theme["font_color"]),
            )
            fig.update_layout(
                paper_bgcolor=self.theme["paper_bgcolor"],
                plot_bgcolor=self.theme["plot_bgcolor"],
            )
            return fig

        # Sort by contribution and take top 10
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        if len(contributions) > 10:
            top_10 = contributions[:10]
            others_value = sum(c["contribution"] for c in contributions[10:])
            top_10.append({"variable": "Others", "contribution": others_value})
            contributions = top_10

        # Create pie chart
        labels = [c["variable"] for c in contributions]
        values = [c["contribution"] for c in contributions]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    marker=dict(
                        colors=px.colors.qualitative.Set3,
                        line=dict(color=self.theme["background"], width=2),
                    ),
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate=(
                        "%{label}<br>Contribution:"
                        " %{value:.2f}<br>%{percent}<extra></extra>"
                    ),
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title="Variable Contribution to Objective",
            paper_bgcolor=self.theme["paper_bgcolor"],
            plot_bgcolor=self.theme["plot_bgcolor"],
            font=dict(color=self.theme["font_color"]),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(0,0,0,0.5)",
            ),
            margin=dict(l=0, r=150, t=50, b=0),
        )

        return fig


def create_chart(
    chart_type: str, solution: Dict[str, Any], problem_data: Dict[str, Any]
) -> Optional[go.Figure]:
    """
    Convenience function to create charts based on problem type.

    Args:
        chart_type: Type of chart ('network', 'sensitivity', 'gantt', 'frontier', 'distribution')
        solution: Solution dictionary
        problem_data: Problem data dictionary

    Returns:
        Plotly Figure or None if chart type not recognized
    """
    generator = ChartGenerator()

    if chart_type == "network":
        return generator.generate_network_chart(solution, problem_data)
    elif chart_type == "sensitivity":
        return generator.generate_sensitivity_chart(solution)
    elif chart_type == "gantt":
        return generator.generate_gantt_chart(solution, problem_data)
    elif chart_type == "frontier":
        return generator.generate_efficient_frontier(solution, problem_data)
    elif chart_type == "distribution":
        return generator.generate_variable_distribution(solution)
    else:
        return None
