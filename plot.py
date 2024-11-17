import plotly.graph_objects as go

def plot_loss(loss_history: list[float]):
    """Display a line chart with the model loss over the iteration/epochs."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Training Loss'))
    fig.update_layout(
        title="Training Loss Over Batches",
        xaxis_title="Batch Number",
        yaxis_title="Loss" )
    fig.show()

def plot_metrics(results: dict):
    """Display main model metrics for test set."""
    fig = go.Figure(data=[
        go.Bar(name='Metrics', x=list(results.keys()), y=list(results.values())) ])
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis_tickformat='.1f' )
    fig.show()
