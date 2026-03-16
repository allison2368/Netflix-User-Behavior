import dash
from dash import dash_table, dcc, html, Input, Output, State
from dash.dependencies import Input, Output, State, MATCH, ALL
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------
# Load Data
# -----------------------
users = pd.read_csv("data/cleaned/users.csv").head(100)
movies = pd.read_csv("data/cleaned/movies.csv").head(100)

# Train churn model
users["churn"] = 1 - users["is_active"]
numeric_cols = users.select_dtypes(include="number").columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ["is_active", "churn"]]

model_df = pd.concat([users[feature_cols], users["churn"]], axis=1).dropna()
X = model_df[feature_cols]
y = model_df["churn"]

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X, y)

users["churn_probability"] = None
users.loc[X.index, "churn_probability"] = model.predict_proba(X)[:, 1]

# -----------------------
# Dash App
# -----------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Customer & Content Analytics Dashboard"),

    # Analysis type selector
    html.Label("Select Analysis"),
    dcc.Dropdown(
        id="analysis-type",
        options=[
            {"label": "Churn Analysis", "value": "churn"},
            {"label": "Content Analysis", "value": "content"}
        ],
        value="churn",
        clearable=False
    ),

    html.Br(),

    # Dynamic controls (placeholder)
    html.Div(id="dynamic-controls"),

    html.Br(),

    # Metric output
    html.Div(id="metric-output", style={"fontSize": "20px", "fontWeight": "bold"}),

    html.Br(),

    # Table
    dash_table.DataTable(
        id="result-table",
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_table={
            "overflowX": "auto",
            "maxHeight": "500px",
            "overflowY": "auto",
        },
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "backgroundColor": "#f0f0f0",
            "fontWeight": "bold",
            "border": "1px solid #ccc",
        },
        style_data={
            "border": "1px solid #eee",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafafa",
            }
        ],
    ),


    html.Br(),

    # Download button
    html.Button("Download Data", id="download-btn"),
    dcc.Download(id="download-data")
])

# -----------------------
# Dynamic UI
# -----------------------
@app.callback(
    Output("dynamic-controls", "children"),
    Input("analysis-type", "value")
)
def update_controls(analysis_type):
    if analysis_type == "churn":
        return html.Div([
            html.Label("Select churn probability threshold (%)"),
            dcc.Slider(
                id={"type": "threshold-slider", "analysis": "churn"},
                min=0, max=100, step=1, value=85,
                marks={i: str(i) for i in range(0, 101, 20)}
            )
        ])
    else:
        return html.Div([
            html.Label("Minimum IMDb Rating"),
            dcc.Slider(
                id={"type": "threshold-slider", "analysis": "content"},
                min=0, max=10, step=0.1, value=8.0,
                marks={i: str(i) for i in range(0, 11)}
            )
        ])

# -----------------------
# Update table + metric
# ----------------------- 
@app.callback(
    Output("metric-output", "children"),
    Output("result-table", "data"),
    Output("result-table", "columns"),
    Input("analysis-type", "value"),
    Input({"type": "threshold-slider", "analysis": ALL}, "value"),
)
def update_results(analysis_type, slider_values):

    # If no slider exists yet (initial load) 
    if not slider_values: 
        return "", [], []
    
    # Get slider values safely
    slider_value = slider_values[0]
    # -----------------------
    # CHURN ANALYSIS 
    # ----------------------- 
    if analysis_type == "churn": 
        threshold = slider_value / 100 
        filtered = users[users["churn_probability"] >= threshold] 

        metric = f"High Risk Users: {len(filtered)}" 
        columns = [{"name": c, "id": c} for c in filtered.columns] 

        return metric, filtered.to_dict("records"), columns 
    
    # ----------------------- 
    # CONTENT ANALYSIS 
    # ----------------------- 
    else: 
        rating_threshold = slider_value 
        filtered = movies[movies["imdb_rating"] >= rating_threshold] 

        metric = f"Movies Above Rating Threshold: {len(filtered)}" 
        columns = [{"name": c, "id": c} for c in filtered.columns] 

        return metric, filtered.to_dict("records"), columns


# -----------------------
# Download callback
# -----------------------
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("result-table", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, table_data):
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_csv, "filtered_data.csv", index=False)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)