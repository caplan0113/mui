import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from lib import *
from scipy.spatial.transform import Rotation as R

allData = load_all()

# Dash アプリの初期化
app = dash.Dash(__name__)

# 3D ベクトル表示用の関数
def create_figure(data):
    pos = data["pos"]; rot = data["rot"]
    xmin, ymin, zmin = np.min(pos, axis=0)
    xmax, ymax, zmax = np.max(pos, axis=0)
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    # print(xmin, ymin, zmin, xmax, ymax, zmax, max_range)
    # mid_x = (xmax + xmin) / 2; mid_y = (ymax + ymin) / 2; mid_z = (zmax + zmin) / 2
    # xmin = mid_x - max_range / 2; xmax = mid_x + max_range / 2
    # ymin = mid_y - max_range / 2; ymax = mid_y + max_range / 2
    # zmin = mid_z - max_range / 2; zmax = mid_z + max_range / 2
    # print(mid_x, mid_y, mid_z, xmin, ymin, zmin, xmax, ymax, zmax)

    base_vec = np.array([0, 0, 1])
    vec = [R.from_euler("xyz", r, degrees=True).apply(base_vec) for r in rot]

    fig = go.Figure(
        data=[
            go.Cone(
                x=[pos[0][0]], y=[pos[0][1]], z=[pos[0][2]],
                u=[vec[0][0]], v=[vec[0][1]], w=[vec[0][2]],
                sizemode="absolute",
                sizeref=1,
                anchor="tail",
                colorscale="Blues"
            )
        ],
        layout=go.Layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax]),
                zaxis=dict(range=[zmin, zmax]),
                # camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # 初期視点
            ),
            updatemenus=[dict(
                type="buttons",
                x=0.1,
                y=0.05,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 16, "redraw": True}, "fromcurrent": True}]
                )]
            ),
                dict(
                    type="buttons",
                    x=0.1,
                    y=0.15,
                    buttons=[dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]
                    )]
                )
            ],
            sliders=[dict(
                steps=[dict(
                    method="animate",
                    args=[[f"{t}"], {"mode": "immediate", "frame": {"duration": 16, "redraw": True}}]
                ) for t in data["time"]],
                currentvalue=dict(visible=True, font=dict(size=14)),
                x=0.1,
                y=0
            )],
            margin=dict(l=0, r=0, t=0, b=0)
        ),
        frames=[go.Frame(
            data=[go.Cone(
                x=[p[0]], y=[p[1]], z=[p[2]],
                u=[v[0]], v=[v[1]], w=[v[2]],
                sizemode="absolute",
                sizeref=1,
                anchor="tail",
                colorscale="Blues"
            )],
            name=f"{t}"
        ) for t, p, v in zip(data["time"], pos, vec)]
    )
    return fig

def create_figure_2d(data):
    pos = data["pos"]
    rot = data["rot"]
    vec = [R.from_euler("xyz", r, degrees=True).apply([0, 0, 1]) for r in rot]

    x = [p[0] for p in pos]
    z = [p[2] for p in pos]
    u = [v[0] for v in vec]
    w = [v[2] for v in vec]

    fig = go.Figure()

    # 最初の矢印（ベクトル）をプロット
    fig.add_trace(go.Scatter(
        x=[x[0], x[0] + u[0]],
        y=[z[0], z[0] + w[0]],
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(width=2, color='blue'),
        name="Vector"
    ))

    # アニメーションフレームを作成
    frames = []
    for i in range(len(data["time"])):
        frames.append(go.Frame(
            data=[go.Scatter(
                x=[x[i], x[i] + u[i]],
                y=[z[i], z[i] + w[i]],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(width=2, color='blue')
            )],
            name=f"{data['time'][i]}"
        ))

    fig.frames = frames

    # アニメーションUI
    fig.update_layout(
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1, range=[0, 30]),
        yaxis=dict(title="Z", range=[10, 30]),
        margin=dict(l=20, r=20, t=20, b=20),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 16, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ],
            x=0.1, y=0
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[f"{t}"], {"mode": "immediate", "frame": {"duration": 16, "redraw": True}}],
                        label=str(t)) for t in data["time"]],
            x=0.1,
            y=-0.1,
            currentvalue=dict(visible=True)
        )]
    )

    return fig


# 初期ベクトル
initial_vector = allData[3][0][0]

# レイアウト
app.layout = html.Div([
    html.Div([
        "User-ID: ", dcc.Input(id="userId", type="number", value=3, step=1, min="3", max="6", style={'width': '50px'}),
        "UI-ID: ", dcc.Input(id="uiId", type="number", value=0, step=1, min="0", max="3", style={'width': '50px'}),
        "Task-ID: ", dcc.Input(id="taskId", type="number", value=0, step=1, min="0", max="8", style={'width': '50px'})
    ]),
    html.Div(id='error-message', style={'color': 'red'}),
    dcc.Graph(id='vector-graph', figure=create_figure_2d(initial_vector))
])

# 入力されたベクトルに基づいて図を更新するコールバック
@app.callback(
    Output('vector-graph', 'figure'),
    Output('error-message', 'children'),
    Input("userId", "value"),
    Input("uiId", "value"),
    Input("taskId", "value")
)
def update_vector(userId, uiId, taskId):
    try:
        # ベクトル成分を入力から取得
        if userId is None or uiId is None or taskId is None:
            raise ValueError("すべてのフィールドに値を入力してください")
        userId = int(userId); uiId = int(uiId); taskId = int(taskId)
        return create_figure_2d(allData[userId][uiId][taskId]), ""
    except Exception as e:
        traceback.print_exc()
        return dash.no_update, f"⚠️ 入力エラー: {str(e)}"

# 実行
if __name__ == '__main__':
    app.run(debug=True)
