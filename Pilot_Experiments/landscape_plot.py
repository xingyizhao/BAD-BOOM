import plotly.graph_objects as go
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot 3D landscape for backdoor loss basin and ASR.")
    parser.add_argument("--threat_scenario", type=str, default="sentiment_steering", help="Threat scenario: sentiment_steering or targeted_refusal.")
    parser.add_argument("--backdoor_attack_method", type=str, default="AddSent", help="Backdoor attack method: AddSent, Sleeper, or VPI.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimization strategy: AdamW, SAM, or BAD-BOOM.")
    parser.add_argument("--model_series", type=str, help="Model series: e.g., Qwen3-0.6B, Qwen3-1.7B, Llama-1B.")

    landscape_args = parser.parse_args()

    landscape_data_path = landscape_args.threat_scenario + "_" + landscape_args.backdoor_attack_method + "_" + landscape_args.model_series + "_" + landscape_args.optimizer
    landscape_data = np.load("Figure/landscape/" + landscape_data_path + ".npy")

    alphas, betas, z_loss, z_asr = landscape_data["alphas"], landscape_data["betas"], landscape_data["z_loss"], landscape_data["z_asr"]

    ####### Loss Landscape #######
    fig_loss_landscape = go.Figure([go.Surface(x=alphas, y=betas, z=z_loss, colorscale="Viridis", contours=dict(z=dict(show=False)), showscale=False)])
    grid_kw = dict(showgrid=True, gridcolor="rgba(0,0,0,0.3)", gridwidth=3)

    fig_loss_landscape.update_layout(
        scene=dict(
            domain=dict(x=[0.0, 0.94], y=[0.0, 1.0]),
            xaxis=dict(title="α", title_font=dict(size=40),
                       tickfont=dict(size=20, family="sans-serif"),
                       ticks="outside", ticklen=0, **grid_kw),
            yaxis=dict(title="β", title_font=dict(size=40),
                       tickfont=dict(size=20, family="sans-serif"),
                       ticks="outside", ticklen=0, **grid_kw),
            zaxis=dict(title="ΔL", title_font=dict(size=40),
                       tickfont=dict(size=20, family="sans-serif"),
                       ticks="outside", ticklen=0, **grid_kw),
            aspectmode="cube",
        ),
        margin=dict(l=4, r=4, t=6, b=4),
    )

    fig_loss_landscape.write_html("Figure/landscape/" + landscape_data_path + "_landscape_plot.html", include_plotlyjs="cdn")

    ####### Loss Contour #######
    fig_loss_contour = go.Figure(data=go.Contour(x=alphas, y=betas, z=z_loss, colorscale="Viridis", contours=dict(showlabels=True, labelfont=dict(size=25)), colorbar=dict(tickfont=dict(size=50))))
    fig_loss_contour.update_layout(
            xaxis=dict(title="α", title_font=dict(size=50), tickfont=dict(size=50, family="sans-serif"), ticks="outside", ticklen=10, tickwidth=6, title_standoff=0),
            yaxis=dict(title="β", title_font=dict(size=50), tickfont=dict(size=50, family="sans-serif"), ticks="outside", ticklen=10, tickwidth=6, title_standoff=0),
            autosize=False, width=800, height=800, margin=dict(l=60, r=80, t=60, b=60),
    )

    fig_loss_contour.write_html("Figure/landscape/" + landscape_data_path + "_contour_plot.html", include_plotlyjs="cdn")

    ####### ASR Landscape #######
    fig_asr_landscape = go.Figure([go.Surface(x=alphas, y=betas, z=z_asr, colorscale="Viridis", contours=dict(z=dict(show=False)), showscale=False)])

    fig_asr_landscape.update_layout(
        scene=dict(
            domain=dict(x=[0.0, 0.94], y=[0.0, 1.0]),
            xaxis=dict(title="α", title_font=dict(size=40),
                       tickfont=dict(size=20, family="sans-serif"),
                       ticks="outside", ticklen=0, **grid_kw),
            yaxis=dict(title="β", title_font=dict(size=40),
                       tickfont=dict(size=20, family="sans-serif"),
                       ticks="outside", ticklen=0, **grid_kw),
            zaxis=dict(title="ASR", title_font=dict(size=40),
                       tickfont=dict(size=20, family="sans-serif"),
                       ticks="outside", ticklen=0, **grid_kw),
            aspectmode="cube",
        ),
        margin=dict(l=4, r=4, t=6, b=4),
    )
    fig_asr_landscape.write_html("Figure/landscape/" + landscape_data_path + "_asr_landscape_plot.html", include_plotlyjs="cdn")

    ####### ASR Contour #######
    fig_asr_contour = go.Figure(data=go.Contour(x=alphas, y=betas, z=z_asr, colorscale="Viridis", contours=dict(showlabels=True, labelfont=dict(size=20)), colorbar=dict(tickfont=dict(size=50))))

    fig_asr_contour.update_layout(
        xaxis=dict(title="α", title_font=dict(size=50), tickfont=dict(size=50, family="sans-serif"), ticks="outside", ticklen=10, tickwidth=6, title_standoff=0),
        yaxis=dict(title="β", title_font=dict(size=50), tickfont=dict(size=50, family="sans-serif"), ticks="outside", ticklen=10, tickwidth=6, title_standoff=0),
        autosize=False, width=800, height=800, margin=dict(l=60, r=80, t=60, b=60),
    )
    fig_asr_contour.write_html("Figure/landscape/" + landscape_data_path + "_asr_contour_plot.html", include_plotlyjs="cdn")

if __name__ == "__main__":
    main()