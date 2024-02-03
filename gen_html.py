import re
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

def plot_similarity(filename):
    # Read the selected file
    df = pd.read_csv(filename)

    # Extract the second last four chunks of the layer name
    df['Layer'] = df['Layer'].apply(lambda x: '.'.join([i.zfill(2) if i.isdigit() and len(i) < 2 else i for i in x.split('.')[-5:]]))
    df = df.sort_values('Layer')

    df.set_index('Layer', inplace=True)

    # Filter out 'embed_tokens' from Cosine Similarities
    df_cosine = df[~df.index.str.contains('embed_tokens')]

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Similarity", "Cosine Similarity"))

    # Add traces with custom hover text
    fig.add_trace(
        go.Bar(x=df.index, y=df['Similarity'], name='Similarity', hovertemplate='%{x} %{y}'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df_cosine.index, y=df_cosine['Cosine Similarity'], name='Cosine Similarity', hovertemplate='%{x} %{y}'),
        row=2, col=1
    )

    for y_line in range(0, 101, 2):
        fig.add_shape(type="line",
                      x0=0, y0=y_line, x1=1, y1=y_line,
                      xref="paper", yref="y",
                      line=dict(color="gray", width=0.5),
                      row=1, col=1)
    #    fig.add_shape(type="line",
    #                  x0=0, y0=y_line, x1=1, y1=y_line,
    #                  xref="paper", yref="y",
    #                  line=dict(color="gray", width=0.5),
    #                  row=2, col=1)

    # Add mean lines to the plot
    fig.add_shape(type="line",
                  y0=df['Similarity'][:-1].mean(), y1=df['Similarity'][:-1].mean(),
                  xref="paper", x0=0, x1=1,
                  line=dict(color="Red", width=2), row=1, col=1)
    fig.add_annotation(text="Mean Similarity", x=0.5, y=df['Similarity'][:-1].mean(),
                       xref="paper", xanchor="center", yanchor="bottom",
                       showarrow=False, font=dict(color="Red"), row=1, col=1)

    fig.add_shape(type="line",
                  y0=df['Cosine Similarity'][:-1].mean(), y1=df['Cosine Similarity'][:-1].mean(),
                  xref="paper", x0=0, x1=1,
                  line=dict(color="Red", width=2), row=2, col=1)
    fig.add_annotation(text="Mean Cosine Similarity", x=0.5, y=df['Cosine Similarity'][:-1].mean(),
                       xref="paper", xanchor="center", yanchor="bottom",
                       showarrow=False, font=dict(color="Red"), row=2, col=1)

    # Increase the size of the plot
    fig.update_layout(height=800, width=1000, title_text="Layer Similarities",
                      legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=1),
                      hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
                      xaxis=dict(tickangle=-45))

    fig.update_yaxes(autorange=True, row=2, col=1) # update the yaxes to be automatic for the second plot (Cosine Similarity)

    fig.write_html(f'{filename}.html'.replace('.csv', ''))

    return fig.to_html(include_plotlyjs='cdn')

csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

iface = gr.Interface(fn=plot_similarity,
                     inputs=gr.inputs.Dropdown(choices=csv_files),
                     outputs=gr.outputs.HTML())

iface.launch()
