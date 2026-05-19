# Gradio Spaces' hot-reload picks up the ``demo`` Blocks but only honours
# theme/css when they're passed to ``launch()``. We pull both from the
# miro-t2i package and inject them here so the Space matches the local
# look.
import gradio as gr

from miro.app import app as demo, CSS

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
    theme=gr.themes.Default(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="md",
    ),
    css=CSS,
)
