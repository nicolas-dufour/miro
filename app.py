"""Gradio demo for MIRO.

Run locally:
    pip install miro-t2i[demo]
    python app.py

Deploy to a HuggingFace Space (Gradio SDK) by pushing this file + a
``requirements.txt`` containing ``miro-t2i[demo]``.
"""
from __future__ import annotations

import os
import random

import gradio as gr
import torch
from PIL import Image

from miro import MiroPipeline

# HuggingFace Spaces ZeroGPU: allocates the GPU on-demand per ``@spaces.GPU``
# call. Outside Spaces the import fails — the ``_gpu_decorator`` below
# degrades to a no-op and ``generate`` just uses whatever device is local.
try:
    import spaces  # type: ignore
except ImportError:
    spaces = None

REPO = os.environ.get("MIRO_REPO", "nicolas-dufour/miro")
ZERO_GPU = spaces is not None
DEVICE = "cuda" if (torch.cuda.is_available() and not ZERO_GPU) else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
STEP = 1 / 64  # match the 64-bin coherence quantisation used during training

# (key, short_label, color)  — colors are the per-reward palette from the
# project page (assets/js/ui/image-carousels.js).
REWARDS = [
    ("clip_score",         "CLIP",        "#4fd2c8"),
    ("aesthetic_score",    "Aesthetic",   "#69c869"),
    ("image_reward_score", "ImageReward", "#ff6978"),
    ("pick_a_score_score", "PickScore",   "#ff96c8"),
    ("hpsv2_score",        "HPSv2",       "#6496ff"),
    ("vqa_score",          "VQA",         "#e1d34f"),
    ("sciscore_score",     "SciScore",    "#c769e6"),
]
N = len(REWARDS)

EXAMPLES = [
    "Photography closeup portrait of an adorable rusty broken­down steampunk "
    "robot covered in budding vegetation, surrounded by tall grass, misty "
    "futuristic sci­fi forest environment.",
    "An armchair in the shape of an avocado.",
    "A rocket painted on a brick wall.",
    "Robots meditating on top of a skyscraper.",
    "The Eiffel tower made of French fries.",
    "An oil painting of rain at a traditional Chinese town.",
    "A cute chonky cat looking angry at you while laying on your laptop.",
    "A realistic photo of a castle at dawn.",
    "A hedgehog using a calculator.",
    "A man with an apple instead of a head.",
    "An elephant under the sea.",
    "Lego Arnold Schwarzenegger.",
    "A magnifying glass over a page of a 1950s batman comic.",
    "A pencil sketch of a woman's face.",
    "A frog drinking coffee, fancy digital art.",
    "A teddy bear wearing a blue ribbon taking a selfie in a small boat in the "
    "center of a lake.",
    "A groundhog wearing a straw hat stands on top of a table.",
    "A cute little matte low poly isometric cherry blossom forest island, "
    "waterfalls, lighting, soft shadows, trending on Artstation, 3d render, "
    "monument valley, fez video game.",
    "Abraham Lincoln in the GoldenEye 007 video game for Nintendo64 released in 1997.",
    "A still of a group of wild animals including giraffes and lions in the "
    "movie The Matrix from 1999.",
    "Cartoon characters, mini characters, hand-made, illustrations, robot kids, "
    "color expressions, boy, short brown hair, curly hair, blue eyes, "
    "technological age, cyberpunk, big eyes, cute, mini, detailed light and "
    "shadow, high detail.",
    "Five dogs on the street.",
]

print(f"Loading {REPO} (zero_gpu={ZERO_GPU}, device={DEVICE}, dtype={DTYPE}) …", flush=True)
pipe = MiroPipeline.from_pretrained(REPO)
if not ZERO_GPU:
    pipe = pipe.to(DEVICE, DTYPE)
print(f"  coherence_keys = {pipe.coherence_keys}", flush=True)


def _gpu_decorator(func):
    """Wrap GPU-using callable with ``spaces.GPU(...)`` when on HF Spaces.

    Outside Spaces, this is the identity decorator.
    """
    if spaces is not None:
        return spaces.GPU(duration=60)(func)
    return func


def _make_grid(images: list[Image.Image], gap: int = 4, bg=(245, 245, 245)) -> Image.Image:
    """Compose 1-4 images into a single PIL canvas.

    1 → single image, 2 → 1×2 horizontal, 3 or 4 → 2×2 (slot 4 blank for n=3).
    """
    n = len(images)
    if n == 1:
        return images[0]
    w, h = images[0].size
    if n == 2:
        canvas = Image.new("RGB", (2 * w + gap, h), bg)
        canvas.paste(images[0], (0, 0))
        canvas.paste(images[1], (w + gap, 0))
        return canvas
    canvas = Image.new("RGB", (2 * w + gap, 2 * h + gap), bg)
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (w + gap, 0))
    canvas.paste(images[2], (0, h + gap))
    if n >= 4:
        canvas.paste(images[3], (w + gap, h + gap))
    return canvas


@_gpu_decorator
def generate(prompt, num_steps, guidance, seed, num_images, *slider_values):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    pos_values = slider_values[:N]
    neg_values = slider_values[N:]
    reward_targets = {k[0]: v for k, v in zip(REWARDS, pos_values)}
    negative_reward_targets = {k[0]: v for k, v in zip(REWARDS, neg_values)}
    if int(seed) < 0:
        seed = random.randint(0, 2**31 - 1)

    # On ZeroGPU the pipeline lives on CPU at module load; move to CUDA the
    # first time we run inside a ``@spaces.GPU`` slot.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if next(pipe.network.parameters()).device.type != device:
        pipe.to(device, DTYPE)

    gen = torch.Generator(device).manual_seed(int(seed))
    n = int(num_images)
    images = pipe(
        prompt,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance),
        num_images_per_prompt=n,
        reward_targets=reward_targets,
        negative_reward_targets=negative_reward_targets,
        generator=gen,
    )
    # Compose all samples into one PNG so the download button can serve a
    # single file alongside the individually-clickable gallery tiles.
    grid_path = "/tmp/miro_grid.png"
    _make_grid(images).save(grid_path)

    # Choose column count to keep the layout balanced:
    #   1 → 1 col, 2 → 2 col, 3 → 3 col (single centred row), 4 → 2 col (2×2)
    cols = {1: 1, 2: 2, 3: 3, 4: 2}.get(n, 2)
    return (
        gr.update(value=images, columns=cols),
        gr.DownloadButton(value=grid_path, visible=True),
    )


def reset_positive():
    return [1.0] * N


def reset_negative():
    return [0.0] * N


# Per-reward gradient styling. Targets gradio 6's actual slider DOM:
# ``<input type='range'>`` with svelte-hashed classes. Gradio paints the
# track via ``::-webkit-slider-runnable-track`` and the variable
# ``--slider-color`` for the filled portion; we override that pseudo-element
# entirely with a full-width gradient so the bar reads as a single colored
# strip per reward (and the same color sits behind the positive *and*
# negative slider — the "prolongation" the user asked for).
_per_reward_css = []
for key, _, color in REWARDS:
    _per_reward_css.append(f"""
.reward-{key} {{ --slider-color: {color}; }}
.reward-{key} input[type='range']::-webkit-slider-runnable-track {{
  background: linear-gradient(to right, var(--neutral-200, #e5e7eb), {color}) !important;
  height: 8px !important;
  border-radius: 4px !important;
}}
.reward-{key} input[type='range']::-moz-range-track {{
  background: linear-gradient(to right, var(--neutral-200, #e5e7eb), {color}) !important;
  height: 8px !important;
  border-radius: 4px !important;
}}
.reward-{key} input[type='range']::-moz-range-progress {{
  background: transparent !important;
}}
.reward-{key} input[type='range']::-webkit-slider-thumb {{
  background-color: {color} !important;
  border: 2px solid white !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
}}
.reward-{key} input[type='range']::-moz-range-thumb {{
  background-color: {color} !important;
  border: 2px solid white !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
}}
""")

CSS = """
.gradio-container {
  max-width: 1280px !important;
  margin: auto;
  padding-top: 8px !important;
  background: var(--background-fill-primary);
}

/* ── Header ─────────────────────────────────────────────────────────────── */
#header {
  text-align: center;
  padding: 10px 16px 14px;
  border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 12px;
}
#header h1 {
  font-size: 28px;
  margin: 0 14px 0 0;
  display: inline-block;
  vertical-align: middle;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #f97316, #ec4899, #8b5cf6);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
}
#header span.sub {
  font-size: 13px;
  color: var(--body-text-color-subdued);
  vertical-align: middle;
  font-weight: 500;
}
#badges { display: inline-block; margin-left: 12px; vertical-align: middle; }
#badges a {
  display: inline-block;
  margin: 0 3px;
  padding: 3px 11px;
  border-radius: 999px;
  background: var(--background-fill-secondary);
  color: var(--body-text-color);
  text-decoration: none;
  font-size: 12px;
  border: 1px solid var(--border-color-primary);
  transition: all .15s ease;
}
#badges a:hover {
  transform: translateY(-1px);
  border-color: var(--body-text-color);
  box-shadow: 0 2px 6px rgba(0,0,0,.06);
}

/* ── Section labels (inline, no card) ───────────────────────────────────── */
.section-title {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--body-text-color-subdued);
  margin: 6px 0 4px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.section-title::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border-color-primary), transparent);
}

/* ── Prompt + Generate row ──────────────────────────────────────────────── */
#prompt_box label > span:first-child { display: none !important; }   /* section title replaces it */
#prompt_box textarea {
  font-size: 15px !important;
  line-height: 1.5 !important;
  border-radius: 12px !important;
  border: 1px solid var(--border-color-primary) !important;
  padding: 10px 14px !important;
  background: var(--background-fill-primary) !important;
  transition: all .15s ease;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
#prompt_box textarea::placeholder {
  color: var(--body-text-color-subdued);
  font-style: italic;
}
#prompt_box textarea:focus {
  border-color: #8b5cf6 !important;
  box-shadow: 0 0 0 3px rgba(139, 92, 246, .15) !important;
  outline: none !important;
}

#generate_btn {
  height: 100% !important;
  min-height: 64px !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  white-space: pre-line;
  background: linear-gradient(135deg, #f97316 0%, #ec4899 60%, #8b5cf6 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  letter-spacing: .5px;
  text-transform: uppercase;
  box-shadow: 0 4px 14px rgba(236, 72, 153, .25);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
#generate_btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(236, 72, 153, .4);
  filter: brightness(1.05);
}
#generate_btn:active { transform: translateY(0); }

/* ── Sampling controls row (steps / cfg / seed) ─────────────────────────── */
.controls-row { align-items: flex-start !important; gap: 14px !important; }
.controls-row label > span:first-child {
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--body-text-color-subdued);
  text-transform: uppercase;
  letter-spacing: .6px;
}
/* Force gradio's slider "head" row (label + value box) to a fixed,
   non-wrapping height so all three sliders' value boxes line up. */
.sampling-slider .head {
  flex-wrap: nowrap !important;
  align-items: center !important;
  min-height: 22px !important;
  margin-bottom: 4px !important;
}
.sampling-slider .head label,
.sampling-slider .head > span { white-space: nowrap !important; }
.sampling-slider .slider_input_container {
  align-items: center !important;
  gap: 10px !important;
  height: 28px !important;
}
/* Numeric value box on the right of each slider */
.sampling-slider .tab-like-container {
  height: 26px !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 8px !important;
  background: var(--background-fill-secondary) !important;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0, 0, 0, .04);
  transition: border-color .15s ease, box-shadow .15s ease;
}
.sampling-slider .tab-like-container:focus-within {
  border-color: #64748b !important;
  box-shadow: 0 0 0 2px rgba(100, 116, 139, .15) !important;
}
.sampling-slider input[type='number'] {
  height: 24px !important;
  min-width: 56px !important;
  padding: 0 8px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  text-align: center !important;
  border: none !important;
  border-radius: 0 !important;
  background: transparent !important;
  color: var(--body-text-color) !important;
}
.sampling-slider input[type='number']:focus {
  box-shadow: none !important; outline: none !important;
}
.sampling-slider .reset-button {
  background: var(--background-fill-primary) !important;
  color: var(--body-text-color-subdued) !important;
  border-left: 1px solid var(--border-color-primary) !important;
  font-size: 11px !important;
  padding: 0 6px !important;
  transition: background-color .15s ease, color .15s ease;
}
.sampling-slider .reset-button:hover:not(:disabled) {
  background: var(--background-fill-secondary) !important;
  color: var(--body-text-color) !important;
}

/* ── Mobile layout: image above the controls (gradio's default puts it last) */
@media (max-width: 900px) {
  #main_row { flex-direction: column !important; }
  #image_col   { order: 0 !important; }
  #control_col { order: 1 !important; }
  #output_image { max-width: 100% !important; }
}
/* Neutral gradient for the 3 sampling sliders, matching the reward style */
.sampling-slider input[type='range']::-webkit-slider-runnable-track {
  background: linear-gradient(to right, var(--neutral-200, #e5e7eb), #64748b) !important;
  height: 8px !important;
  border-radius: 4px !important;
}
.sampling-slider input[type='range']::-moz-range-track {
  background: linear-gradient(to right, var(--neutral-200, #e5e7eb), #64748b) !important;
  height: 8px !important;
  border-radius: 4px !important;
}
.sampling-slider input[type='range']::-moz-range-progress {
  background: transparent !important;
}
.sampling-slider input[type='range']::-webkit-slider-thumb {
  background-color: #475569 !important;
  border: 2px solid white !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
}
.sampling-slider input[type='range']::-moz-range-thumb {
  background-color: #475569 !important;
  border: 2px solid white !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
}

/* ── Reward columns ─────────────────────────────────────────────────────── */
.reward-col { padding: 0 6px; }
.reward-col-header {
  text-align: center;
  font-weight: 600;
  font-size: 12px;
  padding: 3px 0 6px;
  color: var(--body-text-color-subdued);
  text-transform: uppercase;
  letter-spacing: .8px;
  border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 6px;
}
.reward-col .gradio-slider { margin: 3px 0 !important; }
.reward-col label > span:first-child { font-size: 12px !important; font-weight: 500; }

/* Reset buttons — subtle, text-style */
.reward-col button {
  background: transparent !important;
  border: 1px dashed var(--border-color-primary) !important;
  color: var(--body-text-color-subdued) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  margin-top: 6px !important;
  transition: all .15s ease;
}
.reward-col button:hover {
  background: var(--background-fill-secondary) !important;
  border-style: solid !important;
  color: var(--body-text-color) !important;
}

/* ── Output gallery ─────────────────────────────────────────────────────── */
#output_gallery {
  border-radius: 14px !important;
  background: linear-gradient(135deg, var(--background-fill-secondary), var(--background-fill-primary));
  box-shadow: 0 4px 20px rgba(0, 0, 0, .06);
  border: 1px solid var(--border-color-primary);
  padding: 6px !important;
}
/* Let the gallery shrink/grow to whatever the cells actually need — the
   gradio defaults of min-height: var(--size-80) / max-height: 55vh combined
   with grid-auto-rows: 1fr stretch cells taller than wide, producing the
   "vertical borders" letterboxing on our square outputs. */
#output_gallery .fixed-height,
#output_gallery .gallery-container {
  min-height: 0 !important;
  max-height: none !important;
  height: auto !important;
}
#output_gallery .grid-container {
  grid-auto-rows: auto !important;   /* row height = cell aspect-ratio */
}
#output_gallery .thumbnail-item {
  aspect-ratio: 1 / 1 !important;
  border-radius: 10px !important;
  overflow: hidden;
  background: var(--background-fill-secondary) !important;
}
#output_gallery .thumbnail-item img,
#output_gallery img {
  object-fit: contain !important;
  width: 100% !important;
  height: 100% !important;
  border-radius: 10px !important;
  background: var(--background-fill-secondary);
}
#output_gallery label { display: none !important; }

#download_btn {
  margin-top: 8px;
  background: var(--background-fill-secondary) !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 8px !important;
  font-size: 12px !important;
  color: var(--body-text-color) !important;
}
#download_btn:hover {
  border-color: #ec4899 !important;
  color: #ec4899 !important;
}

/* Examples accordion: tighter */
.gradio-accordion { border-radius: 12px !important; margin-top: 4px; }
.gradio-accordion > .label-wrap { padding: 6px 12px !important; }

footer { display: none !important; }
""" + "\n".join(_per_reward_css)


with gr.Blocks(title="MIRO — reward-conditioned T2I") as app:
    gr.HTML(
        """
        <div id="header">
          <h1>MIRO</h1>
          <span class="sub">multi-reward conditioned T2I · ICML 2026</span>
          <span id="badges">
            <a href="https://arxiv.org/abs/2510.25897" target="_blank">📄</a>
            <a href="https://nicolas-dufour.github.io/miro/" target="_blank">🌐</a>
            <a href="https://github.com/nicolas-dufour/miro" target="_blank">💻</a>
            <a href="https://huggingface.co/nicolas-dufour/miro" target="_blank">🤗</a>
            <a href="https://pypi.org/project/miro-t2i/" target="_blank">🐍</a>
          </span>
        </div>
        """
    )

    with gr.Row(equal_height=True, elem_id="main_row"):
        # ─── Controls ──────────────────────────────────────────────────────────
        with gr.Column(scale=5, elem_id="control_col"):
            gr.HTML('<div class="section-title">✍️ Prompt</div>')
            with gr.Row(equal_height=True):
                prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Describe what to generate…",
                    lines=2, value=EXAMPLES[0], scale=4,
                    elem_id="prompt_box",
                    container=False,
                )
                btn = gr.Button(
                    "Generate\n4 images", variant="primary",
                    elem_id="generate_btn", scale=1, min_width=120,
                )

            gr.HTML('<div class="section-title">⚙️ Sampling</div>')
            with gr.Row(elem_classes="controls-row", equal_height=True):
                num_steps  = gr.Slider(10, 100, value=50, step=1,
                                        label="Inference steps",
                                        elem_classes="sampling-slider")
                guidance   = gr.Slider(1.0, 15.0, value=7.0, step=0.5,
                                        label="CFG scale",
                                        elem_classes="sampling-slider")
                num_images = gr.Slider(1, 4, value=4, step=1,
                                        label="Images",
                                        elem_classes="sampling-slider")
                seed       = gr.Slider(0, 2**31 - 1, value=3407, step=1,
                                        label="Seed",
                                        elem_classes="sampling-slider",
                                        randomize=False)

            gr.HTML('<div class="section-title">🎚️ Reward axes</div>')
            pos_sliders: list[gr.Slider] = []
            neg_sliders: list[gr.Slider] = []
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="reward-col", scale=1, min_width=160):
                    gr.HTML('<div class="reward-col-header">Positive (1.0)</div>')
                    for key, label, _ in REWARDS:
                        pos_sliders.append(
                            gr.Slider(0.0, 1.0, value=1.0, step=STEP, label=label,
                                      elem_classes=[f"reward-{key}"])
                        )
                    reset_pos_btn = gr.Button("Reset → 1.0", size="sm")
                with gr.Column(elem_classes="reward-col", scale=1, min_width=160):
                    gr.HTML('<div class="reward-col-header">Negative (0.0)</div>')
                    for key, label, _ in REWARDS:
                        neg_sliders.append(
                            gr.Slider(0.0, 1.0, value=0.0, step=STEP, label=label,
                                      elem_classes=[f"reward-{key}"])
                        )
                    reset_neg_btn = gr.Button("Reset → 0.0", size="sm")

        # ─── Output ────────────────────────────────────────────────────────────
        with gr.Column(scale=5, elem_id="image_col"):
            # Gallery → each generated sample is its own tile, so users can
            # right-click / drag any single image. A separate download button
            # below serves the composed 2×2 (or 1×N) grid as one PNG.
            output = gr.Gallery(
                show_label=False,
                elem_id="output_gallery",
                columns=2,
                object_fit="contain",
                allow_preview=True,
                preview=False,
                interactive=False,
            )
            download_btn = gr.DownloadButton(
                "📥 Download all as one image",
                visible=False,
                size="sm",
                elem_id="download_btn",
            )
            with gr.Accordion("Example prompts", open=True):
                gr.Examples(
                    examples=[[e] for e in EXAMPLES],
                    inputs=[prompt],
                    examples_per_page=len(EXAMPLES),   # no pagination
                )

    btn.click(
        generate,
        inputs=[prompt, num_steps, guidance, seed, num_images, *pos_sliders, *neg_sliders],
        outputs=[output, download_btn],
    )
    reset_pos_btn.click(reset_positive, inputs=None, outputs=pos_sliders)
    reset_neg_btn.click(reset_negative, inputs=None, outputs=neg_sliders)


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"}
    app.launch(
        share=share,
        server_name="0.0.0.0",
        theme=gr.themes.Default(
            primary_hue="slate",
            secondary_hue="slate",
            neutral_hue="slate",
            radius_size="md",
        ),
        css=CSS,
    )
