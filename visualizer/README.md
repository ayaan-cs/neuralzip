# Visualizer artboards

Sources of the three-board Claude Design canvas that accompanies the project:
https://claude.ai/artifact/YbhHTJnx1YDDmvtYYfgN6g

* `Main.dc.html` — live byte-by-byte replay: text heat-mapped by bits paid, each
  predictor's top-6 guesses, live mixer weights, play/step/speed controls.
* `Curve.dc.html` — per-10 KB learning curve of each expert and the mixture vs gzip.
* `Pipeline.dc.html` — the pipeline and the key equations.
* `canvas.json` — board layout.

The interactive boards fetch `trace.json` (produced by `python trace.py` at the
repo root) as an uploaded asset of the canvas; the `/_blob/...` URL in the
sources refers to that upload and only resolves inside the artifact.
