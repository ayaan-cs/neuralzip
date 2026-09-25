# Visualizer artboards

Sources of the Claude Design canvas that accompanies the project:
https://claude.ai/artifact/YbhHTJnx1YDDmvtYYfgN6g

* `Main.dc.html` — live byte-by-byte replay: text heat-mapped by bits paid, each
  predictor's top-6 guesses, live mixer weights, play/step/speed controls.
* `Curve.dc.html` — per-10 KB learning curve of each expert and the mixture vs gzip.
* `Pipeline.dc.html` — the pipeline and the key equations.
* `canvas.json` — board layout, for all four boards.

The canvas's fourth board, the animated five-scene walkthrough, lives in
`../playback/Story.dc.html` with its own notes.

The interactive boards fetch `trace.json` (produced by `python trace.py` at the
repo root) as an uploaded asset of the canvas; the `/_blob/...` URL in the
sources refers to that upload and only resolves inside the artifact.

The committed `trace.json` traces the two-expert mixture, i.e.
`python trace.py --model nz-nomatch`, and the boards are written for those two
experts: `Main.dc.html`, `Curve.dc.html` and `../playback/Story.dc.html` each
carry a `titles`/`names` and `colors` map keyed by expert name. Regenerating
the trace for the current headline model
(`python trace.py`, which now also includes the match model) therefore needs a
`match` entry added to both maps, and the new file re-uploaded to the canvas
with the `/_blob/...` URLs repointed at it.
