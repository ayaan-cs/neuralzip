# Playback — the animated walkthrough

`Story.dc.html` is a self-driving, five-scene explanation of what this
compressor does, played against a real trace rather than an illustration.
It is the fourth board of the project canvas:
https://claude.ai/artifact/YbhHTJnx1YDDmvtYYfgN6g

| scene | what it shows |
|---|---|
| 1. Every byte is a bet | Bytes stream in heat-mapped by what they cost; the mixture's top guesses, the byte that actually arrived, and the −log₂ p charged for it |
| 2. Two experts, wrong in different ways | The context model's and the GRU's guesses for the same byte, side by side, with each one's p(actual) |
| 3. The mixer learns who to trust | The weights moving byte by byte, the bits each expert would have charged, and how often the mixture beat both |
| 4. Bits are additive | The running bill against what gzip −9 pays for the same bytes |
| 5. Half a megabyte later | The per-10 KB learning curve drawing itself, ending on the final figures |

It autoplays, has play/pause and per-scene jumps, keeps its position across
reloads, and exposes a playback-speed tweak.

## Data

Every number on the board comes from `trace.json` at the repo root — no
invented figures. The board tries the canvas's uploaded copy
(`/_blob/3a4b03…`) first and falls back to `./trace.json`, so it also works
from a plain directory containing both files.

That trace is the **two-expert** mixture (`python trace.py --model nz-nomatch`),
which is what `nz` was when the board was made. Regenerating it for the current
three-expert `nz` means adding a `match` entry to this board's `colors` and
`names` maps, and to the same maps in `../visualizer/Main.dc.html` and
`../visualizer/Curve.dc.html`, then re-uploading the file to the canvas.

## How it was made

This board was designed in an **AI design workspace** (Claude Design) rather
than hand-written: the structure, motion and scene pacing were iterated
visually on a canvas and then exported here as source. The intent is the thing
prose in the README cannot do — showing the mechanism *moving*, one byte at a
time, so that "compression is prediction" is something you watch happen rather
than something you take on trust.

## Running it

The `.dc.html` format is a Design Component: markup plus a logic class,
rendered by the canvas runtime (`support.js`, supplied by the canvas, not
vendored here). Open the artboard link above to view it. This directory holds
the source so the board is versioned, reviewable and reproducible alongside the
code it describes.
