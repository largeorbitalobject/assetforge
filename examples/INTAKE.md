# Example intake — test corpus for the AssetForge pipeline

Six examples from my own image generations (pixel-art style, made with Midjourney), in `examples/inputs/`. Each routes to a
pipeline branch by `--type`:

    assetforge run examples/inputs/<file> --type <type>

| File | Content | Type | Branch / readiness |
|---|---|---|---|
| object1.webp | carved stone face-block | object | mesh→texture→export (proven) |
| object2.webp | skull guitar pedal (knobs, footswitch) | object | object now; mechanical later (knobs=revolute) |
| object3.webp | retro CRT computer workstation | object | object now |
| character1.webp | yellow humanoid swordsman | creature | mesh+texture OK; rig(organic) M3; animate M4; sword=equipment |
| landscape1.webp | bioluminescent coral garden | scene | HunyuanWorld branch (weights cached 4.8GB; wiring) |
| landscape-building1.webp | white temple in a garden ("place") | scene | HunyuanWorld branch |

Notes: character + landscapes are full scenes — character needs rembg isolation to the figure;
scenes use HunyuanWorld (expects a whole scene). object2/object3 are good `mechanical` M2 test cases.
