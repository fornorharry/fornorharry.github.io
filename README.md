# Littleball91 · Luau Systems Portfolio

Live at **https://fornorharry.github.io/**

Roblox game systems in strict-typed Luau (combat, player data and saving, economy and monetization), each tested in Studio and reviewed as an exploiter would, plus Blender work.

## In this repo

- `index.html`: the portfolio page. One self-contained file with no build step.
- `renders/`: the Blender renders shown on the page, the matching clay and wireframe renders (`*-wire.jpg`), and `stats.json` with each scene's triangle count and render time.
- `blender/build_scenes.py`: the script that builds and renders all three scenes. Run it with `blender -b --factory-startup --python blender/build_scenes.py -- renders blender`.
- `blender/*.blend`: the saved scenes (Blender 5.1).