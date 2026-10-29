"""
Builds and renders the portfolio's Blender scenes: the three shop items from the economy
system's Catalog (Iron Sword, Golden Crown, Health Potion), modelled procedurally and
rendered in Cycles.

Run from a terminal (no Blender window needed):
    blender -b --factory-startup --python build_scenes.py -- <render_dir> <blend_dir> [preview] [sword,crown,potion]

"preview" renders small and fast to check composition. Without it, each scene renders at
1600x1000 and its .blend file is saved to <blend_dir>. Triangle counts and render times
are written to <render_dir>/stats.json.
"""

import bmesh
import bpy
import json
import math
import os
import random
import sys
import time

from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
RENDER_DIR = os.path.abspath(argv[0]) if len(argv) > 0 else os.path.abspath("renders")
BLEND_DIR = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.abspath("blend")
PREVIEW = len(argv) > 2 and argv[2] == "preview"
ONLY = set(argv[3].split(",")) if len(argv) > 3 and argv[3] else None

# Marks the objects that make up the asset itself, so the stats don't count the
# backdrop, the lights or the floating dust.
ASSET_KEY = "portfolio_asset"


# Scene setup ------------------------------------------------------------------------


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def use_gpu(scene):
    """Renders on the GPU when one is available, otherwise falls back to the CPU."""
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return kind
    scene.cycles.device = "CPU"
    return "CPU"


def render_settings(scene):
    device = use_gpu(scene)
    cycles = scene.cycles
    cycles.samples = 16 if PREVIEW else 256
    cycles.use_adaptive_sampling = True
    cycles.use_denoising = True
    try:
        cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    cycles.max_bounces = 12
    cycles.transmission_bounces = 12
    cycles.glossy_bounces = 6
    render = scene.render
    render.resolution_x, render.resolution_y = (640, 400) if PREVIEW else (1600, 1000)
    render.resolution_percentage = 100
    render.image_settings.file_format = "JPEG"
    render.image_settings.quality = 92
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    for look in ("AgX - Punchy", "Punchy", "AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    return device


def world(scene, color, strength=1.0):
    w = bpy.data.worlds.new("World")
    scene.world = w
    try:
        w.use_nodes = True
    except AttributeError:
        pass
    background = w.node_tree.nodes.get("Background") if w.node_tree else None
    if background:
        background.inputs["Color"].default_value = (*color, 1)
        background.inputs["Strength"].default_value = strength
    else:
        w.color = color


def material(name, color, metallic=0.0, rough=0.5, transmission=0.0, ior=1.45,
             emission=None, strength=0.0, coat=0.0, sheen=0.0, sheen_tint=None):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    p = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")

    def put(key, value):
        if key in p.inputs:
            p.inputs[key].default_value = value

    put("Base Color", (*color, 1))
    put("Metallic", metallic)
    put("Roughness", rough)
    put("Transmission Weight", transmission)
    put("IOR", ior)
    put("Coat Weight", coat)
    put("Sheen Weight", sheen)
    if sheen_tint:
        put("Sheen Tint", (*sheen_tint, 1))
    if emission:
        put("Emission Color", (*emission, 1))
        put("Emission Strength", strength)
    m.diffuse_color = (*color, 1)
    return m


# Geometry helpers ---------------------------------------------------------------------


def active():
    return bpy.context.active_object


def finish(ob, mat=None, smooth=False, asset=True, parent=None):
    if mat:
        ob.data.materials.append(mat)
    if smooth:
        for poly in ob.data.polygons:
            poly.use_smooth = True
    ob[ASSET_KEY] = asset
    if parent:
        ob.parent = parent
    return ob


def recalc(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()


def mesh_from(scene, name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    recalc(ob)
    return ob


def lathe(scene, name, profile, segments=32, cap_bottom=True, cap_top=False):
    """Spins an (radius, height) profile around Z, like a lathe or Blender's Screw modifier."""
    verts, faces, n = [], [], len(profile)
    for s in range(segments):
        a = 2 * math.pi * s / segments
        for r, z in profile:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for s in range(segments):
        s2 = (s + 1) % segments
        for i in range(n - 1):
            faces.append((s * n + i, s2 * n + i, s2 * n + i + 1, s * n + i + 1))
    if cap_bottom:
        c = len(verts)
        verts.append((0, 0, profile[0][1]))
        for s in range(segments):
            faces.append((c, ((s + 1) % segments) * n, s * n))
    if cap_top:
        c = len(verts)
        verts.append((0, 0, profile[-1][1]))
        for s in range(segments):
            faces.append((c, s * n + n - 1, ((s + 1) % segments) * n + n - 1))
    return mesh_from(scene, name, verts, faces)


def bevel(ob, width, segments=2):
    md = ob.modifiers.new("Bevel", "BEVEL")
    md.width = width
    md.segments = segments
    md.limit_method = "ANGLE"


def subsurf(ob, levels):
    md = ob.modifiers.new("Subdivision", "SUBSURF")
    md.levels = levels
    md.render_levels = levels


def solidify(ob, thickness):
    md = ob.modifiers.new("Solidify", "SOLIDIFY")
    md.thickness = thickness
    md.offset = 0


def rock(scene, loc, scale, seed, mat, subdiv=2, jitter=0.18, parent=None):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=1, location=loc)
    ob = active()
    rnd = random.Random(seed)
    for v in ob.data.vertices:
        v.co *= 1 + rnd.uniform(-jitter, jitter)
    ob.scale = scale
    ob.rotation_euler = (rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2), rnd.uniform(0, math.pi))
    return finish(ob, mat, parent=parent)


def backdrop(scene, color, rough=0.5):
    """A floor that sweeps up into a back wall, so the render has no horizon line."""
    profile = [(y, 0.0) for y in (-12, -4, -1, 0.5, 1.5)]
    radius = 2.5
    for i in range(1, 14):
        a = -math.pi / 2 + (math.pi / 2) * i / 13
        profile.append((1.5 + radius * math.cos(a), radius + radius * math.sin(a)))
    profile.append((1.5 + radius, 12.0))
    xs = (-14, 14)
    verts = [(x, y, z) for (y, z) in profile for x in xs]
    faces = [(i * 2, i * 2 + 1, i * 2 + 3, i * 2 + 2) for i in range(len(profile) - 1)]
    ob = mesh_from(scene, "Backdrop", verts, faces)
    return finish(ob, material("Backdrop", color, rough=rough), smooth=True, asset=False)


def dust(scene, count, center, spread, color, strength, seed, rmin=0.005, rmax=0.014):
    """Tiny glowing specks. Out of focus near the camera, they become soft bokeh."""
    rnd = random.Random(seed)
    m = material("Dust", color, emission=color, strength=strength)
    for _ in range(count):
        loc = tuple(center[i] + rnd.uniform(-spread[i], spread[i]) for i in range(3))
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=rnd.uniform(rmin, rmax), location=loc)
        finish(active(), m, asset=False)


def look_at(ob, target):
    ob.rotation_euler = (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()


def area_light(scene, name, loc, target, energy, size, color):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size, data.color = energy, size, color
    ob = bpy.data.objects.new(name, data)
    scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)


def point_light(scene, name, loc, energy, color, radius=0.05):
    data = bpy.data.lights.new(name, "POINT")
    data.energy, data.color, data.shadow_soft_size = energy, color, radius
    ob = bpy.data.objects.new(name, data)
    scene.collection.objects.link(ob)
    ob.location = loc
    # Light only: never seen as a hot spot in reflections or through glass.
    ob.visible_glossy = False
    ob.visible_transmission = False


def camera(scene, loc, target, lens, fstop):
    data = bpy.data.cameras.new("Camera")
    data.lens = lens
    ob = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    data.dof.use_dof = True
    data.dof.focus_distance = (Vector(target) - Vector(loc)).length
    data.dof.aperture_fstop = fstop
    scene.camera = ob


def pivot(scene, name):
    empty = bpy.data.objects.new(name, None)
    scene.collection.objects.link(empty)
    return empty


# Scenes -------------------------------------------------------------------------------


def scene_sword():
    scene = reset()
    device = render_settings(scene)
    world(scene, (0.008, 0.01, 0.016))
    backdrop(scene, (0.03, 0.035, 0.05), rough=0.55)

    stone = material("Stone", (0.15, 0.16, 0.18), rough=0.85)
    steel = material("Steel", (0.78, 0.8, 0.83), metallic=1, rough=0.2)
    gold = material("Gold", (1.0, 0.6, 0.18), metallic=1, rough=0.18)
    leather = material("Leather", (0.11, 0.055, 0.03), rough=0.65)
    rune = material("Rune", (0.2, 0.8, 1.0), emission=(0.25, 0.85, 1.0), strength=10)

    # The stone it's stuck in, plus rubble.
    rock(scene, (0, 0, 0.26), (0.95, 0.82, 0.46), 7, stone)
    rock(scene, (0.95, -0.4, 0.08), (0.2, 0.18, 0.13), 11, stone, subdiv=1, jitter=0.25)
    rock(scene, (-0.95, -0.45, 0.06), (0.15, 0.14, 0.1), 13, stone, subdiv=1, jitter=0.25)
    rock(scene, (-0.55, 0.65, 0.07), (0.19, 0.17, 0.12), 17, stone, subdiv=1, jitter=0.25)

    # The sword is built tip-down at the origin, then tilted into the stone by its pivot.
    sword = pivot(scene, "Sword")
    w, t = 0.075, 0.02
    blade = mesh_from(scene, "Blade", [
        (0, 0, 0),
        (w, 0, 0.18), (0, t, 0.18), (-w, 0, 0.18), (0, -t, 0.18),
        (w * 0.86, 0, 0.95), (0, t, 0.95), (-w * 0.86, 0, 0.95), (0, -t, 0.95),
    ], [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1), (5, 8, 7, 6),
    ])
    finish(blade, steel, parent=sword)
    for side in (1, -1):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, side * t, 0.56))
        strip = active()
        strip.scale = (0.009, 0.006, 0.56)
        finish(strip, rune, parent=sword)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.98))
    guard = active()
    guard.scale = (0.56, 0.11, 0.08)
    bevel(guard, 0.012)
    finish(guard, gold, parent=sword)
    for side in (1, -1):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.032, location=(0, side * 0.056, 0.98))
        finish(active(), rune, parent=sword)
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=0.03, depth=0.3, location=(0, 0, 1.17))
    finish(active(), leather, parent=sword)
    for z in (1.06, 1.13, 1.2, 1.27):
        bpy.ops.mesh.primitive_torus_add(major_radius=0.031, minor_radius=0.006, major_segments=12,
                                         minor_segments=6, location=(0, 0, z))
        finish(active(), leather, smooth=True, parent=sword)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.055, location=(0, 0, 1.36))
    finish(active(), gold, parent=sword)
    sword.location = (0.02, 0.0, 0.42)
    sword.rotation_euler = (math.radians(5), math.radians(-9), math.radians(22))

    dust(scene, 55, (0, 0, 1.0), (2.4, 1.8, 1.1), (1.0, 0.7, 0.4), 6, seed=3)
    area_light(scene, "Key", (-2.4, -2.2, 2.6), (0, 0, 0.9), 240, 2.2, (1.0, 0.82, 0.65))
    area_light(scene, "Rim", (1.6, 2.4, 2.2), (0, 0, 0.9), 520, 1.0, (0.45, 0.65, 1.0))
    area_light(scene, "Fill", (2.6, -2.4, 0.9), (0, 0, 0.8), 50, 3.0, (0.55, 0.65, 0.85))
    # A big soft card above the camera, mostly for the metal to reflect.
    area_light(scene, "Card", (0.6, -3.6, 2.9), (0, 0, 1.0), 90, 3.5, (1.0, 0.97, 0.93))
    camera(scene, (1.8, -5.0, 1.3), (0, 0, 0.92), 55, 2.8)
    return scene, device


def scene_crown():
    scene = reset()
    device = render_settings(scene)
    world(scene, (0.015, 0.012, 0.022))
    backdrop(scene, (0.035, 0.022, 0.05), rough=0.5)

    gold = material("Gold", (1.0, 0.72, 0.3), metallic=1, rough=0.15)
    velvet = material("Velvet", (0.2, 0.005, 0.025), rough=0.75, sheen=0.6, sheen_tint=(1.0, 0.25, 0.35))
    cap_velvet = material("Cap Velvet", (0.14, 0.004, 0.02), rough=0.8, sheen=0.6, sheen_tint=(1.0, 0.25, 0.35))
    pearl = material("Pearl", (0.95, 0.93, 0.88), rough=0.15, coat=1)
    ruby = material("Ruby", (0.9, 0.04, 0.1), rough=0.04, transmission=0.8, ior=1.77,
                    emission=(1.0, 0.05, 0.12), strength=0.8)
    sapphire = material("Sapphire", (0.05, 0.2, 0.9), rough=0.04, transmission=0.8, ior=1.77,
                        emission=(0.1, 0.3, 1.0), strength=0.8)

    # Cushion: a bevelled box smoothed into something soft, with a tassel at each corner.
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0.2))
    cushion = active()
    cushion.scale = (0.78, 0.78, 0.2)
    # Apply the scale first, so the bevel rounds evenly instead of squashing with the box.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel(cushion, 0.14, 4)
    subsurf(cushion, 2)
    finish(cushion, velvet, smooth=True)
    for sx in (1, -1):
        for sy in (1, -1):
            bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=0.035, depth=0.12, location=(sx * 0.72, sy * 0.72, 0.16),
                                            rotation=(math.pi, 0, 0))
            finish(active(), gold)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.025, location=(sx * 0.72, sy * 0.72, 0.23))
            finish(active(), gold, smooth=True)

    crown = pivot(scene, "Crown")
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.42, depth=0.16, location=(0, 0, 0.44),
                                        end_fill_type="NOTHING")
    band = active()
    solidify(band, 0.035)
    bevel(band, 0.008)
    finish(band, gold, smooth=True, parent=crown)
    for z in (0.36, 0.52):
        bpy.ops.mesh.primitive_torus_add(major_radius=0.425, minor_radius=0.02, major_segments=48,
                                         minor_segments=8, location=(0, 0, z))
        finish(active(), gold, smooth=True, parent=crown)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.4, location=(0, 0, 0.44))
    cap = active()
    cap.scale = (1, 1, 0.72)
    finish(cap, cap_velvet, smooth=True, parent=crown)
    for i in range(8):
        a = i * math.tau / 8
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.07, depth=0.26,
                                        location=(0.42 * math.cos(a), 0.42 * math.sin(a), 0.65),
                                        rotation=(0, 0, a + math.pi / 4))
        finish(active(), gold, parent=crown)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.034,
                                             location=(0.42 * math.cos(a), 0.42 * math.sin(a), 0.8))
        finish(active(), pearl, smooth=True, parent=crown)
        g = a + math.pi / 8
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.042,
                                              location=(0.452 * math.cos(g), 0.452 * math.sin(g), 0.44),
                                              rotation=(0, 0, g))
        gem = active()
        gem.scale = (0.55, 1, 1)
        finish(gem, ruby if i % 2 == 0 else sapphire, parent=crown)
    # Turn the crown so a ruby faces the camera.
    crown.rotation_euler = (0, 0, -math.pi / 2 - math.pi / 8)

    dust(scene, 45, (0, 0, 0.8), (2.0, 1.4, 0.9), (1.0, 0.75, 0.35), 7, seed=5, rmin=0.004, rmax=0.012)
    area_light(scene, "Key", (-2.1, -2.3, 2.7), (0, 0, 0.5), 320, 2.2, (1.0, 0.8, 0.62))
    area_light(scene, "Rim", (1.9, 2.2, 2.1), (0, 0, 0.5), 420, 1.0, (0.55, 0.65, 1.0))
    area_light(scene, "Top", (0.3, -0.4, 3.2), (0, 0, 0.5), 120, 1.6, (1.0, 0.95, 0.9))
    # A big soft card above the camera, so the gold has something bright to reflect.
    area_light(scene, "Card", (0.0, -3.2, 2.6), (0, 0, 0.5), 130, 3.5, (1.0, 0.97, 0.93))
    camera(scene, (0.0, -3.15, 1.4), (0, 0, 0.48), 55, 3.5)
    return scene, device


def scene_potion():
    scene = reset()
    device = render_settings(scene)
    world(scene, (0.01, 0.008, 0.008))
    backdrop(scene, (0.03, 0.02, 0.025), rough=0.28)

    glass = material("Glass", (0.95, 0.98, 1.0), rough=0.03, transmission=1.0, ior=1.45)
    # Kept dim on purpose: a bright red glow washes out to pink once it's tone-mapped.
    potion = material("Potion", (0.75, 0.0, 0.04), rough=0.05, transmission=0.85, ior=1.33,
                      emission=(1.0, 0.0, 0.04), strength=1.6)
    bubble = material("Bubble", (1.0, 0.5, 0.55), emission=(1.0, 0.5, 0.55), strength=1.5)
    cork = material("Cork", (0.55, 0.38, 0.22), rough=0.9)
    twine = material("Twine", (0.5, 0.4, 0.25), rough=0.95)
    gold = material("Gold", (1.0, 0.7, 0.3), metallic=1, rough=0.22)
    sapphire = material("Sapphire", (0.05, 0.25, 0.95), rough=0.04, transmission=0.85, ior=1.77,
                        emission=(0.1, 0.3, 1.0), strength=1.0)

    # Flask: one lathed profile, given real glass thickness.
    profile = [(0.2, 0.0), (0.27, 0.02), (0.32, 0.08), (0.345, 0.18), (0.34, 0.3), (0.3, 0.42),
               (0.22, 0.52), (0.13, 0.58), (0.095, 0.63), (0.09, 0.8), (0.1, 0.83), (0.112, 0.86),
               (0.1, 0.89), (0.085, 0.9)]
    flask = lathe(scene, "Flask", profile, segments=32, cap_bottom=True)
    solidify(flask, 0.008)
    subsurf(flask, 2)
    finish(flask, glass, smooth=True)
    flask.visible_shadow = False  # lets the potion's glow reach the table through the glass

    # Liquid: the same profile, slightly inset, filled to just above the widest point.
    fill = 0.38
    inner = []
    for (r1, z1), (r2, z2) in zip(profile, profile[1:]):
        if z1 > fill:
            break
        inner.append((r1 * 0.92, max(z1, 0.012)))
        if z2 > fill:
            k = (fill - z1) / (z2 - z1)
            inner.append(((r1 + (r2 - r1) * k) * 0.92, fill))
    liquid = lathe(scene, "Liquid", inner, segments=32, cap_bottom=True, cap_top=True)
    subsurf(liquid, 1)
    finish(liquid, potion, smooth=True)
    liquid.visible_shadow = False

    rnd = random.Random(9)
    for _ in range(16):
        a, r = rnd.uniform(0, math.tau), rnd.uniform(0.0, 0.2)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rnd.uniform(0.006, 0.016),
                                              location=(r * math.cos(a), r * math.sin(a), rnd.uniform(0.06, 0.34)))
        finish(active(), bubble, smooth=True, asset=False)

    bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.075, radius2=0.092, depth=0.13, location=(0, 0, 0.93))
    finish(active(), cork)
    for z in (0.72, 0.745):
        bpy.ops.mesh.primitive_torus_add(major_radius=0.094, minor_radius=0.008, major_segments=24,
                                         minor_segments=6, location=(0, 0, z))
        finish(active(), twine, smooth=True)

    # Coins and a gem on the table: the economy's two currencies.
    def coin(loc, rot):
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.07, depth=0.014, location=loc, rotation=rot)
        ob = active()
        bevel(ob, 0.003)
        finish(ob, gold, smooth=True)

    coin((0.56, -0.18, 0.007), (0, 0, 0))
    coin((0.7, 0.06, 0.007), (0, 0, 0.4))
    coin((0.46, 0.12, 0.05), (math.radians(62), 0, math.radians(20)))
    for k, z in enumerate((0.007, 0.021, 0.035, 0.049)):
        coin((-0.56 + k * 0.006, 0.16, z), (0, 0, k * 0.5))
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.055, location=(-0.42, -0.28, 0.045),
                                          rotation=(0.3, 0.5, 0.2))
    finish(active(), sapphire)

    point_light(scene, "Glow", (0, 0, 0.2), 14, (1.0, 0.05, 0.12), radius=0.15)
    area_light(scene, "Key", (-1.7, -1.6, 1.7), (0, 0, 0.4), 160, 1.4, (1.0, 0.85, 0.72))
    area_light(scene, "Rim", (1.4, 1.9, 1.5), (0, 0, 0.4), 280, 0.8, (0.5, 0.62, 1.0))
    area_light(scene, "Back", (0, 1.2, 2.2), (0, 0, 0.4), 80, 1.2, (1.0, 0.9, 0.85))
    camera(scene, (0.95, -3.15, 0.9), (0.05, 0, 0.47), 60, 2.4)
    return scene, device


# Rendering ----------------------------------------------------------------------------


def asset_stats(scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tris = verts = objects = 0
    materials = set()
    for ob in scene.objects:
        if ob.type != "MESH" or not ob.get(ASSET_KEY):
            continue
        evaluated = ob.evaluated_get(depsgraph)
        me = evaluated.to_mesh()
        me.calc_loop_triangles()
        tris += len(me.loop_triangles)
        verts += len(me.vertices)
        evaluated.to_mesh_clear()
        objects += 1
        for slot in ob.material_slots:
            if slot.material:
                materials.add(slot.material.name)
    return {"tris": tris, "verts": verts, "objects": objects, "materials": len(materials)}


def render(scene, name, device):
    stats = asset_stats(scene)
    scene.render.filepath = os.path.join(RENDER_DIR, name + ".jpg")
    started = time.time()
    bpy.ops.render.render(write_still=True)
    stats.update({
        "render_seconds": round(time.time() - started, 1),
        "device": device,
        "samples": scene.cycles.samples,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    })
    if not PREVIEW:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(BLEND_DIR, name + ".blend"))
    return stats


SCENES = {"sword": scene_sword, "crown": scene_crown, "potion": scene_potion}

os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(BLEND_DIR, exist_ok=True)
results = {}
for scene_name, build in SCENES.items():
    if ONLY and scene_name not in ONLY:
        continue
    built_scene, render_device = build()
    results[scene_name] = render(built_scene, scene_name, render_device)
    print(f"[portfolio] {scene_name}: {results[scene_name]}", flush=True)

with open(os.path.join(RENDER_DIR, "stats-preview.json" if PREVIEW else "stats.json"), "w") as f:
    json.dump(results, f, indent=2)
