"""
Builds and renders the portfolio's Blender scenes: the three shop items from the economy
system's Catalog (Iron Sword, Golden Crown, Health Potion), modelled procedurally,
rendered in Cycles and finished in Blender's compositor.

Run from a terminal (no Blender window needed):
    blender -b --factory-startup --python build_scenes.py -- <render_dir> <blend_dir> [preview] [sword,crown,potion]

Each scene produces two images: the beauty render (<name>.jpg) and a clay render with a
wireframe overlay that shows the topology (<name>-wire.jpg). "preview" renders small and
fast to check composition. Without it, renders are 2400x1500 and each scene's .blend file
is saved to <blend_dir>. Triangle counts and render times go to <render_dir>/stats.json.
"""

import bmesh
import bpy
import json
import math
import os
import random
import sys
import time

from mathutils import Matrix, Vector, noise

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
RENDER_DIR = os.path.abspath(argv[0]) if len(argv) > 0 else os.path.abspath("renders")
BLEND_DIR = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.abspath("blend")
PREVIEW = len(argv) > 2 and argv[2] == "preview"
ONLY = set(argv[3].split(",")) if len(argv) > 3 and argv[3] else None

# The objects that make up the item itself, for the stats (not the stone, cushion or table).
ASSET_KEY = "portfolio_asset"
# Floating specks and background lights, hidden in the clay render.
DECOR_KEY = "portfolio_decor"

UP = Vector((0, 0, 1))


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
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return kind
    scene.cycles.device = "CPU"
    return "CPU"


def render_settings(scene, samples=640):
    device = use_gpu(scene)
    cycles = scene.cycles
    cycles.samples = 32 if PREVIEW else samples
    cycles.use_adaptive_sampling = True
    cycles.use_denoising = True
    try:
        cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    cycles.max_bounces = 16
    cycles.transmission_bounces = 16
    cycles.glossy_bounces = 8
    cycles.volume_bounces = 2
    render = scene.render
    render.resolution_x, render.resolution_y = (800, 500) if PREVIEW else (2400, 1500)
    render.resolution_percentage = 100
    render.image_settings.file_format = "JPEG"
    render.image_settings.quality = 90
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


def world(scene, color, strength=1.0, haze=0.0, haze_color=(1, 1, 1)):
    w = bpy.data.worlds.new("World")
    scene.world = w
    try:
        w.use_nodes = True
    except AttributeError:
        pass
    nt = w.node_tree
    nt.nodes["Background"].inputs["Color"].default_value = (*color, 1)
    nt.nodes["Background"].inputs["Strength"].default_value = strength
    if haze > 0:
        # A thin fog through the whole scene, so spotlights draw visible beams.
        volume = nt.nodes.new("ShaderNodeVolumePrincipled")
        volume.inputs["Density"].default_value = haze
        volume.inputs["Anisotropy"].default_value = 0.45
        volume.inputs["Color"].default_value = (*haze_color, 1)
        output = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
        nt.links.new(volume.outputs["Volume"], output.inputs["Volume"])


def set_menu(socket, options):
    for option in options:
        try:
            socket.default_value = option
            return True
        except (TypeError, ValueError):
            continue
    return False


def set_vector(socket, values):
    try:
        socket.default_value = values
    except (TypeError, ValueError):
        socket.default_value = (*values, 0)


def finishing(scene, bloom=0.5, vignette=0.6):
    """Compositor: bloom on the brightest highlights, a touch of chromatic aberration,
    and a soft vignette that pulls the eye to the middle."""
    try:
        tree = bpy.data.node_groups.new("Finish", "CompositorNodeTree")
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        n, link = tree.nodes, tree.links.new
        layers = n.new("CompositorNodeRLayers")
        layers.scene = scene
        glare = n.new("CompositorNodeGlare")
        set_menu(glare.inputs["Type"], ("Bloom", "BLOOM"))
        set_menu(glare.inputs["Quality"], ("High", "HIGH"))
        glare.inputs["Threshold"].default_value = 1.0
        glare.inputs["Strength"].default_value = bloom
        glare.inputs["Size"].default_value = 0.7
        lens = n.new("CompositorNodeLensdist")
        lens.inputs["Dispersion"].default_value = 0.012
        mask = n.new("CompositorNodeEllipseMask")
        set_vector(mask.inputs["Size"], (0.92, 0.82))
        blur = n.new("CompositorNodeBlur")
        set_vector(blur.inputs["Size"], (320, 320))
        set_menu(blur.inputs["Type"], ("Gaussian", "GAUSS"))
        mix = n.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = vignette
        out = n.new("NodeGroupOutput")
        link(layers.outputs["Image"], glare.inputs["Image"])
        link(glare.outputs["Image"], lens.inputs["Image"])
        link(lens.outputs["Image"], mix.inputs[6])
        link(mask.outputs["Mask"], blur.inputs["Image"])
        link(blur.outputs["Image"], mix.inputs[7])
        link(mix.outputs[2], out.inputs[0])
        scene.compositing_node_group = tree
        scene.render.use_compositing = True
    except Exception as err:  # the render still works without the finishing pass
        print(f"[portfolio] compositor skipped: {err}", flush=True)


# Materials ----------------------------------------------------------------------------


def principled(m):
    return next(node for node in m.node_tree.nodes if node.type == "BSDF_PRINCIPLED")


def material(name, color, metallic=0.0, rough=0.5, transmission=0.0, ior=1.45, emission=None,
             strength=0.0, coat=0.0, sheen=0.0, sheen_tint=None):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    p = principled(m)

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


def texture(m, kind, scale, detail=6.0, coord="Object", randomness=1.0):
    nt = m.node_tree
    coords = nt.nodes.new("ShaderNodeTexCoord")
    tex = nt.nodes.new(kind)
    tex.inputs["Scale"].default_value = scale
    if "Detail" in tex.inputs:
        tex.inputs["Detail"].default_value = detail
    if "Randomness" in tex.inputs:
        tex.inputs["Randomness"].default_value = randomness
    nt.links.new(coords.outputs[coord], tex.inputs["Vector"])
    return tex


def drive_roughness(m, source, low, high):
    nt = m.node_tree
    remap = nt.nodes.new("ShaderNodeMapRange")
    remap.inputs["To Min"].default_value = low
    remap.inputs["To Max"].default_value = high
    nt.links.new(source, remap.inputs["Value"])
    nt.links.new(remap.outputs["Result"], principled(m).inputs["Roughness"])


def drive_bump(m, source, strength, distance=0.01):
    nt = m.node_tree
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    nt.links.new(source, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], principled(m).inputs["Normal"])


def ramp(m, source, stops, target="Base Color"):
    nt = m.node_tree
    node = nt.nodes.new("ShaderNodeValToRGB")
    elements = node.color_ramp.elements
    elements[0].position, elements[0].color = stops[0][0], (*stops[0][1], 1)
    elements[1].position, elements[1].color = stops[-1][0], (*stops[-1][1], 1)
    for position, color in stops[1:-1]:
        elements.new(position).color = (*color, 1)
    nt.links.new(source, node.inputs["Factor"])
    if target:
        nt.links.new(node.outputs["Color"], principled(m).inputs[target])
    return node


def gold(name="Gold", hammered=True):
    """Polished gold with fine roughness variation, optionally hand-hammered."""
    m = material(name, (1.0, 0.7, 0.32), metallic=1, rough=0.14)
    drive_roughness(m, texture(m, "ShaderNodeTexNoise", 40).outputs["Factor"], 0.08, 0.24)
    if hammered:
        drive_bump(m, texture(m, "ShaderNodeTexVoronoi", 60).outputs["Distance"], 0.12, 0.004)
    return m


def steel():
    m = material("Steel", (0.8, 0.82, 0.85), metallic=1, rough=0.16)
    principled(m).inputs["Anisotropic"].default_value = 0.6
    drive_roughness(m, texture(m, "ShaderNodeTexNoise", 25, detail=10).outputs["Factor"], 0.1, 0.3)
    drive_bump(m, texture(m, "ShaderNodeTexNoise", 400, detail=2).outputs["Factor"], 0.04, 0.002)
    return m


def gem(name, color, glow=0.35):
    return material(name, color, rough=0.0, transmission=1.0, ior=1.9, emission=color, strength=glow)


def velvet(name, color):
    # Sheen is kept low: strong sheen reads pink once it's tone-mapped.
    m = material(name, color, rough=0.75, sheen=0.45, sheen_tint=(1.0, 0.25, 0.3))
    drive_bump(m, texture(m, "ShaderNodeTexNoise", 260, detail=3).outputs["Factor"], 0.08, 0.002)
    return m


def stone():
    """Rock with darker cracks, and moss creeping over the faces that point up."""
    m = material("Stone", (0.2, 0.21, 0.23), rough=0.85)
    nt = m.node_tree
    p = principled(m)
    cells = texture(m, "ShaderNodeTexVoronoi", 3.5)
    grit = texture(m, "ShaderNodeTexNoise", 12, detail=10)
    rock_color = ramp(m, cells.outputs["Distance"], [(0.0, (0.07, 0.075, 0.08)), (0.55, (0.17, 0.18, 0.19)), (1.0, (0.24, 0.25, 0.26))], target=None)
    geometry = nt.nodes.new("ShaderNodeNewGeometry")
    split = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geometry.outputs["Normal"], split.inputs["Vector"])
    facing_up = nt.nodes.new("ShaderNodeMapRange")
    facing_up.inputs["From Min"].default_value = 0.8
    facing_up.inputs["From Max"].default_value = 0.97
    nt.links.new(split.outputs["Z"], facing_up.inputs["Value"])
    patchy = nt.nodes.new("ShaderNodeMath")
    patchy.operation = "MULTIPLY"
    nt.links.new(facing_up.outputs["Result"], patchy.inputs[0])
    nt.links.new(grit.outputs["Factor"], patchy.inputs[1])
    moss_mask = nt.nodes.new("ShaderNodeMapRange")
    moss_mask.inputs["From Min"].default_value = 0.35
    moss_mask.inputs["From Max"].default_value = 0.5
    nt.links.new(patchy.outputs["Value"], moss_mask.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(moss_mask.outputs["Result"], mix.inputs[0])
    nt.links.new(rock_color.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (0.025, 0.06, 0.018, 1)
    nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    drive_bump(m, grit.outputs["Factor"], 0.35, 0.02)
    return m


def clay():
    """Grey clay with a wireframe overlay, for the topology renders."""
    m = material("Clay Wire", (0.5, 0.5, 0.52), rough=0.55)
    nt = m.node_tree
    wire = nt.nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = True
    wire.inputs["Size"].default_value = 0.8
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(wire.outputs["Factor"], mix.inputs[0])
    mix.inputs[6].default_value = (0.5, 0.5, 0.52, 1)
    mix.inputs[7].default_value = (0.05, 0.055, 0.07, 1)
    nt.links.new(mix.outputs[2], principled(m).inputs["Base Color"])
    return m


# Geometry helpers ---------------------------------------------------------------------


def link_object(scene, ob):
    scene.collection.objects.link(ob)
    return ob


def select_only(ob):
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def finish(ob, mat=None, smooth=False, asset=False, parent=None, decor=False):
    if mat:
        ob.data.materials.append(mat)
    if smooth:
        for poly in ob.data.polygons:
            poly.use_smooth = True
    ob[ASSET_KEY] = asset
    ob[DECOR_KEY] = decor
    if parent:
        ob.parent = parent
    return ob


def smooth_by_angle(ob, degrees=35):
    """Smooth shading, but with crisp edges where faces meet at a sharp angle."""
    for poly in ob.data.polygons:
        poly.use_smooth = True
    try:
        select_only(ob)
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(degrees))
    except Exception:
        pass


def active():
    return bpy.context.active_object


def recalc(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()


def mesh_from(scene, name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    me.update()
    ob = link_object(scene, bpy.data.objects.new(name, me))
    recalc(ob)
    return ob


def lathe(scene, name, profile, segments=48, cap_bottom=False, cap_top=False, closed=False,
          start=0.0, end=math.tau):
    """Spins an (radius, height) profile around Z. `closed` joins the profile's ends (for a
    ring's cross-section). A start/end short of a full turn makes a partial shell."""
    full = abs((end - start) - math.tau) < 1e-6
    steps = segments if full else segments + 1
    n = len(profile)
    verts, faces = [], []
    for s in range(steps):
        a = start + (end - start) * s / segments
        for r, z in profile:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    rows = n if closed else n - 1
    for s in range(segments):
        s2 = (s + 1) % steps if full else s + 1
        for i in range(rows):
            i2 = (i + 1) % n
            faces.append((s * n + i, s2 * n + i, s2 * n + i2, s * n + i2))
    if cap_bottom:
        c = len(verts)
        verts.append((0, 0, profile[0][1]))
        for s in range(segments):
            faces.append((c, ((s + 1) % steps) * n, s * n))
    if cap_top:
        c = len(verts)
        verts.append((0, 0, profile[-1][1]))
        for s in range(segments):
            faces.append((c, s * n + n - 1, ((s + 1) % steps) * n + n - 1))
    return mesh_from(scene, name, verts, faces)


def loft(scene, name, rings, tip_start=None, tip_end=None, cap_start=False, cap_end=False):
    """Skins a list of equal-length point rings, with optional pointed or flat ends."""
    k = len(rings[0])
    verts = [v for ring in rings for v in ring]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(k):
            j2 = (j + 1) % k
            faces.append((i * k + j, i * k + j2, (i + 1) * k + j2, (i + 1) * k + j))
    if tip_start is not None:
        c = len(verts)
        verts.append(tip_start)
        faces += [(c, (j + 1) % k, j) for j in range(k)]
    elif cap_start:
        faces.append(tuple(reversed(range(k))))
    if tip_end is not None:
        c = len(verts)
        last = (len(rings) - 1) * k
        verts.append(tip_end)
        faces += [(c, last + j, last + (j + 1) % k) for j in range(k)]
    elif cap_end:
        faces.append(tuple((len(rings) - 1) * k + j for j in range(k)))
    return mesh_from(scene, name, verts, faces)


def tube(scene, name, path, radius, sides=10, closed=False):
    """Sweeps a circle along a path. Uses parallel transport, so the tube never twists or
    kinks, even when the path turns vertical."""
    path = [Vector(p) for p in path]
    n = len(path)
    tangents = []
    for i in range(n):
        if closed:
            t = path[(i + 1) % n] - path[i - 1]
        else:
            t = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
        tangents.append(t.normalized())
    ref = UP if abs(tangents[0].dot(UP)) < 0.9 else Vector((1, 0, 0))
    normal = tangents[0].cross(ref).normalized()
    verts, faces = [], []
    for i, p in enumerate(path):
        t = tangents[i]
        normal = (normal - t * normal.dot(t)).normalized()
        binormal = t.cross(normal)
        r = radius(i / max(n - 1, 1)) if callable(radius) else radius
        for s in range(sides):
            a = math.tau * s / sides
            verts.append(p + (normal * math.cos(a) + binormal * math.sin(a)) * r)
    for i in range(n if closed else n - 1):
        i2 = (i + 1) % n
        for s in range(sides):
            s2 = (s + 1) % sides
            faces.append((i * sides + s, i * sides + s2, i2 * sides + s2, i2 * sides + s))
    if not closed:
        faces.append(tuple(reversed(range(sides))))
        faces.append(tuple((n - 1) * sides + s for s in range(sides)))
    return mesh_from(scene, name, verts, faces)


def spheres(scene, name, centers, radius, segments=12, rings=8):
    """Many small spheres in one mesh: one object instead of hundreds."""
    bm = bmesh.new()
    for i, c in enumerate(centers):
        r = radius(i) if callable(radius) else radius
        bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=r,
                                  matrix=Matrix.Translation(Vector(c)))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = link_object(scene, bpy.data.objects.new(name, me))
    for poly in ob.data.polygons:
        poly.use_smooth = True
    return ob


def orient(direction):
    return UP.rotation_difference(Vector(direction).normalized()).to_matrix()


def cut_gem(scene, name, radius, mat, loc, direction, sides=8, parent=None, asset=True):
    """A faceted gem: a table on top, kite-shaped crown facets, a girdle and a pointed
    pavilion underneath. Built along +Z, then turned to face `direction`."""
    table, crown_h, girdle_h, pavilion_h = radius * 0.55, radius * 0.32, radius * 0.05, radius * 0.8
    local = []
    for s in range(sides):
        a = math.tau * (s + 0.5) / sides
        local.append((table * math.cos(a), table * math.sin(a), girdle_h + crown_h))
    for z in (girdle_h, -girdle_h):
        for s in range(sides):
            a = math.tau * s / sides
            local.append((radius * math.cos(a), radius * math.sin(a), z))
    local.append((0, 0, -girdle_h - pavilion_h))
    T, G, B, culet = 0, sides, 2 * sides, 3 * sides
    faces = [tuple(T + s for s in range(sides))]
    for s in range(sides):
        s2 = (s + 1) % sides
        faces.append((G + s, G + s2, T + s))
        faces.append((T + s, G + s2, T + s2))
        faces.append((B + s, B + s2, G + s2, G + s))
        faces.append((B + s2, B + s, culet))
    rot = orient(direction)
    verts = [Vector(loc) + rot @ Vector(v) for v in local]
    return finish(mesh_from(scene, name, verts, faces), mat, parent=parent, asset=asset)


def ring(scene, name, major, minor, loc, direction, mat, parent=None, asset=True):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor, major_segments=32,
                                     minor_segments=8, location=loc)
    ob = active()
    ob.rotation_euler = orient(direction).to_euler()
    return finish(ob, mat, smooth=True, parent=parent, asset=asset)


def box(scene, name, size, loc, mat, bevel_width=0.0, parent=None, asset=True, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rotation)
    ob = active()
    ob.name = name
    ob.scale = size
    select_only(ob)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel_width:
        bevel(ob, bevel_width, 3)
    finish(ob, mat, parent=parent, asset=asset)
    smooth_by_angle(ob)
    return ob


def bevel(ob, width, segments=2, harden=True):
    md = ob.modifiers.new("Bevel", "BEVEL")
    md.width = width
    md.segments = segments
    md.limit_method = "ANGLE"
    try:
        md.harden_normals = harden
    except AttributeError:
        pass


def subsurf(ob, levels):
    md = ob.modifiers.new("Subdivision", "SUBSURF")
    md.levels = levels
    md.render_levels = levels


def solidify(ob, thickness):
    md = ob.modifiers.new("Solidify", "SOLIDIFY")
    md.thickness = thickness
    md.offset = 0


def displace(ob, strength, size, kind="CLOUDS"):
    tex = bpy.data.textures.new(ob.name + " Displace", kind)
    if hasattr(tex, "noise_scale"):
        tex.noise_scale = size
    md = ob.modifiers.new("Displace", "DISPLACE")
    md.texture = tex
    md.strength = strength
    md.texture_coords = "LOCAL"


def rock(scene, name, loc, scale, seed, mat, subdiv=3, chunkiness=0.16, roughness=0.3):
    """A natural rock: noise pushes a sphere out of shape, the bottom flattens where it
    meets the ground, and a decimate pass breaks it into chunky facets."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=1, location=loc)
    ob = active()
    ob.name = name
    offset = Vector((seed * 3.1, seed * 1.7, seed * 2.3))
    for v in ob.data.vertices:
        p = v.co.copy()
        bulge = 1 + roughness * noise.noise(p * 1.4 + offset) + roughness * 0.35 * noise.noise(p * 4.2 + offset)
        v.co = p.normalized() * bulge
        if v.co.z < -0.45:
            v.co.z = -0.45 + (v.co.z + 0.45) * 0.15
    ob.scale = scale
    ob.rotation_euler = (0, 0, seed)
    md = ob.modifiers.new("Decimate", "DECIMATE")
    md.ratio = chunkiness
    return finish(ob, mat)


def backdrop(scene, mat, floor_z=0.0):
    """A floor that sweeps up into a back wall, so the render has no horizon line."""
    profile = [(y, floor_z) for y in (-14, -5, -1, 0.5, 1.5)]
    radius = 2.5
    for i in range(1, 16):
        a = -math.pi / 2 + (math.pi / 2) * i / 15
        profile.append((1.5 + radius * math.cos(a), floor_z + radius + radius * math.sin(a)))
    profile.append((1.5 + radius, floor_z + 14.0))
    verts = [(x, y, z) for (y, z) in profile for x in (-16, 16)]
    faces = [(i * 2, i * 2 + 1, i * 2 + 3, i * 2 + 2) for i in range(len(profile) - 1)]
    return finish(mesh_from(scene, "Backdrop", verts, faces), mat, smooth=True)


def specks(scene, name, count, center, spread, color, strength, seed, rmin=0.004, rmax=0.012):
    """Floating embers or sparkles. Out of focus near the camera, they become soft bokeh."""
    rnd = random.Random(seed)
    centers = [tuple(center[i] + rnd.uniform(-spread[i], spread[i]) for i in range(3)) for _ in range(count)]
    radii = [rnd.uniform(rmin, rmax) for _ in range(count)]
    ob = spheres(scene, name, centers, lambda i: radii[i], segments=8, rings=5)
    return finish(ob, material(name, color, emission=color, strength=strength), decor=True)


def grass(scene, name, seed, count, inner, outer, mat, avoid=None):
    """Tufts of grass blades as one mesh: each blade tapers to a point and bends as it grows."""
    rnd = random.Random(seed)
    verts, faces = [], []
    for _ in range(count):
        a, d = rnd.uniform(0, math.tau), rnd.uniform(inner, outer)
        base = Vector((d * math.cos(a), d * math.sin(a), 0))
        if avoid and avoid(base):
            continue
        for _ in range(rnd.randint(7, 12)):
            root = base + Vector((rnd.uniform(-0.05, 0.05), rnd.uniform(-0.05, 0.05), -0.01))
            height = rnd.uniform(0.07, 0.2)
            lean_a = rnd.uniform(0, math.tau)
            lean = Vector((math.cos(lean_a), math.sin(lean_a), 0)) * rnd.uniform(0.2, 0.6)
            side = Vector((-lean.y, lean.x, 0)).normalized() if lean.length else Vector((1, 0, 0))
            width = rnd.uniform(0.008, 0.014)
            start = len(verts)
            for step in range(4):
                t = step / 4
                center = root + UP * (t * height) + lean * (t * t * height)
                half = width * (1 - t)
                verts += [center - side * half, center + side * half]
            verts.append(root + UP * height + lean * height)
            for step in range(3):
                i = start + step * 2
                faces.append((i, i + 1, i + 3, i + 2))
            faces.append((start + 6, start + 7, start + 8))
    return finish(mesh_from(scene, name, verts, faces), mat, smooth=True)


def look_at(ob, target):
    ob.rotation_euler = (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()


def area_light(scene, name, loc, target, energy, size, color):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size, data.color = energy, size, color
    ob = link_object(scene, bpy.data.objects.new(name, data))
    ob.location = loc
    look_at(ob, target)


def spot_light(scene, name, loc, target, energy, angle, color, blend=0.4, radius=0.2):
    data = bpy.data.lights.new(name, "SPOT")
    data.energy, data.color, data.spot_size, data.spot_blend = energy, color, math.radians(angle), blend
    data.shadow_soft_size = radius
    ob = link_object(scene, bpy.data.objects.new(name, data))
    ob.location = loc
    look_at(ob, target)


def point_light(scene, name, loc, energy, color, radius=0.05):
    data = bpy.data.lights.new(name, "POINT")
    data.energy, data.color, data.shadow_soft_size = energy, color, radius
    ob = link_object(scene, bpy.data.objects.new(name, data))
    ob.location = loc
    # Light only: never seen as a hot spot in reflections or through glass.
    ob.visible_glossy = False
    ob.visible_transmission = False


def camera(scene, loc, target, lens, fstop):
    data = bpy.data.cameras.new("Camera")
    data.lens = lens
    ob = link_object(scene, bpy.data.objects.new("Camera", data))
    ob.location = loc
    look_at(ob, target)
    data.dof.use_dof = True
    data.dof.focus_distance = (Vector(target) - Vector(loc)).length
    data.dof.aperture_fstop = fstop
    scene.camera = ob


def pivot(scene, name):
    return link_object(scene, bpy.data.objects.new(name, None))


# Scenes -------------------------------------------------------------------------------


def scene_sword():
    scene = reset()
    device = render_settings(scene)
    world(scene, (0.005, 0.007, 0.012), haze=0.012, haze_color=(0.75, 0.85, 1.0))

    soil = material("Soil", (0.03, 0.032, 0.035), rough=0.9)
    ramp(soil, texture(soil, "ShaderNodeTexNoise", 3, detail=8).outputs["Factor"],
         [(0.3, (0.018, 0.02, 0.022)), (0.7, (0.05, 0.048, 0.045))])
    drive_bump(soil, texture(soil, "ShaderNodeTexNoise", 30, detail=10).outputs["Factor"], 0.3, 0.02)
    backdrop(scene, soil)

    rock_mat = stone()
    metal = steel()
    gilt = gold()
    leather = material("Leather", (0.09, 0.045, 0.025), rough=0.6)
    drive_bump(leather, texture(leather, "ShaderNodeTexNoise", 180, detail=4).outputs["Factor"], 0.2, 0.003)
    rune = material("Rune", (0.3, 0.85, 1.0), emission=(0.3, 0.85, 1.0), strength=16)
    sapphire = gem("Sapphire", (0.1, 0.45, 1.0), glow=1.2)
    # Grass darkens toward the roots: a ramp over the mesh's own height.
    blade_grass = material("Grass", (0.1, 0.25, 0.05), rough=0.6)
    grass_gradient = blade_grass.node_tree.nodes.new("ShaderNodeSeparateXYZ")
    coords = blade_grass.node_tree.nodes.new("ShaderNodeTexCoord")
    blade_grass.node_tree.links.new(coords.outputs["Generated"], grass_gradient.inputs["Vector"])
    ramp(blade_grass, grass_gradient.outputs["Z"], [(0.0, (0.015, 0.04, 0.012)), (1.0, (0.22, 0.38, 0.07))])

    # The stone, scattered pebbles and grass around it.
    rock(scene, "Stone", (0, 0, 0.2), (0.88, 0.76, 0.52), 7, rock_mat, chunkiness=0.28, roughness=0.32)
    rnd = random.Random(21)
    for i in range(18):
        a, d = rnd.uniform(0, math.tau), rnd.uniform(0.95, 1.9)
        size = rnd.uniform(0.04, 0.11)
        rock(scene, f"Pebble {i}", (d * math.cos(a), d * math.sin(a), size * 0.35),
             (size, size * rnd.uniform(0.8, 1.1), size * 0.6), 30 + i, rock_mat, subdiv=2, chunkiness=0.35)
    grass(scene, "Grass", 5, 40, 0.85, 2.6, blade_grass)

    # The sword, built tip-down at the origin, then tilted into the stone by its pivot.
    sword = pivot(scene, "Sword")
    W, T = 0.075, 0.021

    def blade_ring(z):
        grow = min(1.0, z / 0.24) ** 0.65
        w, t = W * grow * (1 - 0.06 * z), T * grow
        fuller = 0.5 if 0.32 <= z <= 0.86 else 0.95
        pts = [(w, 0), (0.72 * w, 0.5 * t), (0.32 * w, 0.88 * t), (0, fuller * t), (-0.32 * w, 0.88 * t),
               (-0.72 * w, 0.5 * t), (-w, 0), (-0.72 * w, -0.5 * t), (-0.32 * w, -0.88 * t), (0, -fuller * t),
               (0.32 * w, -0.88 * t), (0.72 * w, -0.5 * t)]
        return [(x, y, z) for x, y in pts]

    heights = [0.04, 0.1, 0.17, 0.24, 0.31, 0.33, 0.5, 0.7, 0.85, 0.87, 0.95]
    blade = loft(scene, "Blade", [blade_ring(z) for z in heights], tip_start=(0, 0, 0), cap_end=True)
    bevel(blade, 0.0025, 2)
    finish(blade, metal, asset=True, parent=sword)
    smooth_by_angle(blade, 30)

    # Glowing runes inlaid in the fuller on both faces.
    z = 0.36
    while z < 0.82:
        length = rnd.uniform(0.018, 0.05)
        for side in (1, -1):
            box(scene, "Rune", (0.007, 0.004, length), (0, side * 0.5 * T, z + length / 2), rune, parent=sword)
        z += length + rnd.uniform(0.012, 0.022)

    # Crossguard: quillons that sweep down toward the blade and taper to round finials.
    guard_rings = []
    for i in range(15):
        x = -0.3 + 0.6 * i / 14
        f = abs(x) / 0.3
        cz = 0.985 - 0.085 * f ** 2.2
        h, d = 0.052 - 0.02 * f, 0.062 - 0.026 * f
        guard_rings.append([(x, -d / 2, cz - h / 2), (x, d / 2, cz - h / 2), (x, d / 2, cz + h / 2), (x, -d / 2, cz + h / 2)])
    guard = loft(scene, "Crossguard", guard_rings, cap_start=True, cap_end=True)
    bevel(guard, 0.007, 3)
    finish(guard, gilt, asset=True, parent=sword)
    smooth_by_angle(guard, 40)
    for side in (1, -1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.03, location=(side * 0.305, 0, 0.9))
        finish(active(), gilt, smooth=True, asset=True, parent=sword)
    box(scene, "Guard Core", (0.1, 0.085, 0.09), (0, 0, 0.985), gilt, bevel_width=0.012, parent=sword)
    for side in (1, -1):
        cut_gem(scene, "Guard Gem", 0.026, sapphire, (0, side * 0.0445, 0.985), (0, side, 0), parent=sword)
        ring(scene, "Bezel", 0.028, 0.005, (0, side * 0.044, 0.985), (0, side, 0), gilt, parent=sword)

    # Grip with a spiral leather wrap, collars, and a gem-set pommel.
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.026, depth=0.3, location=(0, 0, 1.18))
    finish(active(), leather, smooth=True, asset=True, parent=sword)
    helix = [(0.028 * math.cos(t * 0.5), 0.028 * math.sin(t * 0.5), 1.04 + 0.27 * t / (math.tau * 11 * 2))
             for t in [i * 0.25 for i in range(int(math.tau * 11 * 2 / 0.25))]]
    finish(tube(scene, "Wrap", helix, 0.0055, sides=6), leather, smooth=True, asset=True, parent=sword)
    for z in (1.03, 1.33):
        ring(scene, "Collar", 0.03, 0.008, (0, 0, z), (0, 0, 1), gilt, parent=sword)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.05, location=(0, 0, 1.385))
    pommel = active()
    pommel.scale = (1, 0.82, 0.9)
    finish(pommel, gilt, asset=True, parent=sword)
    cut_gem(scene, "Pommel Gem", 0.019, sapphire, (0, -0.042, 1.385), (0, -1, 0), parent=sword)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.014, location=(0, 0, 1.44))
    finish(active(), gilt, smooth=True, asset=True, parent=sword)

    sword.location = (0.02, 0.0, 0.43)
    sword.rotation_euler = (math.radians(5), math.radians(-9), math.radians(22))

    specks(scene, "Embers", 70, (0, 0, 1.0), (2.4, 1.8, 1.1), (1.0, 0.62, 0.3), 12, seed=3)
    specks(scene, "Near Embers", 8, (0.9, -3.0, 1.0), (0.7, 0.4, 0.5), (1.0, 0.6, 0.3), 10, seed=4, rmin=0.006, rmax=0.012)
    area_light(scene, "Key", (-2.4, -2.2, 2.6), (0, 0, 0.9), 260, 2.2, (1.0, 0.82, 0.65))
    area_light(scene, "Rim", (1.6, 2.4, 2.2), (0, 0, 0.9), 620, 1.0, (0.45, 0.65, 1.0))
    area_light(scene, "Fill", (2.6, -2.4, 0.9), (0, 0, 0.8), 40, 3.0, (0.55, 0.65, 0.85))
    area_light(scene, "Card", (0.6, -3.6, 2.9), (0, 0, 1.0), 80, 3.5, (1.0, 0.97, 0.93))
    spot_light(scene, "Shaft", (-1.4, 2.6, 5.0), (0.1, -0.2, 0.9), 3500, 26, (0.72, 0.86, 1.0), blend=0.5)
    camera(scene, (1.3, -4.25, 1.05), (0, 0, 0.98), 52, 2.8)
    finishing(scene, bloom=0.6)
    return scene, device


def scene_crown():
    scene = reset()
    device = render_settings(scene)
    world(scene, (0.01, 0.008, 0.016))
    wall = material("Backdrop", (0.03, 0.022, 0.055), rough=0.6)
    backdrop(scene, wall)

    gilt = gold()
    braid = gold("Gold Braid", hammered=False)
    drive_bump(braid, texture(braid, "ShaderNodeTexWave", 120).outputs["Factor"], 0.4, 0.003)
    pearl = material("Pearl", (0.95, 0.93, 0.88), rough=0.12, coat=1)
    principled(pearl).inputs["Thin Film Thickness"].default_value = 280
    cushion_mat = velvet("Velvet", (0.085, 0.002, 0.011))
    cap_mat = velvet("Cap Velvet", (0.06, 0.002, 0.022))
    stones = [gem("Ruby", (0.95, 0.03, 0.08)), gem("Sapphire", (0.08, 0.3, 1.0)),
              gem("Emerald", (0.05, 0.75, 0.25)), gem("Amethyst", (0.6, 0.15, 0.95))]

    # Cushion: a subdivided box puffed into a pillow, tufted where the crown sits.
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=12, use_grid_fill=True)
    for v in bm.verts:
        x, y, z = v.co
        edge = max(abs(x), abs(y))
        z *= 1 - 0.5 * edge ** 2.5
        x *= 1 - 0.07 * abs(x * y)
        y *= 1 - 0.07 * abs(x * y)
        if z > 0:
            z -= 0.32 * math.exp(-((x * x + y * y) / 0.06))
        v.co = Vector((x * 0.8, y * 0.8, z * 0.21 + 0.21))
    me = bpy.data.meshes.new("Cushion")
    bm.to_mesh(me)
    bm.free()
    cushion = link_object(scene, bpy.data.objects.new("Cushion", me))
    subsurf(cushion, 2)
    finish(cushion, cushion_mat, smooth=True)
    seam = []
    for i in range(160):
        a = math.tau * i / 160
        c, s = math.cos(a), math.sin(a)
        seam.append((0.79 * math.copysign(abs(c) ** 0.22, c), 0.79 * math.copysign(abs(s) ** 0.22, s), 0.21))
    finish(tube(scene, "Piping", seam, 0.017, sides=10, closed=True), braid, smooth=True)
    for sx in (1, -1):
        for sy in (1, -1):
            corner = Vector((sx * 0.735, sy * 0.735, 0.21))
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.028, location=corner)
            finish(active(), braid, smooth=True)
            out = Vector((sx, sy, 0)).normalized()
            cord = [corner + out * (0.02 * t) - UP * (0.07 * t) for t in [i / 8 for i in range(9)]]
            finish(tube(scene, "Cord", cord, 0.006, sides=6), braid, smooth=True)
            bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.045, radius2=0.012, depth=0.13,
                                            location=tuple(corner + out * 0.02 - UP * 0.135))
            finish(active(), braid, smooth=True)

    # The crown itself, in its own space with the band's base at Z0.
    crown = pivot(scene, "Crown")
    Z0, H = 0.32, 0.18
    Z1 = Z0 + H
    band_profile = [(0.43, Z0), (0.447, Z0 + 0.008), (0.447, Z0 + 0.026), (0.432, Z0 + 0.034),
                    (0.432, Z1 - 0.034), (0.447, Z1 - 0.026), (0.447, Z1 - 0.008), (0.43, Z1),
                    (0.405, Z1), (0.405, Z0)]
    band = lathe(scene, "Band", band_profile, segments=96, closed=True)
    finish(band, gilt, asset=True, parent=crown)
    smooth_by_angle(band, 50)
    pearl_rows = [(0.452 * math.cos(math.tau * i / 72), 0.452 * math.sin(math.tau * i / 72), z)
                  for z in (Z0 + 0.017, Z1 - 0.017) for i in range(72)]
    finish(spheres(scene, "Pearl Rows", pearl_rows, 0.0085), pearl, asset=True, parent=crown)

    # Eight points, tall and short in turn, each a bevelled pointed arch with a gem.
    for i in range(8):
        a = math.tau * i / 8
        tall = i % 2 == 0
        h = 0.25 if tall else 0.15
        radial = Vector((math.cos(a), math.sin(a), 0))
        tangent = Vector((-math.sin(a), math.cos(a), 0))
        base = radial * 0.428 + UP * (Z1 - 0.012)
        outline = []
        for k in range(9):
            s = k / 8
            outline.append((-0.066 * (1 - s) ** 1.25, h * s))
        outline += [(-x, z) for x, z in reversed(outline[:-1])]
        verts = []
        for depth in (0.012, -0.012):
            verts += [base + tangent * x + UP * z + radial * depth for x, z in outline]
        k = len(outline)
        faces = [tuple(range(k)), tuple(reversed(range(k, 2 * k)))]
        faces += [(j, j + k, (j + 1) % k + k, (j + 1) % k) for j in range(k)]
        point = mesh_from(scene, "Point", verts, faces)
        bevel(point, 0.004, 2)
        finish(point, gilt, asset=True, parent=crown)
        smooth_by_angle(point, 40)
        gem_at = base + UP * (0.075 * h / 0.25) + radial * 0.014
        cut_gem(scene, "Point Gem", 0.02, stones[i % 3], gem_at, radial, parent=crown)
        ring(scene, "Bezel", 0.022, 0.004, gem_at - radial * 0.002, radial, gilt, parent=crown)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.024 if tall else 0.017,
                                             location=tuple(base + UP * (h + 0.016)))
        finish(active(), pearl, smooth=True, asset=True, parent=crown)
        # Band gems sit between the points.
        b = a + math.tau / 16
        out = Vector((math.cos(b), math.sin(b), 0))
        mid = out * 0.452 + UP * (Z0 + H / 2)
        cut_gem(scene, "Band Gem", 0.03, stones[(i + 1) % 4], mid, out, parent=crown)
        ring(scene, "Bezel", 0.034, 0.006, mid - out * 0.003, out, gilt, parent=crown)

    # Four arches rise from the tall points and meet under an orb and cross.
    top = Z1 + 0.33
    for i in range(0, 8, 2):
        a = math.tau * i / 8
        radial = Vector((math.cos(a), math.sin(a), 0))
        path = [radial * (0.415 * math.cos(t * math.pi / 2)) + UP * (Z1 + 0.33 * math.sin(t * math.pi / 2))
                for t in [j / 32 for j in range(33)]]
        finish(tube(scene, "Arch", path, 0.015, sides=12), gilt, smooth=True, asset=True, parent=crown)
        beads = [radial * (0.415 * math.cos(t * math.pi / 2) + 0.016 * math.cos(t * math.pi / 2))
                 + UP * (Z1 + 0.33 * math.sin(t * math.pi / 2) + 0.016 * math.sin(t * math.pi / 2))
                 for t in [j / 14 for j in range(1, 13)]]
        finish(spheres(scene, "Arch Pearls", beads, 0.011), pearl, asset=True, parent=crown)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.055, location=(0, 0, top + 0.045))
    finish(active(), gilt, smooth=True, asset=True, parent=crown)
    ring(scene, "Orb Band", 0.056, 0.007, (0, 0, top + 0.045), (0, 0, 1), gilt, parent=crown)
    box(scene, "Cross", (0.022, 0.022, 0.13), (0, 0, top + 0.16), gilt, bevel_width=0.005, parent=crown)
    box(scene, "Cross Arm", (0.085, 0.022, 0.022), (0, 0, top + 0.18), gilt, bevel_width=0.005, parent=crown)
    cut_gem(scene, "Cross Gem", 0.012, stones[0], (0, -0.013, top + 0.18), (0, -1, 0), parent=crown)

    # Velvet cap under the arches, with soft folds.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.405, location=(0, 0, Z0 + 0.06))
    cap = active()
    cap.scale = (1, 1, 0.86)
    subsurf(cap, 1)
    displace(cap, 0.025, 0.18)
    finish(cap, cap_mat, smooth=True, asset=True, parent=crown)
    crown.rotation_euler = (0, 0, -math.pi / 2 - math.tau / 16)

    specks(scene, "Sparkles", 40, (0, 0, 0.9), (2.0, 1.4, 0.9), (1.0, 0.78, 0.4), 12, seed=5, rmin=0.003, rmax=0.009)
    area_light(scene, "Key", (-2.1, -2.3, 2.7), (0, 0, 0.55), 340, 2.2, (1.0, 0.8, 0.62))
    area_light(scene, "Rim", (1.9, 2.2, 2.1), (0, 0, 0.55), 480, 1.0, (0.55, 0.65, 1.0))
    area_light(scene, "Top", (0.3, -0.4, 3.2), (0, 0, 0.55), 150, 1.6, (1.0, 0.95, 0.9))
    area_light(scene, "Card", (0.0, -3.2, 2.6), (0, 0, 0.55), 150, 3.5, (1.0, 0.97, 0.93))
    spot_light(scene, "Wall Glow", (0, -0.5, 3.6), (0, 4.0, 1.3), 900, 40, (0.62, 0.5, 1.0), blend=1.0, radius=0.5)
    camera(scene, (0.6, -3.8, 1.45), (0, 0, 0.6), 60, 4.0)
    finishing(scene, bloom=0.45)
    return scene, device


def scene_potion():
    scene = reset()
    device = render_settings(scene)
    world(scene, (0.008, 0.006, 0.005))
    backdrop(scene, material("Backdrop", (0.02, 0.014, 0.012), rough=0.5), floor_z=-0.9)

    # Walnut: noise stretched along the plank becomes long, uneven grain.
    wood = material("Walnut", (0.08, 0.04, 0.02), rough=0.35)
    nt = wood.node_tree
    stretch = nt.nodes.new("ShaderNodeMapping")
    stretch.inputs["Scale"].default_value = (0.5, 9.0, 1.0)
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], stretch.inputs["Vector"])
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 2.0
    grain.inputs["Detail"].default_value = 10
    grain.inputs["Distortion"].default_value = 1.5
    nt.links.new(stretch.outputs["Vector"], grain.inputs["Vector"])
    ramp(wood, grain.outputs["Factor"], [(0.3, (0.045, 0.022, 0.011)), (0.5, (0.09, 0.045, 0.022)), (0.7, (0.14, 0.072, 0.035))])
    drive_bump(wood, grain.outputs["Factor"], 0.08, 0.004)
    glass = material("Glass", (0.96, 0.98, 1.0), rough=0.012, transmission=1.0, ior=1.5)
    liquid_mat = material("Potion", (1, 1, 1), rough=0.02, transmission=1.0, ior=1.33)
    volume = liquid_mat.node_tree.nodes.new("ShaderNodeVolumePrincipled")
    # Clear ruby: almost no scattering (which would turn it milky), deep red absorption,
    # and a gentle emission so it still glows.
    volume.inputs["Color"].default_value = (0.03, 0.0, 0.005, 1)
    volume.inputs["Absorption Color"].default_value = (0.95, 0.03, 0.07, 1)
    volume.inputs["Density"].default_value = 4
    volume.inputs["Emission Color"].default_value = (1.0, 0.04, 0.1, 1)
    volume.inputs["Emission Strength"].default_value = 1.2
    out = next(n for n in liquid_mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
    liquid_mat.node_tree.links.new(volume.outputs["Volume"], out.inputs["Volume"])
    bubble = material("Bubble", (1.0, 0.85, 0.9), rough=0.0, transmission=1.0, ior=1.0,
                      emission=(1.0, 0.55, 0.6), strength=1.5)
    cork_mat = material("Cork", (0.55, 0.38, 0.22), rough=0.9)
    ramp(cork_mat, texture(cork_mat, "ShaderNodeTexVoronoi", 90).outputs["Distance"],
         [(0.0, (0.28, 0.17, 0.08)), (0.4, (0.55, 0.38, 0.22)), (1.0, (0.68, 0.5, 0.3))])
    drive_bump(cork_mat, texture(cork_mat, "ShaderNodeTexVoronoi", 90).outputs["Distance"], 0.5, 0.003)
    wax = material("Wax", (0.32, 0.008, 0.02), rough=0.22, coat=0.6)
    drive_bump(wax, texture(wax, "ShaderNodeTexNoise", 30).outputs["Factor"], 0.1, 0.004)
    twine_mat = material("Twine", (0.55, 0.45, 0.3), rough=0.95)
    drive_bump(twine_mat, texture(twine_mat, "ShaderNodeTexWave", 200).outputs["Factor"], 0.5, 0.002)
    parchment = material("Parchment", (0.85, 0.75, 0.55), rough=0.8)
    ramp(parchment, texture(parchment, "ShaderNodeTexNoise", 18, detail=8).outputs["Factor"],
         [(0.25, (0.55, 0.42, 0.25)), (0.5, (0.82, 0.71, 0.5)), (0.75, (0.9, 0.82, 0.62))])
    drive_bump(parchment, texture(parchment, "ShaderNodeTexNoise", 120).outputs["Factor"], 0.12, 0.002)
    ink = material("Red Ink", (0.62, 0.03, 0.04), rough=0.5)
    gilt = gold("Gold", hammered=False)
    sapphire = gem("Sapphire", (0.08, 0.32, 1.0), glow=0.6)

    # Table: a walnut slab whose front edge drops away into the dark.
    table = box(scene, "Table", (3.0, 1.6, 0.08), (0, 0.15, -0.04), wood, bevel_width=0.008)

    # Flask: one lathed profile, given real glass thickness.
    profile = [(0.2, 0.0), (0.27, 0.02), (0.32, 0.08), (0.345, 0.18), (0.34, 0.3), (0.3, 0.42),
               (0.22, 0.52), (0.13, 0.58), (0.095, 0.63), (0.09, 0.8), (0.1, 0.83), (0.112, 0.86),
               (0.1, 0.89), (0.085, 0.9)]

    def radius_at(z):
        for (r1, z1), (r2, z2) in zip(profile, profile[1:]):
            if z1 <= z <= z2:
                return r1 + (r2 - r1) * (z - z1) / (z2 - z1)
        return profile[-1][0]

    flask = lathe(scene, "Flask", profile, segments=64, cap_bottom=True)
    solidify(flask, 0.008)
    subsurf(flask, 2)
    finish(flask, glass, smooth=True, asset=True)
    flask.visible_shadow = False  # lets the potion's glow reach the table through the glass

    fill = 0.4
    inner = []
    for (r1, z1), (r2, z2) in zip(profile, profile[1:]):
        if z1 > fill:
            break
        inner.append((r1 * 0.93, max(z1, 0.012)))
        if z2 > fill:
            inner.append(((r1 + (r2 - r1) * (fill - z1) / (z2 - z1)) * 0.93, fill))
    liquid = lathe(scene, "Liquid", inner, segments=64, cap_bottom=True, cap_top=True)
    subsurf(liquid, 1)
    finish(liquid, liquid_mat, smooth=True, asset=True)
    liquid.visible_shadow = False

    rnd = random.Random(9)
    bubbles = []
    for _ in range(26):
        z = rnd.uniform(0.05, 0.37)
        a, r = rnd.uniform(0, math.tau), rnd.uniform(0, radius_at(z) * 0.75)
        bubbles.append((r * math.cos(a), r * math.sin(a), z))
    radii = [rnd.uniform(0.005, 0.016) for _ in bubbles]
    finish(spheres(scene, "Bubbles", bubbles, lambda i: radii[i]), bubble, asset=True)

    # Cork, sealed with wax and stamped with a cross.
    cork = lathe(scene, "Cork", [(0.074, 0.84), (0.08, 0.9), (0.089, 0.96), (0.091, 0.972), (0.086, 0.985), (0.0, 0.99)],
                 segments=32, cap_bottom=True)
    finish(cork, cork_mat, smooth=True, asset=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.11, location=(0, 0, 0.985))
    seal = active()
    seal.scale = (1, 1, 0.3)
    subsurf(seal, 1)
    displace(seal, 0.012, 0.08)
    finish(seal, wax, smooth=True, asset=True)
    # Runs of wax down the neck: thin, long, and starting inside the seal so they join it.
    for k in range(6):
        a = math.tau * k / 6 + rnd.uniform(-0.3, 0.3)
        drop = rnd.uniform(0.05, 0.1)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10, radius=1,
                                             location=(0.098 * math.cos(a), 0.098 * math.sin(a), 0.975 - drop / 2))
        drip = active()
        drip.scale = (0.016, 0.016, drop / 2 + 0.012)
        finish(drip, wax, smooth=True, asset=True)
    for size in ((0.05, 0.014, 0.008), (0.014, 0.05, 0.008)):
        box(scene, "Stamp", size, (0, 0, 1.024), gilt, bevel_width=0.002)

    # Twine wound three times around the neck, tied off with two loose ends.
    turns = []
    for i in range(int(math.tau * 3 / 0.15)):
        t = i * 0.15
        z = 0.72 + 0.04 * t / (math.tau * 3)
        turns.append((radius_at(z) * 1.07 * math.cos(t), radius_at(z) * 1.07 * math.sin(t), z))
    finish(tube(scene, "Twine", turns, 0.006, sides=8), twine_mat, smooth=True, asset=True)
    knot = Vector((radius_at(0.74) * 1.1 * math.cos(-1.2), radius_at(0.74) * 1.1 * math.sin(-1.2), 0.745))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.014, location=tuple(knot))
    finish(active(), twine_mat, smooth=True, asset=True)
    for spread in (-0.35, 0.3):
        away = Vector((math.cos(-1.2 + spread), math.sin(-1.2 + spread), 0))
        end = [knot + away * (0.09 * t) - UP * (0.17 * t * t + 0.02 * t) for t in [i / 10 for i in range(11)]]
        finish(tube(scene, "Twine End", end, 0.005, sides=6), twine_mat, smooth=True, asset=True)

    # A parchment label wrapped around the front, printed with a red cross.
    face = math.atan2(-2.9, 0.8)
    label_profile = [(radius_at(z) + 0.006, z) for z in (0.14, 0.17, 0.2, 0.23, 0.26, 0.29)]
    label = lathe(scene, "Label", label_profile, segments=24, start=face - 0.75, end=face + 0.75)
    finish(label, parchment, smooth=True, asset=True)
    front = Vector((math.cos(face), math.sin(face), 0))
    for size in ((0.075, 0.005, 0.024), (0.024, 0.005, 0.075)):
        box(scene, "Label Cross", size, tuple(front * (radius_at(0.215) + 0.0085) + UP * 0.215), ink,
            rotation=(0, 0, face + math.pi / 2))

    # Coins and a gem: the economy's two currencies.
    def coin(loc, rot):
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.07, depth=0.014, location=loc, rotation=rot)
        ob = active()
        bevel(ob, 0.004, 3)
        finish(ob, gilt)
        smooth_by_angle(ob)

    coin((0.56, -0.18, 0.007), (0, 0, 0))
    coin((0.72, 0.06, 0.007), (0, 0, 0.4))
    coin((0.46, 0.13, 0.05), (math.radians(62), 0, math.radians(20)))
    for k, z in enumerate((0.007, 0.021, 0.035, 0.049, 0.063)):
        coin((-0.56 + k * 0.006, 0.16, z), (0, 0, k * 0.5))
    cut_gem(scene, "Gem", 0.06, sapphire, (-0.42, -0.3, 0.03), (0.3, -0.4, 1), asset=False)

    # Out-of-focus lights in the distance: a warm tavern behind the bottle.
    rnd = random.Random(12)
    for k in range(18):
        warmth = rnd.choice(((1.0, 0.55, 0.22), (1.0, 0.7, 0.35), (1.0, 0.45, 0.2)))
        specks(scene, f"Bokeh {k}", 1, (rnd.uniform(-3.2, 3.2), rnd.uniform(2.6, 4.4), rnd.uniform(0.0, 2.2)),
               (0, 0, 0), warmth, rnd.uniform(5, 12), seed=k, rmin=0.04, rmax=0.09)

    point_light(scene, "Glow", (0, 0, 0.2), 16, (1.0, 0.05, 0.12), radius=0.15)
    area_light(scene, "Key", (-1.7, -1.6, 1.7), (0, 0, 0.4), 170, 1.4, (1.0, 0.85, 0.72))
    area_light(scene, "Rim", (1.4, 1.9, 1.5), (0, 0, 0.4), 300, 0.8, (0.5, 0.62, 1.0))
    area_light(scene, "Back", (0, 1.2, 2.2), (0, 0, 0.4), 80, 1.2, (1.0, 0.9, 0.85))
    area_light(scene, "Card", (0.5, -2.6, 1.6), (0, 0, 0.45), 60, 2.5, (1.0, 0.97, 0.93))
    camera(scene, (1.0, -3.8, 0.78), (0.03, 0, 0.5), 70, 2.0)
    finishing(scene, bloom=0.55)
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


def render_wire(scene, name):
    """The same shot in clay with a wireframe overlay, to show the topology. Runs after
    the .blend is saved, so swapping materials here never reaches the saved file."""
    wire = clay()
    matte = material("Matte", (0.05, 0.05, 0.06), rough=0.9)
    for ob in scene.objects:
        if ob.get(DECOR_KEY):
            ob.hide_render = True
        elif ob.type == "MESH":
            ob.data.materials.clear()
            ob.data.materials.append(matte if ob.name == "Backdrop" else wire)
    world(scene, (0.02, 0.02, 0.025))
    # The lights were set for dark materials; clay is much brighter.
    scene.view_settings.exposure = -1.2
    scene.cycles.samples = 24 if PREVIEW else 200
    scene.render.resolution_x, scene.render.resolution_y = (800, 500) if PREVIEW else (1600, 1000)
    scene.render.filepath = os.path.join(RENDER_DIR, name + "-wire.jpg")
    started = time.time()
    bpy.ops.render.render(write_still=True)
    return round(time.time() - started, 1)


SCENES = {"sword": scene_sword, "crown": scene_crown, "potion": scene_potion}

os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(BLEND_DIR, exist_ok=True)
stats_path = os.path.join(RENDER_DIR, "stats-preview.json" if PREVIEW else "stats.json")
# Re-rendering one scene keeps the other scenes' stats.
results = {}
if ONLY and os.path.exists(stats_path):
    with open(stats_path) as f:
        results = json.load(f)
for scene_name, build in SCENES.items():
    if ONLY and scene_name not in ONLY:
        continue
    built_scene, render_device = build()
    results[scene_name] = render(built_scene, scene_name, render_device)
    results[scene_name]["wire_seconds"] = render_wire(built_scene, scene_name)
    print(f"[portfolio] {scene_name}: {results[scene_name]}", flush=True)

with open(stats_path, "w") as f:
    json.dump(results, f, indent=2)
