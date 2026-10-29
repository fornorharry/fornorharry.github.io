"""
Renders one of my own modelled .blend files for the portfolio: a studio beauty shot and a
clay render with a wireframe overlay that shows the topology.

    blender -b <model.blend> --python render_own_model.py -- <render_dir> <name> [preview]

The model and its materials are left exactly as they are in the file. Only the
presentation is replaced: the file's lights are removed and a studio setup (backdrop,
lights, camera, compositor finish) is added. Nothing is saved, so the original .blend
file is never changed.
"""

import bpy
import json
import math
import os
import sys
import time

from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
RENDER_DIR = os.path.abspath(argv[0])
NAME = argv[1]
PREVIEW = len(argv) > 2 and argv[2] == "preview"

scene = bpy.context.scene


def use_gpu():
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


def set_menu(socket, options):
    for option in options:
        try:
            socket.default_value = option
            return
        except (TypeError, ValueError):
            continue


def set_vector(socket, values):
    try:
        socket.default_value = values
    except (TypeError, ValueError):
        socket.default_value = (*values, 0)


def finishing(bloom=0.35, vignette=0.55):
    """Bloom on the brightest highlights and a soft vignette, as in build_scenes.py."""
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
        mask = n.new("CompositorNodeEllipseMask")
        set_vector(mask.inputs["Size"], (0.95, 0.85))
        blur = n.new("CompositorNodeBlur")
        set_vector(blur.inputs["Size"], (320, 320))
        set_menu(blur.inputs["Type"], ("Gaussian", "GAUSS"))
        mix = n.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = vignette
        out = n.new("NodeGroupOutput")
        link(layers.outputs["Image"], glare.inputs["Image"])
        link(glare.outputs["Image"], mix.inputs[6])
        link(mask.outputs["Mask"], blur.inputs["Image"])
        link(blur.outputs["Image"], mix.inputs[7])
        link(mix.outputs[2], out.inputs[0])
        scene.compositing_node_group = tree
        scene.render.use_compositing = True
    except Exception as err:
        print(f"[portfolio] compositor skipped: {err}", flush=True)


def new_material(name, color, rough):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    p = next(node for node in m.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    return m


def clay():
    m = new_material("Clay Wire", (0.5, 0.5, 0.52), 0.55)
    nt = m.node_tree
    wire = nt.nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = True
    wire.inputs["Size"].default_value = 0.8
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(wire.outputs["Factor"], mix.inputs[0])
    mix.inputs[6].default_value = (0.5, 0.5, 0.52, 1)
    mix.inputs[7].default_value = (0.05, 0.055, 0.07, 1)
    p = next(node for node in nt.nodes if node.type == "BSDF_PRINCIPLED")
    nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    return m


def link_new(ob):
    scene.collection.objects.link(ob)
    return ob


def look_at(ob, target):
    ob.rotation_euler = (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()


def area_light(name, loc, target, energy, size, color, shape="RECTANGLE", size_y=None):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.color = energy, color
    data.shape = shape
    data.size = size
    if size_y is not None:
        data.size_y = size_y
    ob = link_new(bpy.data.objects.new(name, data))
    ob.location = loc
    look_at(ob, target)
    # Seen in reflections (so black plastic shows its form) but never as a visible panel.
    ob.visible_camera = False
    return ob


def backdrop(mat, floor_z, center_x):
    """A floor that sweeps up into a back wall, so the render has no horizon line."""
    profile = [(y, floor_z) for y in (-30, -8, -2, 2.0, 6.0)]
    radius = 4.0
    for i in range(1, 16):
        a = -math.pi / 2 + (math.pi / 2) * i / 15
        profile.append((6.0 + radius * math.cos(a), floor_z + radius + radius * math.sin(a)))
    profile.append((6.0 + radius, floor_z + 30.0))
    verts = [(center_x + x, y, z) for (y, z) in profile for x in (-40, 40)]
    faces = [(i * 2, i * 2 + 1, i * 2 + 3, i * 2 + 2) for i in range(len(profile) - 1)]
    me = bpy.data.meshes.new("Backdrop")
    me.from_pydata(verts, [], faces)
    me.update()
    ob = link_new(bpy.data.objects.new("Backdrop", me))
    ob.data.materials.append(mat)
    for poly in ob.data.polygons:
        poly.use_smooth = True
    return ob


def model_meshes():
    return [o for o in scene.objects if o.type == "MESH" and o.visible_get() and not o.hide_render]


def bounds(objects):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for ob in objects:
        for corner in ob.bound_box:
            p = ob.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
    return lo, hi


def stats(objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tris = verts = 0
    materials = set()
    for ob in objects:
        evaluated = ob.evaluated_get(depsgraph)
        me = evaluated.to_mesh()
        me.calc_loop_triangles()
        tris += len(me.loop_triangles)
        verts += len(me.vertices)
        evaluated.to_mesh_clear()
        materials.update(s.material.name for s in ob.material_slots if s.material)
    return {"tris": tris, "verts": verts, "objects": len(objects), "materials": len(materials)}


# Presentation -----------------------------------------------------------------------

model = model_meshes()
info = stats(model)
lo, hi = bounds(model)
center = (lo + hi) / 2
size = (hi - lo).length

# The file's own lights and camera are replaced by the studio setup below.
for ob in [o for o in scene.objects if o.type in ("LIGHT", "CAMERA")]:
    bpy.data.objects.remove(ob, do_unlink=True)

world = bpy.data.worlds.new("Studio")
scene.world = world
try:
    world.use_nodes = True
except AttributeError:
    pass
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.013, 0.016, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

floor = new_material("Studio Floor", (0.035, 0.037, 0.042), 0.32)
backdrop(floor, lo.z - 0.002, center.x)

# Big soft key from the upper front left, cool rims from behind, and long strip lights
# above whose reflections run along the slide and suppressor.
area_light("Key", center + Vector((-2.5, -5.5, 5.0)), center, 1000, 4.0, (1.0, 0.96, 0.9))
area_light("Rim Left", center + Vector((-6.0, 3.0, 3.2)), center, 1700, 1.2, (0.55, 0.72, 1.0),
           shape="RECTANGLE", size_y=5.0)
area_light("Rim Right", center + Vector((6.5, 3.0, 2.6)), center, 1200, 1.2, (0.6, 0.78, 1.0),
           shape="RECTANGLE", size_y=5.0)
area_light("Strip Top", center + Vector((0.0, -1.2, 4.5)), center, 260, 8.0, (1.0, 1.0, 1.0),
           shape="RECTANGLE", size_y=0.5)
area_light("Back Glow", Vector((center.x, 6.5, lo.z + 0.4)), Vector((center.x, 10.0, lo.z + 4.0)),
           700, 6.0, (0.35, 0.5, 0.9))

cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 70
cam = link_new(bpy.data.objects.new("Camera", cam_data))
# Far enough back for the whole model to fit with some room around it.
target = center + Vector((0.0, 0.2, -0.05))
direction = Vector((2.0, -9.6, 3.0)).normalized()
cam.location = target + direction * size * 2.15
look_at(cam, target)
cam_data.dof.use_dof = True
cam_data.dof.focus_distance = (target - cam.location).length
cam_data.dof.aperture_fstop = 4.0
scene.camera = cam

device = use_gpu()
scene.cycles.samples = 48 if PREVIEW else 640
scene.cycles.use_adaptive_sampling = True
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 12
scene.frame_set(scene.frame_current)
render = scene.render
render.resolution_x, render.resolution_y = (800, 500) if PREVIEW else (2400, 1500)
render.resolution_percentage = 100
try:
    render.image_settings.media_type = "IMAGE"
except (AttributeError, TypeError):
    pass
render.image_settings.file_format = "JPEG"
render.image_settings.quality = 90
try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
except TypeError:
    pass
scene.view_settings.exposure = 0.0
finishing()

os.makedirs(RENDER_DIR, exist_ok=True)
render.filepath = os.path.join(RENDER_DIR, NAME + ".jpg")
started = time.time()
bpy.ops.render.render(write_still=True)
info.update({
    "render_seconds": round(time.time() - started, 1),
    "device": device,
    "samples": scene.cycles.samples,
    "resolution": [render.resolution_x, render.resolution_y],
})

# Clay and wireframe pass: same shot, every model material swapped for clay. Nothing is
# saved, so the swap never reaches the original file.
wire = clay()
matte = new_material("Matte", (0.05, 0.05, 0.06), 0.9)
for ob in scene.objects:
    if ob.type == "MESH":
        ob.data.materials.clear()
        ob.data.materials.append(matte if ob.name == "Backdrop" else wire)
scene.view_settings.exposure = -1.6
scene.cycles.samples = 24 if PREVIEW else 200
render.resolution_x, render.resolution_y = (800, 500) if PREVIEW else (1600, 1000)
render.filepath = os.path.join(RENDER_DIR, NAME + "-wire.jpg")
started = time.time()
bpy.ops.render.render(write_still=True)
info["wire_seconds"] = round(time.time() - started, 1)

if not PREVIEW:
    stats_path = os.path.join(RENDER_DIR, "stats.json")
    results = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            results = json.load(f)
    results[NAME] = info
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
print(f"[portfolio] {NAME}: {info}", flush=True)
