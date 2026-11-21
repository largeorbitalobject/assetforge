import bpy, sys, math, os
# args after --
argv = sys.argv[sys.argv.index("--")+1:]
glb = argv[0]; out = argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
# frame all
objs = [o for o in bpy.data.objects if o.type=='MESH']
import mathutils
mn = mathutils.Vector((1e9,1e9,1e9)); mx = mathutils.Vector((-1e9,-1e9,-1e9))
for o in objs:
    for v in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(v)
        for i in range(3):
            mn[i]=min(mn[i],w[i]); mx[i]=max(mx[i],w[i])
center=(mn+mx)/2; size=(mx-mn); rad=max(size)*0.5
# camera
cam_data=bpy.data.cameras.new("cam"); cam=bpy.data.objects.new("cam",cam_data); bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera=cam
dist=rad*3.2
cam.location=(center[0]+dist*0.6, center[1]-dist*0.9, center[2]+dist*0.4)
d=center-cam.location
cam.rotation_euler=d.to_track_quat('-Z','Y').to_euler()
# lights
for ang in [0,120,240]:
    l=bpy.data.lights.new(f"l{ang}",'SUN'); l.energy=3.0
    lo=bpy.data.objects.new(f"l{ang}",l); bpy.context.scene.collection.objects.link(lo)
    lo.rotation_euler=(math.radians(50),0,math.radians(ang))
w=bpy.context.scene.world or bpy.data.worlds.new("w")
bpy.context.scene.world=w; w.use_nodes=True
w.node_tree.nodes["Background"].inputs[1].default_value=0.6
sc=bpy.context.scene
try: sc.render.engine='BLENDER_EEVEE_NEXT'
except Exception:
    try: sc.render.engine='BLENDER_EEVEE'
    except Exception: sc.render.engine='CYCLES'
sc.render.resolution_x=768; sc.render.resolution_y=768
sc.render.film_transparent=True
sc.render.filepath=out
bpy.ops.render.render(write_still=True)
print("RENDERED", out)
