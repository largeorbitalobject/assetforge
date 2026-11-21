import bpy, sys, math, mathutils
glb, tex, out, glb_out = sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
obj=[o for o in bpy.context.scene.objects if o.type=='MESH'][0]
m=bpy.data.materials.new("pbr"); m.use_nodes=True; nt=m.node_tree; b=nt.nodes.get("Principled BSDF")
def imgnode(path, noncolor):
    n=nt.nodes.new("ShaderNodeTexImage"); n.image=bpy.data.images.load(path)
    if noncolor: n.image.colorspace_settings.name='Non-Color'
    return n
base=imgnode(f"{tex}/pbr_base.png",False); met=imgnode(f"{tex}/pbr_metal.png",True); rgh=imgnode(f"{tex}/pbr_rough.png",True)
nt.links.new(base.outputs["Color"], b.inputs["Base Color"])
nt.links.new(met.outputs["Color"], b.inputs["Metallic"])
nt.links.new(rgh.outputs["Color"], b.inputs["Roughness"])
obj.data.materials.clear(); obj.data.materials.append(m)
for p in obj.data.polygons: p.use_smooth=True
bpy.ops.export_scene.gltf(filepath=glb_out, use_selection=False, export_format='GLB')
# render with environment for metal reflections
mn=mathutils.Vector((1e9,)*3); mx=mathutils.Vector((-1e9,)*3)
for c in obj.bound_box:
    w=obj.matrix_world@mathutils.Vector(c); mn=mathutils.Vector(map(min,mn,w)); mx=mathutils.Vector(map(max,mx,w))
center=(mn+mx)/2; rad=max(mx-mn)*0.5 or 1
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=48
sc.render.resolution_x=520; sc.render.resolution_y=720; sc.render.image_settings.file_format='PNG'
# studio-ish world (gradient) for metal to catch reflections
sc.world=bpy.data.worlds.new("W"); sc.world.use_nodes=True
bg=sc.world.node_tree.nodes['Background']; bg.inputs[0].default_value=(0.5,0.53,0.58,1); bg.inputs[1].default_value=0.7
for nm,en,rot in [("k",3.5,(math.radians(48),0,math.radians(40))),("f",1.4,(math.radians(62),0,math.radians(-110))),("r",2.2,(math.radians(-25),0,math.radians(165)))]:
    s=bpy.data.objects.new(nm,bpy.data.lights.new(nm,'SUN')); sc.collection.objects.link(s); s.data.energy=en; s.rotation_euler=rot
camd=bpy.data.cameras.new("C"); cam=bpy.data.objects.new("C",camd); sc.collection.objects.link(cam); sc.camera=cam
paths=[]
for nm,az in [('front',0),('threeq',35),('side',92)]:
    a=math.radians(az); el=math.radians(8); dist=rad*2.7
    cam.location=center+mathutils.Vector((dist*math.sin(a)*math.cos(el),-dist*math.cos(a)*math.cos(el),dist*math.sin(el)+rad*0.1))
    d=center-cam.location; cam.rotation_euler=d.to_track_quat('-Z','Y').to_euler()
    p=out.replace('.png',f'_{nm}.png'); sc.render.filepath=p; bpy.ops.render.render(write_still=True); paths.append(p)
print("RENDERED", " ".join(paths))
