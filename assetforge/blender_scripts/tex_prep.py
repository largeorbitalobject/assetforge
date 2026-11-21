import bpy, sys, math, mathutils, os, json
glb, outdir = sys.argv[-2], sys.argv[-1]
os.makedirs(outdir, exist_ok=True)
TARGET_FACES=120000

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
# join
ctx=bpy.context
for o in bpy.context.scene.objects: o.select_set(False)
for o in meshes: o.select_set(True)
ctx.view_layer.objects.active=meshes[0]
if len(meshes)>1: bpy.ops.object.join()
obj=ctx.view_layer.objects.active
# decimate
nf=len(obj.data.polygons)
if nf>TARGET_FACES:
    m=obj.modifiers.new("dec","DECIMATE"); m.ratio=TARGET_FACES/nf
    bpy.ops.object.modifier_apply(modifier=m.name)
print("faces after decimate:", len(obj.data.polygons))
# smart uv unwrap
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.002)
bpy.ops.object.mode_set(mode='OBJECT')
# bounds
mn=mathutils.Vector((1e9,)*3); mx=mathutils.Vector((-1e9,)*3)
for c in obj.bound_box:
    w=obj.matrix_world@mathutils.Vector(c)
    mn=mathutils.Vector(map(min,mn,w)); mx=mathutils.Vector(map(max,mx,w))
center=(mn+mx)/2; size=(mx-mn); rad=max(size)*0.5 or 1.0
cam_dist=rad*3.0

scn=bpy.context.scene
scn.render.engine='CYCLES'; scn.cycles.device='CPU'
scn.render.resolution_x=1024; scn.render.resolution_y=1024
scn.render.film_transparent=True
scn.view_settings.view_transform='Raw'
# camera (ortho)
cam_d=bpy.data.cameras.new("Cam"); cam=bpy.data.objects.new("Cam",cam_d); scn.collection.objects.link(cam)
cam_d.type='ORTHO'; cam_d.ortho_scale=rad*2.25; scn.camera=cam
# world black
scn.world=bpy.data.worlds.new("W"); scn.world.use_nodes=True
scn.world.node_tree.nodes['Background'].inputs[0].default_value=(0,0,0,1)

# materials
def mat_uv():
    m=bpy.data.materials.new("uv"); m.use_nodes=True; nt=m.node_tree; nt.nodes.clear()
    uv=nt.nodes.new("ShaderNodeUVMap"); em=nt.nodes.new("ShaderNodeEmission"); out=nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(uv.outputs["UV"],em.inputs["Color"]); nt.links.new(em.outputs[0],out.inputs["Surface"]); return m
def mat_facing():
    m=bpy.data.materials.new("fac"); m.use_nodes=True; nt=m.node_tree; nt.nodes.clear()
    lw=nt.nodes.new("ShaderNodeLayerWeight"); em=nt.nodes.new("ShaderNodeEmission"); out=nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(lw.outputs["Facing"],em.inputs["Color"]); nt.links.new(em.outputs[0],out.inputs["Surface"]); return m
def mat_clay():
    m=bpy.data.materials.new("clay"); m.use_nodes=True; b=m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value=(0.6,0.55,0.5,1); b.inputs["Roughness"].default_value=0.6; return m
MU,MF,MC=mat_uv(),mat_facing(),mat_clay()
def setmat(m):
    obj.data.materials.clear(); obj.data.materials.append(m)
def smooth():
    for p in obj.data.polygons: p.use_smooth=True
smooth()

# compositor for depth
scn.use_nodes=True; scn.view_layers[0].use_pass_z=True
tree=scn.node_tree; tree.nodes.clear()
rl=tree.nodes.new("CompositorNodeRLayers")
mr=tree.nodes.new("CompositorNodeMapRange")
mr.inputs['From Min'].default_value=cam_dist-rad*1.2; mr.inputs['From Max'].default_value=cam_dist+rad*1.2
mr.inputs['To Min'].default_value=1.0; mr.inputs['To Max'].default_value=0.0; mr.use_clamp=True
comp=tree.nodes.new("CompositorNodeComposite")
tree.links.new(rl.outputs['Depth'],mr.inputs['Value']); tree.links.new(mr.outputs[0],comp.inputs['Image'])

def place_cam(az,el):
    a=math.radians(az); e=math.radians(el)
    cam.location=center+mathutils.Vector((cam_dist*math.sin(a)*math.cos(e), -cam_dist*math.cos(a)*math.cos(e), cam_dist*math.sin(e)))
    d=center-cam.location; cam.rotation_euler=d.to_track_quat('-Z','Y').to_euler()

def render(path,depth8=False,bits='16'):
    scn.render.image_settings.file_format='PNG'; scn.render.image_settings.color_depth=('8' if depth8 else bits)
    scn.render.image_settings.color_mode='RGBA'
    scn.render.filepath=path; bpy.ops.render.render(write_still=True)

VIEWS={'front':(0,8),'fl':(45,8),'fr':(-45,8),'back':(180,8),'bl':(135,8),'br':(-135,8),'top':(0,75)}
scn.cycles.samples=2
key=bpy.data.objects.new("k",bpy.data.lights.new("k","SUN")); scn.collection.objects.link(key); key.data.energy=3
meta={'center':list(center),'rad':rad,'views':{}}
for name,(az,el) in VIEWS.items():
    place_cam(az,el)
    # depth (compositor on)
    scn.use_nodes=True; setmat(MC)  # any mat; depth from geometry
    render(f"{outdir}/depth_{name}.png", depth8=True)
    # uv (compositor off, raw)
    scn.use_nodes=False; setmat(MU); render(f"{outdir}/uv_{name}.png", bits='16')
    setmat(MF); render(f"{outdir}/facing_{name}.png", bits='16')
    # clay beauty (standard view)
    scn.view_settings.view_transform='Standard'; setmat(MC); render(f"{outdir}/clay_{name}.png", depth8=True)
    scn.view_settings.view_transform='Raw'
    meta['views'][name]={'az':az,'el':el}
    print("view done:",name)
# export working mesh
bpy.ops.object.select_all(action='SELECT')
setmat(MC)
bpy.ops.export_scene.gltf(filepath=f"{outdir}/working.glb", use_selection=False, export_format='GLB')
json.dump(meta, open(f"{outdir}/meta.json","w"))
print("PREP_DONE faces", len(obj.data.polygons))
