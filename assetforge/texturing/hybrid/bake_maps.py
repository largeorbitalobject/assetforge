import bpy, sys, math, mathutils, json
glb, outdir = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
obj=[o for o in bpy.context.scene.objects if o.type=='MESH'][0]
bpy.context.view_layer.objects.active=obj; obj.select_set(True)
mn=mathutils.Vector((1e9,)*3); mx=mathutils.Vector((-1e9,)*3)
for c in obj.bound_box:
    w=obj.matrix_world@mathutils.Vector(c); mn=mathutils.Vector(map(min,mn,w)); mx=mathutils.Vector(map(max,mx,w))
center=(mn+mx)/2; rad=max(mx-mn)*0.5 or 1.0
cam_dist=rad*3.0; ortho=rad*2.25; RES=1024
VIEWS={'front':(0,8),'fl':(45,8),'fr':(-45,8),'back':(180,8),'bl':(135,8),'br':(-135,8)}
scn=bpy.context.scene
cam_d=bpy.data.cameras.new("C"); cam=bpy.data.objects.new("C",cam_d); scn.collection.objects.link(cam)
cam_d.type='ORTHO'; cam_d.ortho_scale=ortho; scn.camera=cam
def place(az,el):
    a=math.radians(az); e=math.radians(el)
    cam.location=center+mathutils.Vector((cam_dist*math.sin(a)*math.cos(e),-cam_dist*math.cos(a)*math.cos(e),cam_dist*math.sin(e)))
    d=center-cam.location; cam.rotation_euler=d.to_track_quat('-Z','Y').to_euler(); bpy.context.view_layer.update()
cams={'meta':{'cam_dist':cam_dist,'rad':rad,'ortho':ortho,'res':RES,'center':list(center)},'views':{}}
for v,(az,el) in VIEWS.items():
    place(az,el); cams['views'][v]={'M':[list(r) for r in cam.matrix_world]}
json.dump(cams, open(f"{outdir}/cams.json","w"))
AT=2048
scn.render.engine='CYCLES'; scn.cycles.device='CPU'; scn.cycles.samples=1
scn.render.bake.margin=12; scn.render.bake.use_clear=True; scn.view_settings.view_transform='Raw'
def bake(kind, path):
    m=bpy.data.materials.new(kind); m.use_nodes=True; nt=m.node_tree; nt.nodes.clear()
    geo=nt.nodes.new("ShaderNodeNewGeometry"); em=nt.nodes.new("ShaderNodeEmission"); out=nt.nodes.new("ShaderNodeOutputMaterial")
    if kind=="pos":
        sub=nt.nodes.new("ShaderNodeVectorMath"); sub.operation='SUBTRACT'; sub.inputs[1].default_value=center
        mul=nt.nodes.new("ShaderNodeVectorMath"); mul.operation='SCALE'; mul.inputs['Scale'].default_value=1.0/(2*rad)
        add=nt.nodes.new("ShaderNodeVectorMath"); add.operation='ADD'; add.inputs[1].default_value=(0.5,0.5,0.5)
        nt.links.new(geo.outputs['Position'],sub.inputs[0]); nt.links.new(sub.outputs[0],mul.inputs[0])
        nt.links.new(mul.outputs[0],add.inputs[0]); nt.links.new(add.outputs[0],em.inputs['Color'])
    else:
        mul=nt.nodes.new("ShaderNodeVectorMath"); mul.operation='SCALE'; mul.inputs['Scale'].default_value=0.5
        add=nt.nodes.new("ShaderNodeVectorMath"); add.operation='ADD'; add.inputs[1].default_value=(0.5,0.5,0.5)
        nt.links.new(geo.outputs['Normal'],mul.inputs[0]); nt.links.new(mul.outputs[0],add.inputs[0]); nt.links.new(add.outputs[0],em.inputs['Color'])
    nt.links.new(em.outputs[0],out.inputs['Surface'])
    img=bpy.data.images.new(kind, AT, AT, alpha=True, float_buffer=True)
    tex=nt.nodes.new("ShaderNodeTexImage"); tex.image=img; nt.nodes.active=tex
    obj.data.materials.clear(); obj.data.materials.append(m)
    scn.cycles.bake_type='EMIT'; bpy.ops.object.bake(type='EMIT')
    img.file_format='OPEN_EXR'; img.filepath_raw=path; img.save(); print("baked",kind)
bake("pos", f"{outdir}/pos_atlas.exr"); bake("nrm", f"{outdir}/nrm_atlas.exr")
print("BAKE_MAPS_DONE")
