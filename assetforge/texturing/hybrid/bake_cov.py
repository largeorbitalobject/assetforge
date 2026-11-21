import bpy, sys
glb, outdir = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
obj=[o for o in bpy.context.scene.objects if o.type=='MESH'][0]
bpy.context.view_layer.objects.active=obj; obj.select_set(True)
scn=bpy.context.scene; scn.render.engine='CYCLES'; scn.cycles.device='CPU'; scn.cycles.samples=1
scn.render.bake.margin=12; scn.render.bake.use_clear=True; scn.view_settings.view_transform='Raw'
AT=2048
m=bpy.data.materials.new("cov"); m.use_nodes=True; nt=m.node_tree; nt.nodes.clear()
em=nt.nodes.new("ShaderNodeEmission"); em.inputs['Color'].default_value=(1,1,1,1); out=nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs[0],out.inputs['Surface'])
img=bpy.data.images.new("cov",AT,AT,alpha=True); tex=nt.nodes.new("ShaderNodeTexImage"); tex.image=img; nt.nodes.active=tex
obj.data.materials.clear(); obj.data.materials.append(m)
scn.cycles.bake_type='EMIT'; bpy.ops.object.bake(type='EMIT')
img.file_format='PNG'; img.filepath_raw=f"{outdir}/cov_atlas.png"; img.save(); print("COV_DONE")
