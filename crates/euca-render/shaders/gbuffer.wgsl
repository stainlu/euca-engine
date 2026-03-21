struct InstanceData { model: mat4x4<f32>, normal_matrix: mat4x4<f32> }
struct MaterialUniforms { albedo: vec4<f32>, metallic: f32, roughness: f32, has_normal_map: f32, has_metallic_roughness_tex: f32, emissive: vec3<f32>, has_emissive_tex: f32, has_ao_tex: f32, alpha_mode: f32, alpha_cutoff: f32 }
struct GBufferSceneUniforms { camera_vp: mat4x4<f32> }
@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(0) var<uniform> scene: GBufferSceneUniforms;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var albedo_tex: texture_2d<f32>;
@group(2) @binding(2) var albedo_sampler: sampler;
@group(2) @binding(3) var normal_tex: texture_2d<f32>;
@group(2) @binding(4) var metallic_roughness_tex: texture_2d<f32>;
@group(2) @binding(5) var ao_tex: texture_2d<f32>;
@group(2) @binding(6) var emissive_tex: texture_2d<f32>;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) tangent: vec3<f32>, @location(3) uv: vec2<f32> }
struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) world_normal: vec3<f32>, @location(2) world_tangent: vec3<f32>, @location(3) uv: vec2<f32> }
@vertex fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput { let model = instances[iid].model; let normal_mat = instances[iid].normal_matrix; var out: VertexOutput; let world_pos = (model * vec4<f32>(in.position, 1.0)).xyz; out.clip_position = scene.camera_vp * vec4<f32>(world_pos, 1.0); out.world_pos = world_pos; out.world_normal = normalize((normal_mat * vec4<f32>(in.normal, 0.0)).xyz); out.world_tangent = normalize((model * vec4<f32>(in.tangent, 0.0)).xyz); out.uv = in.uv; return out; }
struct GBufferOutput { @location(0) albedo: vec4<f32>, @location(1) normal_roughness: vec4<f32>, @location(2) material_params: vec4<f32> }
@fragment fn fs_main(in: VertexOutput) -> GBufferOutput { let tex_color = textureSample(albedo_tex, albedo_sampler, in.uv); let albedo = material.albedo.rgb * tex_color.rgb; let alpha = material.albedo.a * tex_color.a; if material.alpha_mode > 0.5 && material.alpha_mode < 1.5 { if alpha < material.alpha_cutoff { discard; } } var metallic = material.metallic; var roughness = material.roughness; if material.has_metallic_roughness_tex > 0.5 { let mr = textureSample(metallic_roughness_tex, albedo_sampler, in.uv); roughness = material.roughness * mr.g; metallic = material.metallic * mr.b; } roughness = max(roughness, 0.04); var N: vec3<f32>; if material.has_normal_map > 0.5 { let s = textureSample(normal_tex, albedo_sampler, in.uv).rgb; let tn = s * 2.0 - 1.0; let T = normalize(in.world_tangent); let Nv = normalize(in.world_normal); let B = cross(Nv, T); N = normalize(T * tn.x + B * tn.y + Nv * tn.z); } else { N = normalize(in.world_normal); } var ao = 1.0; if material.has_ao_tex > 0.5 { ao = textureSample(ao_tex, albedo_sampler, in.uv).r; } var ef = 0.0; var ec = material.emissive; if material.has_emissive_tex > 0.5 { ec = ec * textureSample(emissive_tex, albedo_sampler, in.uv).rgb; } if dot(ec, vec3<f32>(0.2126, 0.7152, 0.0722)) > 0.001 { ef = 1.0; } var out: GBufferOutput; out.albedo = vec4<f32>(albedo, alpha); out.normal_roughness = vec4<f32>(N * 0.5 + 0.5, roughness); out.material_params = vec4<f32>(metallic, ao, ef, 0.0); return out; }
