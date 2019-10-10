use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: nalgebra::Vector4<f32>,
    pub texture_coordinate: [f32; 2],
}

// TODO: Document whether inside is positive or negative
#[derive(Clone)]
pub struct Region {
    pub facets: Vec<nalgebra::RowVector5<f32>>,
}

pub struct TextureData<'r> {
    pub width: i32,
    pub height: i32,
    pub data: &'r [u8],
}

pub struct Uniforms {
    pub four_camera: nalgebra::Matrix5<f32>,
    pub three_camera: nalgebra::Matrix4<f32>,
    pub three_screen_size: [f32; 3],
}

pub struct Mesh {
    pub triangles: Box<dyn Iterator<Item = [Vertex; 3]>>,
    pub regions: Box<dyn Iterator<Item = Region>>,
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            triangles: Box::new(std::iter::empty()),
            regions: Box::new(std::iter::empty()),
        }
    }
}

impl std::ops::Add for Mesh {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            triangles: Box::new(self.triangles.chain(other.triangles)),
            regions: Box::new(self.regions.chain(other.regions)),
        }
    }
}

impl Mesh {
    pub fn transform(mat: nalgebra::Matrix5<f32>) -> Option<impl Fn(Mesh) -> Mesh> {
        let mat_inv = mat.try_inverse()?;
        Some(move |Mesh { triangles, regions }| Mesh {
            triangles: Box::new(triangles.map(move |mut vertices| {
                for vertex in vertices.iter_mut() {
                    let old_pos: nalgebra::Vector5<f32> = vertex.position.fixed_resize(1.);
                    let new_pos: nalgebra::Vector5<f32> = mat * old_pos;
                    vertex.position = new_pos.fixed_rows::<nalgebra::U4>(0) / new_pos[4];
                }
                vertices
            })),
            regions: Box::new(regions.map(move |mut region| {
                for facet in region.facets.iter_mut() {
                    *facet *= mat_inv;
                }
                region
            })),
        })
    }
}

pub struct Renderer {
    permanent: RendererPermanent,
    regenerated: RendererRegenerated,
}

impl Renderer {
    pub fn new(canvas: &web_sys::HtmlCanvasElement, texture_data: TextureData, mesh: Mesh) -> Self {
        let gl = canvas
            .get_context("webgl2")
            .unwrap_throw()
            .unwrap_throw()
            .dyn_into::<GL>()
            .unwrap_throw();

        let permanent = RendererPermanent::new(gl, texture_data);
        Self {
            regenerated: permanent.regenerate_mesh(mesh),
            permanent,
        }
    }

    pub fn update_mesh(&mut self, mesh: Mesh) {
        self.regenerated = self.permanent.regenerate_mesh(mesh);
    }

    pub fn clear_screen(&self) {
        self.permanent.gl.clear_color(1., 1., 1., 1.);
        self.permanent.gl.clear(GL::COLOR_BUFFER_BIT);
    }

    pub fn render(&self, viewport: ([i32; 2], [i32; 2]), uniforms: &Uniforms) {
        let four_camera_pos: nalgebra::Vector5<f32> = uniforms
            .four_camera
            .try_inverse()
            .expect_throw("Projection matrix should be invertible.")
            * nalgebra::Vector5::new(0., 0., 0., -1., 0.);
        let four_camera_pos: nalgebra::Vector4<f32> =
            four_camera_pos.remove_row(4) / four_camera_pos[4];

        // Ignore depth in the 4->3 projection.
        let four_camera = uniforms.four_camera.remove_row(3);
        let four_camera_a = four_camera.remove_column(4);
        let four_camera_b = four_camera.column(4);

        let gl = &self.permanent.gl;

        //

        gl.use_program(Some(&self.regenerated.program));
        gl.bind_vertex_array(Some(&self.permanent.vao));

        gl.uniform_matrix4fv_with_f32_array(
            self.regenerated.uniform_locations.three_camera.as_ref(),
            false,
            &uniforms
                .three_camera
                .into_iter()
                .copied()
                .collect::<Vec<_>>(),
        );

        gl.uniform_matrix4fv_with_f32_array(
            self.regenerated.uniform_locations.four_camera_a.as_ref(),
            false,
            &four_camera_a.into_iter().copied().collect::<Vec<_>>(),
        );

        gl.uniform4f(
            self.regenerated.uniform_locations.four_camera_b.as_ref(),
            four_camera_b[0],
            four_camera_b[1],
            four_camera_b[2],
            four_camera_b[3],
        );

        gl.uniform4f(
            self.regenerated.uniform_locations.four_camera_pos.as_ref(),
            four_camera_pos[0],
            four_camera_pos[1],
            four_camera_pos[2],
            four_camera_pos[3],
        );

        gl.uniform3f(
            self.regenerated
                .uniform_locations
                .three_screen_size
                .as_ref(),
            uniforms.three_screen_size[0],
            uniforms.three_screen_size[1],
            uniforms.three_screen_size[2],
        );

        gl.bind_texture(GL::TEXTURE_2D, Some(&self.permanent.texture));
        gl.uniform1i(self.regenerated.uniform_locations.texture.as_ref(), 0);

        gl.viewport(viewport.0[0], viewport.0[1], viewport.1[0], viewport.1[1]);

        gl.draw_arrays(GL::TRIANGLES, 0, self.regenerated.num_tris);
    }
}

struct RendererPermanent {
    gl: GL,
    vertex_shader: web_sys::WebGlShader,
    vao: web_sys::WebGlVertexArrayObject,
    vertex_buffer: web_sys::WebGlBuffer,
    texture: web_sys::WebGlTexture,
}

impl Drop for RendererPermanent {
    fn drop(&mut self) {
        self.gl.delete_shader(Some(&self.vertex_shader));
        self.gl.delete_vertex_array(Some(&self.vao));
        self.gl.delete_buffer(Some(&self.vertex_buffer));
        self.gl.delete_texture(Some(&self.texture));
    }
}

impl RendererPermanent {
    fn new(gl: GL, texture_data: TextureData) -> Self {
        gl.enable(GL::BLEND);
        gl.blend_func(GL::DST_COLOR, GL::ZERO);

        let vertex_shader = gl.create_shader(GL::VERTEX_SHADER).unwrap_throw();
        gl.shader_source(&vertex_shader, VERTEX_SHADER_SOURCE);
        gl.compile_shader(&vertex_shader);

        let vao = gl.create_vertex_array().unwrap_throw();
        gl.bind_vertex_array(Some(&vao));

        let vertex_buffer = gl.create_buffer().unwrap_throw();

        let texture = gl.create_texture().unwrap_throw();
        gl.bind_texture(GL::TEXTURE_2D, Some(&texture));

        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            GL::TEXTURE_2D,
            0,                   // level
            GL::RGBA as i32,     // internal_format
            texture_data.width,  // width
            texture_data.height, // height
            0,                   // border
            GL::RGBA,            // format
            GL::UNSIGNED_BYTE,   // type
            Some(texture_data.data),
        )
        .expect_throw("tex_image_2d failed.");
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);

        Self {
            gl,
            vertex_shader,
            vao,
            vertex_buffer,
            texture,
        }
    }
}

struct RendererRegenerated {
    gl: GL,

    program: web_sys::WebGlProgram,
    uniform_locations: UniformLocations,

    pos_loc: u32,
    texcoord_loc: u32,

    num_tris: i32,
}

impl Drop for RendererRegenerated {
    fn drop(&mut self) {
        self.gl.enable_vertex_attrib_array(self.pos_loc);
        self.gl.enable_vertex_attrib_array(self.texcoord_loc);

        self.gl.delete_program(Some(&self.program));
    }
}

impl RendererPermanent {
    fn regenerate_mesh(&self, Mesh { triangles, regions }: Mesh) -> RendererRegenerated {
        let fragment_shader = self.gl.create_shader(GL::FRAGMENT_SHADER).unwrap_throw();
        self.gl
            .shader_source(&fragment_shader, &regions_to_shader(regions));
        self.gl.compile_shader(&fragment_shader);

        let program = self.gl.create_program().unwrap_throw();
        self.gl.attach_shader(&program, &self.vertex_shader);
        self.gl.attach_shader(&program, &fragment_shader);
        self.gl.link_program(&program);

        self.gl.delete_shader(Some(&fragment_shader));

        let pos_loc = self.gl.get_attrib_location(&program, "pos") as u32;
        let texcoord_loc = self.gl.get_attrib_location(&program, "texcoord") as u32;

        self.gl
            .bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        self.gl.enable_vertex_attrib_array(pos_loc);
        self.gl
            .vertex_attrib_pointer_with_i32(pos_loc, 4, GL::FLOAT, false, 6 * 4, 0);
        self.gl.enable_vertex_attrib_array(texcoord_loc);
        self.gl
            .vertex_attrib_pointer_with_i32(texcoord_loc, 2, GL::FLOAT, false, 6 * 4, 4 * 4);

        let mut data = Vec::new();
        for vertices in triangles {
            for Vertex {
                position,
                texture_coordinate,
            } in &vertices
            {
                data.extend(position.into_iter());
                data.push(texture_coordinate[0]);
                data.push(texture_coordinate[1]);
            }
        }

        self.gl.buffer_data_with_array_buffer_view(
            GL::ARRAY_BUFFER,
            &as_f32_array(&data).into(),
            GL::STATIC_DRAW,
        );

        let uniform_locations = UniformLocations {
            four_camera_a: self.gl.get_uniform_location(&program, "four_camera_a"),
            four_camera_b: self.gl.get_uniform_location(&program, "four_camera_b"),
            three_camera: self.gl.get_uniform_location(&program, "three_camera"),
            four_camera_pos: self.gl.get_uniform_location(&program, "four_camera_pos"),
            three_screen_size: self.gl.get_uniform_location(&program, "three_screen_size"),
            texture: self.gl.get_uniform_location(&program, "tex"),
        };

        RendererRegenerated {
            gl: self.gl.clone(),
            uniform_locations,
            program,
            pos_loc,
            texcoord_loc,
            num_tris: (data.len() / 6) as i32,
        }
    }
}

struct UniformLocations {
    four_camera_a: Option<web_sys::WebGlUniformLocation>,
    four_camera_b: Option<web_sys::WebGlUniformLocation>,
    three_camera: Option<web_sys::WebGlUniformLocation>,
    four_camera_pos: Option<web_sys::WebGlUniformLocation>,
    three_screen_size: Option<web_sys::WebGlUniformLocation>,
    texture: Option<web_sys::WebGlUniformLocation>,
}

type GL = web_sys::WebGl2RenderingContext;

fn regions_to_shader(regions: impl Iterator<Item = Region>) -> String {
    let mut out = String::new();

    out += r"#version 300 es

precision mediump float;

in vec4 vpos;
in vec2 vtexcoord;
in vec4 vdata;

out vec4 color;

uniform vec4 four_camera_pos;
uniform sampler2D tex;
uniform vec3 three_screen_size;

vec2 clip(vec2 minmax, vec4 pos, vec4 target, vec4 abcd, float e) {
    float x = dot(abcd, pos) + e;
    float y = dot(abcd, target) + e;

    if (x > y) {
        minmax.x = max(minmax.x, x/(x-y));
    } else {
        minmax.y = min(minmax.y, x/(x-y));
    }

    return minmax;
}

bool intersects_scene(vec4 pos, vec4 target) {
    vec2 minmax;
";

    for r in regions {
        out += "    minmax = vec2(0., 0.999);\n";

        for h in r.facets {
            out += &format!(
                    "    minmax = clip(minmax, pos, target, vec4({:.9}, {:.9}, {:.9}, {:.9}), {:.9});\n",
                    h[0], h[1], h[2], h[3], h[4]
                )
        }

        out += r"
    if (minmax.y > minmax.x) {
        return true;
    }
"
    }

    out += "
    return false;
}

void main() {

    vec3 data = vdata.xyz / vdata.w;

    if (abs(data.x) > three_screen_size.x || abs(data.y) > three_screen_size.y || abs(data.z) > three_screen_size.z || abs(vdata.w) < 0.) {
        // Outside three-screen, so invisible.
        color = vec4(1.);
    } else if (intersects_scene(four_camera_pos, vpos)) {
        // Occluded, so invisible.
        color = vec4(1.);
    } else {
        color = (texture(tex, vtexcoord) + vec4(1.0)) / 2.0;
    }
}

";

    out
}

const VERTEX_SHADER_SOURCE: &str = r"#version 300 es

in vec4 pos;
in vec2 texcoord;

out vec4 vpos;
out vec2 vtexcoord;
out vec4 vdata;

uniform mat4 four_camera_a;
uniform vec4 four_camera_b;

uniform mat4 three_camera;

void main() {
    vpos = pos;
    vtexcoord = texcoord;

    vdata = four_camera_a * pos + four_camera_b;

    gl_Position = three_camera * vdata.yxzw;
}
";

pub fn as_f32_array(v: &[f32]) -> js_sys::Float32Array {
    let memory_buffer = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .unwrap_throw()
        .buffer();

    let location = v.as_ptr() as u32 / 4;

    js_sys::Float32Array::new(&memory_buffer).subarray(location, location + v.len() as u32)
}