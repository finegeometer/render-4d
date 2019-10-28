//! Display a perspective projection of a 4D scene.
//!
//! In the same way that a 3D scene can be projected onto a 2D screen,
//! a 4D scene can be projected onto a 3D screen.
//!
//! This crate provides a [`Renderer`] that handles
//! projecting from 4D to 3D, with hidden surface removal,
//! and then 3D to 2D.
//!
//! Unlike games like <https://www.urticator.net/maze/>, which display the scene as a line-art drawing,
//! this will draw 2D surfaces. (A plane-art drawing.)
//!
//! # Examples
//!
//! ## Basics
//!
//! ```no_run
//! # use render_4d::*;
//! let canvas: web_sys::HtmlCanvasElement = unimplemented!();
//! # let texture_data: TextureData = unimplemented!();
//! # let hypercube_mesh: Mesh = unimplemented!();
//!
//! let renderer = Renderer::new(&canvas, texture_data, hypercube_mesh);
//!
//!
//! // On each frame ...
//!
//! # let canvas_width = 800;
//! # let canvas_height = 800;
//! # let four_camera: nalgebra::Matrix5<f32> = unimplemented!();
//! # let three_camera: nalgebra::Matrix4<f32> = unimplemented!();
//! let uniforms = Uniforms {
//!     four_camera,
//!     three_camera,
//!     three_screen_size: [1., 1., 1.],
//! };
//!
//! renderer.clear_screen();
//! renderer.render(([0, 0], [canvas_width, canvas_height]), &uniforms);
//!
//! ```
//!
//! ## Bigger example
//!
//! [Link](https://github.com/finegeometer/render-4d-rs/blob/master/examples/example/README.md)
//!
//! # Limitations
//!
//! This crate currently only supports WebAssembly.
//!
//! This crate is currently not suitable for cases where there is continuous motion in the 4D scene,
//! as changing the scene recompiles a fragment shader.
//!
//! This crate is nowhere close to stable. Large API changes will probably be made in the future (unless this project is abandoned).
//!
//! # Panics
//!
//! Panics should only happen if the WebGL API throws exceptions, or if the projection matrix is singular.
//!
//! [`Renderer`]: struct.Renderer.html

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

mod mesh;
pub use mesh::Mesh;

/// The required information for a texture.
/// ```
/// # use render_4d::*;
/// let texture_data = TextureData {
///     width: 2,
///     height: 2,
///     depth: 2,
///     data: &[
///         0x00, 0x00, 0x00, 0xFF,
///         0x00, 0x00, 0xFF, 0xFF,
///         0x00, 0xFF, 0x00, 0xFF,
///         0x00, 0xFF, 0xFF, 0xFF,
///         0xFF, 0x00, 0x00, 0xFF,
///         0xFF, 0x00, 0xFF, 0xFF,
///         0xFF, 0xFF, 0x00, 0xFF,
///         0xFF, 0xFF, 0xFF, 0xFF,
///     ],
/// };
/// ```
#[derive(Debug, Copy, Clone)]
pub struct TextureData<'r> {
    /// The width of the texture, in pixels.
    pub width: i32,
    /// The height of the texture, in pixels.
    pub height: i32,
    /// The depth of the texture, in pixels.
    pub depth: i32,
    /// An array of bytes, which should be of width `4 * width * height * depth`.
    /// Pixel (x,y) of the texture has RGBA values `data[((z * height + y) * width + x) * 4 .. ((z * height + y) * width + x) * 4 + 3]`.
    pub data: &'r [u8],
}

/// ```
/// # use render_4d::*;
/// let uniforms = Uniforms {
///     four_camera: nalgebra::Matrix5::new(
///         1., 0., 0., 0., 0.,
///         0., 1., 0., 0., 0.,
///         0., 0., 1., 0., 0.,
///         0., 0., 0., -2.5/1.5, -2./1.5,
///         0., 0., 0., -1., 0.,
///     ),
///     three_camera: nalgebra::Matrix4::new(
///         1., 0., 0., 0.,
///         0., 1., 0., 0.,
///         0., 0., -100.01/99.99, -2./99.99,
///         0., 0., -1., 0.,
///     ),
///     three_screen_size: [1., 1., 1.],
/// };
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Uniforms {
    /// The projection matrix for the camera in 4D space. Should be invertible.
    pub four_camera: nalgebra::Matrix5<f32>,
    /// The projection matrix for the camera in the 3D screen.
    pub three_camera: nalgebra::Matrix4<f32>,
    /// The size of the 3D screen.
    pub three_screen_size: [f32; 3],
}

/// Handles rendering of a 4D scene.
/// ```no_run
/// # use render_4d::*;
/// # let canvas: web_sys::HtmlCanvasElement = unimplemented!();
/// # let texture_data: TextureData = unimplemented!();
/// # let hypercube_mesh: Mesh = unimplemented!();
/// let renderer = Renderer::new(&canvas, texture_data, hypercube_mesh);
///
/// // On each frame...
/// # let canvas_width = 800;
/// # let canvas_height = 800;
/// # let uniforms: Uniforms = unimplemented!();
/// renderer.clear_screen();
/// renderer.render(([0, 0], [canvas_width, canvas_height]), &uniforms);
/// ```
#[derive(Debug)]
pub struct Renderer {
    permanent: RendererPermanent,
    regenerated: RendererRegenerated,
}

impl Renderer {
    /// Create a [`Renderer`].
    /// Arguments:
    /// - A canvas. This will be used for rendering.
    /// - Texture data. There is only one texture; if you want more, draw several pictures on different parts of the texture.
    /// - A mesh. This is the 4D scene that will be rendered.
    ///
    /// [`Renderer`]: struct.Renderer.html
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

    /// Update the mesh.
    /// This is useful if something changes in the world.
    ///
    /// ## Warning
    /// At present, this function recompiles a fragment shader, so it might be somewhat slow.
    pub fn update_mesh(&mut self, mesh: Mesh) {
        self.regenerated = self.permanent.regenerate_mesh(mesh);
    }

    /// Clear the screen. This should be done at the beginning of each frame.
    pub fn clear_screen(&self) {
        self.permanent.gl.clear_color(1., 1., 1., 1.);
        self.permanent.gl.clear(GL::COLOR_BUFFER_BIT);
    }

    /// Render the 4D scene to a 3D screen, and then display the result on some part of the canvas.
    /// You should call this once per frame, or twice if you are in VR.
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
            uniforms.three_camera.as_slice(),
        );

        gl.uniform_matrix4fv_with_f32_array(
            self.regenerated.uniform_locations.four_camera_a.as_ref(),
            false,
            four_camera_a.as_slice(),
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

        gl.bind_texture(GL::TEXTURE_3D, Some(&self.permanent.texture));
        gl.uniform1i(self.regenerated.uniform_locations.texture.as_ref(), 0);

        gl.viewport(viewport.0[0], viewport.0[1], viewport.1[0], viewport.1[1]);

        gl.draw_arrays(GL::TRIANGLES, 0, self.regenerated.num_tris);
    }
}

#[derive(Debug)]
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
        gl.bind_texture(GL::TEXTURE_3D, Some(&texture));

        gl.tex_image_3d_with_opt_u8_array(
            GL::TEXTURE_3D,
            0,                   // level
            GL::RGBA as i32,     // internal_format
            texture_data.width,  // width
            texture_data.height, // height
            texture_data.depth,  // depth
            0,                   // border
            GL::RGBA,            // format
            GL::UNSIGNED_BYTE,   // type
            Some(texture_data.data),
        )
        .expect_throw("tex_image_3d failed.");
        gl.tex_parameteri(GL::TEXTURE_3D, GL::TEXTURE_MIN_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_3D, GL::TEXTURE_MAG_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_3D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(GL::TEXTURE_3D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);

        Self {
            gl,
            vertex_shader,
            vao,
            vertex_buffer,
            texture,
        }
    }
}

#[derive(Debug)]
struct RendererRegenerated {
    gl: GL,

    program: web_sys::WebGlProgram,
    uniform_locations: UniformLocations,

    pos_loc: u32,
    facets_loc: u32,

    num_tris: i32,
}

impl Drop for RendererRegenerated {
    fn drop(&mut self) {
        self.gl.disable_vertex_attrib_array(self.pos_loc);
        self.gl.disable_vertex_attrib_array(self.facets_loc);

        self.gl.delete_program(Some(&self.program));
    }
}

impl RendererPermanent {
    fn regenerate_mesh(&self, mesh: Mesh) -> RendererRegenerated {
        let fragment_shader = self.gl.create_shader(GL::FRAGMENT_SHADER).unwrap_throw();
        self.gl.shader_source(&fragment_shader, &mesh.to_shader());
        self.gl.compile_shader(&fragment_shader);

        web_sys::console::log_1(
            &self
                .gl
                .get_shader_info_log(&fragment_shader)
                .unwrap_throw()
                .into(),
        );

        let program = self.gl.create_program().unwrap_throw();
        self.gl.attach_shader(&program, &self.vertex_shader);
        self.gl.attach_shader(&program, &fragment_shader);
        self.gl.link_program(&program);

        self.gl.delete_shader(Some(&fragment_shader));

        let pos_loc = self.gl.get_attrib_location(&program, "pos") as u32;
        let facets_loc = self.gl.get_attrib_location(&program, "facets") as u32;

        self.gl
            .bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        self.gl.enable_vertex_attrib_array(pos_loc);
        self.gl
            .vertex_attrib_pointer_with_i32(pos_loc, 4, GL::FLOAT, false, 6 * 4, 0);
        self.gl.enable_vertex_attrib_array(facets_loc);
        self.gl
            .vertex_attrib_pointer_with_i32(facets_loc, 2, GL::FLOAT, false, 6 * 4, 4 * 4);

        let data = mesh.to_vertex_buffer();

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
            facets_loc,
            num_tris: (data.len() / 6) as i32,
        }
    }
}

#[derive(Debug)]
struct UniformLocations {
    four_camera_a: Option<web_sys::WebGlUniformLocation>,
    four_camera_b: Option<web_sys::WebGlUniformLocation>,
    three_camera: Option<web_sys::WebGlUniformLocation>,
    four_camera_pos: Option<web_sys::WebGlUniformLocation>,
    three_screen_size: Option<web_sys::WebGlUniformLocation>,
    texture: Option<web_sys::WebGlUniformLocation>,
}

type GL = web_sys::WebGl2RenderingContext;

const VERTEX_SHADER_SOURCE: &str = r"#version 300 es

in vec4 pos;
// Secretly integers
in vec2 facets;

out vec4 vpos;
// Secretly integers
out vec2 vfacets;
out vec4 v_three_screen_pos;

uniform mat4 four_camera_a;
uniform vec4 four_camera_b;

uniform mat4 three_camera;

void main() {
    vpos = pos;
    vfacets = facets;

    v_three_screen_pos = four_camera_a * pos + four_camera_b;

    gl_Position = three_camera * v_three_screen_pos.yxzw;
}
";

fn as_f32_array(v: &[f32]) -> js_sys::Float32Array {
    let memory_buffer = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .unwrap_throw()
        .buffer();

    let location = v.as_ptr() as u32 / 4;

    js_sys::Float32Array::new(&memory_buffer).subarray(location, location + v.len() as u32)
}
