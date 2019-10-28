#![forbid(unsafe_code)]

mod fps;
// mod world;

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
pub fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    web_sys::window()
        .unwrap_throw()
        .request_animation_frame(&State::new().0.borrow().animation_frame_closure)
        .unwrap_throw();
}

#[derive(Clone)]
struct State(Rc<RefCell<Model>>);

struct Model {
    animation_frame_closure: js_sys::Function,
    keys: HashSet<String>,
    fps: Option<fps::FrameCounter>,
    renderer: render_4d::Renderer,
    vr_status: VrStatus,

    window: web_sys::Window,
    document: web_sys::Document,
    canvas: web_sys::HtmlCanvasElement,
    info_box: web_sys::HtmlParagraphElement,
    slice_slider: web_sys::HtmlInputElement,

    position: nalgebra::Vector4<f32>,
    orientation: nalgebra::UnitQuaternion<f32>,
}

enum VrStatus {
    Searching,
    NotSupported,
    NotFound,
    Known(web_sys::VrDisplay),
    RequestedPresentation(web_sys::VrDisplay),
    Presenting(web_sys::VrDisplay),
}

impl Model {
    fn new() -> Self {
        let window = web_sys::window().unwrap_throw();
        let document = window.document().unwrap_throw();
        let body = document.body().unwrap_throw();

        let canvas = document
            .create_element("canvas")
            .unwrap_throw()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap_throw();
        canvas.set_attribute("width", "1600").unwrap_throw();
        canvas.set_attribute("height", "800").unwrap_throw();
        body.append_child(&canvas).unwrap_throw();

        let info_box = document
            .create_element("p")
            .unwrap_throw()
            .dyn_into::<web_sys::HtmlParagraphElement>()
            .unwrap_throw();
        body.append_child(&info_box).unwrap_throw();

        let slice_slider = document
            .create_element("input")
            .unwrap_throw()
            .dyn_into::<web_sys::HtmlInputElement>()
            .unwrap_throw();

        slice_slider.set_type("range");
        slice_slider.set_min("1");
        slice_slider.set_max("10");
        slice_slider.set_value("10");
        body.append_child(&slice_slider).unwrap_throw();

        let renderer = render_4d::Renderer::new(
            &canvas,
            // render_4d::TextureData {
            //     width: 64,
            //     height: 64,
            //     data: include_bytes!("../resources/texture"),
            // },
            // render_4d::TextureData {
            //     width: 2,
            //     height: 2,
            //     depth: 2,
            //     data: &[
            //         0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xDD, 0xFF, 0x00, 0xDD, 0x00, 0xFF, 0x00,
            //         0xDD, 0xDD, 0xFF, 0xDD, 0x00, 0x00, 0xFF, 0xDD, 0x00, 0xDD, 0xFF, 0xDD, 0xDD,
            //         0x00, 0xFF, 0xDD, 0xDD, 0xDD, 0xFF,
            //     ],
            // },
            render_4d::TextureData {
                width: 3,
                height: 3,
                depth: 3,
                data: &[
                    0x80, 0x80, 0x80, 0xFF, 0xFF, 0x80, 0x80, 0xFF, 0x80, 0x80, 0x80, 0xFF, 0x80,
                    0xFF, 0x80, 0xFF, 0xFF, 0xFF, 0x80, 0xFF, 0x80, 0xFF, 0x80, 0xFF, 0x80, 0x80,
                    0x80, 0xFF, 0xFF, 0x80, 0x80, 0xFF, 0x80, 0x80, 0x80, 0xFF, 0x80, 0x80, 0xFF,
                    0xFF, 0xFF, 0x80, 0xFF, 0xFF, 0x80, 0x80, 0xFF, 0xFF, 0x80, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0x80, 0xFF, 0xFF, 0xFF, 0x80, 0x80, 0xFF, 0xFF, 0xFF,
                    0x80, 0xFF, 0xFF, 0x80, 0x80, 0xFF, 0xFF, 0x80, 0x80, 0xFF, 0xFF, 0xFF, 0x80,
                    0xFF, 0xFF, 0x80, 0x80, 0xFF, 0xFF, 0x80, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0x80, 0xFF, 0xFF, 0xFF, 0x80, 0x80, 0xFF, 0xFF, 0xFF, 0x80, 0xFF, 0xFF,
                    0x80, 0x80, 0xFF, 0xFF,
                ],
            },
            render_4d::Mesh::hypercube(
                [
                    nalgebra::Vector4::new(-1., -1., -1., -6.),
                    nalgebra::Vector4::new(1., 1., 1., -4.),
                ],
                [[0., 0., 0.], [1., 1., 1.]],
            ),
            // world::scene(),
        );

        Self {
            animation_frame_closure: JsValue::undefined().into(),
            fps: None,
            keys: HashSet::new(),
            renderer,
            vr_status: VrStatus::Searching,

            window,
            document,
            canvas,
            info_box,
            slice_slider,

            position: nalgebra::Vector4::zeros(),
            orientation: nalgebra::UnitQuaternion::identity(),
        }
    }
}

pub enum Msg {
    Click,
    MouseMove([i32; 2]),
    KeyDown(String),
    KeyUp(String),

    GotVRDisplays(js_sys::Array),
    DisplayPresenting(web_sys::VrDisplay),
}

impl State {
    fn new() -> Self {
        let out = Self(Rc::new(RefCell::new(Model::new())));

        {
            let model: &mut Model = &mut out.0.borrow_mut();

            let navigator: web_sys::Navigator = model.window.navigator();
            if js_sys::Reflect::has(&navigator, &"getVRDisplays".into()).unwrap_throw() {
                let state = out.clone();
                let closure = Closure::once(move |vr_displays| {
                    state.update(Msg::GotVRDisplays(js_sys::Array::from(&vr_displays)));
                });
                navigator.get_vr_displays().unwrap_throw().then(&closure);
                closure.forget();
            } else {
                web_sys::console::error_1(
                    &"WebVR is not supported by this browser, on this computer.".into(),
                );

                model.vr_status = VrStatus::NotSupported;
            }

            out.event_listener(&model.canvas, "mousedown", move |_| Msg::Click);
            out.event_listener(&model.canvas, "mousemove", |evt| {
                let evt = evt.dyn_into::<web_sys::MouseEvent>().unwrap_throw();
                Msg::MouseMove([evt.movement_x(), evt.movement_y()])
            });
            out.event_listener(&model.document, "keydown", |evt| {
                let evt = evt.dyn_into::<web_sys::KeyboardEvent>().unwrap_throw();
                Msg::KeyDown(evt.key())
            });
            out.event_listener(&model.document, "keyup", |evt| {
                let evt = evt.dyn_into::<web_sys::KeyboardEvent>().unwrap_throw();
                Msg::KeyUp(evt.key())
            });

            let state = out.clone();
            let closure: Closure<dyn FnMut(f64)> = Closure::wrap(Box::new(move |timestamp| {
                state.frame(timestamp);
            }));
            model.animation_frame_closure =
                closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
            closure.forget();
        }

        out
    }

    fn update(&self, msg: Msg) {
        let model: &mut Model = &mut self.0.borrow_mut();

        match msg {
            Msg::Click => {
                if model.document.pointer_lock_element().is_none() {
                    model.canvas.request_pointer_lock();
                }
                if let VrStatus::Known(display) = &model.vr_status {
                    let mut layer = web_sys::VrLayer::new();
                    layer.source(Some(&model.canvas));
                    let layers = js_sys::Array::new();
                    layers.set(0, layer.as_ref().clone());

                    let state = self.clone();
                    let display_ = display.clone();
                    let closure =
                        Closure::once(move |_| state.update(Msg::DisplayPresenting(display_)));
                    display
                        .request_present(&layers)
                        .unwrap_throw()
                        .then(&closure);
                    closure.forget();

                    model.vr_status = VrStatus::RequestedPresentation(display.clone());
                }
            }
            Msg::KeyDown(k) => {
                model.keys.insert(k.to_lowercase());
            }
            Msg::KeyUp(k) => {
                model.keys.remove(&k.to_lowercase());
            }
            Msg::MouseMove([x, y]) => {
                if model.document.pointer_lock_element().is_some() {
                    model.orientation *= nalgebra::UnitQuaternion::new(nalgebra::Vector3::new(
                        y as f32 * 3e-3,
                        -x as f32 * 3e-3,
                        0.,
                    ));
                }
            }
            Msg::GotVRDisplays(vr_displays) => {
                if vr_displays.length() == 0 {
                    model.vr_status = VrStatus::NotFound;
                } else {
                    model.vr_status = VrStatus::Known(vr_displays.get(0).dyn_into().unwrap_throw());
                }
            }
            Msg::DisplayPresenting(display) => model.vr_status = VrStatus::Presenting(display),
        }
    }

    fn frame(&self, timestamp: f64) {
        let model: &mut Model = &mut self.0.borrow_mut();

        if let VrStatus::Presenting(display) = &model.vr_status {
            display
                .request_animation_frame(&model.animation_frame_closure)
                .unwrap_throw();
        } else {
            model
                .window
                .request_animation_frame(&model.animation_frame_closure)
                .unwrap_throw();
        }

        if let Some(fps) = &mut model.fps {
            let dt = fps.frame(timestamp);
            model.info_box.set_inner_text(&format!("{}", fps));

            model.move_player(dt);

            let mut uniforms = render_4d::Uniforms {
                four_camera: model.four_camera(),
                three_camera: nalgebra::Matrix4::new(
                    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., -1., 2.98, 0., 0., -1., 3.,
                ),
                three_screen_size: [1., 1., 0.1 * model.slice_slider.value_as_number() as f32],
            };

            model.renderer.clear_screen();

            if let VrStatus::Presenting(display) = &model.vr_status {
                let frame_data = web_sys::VrFrameData::new().unwrap_throw();
                display.get_frame_data(&frame_data);

                uniforms.three_camera = nalgebra::Matrix4::from_iterator(
                    frame_data.left_projection_matrix().unwrap_throw(),
                ) * nalgebra::Matrix4::from_iterator(
                    frame_data.left_view_matrix().unwrap_throw(),
                );

                model.renderer.render(([0, 0], [800, 800]), &uniforms);

                uniforms.three_camera = nalgebra::Matrix4::from_iterator(
                    frame_data.right_projection_matrix().unwrap_throw(),
                ) * nalgebra::Matrix4::from_iterator(
                    frame_data.right_view_matrix().unwrap_throw(),
                );

                model.renderer.render(([800, 0], [800, 800]), &uniforms);

                display.submit_frame();
            } else {
                model.renderer.render(([0, 0], [800, 800]), &uniforms);
            }
        } else {
            model.fps = Some(<fps::FrameCounter>::new(timestamp));
        }
    }

    fn event_listener(
        &self,
        target: &web_sys::EventTarget,
        event: &str,
        msg: impl Fn(web_sys::Event) -> Msg + 'static,
    ) {
        let state = self.clone();
        let closure: Closure<dyn FnMut(web_sys::Event)> = Closure::wrap(Box::new(move |evt| {
            state.update(msg(evt));
        }));
        target
            .add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())
            .unwrap_throw();
        closure.forget();
    }
}

impl Model {
    #[rustfmt::skip]
    fn four_camera(&self) -> nalgebra::Matrix5<f32> {
        let fov: f32 = 1.57;
        let x = (fov / 2.).tan();
        let projection = nalgebra::Matrix5::new(
            x, 0., 0., 0., 0.,
            0., x, 0., 0., 0.,
            0., 0., x, 0., 0.,
            0., 0., 0., -2.5/1.5, -2./1.5,
            0., 0., 0., -1., 0.,
        );

        let mut rotation = nalgebra::Matrix5::identity();
        rotation.fixed_slice_mut::<nalgebra::U3, nalgebra::U3>(1, 1)
            .copy_from(&self.orientation.conjugate().to_rotation_matrix().matrix());

        let translation = nalgebra::Translation::from(-self.position).to_homogeneous();

        projection * rotation * translation
    }

    fn move_player(&mut self, dt: f64) {
        let mut orientation = nalgebra::Matrix4::identity();
        orientation
            .fixed_slice_mut::<nalgebra::U3, nalgebra::U3>(1, 1)
            .copy_from(&self.orientation.to_rotation_matrix().matrix());
        orientation *= dt as f32;

        if self.keys.contains(" ") {
            self.position += orientation * nalgebra::Vector4::new(1., 0., 0., 0.);
        }
        if self.keys.contains("shift") {
            self.position += orientation * nalgebra::Vector4::new(-1., 0., 0., 0.);
        }
        if self.keys.contains("w") {
            self.position += orientation * nalgebra::Vector4::new(0., 0., 0., -1.);
        }
        if self.keys.contains("s") {
            self.position += orientation * nalgebra::Vector4::new(0., 0., 0., 1.);
        }
        if self.keys.contains("d") {
            self.position += orientation * nalgebra::Vector4::new(0., 1., 0., 0.);
        }
        if self.keys.contains("a") {
            self.position += orientation * nalgebra::Vector4::new(0., -1., 0., 0.);
        }
        if self.keys.contains("q") {
            self.position += orientation * nalgebra::Vector4::new(0., 0., 1., 0.);
        }
        if self.keys.contains("e") {
            self.position += orientation * nalgebra::Vector4::new(0., 0., -1., 0.);
        }
    }
}
