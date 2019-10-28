// // impl Mesh {
// //     /// Given a matrix, produce a function that will transform a mesh by that transformation.
// //     /// Fails if the matrix isn't invertible.
// //     /// ```
// //     /// # use render_4d::*;
// //     /// # fn f() -> Option<Mesh> {
// //     /// let mat: nalgebra::Matrix5<f32> = nalgebra::Matrix5::new(
// //     ///     1.,  1.,  1.,  1., 0.,
// //     ///     1.,  1., -1., -1., 0.,
// //     ///     1., -1.,  1., -1., 0.,
// //     ///     1., -1., -1.,  1., 0.,
// //     ///     0.,  0.,  0.,  0., 1.,
// //     /// );
// //     /// # let hypercube_mesh: Mesh = unimplemented!();
// //     /// let transformed_mesh = (Mesh::transform(mat)?)(hypercube_mesh);
// //     /// # Some(transformed_mesh)
// //     /// # }
// //     /// ```
// //     pub fn transform(mat: nalgebra::Matrix5<f32>) -> Option<impl Fn(Mesh) -> Mesh> {
// //         let mat_inv = mat.try_inverse()?;
// //         Some(move |Mesh { triangles, regions }| Mesh {
// //             triangles: Box::new(triangles.map(move |mut vertices| {
// //                 for vertex in vertices.iter_mut() {
// //                     let old_pos: nalgebra::Vector5<f32> = vertex.position.fixed_resize(1.);
// //                     let new_pos: nalgebra::Vector5<f32> = mat * old_pos;
// //                     vertex.position = new_pos.fixed_rows::<nalgebra::U4>(0) / new_pos[4];
// //                 }
// //                 vertices
// //             })),
// //             regions: Box::new(regions.map(move |mut region| {
// //                 for facet in region.facets.iter_mut() {
// //                     *facet *= mat_inv;
// //                 }
// //                 region
// //             })),
// //         })
// //     }
// // }

// // fn regions_to_shader(regions: impl Iterator<Item = Region>) -> String {
// //     let mut out = String::new();

// //     out += r"#version 300 es

// // precision mediump float;

// // in vec4 vpos;
// // in vec3 vtexcoord;
// // in vec4 vdata;

// // out vec4 color;

// // uniform vec4 four_camera_pos;
// // uniform sampler3D tex;
// // uniform vec3 three_screen_size;

// // vec2 clip(vec2 minmax, vec4 pos, vec4 target, vec4 abcd, float e) {
// //     float x = dot(abcd, pos) + e;
// //     float y = dot(abcd, target) + e;

// //     if (x > y) {
// //         minmax.x = max(minmax.x, x/(x-y));
// //     } else {
// //         minmax.y = min(minmax.y, x/(x-y));
// //     }

// //     return minmax;
// // }

// // bool intersects_scene(vec4 pos, vec4 target) {
// //     vec2 minmax;
// // ";

// //     for r in regions {
// //         out += "    minmax = vec2(0., 0.999);\n";

// //         for h in r.facets {
// //             out += &format!(
// //                     "    minmax = clip(minmax, pos, target, vec4({:.9}, {:.9}, {:.9}, {:.9}), {:.9});\n",
// //                     h[0], h[1], h[2], h[3], h[4]
// //                 )
// //         }

// //         out += r"
// //     if (minmax.y > minmax.x) {
// //         return true;
// //     }
// // "
// //     }

// //     out += "
// //     return false;
// // }

// // void main() {

// //     vec3 data = vdata.xyz / vdata.w;

// //     if (abs(data.x) > three_screen_size.x || abs(data.y) > three_screen_size.y || abs(data.z) > three_screen_size.z || abs(vdata.w) < 0.) {
// //         // Outside three-screen.
// //         discard;
// //     } else if (intersects_scene(four_camera_pos, vpos)) {
// //         // Occluded.
// //         discard;
// //     } else {
// //         color = (texture(tex, vtexcoord) + vec4(1.0)) / 2.0;
// //     }
// // }

// // ";

// //     out
// // }

// #[derive(Debug, Default)]
// pub struct MeshBuilder {
//     facets: Vec<FacetData>,
// }

// #[derive(Debug, Copy, Clone)]
// pub struct Facet(isize);

// #[derive(Debug, Clone)]
// struct FacetData {
//     texture_embedding: nalgebra::Matrix5x4<f32>,
//     ridges: Vec<RidgeData>,
// }

// #[derive(Debug, Clone)]
// struct RidgeData {
//     // Relative to source
//     target: isize,
//     // Relative to source
//     edges: Vec<isize>,
// }

// impl MeshBuilder {
//     pub fn create_facet(&mut self, texture_embedding: nalgebra::Matrix5x4<f32>) -> Facet {
//         self.facets.push(FacetData {
//             texture_embedding,
//             ridges: Vec::new(),
//         });
//         Facet(self.facets.len() as isize - 1)
//     }

//     pub fn create_ridge(
//         &mut self,
//         [Facet(facet1), Facet(facet2)]: [Facet; 2],
//         neighbor_loop: &[Facet],
//     ) {
//         self.facets[facet1 as usize].ridges.push(RidgeData {
//             target: facet2 - facet1,
//             edges: neighbor_loop.iter().map(|Facet(x)| x - facet1).collect(),
//         });
//         self.facets[facet2 as usize].ridges.push(RidgeData {
//             target: facet1 - facet2,
//             edges: neighbor_loop.iter().map(|Facet(x)| x - facet2).collect(),
//         });
//     }
// }

// #[derive(Debug, Clone)]
// pub struct Mesh {
//     facets: Vec<FacetData>,
// }

// impl From<MeshBuilder> for Mesh {
//     fn from(MeshBuilder { facets }: MeshBuilder) -> Self {
//         Self { facets }
//     }
// }

// impl Default for Mesh {
//     fn default() -> Self {
//         Self { facets: Vec::new() }
//     }
// }

// impl std::ops::Add for Mesh {
//     type Output = Self;
//     fn add(mut self, other: Self) -> Self {
//         self.facets.extend(other.facets);
//         self
//     }
// }

// impl Mesh {
//     pub(crate) fn to_vertex_buffer(&self) -> Vec<f32> {
//         let mut out = Vec::new();

//         for (
//             facet,
//             FacetData {
//                 texture_embedding,
//                 ridges,
//             },
//         ) in self.facets.iter().enumerate()
//         {
//             let facet = facet as isize;
//             for RidgeData { target, edges } in ridges {
//                 let plane1 =
//                     to_hyperplane(self.facets[(target + facet) as usize].texture_embedding)
//                         * texture_embedding;

//                 let planes: Vec<nalgebra::RowVector4<f32>> = edges
//                     .iter()
//                     .map(|e| {
//                         to_hyperplane(self.facets[(e + facet) as usize].texture_embedding)
//                             * texture_embedding
//                     })
//                     .collect();

//                 let vertices: Vec<(nalgebra::Vector4<f32>, nalgebra::Vector3<f32>)> = planes
//                     .iter()
//                     .zip(planes.iter().cycle().skip(1))
//                     .map(|(&plane2, &plane3)| {
//                         let texcoord = intersect_three_planes([plane1, plane2, plane3]);
//                         let position = texture_embedding * texcoord;

//                         (
//                             position.remove_row(4) / position[4],
//                             texcoord.remove_row(3) / texcoord[3],
//                         )
//                     })
//                     .collect();

//                 for (position, texcoord) in vertices[1..]
//                     .windows(2)
//                     .flat_map(|w| std::iter::once(&vertices[0]).chain(w.iter()))
//                 {
//                     out.extend(position.iter().copied());
//                     out.extend(texcoord.iter().copied());
//                 }
//             }
//         }

//         out
//     }

//     pub(crate) fn to_shader(&self) -> String {
//         let mut initialize_facets = String::new();

//         for (
//             i,
//             FacetData {
//                 texture_embedding, ..
//             },
//         ) in self.facets.iter().enumerate()
//         {
//             let hyperplane = to_hyperplane(*texture_embedding);
//             initialize_facets += &format!("facets_a[{i}] = mat4(", i = i);
//             for x in texture_embedding.remove_row(4).as_slice() {
//                 initialize_facets += &format!("{:.9}, ", x)
//             }
//             initialize_facets += &format!(");\nfacets_b[{i}] = vec4(", i = i);
//             for x in texture_embedding.row(4).iter() {
//                 initialize_facets += &format!("{:.9}, ", x)
//             }
//             initialize_facets += &format!(");\nhyperplanes_a[{i}] = vec4(", i = i);
//             for x in hyperplane.iter().take(4) {
//                 initialize_facets += &format!("{:.9}, ", x)
//             }
//             initialize_facets += &format!(
//                 ");\nhyperplanes_b[{i}] = {val};\n",
//                 i = i,
//                 val = hyperplane[4]
//             );
//         }

//         let intersect_scene: String = unimplemented!();

//         format!(
//             r"#version 300 es
// precision mediump float;
// in vec4 vpos;
// in int vfacet;
// in vec4 v_three_screen_pos;

// out vec4 color;

// uniform vec4 four_camera_pos;
// uniform sampler2D tex;
// uniform vec3 three_screen_size;

// mat4[{num_facets}] facets_a;
// vec4[{num_facets}] facets_b;
// vec4[{num_facets}] hyperplanes_a;
// float[{num_facets}] hyperplanes_b;
// {initialize_facets}

// float intersect_facet(vec4 t0, vec4 t1, int facet) {{
//     float x = dot(hyperplanes_a[facet], t0) + hyperplanes_b[facet];
//     float y = dot(hyperplanes_a[facet], t1) + hyperplanes_b[facet];
//     return x/(x-y);
// }}

// vec4 intersection_color(vec4 start, vec4 end, int facet) {{
//     mat4 a = facets_a[facet];
//     vec4 b = end - start;
//     vec4 c = facets_b[facet];
//     mat4 adjugate_a = mat4(

//         determinant(mat4(a[1][1], a[1][2], a[1][3],    c[1],  a[2][1], a[2][2], a[2][3],    c[1],  a[3][1], a[3][2], a[3][3],    c[1],     b[1],    b[2],    b[3],      1.)),
//         determinant(mat4(a[2][1], a[2][2], a[2][3],    c[1],  a[3][1], a[3][2], a[3][3],    c[1],     b[1],    b[2],    b[3],      1.,  a[0][1], a[0][2], a[0][3],    c[0])),
//         determinant(mat4(a[3][1], a[3][2], a[3][3],    c[1],     b[1],    b[2],    b[3],      1.,  a[0][1], a[0][2], a[0][3],    c[0],  a[1][1], a[1][2], a[1][3],    c[1])),
//         determinant(mat4(   b[1],    b[2],    b[3],      1.,  a[0][1], a[0][2], a[0][3],    c[0],  a[1][1], a[1][2], a[1][3],    c[1],  a[2][1], a[2][2], a[2][3],    c[1])),

//         determinant(mat4(a[1][2], a[1][3],    c[1], a[1][0],  a[2][2], a[2][3],    c[1], a[2][0],  a[3][2], a[3][3],    c[1], a[3][0],     b[2],    b[3],      1.,    b[0])),
//         determinant(mat4(a[2][2], a[2][3],    c[1], a[2][0],  a[3][2], a[3][3],    c[1], a[3][0],     b[2],    b[3],      1.,    b[0],  a[0][2], a[0][3],    c[0], a[0][0])),
//         determinant(mat4(a[3][2], a[3][3],    c[1], a[3][0],     b[2],    b[3],      1.,    b[0],  a[0][2], a[0][3],    c[0], a[0][0],  a[1][2], a[1][3],    c[1], a[1][0])),
//         determinant(mat4(   b[2],    b[3],      1.,    b[0],  a[0][2], a[0][3],    c[0], a[0][0],  a[1][2], a[1][3],    c[1], a[1][0],  a[2][2], a[2][3],    c[1], a[2][0])),

//         determinant(mat4(a[1][3],    c[1], a[1][0], a[1][1],  a[2][3],    c[1], a[2][0], a[2][1],  a[3][3],    c[1], a[3][0], a[3][1],     b[3],      1.,    b[0],    b[1])),
//         determinant(mat4(a[2][3],    c[1], a[2][0], a[2][1],  a[3][3],    c[1], a[3][0], a[3][1],     b[3],      1.,    b[0],    b[1],  a[0][3],    c[0], a[0][0], a[0][1])),
//         determinant(mat4(a[3][3],    c[1], a[3][0], a[3][1],     b[3],      1.,    b[0],    b[1],  a[0][3],    c[0], a[0][0], a[0][1],  a[1][3],    c[1], a[1][0], a[1][1])),
//         determinant(mat4(   b[3],      1.,    b[0],    b[1],  a[0][3],    c[0], a[0][0], a[0][1],  a[1][3],    c[1], a[1][0], a[1][1],  a[2][3],    c[1], a[2][0], a[2][1])),

//         determinant(mat4(   c[1], a[1][0], a[1][1], a[1][2],     c[1], a[2][0], a[2][1], a[2][2],     c[1], a[3][0], a[3][1], a[3][2],       1.,    b[0],    b[1],    b[2])),
//         determinant(mat4(   c[1], a[2][0], a[2][1], a[2][2],     c[1], a[3][0], a[3][1], a[3][2],       1.,    b[0],    b[1],    b[2],     c[0], a[0][0], a[0][1], a[0][2])),
//         determinant(mat4(   c[1], a[3][0], a[3][1], a[3][2],       1.,    b[0],    b[1],    b[2],     c[0], a[0][0], a[0][1], a[0][2],     c[1], a[1][0], a[1][1], a[1][2])),
//         determinant(mat4(     1.,    b[0],    b[1],    b[2],     c[0], a[0][0], a[0][1], a[0][2],     c[1], a[1][0], a[1][1], a[1][2],     c[1], a[2][0], a[2][1], a[2][2])),
//     );

//     vec4 adjugate_b = vec4(
//         determinant(mat4(a[1][0], a[1][1], a[1][2], a[1][3],  a[2][0], a[2][1], a[2][2], a[2][3],  a[3][0], a[3][1], a[3][2], a[3][3],     b[0],    b[1],    b[2],    b[3])),
//         determinant(mat4(a[2][0], a[2][1], a[2][2], a[2][3],  a[3][0], a[3][1], a[3][2], a[3][3],     b[0],    b[1],    b[2],    b[3],  a[0][0], a[0][1], a[0][2], a[0][3])),
//         determinant(mat4(a[3][0], a[3][1], a[3][2], a[3][3],     b[0],    b[1],    b[2],    b[3],  a[0][0], a[0][1], a[0][2], a[0][3],  a[1][0], a[1][1], a[1][2], a[1][3])),
//         determinant(mat4(   b[0],    b[1],    b[2],    b[3],  a[0][0], a[0][1], a[0][2], a[0][3],  a[1][0], a[1][1], a[1][2], a[1][3],  a[2][0], a[2][1], a[2][2], a[2][3])),
//     )

//     vec4 texcoords = adjugate_a * start + adjugate_b;

//     return texture(tex, texcoords.xyz / texcoords.w);
// }}

// bool intersect_scene(vec4 start, vec4 end, out float t, out int facet) {{
//     float[{num_facets}] intersections;
//     for (int i = 0; i < {num_facets}; i++) {{
//         intersections[i] = intersect_facet(start, end, i);
//     }}

//     return {intersect_scene};
// }}

// void main() {{

//     vec3 three_screen_pos = v_three_screen_pos.xyz / v_three_screen_pos.w;

//     if (abs(three_screen_pos.x) > three_screen_size.x || abs(three_screen_pos.y) > three_screen_size.y || abs(three_screen_pos.z) > three_screen_size.z || abs(v_three_screen_pos) < 0.) {{
//         // Outside three-screen, so invisible.
//         color = vec4(1.);
//         discard;
//     }}

//     float t;
//     int facet;

//     if (intersect_scene(four_camera_pos, vpos, t, facet)) {{
//         if (t < 0.999) {{
//             // Occluded, so invisible.
//             discard;
//         }}
//     }}

//     if dot(hyperplanes_a[vfacet], end - start) < 0.0 {{
//         // Facing toward facet
//         color = intersection_color(four_camera_pos, vpos, vfacet);
//     }} else {{
//         // Facing away; raytrace to next facet
//         vec4 v = vpos - four_camera_pos;
//         if (intersect_scene(vpos + 0.001*v, vpos + 1000*v, t, facet)) {{
//             // Hit something else
//             color = intersection_color(four_camera_pos, vpos, facet);
//         }} else {{
//             // Nothing beyond but sky
//             color = vec4(0.5, 0.5, 1.0);
//         }}
//     }}
// }}

// ",
//             num_facets = self.facets.len(),
//             initialize_facets = initialize_facets,
//             intersect_scene = intersect_scene,
//         )
//     }
// }

// fn to_hyperplane(embedding: nalgebra::Matrix5x4<f32>) -> nalgebra::RowVector5<f32> {
//     let x1 = embedding.remove_row(0).determinant();
//     let x2 = -embedding.remove_row(1).determinant();
//     let x3 = embedding.remove_row(2).determinant();
//     let x4 = -embedding.remove_row(3).determinant();
//     let x5 = embedding.remove_row(4).determinant();
//     nalgebra::RowVector5::new(x1, x2, x3, x4, x5)
// }

// fn intersect_three_planes(planes: [nalgebra::RowVector4<f32>; 3]) -> nalgebra::Vector4<f32> {
//     let mat = <nalgebra::Matrix3x4<f32>>::from_rows(&planes);
//     let x1 = mat.remove_column(0).determinant();
//     let x2 = -mat.remove_column(1).determinant();
//     let x3 = mat.remove_column(2).determinant();
//     let x4 = -mat.remove_column(3).determinant();
//     nalgebra::Vector4::new(x1, x2, x3, x4)
// }

mod geometry;

use petgraph::prelude::*;

pub struct Mesh(pub Graph<nalgebra::Matrix5x4<f32>, Vec<nalgebra::Vector4<f32>>, Undirected>);

impl Mesh {
    // Vertex Buffer:
    //   vec4 pos
    //   vec2 facet_indices
    pub(crate) fn to_vertex_buffer(&self) -> Vec<f32> {
        self.0
            .edge_references()
            .flat_map(|e| {
                let vertices: &[nalgebra::Vector4<f32>] = e.weight();

                let facets_in_order = {
                    let m = embedding_inverse(self.0[e.source()]).remove_row(4);
                    let a: nalgebra::Vector4<f32> = m * (vertices[0].insert_row(4, 1.0));
                    let b: nalgebra::Vector4<f32> = m * (vertices[1].insert_row(4, 1.0));
                    let c: nalgebra::Vector4<f32> = m * (vertices[2].insert_row(4, 1.0));

                    let a: nalgebra::Vector3<f32> = a.remove_row(3) / a[3];
                    let b: nalgebra::Vector3<f32> = b.remove_row(3) / b[3];
                    let c: nalgebra::Vector3<f32> = c.remove_row(3) / c[3];

                    let poly_normal = (b - a).cross(&(c - a));

                    let facet_boundary = to_hyperplane(self.0[e.target()]) * self.0[e.source()];

                    let dot: nalgebra::Matrix1<f32> = facet_boundary.remove_column(3) * poly_normal;

                    if dot[0] > 0.0 {
                        [e.source().index() as f32, e.target().index() as f32]
                    } else {
                        [e.target().index() as f32, e.source().index() as f32]
                    }
                };

                vertices[1..].windows(2).flat_map(move |w| {
                    std::iter::once(vertices[0])
                        .chain(w.iter().copied())
                        .flat_map(move |v: nalgebra::Vector4<f32>| {
                            v.as_slice()
                                .to_vec()
                                .into_iter()
                                .chain(std::iter::once(facets_in_order[0]))
                                .chain(std::iter::once(facets_in_order[1]))
                        })
                })
            })
            .collect()
    }
    pub(crate) fn to_shader(&self) -> String {
        let num_facets: usize = self.0.node_count();

        let mut facets_a = String::new();
        let mut facets_b = String::new();
        let mut facets_c = String::new();
        let mut facets_d = String::new();
        for node in self.0.node_indices() {
            let m: nalgebra::Matrix5<f32> = embedding_inverse(self.0[node]);

            facets_a += "mat4(";
            for x in m.remove_row(4).remove_column(4).as_slice() {
                facets_a += &format!("{:.9},", x)
            }
            facets_a.pop();

            facets_b += "vec4(";
            for x in m.remove_row(4).column(4).as_slice() {
                facets_b += &format!("{:.9},", x)
            }
            facets_b.pop();

            facets_c += "vec4(";
            for x in m.row(4).remove_column(4).as_slice() {
                facets_c += &format!("{:.9},", x)
            }
            facets_c.pop();

            facets_d += &format!("{:9},", m[(4, 4)]);

            facets_a += "),";
            facets_b += "),";
            facets_c += "),";
        }
        facets_a.pop();
        facets_b.pop();
        facets_c.pop();
        facets_d.pop();

        let mut intersect_scene = String::new();
        for node in self.0.node_indices() {
            let i = node.index();
            let mut intersection_in_bounds = String::new();
            for neighbor in self.0.neighbors(node) {
                let j = neighbor.index();
                intersection_in_bounds += &format!(
                    "&& ((intersections[{j}] < intersections[{i}]) != backfacing[{j}])",
                    i = i,
                    j = j
                );
            }

            intersect_scene += &format!(
                r"
    if (!backfacing[{i}] && intersections[{i}] > 0.0 && intersections[{i}] < t {intersection_in_bounds}) {{
        facet = {i};
        t = intersections[{i}];
    }}",
                i = i,
                intersection_in_bounds = intersection_in_bounds
            );
        }

        format!(
            r"#version 300 es
precision mediump float;
in vec4 vpos;
// Secretly integers
in vec2 vfacets;
in vec4 v_three_screen_pos;

out vec4 color;

precision lowp sampler3D;

uniform vec4 four_camera_pos;
uniform sampler3D tex;
uniform vec3 three_screen_size;

// Elements of `facet_*` are `mat5`s
// Inverse of this:
// [    5x4    |   5x1  ]
// [           |        ]
// [ embedding |  some  ]
// [           | vector ]

mat4 facets_a[{num_facets}] = mat4[{num_facets}]({facets_a});
vec4 facets_b[{num_facets}] = vec4[{num_facets}]({facets_b});
vec4 facets_c[{num_facets}] = vec4[{num_facets}]({facets_c});
float facets_d[{num_facets}] = float[{num_facets}]({facets_d});


float intersect_facet(vec4 t0, vec4 t1, int facet) {{
    float x = dot(facets_c[facet], t0) + facets_d[facet];
    float y = dot(facets_c[facet], t1) + facets_d[facet];
    return x/(x-y);
}}

vec4 intersection_color(vec4 start, vec4 end, int facet) {{
    float t = intersect_facet(start, end, facet);
    vec4 texcoords = facets_a[facet] * mix(start, end, t) + facets_b[facet];

    return (texture(tex, texcoords.xyz / texcoords.w) + vec4(1.0)) / 2.0;
}}


bool intersect_scene(vec4 start, vec4 end, out float t, out int facet) {{
    float[{num_facets}] intersections;
    bool[{num_facets}] backfacing;
    for (int i = 0; i < {num_facets}; i++) {{
        intersections[i] = intersect_facet(start, end, i);
        backfacing[i] = dot(facets_c[i], end - start) > 0.0;
    }}

    t = 1.0;
    facet = -1;

    {intersect_scene}

    return facet >= 0;
}}

void main() {{

    vec3 three_screen_pos = v_three_screen_pos.xyz / v_three_screen_pos.w;

    if (abs(three_screen_pos.x) > three_screen_size.x || abs(three_screen_pos.y) > three_screen_size.y || abs(three_screen_pos.z) > three_screen_size.z || abs(v_three_screen_pos.w) < 0.) {{
        // Outside three-screen, so invisible.
        color = vec4(1.);
        discard;
    }}

    float t;
    int facet;

    if (intersect_scene(four_camera_pos, mix(four_camera_pos, vpos, 0.999), t, facet)) {{
        // Occluded, so invisible.
        discard;
    }}

    int my_facet;
    if (gl_FrontFacing) {{
        my_facet = int(round(vfacets[0]));
    }} else {{
        my_facet = int(round(vfacets[1]));
    }}

    if (dot(facets_c[my_facet], vpos - four_camera_pos) < 0.0) {{
        // Facing toward facet
        color = intersection_color(four_camera_pos, vpos, my_facet);
    }} else {{
        // Facing away; raytrace to next facet
        if (intersect_scene(mix(four_camera_pos, vpos, 1.001), mix(four_camera_pos, vpos, 1000.0), t, facet)) {{
            // Hit something else
            color = intersection_color(four_camera_pos, vpos, facet);
        }} else {{
            // Nothing beyond but sky
            color = vec4(0.5, 0.5, 1.0, 1.0);
        }}
    }}
}}


",
            num_facets = num_facets,
            facets_a = facets_a,
            facets_b = facets_b,
            facets_c = facets_c,
            facets_d = facets_d,
            intersect_scene = intersect_scene,
        )
    }
}

fn to_hyperplane(embedding: nalgebra::Matrix5x4<f32>) -> nalgebra::RowVector5<f32> {
    let x1 = embedding.remove_row(0).determinant();
    let x2 = -embedding.remove_row(1).determinant();
    let x3 = embedding.remove_row(2).determinant();
    let x4 = -embedding.remove_row(3).determinant();
    let x5 = embedding.remove_row(4).determinant();
    nalgebra::RowVector5::new(x1, x2, x3, x4, x5)
}

fn embedding_inverse(embedding: nalgebra::Matrix5x4<f32>) -> nalgebra::Matrix5<f32> {
    let h: nalgebra::RowVector5<f32> = to_hyperplane(embedding);
    let h: [f32; 5] = h.into();
    let h: nalgebra::Vector5<f32> = h.into();

    let mut m = embedding.insert_column(4, 0.0);
    m.column_mut(4).copy_from(&h);
    m.try_inverse_mut();
    m
}
