use render_4d::{Mesh, Region, Vertex};

/// phi^(-0)
const P0: f32 = 1.;
/// phi^(-1)
const P1: f32 = 0.618_034;
/// phi^(-2)
const P2: f32 = 0.381_966_02;

// Vertices of a dodecahedron, or faces of an icosahedron.
const SE: [f32; 3] = [P0, P2, 0.];
const SW: [f32; 3] = [P0, -P2, 0.];
const NE: [f32; 3] = [-P0, P2, 0.];
const NW: [f32; 3] = [-P0, -P2, 0.];
const EU: [f32; 3] = [0., P0, P2];
const ED: [f32; 3] = [0., P0, -P2];
const WU: [f32; 3] = [0., -P0, P2];
const WD: [f32; 3] = [0., -P0, -P2];
const US: [f32; 3] = [P2, 0., P0];
const UN: [f32; 3] = [-P2, 0., P0];
const DS: [f32; 3] = [P2, 0., -P0];
const DN: [f32; 3] = [-P2, 0., -P0];
const SEU: [f32; 3] = [P1, P1, P1];
const SED: [f32; 3] = [P1, P1, -P1];
const SWU: [f32; 3] = [P1, -P1, P1];
const SWD: [f32; 3] = [P1, -P1, -P1];
const NEU: [f32; 3] = [-P1, P1, P1];
const NED: [f32; 3] = [-P1, P1, -P1];
const NWU: [f32; 3] = [-P1, -P1, P1];
const NWD: [f32; 3] = [-P1, -P1, -P1];

// Vertices of an icosahedron, or faces of a dodecahedron.
const ES: [f32; 3] = [P1, P0, 0.];
const WS: [f32; 3] = [P1, -P0, 0.];
const EN: [f32; 3] = [-P1, P0, 0.];
const WN: [f32; 3] = [-P1, -P0, 0.];
const UE: [f32; 3] = [0., P1, P0];
const DE: [f32; 3] = [0., P1, -P0];
const UW: [f32; 3] = [0., -P1, P0];
const DW: [f32; 3] = [0., -P1, -P0];
const SU: [f32; 3] = [P0, 0., P1];
const NU: [f32; 3] = [-P0, 0., P1];
const SD: [f32; 3] = [P0, 0., -P1];
const ND: [f32; 3] = [-P0, 0., -P1];

const DODECAHEDRON_FACES: [[f32; 3]; 12] = [ES, WS, EN, WN, UE, DE, UW, DW, SU, NU, SD, ND];

/// The faces of a rhombic triacontahedron.
/// For each face, first lists the obtuse vertices, then the acute vertices.
/// Faces oriented such that in ([a,b], [c,d]), the triple product of a, b, and c ahould be positive.
/// Sized such that two adjacent vertices always have dot product one.
/// This means that if you reinterpret one set of vertices as faces, they touch the other set of vertices.
#[allow(clippy::type_complexity)]
const RHOMBIC_TRIACONTAHEDRON: [([[f32; 3]; 2], [[f32; 3]; 2]); 30] = [
    ([NE, NW], [NU, ND]),
    ([SE, SW], [SD, SU]),
    ([EU, ED], [EN, ES]),
    ([WU, WD], [WS, WN]),
    ([UN, US], [UE, UW]),
    ([DN, DS], [DW, DE]),
    ([NE, NEU], [EN, NU]),
    ([SE, SEU], [SU, ES]),
    ([NW, NWU], [NU, WN]),
    ([SW, SWU], [WS, SU]),
    ([NE, NED], [ND, EN]),
    ([SE, SED], [ES, SD]),
    ([NW, NWD], [WN, ND]),
    ([SW, SWD], [SD, WS]),
    ([EU, NEU], [UE, EN]),
    ([EU, SEU], [ES, UE]),
    ([WU, NWU], [WN, UW]),
    ([WU, SWU], [UW, WS]),
    ([ED, NED], [EN, DE]),
    ([ED, SED], [DE, ES]),
    ([WD, NWD], [DW, WN]),
    ([WD, SWD], [WS, DW]),
    ([UN, NEU], [NU, UE]),
    ([US, SEU], [UE, SU]),
    ([UN, NWU], [UW, NU]),
    ([US, SWU], [SU, UW]),
    ([DN, NED], [DE, ND]),
    ([DS, SED], [SD, DE]),
    ([DN, NWD], [ND, DW]),
    ([DS, SWD], [DW, SD]),
];

#[test]
fn test_rhombic_triacontahedron() {
    for &([a, b], [c, d]) in &RHOMBIC_TRIACONTAHEDRON {
        let a: nalgebra::Vector3<f32> = a.into();
        let b: nalgebra::Vector3<f32> = b.into();
        let c: nalgebra::Vector3<f32> = c.into();
        let d: nalgebra::Vector3<f32> = d.into();
        for &[v1, v2] in &[[a, c], [a, d], [b, c], [b, d]] {
            assert!((v1.dot(&v2) - 1.).abs() < 1e-6)
        }

        // 2 / phi^3 = 0.472_135_96
        assert!((a.cross(&b).dot(&c) - 0.472_135_96).abs() < 1e-6);
        assert!((a.cross(&b).dot(&d) + 0.472_135_96).abs() < 1e-6);
    }
}

fn dodecahedron(
    texcoords: [[f32; 2]; 3],
    height: f32,
    size: f32,
) -> impl Iterator<Item = [Vertex; 3]> {
    RHOMBIC_TRIACONTAHEDRON
        .iter()
        .flat_map(move |&([a, b], [c, d])| {
            let a: nalgebra::Vector3<f32> = a.into();
            let b: nalgebra::Vector3<f32> = b.into();
            let mut c: nalgebra::Vector3<f32> = c.into();
            let mut d: nalgebra::Vector3<f32> = d.into();
            c *= 0.723_606_8;
            d *= 0.723_606_8;

            std::iter::once([c, a, b])
                .chain(std::iter::once([d, b, a]))
                .map(move |[v1, v2, v3]| {
                    [
                        Vertex {
                            position: (v1 * size).insert_row(0, height),
                            texture_coordinate: texcoords[0],
                        },
                        Vertex {
                            position: (v2 * size).insert_row(0, height),
                            texture_coordinate: texcoords[1],
                        },
                        Vertex {
                            position: (v3 * size).insert_row(0, height),
                            texture_coordinate: texcoords[2],
                        },
                    ]
                })
        })
}

fn dodecahedron_edge_walls(
    texcoords: [[f32; 2]; 4],
    heights: (f32, f32),
    sizes: (f32, f32),
) -> impl Iterator<Item = [Vertex; 3]> {
    RHOMBIC_TRIACONTAHEDRON
        .iter()
        .flat_map(move |&([a, b], _)| {
            let a: nalgebra::Vector3<f32> = a.into();
            let b: nalgebra::Vector3<f32> = b.into();

            std::iter::once([
                Vertex {
                    position: (a * sizes.0).insert_row(0, heights.0),
                    texture_coordinate: texcoords[0],
                },
                Vertex {
                    position: (b * sizes.0).insert_row(0, heights.0),
                    texture_coordinate: texcoords[1],
                },
                Vertex {
                    position: (b * sizes.1).insert_row(0, heights.1),
                    texture_coordinate: texcoords[3],
                },
            ])
            .chain(std::iter::once([
                Vertex {
                    position: (b * sizes.1).insert_row(0, heights.1),
                    texture_coordinate: texcoords[3],
                },
                Vertex {
                    position: (a * sizes.1).insert_row(0, heights.1),
                    texture_coordinate: texcoords[2],
                },
                Vertex {
                    position: (a * sizes.0).insert_row(0, heights.0),
                    texture_coordinate: texcoords[0],
                },
            ]))
        })
}

fn dodecahedral_prism(
    texcoords_dodecahedron: [[f32; 2]; 3],
    texcoords_vertical: [[f32; 2]; 4],
    heights: (f32, f32),
    sizes: (f32, f32),
) -> Mesh {
    let triangles = Box::new(
        dodecahedron(texcoords_dodecahedron, heights.0, sizes.0)
            .chain(dodecahedron(texcoords_dodecahedron, heights.1, sizes.1))
            .chain(dodecahedron_edge_walls(texcoords_vertical, heights, sizes)),
    );

    let mut facets = Vec::new();
    facets.push(nalgebra::RowVector5::new(-1., 0., 0., 0., heights.0));
    facets.push(nalgebra::RowVector5::new(1., 0., 0., 0., -heights.1));
    let x = (sizes.1 - sizes.0) / (heights.0 - heights.1);
    let c = (heights.1 * sizes.0 - heights.0 * sizes.1) / (heights.0 - heights.1);
    for &f in &DODECAHEDRON_FACES {
        let f: nalgebra::RowVector3<f32> = f.into();
        facets.push(f.insert_column(0, x).insert_column(4, c))
    }
    let regions = Box::new(std::iter::once(Region { facets }));
    Mesh { triangles, regions }
}

fn tree() -> Mesh {
    dodecahedral_prism(
        [[0., 0.], [55. / 64., 0.], [0., 55. / 64.]],
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        (0., 1.),
        (0.5, 0.5),
    ) + dodecahedral_prism(
        [[0., 1.], [0., 1.], [0., 1.]],
        [[0., 1.], [0., 1.], [1., 1.], [1., 1.]],
        (1., 6.),
        (2., 0.),
    )
}

pub fn scene() -> Mesh {
    let t1 =
        render_4d::Mesh::transform(nalgebra::Translation4::new(-1.5, 0., 0., -5.).to_homogeneous())
            .unwrap();
    let t2 = render_4d::Mesh::transform(
        nalgebra::Translation4::new(-1.5, 0., 0., -10.).to_homogeneous(),
    )
    .unwrap();

    t1(tree()) + t2(tree())
}
