use super::*;
use petgraph::prelude::*;

impl Mesh {
    pub fn hypercube(position: [nalgebra::Vector4<f32>; 2], tex_coords: [[f32; 3]; 2]) -> Self {
        // Maps texture onto [-1..1, -1..1, -1..1]
        let texture_to_unit_cube = {
            let tex_corner_1 = nalgebra::Vector3::from(tex_coords[0]).insert_row(3, 1.0);
            let tex_corner_2 = nalgebra::Vector3::from(tex_coords[1]).insert_row(3, 1.0);

            let mut out = nalgebra::Matrix4::from_diagonal(&(tex_corner_2 - tex_corner_1));
            out.column_mut(3).copy_from(&(tex_corner_2 + tex_corner_1));
            out.try_inverse_mut();
            out
        };

        // Maps [-1..1, -1..1, -1..1, -1..1] onto hypercube
        let unit_hypercube_to_result = {
            let corner_1 = position[0].insert_row(4, 1.0);
            let corner_2 = position[1].insert_row(4, 1.0);

            let mut out = nalgebra::Matrix5::from_diagonal(&(corner_2 - corner_1));
            out.column_mut(4).copy_from(&(corner_2 + corner_1));
            out
        };

        #[rustfmt::skip]
        let unit_cube_to_unit_hypercube: [nalgebra::Matrix5x4<f32>; 8] = [
            nalgebra::Matrix5x4::new( 0.,  0.,  0.,  1.,    0.,  0., -1.,  0.,    0.,  1.,  0.,  0.,    1.,  0.,  0.,  0.,    0.,  0.,  0., 1.),
            nalgebra::Matrix5x4::new( 0.,  0.,  1.,  0.,    0.,  0.,  0.,  1.,   -1.,  0.,  0.,  0.,    0.,  1.,  0.,  0.,    0.,  0.,  0., 1.),
            nalgebra::Matrix5x4::new( 0., -1.,  0.,  0.,    1.,  0.,  0.,  0.,    0.,  0.,  0.,  1.,    0.,  0.,  1.,  0.,    0.,  0.,  0., 1.),
            nalgebra::Matrix5x4::new(-1.,  0.,  0.,  0.,    0., -1.,  0.,  0.,    0.,  0., -1.,  0.,    0.,  0.,  0.,  1.,    0.,  0.,  0., 1.),
            nalgebra::Matrix5x4::new( 0.,  0.,  0., -1.,    0.,  0.,  1.,  0.,    0., -1.,  0.,  0.,   -1.,  0.,  0.,  0.,    0.,  0.,  0., 1.),
            nalgebra::Matrix5x4::new( 0.,  0., -1.,  0.,    0.,  0.,  0., -1.,    1.,  0.,  0.,  0.,    0., -1.,  0.,  0.,    0.,  0.,  0., 1.),
            nalgebra::Matrix5x4::new( 0.,  1.,  0.,  0.,   -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    0.,  0., -1.,  0.,    0.,  0.,  0., 1.),
            nalgebra::Matrix5x4::new( 1.,  0.,  0.,  0.,    0.,  1.,  0.,  0.,    0.,  0.,  1.,  0.,    0.,  0.,  0., -1.,    0.,  0.,  0., 1.),
        ];

        let mut graph = Graph::new_undirected();

        let nodes: Vec<_> = unit_cube_to_unit_hypercube
            .iter()
            .map(|m| graph.add_node(unit_hypercube_to_result * m * texture_to_unit_cube))
            .collect();

        for &[i, j, k, l] in &[
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [0, 3, 1, 2],
            [1, 2, 0, 3],
            [1, 3, 0, 2],
            [2, 3, 0, 1],
        ] {
            for &[flip_1, flip_2] in &[[false, false], [false, true], [true, true], [true, false]] {
                let n1 = nodes[i + 4 * flip_1 as usize];
                let n2 = nodes[j + 4 * flip_2 as usize];

                let mut center_face = nalgebra::Vector5::zeros();
                center_face[i] = if flip_1 { -1. } else { 1. };
                center_face[j] = if flip_2 { -1. } else { 1. };
                center_face[4] = 1.;

                let mut v1 = nalgebra::Vector5::zeros();
                v1[k] = 1.;
                let mut v2 = nalgebra::Vector5::zeros();
                v2[l] = 1.;

                let edge = vec![
                    {
                        let e = unit_hypercube_to_result * (center_face + v1 + v2);
                        e.remove_row(4) / e[4]
                    },
                    {
                        let e = unit_hypercube_to_result * (center_face + v1 - v2);
                        e.remove_row(4) / e[4]
                    },
                    {
                        let e = unit_hypercube_to_result * (center_face - v1 - v2);
                        e.remove_row(4) / e[4]
                    },
                    {
                        let e = unit_hypercube_to_result * (center_face - v1 + v2);
                        e.remove_row(4) / e[4]
                    },
                ];

                graph.add_edge(n1, n2, edge);
            }
        }

        Self(graph)
    }
}
