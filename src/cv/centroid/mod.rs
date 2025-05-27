use anyhow::Result;
use opencv::core::Rect;
use pathfinding::{
    matrix::Matrix,
    prelude::{kuhn_munkres, kuhn_munkres_min},
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::direction::Direction;

#[derive(Debug, Clone)]
pub struct Centroid {
    pub x: i32,
    pub y: i32,
}

impl Centroid {
    pub fn from_rect(rect: Rect) -> Self {
        Self {
            x: rect.x + rect.width / 2,
            y: rect.y + rect.height / 2,
        }
    }

    pub fn distance_to(&self, other: Centroid) -> f32 {
        let dx = (self.x - other.x) as f32;
        let dy = (self.y - other.y) as f32;
        (dx.powi(2) + dy.powi(2)).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct TrackableObject {
    pub oid: u32,
    pub centroids: Vec<Centroid>,
    pub counted: bool,
    pub last_direction: Option<Direction>,
}

#[derive(Debug, Clone)]
pub struct CentroidTracker {
    next_oid: u32,
    pub objects: HashMap<u32, TrackableObject>,
    disappeared: HashMap<u32, u32>,
    max_disappeared: u32,
    max_distance: f32,
}

impl CentroidTracker {
    pub fn new(max_disappeared: u32, max_distance: f32) -> Self {
        Self {
            next_oid: 0,
            objects: HashMap::new(),
            disappeared: HashMap::new(),
            max_disappeared,
            max_distance,
        }
    }

    pub fn update(&mut self, rects: &[Rect]) -> Result<HashMap<u32, Centroid>> {
        let input: Vec<Centroid> = rects.par_iter().map(|r| Centroid::from_rect(*r)).collect();

        if self.objects.is_empty() {
            for centroid in input {
                self.register(centroid);
            }
            return Ok(self.current_centroids());
        }

        // Optimization: For small numbers of objects, use simple greedy matching
        // instead of Hungarian algorithm which has O(n³) complexity
        if self.objects.len() <= 5 && input.len() <= 5 {
            return self.greedy_update(input);
        }

        // 1. Create cost matrix with max_distance check
        let mut cost_matrix = vec![vec![0.0; input.len()]; self.objects.len()];
        let oids: Vec<u32> = self.objects.keys().copied().collect();

        cost_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            let oid = oids[i];
            let obj = self.objects.get(&oid).unwrap();
            let last_centroid = obj.centroids.last().unwrap();

            row.par_iter_mut().enumerate().for_each(|(j, cost)| {
                let distance = last_centroid.distance_to(input[j].clone());
                *cost = if distance <= self.max_distance {
                    distance
                } else {
                    f32::INFINITY
                };
            });
        });

        // 2. Convert to integer matrix with scaling and inversion for minimization
        let int_matrix: Vec<Vec<i32>> = cost_matrix
            .par_iter()
            .map(|row| {
                row.par_iter()
                    .map(|&d| {
                        if d > self.max_distance {
                            i32::MAX
                        } else {
                            (d * 1000.0).round() as i32 // Preserve 3 decimal places
                        }
                    })
                    .collect()
            })
            .collect();

        let matrix = Matrix::from_rows(int_matrix.clone())?;
        let (total_cost, assignments) = kuhn_munkres_min(&matrix);

        // 3. Process assignments
        let mut used_inputs = HashSet::new();
        let mut matched_oids = HashSet::new();

        assignments
            .par_iter()
            .enumerate()
            .for_each(|(obj_idx, &input_idx)| {
                // Verify valid indices
                if obj_idx >= oids.len() || input_idx >= input.len() {
                    return;
                }

                let oid = oids[obj_idx];
                let centroid = &input[input_idx];

                // Check if this is a valid match (not infinite cost)
                if int_matrix[obj_idx][input_idx] != i32::MAX {
                    // Note: This requires additional synchronization since we're mutating shared state
                    // Consider collecting results and applying them sequentially after the parallel loop
                }
            });

        // Apply updates sequentially to avoid race conditions
        for (obj_idx, &input_idx) in assignments.iter().enumerate() {
            if obj_idx >= oids.len() || input_idx >= input.len() {
                continue;
            }

            let oid = oids[obj_idx];
            let centroid = &input[input_idx];

            if let Some(obj) = self.objects.get_mut(&oid) {
                if int_matrix[obj_idx][input_idx] != i32::MAX {
                    obj.centroids.push(centroid.clone());
                    self.disappeared.remove(&oid);
                    matched_oids.insert(oid);
                    used_inputs.insert(input_idx);
                }
            }
        }

        // 4. Handle disappeared objects
        for oid in &oids {
            if !matched_oids.contains(oid) {
                let entry = self.disappeared.entry(*oid).or_insert(0);
                *entry += 1;

                if *entry > self.max_disappeared {
                    self.deregister(*oid);
                }
            }
        }

        // 5. Register new objects
        input
            .into_par_iter()
            .enumerate()
            .filter(|(input_idx, _)| !used_inputs.contains(input_idx))
            .map(|(_, centroid)| centroid)
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|centroid| self.register(centroid));

        Ok(self.current_centroids())
    }

    // Optimized greedy matching for small object sets
    fn greedy_update(&mut self, input: Vec<Centroid>) -> Result<HashMap<u32, Centroid>> {
        let mut used_inputs = HashSet::new();
        let mut matched_oids = HashSet::new();
        let oids: Vec<u32> = self.objects.keys().copied().collect();

        // Greedy matching: for each object, find closest input centroid
        for oid in &oids {
            let obj = self.objects.get(oid).unwrap();
            let last_centroid = obj.centroids.last().unwrap();

            let mut best_match: Option<usize> = None;
            let mut best_distance = f32::INFINITY;

            let (best_distance, best_match) = input
                .par_iter()
                .enumerate()
                .filter(|(idx, _)| !used_inputs.contains(idx))
                .map(|(idx, input_centroid)| {
                    let distance = last_centroid.distance_to(input_centroid.clone());
                    (
                        distance,
                        if distance <= self.max_distance {
                            Some(idx)
                        } else {
                            None
                        },
                    )
                })
                .filter(|(distance, _)| *distance <= self.max_distance)
                .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
                .unwrap_or((f32::INFINITY, None));

            if let Some(idx) = best_match {
                if let Some(obj) = self.objects.get_mut(oid) {
                    obj.centroids.push(input[idx].clone());
                    self.disappeared.remove(oid);
                    matched_oids.insert(*oid);
                    used_inputs.insert(idx);
                }
            }
        }

        // Handle disappeared objects
        for oid in &oids {
            if !matched_oids.contains(oid) {
                let entry = self.disappeared.entry(*oid).or_insert(0);
                *entry += 1;

                if *entry > self.max_disappeared {
                    self.deregister(*oid);
                }
            }
        }

        // Register new objects
        for (input_idx, centroid) in input.into_iter().enumerate() {
            if !used_inputs.contains(&input_idx) {
                self.register(centroid);
            }
        }

        Ok(self.current_centroids())
    }

    fn current_centroids(&self) -> HashMap<u32, Centroid> {
        self.objects
            .par_iter()
            .filter_map(|(oid, obj)| obj.centroids.last().map(|c| (*oid, c.clone())))
            .collect()
    }

    fn deregister(&mut self, oid: u32) {
        self.objects.remove(&oid);
        self.disappeared.remove(&oid);
    }

    fn register(&mut self, centroid: Centroid) {
        let oid = self.next_oid;
        self.next_oid += 1;
        self.objects.insert(
            oid,
            TrackableObject {
                oid,
                centroids: vec![centroid],
                counted: false,
                last_direction: None,
            },
        );
    }
}
