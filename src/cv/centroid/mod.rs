use anyhow::Result;
use std::collections::{HashMap, HashSet};

use opencv::core::Rect;
use pathfinding::{matrix::Matrix, prelude::{kuhn_munkres, kuhn_munkres_min}};

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
    disapeared: HashMap<u32, u32>,
    max_disapeared: u32,
    max_distance: f32,
}

impl CentroidTracker {
    pub fn new(max_disapeared: u32, max_distance: f32) -> Self {
        Self {
            next_oid: 0,
            objects: HashMap::new(),
            disapeared: HashMap::new(),
            max_disapeared,
            max_distance,
        }
    }

    pub fn update(&mut self, rects: &[Rect]) -> Result<HashMap<u32, Centroid>> {
        let input: Vec<Centroid> = rects.iter().map(|r| Centroid::from_rect(*r)).collect();

        if self.objects.is_empty() {
            for centroid in input {
                self.register(centroid);
            }
            return Ok(self.current_centroids());
        }

        // 1. Create cost matrix with max_distance check
        let mut cost_matrix = vec![vec![0.0; input.len()]; self.objects.len()];
        let oids: Vec<u32> = self.objects.keys().copied().collect();
        
        for (i, oid) in oids.iter().enumerate() {
            let obj = self.objects.get(oid).unwrap();
            let last_centroid = obj.centroids.last().unwrap();
            
            for (j, input_centroid) in input.iter().enumerate() {
                let distance = last_centroid.distance_to(input_centroid.clone());
                cost_matrix[i][j] = if distance <= self.max_distance {
                    distance
                } else {
                    f32::INFINITY
                };
            }
        }

        // 2. Convert to integer matrix with scaling and inversion for minimization
        let int_matrix: Vec<Vec<i32>> = cost_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&d| {
                        if d > self.max_distance {
                            i32::MAX
                        } else {
                            (d * 1000.0).round() as i32  // Preserve 3 decimal places
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

        for (obj_idx, &input_idx) in assignments.iter().enumerate() {
            // Verify valid indices
            if obj_idx >= oids.len() || input_idx >= input.len() {
                continue;
            }

            let oid = oids[obj_idx];
            let centroid = &input[input_idx];

            if let Some(obj) = self.objects.get_mut(&oid) {
                // Check if this is a valid match (not infinite cost)
                if int_matrix[obj_idx][input_idx] != i32::MAX {
                    obj.centroids.push(centroid.clone());
                    self.disapeared.remove(&oid);
                    matched_oids.insert(oid);
                    used_inputs.insert(input_idx);
                }
            }
        }

        // 4. Handle disappeared objects
        for oid in &oids {
            if !matched_oids.contains(oid) {
                let entry = self.disapeared.entry(*oid).or_insert(0);
                *entry += 1;
                
                if *entry > self.max_disapeared {
                    self.deregister(*oid);
                }
            }
        }

        // 5. Register new objects
        for (input_idx, centroid) in input.into_iter().enumerate() {
            if !used_inputs.contains(&input_idx) {
                self.register(centroid);
            }
        }

        Ok(self.current_centroids())
    }

    fn current_centroids(&self) -> HashMap<u32, Centroid> {
        self.objects
            .iter()
            .filter_map(|(oid, obj)| obj.centroids.last().map(|c| (*oid, c.clone())))
            .collect()
    }

    fn deregister(&mut self, oid: u32) {
        self.objects.remove(&oid);
        self.disapeared.remove(&oid);
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
