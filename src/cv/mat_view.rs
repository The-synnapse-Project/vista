use anyhow::{Context, Result, anyhow};
use log::{error, info};
use opencv::core::{
    CV_32FC2, DataType, Mat, MatTrait, MatTraitConst, MatTraitConstManual, MatTraitManual, Scalar_,
};
use rayon::prelude::*;

pub struct MatViewND<'a, T> {
    dims: Vec<i32>,
    strides: Vec<usize>,
    data: &'a mut [T],
}

impl<'a, T: DataType> MatViewND<'a, T> {
    pub fn new(mat: &'a mut Mat) -> opencv::Result<Self> {
        let dims = mat.mat_size();

        // Calculate strides based on matrix dimensions
        let mut strides: Vec<usize> = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1] as usize;
        }

        info!("Strides calculated: {:?}", strides);
        Ok(Self {
            dims: dims.to_vec(),
            strides,
            data: mat.data_typed_mut::<T>()?,
        })
    }

    /// Safe access with bounds checking (optimized)
    pub fn get(&self, indices: &[i32]) -> Result<&T> {
        // Fast path for common case of 4D tensor access
        if indices.len() == 4 && self.dims.len() == 4 {
            let idx0 = indices[0] as usize;
            let idx1 = indices[1] as usize;
            let idx2 = indices[2] as usize;
            let idx3 = indices[3] as usize;

            // Bounds check
            if idx0 < self.dims[0] as usize
                && idx1 < self.dims[1] as usize
                && idx2 < self.dims[2] as usize
                && idx3 < self.dims[3] as usize
            {
                let offset = idx0 * self.strides[0]
                    + idx1 * self.strides[1]
                    + idx2 * self.strides[2]
                    + idx3 * self.strides[3];
                return self.data.get(offset).context("Index out of bounds");
            }
        }

        // Fallback to general case
        self.validate_indices(indices)?;
        let offset = self.calculate_offset(indices);
        self.data.get(offset).context("Index out of bounds")
    }

    /// Mutable access with bounds checking
    pub fn get_mut(&mut self, indices: &[i32]) -> Result<&mut T> {
        self.validate_indices(indices)?;
        let offset = self.calculate_offset(indices);
        self.data.get_mut(offset).context("Index out of bounds")
    }

    fn validate_indices(&self, indices: &[i32]) -> Result<()> {
        if indices.len() != self.dims.len() {
            return Err(anyhow!(
                "Invalid index dimensions: expected {}, got {}",
                self.dims.len(),
                indices.len()
            ));
        }

        indices.iter().enumerate().try_for_each(|(i, &idx)| {
            if idx < 0 || idx >= self.dims[i] {
                Err(anyhow!(
                    "Index {} out of bounds for dimension {} (0..{})",
                    idx,
                    i,
                    self.dims[i] - 1
                ))
            } else {
                Ok(())
            }
        })?;

        Ok(())
    }

    fn calculate_offset(&self, indices: &[i32]) -> usize {
        indices
            .par_iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx as usize * stride)
            .sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + use<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }
}
