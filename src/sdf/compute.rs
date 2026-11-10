use algebrix::{IVec3, Vec3};

use super::field::Edit;

/// One brick's edits to apply to its samples, for an [`SdfCompute`].
#[derive(Debug, Clone, Copy)]
pub struct EditSpan {
    pub edit: Edit,
    /// Lattice samples of the brick it reaches, inclusive, in lattice coordinates.
    pub low: IVec3,
    pub high: IVec3,
}

/// One brick for an [`SdfCompute`]: its samples as they start, the base ground or what an
/// earlier edit left, and the edits to apply in order.
#[derive(Debug)]
pub struct BrickSamples<'a> {
    /// Lattice coordinates of its first sample, x fastest, then y, then z.
    pub low: IVec3,
    /// Samples along each side.
    pub side: usize,
    /// Meters between samples, a sample at lattice `g` sits at `g * voxel` from the volume origin.
    pub voxel: f32,
    pub edits: &'a [EditSpan],
    /// `side` cubed samples, applied in place.
    pub values: &'a mut [f32],
}

/// Applies edits to brick samples somewhere other than the job's own threads, as a gpu. Each
/// sample in an edit's span becomes `edit.op` applied to it and `edit.shape`'s distance at the
/// sample's place, edits in order. Sampling the base ground and meshing stay on the cpu. The
/// solver never needs one, [`CpuEdits`] is the reference an adapter is checked against.
pub trait SdfCompute: Sync {
    /// Applies every brick's edits. False leaves `bricks` to the cpu, as a device that is lost
    /// or busy, and nothing it wrote is used.
    fn apply(&self, bricks: &mut [BrickSamples<'_>]) -> bool;
}

/// The cpu reference of [`SdfCompute`], the same arithmetic the bricks use on their own.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuEdits;

impl SdfCompute for CpuEdits {
    fn apply(&self, bricks: &mut [BrickSamples<'_>]) -> bool {
        for brick in bricks.iter_mut() {
            let side = brick.side;
            for span in brick.edits {
                let (from, to) = (span.low, span.high);
                for gz in from.z..=to.z {
                    for gy in from.y..=to.y {
                        for gx in from.x..=to.x {
                            let p = Vec3::new(gx as f32, gy as f32, gz as f32) * brick.voxel;
                            let slot = (gx - brick.low.x) as usize
                                + side
                                    * ((gy - brick.low.y) as usize
                                        + side * (gz - brick.low.z) as usize);
                            brick.values[slot] = span
                                .edit
                                .op
                                .apply(brick.values[slot], span.edit.shape.distance(p));
                        }
                    }
                }
            }
        }
        true
    }
}
