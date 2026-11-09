use algebrix::{DVec3, IVec3, Quat, Vec3};

use super::SnapshotError;
use crate::{BodyHandle, GridPosition};

pub(crate) fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub(crate) struct Writer {
    pub(crate) data: Vec<u8>,
}

impl Writer {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn bytes(&mut self, value: &[u8]) {
        self.data.extend_from_slice(value);
    }

    pub(crate) fn u8(&mut self, value: u8) {
        self.data.push(value);
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    pub(super) fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    pub(crate) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    pub(crate) fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    // storage indexes are u32, so every length fits
    pub(crate) fn len(&mut self, value: usize) {
        self.u32(value as u32);
    }

    pub(crate) fn vec3(&mut self, value: Vec3) {
        self.f32(value.x);
        self.f32(value.y);
        self.f32(value.z);
    }

    pub(crate) fn dvec3(&mut self, value: DVec3) {
        self.f64(value.x);
        self.f64(value.y);
        self.f64(value.z);
    }

    pub(crate) fn position(&mut self, value: GridPosition) {
        self.i32(value.cell.x);
        self.i32(value.cell.y);
        self.i32(value.cell.z);
        self.vec3(value.local);
    }

    pub(crate) fn body(&mut self, handle: BodyHandle) {
        self.u32(handle.index);
        self.u32(handle.generation);
    }

    pub(crate) fn quat(&mut self, value: Quat) {
        for part in value.to_array() {
            self.f32(part);
        }
    }
}

pub(crate) struct Reader<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) at: usize,
    /// The snapshot's format version, what it holds depends on it.
    pub(crate) version: u32,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], SnapshotError> {
        let end = self.at.checked_add(N).ok_or(SnapshotError::Truncated)?;
        let bytes = self
            .data
            .get(self.at..end)
            .ok_or(SnapshotError::Truncated)?;
        self.at = end;
        bytes.try_into().map_err(|_| SnapshotError::Truncated)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, SnapshotError> {
        Ok(self.take::<1>()?[0])
    }

    pub(crate) fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    pub(super) fn i32(&mut self) -> Result<i32, SnapshotError> {
        Ok(i32::from_le_bytes(self.take()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, SnapshotError> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, SnapshotError> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, SnapshotError> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// Reads a count, checks it against `max` and the bytes left before anything allocates.
    pub(crate) fn count(
        &mut self,
        record_bytes: usize,
        max: usize,
    ) -> Result<usize, SnapshotError> {
        let count = self.u32()? as usize;
        if count > max {
            return Err(SnapshotError::Invalid("count over limit"));
        }
        let left = self.data.len() - self.at;
        if count
            .checked_mul(record_bytes)
            .is_none_or(|bytes| bytes > left)
        {
            return Err(SnapshotError::Truncated);
        }
        Ok(count)
    }

    pub(crate) fn vec3(&mut self) -> Result<Vec3, SnapshotError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }

    pub(crate) fn dvec3(&mut self) -> Result<DVec3, SnapshotError> {
        Ok(DVec3::new(self.f64()?, self.f64()?, self.f64()?))
    }

    pub(crate) fn position(&mut self) -> Result<GridPosition, SnapshotError> {
        let cell = IVec3::new(self.i32()?, self.i32()?, self.i32()?);
        let position = GridPosition {
            cell,
            local: self.vec3()?,
        };
        if position.rebased() != Some(position) {
            return Err(SnapshotError::Invalid("position"));
        }
        Ok(position)
    }

    pub(crate) fn handle(&mut self) -> Result<BodyHandle, SnapshotError> {
        Ok(BodyHandle {
            index: self.u32()?,
            generation: self.u32()?,
        })
    }

    pub(crate) fn quat(&mut self) -> Result<Quat, SnapshotError> {
        Ok(Quat::new(
            self.f32()?,
            self.f32()?,
            self.f32()?,
            self.f32()?,
        ))
    }

    pub(crate) fn unit_quat(&mut self) -> Result<Quat, SnapshotError> {
        let value = self.quat()?;
        let error = (value.length_squared() - 1.0).abs();
        if error.is_nan() || error >= 1e-3 {
            return Err(SnapshotError::Invalid("joint frame"));
        }
        Ok(value)
    }
}
