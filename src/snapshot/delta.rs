use super::{SnapshotError, fnv1a};
use crate::world::PhysicsWorld;

const MAGIC: [u8; 4] = *b"SLDT";
pub const DELTA_VERSION: u32 = 1;
/// Magic, version, the base's length and sum, the new length, then the runs, then the sum.
const HEAD: usize = 4 + 4 + 4 + 8 + 4;
/// Most bytes a delta may make, as a full snapshot.
const MOST_BYTES: usize = 1 << 30;

fn varint(out: &mut Vec<u8>, mut value: usize) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint(data: &[u8], at: &mut usize) -> Result<usize, SnapshotError> {
    let mut value = 0_usize;
    for shift in (0..64).step_by(7) {
        let byte = *data.get(*at).ok_or(SnapshotError::Truncated)?;
        *at += 1;
        value |= usize::from(byte & 0x7f)
            .checked_shl(shift)
            .ok_or(SnapshotError::Invalid("delta run"))?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(SnapshotError::Invalid("delta run"))
}

/// `new` as what differs from `base`: runs of bytes that are the same, each followed by the
/// bytes that are not, xored with the base's.
pub fn snapshot_delta(base: &[u8], new: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD + new.len() / 8 + 16);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&DELTA_VERSION.to_le_bytes());
    out.extend_from_slice(&(base.len() as u32).to_le_bytes());
    out.extend_from_slice(&fnv1a(base).to_le_bytes());
    out.extend_from_slice(&(new.len() as u32).to_le_bytes());
    let changed = |at: usize| base.get(at).copied().unwrap_or(0) ^ new[at];
    let mut at = 0;
    while at < new.len() {
        let start = at;
        while at < new.len() && changed(at) == 0 {
            at += 1;
        }
        let same = at - start;
        let from = at;
        // a few same bytes inside a changed stretch cost more as a run than as literals
        while at < new.len() {
            if changed(at) != 0 {
                at += 1;
                continue;
            }
            let quiet = (at..new.len().min(at + 4)).all(|i| changed(i) == 0);
            if quiet {
                break;
            }
            at += 1;
        }
        varint(&mut out, same);
        varint(&mut out, at - from);
        out.extend((from..at).map(changed));
    }
    let sum = fnv1a(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

/// The snapshot `delta` makes from `base`.
///
/// # Errors
/// Fails on a delta that is not one of this version, cut short or changed, or made from
/// another base.
pub fn apply_snapshot_delta(base: &[u8], delta: &[u8]) -> Result<Vec<u8>, SnapshotError> {
    if delta.len() < HEAD + 8 || delta[..4] != MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let (body, sum) = delta.split_at(delta.len() - 8);
    let sum = u64::from_le_bytes(sum.try_into().map_err(|_| SnapshotError::Truncated)?);
    if fnv1a(body) != sum {
        return Err(SnapshotError::BadChecksum);
    }
    let word = |at: usize| u32::from_le_bytes([body[at], body[at + 1], body[at + 2], body[at + 3]]);
    let version = word(4);
    if version != DELTA_VERSION {
        return Err(SnapshotError::UnsupportedVersion(version));
    }
    let base_sum = u64::from_le_bytes(
        body[12..20]
            .try_into()
            .map_err(|_| SnapshotError::Truncated)?,
    );
    if word(8) as usize != base.len() || fnv1a(base) != base_sum {
        return Err(SnapshotError::Invalid("delta of another base"));
    }
    let length = word(20) as usize;
    if length > MOST_BYTES {
        return Err(SnapshotError::TooLarge);
    }
    let mut new = Vec::with_capacity(length.min(base.len() + body.len() * 2 + 64));
    let mut at = HEAD;
    while at < body.len() {
        let same = read_varint(body, &mut at)?;
        let differs = read_varint(body, &mut at)?;
        let end = new
            .len()
            .checked_add(same)
            .and_then(|end| end.checked_add(differs))
            .ok_or(SnapshotError::Invalid("delta run"))?;
        if end > length || differs > body.len() - at {
            return Err(SnapshotError::Invalid("delta run"));
        }
        for _ in 0..same {
            new.push(base.get(new.len()).copied().unwrap_or(0));
        }
        for byte in &body[at..at + differs] {
            new.push(base.get(new.len()).copied().unwrap_or(0) ^ byte);
        }
        at += differs;
    }
    if new.len() != length {
        return Err(SnapshotError::Invalid("delta length"));
    }
    Ok(new)
}

impl PhysicsWorld {
    /// This world as what changed since `base`, an earlier [`Self::save_snapshot`]. Bodies
    /// lying still cost next to nothing, so it suits sending a world often.
    pub fn save_delta(&self, base: &[u8]) -> Vec<u8> {
        snapshot_delta(base, &self.save_snapshot())
    }

    /// The world `delta` makes from `base`, checked as [`Self::load_snapshot`] checks.
    ///
    /// # Errors
    /// As [`apply_snapshot_delta`] and [`Self::load_snapshot`].
    pub fn load_delta(base: &[u8], delta: &[u8]) -> Result<Self, SnapshotError> {
        Self::load_snapshot(&apply_snapshot_delta(base, delta)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::testing::world;
    use crate::{RigidBodyDesc, Shape};
    use algebrix::Vec3;

    fn pile() -> PhysicsWorld {
        let mut world = world();
        world
            .spawn(RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(
                20.0, 0.5, 20.0,
            ))))
            .unwrap();
        for i in 0..200 {
            let at = Vec3::new((i % 10) as f32 * 1.5, 1.0 + (i / 10) as f32 * 1.2, 0.0);
            world
                .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.4)).local_position(at))
                .unwrap();
        }
        world
    }

    #[test]
    fn a_delta_brings_back_the_same_world_byte_for_byte() {
        let mut world = pile();
        let base = world.save_snapshot();
        for _ in 0..3 {
            world.step_fixed().unwrap();
        }
        let delta = world.save_delta(&base);
        let loaded = PhysicsWorld::load_delta(&base, &delta).unwrap();
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
        // nothing moved, next to nothing sent
        let still = snapshot_delta(&base, &base);
        assert!(still.len() < 48, "{}", still.len());
    }

    #[test]
    fn a_world_lying_still_costs_little() {
        let mut world = pile();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        let base = world.save_snapshot();
        world.step_fixed().unwrap();
        let delta = world.save_delta(&base);
        assert!(
            delta.len() * 20 < base.len(),
            "{} of {}",
            delta.len(),
            base.len()
        );
    }

    #[test]
    fn bodies_coming_and_going_still_round_trip() {
        let mut world = pile();
        let base = world.save_snapshot();
        let gone: Vec<_> = world.body_handles().take(30).collect();
        for body in gone.into_iter().skip(1) {
            world.despawn(body).unwrap();
        }
        for i in 0..5 {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.3)))
                        .local_position(Vec3::new(i as f32, 9.0, 3.0)),
                )
                .unwrap();
        }
        world.step_fixed().unwrap();
        let loaded = PhysicsWorld::load_delta(&base, &world.save_delta(&base)).unwrap();
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
    }

    #[test]
    fn bad_deltas_are_refused_and_never_panic() {
        let mut world = pile();
        let base = world.save_snapshot();
        world.step_fixed().unwrap();
        let delta = world.save_delta(&base);
        let mut other = base.clone();
        other[40] ^= 1;
        assert!(apply_snapshot_delta(&other, &delta).is_err());
        assert!(apply_snapshot_delta(&base, &delta[..delta.len() - 1]).is_err());
        assert!(apply_snapshot_delta(&base, b"SLDT").is_err());
        let mut seed = 5_u64;
        for _ in 0..2000 {
            let mut broken = delta.clone();
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let at = (seed >> 33) as usize % broken.len();
            broken[at] ^= 1 << ((seed >> 20) % 8);
            // resealed so the runs themselves are tried, not only the sum
            let end = broken.len() - 8;
            let sum = fnv1a(&broken[..end]);
            broken[end..].copy_from_slice(&sum.to_le_bytes());
            let _ = PhysicsWorld::load_delta(&base, &broken);
        }
    }
}
