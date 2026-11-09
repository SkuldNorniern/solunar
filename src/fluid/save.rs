use super::boundary::Walls;
use super::couple::Coupled;
use super::flow::Flows;
use super::terrain::TerrainField;
use super::{FluidSettings, FluidSolver, FluidWorld};
use crate::snapshot::{Reader, SnapshotError, Writer, fnv1a};

const MAGIC: [u8; 4] = *b"SLFL";
pub const FLUID_SNAPSHOT_VERSION: u32 = 3;
const CHECKSUM_BYTES: usize = 8;
/// Largest fluid snapshot read.
pub const MAX_FLUID_SNAPSHOT_BYTES: usize = 1 << 30;
/// Most coupled bodies or terrain regions one snapshot may hold.
const MAX_PARTS: usize = 1 << 16;

impl FluidWorld {
    /// Everything the next step needs, so a loaded copy steps on exactly as this one would.
    /// Coupled bodies are named by handle, they belong to the physics world saved beside it.
    pub fn save_snapshot(&self) -> Vec<u8> {
        let mut out = Writer::with_capacity(64 + self.len() * 28 + self.walls.positions.len() * 16);
        out.bytes(&MAGIC);
        out.u32(FLUID_SNAPSHOT_VERSION);
        let s = &self.settings;
        out.u8(match s.solver {
            FluidSolver::Wcsph => 0,
            FluidSolver::Dfsph => 1,
        });
        for value in [
            s.spacing,
            s.rest_density,
            s.support,
            s.sound_speed,
            s.viscosity,
            s.courant,
        ] {
            out.f32(value);
        }
        out.u64(s.max_particles as u64);
        out.vec3(s.gravity);
        out.u32(s.max_substeps);
        out.len(self.len());
        for index in 0..self.len() {
            out.u32(self.ids[index]);
            out.vec3(self.positions[index]);
            out.vec3(self.velocities[index]);
        }
        out.u32(self.next_id);
        self.walls.save(&mut out);
        self.flows.save(&mut out);
        out.len(self.coupled.len());
        for coupled in &self.coupled {
            coupled.save(&mut out);
        }
        out.len(self.terrains.len());
        for terrain in &self.terrains {
            terrain.save(&mut out);
        }
        match self.domain {
            None => out.u8(0),
            Some((low, high)) => {
                out.u8(1);
                out.vec3(low);
                out.vec3(high);
            }
        }
        let checksum = fnv1a(&out.data);
        out.u64(checksum);
        out.data
    }

    /// Input is untrusted. Bad data returns an error, never panics.
    ///
    /// # Errors
    /// Fails on oversized, truncated, corrupt or inconsistent data, or an unknown version.
    pub fn load_snapshot(data: &[u8]) -> Result<Self, SnapshotError> {
        if data.len() > MAX_FLUID_SNAPSHOT_BYTES {
            return Err(SnapshotError::TooLarge);
        }
        if data.len() < MAGIC.len() + 4 + CHECKSUM_BYTES {
            return Err(SnapshotError::Truncated);
        }
        if data[..4] != MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let (body, stored) = data.split_at(data.len() - CHECKSUM_BYTES);
        let mut checksum = Reader {
            data: stored,
            at: 0,
            version: 0,
        };
        if checksum.u64()? != fnv1a(body) {
            return Err(SnapshotError::BadChecksum);
        }
        let mut reader = Reader {
            data: body,
            at: MAGIC.len(),
            version: 0,
        };
        let version = reader.u32()?;
        if !(1..=FLUID_SNAPSHOT_VERSION).contains(&version) {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        reader.version = version;
        let solver = match reader.u8()? {
            0 => FluidSolver::Wcsph,
            1 => FluidSolver::Dfsph,
            _ => return Err(SnapshotError::Invalid("fluid solver")),
        };
        let mut floats = [0.0_f32; 6];
        for value in &mut floats {
            *value = reader.f32()?;
        }
        let [
            spacing,
            rest_density,
            support,
            sound_speed,
            viscosity,
            courant,
        ] = floats;
        let max_particles = usize::try_from(reader.u64()?)
            .map_err(|_| SnapshotError::Invalid("fluid particle cap"))?;
        let settings = FluidSettings {
            solver,
            spacing,
            rest_density,
            support,
            max_particles,
            sound_speed,
            viscosity,
            gravity: reader.vec3()?,
            courant,
            max_substeps: reader.u32()?,
        };
        let mut fluid =
            Self::new(settings).map_err(|_| SnapshotError::Invalid("fluid settings"))?;
        let count = reader.count(28, max_particles)?;
        for _ in 0..count {
            let (id, at, velocity) = (reader.u32()?, reader.vec3()?, reader.vec3()?);
            if !at.is_finite() || !velocity.is_finite() {
                return Err(SnapshotError::Invalid("fluid particle"));
            }
            fluid.ids.push(id);
            fluid.positions.push(at);
            fluid.velocities.push(velocity);
        }
        fluid.densities = vec![settings.rest_density; count];
        fluid.next_id = reader.u32()?;
        let mut seen = fluid.ids.clone();
        seen.sort_unstable();
        if seen.windows(2).any(|pair| pair[0] == pair[1])
            || seen.last().is_some_and(|id| *id >= fluid.next_id)
        {
            return Err(SnapshotError::Invalid("fluid particle id"));
        }
        fluid.walls = Walls::load(&mut reader, max_particles)?;
        fluid.flows = Flows::load(&mut reader, MAX_PARTS)?;
        let coupled = reader.count(12, MAX_PARTS)?;
        let mut room = max_particles;
        for _ in 0..coupled {
            let part = Coupled::load(&mut reader, room, fluid.settings.spacing)?;
            room -= part.point_count();
            fluid.coupled.push(part);
        }
        if fluid
            .coupled
            .windows(2)
            .any(|pair| pair[0].body() >= pair[1].body())
        {
            return Err(SnapshotError::Invalid("fluid body order"));
        }
        let terrains = reader.count(32, MAX_PARTS)?;
        for _ in 0..terrains {
            let terrain = TerrainField::load(&mut reader)?;
            if terrain.owner >= fluid.walls.next() {
                return Err(SnapshotError::Invalid("fluid terrain owner"));
            }
            fluid.terrains.push(terrain);
        }
        if version >= 2 {
            let domain = match reader.u8()? {
                0 => None,
                1 => Some((reader.vec3()?, reader.vec3()?)),
                _ => return Err(SnapshotError::Invalid("fluid domain flag")),
            };
            fluid
                .set_domain(domain)
                .map_err(|_| SnapshotError::Invalid("fluid domain"))?;
        }
        if reader.at != body.len() {
            return Err(SnapshotError::Invalid("trailing bytes"));
        }
        fluid.settle_walls();
        Ok(fluid)
    }
}

#[cfg(test)]
mod tests {
    use algebrix::Vec3;

    use super::*;
    use crate::Sequential;

    const TICK: f32 = 1.0 / 60.0;

    fn pool() -> FluidWorld {
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        fluid
            .add_box_container(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.6))
            .unwrap();
        fluid
            .add_wall(Vec3::new(0.45, 0.0, 0.0), Vec3::new(0.55, 0.3, 0.6))
            .unwrap();
        fluid
            .fill_box(Vec3::ZERO, Vec3::new(0.4, 0.5, 0.6))
            .unwrap();
        fluid
            .add_emitter(
                Vec3::new(0.05, 0.7, 0.2),
                Vec3::new(0.2, 0.8, 0.4),
                Vec3::new(1.0, 0.0, 0.0),
            )
            .unwrap();
        fluid
            .add_drain(Vec3::new(0.8, 0.0, 0.0), Vec3::new(1.0, 0.1, 0.6))
            .unwrap();
        fluid
    }

    fn steps(fluid: &mut FluidWorld, count: usize) {
        for _ in 0..count {
            fluid.step(TICK, &Sequential).unwrap();
        }
    }

    #[test]
    fn water_past_the_domain_is_taken_out_and_the_domain_kept() {
        let mut fluid = pool();
        // the pool's water is 0.5 m deep, the domain ends at 0.3 m
        let (low, high) = (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(2.0, 0.3, 2.0));
        fluid.set_domain(Some((low, high))).unwrap();
        let before = fluid.len();
        let report = fluid.step(TICK, &Sequential).unwrap();
        assert!(report.lost > 0);
        assert!(fluid.positions().iter().all(|at| at.y <= 0.3));
        assert!(fluid.len() + report.lost as usize <= before + report.emitted as usize);
        let loaded = FluidWorld::load_snapshot(&fluid.save_snapshot()).unwrap();
        assert_eq!(loaded.domain(), Some((low, high)));
        assert!(fluid.set_domain(Some((high, low))).is_err());
        assert!(
            fluid
                .set_domain(Some((low, Vec3::splat(f32::NAN))))
                .is_err()
        );
    }

    #[test]
    fn a_loaded_fluid_steps_on_exactly_as_the_saved_one() {
        let mut fluid = pool();
        steps(&mut fluid, 20);
        let saved = fluid.save_snapshot();
        let mut loaded = FluidWorld::load_snapshot(&saved).unwrap();
        assert_eq!(loaded.save_snapshot(), saved);
        assert_eq!(loaded.state_hash(), fluid.state_hash());
        steps(&mut fluid, 30);
        steps(&mut loaded, 30);
        assert_eq!(loaded.state_hash(), fluid.state_hash());
        assert_eq!(loaded.len(), fluid.len());
    }

    #[test]
    fn a_cut_or_flipped_fluid_snapshot_is_refused_without_a_panic() {
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        fluid
            .add_box_container(Vec3::ZERO, Vec3::new(0.3, 0.3, 0.3))
            .unwrap();
        fluid
            .fill_box(Vec3::ZERO, Vec3::new(0.15, 0.15, 0.15))
            .unwrap();
        fluid.add_drain(Vec3::ZERO, Vec3::splat(0.05)).unwrap();
        let saved = fluid.save_snapshot();
        for end in 0..saved.len() {
            assert!(
                FluidWorld::load_snapshot(&saved[..end]).is_err(),
                "cut at {end}"
            );
        }
        for at in 0..saved.len() {
            let mut flipped = saved.clone();
            flipped[at] ^= 0x41;
            assert!(FluidWorld::load_snapshot(&flipped).is_err(), "flip at {at}");
        }
        // a good checksum over bad content is still refused
        let mut bad = saved[..saved.len() - CHECKSUM_BYTES].to_vec();
        // the solver tag sits right after magic and version
        bad[8] = 7;
        let sum = fnv1a(&bad);
        bad.extend_from_slice(&sum.to_le_bytes());
        assert_eq!(
            FluidWorld::load_snapshot(&bad).err(),
            Some(SnapshotError::Invalid("fluid solver"))
        );
        let mut later = saved[..saved.len() - CHECKSUM_BYTES].to_vec();
        later[4..8].copy_from_slice(&(FLUID_SNAPSHOT_VERSION + 1).to_le_bytes());
        let sum = fnv1a(&later);
        later.extend_from_slice(&sum.to_le_bytes());
        assert_eq!(
            FluidWorld::load_snapshot(&later).err(),
            Some(SnapshotError::UnsupportedVersion(
                FLUID_SNAPSHOT_VERSION + 1
            ))
        );
    }
}
