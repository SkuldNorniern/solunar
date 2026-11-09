use algebrix::Vec3;

use super::{FluidWorld, ShallowWater};
use crate::error::PhysicsError;

/// Where water changes form between shallow columns and particles, in the same local space.
/// Columns whose middle is inside `release` in x and z become particles, from their floor up.
/// Particles inside `settle` at or under the sheet's surface become column water again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Handoff {
    pub release_low: Vec3,
    pub release_high: Vec3,
    pub settle_low: Vec3,
    pub settle_high: Vec3,
}

/// What one hand off moved. Volumes in cubic meters, each way.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HandoffReport {
    pub released: u32,
    pub settled: u32,
    pub to_particles: f64,
    pub to_columns: f64,
}

fn inside(at: Vec3, low: Vec3, high: Vec3) -> bool {
    at.x >= low.x
        && at.y >= low.y
        && at.z >= low.z
        && at.x <= high.x
        && at.y <= high.y
        && at.z <= high.z
}

/// Moves water between `water` and `fluid` as `handoff` says, keeping volume and momentum: a
/// column loses exactly what its new particles hold and carry, a column gains exactly what the
/// particles it takes in held and carried. Run once a tick after both have stepped.
///
/// # Errors
/// Fails on a non finite or inside out box, or when the fluid is full. A full fluid leaves the
/// columns it could not empty as they were.
pub fn hand_off(
    water: &mut ShallowWater,
    fluid: &mut FluidWorld,
    handoff: &Handoff,
) -> Result<HandoffReport, PhysicsError> {
    for (low, high) in [
        (handoff.release_low, handoff.release_high),
        (handoff.settle_low, handoff.settle_high),
    ] {
        if !low.is_finite() || !high.is_finite() {
            return Err(PhysicsError::NonFinite("hand off box"));
        }
        if high.x < low.x || high.y < low.y || high.z < low.z {
            return Err(PhysicsError::OutOfRange);
        }
    }
    let mut report = HandoffReport::default();
    let volume = f64::from(fluid.particle_mass() / fluid.settings().rest_density);
    let cell = water.settings.cell;
    let area = f64::from(cell) * f64::from(cell);
    settle(water, fluid, handoff, volume, area, &mut report);
    release(water, fluid, handoff, volume, area, &mut report)?;
    Ok(report)
}

fn settle(
    water: &mut ShallowWater,
    fluid: &mut FluidWorld,
    handoff: &Handoff,
    volume: f64,
    area: f64,
    report: &mut HandoffReport,
) {
    let spacing = fluid.settings().spacing;
    let mut keep = vec![true; fluid.len()];
    for (index, kept) in keep.iter_mut().enumerate() {
        let at = fluid.positions[index];
        if !inside(at, handoff.settle_low, handoff.settle_high) {
            continue;
        }
        let Some(column) = water.cell_at(at) else {
            continue;
        };
        // touching the sheet, or the floor where it is dry
        if at.y > water.surface(column) + spacing {
            continue;
        }
        let slot = water.index(column.0, column.1);
        let share = (volume / area) as f32;
        let velocity = fluid.velocities[index];
        water.depth[slot] += share;
        water.flow_x[slot] += share * velocity.x;
        water.flow_z[slot] += share * velocity.z;
        *kept = false;
        report.settled += 1;
        report.to_columns += volume;
    }
    if report.settled > 0 {
        fluid.keep_only(&keep);
    }
}

fn release(
    water: &mut ShallowWater,
    fluid: &mut FluidWorld,
    handoff: &Handoff,
    volume: f64,
    area: f64,
    report: &mut HandoffReport,
) -> Result<(), PhysicsError> {
    let spacing = fluid.settings().spacing;
    let cell = water.settings.cell;
    let across = ((cell / spacing).floor() as usize).max(1);
    let clear = (0.9 * spacing).powi(2);
    let (width, depth_cells) = water.size;
    for z in 0..depth_cells {
        for x in 0..width {
            let middle =
                water.origin + Vec3::new((x as f32 + 0.5) * cell, 0.0, (z as f32 + 0.5) * cell);
            let (low, high) = (handoff.release_low, handoff.release_high);
            if middle.x < low.x || middle.x > high.x || middle.z < low.z || middle.z > high.z {
                continue;
            }
            let slot = water.index(x, z);
            let depth = water.depth[slot];
            let wanted = (f64::from(depth) * area / volume).floor() as usize;
            if wanted == 0 {
                continue;
            }
            let velocity = water.velocity((x, z));
            let floor = water.floor[slot];
            let corner = water.origin + Vec3::new(x as f32 * cell, floor, z as f32 * cell);
            let layers = (f64::from(depth) / f64::from(spacing)).ceil() as usize + 2;
            // water already in the column keeps its place
            let near: Vec<Vec3> = fluid
                .positions
                .iter()
                .copied()
                .filter(|at| {
                    at.x >= corner.x - spacing
                        && at.x <= corner.x + cell + spacing
                        && at.z >= corner.z - spacing
                        && at.z <= corner.z + cell + spacing
                })
                .collect();
            let mut placed = 0;
            'fill: for layer in 0..layers {
                for j in 0..across {
                    for i in 0..across {
                        if placed == wanted {
                            break 'fill;
                        }
                        let at = corner
                            + Vec3::new(i as f32 + 0.5, layer as f32 + 0.5, j as f32 + 0.5)
                                * spacing;
                        if near
                            .iter()
                            .any(|other| (*other - at).length_squared() < clear)
                        {
                            continue;
                        }
                        fluid.add_particle(at, velocity)?;
                        placed += 1;
                    }
                }
            }
            if placed == 0 {
                continue;
            }
            let taken = (placed as f64 * volume / area) as f32;
            let left = (depth - taken).max(0.0);
            water.depth[slot] = left;
            water.flow_x[slot] = left * velocity.x;
            water.flow_z[slot] = left * velocity.z;
            report.released += placed as u32;
            report.to_particles += placed as f64 * volume;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sequential;
    use crate::fluid::{FluidSettings, ShallowSettings};

    const TICK: f32 = 1.0 / 60.0;

    /// 6 m by 1 m in 0.25 m cells, the upper half 0.5 m higher, water 0.3 m deep on it.
    fn falls() -> (ShallowWater, FluidWorld, Handoff) {
        let (width, across) = (24, 4);
        let bed: Vec<f32> = (0..width * across)
            .map(|index| if index % width < 12 { 0.5 } else { 0.0 })
            .collect();
        let mut water =
            ShallowWater::new(ShallowSettings::default(), Vec3::ZERO, (width, across), bed)
                .unwrap();
        water.fill((0, 0), (9, across - 1), 0.8).unwrap();
        let mut fluid = FluidWorld::new(FluidSettings {
            spacing: 0.05,
            ..FluidSettings::default()
        })
        .unwrap();
        fluid
            .add_box_container(Vec3::ZERO, Vec3::new(6.0, 2.0, 1.0))
            .unwrap();
        fluid
            .add_wall(Vec3::ZERO, Vec3::new(3.0, 0.5, 1.0))
            .unwrap();
        let handoff = Handoff {
            release_low: Vec3::new(2.75, 0.0, 0.0),
            release_high: Vec3::new(3.0, 2.0, 1.0),
            settle_low: Vec3::new(4.0, 0.0, 0.0),
            settle_high: Vec3::new(6.0, 2.0, 1.0),
        };
        (water, fluid, handoff)
    }

    fn total(water: &ShallowWater, fluid: &FluidWorld) -> f64 {
        let each = f64::from(fluid.particle_mass() / fluid.settings().rest_density);
        water.volume() + fluid.len() as f64 * each
    }

    #[test]
    fn water_goes_over_the_falls_as_particles_and_lands_as_columns() {
        let (mut water, mut fluid, handoff) = falls();
        let start = total(&water, &fluid);
        let (mut released, mut settled) = (0, 0);
        for _ in 0..90 {
            water.step(TICK, &Sequential).unwrap();
            fluid.step(TICK, &Sequential).unwrap();
            let report = hand_off(&mut water, &mut fluid, &handoff).unwrap();
            released += report.released;
            settled += report.settled;
            let now = total(&water, &fluid);
            assert!((now - start).abs() < start * 1e-4, "{now} of {start}");
        }
        assert!(released > 100, "{released} released");
        assert!(settled > 0, "{settled} settled");
        let below = water.cell_at(Vec3::new(5.0, 0.0, 0.5)).unwrap();
        assert!(water.surface(below) > 0.0);
    }

    #[test]
    fn one_hand_off_keeps_volume_and_momentum() {
        let (mut water, mut fluid, handoff) = falls();
        for _ in 0..20 {
            water.step(TICK, &Sequential).unwrap();
        }
        // moving particles in the settle box, touching the dry floor
        for x in 0..10 {
            fluid
                .add_particle(
                    Vec3::new(4.2 + x as f32 * 0.1, 0.03, 0.5),
                    Vec3::new(1.5, -0.5, 0.25),
                )
                .unwrap();
        }
        let each = f64::from(fluid.particle_mass() / fluid.settings().rest_density);
        let cell = f64::from(water.settings.cell);
        let momentum = |water: &ShallowWater, fluid: &FluidWorld| {
            let columns: (f64, f64) = water
                .flow_x
                .iter()
                .zip(&water.flow_z)
                .map(|(x, z)| (f64::from(*x) * cell * cell, f64::from(*z) * cell * cell))
                .fold((0.0, 0.0), |sum, (x, z)| (sum.0 + x, sum.1 + z));
            let particles = fluid.velocities().iter().fold((0.0, 0.0), |sum, v| {
                (sum.0 + f64::from(v.x) * each, sum.1 + f64::from(v.z) * each)
            });
            (columns.0 + particles.0, columns.1 + particles.1)
        };
        let (volume, (mx, mz)) = (total(&water, &fluid), momentum(&water, &fluid));
        let report = hand_off(&mut water, &mut fluid, &handoff).unwrap();
        assert!(report.released > 0 && report.settled == 10, "{report:?}");
        assert!((total(&water, &fluid) - volume).abs() < volume * 1e-5);
        let (nx, nz) = momentum(&water, &fluid);
        assert!((nx - mx).abs() < 1e-3 * mx.abs().max(1.0), "{mx} {nx}");
        assert!((nz - mz).abs() < 1e-3 * mz.abs().max(1.0), "{mz} {nz}");
        // bad boxes are refused
        let inside_out = Handoff {
            settle_low: Vec3::splat(1.0),
            settle_high: Vec3::ZERO,
            ..handoff
        };
        assert!(hand_off(&mut water, &mut fluid, &inside_out).is_err());
    }
}
