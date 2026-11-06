use algebrix::Vec3;

use super::FluidWorld;
use crate::error::PhysicsError;

/// New water goes only where none is nearer than this share of a spacing.
const CLEAR_SHARE: f32 = 0.9;

/// An emitter or drain, so it can be taken away again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FlowId(u32);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    /// Keeps its lattice slots filled, new water moving at this velocity.
    Emitter(Vec3),
    Drain,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Flow {
    id: u32,
    low: Vec3,
    high: Vec3,
    kind: Kind,
}

#[derive(Debug, Clone, Default)]
pub(super) struct Flows {
    list: Vec<Flow>,
    next: u32,
}

impl FluidWorld {
    /// A box that tops itself up after each step: every lattice slot in it with no water gets a
    /// particle moving at `velocity`. Water already there stops it, so a full reservoir keeps its
    /// level at the box's top.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box or velocity.
    pub fn add_emitter(
        &mut self,
        low: Vec3,
        high: Vec3,
        velocity: Vec3,
    ) -> Result<FlowId, PhysicsError> {
        if !velocity.is_finite() {
            return Err(PhysicsError::NonFinite("emitter velocity"));
        }
        self.add_flow(low, high, Kind::Emitter(velocity))
    }

    /// A box that takes away the water in it after each step.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box.
    pub fn add_drain(&mut self, low: Vec3, high: Vec3) -> Result<FlowId, PhysicsError> {
        self.add_flow(low, high, Kind::Drain)
    }

    /// Returns false for an id that is not there.
    pub fn remove_flow(&mut self, id: FlowId) -> bool {
        let before = self.flows.list.len();
        self.flows.list.retain(|flow| flow.id != id.0);
        self.flows.list.len() != before
    }

    fn add_flow(&mut self, low: Vec3, high: Vec3, kind: Kind) -> Result<FlowId, PhysicsError> {
        if !low.is_finite() || !high.is_finite() {
            return Err(PhysicsError::NonFinite("flow box"));
        }
        if high.x <= low.x || high.y <= low.y || high.z <= low.z {
            return Err(PhysicsError::OutOfRange);
        }
        let id = self.flows.next;
        self.flows.next = self
            .flows
            .next
            .checked_add(1)
            .ok_or(PhysicsError::FluidFull)?;
        self.flows.list.push(Flow {
            id,
            low,
            high,
            kind,
        });
        Ok(FlowId(id))
    }

    /// Drains first, then emitters, each in the order they were added. Returns how many particles
    /// came in and went out.
    pub(super) fn run_flows(&mut self) -> Result<(u32, u32), PhysicsError> {
        if self.flows.list.is_empty() {
            return Ok((0, 0));
        }
        let mut drained = 0;
        for flow in self.flows.list.clone() {
            if flow.kind != Kind::Drain {
                continue;
            }
            let inside = |at: &Vec3| {
                at.x >= flow.low.x
                    && at.y >= flow.low.y
                    && at.z >= flow.low.z
                    && at.x <= flow.high.x
                    && at.y <= flow.high.y
                    && at.z <= flow.high.z
            };
            let keep: Vec<bool> = self.positions.iter().map(|at| !inside(at)).collect();
            let gone = keep.iter().filter(|kept| !**kept).count();
            if gone > 0 {
                drained += gone as u32;
                let mut index = 0;
                let mut take = |_: &u32| {
                    index += 1;
                    keep[index - 1]
                };
                self.ids.retain(&mut take);
                for list in [&mut self.positions, &mut self.velocities] {
                    let mut index = 0;
                    list.retain(|_| {
                        index += 1;
                        keep[index - 1]
                    });
                }
                let mut index = 0;
                self.densities.retain(|_| {
                    index += 1;
                    keep[index - 1]
                });
            }
        }
        let mut emitted = 0;
        let spacing = self.settings.spacing;
        for flow in self.flows.list.clone() {
            let Kind::Emitter(velocity) = flow.kind else {
                continue;
            };
            let size = flow.high - flow.low;
            let count = |extent: f32| ((extent / spacing).floor() as i64).clamp(0, 1 << 12);
            let cells = (count(size.x), count(size.y), count(size.z));
            let total = cells.0 * cells.1 * cells.2;
            if total == 0 {
                continue;
            }
            // a slot is taken when water is nearer its center than most of a spacing. by cell
            // alone, water a hair past the slot's edge left it empty, the new particle landed on
            // top of it, the squeeze threw water out and emptied more slots
            let clear = (CLEAR_SHARE * spacing).powi(2);
            let mut taken = vec![false; total as usize];
            for at in &self.positions {
                let local = (*at - flow.low) / spacing - Vec3::splat(0.5);
                let nearest = |value: f32| value.round() as i64;
                let (cx, cy, cz) = (nearest(local.x), nearest(local.y), nearest(local.z));
                for y in cy - 1..=cy + 1 {
                    for z in cz - 1..=cz + 1 {
                        for x in cx - 1..=cx + 1 {
                            if !((0..cells.0).contains(&x)
                                && (0..cells.1).contains(&y)
                                && (0..cells.2).contains(&z))
                            {
                                continue;
                            }
                            let center = flow.low
                                + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5)
                                    * spacing;
                            if (*at - center).length_squared() < clear {
                                taken[((y * cells.2 + z) * cells.0 + x) as usize] = true;
                            }
                        }
                    }
                }
            }
            for y in 0..cells.1 {
                for z in 0..cells.2 {
                    for x in 0..cells.0 {
                        if taken[((y * cells.2 + z) * cells.0 + x) as usize] {
                            continue;
                        }
                        let at = flow.low
                            + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * spacing;
                        self.add_particle(at, velocity)?;
                        emitted += 1;
                    }
                }
            }
        }
        Ok((emitted, drained))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::Sequential;
    use crate::fluid::FluidSettings;

    const TICK: f32 = 1.0 / 60.0;

    fn channel() -> FluidWorld {
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        fluid
            .add_box_container(Vec3::ZERO, Vec3::new(2.0, 0.6, 0.3))
            .unwrap();
        fluid
    }

    #[test]
    fn an_emitter_fills_its_box_once_and_tops_it_up() {
        let mut fluid = channel();
        fluid
            .add_emitter(Vec3::ZERO, Vec3::new(0.2, 0.3, 0.3), Vec3::ZERO)
            .unwrap();
        let report = fluid.step(TICK, &Sequential).unwrap();
        assert_eq!(report.emitted, 4 * 6 * 6);
        assert_eq!(fluid.len(), 4 * 6 * 6);
        // full now, the water spreads out of it and gets replaced
        let mut total = report.emitted;
        for _ in 0..30 {
            total += fluid.step(TICK, &Sequential).unwrap().emitted;
        }
        assert_eq!(fluid.len() as u32, total);
        assert!(total > 4 * 6 * 6);
    }

    #[test]
    fn a_river_from_an_emitter_to_a_drain_keeps_its_count_and_books() {
        let mut fluid = channel();
        let speed = Vec3::new(1.0, 0.0, 0.0);
        fluid
            .add_emitter(Vec3::ZERO, Vec3::new(0.15, 0.3, 0.3), speed)
            .unwrap();
        fluid
            .add_drain(Vec3::new(1.8, 0.0, 0.0), Vec3::new(2.0, 0.6, 0.3))
            .unwrap();
        let (mut emitted, mut drained) = (0, 0);
        let mut counts = Vec::new();
        for _ in 0..240 {
            let report = fluid.step(TICK, &Sequential).unwrap();
            emitted += report.emitted;
            drained += report.drained;
            assert_eq!(report.held_back, 0);
            counts.push(fluid.len());
        }
        assert_eq!(fluid.len() as u32, emitted - drained);
        assert!(drained > 0, "water reached the drain");
        // the last second neither fills up nor empties
        let late = &counts[180..];
        let (low, high) = (late.iter().min().unwrap(), late.iter().max().unwrap());
        let spread = (*high - *low) as f32 / *high as f32;
        // it sloshes a little between emitter and drain
        assert!(spread < 0.15, "{low} {high}");
        assert!(fluid.positions().iter().all(|at| at.x < 1.8 + 0.1));
    }

    #[test]
    fn an_emitter_never_puts_water_on_top_of_water() {
        let mut fluid = channel();
        let spacing = fluid.settings().spacing;
        // a particle a little past the first slot's edge
        fluid
            .add_particle(
                Vec3::new(spacing * 1.05, spacing * 0.5, spacing * 0.5),
                Vec3::ZERO,
            )
            .unwrap();
        fluid
            .add_emitter(Vec3::ZERO, Vec3::new(0.1, 0.05, 0.05), Vec3::ZERO)
            .unwrap();
        fluid.run_flows().unwrap();
        let positions = fluid.positions();
        for (a, at) in positions.iter().enumerate() {
            for other in &positions[a + 1..] {
                assert!((*at - *other).length() > 0.5 * spacing, "{at:?} {other:?}");
            }
        }
    }

    #[test]
    fn flows_refuse_bad_boxes_and_go_away() {
        let mut fluid = channel();
        assert!(fluid.add_drain(Vec3::ONE, Vec3::ZERO).is_err());
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(fluid.add_emitter(Vec3::ZERO, Vec3::ONE, nan).is_err());
        assert!(fluid.add_drain(nan, Vec3::ONE).is_err());
        let drain = fluid.add_drain(Vec3::ZERO, Vec3::ONE).unwrap();
        assert!(fluid.remove_flow(drain));
        assert!(!fluid.remove_flow(drain));
    }

    #[test]
    fn a_full_world_stops_the_emitter_with_an_error() {
        let mut fluid = FluidWorld::new(FluidSettings {
            max_particles: 10,
            ..FluidSettings::default()
        })
        .unwrap();
        fluid
            .add_emitter(Vec3::ZERO, Vec3::splat(0.2), Vec3::ZERO)
            .unwrap();
        assert_eq!(fluid.step(TICK, &Sequential), Err(PhysicsError::FluidFull));
        assert_eq!(fluid.len(), 10);
    }
}
