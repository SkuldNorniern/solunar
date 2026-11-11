use algebrix::{DVec3, Quat, Vec3};

use crate::QueryFilter;
use crate::body::BodyHandle;
use crate::error::PhysicsError;
use crate::frame::FrameId;
use crate::position::GridPosition;
use crate::shape::Shape;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::world::PhysicsWorld;

/// Where a held end is now, from the rope's first point, and the body holding it with its arm.
type Pin = Option<(Vec3, Option<(BodyHandle, Vec3)>)>;

/// Rope steps a world step: small steps, one pass each, as xpbd does best.
const ROPE_SUBSTEPS: u32 = 8;
/// Most segments in one rope.
pub const MAX_SEGMENTS: u32 = 1024;
/// Most ropes a world holds.
const MAX_ROPES: usize = 1 << 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RopeHandle {
    index: u32,
    generation: u32,
}

/// What holds an end of a rope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeEnd {
    Free,
    /// Held at a point of the rope's frame.
    Fixed(GridPosition),
    /// Tied to a body at `local`, in its own axes. The rope pulls on it.
    Body {
        body: BodyHandle,
        local: Vec3,
    },
}

/// A rope or cable: particles along `segments` equal segments from `start` to `end`, drawn
/// straight at first. Stretch and bend are compliant constraints, so how stiff it is does not
/// hang on the step: compliance is meters per newton, zero for as stiff as the step allows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeDesc {
    pub start: GridPosition,
    pub end: GridPosition,
    pub segments: u32,
    /// Its thickness, for touching bodies, meters.
    pub radius: f32,
    /// Kilograms a meter.
    pub mass_per_meter: f32,
    pub stretch_compliance: f32,
    pub bend_compliance: f32,
    /// How fast stretching along it dies out, per second, as a real rope's own losses do.
    /// Zero keeps a soft rope bouncing on its own stretch forever.
    pub damping: f32,
    pub friction: f32,
    /// Newtons a segment takes before it parts, infinite for never.
    pub break_force: f32,
    pub frame: FrameId,
    /// What it touches.
    pub filter: QueryFilter,
    pub ends: [RopeEnd; 2],
}

impl RopeDesc {
    pub fn new(start: GridPosition, end: GridPosition, segments: u32) -> Self {
        Self {
            start,
            end,
            segments,
            radius: 0.02,
            mass_per_meter: 0.5,
            stretch_compliance: 0.0,
            bend_compliance: 1e-2,
            damping: 5.0,
            friction: 0.5,
            break_force: f32::INFINITY,
            frame: FrameId::ROOT,
            filter: QueryFilter::default(),
            ends: [RopeEnd::Free; 2],
        }
    }

    fn is_valid(&self) -> bool {
        let at_least_zero = |v: f32| v.is_finite() && v >= 0.0;
        self.start.local.is_finite()
            && self.end.local.is_finite()
            && (1..=MAX_SEGMENTS).contains(&self.segments)
            && self.radius.is_finite()
            && self.radius > 0.0
            && self.mass_per_meter.is_finite()
            && self.mass_per_meter > 0.0
            && at_least_zero(self.stretch_compliance)
            && at_least_zero(self.bend_compliance)
            && at_least_zero(self.damping)
            && at_least_zero(self.friction)
            && !self.break_force.is_nan()
            && self.break_force > 0.0
            && self.ends.iter().all(|end| match end {
                RopeEnd::Free => true,
                RopeEnd::Fixed(at) => at.local.is_finite(),
                RopeEnd::Body { local, .. } => local.is_finite(),
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Rope {
    desc: RopeDesc,
    /// Segment length at rest.
    rest: f32,
    points: Vec<GridPosition>,
    velocity: Vec<Vec3>,
    /// Newtons each segment carried last step, the most over its substeps.
    tension: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Ropes {
    slots: Vec<Option<Rope>>,
    generations: Vec<u32>,
    /// Ropes that parted last step: the one cut and the new one beyond the cut.
    broken: Vec<(RopeHandle, RopeHandle)>,
}

impl Ropes {
    fn insert(&mut self, rope: Rope) -> Result<RopeHandle, PhysicsError> {
        let index = match self.slots.iter().position(Option::is_none) {
            Some(free) => free,
            None => {
                if self.slots.len() >= MAX_ROPES {
                    return Err(PhysicsError::InvalidShape);
                }
                self.slots.push(None);
                self.generations.push(0);
                self.slots.len() - 1
            }
        };
        self.slots[index] = Some(rope);
        Ok(RopeHandle {
            index: index as u32,
            generation: self.generations[index],
        })
    }

    fn get(&self, handle: RopeHandle) -> Option<&Rope> {
        let index = handle.index as usize;
        if self.generations.get(index) != Some(&handle.generation) {
            return None;
        }
        self.slots.get(index)?.as_ref()
    }

    pub(crate) fn save(&self, out: &mut Writer) {
        out.len(self.slots.len());
        for (slot, generation) in self.slots.iter().zip(&self.generations) {
            out.u32(*generation);
            let Some(rope) = slot else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            let d = &rope.desc;
            out.position(d.start);
            out.position(d.end);
            out.u32(d.segments);
            for value in [
                d.radius,
                d.mass_per_meter,
                d.stretch_compliance,
                d.bend_compliance,
                d.damping,
                d.friction,
                d.break_force,
            ] {
                out.f32(value);
            }
            out.u32(d.frame.index());
            out.u32(d.filter.frame.index());
            out.u32(d.filter.mask);
            match d.filter.exclude {
                None => out.u8(0),
                Some(body) => {
                    out.u8(1);
                    out.body(body);
                }
            }
            for end in d.ends {
                match end {
                    RopeEnd::Free => out.u8(0),
                    RopeEnd::Fixed(at) => {
                        out.u8(1);
                        out.position(at);
                    }
                    RopeEnd::Body { body, local } => {
                        out.u8(2);
                        out.body(body);
                        out.vec3(local);
                    }
                }
            }
            out.f32(rope.rest);
            out.len(rope.points.len());
            for ((point, velocity), tension) in
                rope.points.iter().zip(&rope.velocity).zip(&rope.tension)
            {
                out.position(*point);
                out.vec3(*velocity);
                out.f32(*tension);
            }
        }
    }

    pub(crate) fn load(reader: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let bad = || SnapshotError::Invalid("rope");
        let count = reader.count(5, MAX_ROPES)?;
        let mut ropes = Self::default();
        for _ in 0..count {
            ropes.generations.push(reader.u32()?);
            match reader.u8()? {
                0 => {
                    ropes.slots.push(None);
                    continue;
                }
                1 => {}
                _ => return Err(bad()),
            }
            let (start, end, segments) = (reader.position()?, reader.position()?, reader.u32()?);
            let mut desc = RopeDesc::new(start, end, segments);
            desc.radius = reader.f32()?;
            desc.mass_per_meter = reader.f32()?;
            desc.stretch_compliance = reader.f32()?;
            desc.bend_compliance = reader.f32()?;
            desc.damping = reader.f32()?;
            desc.friction = reader.f32()?;
            desc.break_force = reader.f32()?;
            desc.frame = FrameId(reader.u32()?);
            desc.filter.frame = FrameId(reader.u32()?);
            desc.filter.mask = reader.u32()?;
            desc.filter.exclude = match reader.u8()? {
                0 => None,
                1 => Some(reader.handle()?),
                _ => return Err(bad()),
            };
            for end in &mut desc.ends {
                *end = match reader.u8()? {
                    0 => RopeEnd::Free,
                    1 => RopeEnd::Fixed(reader.position()?),
                    2 => RopeEnd::Body {
                        body: reader.handle()?,
                        local: reader.vec3()?,
                    },
                    _ => return Err(bad()),
                };
            }
            let rest = reader.f32()?;
            let points = reader.count(40, MAX_SEGMENTS as usize + 1)?;
            if !desc.is_valid() || !rest.is_finite() || rest <= 0.0 || points < 2 {
                return Err(bad());
            }
            let mut rope = Rope {
                desc,
                rest,
                points: Vec::with_capacity(points),
                velocity: Vec::with_capacity(points),
                tension: Vec::with_capacity(points),
            };
            for _ in 0..points {
                let (at, velocity, tension) = (reader.position()?, reader.vec3()?, reader.f32()?);
                if !at.local.is_finite() || !velocity.is_finite() || !tension.is_finite() {
                    return Err(bad());
                }
                rope.points.push(at);
                rope.velocity.push(velocity);
                rope.tension.push(tension);
            }
            ropes.slots.push(Some(rope));
        }
        Ok(ropes)
    }
}

impl PhysicsWorld {
    /// Whether every rope stands and looks in frames this world has, for a loaded snapshot.
    pub(crate) fn ropes_in_known_frames(&self, ropes: &Ropes) -> bool {
        ropes.slots.iter().flatten().all(|rope| {
            self.frames.contains(rope.desc.frame) && self.frames.contains(rope.desc.filter.frame)
        })
    }

    /// # Errors
    /// Fails on a bad desc, an unknown frame or a body end on a body the world does not hold.
    pub fn add_rope(&mut self, desc: RopeDesc) -> Result<RopeHandle, PhysicsError> {
        if !desc.is_valid() {
            return Err(PhysicsError::InvalidShape);
        }
        if !self.frames.contains(desc.frame) {
            return Err(PhysicsError::InvalidFrame);
        }
        for end in desc.ends {
            if let RopeEnd::Body { body, .. } = end {
                self.bodies.dense_index(body)?;
            }
        }
        let span = desc.end.offset_from(desc.start);
        let length = span.length() as f32;
        if length <= 0.0 {
            return Err(PhysicsError::InvalidShape);
        }
        let count = desc.segments as usize + 1;
        let points: Vec<GridPosition> = (0..count)
            .map(|i| {
                desc.start
                    .shifted(span * (i as f64 / desc.segments as f64))
                    .ok_or(PhysicsError::OutOfRange)
            })
            .collect::<Result<_, _>>()?;
        self.ropes.insert(Rope {
            desc,
            rest: length / desc.segments as f32,
            velocity: vec![Vec3::ZERO; count],
            tension: vec![0.0; count],
            points,
        })
    }

    /// # Errors
    /// Fails on an unknown handle.
    pub fn remove_rope(&mut self, handle: RopeHandle) -> Result<(), PhysicsError> {
        self.ropes.get(handle).ok_or(PhysicsError::InvalidHandle)?;
        let index = handle.index as usize;
        self.ropes.slots[index] = None;
        self.ropes.generations[index] = self.ropes.generations[index].wrapping_add(1);
        Ok(())
    }

    /// The rope's particles from its start, in its frame.
    ///
    /// # Errors
    /// Fails on an unknown handle.
    pub fn rope_points(&self, handle: RopeHandle) -> Result<&[GridPosition], PhysicsError> {
        Ok(&self
            .ropes
            .get(handle)
            .ok_or(PhysicsError::InvalidHandle)?
            .points)
    }

    /// Newtons each segment carried last step, from its start.
    ///
    /// # Errors
    /// Fails on an unknown handle.
    pub fn rope_tension(&self, handle: RopeHandle) -> Result<&[f32], PhysicsError> {
        let rope = self.ropes.get(handle).ok_or(PhysicsError::InvalidHandle)?;
        Ok(&rope.tension[..rope.points.len() - 1])
    }

    /// Ropes that parted in the last step: the rope cut, which keeps the start's side, and the
    /// new one beyond the cut.
    pub fn broken_ropes(&self) -> &[(RopeHandle, RopeHandle)] {
        &self.ropes.broken
    }

    /// After the rigid step: each rope in handle order on its own small steps, its body ends
    /// pulled as the bodies now stand, what it pulls given to them after.
    pub(crate) fn step_ropes(&mut self, dt: f32) -> Result<(), PhysicsError> {
        self.ropes.broken.clear();
        if self.ropes.slots.iter().all(Option::is_none) {
            return Ok(());
        }
        let mut slots = std::mem::take(&mut self.ropes.slots);
        let mut result = Ok(());
        let mut parted = Vec::new();
        for (index, slot) in slots.iter_mut().enumerate() {
            let Some(rope) = slot else {
                continue;
            };
            match self.step_rope(rope, dt) {
                Ok(Some(cut)) => parted.push((index, cut)),
                Ok(None) => {}
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.ropes.slots = slots;
        result?;
        for (index, cut) in parted {
            let handle = RopeHandle {
                index: index as u32,
                generation: self.ropes.generations[index],
            };
            let Some(rope) = self.ropes.slots[index].as_mut() else {
                continue;
            };
            let mut far = rope.clone();
            far.points.drain(..=cut);
            far.velocity.drain(..=cut);
            far.tension.drain(..=cut);
            far.desc.ends[0] = RopeEnd::Free;
            far.desc.segments = (far.points.len() - 1) as u32;
            rope.points.truncate(cut + 1);
            rope.velocity.truncate(cut + 1);
            rope.tension.truncate(cut + 1);
            rope.desc.ends[1] = RopeEnd::Free;
            rope.desc.segments = cut as u32;
            // a piece of a single point is no rope: parted at an end, the rope only lets go
            match (rope.points.len() >= 2, far.points.len() >= 2) {
                (true, true) => {
                    let new = self.ropes.insert(far)?;
                    self.ropes.broken.push((handle, new));
                }
                (false, _) => {
                    *rope = far;
                    self.ropes.broken.push((handle, handle));
                }
                (true, false) => self.ropes.broken.push((handle, handle)),
            }
        }
        Ok(())
    }

    /// One rope through one world step. The segment it parts at, if one carried more than its
    /// break force.
    // segments read neighbouring points of several arrays by index, plainer than zips
    #[allow(clippy::needless_range_loop)]
    fn step_rope(&mut self, rope: &mut Rope, dt: f32) -> Result<Option<usize>, PhysicsError> {
        let desc = rope.desc;
        let count = rope.points.len();
        let origin = rope.points[0];
        let h = dt / ROPE_SUBSTEPS as f32;
        let mass = desc.mass_per_meter * rope.rest;
        let mut x: Vec<Vec3> = rope
            .points
            .iter()
            .map(|p| p.offset_from(origin).as_vec3())
            .collect();
        let mut v = rope.velocity.clone();
        // where each held end is now, and the body pulled
        let mut pinned: [Pin; 2] = [None, None];
        for (side, end) in desc.ends.iter().enumerate() {
            pinned[side] = match *end {
                RopeEnd::Free => None,
                RopeEnd::Fixed(at) => Some((at.offset_from(origin).as_vec3(), None)),
                RopeEnd::Body { body, local } => match self.body_state(body) {
                    Ok(state) => {
                        let arm = state.rotation.rotate_vec3(local);
                        let at = state
                            .position
                            .shifted(DVec3::from_vec3(arm))
                            .ok_or(PhysicsError::OutOfRange)?;
                        Some((at.offset_from(origin).as_vec3(), Some((body, arm))))
                    }
                    // the body is gone, the end goes free
                    Err(_) => None,
                },
            };
        }
        let index_of = |side: usize| if side == 0 { 0 } else { count - 1 };
        let mut w = vec![1.0 / mass; count];
        for (side, pin) in pinned.iter().enumerate() {
            if let Some((at, _)) = pin {
                w[index_of(side)] = 0.0;
                x[index_of(side)] = *at;
            }
        }
        let gravity = self
            .gravity
            .at(desc.frame, origin, rope.velocity[count / 2]);
        let mut tension = vec![0.0_f32; count];
        let mut pulled = [Vec3::ZERO; 2];
        let stretch = desc.stretch_compliance / (h * h);
        let bend = desc.bend_compliance / (h * h);
        let mut previous = x.clone();
        let sphere = Shape::sphere(desc.radius);
        for _ in 0..ROPE_SUBSTEPS {
            for i in 0..count {
                previous[i] = x[i];
                if w[i] > 0.0 {
                    v[i] += gravity * h;
                    x[i] += v[i] * h;
                }
            }
            for k in 0..count - 1 {
                let (i, j) = (k, k + 1);
                let total = w[i] + w[j] + stretch;
                if total <= 0.0 {
                    continue;
                }
                let d = x[j] - x[i];
                let length = d.length();
                if length < 1e-9 {
                    continue;
                }
                let n = d / length;
                let lambda = -(length - rope.rest) / total;
                x[i] -= n * (w[i] * lambda);
                x[j] += n * (w[j] * lambda);
                let pull = -lambda / (h * h);
                tension[k] = tension[k].max(pull);
                if i == 0 && w[0] == 0.0 {
                    pulled[0] += n * (pull * h);
                }
                if j == count - 1 && w[j] == 0.0 {
                    pulled[1] -= n * (pull * h);
                }
            }
            for k in 0..count.saturating_sub(2) {
                let (i, j) = (k, k + 2);
                let total = w[i] + w[j] + bend;
                if total <= 0.0 {
                    continue;
                }
                let d = x[j] - x[i];
                let length = d.length();
                if length < 1e-9 {
                    continue;
                }
                let n = d / length;
                let lambda = -(length - 2.0 * rope.rest).min(0.0) / total;
                x[i] -= n * (w[i] * lambda);
                x[j] += n * (w[j] * lambda);
            }
            for i in 0..count {
                if w[i] == 0.0 {
                    continue;
                }
                let at = origin
                    .shifted(DVec3::from_vec3(x[i]))
                    .ok_or(PhysicsError::OutOfRange)?;
                if let Some((normal, depth)) =
                    self.deepest_overlap(&sphere, Quat::IDENTITY, at, &desc.filter)?
                {
                    x[i] += normal * depth;
                    // friction holds what it can of the slide this substep
                    let moved = x[i] - previous[i];
                    let slide = moved - normal * moved.dot(normal);
                    let length = slide.length();
                    if length > 1e-9 {
                        x[i] -= slide * (desc.friction * depth / length).min(1.0);
                    }
                }
            }
            for i in 0..count {
                if w[i] > 0.0 {
                    v[i] = (x[i] - previous[i]) / h;
                } else {
                    v[i] = Vec3::ZERO;
                }
            }
            // stretching speed along each segment bleeds off, shared by mass so momentum stays
            let keep = (-desc.damping * h).exp();
            for k in 0..count - 1 {
                let (i, j) = (k, k + 1);
                let total = w[i] + w[j];
                let d = x[j] - x[i];
                let length = d.length();
                if total <= 0.0 || length < 1e-9 {
                    continue;
                }
                let n = d / length;
                let lost = (v[j] - v[i]).dot(n) * (1.0 - keep);
                v[i] += n * (lost * w[i] / total);
                v[j] -= n * (lost * w[j] / total);
            }
        }
        // what the rope pulled on its body ends, given once for the step
        for (side, pin) in pinned.iter().enumerate() {
            if let Some((_, Some((body, arm)))) = pin {
                let impulse = pulled[side];
                if impulse.length_squared() > 0.0 {
                    self.apply_impulse_at(*body, impulse, *arm)?;
                }
            }
        }
        for i in 0..count {
            rope.points[i] = origin
                .shifted(DVec3::from_vec3(x[i]))
                .ok_or(PhysicsError::OutOfRange)?;
            rope.velocity[i] = v[i];
        }
        rope.tension = tension;
        let cut = (0..count - 1)
            .filter(|k| rope.tension[*k] > desc.break_force)
            .max_by(|a, b| rope.tension[*a].total_cmp(&rope.tension[*b]));
        Ok(cut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::testing::at;
    use crate::world::testing::world;
    use crate::{PhysicsSettings, RigidBodyDesc};

    /// A 2 m rope of 1 kg hanging straight down from a fixed top.
    fn hanging(world: &mut PhysicsWorld, segments: u32, compliance: f32) -> RopeHandle {
        let top = at(0.0, 10.0, 0.0);
        let mut desc = RopeDesc::new(top, at(0.0, 8.0, 0.0), segments);
        desc.stretch_compliance = compliance;
        desc.ends[0] = RopeEnd::Fixed(top);
        world.add_rope(desc).unwrap()
    }

    fn run(world: &mut PhysicsWorld, seconds: f32) {
        let steps = (seconds / world.settings().fixed_timestep).round() as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
    }

    fn length(world: &PhysicsWorld, rope: RopeHandle) -> f64 {
        world
            .rope_points(rope)
            .unwrap()
            .windows(2)
            .map(|pair| pair[1].offset_from(pair[0]).length())
            .sum()
    }

    #[test]
    fn a_hanging_rope_stays_below_its_anchor_and_keeps_its_length() {
        let mut world = world();
        let rope = hanging(&mut world, 20, 0.0);
        run(&mut world, 3.0);
        let points = world.rope_points(rope).unwrap();
        let bottom = points.last().unwrap().local;
        assert!(
            bottom.x.abs() < 0.01 && (bottom.y - 8.0).abs() < 0.03,
            "{bottom:?}"
        );
        assert!((length(&world, rope) - 2.0).abs() < 0.02);
        // the top segment holds the whole rope, 1 kg
        let top = world.rope_tension(rope).unwrap()[0];
        assert!((top - 9.81).abs() < 0.5, "{top}");
    }

    /// A soft 2 m, 1 kg rope's stretch sampled each step for 4 s after 2 s: mean and spread.
    fn soft_stretch(hertz: f32, damping: f32) -> (f64, f64) {
        let mut world = PhysicsWorld::new(PhysicsSettings {
            fixed_timestep: 1.0 / hertz,
            ..PhysicsSettings::default()
        })
        .unwrap();
        let top = at(0.0, 10.0, 0.0);
        let mut desc = RopeDesc::new(top, at(0.0, 8.0, 0.0), 10);
        // 0.01 m a newton a segment, a bungee
        desc.stretch_compliance = 0.01;
        desc.damping = damping;
        desc.ends[0] = RopeEnd::Fixed(top);
        let rope = world.add_rope(desc).unwrap();
        run(&mut world, 2.0);
        let samples: Vec<f64> = (0..(4.0 * hertz) as usize)
            .map(|_| {
                world.step_fixed().unwrap();
                length(&world, rope) - 2.0
            })
            .collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let spread = samples.iter().map(|s| (s - mean).abs()).fold(0.0, f64::max);
        (mean, spread)
    }

    #[test]
    fn compliance_sets_the_stretch_whatever_the_step() {
        // each segment stretches by the weight below it times its compliance
        let expected: f64 = (1..=10).map(|k| 0.01 * 9.81 * 0.1 * f64::from(k)).sum();
        for hertz in [60.0, 240.0] {
            let (mean, _) = soft_stretch(hertz, 5.0);
            assert!(
                (mean / expected - 1.0).abs() < 0.1,
                "{hertz} hz: {mean} {expected}"
            );
        }
    }

    #[test]
    fn damping_calms_a_bouncing_rope() {
        let (_, free) = soft_stretch(60.0, 0.0);
        let (_, damped) = soft_stretch(60.0, 50.0);
        assert!(damped < 0.7 * free, "{free} {damped}");
    }

    #[test]
    fn a_rope_lifts_a_hanging_body_and_lets_it_swing() {
        let mut world = world();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.2))
                    .density(1000.0)
                    .local_position(Vec3::new(3.0, 10.0, 0.0)),
            )
            .unwrap();
        let top = at(0.0, 10.0, 0.0);
        let mut desc = RopeDesc::new(top, at(2.8, 10.0, 0.0), 16);
        desc.ends = [
            RopeEnd::Fixed(top),
            RopeEnd::Body {
                body: ball,
                local: Vec3::new(-0.2, 0.0, 0.0),
            },
        ];
        desc.filter.exclude = Some(ball);
        let rope = world.add_rope(desc).unwrap();
        run(&mut world, 3.0);
        // it swung down on the rope, not fallen free
        let state = world.body_state(ball).unwrap();
        let from_top = state.position.offset_from(top).length();
        assert!(from_top < 3.3, "{from_top}");
        assert!(state.position.to_dvec3().y < 9.0);
        assert!(world.rope_tension(rope).unwrap()[0] > 100.0);
    }

    #[test]
    fn a_rope_drapes_over_a_box_and_does_not_pass_through() {
        let mut world = world();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.0, 5.0, 0.0)),
            )
            .unwrap();
        let rope = world
            .add_rope(RopeDesc::new(at(-1.5, 6.0, 0.0), at(1.5, 6.0, 0.0), 30))
            .unwrap();
        run(&mut world, 2.0);
        let points = world.rope_points(rope).unwrap();
        let middle = points[15].local;
        assert!(middle.y > 5.45 && middle.y < 5.6, "{middle:?}");
        for point in points {
            let p = point.local;
            let inside = p.x.abs() < 0.48 && (p.y - 5.0).abs() < 0.48 && p.z.abs() < 0.48;
            assert!(!inside, "{p:?}");
        }
    }

    #[test]
    fn a_rope_past_its_break_force_parts_in_two() {
        let mut world = world();
        let top = at(0.0, 10.0, 0.0);
        let mut desc = RopeDesc::new(top, at(0.0, 6.0, 0.0), 20);
        desc.ends[0] = RopeEnd::Fixed(top);
        desc.mass_per_meter = 5.0;
        // 20 kg hangs from the top, past 150 n it parts
        desc.break_force = 150.0;
        let rope = world.add_rope(desc).unwrap();
        let mut parted = Vec::new();
        for _ in 0..120 {
            world.step_fixed().unwrap();
            parted.extend_from_slice(world.broken_ropes());
        }
        assert_eq!(parted.len(), 1, "{parted:?}");
        let (cut, _) = parted[0];
        assert_eq!(cut, rope);
        // it gave way at the top, the most loaded segment, and the rest fell
        let points = world.rope_points(rope).unwrap();
        assert!(points.len() >= 2);
        assert!(points[0].local.y < 9.5, "{:?}", points[0]);
    }

    #[test]
    fn ropes_come_back_the_same_after_a_snapshot_and_bad_ones_are_refused() {
        let mut world = world();
        let rope = hanging(&mut world, 12, 0.001);
        run(&mut world, 0.5);
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        run(&mut world, 0.5);
        run(&mut loaded, 0.5);
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
        assert_eq!(
            loaded.rope_points(rope).unwrap(),
            world.rope_points(rope).unwrap()
        );
        let mut bad = RopeDesc::new(at(0.0, 0.0, 0.0), at(1.0, 0.0, 0.0), 0);
        assert!(world.add_rope(bad).is_err());
        bad.segments = 4;
        bad.radius = f32::NAN;
        assert!(world.add_rope(bad).is_err());
        let same = RopeDesc::new(at(0.0, 0.0, 0.0), at(0.0, 0.0, 0.0), 4);
        assert!(world.add_rope(same).is_err());
        world.remove_rope(rope).unwrap();
        assert!(world.rope_points(rope).is_err());
    }
}
