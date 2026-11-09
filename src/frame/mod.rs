use algebrix::{DQuat, DVec3};

use crate::PhysicsError;

mod motion;

pub use crate::orbit::Orbit;
pub use motion::FrameMotion;

pub const MAX_FRAME_DEPTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(pub(crate) u32);

impl FrameId {
    pub const ROOT: Self = Self(0);

    pub fn index(self) -> u32 {
        self.0
    }
}

/// Pose and motion of a frame in its parent. `angular_velocity` is in parent axes. With a
/// `motion` the pose is where it starts at time 0, and the frame moves on from there: a body in it
/// feels that. Without one the velocities only describe it, for transfers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameDesc {
    pub parent: FrameId,
    pub translation: DVec3,
    pub rotation: DQuat,
    pub linear_velocity: DVec3,
    pub angular_velocity: DVec3,
    pub motion: FrameMotion,
    /// How far from its origin the frame holds bodies, meters, zero for no limit. A body
    /// going out past it, with a tenth more so it does not flip back, goes to the parent. One in
    /// the parent coming within nine tenths of it comes in.
    pub reach: f64,
}

impl FrameDesc {
    pub fn new(parent: FrameId) -> Self {
        Self {
            parent,
            translation: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            linear_velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
            motion: FrameMotion::default(),
            reach: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameState {
    pub position: DVec3,
    pub rotation: DQuat,
    pub linear_velocity: DVec3,
    pub angular_velocity: DVec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frames {
    /// Each frame as it is at `time`.
    descs: Vec<FrameDesc>,
    /// As added, at time 0.
    bases: Vec<FrameDesc>,
    depths: Vec<u32>,
    time: f64,
}

impl Default for Frames {
    fn default() -> Self {
        Self::new()
    }
}

impl Frames {
    pub fn new() -> Self {
        Self {
            descs: vec![FrameDesc::new(FrameId::ROOT)],
            bases: vec![FrameDesc::new(FrameId::ROOT)],
            depths: vec![0],
            time: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.descs.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn contains(&self, frame: FrameId) -> bool {
        (frame.0 as usize) < self.descs.len()
    }

    pub fn get(&self, frame: FrameId) -> Result<&FrameDesc, PhysicsError> {
        self.descs
            .get(frame.0 as usize)
            .ok_or(PhysicsError::InvalidFrame)
    }

    /// # Errors
    /// Fails on an unknown parent, a chain past [`MAX_FRAME_DEPTH`], non finite values or a zero rotation.
    pub fn add(&mut self, desc: FrameDesc) -> Result<FrameId, PhysicsError> {
        let rotation = validated_rotation(desc.rotation)?;
        self.restore(FrameDesc { rotation, ..desc })
    }

    /// Same checks as [`Self::add`], but keeps the rotation bits as they are.
    pub(crate) fn restore(&mut self, desc: FrameDesc) -> Result<FrameId, PhysicsError> {
        let parent_depth = *self
            .depths
            .get(desc.parent.0 as usize)
            .ok_or(PhysicsError::InvalidFrame)?;
        let depth = parent_depth + 1;
        if depth as usize > MAX_FRAME_DEPTH {
            return Err(PhysicsError::FrameTooDeep);
        }
        let error = (length_squared(desc.rotation) - 1.0).abs();
        if error.is_nan() || error >= 1e-6 {
            return Err(PhysicsError::NonFinite("frame rotation"));
        }
        if !finite(desc.translation)
            || !finite(desc.linear_velocity)
            || !finite(desc.angular_velocity)
            || !desc.motion.is_valid()
            || !desc.reach.is_finite()
            || desc.reach < 0.0
        {
            return Err(PhysicsError::NonFinite("frame"));
        }
        let id = u32::try_from(self.descs.len()).map_err(|_| PhysicsError::OutOfRange)?;
        self.descs.push(desc.motion.moved(&desc, self.time));
        self.bases.push(desc);
        self.depths.push(depth);
        Ok(FrameId(id))
    }

    /// Frames as added, at time 0.
    pub(crate) fn bases(&self) -> &[FrameDesc] {
        &self.bases
    }

    /// Seconds of world time the frames are at.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Moves every frame with a motion to where it is at `time`.
    pub(crate) fn advance_to(&mut self, time: f64) {
        self.time = time;
        for (desc, base) in self.descs.iter_mut().zip(&self.bases).skip(1) {
            if !base.motion.is_still() {
                *desc = base.motion.moved(base, time);
            }
        }
    }

    /// Whether `frame` or one it sits in moves.
    pub(crate) fn moves(&self, frame: FrameId) -> bool {
        let mut current = frame;
        while current != FrameId::ROOT {
            let base = &self.bases[current.0 as usize];
            if !base.motion.is_still() {
                return true;
            }
            current = base.parent;
        }
        false
    }

    /// Where `frame`'s origin is in the root and how it moves there, at `time`.
    pub(crate) fn origin_at(&self, frame: FrameId, time: f64) -> FrameState {
        let mut state = FrameState {
            position: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            linear_velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        };
        let mut current = frame;
        while current != FrameId::ROOT {
            let base = &self.bases[current.0 as usize];
            state = to_parent(&base.motion.moved(base, time), state);
            current = base.parent;
        }
        state
    }

    pub fn root_rotation(&self, frame: FrameId) -> Result<DQuat, PhysicsError> {
        let mut rotation = DQuat::IDENTITY;
        let mut current = frame;
        while current != FrameId::ROOT {
            let desc = self.get(current)?;
            rotation = desc.rotation * rotation;
            current = desc.parent;
        }
        self.get(frame)?;
        Ok(rotation)
    }

    /// # Errors
    /// Fails on an unknown frame.
    pub fn to_root(&self, frame: FrameId, state: FrameState) -> Result<FrameState, PhysicsError> {
        let mut state = state;
        let mut current = frame;
        self.get(frame)?;
        while current != FrameId::ROOT {
            let desc = self.get(current)?;
            state = to_parent(desc, state);
            current = desc.parent;
        }
        Ok(state)
    }

    /// # Errors
    /// Fails on an unknown frame.
    pub fn from_root(&self, frame: FrameId, state: FrameState) -> Result<FrameState, PhysicsError> {
        let mut path = [FrameId::ROOT; MAX_FRAME_DEPTH];
        let mut len = 0;
        let mut current = frame;
        self.get(frame)?;
        while current != FrameId::ROOT {
            path[len] = current;
            len += 1;
            current = self.get(current)?.parent;
        }
        let mut state = state;
        for id in path[..len].iter().rev() {
            state = from_parent(self.get(*id)?, state);
        }
        Ok(state)
    }

    /// # Errors
    /// Fails on an unknown frame.
    pub fn transfer(
        &self,
        from: FrameId,
        to: FrameId,
        state: FrameState,
    ) -> Result<FrameState, PhysicsError> {
        if from == to {
            self.get(from)?;
            return Ok(state);
        }
        self.from_root(to, self.to_root(from, state)?)
    }
}

fn to_parent(frame: &FrameDesc, state: FrameState) -> FrameState {
    let arm = frame.rotation.mul_vec3(state.position);
    FrameState {
        position: frame.translation + arm,
        rotation: (frame.rotation * state.rotation).normalize(),
        linear_velocity: frame.linear_velocity
            + frame.angular_velocity.cross(arm)
            + frame.rotation.mul_vec3(state.linear_velocity),
        angular_velocity: frame.angular_velocity + frame.rotation.mul_vec3(state.angular_velocity),
    }
}

fn from_parent(frame: &FrameDesc, state: FrameState) -> FrameState {
    let inverse = frame.rotation.conjugate();
    let arm = state.position - frame.translation;
    FrameState {
        position: inverse.mul_vec3(arm),
        rotation: (inverse * state.rotation).normalize(),
        linear_velocity: inverse.mul_vec3(
            state.linear_velocity - frame.linear_velocity - frame.angular_velocity.cross(arm),
        ),
        angular_velocity: inverse.mul_vec3(state.angular_velocity - frame.angular_velocity),
    }
}

fn finite(v: DVec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn validated_rotation(rotation: DQuat) -> Result<DQuat, PhysicsError> {
    let squared = length_squared(rotation);
    if !squared.is_finite() || squared < 1e-12 {
        return Err(PhysicsError::NonFinite("frame rotation"));
    }
    Ok(rotation.normalize())
}

fn length_squared(rotation: DQuat) -> f64 {
    rotation.x * rotation.x
        + rotation.y * rotation.y
        + rotation.z * rotation.z
        + rotation.w * rotation.w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec3, b: DVec3) {
        assert!((a - b).length() < 1e-9, "{a:?} != {b:?}");
    }

    fn state(position: DVec3, velocity: DVec3) -> FrameState {
        FrameState {
            position,
            rotation: DQuat::IDENTITY,
            linear_velocity: velocity,
            angular_velocity: DVec3::ZERO,
        }
    }

    #[test]
    fn body_at_rest_in_spinning_frame_moves_in_root() {
        let mut frames = Frames::new();
        let spinning = frames
            .add(FrameDesc {
                angular_velocity: DVec3::new(0.0, 2.0, 0.0),
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let root = frames
            .to_root(spinning, state(DVec3::new(3.0, 0.0, 0.0), DVec3::ZERO))
            .unwrap();
        close(root.linear_velocity, DVec3::new(0.0, 0.0, -6.0));
    }

    #[test]
    fn translated_rotated_frame_moves_position() {
        let mut frames = Frames::new();
        let frame = frames
            .add(FrameDesc {
                translation: DVec3::new(10.0, 0.0, 0.0),
                rotation: DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2),
                linear_velocity: DVec3::new(0.0, 1.0, 0.0),
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let root = frames.to_root(frame, state(DVec3::X, DVec3::X)).unwrap();
        close(root.position, DVec3::new(10.0, 1.0, 0.0));
        close(root.linear_velocity, DVec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn transfer_round_trips_through_nested_frames() {
        let mut frames = Frames::new();
        let planet = frames
            .add(FrameDesc {
                translation: DVec3::new(1.0e9, -2.0e8, 3.0e7),
                rotation: DQuat::from_axis_angle(DVec3::Y, 0.7),
                linear_velocity: DVec3::new(30_000.0, 0.0, 0.0),
                angular_velocity: DVec3::new(0.0, 7e-5, 0.0),
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let ship = frames
            .add(FrameDesc {
                translation: DVec3::new(6.4e6, 0.0, 0.0),
                rotation: DQuat::from_axis_angle(DVec3::X, -1.2),
                angular_velocity: DVec3::new(0.1, 0.0, 0.0),
                ..FrameDesc::new(planet)
            })
            .unwrap();
        let start = FrameState {
            position: DVec3::new(1.0, 2.0, 3.0),
            rotation: DQuat::from_axis_angle(DVec3::Z, 0.3),
            linear_velocity: DVec3::new(-1.0, 0.5, 2.0),
            angular_velocity: DVec3::new(0.2, -0.1, 0.4),
        };
        let moved = frames.transfer(ship, planet, start).unwrap();
        let back = frames.transfer(planet, ship, moved).unwrap();
        assert!((back.position - start.position).length() < 1e-6);
        assert!((back.linear_velocity - start.linear_velocity).length() < 1e-6);
        close(back.angular_velocity, start.angular_velocity);
    }

    #[test]
    fn unknown_and_bad_frames_fail() {
        let mut frames = Frames::new();
        let missing = FrameId(9);
        assert_eq!(
            frames.add(FrameDesc::new(missing)),
            Err(PhysicsError::InvalidFrame)
        );
        assert_eq!(
            frames.to_root(missing, state(DVec3::ZERO, DVec3::ZERO)),
            Err(PhysicsError::InvalidFrame)
        );
        let bad = FrameDesc {
            translation: DVec3::new(f64::NAN, 0.0, 0.0),
            ..FrameDesc::new(FrameId::ROOT)
        };
        assert!(frames.add(bad).is_err());
        let zero = FrameDesc {
            rotation: DQuat::new(0.0, 0.0, 0.0, 0.0),
            ..FrameDesc::new(FrameId::ROOT)
        };
        assert!(frames.add(zero).is_err());
    }

    #[test]
    fn depth_is_capped() {
        let mut frames = Frames::new();
        let mut parent = FrameId::ROOT;
        for _ in 0..MAX_FRAME_DEPTH {
            parent = frames.add(FrameDesc::new(parent)).unwrap();
        }
        assert_eq!(
            frames.add(FrameDesc::new(parent)),
            Err(PhysicsError::FrameTooDeep)
        );
        let root = frames
            .to_root(parent, state(DVec3::X, DVec3::ZERO))
            .unwrap();
        close(root.position, DVec3::X);
        assert!(frames.from_root(parent, root).is_ok());
    }
}
