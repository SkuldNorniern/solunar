use algebrix::{DQuat, DVec3};

use super::FrameDesc;
use crate::orbit::Orbit;

/// How a frame moves in its parent over the world's time, worked out from the time each step
/// rather than stepped, so it never drifts. None of it set, the frame stays where it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMotion {
    /// Turning about this axis in parent axes, at its length in radians a second, from the
    /// desc's rotation at time 0.
    pub spin: DVec3,
    /// Going round the parent's origin, plus the desc's translation.
    pub orbit: Option<Orbit>,
}

impl Default for FrameMotion {
    fn default() -> Self {
        Self {
            spin: DVec3::ZERO,
            orbit: None,
        }
    }
}

impl FrameMotion {
    pub(crate) fn is_valid(&self) -> bool {
        let spin = self.spin;
        spin.x.is_finite()
            && spin.y.is_finite()
            && spin.z.is_finite()
            && self.orbit.is_none_or(|orbit| orbit.is_valid())
    }

    pub(crate) fn is_still(&self) -> bool {
        self.spin == DVec3::ZERO && self.orbit.is_none()
    }

    /// `base` at `time`.
    pub(crate) fn moved(&self, base: &FrameDesc, time: f64) -> FrameDesc {
        let mut desc = *base;
        let rate = self.spin.length();
        if rate > 0.0 {
            let turn = DQuat::from_axis_angle(self.spin / rate, rate * time);
            desc.rotation = (turn * base.rotation).normalize();
            desc.angular_velocity = self.spin;
        }
        if let Some(orbit) = self.orbit {
            let (position, velocity) = orbit.state_at(time);
            desc.translation = base.translation + position;
            desc.linear_velocity = base.linear_velocity + velocity;
        }
        desc
    }
}
