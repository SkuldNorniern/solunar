use algebrix::Vec3;

use crate::body::Bodies;
use crate::{BodyHandle, PhysicsError};

use super::rows::{JointPoint, JointRow};
use super::{JOINT_IMPULSES, JointDesc, JointHandle, JointKind, JointMotor};

const FREE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointState {
    pub(crate) desc: JointDesc,
    pub(crate) impulses: [f32; JOINT_IMPULSES],
    pub(crate) force: Vec3,
    pub(crate) torque: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointSlot {
    pub(crate) generation: u32,
    pub(crate) state: Option<JointState>,
}

#[derive(Debug, Clone, Default)]
pub struct Joints {
    pub(crate) slots: Vec<JointSlot>,
    pub(crate) free: Vec<u32>,
    excluded: Vec<(BodyHandle, BodyHandle)>,
    excluded_stale: bool,
}

impl Joints {
    pub fn len(&self) -> usize {
        self.live().count()
    }

    pub fn is_empty(&self) -> bool {
        self.live().next().is_none()
    }

    pub(crate) fn live(&self) -> impl Iterator<Item = (usize, &JointState)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.state.as_ref().map(|state| (index, state)))
    }

    pub(crate) fn get(&self, handle: JointHandle) -> Result<&JointState, PhysicsError> {
        match self.slots.get(handle.index as usize) {
            Some(JointSlot {
                generation,
                state: Some(state),
            }) if *generation == handle.generation => Ok(state),
            _ => Err(PhysicsError::InvalidHandle),
        }
    }

    pub(crate) fn get_mut(&mut self, handle: JointHandle) -> Result<&mut JointState, PhysicsError> {
        match self.slots.get_mut(handle.index as usize) {
            Some(JointSlot {
                generation,
                state: Some(state),
            }) if *generation == handle.generation => Ok(state),
            _ => Err(PhysicsError::InvalidHandle),
        }
    }

    pub(crate) fn insert(&mut self, desc: JointDesc) -> Result<JointHandle, PhysicsError> {
        let state = JointState {
            desc: JointDesc {
                frame_a: desc.frame_a.normalize(),
                frame_b: desc.frame_b.normalize(),
                ..desc
            },
            impulses: [0.0; JOINT_IMPULSES],
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
        };
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                let index = u32::try_from(self.slots.len())
                    .ok()
                    .filter(|index| *index != FREE)
                    .ok_or(PhysicsError::OutOfRange)?;
                self.slots.push(JointSlot {
                    generation: 0,
                    state: None,
                });
                index
            }
        };
        let slot = &mut self.slots[index as usize];
        slot.state = Some(state);
        self.excluded_stale = true;
        Ok(JointHandle {
            index,
            generation: slot.generation,
        })
    }

    pub(crate) fn remove(&mut self, handle: JointHandle) -> Result<JointState, PhysicsError> {
        let state = *self.get(handle)?;
        let slot = &mut self.slots[handle.index as usize];
        slot.state = None;
        // a slot at the last generation is retired, so old handles can never match again
        if slot.generation < u32::MAX {
            slot.generation += 1;
            self.free.push(handle.index);
        }
        self.excluded_stale = true;
        Ok(state)
    }

    pub(crate) fn mark_loaded(&mut self) {
        self.excluded_stale = true;
    }

    pub(crate) fn handles(&self) -> impl Iterator<Item = JointHandle> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.state.as_ref()?;
            Some(JointHandle {
                index: index as u32,
                generation: slot.generation,
            })
        })
    }

    pub(crate) fn touching(&self, body: BodyHandle) -> impl Iterator<Item = JointHandle> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(move |(index, slot)| {
                let state = slot.state.as_ref()?;
                (state.desc.a == body || state.desc.b == body).then_some(JointHandle {
                    index: index as u32,
                    generation: slot.generation,
                })
            })
    }

    pub(crate) fn refresh_exclusions(&mut self) {
        if !self.excluded_stale {
            return;
        }
        let mut excluded = std::mem::take(&mut self.excluded);
        excluded.clear();
        excluded.extend(self.live().filter_map(|(_, state)| {
            let desc = &state.desc;
            (!desc.collide_connected).then_some((desc.a.min(desc.b), desc.a.max(desc.b)))
        }));
        excluded.sort_unstable();
        excluded.dedup();
        self.excluded = excluded;
        self.excluded_stale = false;
    }

    pub(crate) fn excludes(&self, a: BodyHandle, b: BodyHandle) -> bool {
        !self.excluded.is_empty() && self.excluded.binary_search(&(a.min(b), a.max(b))).is_ok()
    }

    /// Dense pairs linked by a joint, and whether a motor keeps it running.
    pub(crate) fn links<'a>(
        &'a self,
        bodies: &'a Bodies,
    ) -> impl Iterator<Item = (usize, usize, bool)> + 'a {
        self.live().filter_map(move |(_, state)| {
            let a = bodies.dense_index(state.desc.a).ok()?;
            let b = bodies.dense_index(state.desc.b).ok()?;
            let running = match state.desc.kind {
                JointKind::Axes(axes) => axes.iter().any(|axis| {
                    matches!(axis.motor, Some(JointMotor::Velocity { speed, .. }) if speed != 0.0)
                }),
                JointKind::Distance { .. } => false,
            };
            Some((a, b, running))
        })
    }

    /// Runs after the step commits, so a failed step keeps the old impulses.
    pub(crate) fn store(
        &mut self,
        impulses: &[[f32; JOINT_IMPULSES]],
        points: &[JointPoint],
        rows: &[JointRow],
        dt: f32,
    ) {
        for (slot, working) in self.slots.iter_mut().zip(impulses) {
            if let Some(state) = slot.state.as_mut() {
                state.impulses = *working;
            }
        }
        let touched = points
            .iter()
            .map(|point| point.joint)
            .chain(rows.iter().map(|row| row.joint));
        for joint in touched {
            if let Some(state) = self.slots[joint as usize].state.as_mut() {
                state.force = Vec3::ZERO;
                state.torque = Vec3::ZERO;
            }
        }
        for point in points {
            let Some(state) = self.slots[point.joint as usize].state.as_mut() else {
                continue;
            };
            state.force += point.impulse * dt.recip();
        }
        for row in rows {
            let Some(state) = self.slots[row.joint as usize].state.as_mut() else {
                continue;
            };
            if row.linear == Vec3::ZERO {
                state.torque += row.ang_b * (row.impulse / dt);
            } else {
                state.force += row.linear * (row.impulse / dt);
            }
        }
    }
}
