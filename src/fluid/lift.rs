use algebrix::Vec3;

use super::FluidWorld;
use crate::body::BodyHandle;
use crate::substep::SubstepPose;

/// Most probes one body takes, a big body gets them coarser.
pub(super) const MAX_PROBES: usize = 4096;
/// Drag coefficient of a body moving through the water, about a box's.
const DRAG: f32 = 1.0;
/// How far around a body the water is read, in kernel radii.
const READ_REACH: f32 = 3.0;

/// What the water does to one body over a step, and the particles it was read from, which give
/// it back.
#[derive(Debug, Clone)]
pub(super) struct Lift {
    pub(super) body: BodyHandle,
    pub(super) impulse: Vec3,
    pub(super) angular: Vec3,
    near: Vec<u32>,
}

impl FluidWorld {
    /// The water on each coupled body over `dt`, read as one smooth field around it: lift from
    /// how deep its probes are under the water's level there, drag from the water's flow past
    /// them. Single particles' noise averages out over the many read. Bodies `pose` does not know
    /// are skipped.
    pub(super) fn lifts(
        &self,
        dt: f32,
        pose: impl Fn(BodyHandle) -> Option<SubstepPose>,
    ) -> Vec<Lift> {
        let settings = &self.settings;
        let gravity = settings.gravity.length();
        let up = if gravity > 0.0 {
            -settings.gravity / gravity
        } else {
            Vec3::Y
        };
        let reach = READ_REACH * settings.radius();
        let column = settings.radius();
        let mut lifts = Vec::new();
        for known in &self.coupled {
            let Some(state) = pose(known.body()) else {
                continue;
            };
            let center = state.position.to_dvec3().as_vec3();
            let probes: Vec<(Vec3, Vec3)> = known
                .probes
                .iter()
                .map(|local| {
                    let arm = state.rotation.rotate_vec3(*local);
                    let velocity = state.linear_velocity + state.angular_velocity.cross(arm);
                    (arm, velocity)
                })
                .collect();
            let (low, high) = probes.iter().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(low, high), (arm, _)| (low.min(center + *arm), high.max(center + *arm)),
            );
            let (low, high) = (low - Vec3::splat(reach), high + Vec3::splat(reach));
            let near: Vec<u32> = self
                .positions
                .iter()
                .enumerate()
                .filter(|(_, at)| at.clamp(low, high) == **at)
                .map(|(index, _)| index as u32)
                .collect();
            if near.is_empty() {
                continue;
            }
            // the level is the middle of the columns' tops, so a splash or a hollow does not move it
            let mut tops: Vec<((i32, i32), f32)> = near
                .iter()
                .map(|&index| {
                    let at = self.positions[index as usize];
                    let key = (
                        (at.x / column).floor() as i32,
                        (at.z / column).floor() as i32,
                    );
                    (key, at.dot(up))
                })
                .collect();
            tops.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
            tops.dedup_by_key(|item| item.0);
            let mut heights: Vec<f32> = tops.iter().map(|item| item.1).collect();
            heights.sort_unstable_by(f32::total_cmp);
            // the mean of the middle half: a median alone jumped a particle at a time
            let quarter = heights.len() / 4;
            let middle_half = &heights[quarter..heights.len() - quarter];
            let level =
                middle_half.iter().sum::<f32>() / middle_half.len() as f32 + settings.spacing / 2.0;
            let (mut flow, mut wet) = (Vec3::ZERO, 0_u32);
            for &index in &near {
                if self.positions[index as usize].dot(up) <= level {
                    flow += self.velocities[index as usize];
                    wet += 1;
                }
            }
            let flow = flow / wet.max(1) as f32;
            // how much of each probe's cell is under, so lift grows smoothly as a body sinks
            // rather than a layer of probes at a time
            let spacing = known.probe_spacing;
            let under: Vec<(Vec3, Vec3, f32)> = probes
                .iter()
                .filter_map(|(arm, velocity)| {
                    let depth = level - (center + *arm).dot(up);
                    let share = (depth / spacing + 0.5).clamp(0.0, 1.0);
                    (share > 0.0).then_some((*arm, *velocity, share))
                })
                .collect();
            let wetted: f32 = under.iter().map(|item| item.2).sum();
            if wetted <= 0.0 {
                continue;
            }
            let volume = wetted * spacing.powi(3);
            let density = settings.rest_density;
            let middle = under
                .iter()
                .map(|(arm, _, share)| *arm * *share)
                .sum::<Vec3>()
                / wetted;
            let (mut force, mut torque) = (Vec3::ZERO, Vec3::ZERO);
            let buoyancy = up * (density * gravity * volume);
            force += buoyancy;
            torque += middle.cross(buoyancy);
            // drag on each probe's share of the wetted area, so a turning body is slowed too
            let area = volume.powf(2.0 / 3.0) / wetted;
            for (arm, velocity, share) in &under {
                let past = flow - *velocity;
                let drag = past * (0.5 * DRAG * density * area * share * past.length());
                force += drag;
                torque += arm.cross(drag);
            }
            // a body bobbing or rocking sends waves off, which the particles get from it as a wall
            // but give nothing back for, so this stands in: damping on its waterplane at their
            // speed, on each probe there, so rocking is calmed as bobbing is
            let waterline: Vec<&(Vec3, Vec3, f32)> =
                under.iter().filter(|item| item.2 < 1.0).collect();
            let plane = waterline.len() as f32 * spacing * spacing;
            let waves = density * plane * (gravity * plane.sqrt()).sqrt();
            let share = waves / waterline.len().max(1) as f32;
            for (arm, velocity, _) in waterline {
                let calm = up * (share * (flow - *velocity).dot(up));
                force += calm;
                torque += arm.cross(calm);
            }
            lifts.push(Lift {
                body: known.body(),
                impulse: force * dt,
                angular: torque * dt,
                near,
            });
        }
        lifts
    }

    /// The water read for `lift` loses what the body got, shared by mass, so water and bodies
    /// keep their momentum.
    pub(super) fn give_back(&mut self, lift: &Lift) {
        let change = lift.impulse / (self.mass * lift.near.len() as f32);
        for &index in &lift.near {
            self.velocities[index as usize] -= change;
        }
    }
}
