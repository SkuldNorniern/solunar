use algebrix::Vec3;

use crate::{BodyHandle, FrameId, PhysicsError, RigidBodyDesc};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Command {
    CreateBody(RigidBodyDesc),
    DestroyBody(BodyHandle),
    ApplyForce(BodyHandle, Vec3),
    ApplyTorque(BodyHandle, Vec3),
    ApplyImpulse(BodyHandle, Vec3),
    ApplyImpulseAt(BodyHandle, Vec3, Vec3),
    SetVelocity(BodyHandle, Vec3, Vec3),
    TransferToFrame(BodyHandle, FrameId),
    Wake(BodyHandle),
}

/// Commands recorded between steps, applied in order at a safe boundary.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CommandBuffer {
    pub(crate) commands: Vec<Command>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, command: Command) {
        self.commands.push(command);
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

/// `created` follows create command order. `rejected` holds the command index and the reason.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ApplyReport {
    pub created: Vec<BodyHandle>,
    pub rejected: Vec<(usize, PhysicsError)>,
}
