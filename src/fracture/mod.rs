mod voronoi;

use std::collections::HashMap;

use algebrix::{DVec3, Quat, Vec3};

pub use voronoi::{
    FracturePattern, MAX_CELLS, PatternBond, PatternPiece, voronoi_box, voronoi_box_stretched,
};

use crate::body::{Bodies, BodyHandle, BodyKind, RigidBodyDesc};
use crate::compound::{CompoundChild, CompoundInfo};
use crate::error::PhysicsError;
use crate::material::Material;
use crate::position::GridPosition;
use crate::shape::Shape;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::world::PhysicsWorld;

/// Loads under this share of a bond's strength leave no mark.
const FATIGUE_FROM: f32 = 0.25;
/// Damage a load of a bond's full strength would leave, were it not already enough to break it.
const FATIGUE_RATE: f32 = 0.2;

/// Per piece, the strongest push on it this step and from which body.
type Hits = [Option<(f32, BodyHandle)>];
/// Force on each piece of one breakable, and the body that pushed hardest on it.
type PieceLoads = (Vec<f32>, Vec<Option<(f32, BodyHandle)>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BreakableHandle {
    index: u32,
    generation: u32,
}

/// Something that breaks along a pattern's faces. Whole it is one body made of all its pieces.
/// A piece pushed harder than the faces holding it can take comes loose, and each part still
/// holding together becomes a body of its own, the mass and the motion it had kept.
#[derive(Debug, Clone, PartialEq)]
pub struct BreakableDesc {
    pub pattern: FracturePattern,
    pub position: GridPosition,
    pub rotation: Quat,
    /// Fixed stands where it is, for walls and pillars. Pieces that come loose fall, the ones
    /// in `anchored` and what holds on to them stay.
    pub fixed: bool,
    /// Per piece, for a fixed one. Empty means the pieces on the pattern's bottom face.
    pub anchored: Vec<bool>,
    pub material: Material,
    /// Pascals a face takes before it tears. The load is the step's push over the face, so
    /// these are game scale, see [`FractureMaterial`].
    pub strength: f32,
    /// Damage a face that breaks hands each face around its two pieces, 0 to 1. High for glass,
    /// whose cracks run on and shatter it, low for wood.
    pub crack_spread: f32,
    pub grain: Option<Grain>,
}

/// A direction a material is stronger across than along, as wood's fibres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grain {
    /// In the pattern's axes.
    pub axis: Vec3,
    /// Strength of a face lying along the grain, as a share of one across it, 0 to 1.
    pub split: f32,
}

impl BreakableDesc {
    pub fn new(pattern: FracturePattern, position: GridPosition) -> Self {
        Self {
            pattern,
            position,
            rotation: Quat::IDENTITY,
            fixed: false,
            anchored: Vec::new(),
            material: Material::default(),
            strength: 2.0e6,
            crack_spread: 0.0,
            grain: None,
        }
    }

    /// A box of `half` extents of `kind`, cut into pieces as small as the material breaks. Wood
    /// takes its grain along the box's longest side.
    ///
    /// # Errors
    /// Fails on a bad box.
    pub fn of(
        kind: &FractureMaterial,
        half: Vec3,
        position: GridPosition,
        seed: u64,
    ) -> Result<Self, PhysicsError> {
        let volume = 8.0 * half.x * half.y * half.z;
        let cells = ((volume * kind.pieces_per_cubic_meter).round() as usize)
            .clamp(kind.fewest_pieces, MAX_CELLS);
        let longest = if half.x >= half.y && half.x >= half.z {
            Vec3::X
        } else if half.y >= half.z {
            Vec3::Y
        } else {
            Vec3::Z
        };
        let stretch = Vec3::ONE + longest * (kind.grain_stretch - 1.0);
        let pattern = voronoi_box_stretched(half, cells, seed, stretch)?;
        Ok(Self {
            material: Material {
                density: kind.density,
                friction: kind.friction,
                restitution: kind.restitution,
                ..Material::default()
            },
            strength: kind.strength,
            crack_spread: kind.crack_spread,
            grain: (kind.grain_split < 1.0).then_some(Grain {
                axis: longest,
                split: kind.grain_split,
            }),
            ..Self::new(pattern, position)
        })
    }
}

/// How a material breaks, game scale: strengths are tuned so a 190 kg steel ball at 32 m/s
/// breaks a 24 cm concrete wall, not measured tensile strengths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractureMaterial {
    pub density: f32,
    pub strength: f32,
    pub crack_spread: f32,
    /// Strength along the grain as a share of across it, 1 for none.
    pub grain_split: f32,
    /// How drawn out pieces are along the grain, 1 for none.
    pub grain_stretch: f32,
    pub pieces_per_cubic_meter: f32,
    pub fewest_pieces: usize,
    pub friction: f32,
    pub restitution: f32,
}

impl FractureMaterial {
    /// Shatters into many shards, cracks run on.
    pub const GLASS: Self = Self {
        density: 2500.0,
        strength: 1.5e5,
        crack_spread: 0.6,
        grain_split: 1.0,
        grain_stretch: 1.0,
        pieces_per_cubic_meter: 4000.0,
        fewest_pieces: 24,
        friction: 0.4,
        restitution: 0.1,
    };
    pub const CERAMIC: Self = Self {
        density: 2300.0,
        strength: 1.2e5,
        crack_spread: 0.45,
        grain_split: 1.0,
        grain_stretch: 1.0,
        pieces_per_cubic_meter: 1500.0,
        fewest_pieces: 12,
        friction: 0.5,
        restitution: 0.1,
    };
    pub const ICE: Self = Self {
        density: 917.0,
        strength: 8.0e4,
        crack_spread: 0.35,
        grain_split: 1.0,
        grain_stretch: 1.0,
        pieces_per_cubic_meter: 600.0,
        fewest_pieces: 12,
        friction: 0.05,
        restitution: 0.1,
    };
    /// Breaks in chunks.
    pub const CONCRETE: Self = Self {
        density: 2400.0,
        strength: 2.0e5,
        crack_spread: 0.2,
        grain_split: 1.0,
        grain_stretch: 1.0,
        pieces_per_cubic_meter: 60.0,
        fewest_pieces: 12,
        friction: 0.7,
        restitution: 0.05,
    };
    pub const STONE: Self = Self {
        density: 2600.0,
        strength: 4.0e5,
        crack_spread: 0.15,
        grain_split: 1.0,
        grain_stretch: 1.0,
        pieces_per_cubic_meter: 40.0,
        fewest_pieces: 8,
        friction: 0.7,
        restitution: 0.05,
    };
    /// Splits along its grain into long splinters, cracks stay local.
    pub const WOOD: Self = Self {
        density: 600.0,
        strength: 3.0e5,
        crack_spread: 0.05,
        grain_split: 0.15,
        grain_stretch: 5.0,
        pieces_per_cubic_meter: 300.0,
        fewest_pieces: 10,
        friction: 0.6,
        restitution: 0.15,
    };
}

/// A part of a breakable that came apart in the last step and is a body now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractureEvent {
    pub breakable: BreakableHandle,
    pub body: BodyHandle,
    pub pieces: u32,
    pub mass: f32,
}

#[derive(Debug, Clone)]
struct Piece {
    info: CompoundInfo,
    volume: f32,
    /// Where its middle is in the pattern.
    center: Vec3,
    anchored: bool,
}

#[derive(Debug, Clone, Copy)]
struct Bond {
    a: u32,
    b: u32,
    /// Newtons.
    strength: f32,
    damage: f32,
    broken: bool,
}

#[derive(Debug, Clone)]
struct Group {
    body: BodyHandle,
    /// Pieces in compound part order.
    pieces: Vec<u32>,
    /// How the body sits in the pattern, to get the pattern's place back from the body's.
    info: CompoundInfo,
}

#[derive(Debug, Clone)]
pub(crate) struct Breakable {
    pieces: Vec<Piece>,
    bonds: Vec<Bond>,
    groups: Vec<Group>,
    material: Material,
    spread: f32,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Breakables {
    slots: Vec<Option<Breakable>>,
    generations: Vec<u32>,
    events: Vec<FractureEvent>,
    /// Velocities as the step started, while any breakable is there.
    pub(crate) before: Vec<Vec3>,
}

impl Breakables {
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// Whether `body` is a group of some breakable's pieces.
    pub(crate) fn holds(&self, body: BodyHandle) -> bool {
        self.slots
            .iter()
            .flatten()
            .any(|breakable| breakable.groups.iter().any(|group| group.body == body))
    }

    pub(crate) fn save(&self, out: &mut Writer) {
        out.len(self.slots.len());
        for (slot, generation) in self.slots.iter().zip(&self.generations) {
            out.u32(*generation);
            match slot {
                None => out.u8(0),
                Some(breakable) => {
                    out.u8(1);
                    breakable.save(out);
                }
            }
        }
    }

    /// Shapes are checked against the loaded world after, see [`Self::shapes`].
    pub(crate) fn load(
        reader: &mut Reader<'_>,
        bodies: &Bodies,
        max: usize,
    ) -> Result<Self, SnapshotError> {
        let count = reader.count(5, max)?;
        let mut breakables = Self::default();
        for _ in 0..count {
            breakables.generations.push(reader.u32()?);
            breakables.slots.push(match reader.u8()? {
                0 => None,
                1 => Some(Breakable::load(reader, bodies, max)?),
                _ => return Err(SnapshotError::Invalid("breakable flag")),
            });
        }
        Ok(breakables)
    }

    /// Every shape the breakables name.
    pub(crate) fn shapes(&self) -> impl Iterator<Item = &Shape> + '_ {
        self.slots.iter().flatten().flat_map(|breakable| {
            let pieces = breakable.pieces.iter().map(|piece| &piece.info.shape);
            let groups = breakable.groups.iter().map(|group| &group.info.shape);
            pieces.chain(groups)
        })
    }
}

/// A face lying along the grain holds `split` of a face across it, between them by the cosine.
fn grain_factor(grain: Option<Grain>, normal: Vec3) -> f32 {
    let Some(grain) = grain else {
        return 1.0;
    };
    let across = normal.dot(grain.axis.normalize()).abs();
    grain.split + (1.0 - grain.split) * across
}

fn save_info(out: &mut Writer, info: &CompoundInfo) {
    out.shape(&info.shape);
    out.vec3(info.center);
    out.quat(info.axes);
}

fn load_info(reader: &mut Reader<'_>) -> Result<CompoundInfo, SnapshotError> {
    let info = CompoundInfo {
        shape: reader.shape()?,
        center: reader.vec3()?,
        axes: reader.unit_quat()?,
    };
    if !info.center.is_finite() {
        return Err(SnapshotError::Invalid("breakable placement"));
    }
    Ok(info)
}

impl Breakable {
    fn save(&self, out: &mut Writer) {
        out.material(&self.material);
        out.f32(self.spread);
        out.len(self.pieces.len());
        for piece in &self.pieces {
            save_info(out, &piece.info);
            out.f32(piece.volume);
            out.vec3(piece.center);
            out.u8(u8::from(piece.anchored));
        }
        out.len(self.bonds.len());
        for bond in &self.bonds {
            out.u32(bond.a);
            out.u32(bond.b);
            out.f32(bond.strength);
            out.f32(bond.damage);
            out.u8(u8::from(bond.broken));
        }
        out.len(self.groups.len());
        for group in &self.groups {
            out.body(group.body);
            save_info(out, &group.info);
            out.len(group.pieces.len());
            for piece in &group.pieces {
                out.u32(*piece);
            }
        }
    }

    fn load(reader: &mut Reader<'_>, bodies: &Bodies, max: usize) -> Result<Self, SnapshotError> {
        let invalid = || SnapshotError::Invalid("breakable");
        let flag = |value: u8| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(SnapshotError::Invalid("breakable flag")),
        };
        let material = reader.material()?;
        let spread = if reader.version >= 7 {
            reader.f32()?
        } else {
            0.0
        };
        if !(0.0..=1.0).contains(&spread) {
            return Err(SnapshotError::Invalid("breakable spread"));
        }
        let count = reader.count(46, MAX_CELLS)?;
        let mut pieces = Vec::with_capacity(count);
        for _ in 0..count {
            let info = load_info(reader)?;
            let (volume, center) = (reader.f32()?, reader.vec3()?);
            let anchored = flag(reader.u8()?)?;
            if !matches!(info.shape, Shape::Convex(_))
                || !volume.is_finite()
                || volume <= 0.0
                || !center.is_finite()
            {
                return Err(invalid());
            }
            pieces.push(Piece {
                info,
                volume,
                center,
                anchored,
            });
        }
        let bonds_count = reader.count(17, MAX_CELLS * MAX_CELLS)?;
        let mut bonds = Vec::with_capacity(bonds_count);
        for _ in 0..bonds_count {
            let (a, b) = (reader.u32()?, reader.u32()?);
            let (strength, damage) = (reader.f32()?, reader.f32()?);
            let broken = flag(reader.u8()?)?;
            if a >= b
                || b as usize >= count
                || !strength.is_finite()
                || strength <= 0.0
                || !damage.is_finite()
                || damage < 0.0
            {
                return Err(invalid());
            }
            bonds.push(Bond {
                a,
                b,
                strength,
                damage,
                broken,
            });
        }
        let groups_count = reader.count(48, max)?;
        let mut groups = Vec::with_capacity(groups_count);
        let mut placed = vec![false; count];
        for _ in 0..groups_count {
            let body = reader.handle()?;
            let info = load_info(reader)?;
            let members = reader.count(4, count)?;
            let mut group_pieces = Vec::with_capacity(members);
            for _ in 0..members {
                let piece = reader.u32()? as usize;
                if piece >= count || placed[piece] {
                    return Err(invalid());
                }
                placed[piece] = true;
                group_pieces.push(piece as u32);
            }
            if group_pieces.is_empty() || bodies.dense_index(body).is_err() {
                return Err(invalid());
            }
            groups.push(Group {
                body,
                pieces: group_pieces,
                info,
            });
        }
        if placed.iter().any(|in_group| !in_group) {
            return Err(invalid());
        }
        Ok(Self {
            pieces,
            bonds,
            groups,
            material,
            spread,
        })
    }
}

/// The pattern's place in the world from a group body's pose.
fn pattern_pose(
    info: &CompoundInfo,
    position: GridPosition,
    rotation: Quat,
) -> Option<(GridPosition, Quat)> {
    let turn = (rotation * info.axes.conjugate()).normalize();
    let back = turn.rotate_vec3(info.center);
    Some((position.shifted(-DVec3::from_vec3(back))?, turn))
}

impl PhysicsWorld {
    /// # Errors
    /// Fails on an empty pattern, a bad piece, a strength that is not finite and positive, or
    /// an `anchored` list of the wrong length.
    pub fn add_breakable(&mut self, desc: BreakableDesc) -> Result<BreakableHandle, PhysicsError> {
        let count = desc.pattern.pieces.len();
        let grain_ok = desc.grain.is_none_or(|grain| {
            grain.axis.is_finite()
                && grain.axis.length() > 1e-6
                && grain.split.is_finite()
                && grain.split > 0.0
                && grain.split <= 1.0
        });
        if count == 0
            || !desc.strength.is_finite()
            || desc.strength <= 0.0
            || !(0.0..=1.0).contains(&desc.crack_spread)
            || !grain_ok
            || !(desc.anchored.is_empty() || desc.anchored.len() == count)
        {
            return Err(PhysicsError::InvalidShape);
        }
        let bottom = -desc.pattern.half.y;
        let mut pieces = Vec::with_capacity(count);
        for (index, piece) in desc.pattern.pieces.iter().enumerate() {
            // from its own middle, so the hull's tolerance goes by the piece and not the box
            let middle = piece.points.iter().fold(Vec3::ZERO, |sum, p| sum + *p)
                / piece.points.len().max(1) as f32;
            let local: Vec<Vec3> = piece.points.iter().map(|p| *p - middle).collect();
            let mut info = self.add_convex(&local)?;
            info.center += middle;
            let anchored = desc.fixed
                && match desc.anchored.get(index) {
                    Some(flag) => *flag,
                    None => piece
                        .points
                        .iter()
                        .any(|p| p.y <= bottom + 1e-4 * desc.pattern.half.y.abs()),
                };
            pieces.push(Piece {
                info,
                volume: piece.volume,
                center: piece.center,
                anchored,
            });
        }
        let bonds = desc
            .pattern
            .bonds
            .iter()
            .map(|bond| Bond {
                a: bond.a,
                b: bond.b,
                strength: bond.area * desc.strength * grain_factor(desc.grain, bond.normal),
                damage: 0.0,
                broken: false,
            })
            .collect();
        let mut breakable = Breakable {
            pieces,
            bonds,
            groups: Vec::new(),
            material: desc.material,
            spread: desc.crack_spread,
        };
        let all: Vec<u32> = (0..count as u32).collect();
        let kind = if desc.fixed {
            BodyKind::Static
        } else {
            BodyKind::Dynamic
        };
        let group = self.spawn_group(
            &breakable,
            all,
            (desc.position, desc.rotation),
            kind,
            (Vec3::ZERO, Vec3::ZERO),
        )?;
        breakable.groups.push(group);
        let slots = &mut self.breakables;
        let index = match slots.slots.iter().position(Option::is_none) {
            Some(free) => {
                slots.slots[free] = Some(breakable);
                free
            }
            None => {
                slots.slots.push(Some(breakable));
                slots.generations.push(0);
                slots.slots.len() - 1
            }
        };
        Ok(BreakableHandle {
            index: index as u32,
            generation: self.breakables.generations[index],
        })
    }

    /// The bodies a breakable is now, the whole first, then its parts in the order they broke
    /// off. Empty for an unknown handle.
    pub fn breakable_bodies(&self, handle: BreakableHandle) -> Vec<BodyHandle> {
        self.breakable(handle)
            .map(|b| b.groups.iter().map(|group| group.body).collect())
            .unwrap_or_default()
    }

    /// Parts that broke off in the last step.
    pub fn fracture_events(&self) -> &[FractureEvent] {
        &self.breakables.events
    }

    fn breakable(&self, handle: BreakableHandle) -> Option<&Breakable> {
        let index = handle.index as usize;
        if self.breakables.generations.get(index) != Some(&handle.generation) {
            return None;
        }
        self.breakables.slots.get(index)?.as_ref()
    }

    fn spawn_group(
        &mut self,
        breakable: &Breakable,
        pieces: Vec<u32>,
        (at, turn): (GridPosition, Quat),
        kind: BodyKind,
        (linear, angular): (Vec3, Vec3),
    ) -> Result<Group, PhysicsError> {
        let info = if let [single] = pieces[..] {
            breakable.pieces[single as usize].info
        } else {
            let children: Vec<CompoundChild> = pieces
                .iter()
                .map(|piece| {
                    let info = breakable.pieces[*piece as usize].info;
                    CompoundChild {
                        rotation: info.axes,
                        ..CompoundChild::new(info.shape, info.center)
                    }
                })
                .collect();
            self.add_owned_compound(&children)?
        };
        let center = turn.rotate_vec3(info.center);
        let position = at
            .shifted(DVec3::from_vec3(center))
            .ok_or(PhysicsError::OutOfRange)?;
        let rotation = (turn * info.axes).normalize();
        let mut desc = RigidBodyDesc::new(kind, info.shape)
            .material(breakable.material)
            .position(position)
            .rotation(rotation);
        if kind == BodyKind::Dynamic {
            desc = desc.linear_velocity(linear).angular_velocity(angular);
        }
        let body = self.spawn(desc)?;
        Ok(Group { body, pieces, info })
    }

    /// Velocities before the step, for parts that break off to take their share of a hit.
    pub(crate) fn remember_velocities(&mut self) {
        if self.breakables.is_empty() {
            self.breakables.before.clear();
            return;
        }
        self.breakables.before.clear();
        self.breakables
            .before
            .extend_from_slice(&self.bodies.linear);
    }

    /// Loads each piece's faces with what pushed on it this step, breaks the ones that give,
    /// and makes a body of every part that came apart. Runs after the step, between solves.
    pub(crate) fn update_breakables(&mut self) -> Result<(), PhysicsError> {
        self.breakables.events.clear();
        if self.breakables.is_empty() {
            return Ok(());
        }
        let dt = self.settings.fixed_timestep;
        // by handle, a split moves dense indexes
        let before: HashMap<BodyHandle, Vec3> = self
            .breakables
            .before
            .iter()
            .enumerate()
            .filter(|(dense, _)| *dense < self.bodies.len())
            .map(|(dense, velocity)| (self.bodies.handle_at(dense), *velocity))
            .collect();
        let loads = self.piece_loads(dt);
        for (index, (force, hitter)) in loads.into_iter().enumerate() {
            let Some(mut breakable) = self.breakables.slots[index].take() else {
                continue;
            };
            let handle = BreakableHandle {
                index: index as u32,
                generation: self.breakables.generations[index],
            };
            let result = self.load_and_split(handle, &mut breakable, (&force, &hitter), &before);
            self.breakables.slots[index] = Some(breakable);
            result?;
        }
        Ok(())
    }

    /// For each breakable, the force on each of its pieces from the last step's contacts and the
    /// body that pushed hardest on it. One pass over the contacts, each side found by its body in
    /// one sorted list: searched breakable by breakable and group by group it was 1.2 ms a step
    /// with a few hundred pieces loose.
    fn piece_loads(&self, dt: f32) -> Vec<PieceLoads> {
        let slots = &self.breakables.slots;
        let mut owners: Vec<(BodyHandle, u32, u32)> = Vec::new();
        for (slot, breakable) in slots.iter().enumerate() {
            let Some(breakable) = breakable else {
                continue;
            };
            for (group, held) in breakable.groups.iter().enumerate() {
                owners.push((held.body, slot as u32, group as u32));
            }
        }
        owners.sort_unstable();
        let mut loads: Vec<PieceLoads> = slots
            .iter()
            .map(|breakable| {
                let count = breakable
                    .as_ref()
                    .map_or(0, |breakable| breakable.pieces.len());
                (vec![0.0; count], vec![None; count])
            })
            .collect();
        for entry in &self.contact_cache {
            for (body, part, other) in [
                (entry.a, entry.part_a, entry.b),
                (entry.b, entry.part_b, entry.a),
            ] {
                let Ok(at) = owners.binary_search_by_key(&body, |owner| owner.0) else {
                    continue;
                };
                let (_, slot, group) = owners[at];
                let Some(breakable) = &slots[slot as usize] else {
                    continue;
                };
                let Some(&piece) = breakable.groups[group as usize].pieces.get(part as usize)
                else {
                    continue;
                };
                let load: f32 = entry.points[..entry.count]
                    .iter()
                    .map(|point| point.pushed / dt)
                    .sum();
                let (force, hitter) = &mut loads[slot as usize];
                force[piece as usize] += load;
                let best = &mut hitter[piece as usize];
                if best.is_none_or(|(held, _)| load > held) {
                    *best = Some((load, other));
                }
            }
        }
        loads
    }

    fn load_and_split(
        &mut self,
        handle: BreakableHandle,
        breakable: &mut Breakable,
        (force, hitter): (&[f32], &Hits),
        before: &HashMap<BodyHandle, Vec3>,
    ) -> Result<(), PhysicsError> {
        let mut held = vec![0.0_f32; breakable.pieces.len()];
        for bond in breakable.bonds.iter().filter(|bond| !bond.broken) {
            held[bond.a as usize] += bond.strength;
            held[bond.b as usize] += bond.strength;
        }
        let mut broke = Vec::new();
        for (index, bond) in breakable
            .bonds
            .iter_mut()
            .enumerate()
            .filter(|(_, bond)| !bond.broken)
        {
            // a piece's push shared over its faces by how much each holds
            let share = |piece: u32| {
                let total = held[piece as usize];
                if total > 0.0 {
                    force[piece as usize] * bond.strength / total
                } else {
                    0.0
                }
            };
            let load = share(bond.a).max(share(bond.b));
            let ratio = load / bond.strength;
            if ratio > FATIGUE_FROM {
                bond.damage += ratio * FATIGUE_RATE;
            }
            if ratio >= 1.0 || bond.damage >= 1.0 {
                bond.broken = true;
                broke.push(index);
            }
        }
        if broke.is_empty() {
            return Ok(());
        }
        // the crack runs on: faces around each broken one's pieces take damage, breaking next
        // step past 1, so a brittle thing shatters over a few steps from where it was hit
        if breakable.spread > 0.0 {
            let mut hit = vec![0.0_f32; breakable.pieces.len()];
            for index in &broke {
                let bond = breakable.bonds[*index];
                hit[bond.a as usize] += breakable.spread;
                hit[bond.b as usize] += breakable.spread;
            }
            for bond in breakable.bonds.iter_mut().filter(|bond| !bond.broken) {
                bond.damage += hit[bond.a as usize].max(hit[bond.b as usize]);
            }
        }
        let groups = std::mem::take(&mut breakable.groups);
        for group in groups {
            let parts = connected(&group.pieces, &breakable.bonds);
            if parts.len() == 1 {
                breakable.groups.push(group);
                continue;
            }
            self.split(handle, breakable, group, parts, (hitter, before))?;
        }
        Ok(())
    }

    fn split(
        &mut self,
        handle: BreakableHandle,
        breakable: &mut Breakable,
        group: Group,
        parts: Vec<Vec<u32>>,
        (hitter, before): (&Hits, &HashMap<BodyHandle, Vec3>),
    ) -> Result<(), PhysicsError> {
        let state = self.body_state(group.body)?;
        let (at, turn) = pattern_pose(&group.info, state.position, state.rotation)
            .ok_or(PhysicsError::OutOfRange)?;
        let was_fixed = state.kind != BodyKind::Dynamic;
        let middle = state.position;
        self.despawn(group.body)?;
        for pieces in parts {
            let anchored = pieces
                .iter()
                .any(|p| breakable.pieces[*p as usize].anchored);
            let kind = if was_fixed && anchored {
                BodyKind::Static
            } else {
                BodyKind::Dynamic
            };
            let volume: f32 = pieces
                .iter()
                .map(|p| breakable.pieces[*p as usize].volume)
                .sum();
            let mass = volume * breakable.material.density;
            let center = pieces
                .iter()
                .map(|p| {
                    let piece = &breakable.pieces[*p as usize];
                    piece.center * piece.volume
                })
                .fold(Vec3::ZERO, |sum, c| sum + c)
                / volume;
            let world_center = at
                .shifted(DVec3::from_vec3(turn.rotate_vec3(center)))
                .ok_or(PhysicsError::OutOfRange)?;
            let arm = world_center.offset_from(middle).as_vec3();
            let mut linear = if was_fixed {
                Vec3::ZERO
            } else {
                state.linear_velocity + state.angular_velocity.cross(arm)
            };
            let angular = if was_fixed {
                Vec3::ZERO
            } else {
                state.angular_velocity
            };
            // a loose part hit hard moves on with what hit it, as if they stuck together, so
            // the hit goes into the part and not into the whole it left
            let struck = pieces
                .iter()
                .filter_map(|p| hitter[*p as usize])
                .max_by(|a, b| a.0.total_cmp(&b.0));
            if kind == BodyKind::Dynamic
                && let Some((_, other)) = struck
                && let Ok(dense) = self.bodies.dense_index(other)
                && self.bodies.kind[dense] == BodyKind::Dynamic
                && let Some(before) = before.get(&other).copied()
            {
                let other_mass = self.bodies.inverse_mass[dense].recip();
                if other_mass.is_finite() && mass > 0.0 {
                    let shared = (before * other_mass + linear * mass) / (other_mass + mass);
                    linear = shared;
                    self.bodies.linear[dense] = shared;
                    self.bodies.wake(dense);
                }
            }
            let made = self.spawn_group(breakable, pieces, (at, turn), kind, (linear, angular))?;
            self.breakables.events.push(FractureEvent {
                breakable: handle,
                body: made.body,
                pieces: made.pieces.len() as u32,
                mass,
            });
            breakable.groups.push(made);
        }
        Ok(())
    }
}

/// The pieces of `pieces` still holding together, each part in piece order, parts by their
/// lowest piece.
fn connected(pieces: &[u32], bonds: &[Bond]) -> Vec<Vec<u32>> {
    let local = |piece: u32| pieces.iter().position(|p| *p == piece);
    let mut parent: Vec<usize> = (0..pieces.len()).collect();
    fn root(parent: &mut [usize], mut at: usize) -> usize {
        while parent[at] != at {
            parent[at] = parent[parent[at]];
            at = parent[at];
        }
        at
    }
    for bond in bonds.iter().filter(|bond| !bond.broken) {
        if let (Some(a), Some(b)) = (local(bond.a), local(bond.b)) {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    let mut parts: Vec<Vec<u32>> = Vec::new();
    let mut slot = vec![usize::MAX; pieces.len()];
    for (i, piece) in pieces.iter().enumerate() {
        let r = root(&mut parent, i);
        if slot[r] == usize::MAX {
            slot[r] = parts.len();
            parts.push(Vec::new());
        }
        parts[slot[r]].push(*piece);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::testing::at;
    use crate::{JobDispatch, PhysicsSettings, Sequential, Shape};

    struct Threads(usize);

    impl JobDispatch for Threads {
        fn parallelism(&self) -> usize {
            self.0
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            let workers = self.0;
            std::thread::scope(|scope| {
                for worker in 0..workers {
                    scope.spawn(move || {
                        for index in (0..jobs).filter(|index| index % workers == worker) {
                            job(index);
                        }
                    });
                }
            });
        }
    }

    fn world() -> PhysicsWorld {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(30.0, 1.0, 30.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
        world
    }

    fn mass(world: &PhysicsWorld, bodies: &[BodyHandle]) -> f32 {
        bodies
            .iter()
            .map(|body| {
                let dense = world.bodies.dense_index(*body).unwrap();
                world.bodies.inverse_mass[dense].recip()
            })
            .sum()
    }

    fn glass_cube(world: &mut PhysicsWorld, height: f32, strength: f32) -> BreakableHandle {
        let pattern = voronoi_box(Vec3::splat(0.5), 24, 5).unwrap();
        world
            .add_breakable(BreakableDesc {
                strength,
                material: Material {
                    density: 2500.0,
                    ..Material::default()
                },
                ..BreakableDesc::new(pattern, at(0.0, height, 0.0))
            })
            .unwrap()
    }

    fn world_with_cube(height: f32, strength: f32) -> (PhysicsWorld, BreakableHandle) {
        let mut world = world();
        let cube = glass_cube(&mut world, height, strength);
        (world, cube)
    }

    #[test]
    fn thin_and_finely_cut_patterns_all_build() {
        let mut world = world();
        for (half, cells, seed) in [
            (Vec3::new(1.6, 1.25, 0.12), 60, 7),
            (Vec3::new(3.0, 2.0, 0.08), 120, 1),
            (Vec3::new(0.05, 1.0, 1.0), 40, 4),
            (Vec3::splat(0.4), 200, 9),
        ] {
            let pattern = voronoi_box(half, cells, seed).unwrap();
            world
                .add_breakable(BreakableDesc::new(pattern, at(0.0, 10.0, 0.0)))
                .unwrap();
        }
    }

    #[test]
    fn a_dropped_cube_breaks_when_weak_and_keeps_its_mass() {
        let (mut world, cube) = world_with_cube(6.0, 2.0e4);
        let whole = mass(&world, &world.breakable_bodies(cube));
        assert!((whole - 2500.0).abs() < 2500.0 * 1e-3, "{whole}");
        let mut events = 0;
        for _ in 0..240 {
            world.step_fixed().unwrap();
            events += world.fracture_events().len();
        }
        let parts = world.breakable_bodies(cube);
        assert!(parts.len() > 3, "{} parts", parts.len());
        assert!(events >= parts.len() - 1);
        let kept = mass(&world, &parts);
        assert!((kept - whole).abs() < whole * 1e-3, "{kept} of {whole}");
        // strong enough, it lands whole
        let (mut world, cube) = world_with_cube(6.0, 1.0e9);
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        assert_eq!(world.breakable_bodies(cube).len(), 1);
    }

    #[test]
    fn a_split_keeps_the_momentum_it_had() {
        // in the air, spinning, nothing touching: a split by hand changes no momentum
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        let pattern = voronoi_box(Vec3::new(1.0, 0.5, 0.3), 16, 9).unwrap();
        let handle = world
            .add_breakable(BreakableDesc::new(pattern, at(0.0, 5.0, 0.0)))
            .unwrap();
        let body = world.breakable_bodies(handle)[0];
        world
            .set_velocity(body, Vec3::new(1.0, 2.0, -0.5), Vec3::new(0.3, 1.5, -0.7))
            .unwrap();
        let momentum = |world: &PhysicsWorld| {
            world
                .breakable_bodies(handle)
                .iter()
                .map(|b| {
                    let dense = world.bodies.dense_index(*b).unwrap();
                    world.bodies.linear[dense] / world.bodies.inverse_mass[dense]
                })
                .fold(Vec3::ZERO, |sum, p| sum + p)
        };
        let before = momentum(&world);
        let mut breakable = world.breakables.slots[0].take().unwrap();
        for bond in breakable.bonds.iter_mut().filter(|b| b.a % 2 == 0) {
            bond.broken = true;
        }
        let group = breakable.groups.pop().unwrap();
        let parts = connected(&group.pieces, &breakable.bonds);
        assert!(parts.len() > 1);
        let none = vec![None; breakable.pieces.len()];
        world
            .split(
                handle,
                &mut breakable,
                group,
                parts,
                (&none, &HashMap::new()),
            )
            .unwrap();
        world.breakables.slots[0] = Some(breakable);
        let after = momentum(&world);
        assert!(
            (after - before).length() < before.length() * 1e-3,
            "{before:?} {after:?}"
        );
    }

    fn wall(world: &mut PhysicsWorld) -> BreakableHandle {
        let pattern = voronoi_box(Vec3::new(1.5, 1.0, 0.1), 30, 2).unwrap();
        world
            .add_breakable(BreakableDesc {
                fixed: true,
                strength: 1.0e5,
                material: Material {
                    density: 2400.0,
                    ..Material::default()
                },
                ..BreakableDesc::new(pattern, at(0.0, 1.0, 0.0))
            })
            .unwrap()
    }

    fn world_and_wall() -> (PhysicsWorld, BreakableHandle) {
        let mut world = world();
        let target = wall(&mut world);
        (world, target)
    }

    fn shoot(world: &mut PhysicsWorld, speed: f32) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.15))
                    .density(7800.0)
                    .local_position(Vec3::new(0.2, 1.2, -2.0))
                    .linear_velocity(Vec3::new(0.0, 0.0, speed)),
            )
            .unwrap()
    }

    #[test]
    fn a_fast_ball_breaks_a_wall_a_slow_one_does_not_and_the_footing_stays() {
        let (mut world, target) = world_and_wall();
        shoot(&mut world, 1.0);
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        assert_eq!(world.breakable_bodies(target).len(), 1, "a tap");
        let (mut world, target) = world_and_wall();
        let ball = shoot(&mut world, 40.0);
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let parts = world.breakable_bodies(target);
        assert!(parts.len() > 2, "{} parts", parts.len());
        let kinds: Vec<BodyKind> = parts
            .iter()
            .map(|b| world.body_state(*b).unwrap().kind)
            .collect();
        assert!(kinds.contains(&BodyKind::Static), "the footing stands");
        assert!(kinds.contains(&BodyKind::Dynamic), "pieces fall");
        // the ball went on through, it did not bounce back
        let z = world.body_state(ball).unwrap().position.to_dvec3().z;
        assert!(z > -0.5, "{z}");
    }

    #[test]
    fn a_wall_breaks_on_the_same_after_a_snapshot() {
        let (mut world, target) = world_and_wall();
        shoot(&mut world, 40.0);
        for _ in 0..12 {
            world.step_fixed().unwrap();
        }
        let parts = world.breakable_bodies(target).len();
        assert!(parts > 1, "{parts} parts at the save");
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(
            loaded.breakable_bodies(target),
            world.breakable_bodies(target)
        );
        // a second ball breaks more, from the damage already there
        for world in [&mut world, &mut loaded] {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::sphere(0.15))
                        .density(7800.0)
                        .local_position(Vec3::new(-0.8, 0.6, -2.0))
                        .linear_velocity(Vec3::new(0.0, 0.0, 40.0)),
                )
                .unwrap();
            for _ in 0..60 {
                world.step_fixed().unwrap();
            }
        }
        assert_eq!(loaded.state_hash(), world.state_hash());
        assert_eq!(
            loaded.breakable_bodies(target),
            world.breakable_bodies(target)
        );
    }

    /// A fixed 2 m by 1.6 m panel of `kind`, 6 cm thick, shot once at 32 m/s, and how many
    /// parts it ends in.
    fn shot_panel(kind: &FractureMaterial, spread: Option<f32>) -> usize {
        let mut world = world();
        let mut desc =
            BreakableDesc::of(kind, Vec3::new(1.0, 0.8, 0.03), at(0.0, 0.8, 0.0), 4).unwrap();
        desc.fixed = true;
        if let Some(spread) = spread {
            desc.crack_spread = spread;
        }
        let panel = world.add_breakable(desc).unwrap();
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.12))
                    .density(7800.0)
                    .local_position(Vec3::new(0.1, 0.9, -1.5))
                    .linear_velocity(Vec3::new(0.0, 0.0, 32.0)),
            )
            .unwrap();
        for _ in 0..90 {
            world.step_fixed().unwrap();
        }
        world.breakable_bodies(panel).len()
    }

    #[test]
    fn glass_shatters_where_concrete_breaks_in_chunks() {
        let glass = shot_panel(&FractureMaterial::GLASS, None);
        let concrete = shot_panel(&FractureMaterial::CONCRETE, None);
        assert!(
            glass > 3 * concrete.max(1),
            "glass {glass}, concrete {concrete}"
        );
        // its cracks running on is what shatters it
        let calm = shot_panel(&FractureMaterial::GLASS, Some(0.0));
        assert!(glass > calm, "spread {glass}, none {calm}");
    }

    #[test]
    fn a_material_and_its_spread_survive_a_snapshot() {
        let mut world = world();
        let desc = BreakableDesc::of(
            &FractureMaterial::WOOD,
            Vec3::new(0.1, 1.0, 0.1),
            at(0.0, 3.0, 0.0),
            2,
        )
        .unwrap();
        assert!(desc.grain.is_some_and(|grain| grain.axis == Vec3::Y));
        let post = world.add_breakable(desc).unwrap();
        let loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(loaded.breakable_bodies(post), world.breakable_bodies(post));
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
        let mut bad =
            BreakableDesc::of(&FractureMaterial::GLASS, Vec3::ONE, at(0.0, 3.0, 0.0), 2).unwrap();
        bad.crack_spread = 1.5;
        assert!(world.add_breakable(bad).is_err());
    }

    #[test]
    fn broken_groups_leave_no_compounds_behind() {
        let (mut world, cube) = world_with_cube(4.0, 2.0e5);
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let groups = world.breakable_bodies(cube);
        assert!(groups.len() > 2, "{}", groups.len());
        let held = groups
            .iter()
            .filter(|body| {
                let dense = world.bodies.dense_index(**body).unwrap();
                matches!(world.bodies.shape[dense], Shape::Compound(_))
            })
            .count();
        let live = world
            .bodies
            .assets
            .compounds
            .iter()
            .filter(|compound| compound.life != crate::compound::Life::Freed)
            .count();
        assert_eq!(live, held);
    }

    #[test]
    fn breaking_comes_out_the_same_on_any_worker_count() {
        let run = |dispatch: &dyn JobDispatch| {
            let (mut world, target) = world_and_wall();
            shoot(&mut world, 40.0);
            glass_cube(&mut world, 4.0, 2.0e4);
            for _ in 0..90 {
                world.step_with(dispatch).unwrap();
            }
            world
                .breakable_bodies(target)
                .iter()
                .map(|b| {
                    let state = world.body_state(*b).unwrap();
                    (state.position.to_dvec3().as_vec3(), state.rotation)
                })
                .collect::<Vec<_>>()
        };
        let one = run(&Sequential);
        assert!(one.len() > 2);
        assert_eq!(one, run(&Threads(4)));
    }
}
