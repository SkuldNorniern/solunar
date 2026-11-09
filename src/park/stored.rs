use algebrix::{Quat, Vec3};

use crate::collision::convex::MAX_CONVEX_POINTS;
use crate::compound::{CompoundChild, ConvexId, MAX_COMPOUND_CHILDREN};
use crate::error::PhysicsError;
use crate::material::Material;
use crate::shape::Shape;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::world::PhysicsWorld;

/// A body's shape as it is, not as an id into one world's tables: a parked body can come back
/// into a world that never had it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Geometry {
    Part(Part),
    Compound(Vec<StoredChild>),
}

/// A ball, box or capsule as it is, or a hull by its points, with the id it had so a world
/// that still has it takes it again rather than make another.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Part {
    Plain(Shape),
    Hull { points: Vec<Vec3>, id: ConvexId },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StoredChild {
    part: Part,
    position: Vec3,
    rotation: Quat,
    density_scale: f32,
    surface: Option<Material>,
}

impl PhysicsWorld {
    fn stored_part(&self, shape: &Shape) -> Result<Part, PhysicsError> {
        match *shape {
            Shape::Sphere { .. } | Shape::Cuboid { .. } | Shape::Capsule { .. } => {
                Ok(Part::Plain(*shape))
            }
            Shape::Convex(id) => {
                let hull = self
                    .bodies
                    .assets
                    .convex(shape)
                    .ok_or(PhysicsError::InvalidShape)?;
                Ok(Part::Hull {
                    points: hull.source.clone(),
                    id,
                })
            }
            _ => Err(PhysicsError::InvalidShape),
        }
    }

    /// `shape` as it is, for a body leaving the world. A mesh can not be stored.
    pub(crate) fn store_geometry(&self, shape: &Shape) -> Result<Geometry, PhysicsError> {
        let Shape::Compound(id) = *shape else {
            return Ok(Geometry::Part(self.stored_part(shape)?));
        };
        let compound = self
            .bodies
            .assets
            .compound(id)
            .ok_or(PhysicsError::InvalidShape)?;
        compound
            .children
            .iter()
            .map(|child| {
                Ok(StoredChild {
                    part: self.stored_part(&child.shape)?,
                    position: child.position,
                    rotation: child.rotation,
                    density_scale: child.density_scale,
                    surface: child.surface,
                })
            })
            .collect::<Result<_, _>>()
            .map(Geometry::Compound)
    }

    /// A part back in this world: its hull taken again if this world has that very hull, made
    /// anew if not. How it sits about its own center comes back with it.
    fn restore_part(&mut self, part: &Part) -> Result<(Shape, Vec3, Quat), PhysicsError> {
        match part {
            Part::Plain(shape) => Ok((*shape, Vec3::ZERO, Quat::IDENTITY)),
            Part::Hull { points, id } => {
                let same = Shape::Convex(*id);
                if self
                    .bodies
                    .assets
                    .convex(&same)
                    .is_some_and(|hull| hull.source == *points)
                {
                    return Ok((same, Vec3::ZERO, Quat::IDENTITY));
                }
                let info = self.add_convex(points)?;
                Ok((info.shape, info.center, info.axes))
            }
        }
    }

    /// `geometry` back as a shape of this world, and where its center and axes sit against
    /// those it had: a hull or compound built again comes out rebased.
    pub(crate) fn restore_geometry(
        &mut self,
        geometry: &Geometry,
    ) -> Result<(Shape, Vec3, Quat), PhysicsError> {
        let Geometry::Compound(children) = geometry else {
            let Geometry::Part(part) = geometry else {
                unreachable!()
            };
            return self.restore_part(part);
        };
        let mut built = Vec::with_capacity(children.len());
        for child in children {
            let (shape, center, axes) = self.restore_part(&child.part)?;
            built.push(CompoundChild {
                shape,
                position: child.position + child.rotation.rotate_vec3(center),
                rotation: (child.rotation * axes).normalize(),
                density_scale: child.density_scale,
                surface: child.surface,
            });
        }
        let info = self.add_owned_compound(&built)?;
        Ok((info.shape, info.center, info.axes))
    }
}

fn save_part(out: &mut Writer, part: &Part) {
    match part {
        Part::Plain(shape) => {
            out.u8(0);
            out.shape(shape);
        }
        Part::Hull { points, id } => {
            out.u8(1);
            out.u32(id.0);
            out.len(points.len());
            for point in points {
                out.vec3(*point);
            }
        }
    }
}

fn load_part(reader: &mut Reader<'_>) -> Result<Part, SnapshotError> {
    let bad = SnapshotError::Invalid("parked shape");
    match reader.u8()? {
        0 => {
            let shape = reader.shape()?;
            if !matches!(
                shape,
                Shape::Sphere { .. } | Shape::Cuboid { .. } | Shape::Capsule { .. }
            ) {
                return Err(bad);
            }
            Ok(Part::Plain(shape))
        }
        1 => {
            let id = ConvexId(reader.u32()?);
            let count = reader.count(12, MAX_CONVEX_POINTS)?;
            let mut points = Vec::with_capacity(count);
            for _ in 0..count {
                let point = reader.vec3()?;
                if !point.is_finite() {
                    return Err(bad);
                }
                points.push(point);
            }
            Ok(Part::Hull { points, id })
        }
        _ => Err(bad),
    }
}

pub(crate) fn save_geometry(out: &mut Writer, geometry: &Geometry) {
    match geometry {
        Geometry::Part(part) => {
            out.u8(0);
            save_part(out, part);
        }
        Geometry::Compound(children) => {
            out.u8(1);
            out.len(children.len());
            for child in children {
                save_part(out, &child.part);
                out.vec3(child.position);
                out.quat(child.rotation);
                out.f32(child.density_scale);
                out.u8(u8::from(child.surface.is_some()));
                out.material(&child.surface.unwrap_or_default());
            }
        }
    }
}

pub(crate) fn load_geometry(reader: &mut Reader<'_>) -> Result<Geometry, SnapshotError> {
    let bad = SnapshotError::Invalid("parked shape");
    match reader.u8()? {
        0 => Ok(Geometry::Part(load_part(reader)?)),
        1 => {
            let count = reader.count(2, MAX_COMPOUND_CHILDREN)?;
            if count == 0 {
                return Err(bad);
            }
            let mut children = Vec::with_capacity(count);
            for _ in 0..count {
                let part = load_part(reader)?;
                let (position, rotation, density_scale) =
                    (reader.vec3()?, reader.quat()?, reader.f32()?);
                let has = reader.u8()?;
                let material = reader.material()?;
                let surface = match has {
                    0 => None,
                    1 => Some(material),
                    _ => return Err(bad),
                };
                if !position.is_finite() || !density_scale.is_finite() || density_scale <= 0.0 {
                    return Err(bad);
                }
                children.push(StoredChild {
                    part,
                    position,
                    rotation: rotation.normalize(),
                    density_scale,
                    surface,
                });
            }
            Ok(Geometry::Compound(children))
        }
        _ => Err(bad),
    }
}
