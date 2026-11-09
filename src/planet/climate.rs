use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::error::PhysicsError;

/// Seconds in a year of 365.25 days, for heat capacities given per year.
pub const SECONDS_PER_YEAR: f64 = 365.25 * 86_400.0;

/// A planet's climate at its cheapest: one temperature per band of latitude, warmed by its sun
/// through the seasons, cooled by what it sends to space and evened out by what its air and seas
/// carry toward the poles. An energy balance model, after Budyko, Sellers and North. Long term
/// temperatures, not weather.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClimateDesc {
    /// Sunlight at the planet's distance, W/m^2 across the beam, 1361 for the earth.
    pub solar: f64,
    /// Tilt of the spin axis to the orbit, radians.
    pub obliquity: f64,
    /// Seconds for one orbit, for the seasons.
    pub year: f64,
    /// Bands from pole to pole, each the same area.
    pub bands: usize,
    /// Heat sent to space is `emission + emission_slope * celsius`, W/m^2: the greenhouse.
    pub emission: f64,
    pub emission_slope: f64,
    /// Heat carried toward the poles, W/m^2 per degree, over the sine of latitude.
    pub diffusion: f64,
    /// J/m^2 per degree, the oceans' mixed layer holds most of it.
    pub heat_capacity: f64,
    /// Share of sunlight sent back by ground and sea, and by ice.
    pub albedo: f64,
    pub ice_albedo: f64,
    /// Celsius under which a band is ice covered.
    pub ice_temperature: f64,
}

impl ClimateDesc {
    /// The earth, with north's constants.
    pub fn earth() -> Self {
        Self {
            solar: 1361.0,
            obliquity: 23.44_f64.to_radians(),
            year: SECONDS_PER_YEAR,
            bands: 90,
            emission: 203.3,
            emission_slope: 2.09,
            diffusion: 0.44,
            heat_capacity: 4.2e7,
            albedo: 0.3,
            ice_albedo: 0.62,
            ice_temperature: -10.0,
        }
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let share = |value: f64| (0.0..=1.0).contains(&value);
        let valid = self.solar.is_finite()
            && self.solar >= 0.0
            && self.obliquity.is_finite()
            && self.obliquity.abs() <= PI
            && positive(self.year)
            && (2..=4096).contains(&self.bands)
            && self.emission.is_finite()
            && positive(self.emission_slope)
            && self.diffusion.is_finite()
            && self.diffusion >= 0.0
            && positive(self.heat_capacity)
            && share(self.albedo)
            && share(self.ice_albedo)
            && self.ice_temperature.is_finite();
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidSettings)
        }
    }
}

/// [`ClimateDesc`] running: band temperatures in celsius, south pole first.
#[derive(Debug, Clone, PartialEq)]
pub struct Climate {
    desc: ClimateDesc,
    temperatures: Vec<f64>,
    /// Seconds since the northern spring.
    time: f64,
}

impl Climate {
    /// Every band at `celsius`, at the northern spring.
    ///
    /// # Errors
    /// Fails on a desc out of range.
    pub fn new(desc: ClimateDesc, celsius: f64) -> Result<Self, PhysicsError> {
        desc.validate()?;
        if !celsius.is_finite() {
            return Err(PhysicsError::NonFinite("climate temperature"));
        }
        Ok(Self {
            temperatures: vec![celsius; desc.bands],
            desc,
            time: 0.0,
        })
    }

    pub fn desc(&self) -> &ClimateDesc {
        &self.desc
    }

    pub fn temperatures(&self) -> &[f64] {
        &self.temperatures
    }

    /// Seconds since the northern spring.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The sine of latitude in the middle of each band, south first.
    fn middle(&self, band: usize) -> f64 {
        let width = 2.0 / self.desc.bands as f64;
        -1.0 + (band as f64 + 0.5) * width
    }

    /// Celsius at a latitude in radians, between the two nearest bands.
    pub fn temperature_at(&self, latitude: f64) -> f64 {
        let bands = self.desc.bands;
        let x = latitude.clamp(-FRAC_PI_2, FRAC_PI_2).sin();
        let at = ((x + 1.0) / 2.0 * bands as f64 - 0.5).clamp(0.0, (bands - 1) as f64);
        let low = at.floor() as usize;
        let high = (low + 1).min(bands - 1);
        let share = at - low as f64;
        self.temperatures[low] * (1.0 - share) + self.temperatures[high] * share
    }

    /// Over the whole planet, the bands being the same area.
    pub fn mean(&self) -> f64 {
        self.temperatures.iter().sum::<f64>() / self.desc.bands as f64
    }

    /// Daily mean sunlight on flat ground at the sine of latitude `x`, now, W/m^2.
    pub fn sunlight(&self, x: f64) -> f64 {
        let season = TAU * self.time / self.desc.year;
        let declination = (self.desc.obliquity.sin() * season.sin()).asin();
        daily_sunlight(self.desc.solar, x, declination)
    }

    /// Share of sunlight a band at `celsius` sends back, ice coming in over a few degrees so a
    /// band at the edge does not flicker.
    fn albedo(&self, celsius: f64) -> f64 {
        let ice = 0.5 * (1.0 - ((celsius - self.desc.ice_temperature) / 2.0).tanh());
        self.desc.albedo + (self.desc.ice_albedo - self.desc.albedo) * ice
    }

    /// Heat in and out over the planet now, W/m^2: what it absorbs, what it sends to space.
    pub fn balance(&self) -> (f64, f64) {
        let bands = self.desc.bands as f64;
        let (mut absorbed, mut emitted) = (0.0, 0.0);
        for (band, celsius) in self.temperatures.iter().enumerate() {
            absorbed += self.sunlight(self.middle(band)) * (1.0 - self.albedo(*celsius));
            emitted += self.desc.emission + self.desc.emission_slope * celsius;
        }
        (absorbed / bands, emitted / bands)
    }

    /// Advances `seconds`, implicit in the cooling and the carrying, so a step of weeks is
    /// stable. Sunlight and albedo are taken at the start.
    ///
    /// # Errors
    /// Fails on a negative or non finite step.
    pub fn step(&mut self, seconds: f64) -> Result<(), PhysicsError> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(PhysicsError::InvalidSettings);
        }
        if seconds == 0.0 {
            return Ok(());
        }
        let desc = self.desc;
        let bands = desc.bands;
        let width = 2.0 / bands as f64;
        let hold = desc.heat_capacity / seconds;
        // carrying across the edge above each band, none past the poles
        let edge = |band: usize| {
            let x = -1.0 + (band as f64 + 1.0) * width;
            desc.diffusion * (1.0 - x * x) / (width * width)
        };
        let mut below = vec![0.0; bands];
        let mut middle = vec![0.0; bands];
        let mut above = vec![0.0; bands];
        let mut right = vec![0.0; bands];
        for band in 0..bands {
            let (down, up) = (
                if band > 0 { edge(band - 1) } else { 0.0 },
                if band + 1 < bands { edge(band) } else { 0.0 },
            );
            let celsius = self.temperatures[band];
            let heating = self.sunlight(self.middle(band)) * (1.0 - self.albedo(celsius));
            below[band] = -down;
            above[band] = -up;
            middle[band] = hold + desc.emission_slope + down + up;
            right[band] = hold * celsius + heating - desc.emission;
        }
        solve_tridiagonal(&below, &mut middle, &above, &mut right);
        self.temperatures.copy_from_slice(&right);
        self.time = (self.time + seconds).rem_euclid(desc.year);
        Ok(())
    }
}

/// Daily mean sunlight at the sine of latitude `x` with the sun at `declination`, from the hours
/// it is up.
fn daily_sunlight(solar: f64, x: f64, declination: f64) -> f64 {
    let latitude = x.clamp(-1.0, 1.0).asin();
    let set = (-latitude.tan() * declination.tan())
        .clamp(-1.0, 1.0)
        .acos();
    let (sin, cos) = (latitude.sin(), latitude.cos());
    (solar / PI * (set * sin * declination.sin() + cos * declination.cos() * set.sin())).max(0.0)
}

/// Thomas' algorithm, the answer left in `right`. The system is diagonally dominant here.
fn solve_tridiagonal(below: &[f64], middle: &mut [f64], above: &[f64], right: &mut [f64]) {
    let count = middle.len();
    for row in 1..count {
        let factor = below[row] / middle[row - 1];
        middle[row] -= factor * above[row - 1];
        right[row] -= factor * right[row - 1];
    }
    right[count - 1] /= middle[count - 1];
    for row in (0..count - 1).rev() {
        right[row] = (right[row] - above[row] * right[row + 1]) / middle[row];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Twenty years in steps of a few days, the seasons and all, then the next year's average.
    fn settled(desc: ClimateDesc, start: f64) -> (Climate, Vec<f64>) {
        let mut climate = Climate::new(desc, start).unwrap();
        let step = desc.year / 100.0;
        for _ in 0..20 * 100 {
            climate.step(step).unwrap();
        }
        let mut average = vec![0.0; desc.bands];
        for _ in 0..100 {
            climate.step(step).unwrap();
            for (sum, celsius) in average.iter_mut().zip(climate.temperatures()) {
                *sum += celsius / 100.0;
            }
        }
        (climate, average)
    }

    #[test]
    fn the_earth_settles_near_its_real_temperatures() {
        let (_, average) = settled(ClimateDesc::earth(), 10.0);
        let mean = average.iter().sum::<f64>() / average.len() as f64;
        let equator = (average[44] + average[45]) / 2.0;
        let pole = (average[0] + average[89]) / 2.0;
        // here 12.9, 26.8 and -17.9. measured about 14 over the planet, 27 at the equator, -18
        // at the north pole over a year
        assert!((10.0..18.0).contains(&mean), "{mean}");
        assert!((22.0..32.0).contains(&equator), "{equator}");
        assert!(pole < -10.0, "{pole}");
        // the same either side, the tilt only moves the seasons
        for band in 0..45 {
            assert!((average[band] - average[89 - band]).abs() < 0.5);
        }
    }

    #[test]
    fn settled_it_sends_out_what_it_takes_in() {
        let mut climate = settled(ClimateDesc::earth(), 10.0).0;
        let step = climate.desc().year / 100.0;
        let (mut absorbed, mut emitted) = (0.0, 0.0);
        for _ in 0..100 {
            let (taken, sent) = climate.balance();
            absorbed += taken / 100.0;
            emitted += sent / 100.0;
            climate.step(step).unwrap();
        }
        assert!((absorbed - emitted).abs() < 0.2, "{absorbed} {emitted}");
        assert!((absorbed - 240.0).abs() < 10.0, "{absorbed}");
    }

    #[test]
    fn a_dimmer_sun_freezes_it_over_and_a_brighter_one_warms_it() {
        let earth = ClimateDesc::earth();
        let mean = |solar: f64| {
            let (_, average) = settled(ClimateDesc { solar, ..earth }, 10.0);
            average.iter().sum::<f64>() / average.len() as f64
        };
        let (dim, now, bright) = (mean(1361.0 * 0.85), mean(1361.0), mean(1361.0 * 1.05));
        assert!(dim < now && now < bright, "{dim} {now} {bright}");
        // ice sends the light back, so it runs away: a snowball
        let (_, frozen) = settled(
            ClimateDesc {
                solar: 1361.0 * 0.85,
                ..earth
            },
            10.0,
        );
        assert!(frozen.iter().all(|celsius| *celsius < -10.0), "{frozen:?}");
        // and it remembers: started frozen under today's sun it stays colder than started warm
        let cold = settled(earth, -60.0).1;
        assert!(cold.iter().sum::<f64>() / cold.len() as f64 <= now + 1e-6);
    }

    #[test]
    fn the_seasons_swap_between_the_halves() {
        let mut climate = settled(ClimateDesc::earth(), 10.0).0;
        let quarter = climate.desc().year / 4.0;
        // from spring to the northern summer's end
        while climate.time() > 1.0 {
            climate.step(climate.desc().year / 400.0).unwrap();
            if climate.time() < climate.desc().year / 400.0 {
                break;
            }
        }
        let mut seen = 0.0;
        while seen < quarter * 1.3 {
            climate.step(quarter / 100.0).unwrap();
            seen += quarter / 100.0;
        }
        let north = climate.temperature_at(60.0_f64.to_radians());
        let south = climate.temperature_at(-60.0_f64.to_radians());
        assert!(north > south + 5.0, "{north} {south}");
    }

    #[test]
    fn sunlight_matches_the_textbook_days() {
        let solar = 1361.0;
        // the equator at an equinox: solar over pi
        assert!((daily_sunlight(solar, 0.0, 0.0) - solar / PI).abs() < 1e-9);
        // the pole in its winter gets none, in its summer more than the equator
        let tilt = 23.44_f64.to_radians();
        assert_eq!(daily_sunlight(solar, 1.0, -tilt), 0.0);
        assert!(daily_sunlight(solar, 1.0, tilt) > daily_sunlight(solar, 0.0, tilt));
        // north pole at its solstice: solar times the sine of the tilt
        assert!((daily_sunlight(solar, 1.0, tilt) - solar * tilt.sin()).abs() < 1e-6);
    }

    #[test]
    fn bad_climates_are_refused() {
        let earth = ClimateDesc::earth();
        for bad in [
            ClimateDesc { bands: 1, ..earth },
            ClimateDesc {
                heat_capacity: 0.0,
                ..earth
            },
            ClimateDesc {
                albedo: 1.5,
                ..earth
            },
            ClimateDesc {
                year: f64::NAN,
                ..earth
            },
        ] {
            assert!(Climate::new(bad, 0.0).is_err());
        }
        assert!(Climate::new(earth, f64::INFINITY).is_err());
        let mut climate = Climate::new(earth, 0.0).unwrap();
        assert!(climate.step(-1.0).is_err());
        assert!(climate.step(f64::NAN).is_err());
    }
}
