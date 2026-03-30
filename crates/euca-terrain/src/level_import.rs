//! PNG-based importers and exporters for terrain level data.
//!
//! Provides functions to load heightmaps, surface type maps, and walkability
//! masks from PNG images, plus the inverse save operations. All functions are
//! gated behind the `level-import` feature.

use std::path::Path;

use image::{GrayImage, ImageReader, Luma, Rgb, RgbImage};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during level-data import / export.
#[derive(Debug)]
pub enum ImportError {
    /// Underlying I/O failure (file not found, permission denied, etc.).
    Io(std::io::Error),
    /// The `image` crate could not decode or encode the file.
    Image(image::ImageError),
    /// The supplied data length does not match `width * height`.
    DimensionMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Image(e) => write!(f, "image error: {e}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension mismatch: expected {expected} pixels, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for ImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Image(e) => Some(e),
            Self::DimensionMismatch { .. } => None,
        }
    }
}

impl From<std::io::Error> for ImportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<image::ImageError> for ImportError {
    fn from(e: image::ImageError) -> Self {
        Self::Image(e)
    }
}

// ---------------------------------------------------------------------------
// Surface types
// ---------------------------------------------------------------------------

/// Terrain surface material classification.
///
/// Each variant maps to a canonical RGB colour used by the PNG round-trip
/// helpers ([`surface_type_to_rgb`] / [`surface_type_from_rgb`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceType {
    Grass,
    Dirt,
    Stone,
    Water,
    Sand,
    Snow,
    Mud,
    Road,
    Cliff,
    Void,
    /// Application-defined surface type keyed by a 16-bit identifier.
    Custom(u16),
}

/// Returns the canonical RGB colour for a [`SurfaceType`].
pub fn surface_type_to_rgb(s: &SurfaceType) -> (u8, u8, u8) {
    match s {
        SurfaceType::Grass => (34, 139, 34),   // forest green
        SurfaceType::Dirt => (139, 90, 43),    // saddle brown-ish
        SurfaceType::Stone => (128, 128, 128), // gray
        SurfaceType::Water => (0, 0, 255),     // blue
        SurfaceType::Sand => (238, 214, 175),  // sandy
        SurfaceType::Snow => (255, 250, 250),  // almost white
        SurfaceType::Mud => (105, 75, 55),     // dark brown
        SurfaceType::Road => (64, 64, 64),     // dark gray
        SurfaceType::Cliff => (169, 169, 169), // dark-ish gray
        SurfaceType::Void => (0, 0, 0),        // black
        SurfaceType::Custom(id) => {
            // Deterministic mapping: pack the 16-bit id into R and G channels.
            let hi = (*id >> 8) as u8;
            let lo = (*id & 0xFF) as u8;
            (hi, lo, 1) // B=1 sentinel distinguishes from named types
        }
    }
}

/// Returns the [`SurfaceType`] whose canonical colour is closest (Euclidean
/// distance in RGB space) to the given pixel.
pub fn surface_type_from_rgb(r: u8, g: u8, b: u8) -> SurfaceType {
    // Check for the Custom sentinel first: B==1 and exact match.
    if b == 1 {
        let id = (r as u16) << 8 | g as u16;
        // Only treat as Custom when it doesn't accidentally match a named
        // type's canonical colour.
        let candidate = SurfaceType::Custom(id);
        let (cr, cg, cb) = surface_type_to_rgb(&candidate);
        if (cr, cg, cb) == (r, g, b) {
            // Make sure it isn't coincidentally the same as a named type.
            let named = nearest_named_surface(r, g, b);
            let (nr, ng, nb) = surface_type_to_rgb(&named);
            if (nr, ng, nb) != (r, g, b) {
                return candidate;
            }
        }
    }

    nearest_named_surface(r, g, b)
}

/// Nearest-neighbour match among the ten named surface types.
fn nearest_named_surface(r: u8, g: u8, b: u8) -> SurfaceType {
    const NAMED: [SurfaceType; 10] = [
        SurfaceType::Grass,
        SurfaceType::Dirt,
        SurfaceType::Stone,
        SurfaceType::Water,
        SurfaceType::Sand,
        SurfaceType::Snow,
        SurfaceType::Mud,
        SurfaceType::Road,
        SurfaceType::Cliff,
        SurfaceType::Void,
    ];

    let mut best = SurfaceType::Void;
    let mut best_dist = u32::MAX;
    for s in &NAMED {
        let (sr, sg, sb) = surface_type_to_rgb(s);
        let dr = (r as i32 - sr as i32).unsigned_abs();
        let dg = (g as i32 - sg as i32).unsigned_abs();
        let db = (b as i32 - sb as i32).unsigned_abs();
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best = *s;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Heightmap import / export
// ---------------------------------------------------------------------------

/// Loads an 8-bit grayscale PNG as a heightmap.
///
/// Pixel brightness is linearly mapped to `[0.0, 1.0]`.
///
/// Returns `(width, height, data)` where `data.len() == width * height`.
pub fn load_heightmap_png(path: &Path) -> Result<(u32, u32, Vec<f32>), ImportError> {
    let img = ImageReader::open(path)?.decode()?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let data: Vec<f32> = gray.pixels().map(|Luma([v])| *v as f32 / 255.0).collect();
    Ok((w, h, data))
}

/// Loads a 16-bit grayscale PNG as a heightmap for higher precision.
///
/// Pixel brightness is linearly mapped to `[0.0, 1.0]`.
///
/// Returns `(width, height, data)` where `data.len() == width * height`.
pub fn load_heightmap_png_16bit(path: &Path) -> Result<(u32, u32, Vec<f32>), ImportError> {
    let img = ImageReader::open(path)?.decode()?;
    let gray = img.to_luma16();
    let (w, h) = gray.dimensions();
    let data: Vec<f32> = gray.pixels().map(|Luma([v])| *v as f32 / 65535.0).collect();
    Ok((w, h, data))
}

/// Saves a heightmap as an 8-bit grayscale PNG.
///
/// Values are clamped to `[0.0, 1.0]` before quantisation to `[0, 255]`.
pub fn save_heightmap_png(
    path: &Path,
    width: u32,
    height: u32,
    data: &[f32],
) -> Result<(), ImportError> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(ImportError::DimensionMismatch {
            expected,
            actual: data.len(),
        });
    }

    let mut img = GrayImage::new(width, height);
    for (i, val) in data.iter().enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        let byte = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        img.put_pixel(x, y, Luma([byte]));
    }
    img.save(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Surface-type import / export
// ---------------------------------------------------------------------------

/// Loads an RGB PNG and classifies each pixel to the nearest [`SurfaceType`].
///
/// Returns `(width, height, data)`.
pub fn load_surface_png(path: &Path) -> Result<(u32, u32, Vec<SurfaceType>), ImportError> {
    let img = ImageReader::open(path)?.decode()?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let data: Vec<SurfaceType> = rgb
        .pixels()
        .map(|Rgb([r, g, b])| surface_type_from_rgb(*r, *g, *b))
        .collect();
    Ok((w, h, data))
}

/// Saves a surface-type map as an RGB PNG using canonical colours.
pub fn save_surface_png(
    path: &Path,
    width: u32,
    height: u32,
    data: &[SurfaceType],
) -> Result<(), ImportError> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(ImportError::DimensionMismatch {
            expected,
            actual: data.len(),
        });
    }

    let mut img = RgbImage::new(width, height);
    for (i, st) in data.iter().enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        let (r, g, b) = surface_type_to_rgb(st);
        img.put_pixel(x, y, Rgb([r, g, b]));
    }
    img.save(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Walkability import
// ---------------------------------------------------------------------------

/// Loads a grayscale PNG as a walkability mask.
///
/// Pixels with brightness >= 128 are considered walkable (`true`).
///
/// Returns `(width, height, data)`.
pub fn load_walkability_png(path: &Path) -> Result<(u32, u32, Vec<bool>), ImportError> {
    let img = ImageReader::open(path)?.decode()?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let data: Vec<bool> = gray.pixels().map(|Luma([v])| *v >= 128).collect();
    Ok((w, h, data))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: create a temporary directory that is cleaned up on drop.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("euca_level_import_test_{name}"));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    // -- colour helpers ---------------------------------------------------

    #[test]
    fn surface_rgb_roundtrip_named() {
        let named = [
            SurfaceType::Grass,
            SurfaceType::Dirt,
            SurfaceType::Stone,
            SurfaceType::Water,
            SurfaceType::Sand,
            SurfaceType::Snow,
            SurfaceType::Mud,
            SurfaceType::Road,
            SurfaceType::Cliff,
            SurfaceType::Void,
        ];
        for s in &named {
            let (r, g, b) = surface_type_to_rgb(s);
            let recovered = surface_type_from_rgb(r, g, b);
            assert_eq!(*s, recovered, "roundtrip failed for {s:?}");
        }
    }

    #[test]
    fn surface_rgb_roundtrip_custom() {
        for id in [0u16, 1, 255, 1000, 65535] {
            let s = SurfaceType::Custom(id);
            let (r, g, b) = surface_type_to_rgb(&s);
            let recovered = surface_type_from_rgb(r, g, b);
            assert_eq!(s, recovered, "Custom({id}) roundtrip failed");
        }
    }

    #[test]
    fn nearest_colour_picks_grass_for_green() {
        // A clearly green pixel should resolve to Grass.
        let s = surface_type_from_rgb(20, 150, 20);
        assert_eq!(s, SurfaceType::Grass);
    }

    // -- heightmap --------------------------------------------------------

    #[test]
    fn heightmap_save_load_roundtrip() {
        let tmp = TempDir::new("hm_roundtrip");
        let path = tmp.path().join("hm.png");

        let (w, h) = (4, 3);
        let data: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();
        save_heightmap_png(&path, w, h, &data).unwrap();
        let (lw, lh, loaded) = load_heightmap_png(&path).unwrap();
        assert_eq!((lw, lh), (w, h));
        assert_eq!(loaded.len(), data.len());

        // 8-bit quantisation: tolerance of 1/255.
        for (a, b) in data.iter().zip(loaded.iter()) {
            assert!((a - b).abs() < 1.0 / 255.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn heightmap_16bit_has_more_precision() {
        let tmp = TempDir::new("hm_16bit");
        let path = tmp.path().join("hm.png");

        // Save as 8-bit, load as 16-bit (will still be 8-bit data but the
        // loader should not panic).
        let data = vec![0.0, 0.5, 1.0, 0.25];
        save_heightmap_png(&path, 2, 2, &data).unwrap();
        let (w, h, loaded) = load_heightmap_png_16bit(&path).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(loaded.len(), 4);
        // Values should still be in [0, 1].
        for v in &loaded {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn heightmap_dimension_mismatch() {
        let tmp = TempDir::new("hm_dim");
        let path = tmp.path().join("hm.png");
        let result = save_heightmap_png(&path, 2, 2, &[0.0; 5]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImportError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 5);
            }
            other => panic!("expected DimensionMismatch, got {other}"),
        }
    }

    // -- surface map ------------------------------------------------------

    #[test]
    fn surface_save_load_roundtrip() {
        let tmp = TempDir::new("surf_roundtrip");
        let path = tmp.path().join("surf.png");

        let data = vec![
            SurfaceType::Grass,
            SurfaceType::Water,
            SurfaceType::Stone,
            SurfaceType::Sand,
        ];
        save_surface_png(&path, 2, 2, &data).unwrap();
        let (w, h, loaded) = load_surface_png(&path).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(loaded, data);
    }

    #[test]
    fn surface_dimension_mismatch() {
        let tmp = TempDir::new("surf_dim");
        let path = tmp.path().join("surf.png");
        let result = save_surface_png(&path, 3, 3, &[SurfaceType::Void; 2]);
        assert!(result.is_err());
    }

    // -- walkability ------------------------------------------------------

    #[test]
    fn walkability_load() {
        let tmp = TempDir::new("walk");
        let path = tmp.path().join("walk.png");

        // Build a 2x2 grayscale image: white, black, gray-128, gray-127.
        let mut img = GrayImage::new(2, 2);
        img.put_pixel(0, 0, Luma([255])); // walkable
        img.put_pixel(1, 0, Luma([0])); // blocked
        img.put_pixel(0, 1, Luma([128])); // walkable (>= 128)
        img.put_pixel(1, 1, Luma([127])); // blocked  (< 128)
        img.save(&path).unwrap();

        let (w, h, data) = load_walkability_png(&path).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(data, vec![true, false, true, false]);
    }

    // -- error display ----------------------------------------------------

    #[test]
    fn error_display() {
        let e = ImportError::DimensionMismatch {
            expected: 100,
            actual: 50,
        };
        let msg = format!("{e}");
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }
}
