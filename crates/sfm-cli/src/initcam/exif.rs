//! Minimal EXIF reader for the one thing camera initialization needs: the
//! focal length the camera itself recorded.
//!
//! Hand-rolled rather than pulled in as a dependency. What is needed here is a
//! handful of tags out of one JPEG segment, and the format is small and stable
//! enough that parsing it directly is less code than the glue around a general
//! EXIF library would be - matching how this project already treats small,
//! well-specified problems (see the ArUco dictionary, the seeded PRNG, the
//! glob matcher).

/// What a camera recorded about its own optics.
#[derive(Debug, Clone, Default)]
pub struct ExifFocal {
    /// 35mm-equivalent focal length in millimetres (tag 0xA405).
    pub focal_35mm: Option<f64>,
    /// Physical focal length in millimetres (tag 0x920A).
    pub focal_mm: Option<f64>,
    pub make: Option<String>,
    pub model: Option<String>,
}

impl ExifFocal {
    /// Focal length in pixels, if the 35mm equivalent is known.
    ///
    /// The 35mm equivalent is defined against a 36x24mm frame, so it maps onto
    /// the image's *longer* side regardless of orientation - using the width
    /// unconditionally would be wrong by the aspect ratio for every portrait
    /// photo.
    pub fn focal_px(&self, width: u32, height: u32) -> Option<f64> {
        let f35 = self.focal_35mm?;
        if f35 <= 0.0 {
            return None;
        }
        Some(f35 / 36.0 * width.max(height) as f64)
    }
}

fn u16_at(b: &[u8], off: usize, le: bool) -> Option<u16> {
    let s = b.get(off..off + 2)?;
    Some(if le {
        u16::from_le_bytes([s[0], s[1]])
    } else {
        u16::from_be_bytes([s[0], s[1]])
    })
}

fn u32_at(b: &[u8], off: usize, le: bool) -> Option<u32> {
    let s = b.get(off..off + 4)?;
    Some(if le {
        u32::from_le_bytes([s[0], s[1], s[2], s[3]])
    } else {
        u32::from_be_bytes([s[0], s[1], s[2], s[3]])
    })
}

/// Reads one IFD entry's value as an f64, handling the numeric TIFF types.
/// `tiff` is the TIFF block with offsets relative to its own start.
fn entry_value_f64(tiff: &[u8], entry: usize, le: bool) -> Option<f64> {
    let ty = u16_at(tiff, entry + 2, le)?;
    let value_off = entry + 8;
    match ty {
        3 => u16_at(tiff, value_off, le).map(|v| v as f64), // SHORT
        4 => u32_at(tiff, value_off, le).map(|v| v as f64), // LONG
        5 => {
            // RATIONAL: the four value bytes hold an offset to num/den.
            let off = u32_at(tiff, value_off, le)? as usize;
            let num = u32_at(tiff, off, le)? as f64;
            let den = u32_at(tiff, off + 4, le)? as f64;
            if den == 0.0 {
                None
            } else {
                Some(num / den)
            }
        }
        _ => None,
    }
}

fn entry_value_ascii(tiff: &[u8], entry: usize, le: bool) -> Option<String> {
    let ty = u16_at(tiff, entry + 2, le)?;
    if ty != 2 {
        return None;
    }
    let count = u32_at(tiff, entry + 4, le)? as usize;
    let bytes: &[u8] = if count <= 4 {
        tiff.get(entry + 8..entry + 8 + count)?
    } else {
        let off = u32_at(tiff, entry + 8, le)? as usize;
        tiff.get(off..off + count)?
    };
    let s: String = bytes
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as char)
        .collect();
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Walks one IFD, calling `f` with each entry's tag and byte offset.
fn walk_ifd(tiff: &[u8], ifd_off: usize, le: bool, mut f: impl FnMut(u16, usize)) -> Option<()> {
    let count = u16_at(tiff, ifd_off, le)? as usize;
    // A corrupt or misparsed offset can claim an absurd entry count; bound it
    // by what the buffer could actually hold.
    let max = (tiff.len().saturating_sub(ifd_off + 2)) / 12;
    for i in 0..count.min(max) {
        let entry = ifd_off + 2 + i * 12;
        let tag = u16_at(tiff, entry, le)?;
        f(tag, entry);
    }
    Some(())
}

/// Extracts focal-length tags from a JPEG's EXIF block. Returns `None` when
/// the file has no EXIF at all (common: many messaging apps strip it).
pub fn read_jpeg_exif(bytes: &[u8]) -> Option<ExifFocal> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None; // not a JPEG
    }
    // Scan the segment chain for APP1 carrying an "Exif\0\0" header.
    let mut i = 2usize;
    let tiff = loop {
        if i + 4 > bytes.len() || bytes[i] != 0xFF {
            return None;
        }
        let marker = bytes[i + 1];
        // Start of scan: image data follows, no more metadata segments.
        if marker == 0xDA {
            return None;
        }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if len < 2 {
            return None;
        }
        let seg = bytes.get(i + 4..i + 2 + len)?;
        if marker == 0xE1 && seg.len() > 6 && &seg[0..6] == b"Exif\0\0" {
            break &seg[6..];
        }
        i += 2 + len;
    };

    let le = match tiff.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    if u16_at(tiff, 2, le)? != 42 {
        return None;
    }
    let ifd0 = u32_at(tiff, 4, le)? as usize;

    let mut out = ExifFocal::default();
    let mut exif_ifd: Option<usize> = None;
    walk_ifd(tiff, ifd0, le, |tag, entry| match tag {
        0x010F => out.make = entry_value_ascii(tiff, entry, le),
        0x0110 => out.model = entry_value_ascii(tiff, entry, le),
        0x8769 => exif_ifd = u32_at(tiff, entry + 8, le).map(|v| v as usize),
        _ => {}
    })?;

    // The focal-length tags live in the Exif sub-IFD, not IFD0.
    if let Some(sub) = exif_ifd {
        walk_ifd(tiff, sub, le, |tag, entry| match tag {
            0x920A => out.focal_mm = entry_value_f64(tiff, entry, le),
            0xA405 => out.focal_35mm = entry_value_f64(tiff, entry, le),
            _ => {}
        })?;
    }

    if out.focal_35mm.is_none() && out.focal_mm.is_none() {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal little-endian JPEG+EXIF with the given 35mm focal.
    fn synth_jpeg(f35: u16) -> Vec<u8> {
        let mut tiff: Vec<u8> = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at offset 8
                                                     // IFD0: one entry (ExifIFDPointer), next-IFD = 0
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x8769u16.to_le_bytes());
        tiff.extend_from_slice(&4u16.to_le_bytes()); // LONG
        tiff.extend_from_slice(&1u32.to_le_bytes());
        // IFD0 occupies bytes 0..26 (8 header + 2 count + 12 entry + 4 next),
        // so the sub-IFD starts immediately after it.
        tiff.extend_from_slice(&26u32.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(tiff.len(), 26);
        // Exif sub-IFD: one entry (FocalLengthIn35mmFilm, SHORT)
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0xA405u16.to_le_bytes());
        tiff.extend_from_slice(&3u16.to_le_bytes());
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&(f35 as u32).to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());

        let mut app1: Vec<u8> = b"Exif\0\0".to_vec();
        app1.extend_from_slice(&tiff);
        let mut out: Vec<u8> = vec![0xFF, 0xD8];
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&app1);
        out.extend_from_slice(&[0xFF, 0xDA]); // SOS
        out
    }

    #[test]
    fn reads_focal_35mm_and_converts_to_pixels() {
        let jpg = synth_jpeg(26);
        let e = read_jpeg_exif(&jpg).expect("exif present");
        assert_eq!(e.focal_35mm, Some(26.0));
        // Landscape and portrait must agree: the 35mm equivalent is defined
        // against the frame's long side.
        let land = e.focal_px(4032, 3024).unwrap();
        let port = e.focal_px(3024, 4032).unwrap();
        assert!((land - port).abs() < 1e-9);
        assert!((land - 26.0 / 36.0 * 4032.0).abs() < 1e-6);
    }

    #[test]
    fn returns_none_without_exif() {
        // A JPEG whose segments carry no APP1/Exif at all.
        let bytes = vec![0xFF, 0xD8, 0xFF, 0xDA];
        assert!(read_jpeg_exif(&bytes).is_none());
        assert!(read_jpeg_exif(b"not a jpeg").is_none());
    }

    #[test]
    fn survives_truncated_input() {
        let jpg = synth_jpeg(26);
        for cut in [4, 10, 20, 30, 40] {
            let _ = read_jpeg_exif(&jpg[..cut.min(jpg.len())]);
        }
    }
}
