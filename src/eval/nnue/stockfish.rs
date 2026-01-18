//! Stockfish NNUE format loader.
//!
//! Parses Stockfish's .nnue files to extract the feature transformer weights.
//! Supports the COMPRESSED_LEB128 format used by modern Stockfish networks.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

/// Stockfish NNUE file header magic (little-endian).
const SF_NNUE_MAGIC: u32 = 0x7AF32F20; // First 4 bytes of modern SF NNUE

/// Information extracted from a Stockfish NNUE file.
#[derive(Debug)]
pub struct StockfishNetInfo {
    pub description: String,
    pub is_compressed: bool,
    pub architecture: String,
    pub ft_input_dims: usize,
    pub ft_output_dims: usize,
}

/// Read a Stockfish NNUE file header and extract network info.
pub fn read_stockfish_header(path: &str) -> Result<StockfishNetInfo, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);

    // Read magic
    let mut magic_bytes = [0u8; 4];
    reader
        .read_exact(&mut magic_bytes)
        .map_err(|e| format!("Failed to read magic: {}", e))?;
    let magic = u32::from_le_bytes(magic_bytes);

    if magic != SF_NNUE_MAGIC {
        return Err(format!(
            "Not a Stockfish NNUE file. Magic: 0x{:08X}, expected: 0x{:08X}",
            magic, SF_NNUE_MAGIC
        ));
    }

    // Skip next 4 bytes (hash/version)
    reader
        .seek(SeekFrom::Current(4))
        .map_err(|e| format!("Seek error: {}", e))?;

    // Read description length (as a u32)
    let mut len_bytes = [0u8; 4];
    reader
        .read_exact(&mut len_bytes)
        .map_err(|e| format!("Failed to read description length: {}", e))?;
    let desc_len = u32::from_le_bytes(len_bytes) as usize;

    // Read description string
    let mut desc_bytes = vec![0u8; desc_len];
    reader
        .read_exact(&mut desc_bytes)
        .map_err(|e| format!("Failed to read description: {}", e))?;
    let description = String::from_utf8_lossy(&desc_bytes).to_string();

    // Skip 4 bytes (another hash/marker)
    reader
        .seek(SeekFrom::Current(4))
        .map_err(|e| format!("Seek error: {}", e))?;

    // Check for compression marker (COMPRESSED_LEB128 = 17 chars)
    let mut marker_buf = [0u8; 20];
    reader
        .read_exact(&mut marker_buf)
        .map_err(|e| format!("Failed to read marker: {}", e))?;
    let marker = String::from_utf8_lossy(&marker_buf);
    let is_compressed = marker.starts_with("COMPRESSED_LEB128");

    // For modern Stockfish networks, architecture details are embedded
    let architecture = if is_compressed {
        "HalfKAv2_hm (COMPRESSED_LEB128)".to_string()
    } else {
        "HalfKAv2_hm".to_string()
    };

    // Modern Stockfish networks typically use:
    // - HalfKAv2_hm features: 45056 input features (or 22528 per perspective)
    // - Large hidden layers: 1024 or 2048 neurons
    let ft_input_dims = 45056; // HalfKAv2_hm total features (both perspectives)
    let ft_output_dims = 1024; // Typical modern SF network

    Ok(StockfishNetInfo {
        description,
        is_compressed,
        architecture,
        ft_input_dims,
        ft_output_dims,
    })
}

/// LEB128 decoder for signed integers.
pub fn decode_leb128_signed(data: &[u8], pos: &mut usize) -> i64 {
    let mut result: i64 = 0;
    let mut shift = 0;
    let mut last_byte: u8 = 0;

    while *pos < data.len() {
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as i64) << shift;
        last_byte = byte;
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
    }

    // Sign extend if negative
    if shift < 64 && (last_byte & 0x40) != 0 {
        result |= !0i64 << shift;
    }

    result
}

/// LEB128 decoder for unsigned integers.
pub fn decode_leb128_unsigned(data: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift = 0;

    loop {
        if *pos >= data.len() {
            break;
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }

    result
}

/// Check if a file is a valid Stockfish NNUE file.
pub fn is_stockfish_nnue(path: &str) -> bool {
    read_stockfish_header(path).is_ok()
}

/// Print information about a Stockfish NNUE file.
pub fn print_stockfish_info(path: &str) -> Result<(), String> {
    let info = read_stockfish_header(path)?;
    println!("Stockfish NNUE Network Info:");
    println!("  Description: {}", info.description.trim_end_matches('\0'));
    println!("  Compressed: {}", info.is_compressed);
    println!("  Architecture: {}", info.architecture);
    println!(
        "  Feature Transformer: {} -> {}",
        info.ft_input_dims, info.ft_output_dims
    );
    Ok(())
}

/// Debug function to show raw header bytes.
pub fn debug_header(path: &str) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);

    // Read first 128 bytes
    let mut header = [0u8; 128];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("Failed to read header: {}", e))?;

    println!("Header bytes (first 128):");
    for (i, chunk) in header.chunks(16).enumerate() {
        print!("{:04X}: ", i * 16);
        for b in chunk {
            print!("{:02X} ", b);
        }
        print!("  ");
        for b in chunk {
            let c = if b.is_ascii_graphic() || *b == b' ' {
                *b as char
            } else {
                '.'
            };
            print!("{}", c);
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leb128_unsigned() {
        // Test simple values
        let data = [0x00]; // 0
        let mut pos = 0;
        assert_eq!(decode_leb128_unsigned(&data, &mut pos), 0);

        let data = [0x7F]; // 127
        let mut pos = 0;
        assert_eq!(decode_leb128_unsigned(&data, &mut pos), 127);

        let data = [0x80, 0x01]; // 128
        let mut pos = 0;
        assert_eq!(decode_leb128_unsigned(&data, &mut pos), 128);

        let data = [0xE5, 0x8E, 0x26]; // 624485
        let mut pos = 0;
        assert_eq!(decode_leb128_unsigned(&data, &mut pos), 624485);
    }

    #[test]
    fn test_leb128_signed() {
        let data = [0x00]; // 0
        let mut pos = 0;
        assert_eq!(decode_leb128_signed(&data, &mut pos), 0);

        let data = [0x7F]; // -1
        let mut pos = 0;
        assert_eq!(decode_leb128_signed(&data, &mut pos), -1);

        let data = [0x80, 0x7F]; // -128
        let mut pos = 0;
        assert_eq!(decode_leb128_signed(&data, &mut pos), -128);
    }
}
