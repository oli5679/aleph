//! NNUE file loader.
//!
//! Supports loading custom .nnue files for the dual-head architecture.
//! The format uses LEB128 encoding for compact storage.

use std::fs::File;
use std::io::{BufReader, Read};

use super::features::HALF_DIMENSIONS;
use super::network::{
    FeatureTransformer, Network, PolicyHead, ValueHead, L1_SIZE, NUM_QUANTILES, POLICY_OUTPUT,
    VALUE_HIDDEN1, VALUE_HIDDEN2,
};

/// Magic number for our custom NNUE format.
const NNUE_MAGIC: u32 = 0x414C4550; // "ALEP" in little-endian

/// Version number.
const NNUE_VERSION: u32 = 1;

/// Load a network from a file.
pub fn load_network(path: &str) -> Result<Network, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);

    // Read and verify magic number
    let magic = read_u32(&mut reader)?;
    if magic != NNUE_MAGIC {
        return Err(format!(
            "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
            NNUE_MAGIC, magic
        ));
    }

    // Read and verify version
    let version = read_u32(&mut reader)?;
    if version != NNUE_VERSION {
        return Err(format!(
            "Unsupported version: expected {}, got {}",
            NNUE_VERSION, version
        ));
    }

    // Read architecture parameters
    let half_dims = read_u32(&mut reader)? as usize;
    let l1_size = read_u32(&mut reader)? as usize;

    if half_dims != HALF_DIMENSIONS {
        return Err(format!(
            "Incompatible half dimensions: expected {}, got {}",
            HALF_DIMENSIONS, half_dims
        ));
    }
    if l1_size != L1_SIZE {
        return Err(format!(
            "Incompatible L1 size: expected {}, got {}",
            L1_SIZE, l1_size
        ));
    }

    // Read feature transformer
    let ft = read_feature_transformer(&mut reader)?;

    // Read value head
    let vh = read_value_head(&mut reader)?;

    // Read policy head
    let ph = read_policy_head(&mut reader)?;

    Ok(Network {
        feature_transformer: ft,
        value_head: vh,
        policy_head: ph,
    })
}

/// Read feature transformer weights.
fn read_feature_transformer(reader: &mut BufReader<File>) -> Result<FeatureTransformer, String> {
    let mut ft = FeatureTransformer::default();

    // Read biases
    for i in 0..L1_SIZE {
        ft.biases[i] = read_i16(reader)?;
    }

    // Read weights (stored column-major)
    for i in 0..HALF_DIMENSIONS * L1_SIZE {
        ft.weights[i] = read_i16(reader)?;
    }

    Ok(ft)
}

/// Read value head weights.
fn read_value_head(reader: &mut BufReader<File>) -> Result<ValueHead, String> {
    let mut vh = ValueHead::default();

    // Layer 1
    for i in 0..VALUE_HIDDEN1 {
        vh.b1[i] = read_i32(reader)?;
    }
    for i in 0..L1_SIZE * 2 * VALUE_HIDDEN1 {
        vh.w1[i] = read_i8(reader)?;
    }

    // Layer 2
    for i in 0..VALUE_HIDDEN2 {
        vh.b2[i] = read_i32(reader)?;
    }
    for i in 0..VALUE_HIDDEN1 * VALUE_HIDDEN2 {
        vh.w2[i] = read_i8(reader)?;
    }

    // Output layer
    for i in 0..NUM_QUANTILES {
        vh.b3[i] = read_i32(reader)?;
    }
    for i in 0..VALUE_HIDDEN2 * NUM_QUANTILES {
        vh.w3[i] = read_i8(reader)?;
    }

    Ok(vh)
}

/// Read policy head weights.
fn read_policy_head(reader: &mut BufReader<File>) -> Result<PolicyHead, String> {
    let mut ph = PolicyHead::default();

    // Biases
    for i in 0..POLICY_OUTPUT {
        ph.biases[i] = read_i32(reader)?;
    }

    // Weights
    for i in 0..L1_SIZE * 2 * POLICY_OUTPUT {
        ph.weights[i] = read_i8(reader)?;
    }

    Ok(ph)
}

/// Read a little-endian u32.
fn read_u32(reader: &mut BufReader<File>) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read u32: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian i32.
fn read_i32(reader: &mut BufReader<File>) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read i32: {}", e))?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a little-endian i16.
fn read_i16(reader: &mut BufReader<File>) -> Result<i16, String> {
    let mut buf = [0u8; 2];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read i16: {}", e))?;
    Ok(i16::from_le_bytes(buf))
}

/// Read an i8.
fn read_i8(reader: &mut BufReader<File>) -> Result<i8, String> {
    let mut buf = [0u8; 1];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read i8: {}", e))?;
    Ok(buf[0] as i8)
}

/// Save a network to a file (for training output).
pub fn save_network(network: &Network, path: &str) -> Result<(), String> {
    use std::io::Write;

    let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;

    // Write magic and version
    file.write_all(&NNUE_MAGIC.to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    file.write_all(&NNUE_VERSION.to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))?;

    // Write architecture parameters
    file.write_all(&(HALF_DIMENSIONS as u32).to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    file.write_all(&(L1_SIZE as u32).to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))?;

    // Write feature transformer
    for &b in &network.feature_transformer.biases {
        file.write_all(&b.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
    }
    for &w in &network.feature_transformer.weights {
        file.write_all(&w.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
    }

    // Write value head
    for &b in &network.value_head.b1 {
        file.write_all(&b.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
    }
    for &w in &network.value_head.w1 {
        file.write_all(&[w as u8])
            .map_err(|e| format!("Write error: {}", e))?;
    }
    for &b in &network.value_head.b2 {
        file.write_all(&b.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
    }
    for &w in &network.value_head.w2 {
        file.write_all(&[w as u8])
            .map_err(|e| format!("Write error: {}", e))?;
    }
    for &b in &network.value_head.b3 {
        file.write_all(&b.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
    }
    for &w in &network.value_head.w3 {
        file.write_all(&[w as u8])
            .map_err(|e| format!("Write error: {}", e))?;
    }

    // Write policy head
    for &b in &network.policy_head.biases {
        file.write_all(&b.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
    }
    for &w in &network.policy_head.weights {
        file.write_all(&[w as u8])
            .map_err(|e| format!("Write error: {}", e))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_save_load_roundtrip() {
        let network = Network::new_random();
        let path = "/tmp/test_aleph_nnue.bin";

        // Save
        save_network(&network, path).expect("Failed to save");

        // Load
        let loaded = load_network(path).expect("Failed to load");

        // Verify
        assert_eq!(
            network.feature_transformer.biases,
            loaded.feature_transformer.biases
        );
        assert_eq!(
            network.feature_transformer.weights.len(),
            loaded.feature_transformer.weights.len()
        );
        assert_eq!(network.value_head.b1, loaded.value_head.b1);
        assert_eq!(network.policy_head.biases, loaded.policy_head.biases);

        // Cleanup
        fs::remove_file(path).ok();
    }
}
