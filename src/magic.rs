use crate::bitboard::Bitboard;
use crate::types::Square;

// Well-tested magic numbers from Chess Programming Wiki / Stockfish
static ROOK_MAGICS: [u64; 64] = [
    0xa8002c000108020u64, 0x6c00049b0002001u64, 0x100200010090040u64, 0x2480041000800801u64,
    0x280028004000800u64, 0x900410008040022u64, 0x280020001001080u64, 0x2880002041000080u64,
    0xa000800080400034u64, 0x4808020004000u64, 0x2290802004801000u64, 0x411000d00100020u64,
    0x402800800040080u64, 0xb000401004208u64, 0x2409000100040200u64, 0x1002100004082u64,
    0x22878001e24000u64, 0x1090810021004010u64, 0x801030040200012u64, 0x500808008001000u64,
    0xa08018014000880u64, 0x8000808004000200u64, 0x201008080010200u64, 0x801020000441091u64,
    0x800080204005u64, 0x1040200040100048u64, 0x120200402082u64, 0xd14880480100080u64,
    0x12040280080080u64, 0x100040080020080u64, 0x9020010080800200u64, 0x813241200148449u64,
    0x491604001800080u64, 0x100401000402001u64, 0x4820010021001040u64, 0x400402202000812u64,
    0x209009005000802u64, 0x810800601800400u64, 0x4301083214000150u64, 0x204026458e001401u64,
    0x40204000808000u64, 0x8001008040010020u64, 0x8410820820420010u64, 0x1003001000090020u64,
    0x804040008008080u64, 0x12000810020004u64, 0x1000100200040208u64, 0x430000a044020001u64,
    0x280009023410300u64, 0xe0100040002240u64, 0x200100401700u64, 0x2244100408008080u64,
    0x8000400801980u64, 0x2000810040200u64, 0x8010100228810400u64, 0x2000009044210200u64,
    0x4080008040102101u64, 0x40002080411d01u64, 0x2005524060000901u64, 0x502001008400422u64,
    0x489a000810200402u64, 0x1004400080a13u64, 0x4000011008020084u64, 0x26002114058042u64,
];

static BISHOP_MAGICS: [u64; 64] = [
    0x89a1121896040240u64, 0x2004844802002010u64, 0x2068080051921000u64, 0x62880a0220200808u64,
    0x4042004000000u64, 0x100822020200011u64, 0xc00444222012000au64, 0x28808801216001u64,
    0x400492088408100u64, 0x201c401040c0084u64, 0x840800910a0010u64, 0x82080240060u64,
    0x2000840504006000u64, 0x30010c4108405004u64, 0x1008005410080802u64, 0x8144042209100900u64,
    0x208081020014400u64, 0x4800201208ca00u64, 0xf18140408012008u64, 0x1004002802102001u64,
    0x841000820080811u64, 0x40200200a42008u64, 0x800054042000u64, 0x88010400410c9000u64,
    0x520040470104290u64, 0x1004040051500081u64, 0x2002081833080021u64, 0x400c00c010142u64,
    0x941408200c002000u64, 0x658810000806011u64, 0x188071040440a00u64, 0x4800404002011c00u64,
    0x104442040404200u64, 0x511080202091021u64, 0x4022401120400u64, 0x80c0040400080120u64,
    0x8040010040820802u64, 0x480810700020090u64, 0x102008e00040242u64, 0x809005202050100u64,
    0x8002024220104080u64, 0x431008804142000u64, 0x19001802081400u64, 0x200014208040080u64,
    0x3308082008200100u64, 0x41010500040c020u64, 0x4012020c04210308u64, 0x208220a202004080u64,
    0x111040120082000u64, 0x6803040141280a00u64, 0x2101004202410000u64, 0x8200000041108022u64,
    0x21082088000u64, 0x2410204010040u64, 0x40100400809000u64, 0x822088220820214u64,
    0x40808090012004u64, 0x910224040218c9u64, 0x402814422015008u64, 0x90014004842410u64,
    0x1000042304105u64, 0x10008830412a00u64, 0x2520081090008908u64, 0x40102000a0a60140u64,
];

// Attack tables - computed at startup
// Rook: max 12 bits = 4096 entries per square, 64 squares = 262144 max
// Bishop: max 9 bits = 512 entries per square, 64 squares = 32768 max
// Total: ~300K entries is safe
static mut ROOK_ATTACKS: [[Bitboard; 4096]; 64] = [[Bitboard(0); 4096]; 64];
static mut BISHOP_ATTACKS: [[Bitboard; 512]; 64] = [[Bitboard(0); 512]; 64];
static mut ROOK_MASKS: [Bitboard; 64] = [Bitboard(0); 64];
static mut BISHOP_MASKS: [Bitboard; 64] = [Bitboard(0); 64];
static mut ROOK_SHIFTS: [u8; 64] = [0; 64];
static mut BISHOP_SHIFTS: [u8; 64] = [0; 64];

static INIT: std::sync::Once = std::sync::Once::new();

fn rook_mask(sq: Square) -> Bitboard {
    let rank = sq.rank();
    let file = sq.file();
    let mut mask = 0u64;

    // North (excluding edge)
    for r in (rank + 1)..7 {
        mask |= 1 << (r * 8 + file);
    }
    // South (excluding edge)
    for r in 1..rank {
        mask |= 1 << (r * 8 + file);
    }
    // East (excluding edge)
    for f in (file + 1)..7 {
        mask |= 1 << (rank * 8 + f);
    }
    // West (excluding edge)
    for f in 1..file {
        mask |= 1 << (rank * 8 + f);
    }

    Bitboard(mask)
}

fn bishop_mask(sq: Square) -> Bitboard {
    let rank = sq.rank() as i8;
    let file = sq.file() as i8;
    let mut mask = 0u64;

    // North-East
    let (mut r, mut f) = (rank + 1, file + 1);
    while r < 7 && f < 7 {
        mask |= 1 << (r * 8 + f);
        r += 1;
        f += 1;
    }
    // North-West
    let (mut r, mut f) = (rank + 1, file - 1);
    while r < 7 && f > 0 {
        mask |= 1 << (r * 8 + f);
        r += 1;
        f -= 1;
    }
    // South-East
    let (mut r, mut f) = (rank - 1, file + 1);
    while r > 0 && f < 7 {
        mask |= 1 << (r * 8 + f);
        r -= 1;
        f += 1;
    }
    // South-West
    let (mut r, mut f) = (rank - 1, file - 1);
    while r > 0 && f > 0 {
        mask |= 1 << (r * 8 + f);
        r -= 1;
        f -= 1;
    }

    Bitboard(mask)
}

fn rook_attacks_slow(sq: Square, occupied: Bitboard) -> Bitboard {
    let rank = sq.rank() as i8;
    let file = sq.file() as i8;
    let mut attacks = 0u64;

    // North
    for r in (rank + 1)..8 {
        let bit = 1 << (r * 8 + file);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
    }
    // South
    for r in (0..rank).rev() {
        let bit = 1 << (r * 8 + file);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
    }
    // East
    for f in (file + 1)..8 {
        let bit = 1 << (rank * 8 + f);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
    }
    // West
    for f in (0..file).rev() {
        let bit = 1 << (rank * 8 + f);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
    }

    Bitboard(attacks)
}

fn bishop_attacks_slow(sq: Square, occupied: Bitboard) -> Bitboard {
    let rank = sq.rank() as i8;
    let file = sq.file() as i8;
    let mut attacks = 0u64;

    // North-East
    let (mut r, mut f) = (rank + 1, file + 1);
    while r < 8 && f < 8 {
        let bit = 1 << (r * 8 + f);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
        r += 1;
        f += 1;
    }
    // North-West
    let (mut r, mut f) = (rank + 1, file - 1);
    while r < 8 && f >= 0 {
        let bit = 1 << (r * 8 + f);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
        r += 1;
        f -= 1;
    }
    // South-East
    let (mut r, mut f) = (rank - 1, file + 1);
    while r >= 0 && f < 8 {
        let bit = 1 << (r * 8 + f);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
        r -= 1;
        f += 1;
    }
    // South-West
    let (mut r, mut f) = (rank - 1, file - 1);
    while r >= 0 && f >= 0 {
        let bit = 1 << (r * 8 + f);
        attacks |= bit;
        if occupied.0 & bit != 0 {
            break;
        }
        r -= 1;
        f -= 1;
    }

    Bitboard(attacks)
}

fn index_to_occupancy(index: usize, mask: Bitboard) -> Bitboard {
    let mut occ = Bitboard::EMPTY;
    let mut mask = mask;
    let mut idx = index;

    while mask.is_not_empty() {
        let sq = mask.pop_lsb();
        if idx & 1 != 0 {
            occ |= Bitboard::from_sq(sq);
        }
        idx >>= 1;
    }

    occ
}

fn init_magics() {
    // Initialize rook magics
    for sq_idx in 0..64 {
        let sq = Square(sq_idx);
        let mask = rook_mask(sq);
        let bits = mask.count() as u8;
        let shift = 64 - bits;

        unsafe {
            ROOK_MASKS[sq_idx as usize] = mask;
            ROOK_SHIFTS[sq_idx as usize] = shift;

            let size = 1usize << bits;
            for i in 0..size {
                let occ = index_to_occupancy(i, mask);
                let attacks = rook_attacks_slow(sq, occ);
                let magic = ROOK_MAGICS[sq_idx as usize];
                let idx = ((occ.0.wrapping_mul(magic)) >> shift) as usize;
                ROOK_ATTACKS[sq_idx as usize][idx] = attacks;
            }
        }
    }

    // Initialize bishop magics
    for sq_idx in 0..64 {
        let sq = Square(sq_idx);
        let mask = bishop_mask(sq);
        let bits = mask.count() as u8;
        let shift = 64 - bits;

        unsafe {
            BISHOP_MASKS[sq_idx as usize] = mask;
            BISHOP_SHIFTS[sq_idx as usize] = shift;

            let size = 1usize << bits;
            for i in 0..size {
                let occ = index_to_occupancy(i, mask);
                let attacks = bishop_attacks_slow(sq, occ);
                let magic = BISHOP_MAGICS[sq_idx as usize];
                let idx = ((occ.0.wrapping_mul(magic)) >> shift) as usize;
                BISHOP_ATTACKS[sq_idx as usize][idx] = attacks;
            }
        }
    }
}

pub fn init() {
    INIT.call_once(|| {
        init_magics();
        init_knight_attacks();
        init_king_attacks();
        init_pawn_attacks();
    });
}

#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    unsafe {
        let mask = ROOK_MASKS[sq.index()];
        let magic = ROOK_MAGICS[sq.index()];
        let shift = ROOK_SHIFTS[sq.index()];
        let idx = (((occupied & mask).0.wrapping_mul(magic)) >> shift) as usize;
        ROOK_ATTACKS[sq.index()][idx]
    }
}

#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    unsafe {
        let mask = BISHOP_MASKS[sq.index()];
        let magic = BISHOP_MAGICS[sq.index()];
        let shift = BISHOP_SHIFTS[sq.index()];
        let idx = (((occupied & mask).0.wrapping_mul(magic)) >> shift) as usize;
        BISHOP_ATTACKS[sq.index()][idx]
    }
}

#[inline]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}

// Pre-computed knight and king attacks (non-sliding pieces)
static mut KNIGHT_ATTACKS: [Bitboard; 64] = [Bitboard(0); 64];
static mut KING_ATTACKS: [Bitboard; 64] = [Bitboard(0); 64];
static mut PAWN_ATTACKS: [[Bitboard; 64]; 2] = [[Bitboard(0); 64]; 2];

fn init_knight_attacks() {
    for sq_idx in 0..64 {
        let sq = Square(sq_idx);
        let bb = Bitboard::from_sq(sq);
        let mut attacks = Bitboard::EMPTY;

        // 8 knight move directions
        attacks |= Bitboard((bb.0 << 17) & !Bitboard::FILE_A.0); // NNE
        attacks |= Bitboard((bb.0 << 15) & !Bitboard::FILE_H.0); // NNW
        attacks |= Bitboard((bb.0 << 10) & Bitboard::NOT_FILE_AB.0); // NEE
        attacks |= Bitboard((bb.0 << 6) & Bitboard::NOT_FILE_GH.0); // NWW
        attacks |= Bitboard((bb.0 >> 6) & Bitboard::NOT_FILE_AB.0); // SEE
        attacks |= Bitboard((bb.0 >> 10) & Bitboard::NOT_FILE_GH.0); // SWW
        attacks |= Bitboard((bb.0 >> 15) & !Bitboard::FILE_A.0); // SSE
        attacks |= Bitboard((bb.0 >> 17) & !Bitboard::FILE_H.0); // SSW

        unsafe {
            KNIGHT_ATTACKS[sq_idx as usize] = attacks;
        }
    }
}

fn init_king_attacks() {
    for sq_idx in 0..64 {
        let sq = Square(sq_idx);
        let bb = Bitboard::from_sq(sq);
        let mut attacks = Bitboard::EMPTY;

        attacks |= bb.north();
        attacks |= bb.south();
        attacks |= bb.east();
        attacks |= bb.west();
        attacks |= bb.north_east();
        attacks |= bb.north_west();
        attacks |= bb.south_east();
        attacks |= bb.south_west();

        unsafe {
            KING_ATTACKS[sq_idx as usize] = attacks;
        }
    }
}

fn init_pawn_attacks() {
    for sq_idx in 0..64 {
        let sq = Square(sq_idx);
        let bb = Bitboard::from_sq(sq);

        // White pawn attacks (moving north)
        unsafe {
            PAWN_ATTACKS[0][sq_idx as usize] = bb.north_east() | bb.north_west();
        }

        // Black pawn attacks (moving south)
        unsafe {
            PAWN_ATTACKS[1][sq_idx as usize] = bb.south_east() | bb.south_west();
        }
    }
}

pub fn init_all() {
    init();
}

#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    unsafe { KNIGHT_ATTACKS[sq.index()] }
}

#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    unsafe { KING_ATTACKS[sq.index()] }
}

#[inline]
pub fn pawn_attacks(sq: Square, color: crate::types::Color) -> Bitboard {
    unsafe { PAWN_ATTACKS[color.index()][sq.index()] }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        init_all();
    }

    #[test]
    fn test_rook_attacks_empty_board() {
        setup();
        let attacks = rook_attacks(Square::E4, Bitboard::EMPTY);
        // E4 rook should attack e1-e8 (minus e4) and a4-h4 (minus e4) = 14 squares
        assert_eq!(attacks.count(), 14);
        assert!(attacks.contains(Square::E1));
        assert!(attacks.contains(Square::E8));
        assert!(attacks.contains(Square::A4));
        assert!(attacks.contains(Square::H4));
        assert!(!attacks.contains(Square::E4)); // Not self
    }

    #[test]
    fn test_rook_attacks_blocked() {
        setup();
        let blockers = Bitboard::from_sq(Square::E6);
        let attacks = rook_attacks(Square::E4, blockers);
        assert!(attacks.contains(Square::E6)); // Can capture blocker
        assert!(!attacks.contains(Square::E8)); // Blocked
    }

    #[test]
    fn test_bishop_attacks_empty_board() {
        setup();
        let attacks = bishop_attacks(Square::E4, Bitboard::EMPTY);
        // E4 bishop attacks diagonals
        assert!(attacks.contains(Square::D3));
        assert!(attacks.contains(Square::F5));
        assert!(attacks.contains(Square::H1));
        assert!(attacks.contains(Square::A8));
    }

    #[test]
    fn test_knight_attacks() {
        setup();
        let attacks = knight_attacks(Square::E4);
        assert_eq!(attacks.count(), 8);
        assert!(attacks.contains(Square::D6));
        assert!(attacks.contains(Square::F6));
        assert!(attacks.contains(Square::G5));
        assert!(attacks.contains(Square::G3));
    }

    #[test]
    fn test_knight_attacks_corner() {
        setup();
        let attacks = knight_attacks(Square::A1);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::C2));
        assert!(attacks.contains(Square::B3));
    }

    #[test]
    fn test_king_attacks() {
        setup();
        let attacks = king_attacks(Square::E4);
        assert_eq!(attacks.count(), 8);
    }

    #[test]
    fn test_king_attacks_corner() {
        setup();
        let attacks = king_attacks(Square::A1);
        assert_eq!(attacks.count(), 3);
    }

    #[test]
    fn test_pawn_attacks() {
        setup();
        use crate::types::Color;

        let white_attacks = pawn_attacks(Square::E4, Color::White);
        assert_eq!(white_attacks.count(), 2);
        assert!(white_attacks.contains(Square::D5));
        assert!(white_attacks.contains(Square::F5));

        let black_attacks = pawn_attacks(Square::E4, Color::Black);
        assert_eq!(black_attacks.count(), 2);
        assert!(black_attacks.contains(Square::D3));
        assert!(black_attacks.contains(Square::F3));
    }
}
