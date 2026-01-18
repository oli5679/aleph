use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const ALL: [Piece; 6] = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];

    pub fn from_char(c: char) -> Option<Piece> {
        match c.to_ascii_lowercase() {
            'p' => Some(Piece::Pawn),
            'n' => Some(Piece::Knight),
            'b' => Some(Piece::Bishop),
            'r' => Some(Piece::Rook),
            'q' => Some(Piece::Queen),
            'k' => Some(Piece::King),
            _ => None,
        }
    }

    pub fn to_char(self, color: Color) -> char {
        let c = match self {
            Piece::Pawn => 'p',
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            Piece::King => 'k',
        };
        if color == Color::White {
            c.to_ascii_uppercase()
        } else {
            c
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Default)]
#[repr(transparent)]
pub struct Square(pub u8);

impl Square {
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);
    pub const A2: Square = Square(8);
    pub const B2: Square = Square(9);
    pub const C2: Square = Square(10);
    pub const D2: Square = Square(11);
    pub const E2: Square = Square(12);
    pub const F2: Square = Square(13);
    pub const G2: Square = Square(14);
    pub const H2: Square = Square(15);
    pub const A3: Square = Square(16);
    pub const B3: Square = Square(17);
    pub const C3: Square = Square(18);
    pub const D3: Square = Square(19);
    pub const E3: Square = Square(20);
    pub const F3: Square = Square(21);
    pub const G3: Square = Square(22);
    pub const H3: Square = Square(23);
    pub const A4: Square = Square(24);
    pub const B4: Square = Square(25);
    pub const C4: Square = Square(26);
    pub const D4: Square = Square(27);
    pub const E4: Square = Square(28);
    pub const F4: Square = Square(29);
    pub const G4: Square = Square(30);
    pub const H4: Square = Square(31);
    pub const A5: Square = Square(32);
    pub const B5: Square = Square(33);
    pub const C5: Square = Square(34);
    pub const D5: Square = Square(35);
    pub const E5: Square = Square(36);
    pub const F5: Square = Square(37);
    pub const G5: Square = Square(38);
    pub const H5: Square = Square(39);
    pub const A6: Square = Square(40);
    pub const B6: Square = Square(41);
    pub const C6: Square = Square(42);
    pub const D6: Square = Square(43);
    pub const E6: Square = Square(44);
    pub const F6: Square = Square(45);
    pub const G6: Square = Square(46);
    pub const H6: Square = Square(47);
    pub const A7: Square = Square(48);
    pub const B7: Square = Square(49);
    pub const C7: Square = Square(50);
    pub const D7: Square = Square(51);
    pub const E7: Square = Square(52);
    pub const F7: Square = Square(53);
    pub const G7: Square = Square(54);
    pub const H7: Square = Square(55);
    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    #[inline]
    pub const fn new(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8);
        Self(rank * 8 + file)
    }

    #[inline]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn flip_rank(self) -> Self {
        Self(self.0 ^ 56)
    }

    pub fn from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Self::new(file, rank))
        } else {
            None
        }
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{}{}", file, rank)
    }
}

/// 16-bit move encoding
/// Bits 0-5:   from square
/// Bits 6-11:  to square
/// Bits 12-13: promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
/// Bits 14-15: flags (0=normal, 1=promotion, 2=en passant, 3=castling)
#[derive(Copy, Clone, Eq, PartialEq, Default)]
#[repr(transparent)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Self = Self(0);

    const FROM_MASK: u16 = 0x3F;
    const TO_MASK: u16 = 0x3F << 6;
    const PROMO_MASK: u16 = 0x3 << 12;
    const FLAG_MASK: u16 = 0x3 << 14;

    pub const FLAG_NORMAL: u16 = 0;
    pub const FLAG_PROMOTION: u16 = 1 << 14;
    pub const FLAG_EP: u16 = 2 << 14;
    pub const FLAG_CASTLE: u16 = 3 << 14;

    #[inline]
    pub const fn new(from: Square, to: Square) -> Self {
        Self(from.0 as u16 | ((to.0 as u16) << 6))
    }

    #[inline]
    pub const fn new_promotion(from: Square, to: Square, promo: Piece) -> Self {
        let promo_bits = match promo {
            Piece::Knight => 0,
            Piece::Bishop => 1,
            Piece::Rook => 2,
            Piece::Queen => 3,
            _ => 0,
        };
        Self(from.0 as u16 | ((to.0 as u16) << 6) | (promo_bits << 12) | Self::FLAG_PROMOTION)
    }

    #[inline]
    pub const fn new_ep(from: Square, to: Square) -> Self {
        Self(from.0 as u16 | ((to.0 as u16) << 6) | Self::FLAG_EP)
    }

    #[inline]
    pub const fn new_castle(from: Square, to: Square) -> Self {
        Self(from.0 as u16 | ((to.0 as u16) << 6) | Self::FLAG_CASTLE)
    }

    #[inline]
    pub const fn from(self) -> Square {
        Square((self.0 & Self::FROM_MASK) as u8)
    }

    #[inline]
    pub const fn to(self) -> Square {
        Square(((self.0 & Self::TO_MASK) >> 6) as u8)
    }

    #[inline]
    pub const fn promotion_piece(self) -> Piece {
        match (self.0 & Self::PROMO_MASK) >> 12 {
            0 => Piece::Knight,
            1 => Piece::Bishop,
            2 => Piece::Rook,
            _ => Piece::Queen,
        }
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_promotion(self) -> bool {
        (self.0 & Self::FLAG_MASK) == Self::FLAG_PROMOTION
    }

    #[inline]
    pub const fn is_ep(self) -> bool {
        (self.0 & Self::FLAG_MASK) == Self::FLAG_EP
    }

    #[inline]
    pub const fn is_castle(self) -> bool {
        (self.0 & Self::FLAG_MASK) == Self::FLAG_CASTLE
    }

    pub fn from_uci(s: &str) -> Option<Self> {
        if s.len() < 4 || s.len() > 5 {
            return None;
        }
        let from = Square::from_str(&s[0..2])?;
        let to = Square::from_str(&s[2..4])?;
        if s.len() == 5 {
            let promo = match s.chars().nth(4)? {
                'n' => Piece::Knight,
                'b' => Piece::Bishop,
                'r' => Piece::Rook,
                'q' => Piece::Queen,
                _ => return None,
            };
            Some(Self::new_promotion(from, to, promo))
        } else {
            Some(Self::new(from, to))
        }
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if self.is_promotion() {
            let c = match self.promotion_piece() {
                Piece::Knight => 'n',
                Piece::Bishop => 'b',
                Piece::Rook => 'r',
                Piece::Queen => 'q',
                _ => '?',
            };
            write!(f, "{}", c)?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Default, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const NONE: Self = Self(0);
    pub const WK: u8 = 1;
    pub const WQ: u8 = 2;
    pub const BK: u8 = 4;
    pub const BQ: u8 = 8;
    pub const ALL: Self = Self(Self::WK | Self::WQ | Self::BK | Self::BQ);

    #[inline]
    pub const fn has(self, right: u8) -> bool {
        (self.0 & right) != 0
    }

    #[inline]
    pub const fn remove(self, right: u8) -> Self {
        Self(self.0 & !right)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square() {
        assert_eq!(Square::new(0, 0), Square::A1);
        assert_eq!(Square::new(7, 7), Square::H8);
        assert_eq!(Square::A1.file(), 0);
        assert_eq!(Square::A1.rank(), 0);
        assert_eq!(Square::H8.file(), 7);
        assert_eq!(Square::H8.rank(), 7);
        assert_eq!(format!("{}", Square::E4), "e4");
        assert_eq!(Square::from_str("e4"), Some(Square::new(4, 3)));
    }

    #[test]
    fn test_move() {
        let mv = Move::new(Square::E2, Square::E4);
        assert_eq!(mv.from(), Square::E2);
        assert_eq!(mv.to(), Square::E4);
        assert!(!mv.is_promotion());
        assert_eq!(format!("{}", mv), "e2e4");

        let promo = Move::new_promotion(Square::E7, Square::E8, Piece::Queen);
        assert!(promo.is_promotion());
        assert_eq!(promo.promotion_piece(), Piece::Queen);
        assert_eq!(format!("{}", promo), "e7e8q");
    }

    #[test]
    fn test_color() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }
}
