use crate::types::Square;
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

#[derive(Copy, Clone, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const ALL: Self = Self(!0);

    pub const FILE_A: Self = Self(0x0101_0101_0101_0101);
    pub const FILE_B: Self = Self(0x0202_0202_0202_0202);
    pub const FILE_G: Self = Self(0x4040_4040_4040_4040);
    pub const FILE_H: Self = Self(0x8080_8080_8080_8080);

    pub const RANK_1: Self = Self(0x0000_0000_0000_00FF);
    pub const RANK_2: Self = Self(0x0000_0000_0000_FF00);
    pub const RANK_3: Self = Self(0x0000_0000_00FF_0000);
    pub const RANK_4: Self = Self(0x0000_0000_FF00_0000);
    pub const RANK_5: Self = Self(0x0000_00FF_0000_0000);
    pub const RANK_6: Self = Self(0x0000_FF00_0000_0000);
    pub const RANK_7: Self = Self(0x00FF_0000_0000_0000);
    pub const RANK_8: Self = Self(0xFF00_0000_0000_0000);

    pub const NOT_FILE_A: Self = Self(!Self::FILE_A.0);
    pub const NOT_FILE_H: Self = Self(!Self::FILE_H.0);
    pub const NOT_FILE_AB: Self = Self(!(Self::FILE_A.0 | Self::FILE_B.0));
    pub const NOT_FILE_GH: Self = Self(!(Self::FILE_G.0 | Self::FILE_H.0));

    #[inline]
    pub const fn from_sq(sq: Square) -> Self {
        Self(1 << sq.0)
    }

    #[inline]
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 & (1 << sq.0)) != 0
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(self.0.trailing_zeros() as u8)
    }

    #[inline]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1;
        sq
    }

    #[inline]
    pub const fn north(self) -> Self {
        Self(self.0 << 8)
    }

    #[inline]
    pub const fn south(self) -> Self {
        Self(self.0 >> 8)
    }

    #[inline]
    pub const fn east(self) -> Self {
        Self((self.0 << 1) & Self::NOT_FILE_A.0)
    }

    #[inline]
    pub const fn west(self) -> Self {
        Self((self.0 >> 1) & Self::NOT_FILE_H.0)
    }

    #[inline]
    pub const fn north_east(self) -> Self {
        Self((self.0 << 9) & Self::NOT_FILE_A.0)
    }

    #[inline]
    pub const fn north_west(self) -> Self {
        Self((self.0 << 7) & Self::NOT_FILE_H.0)
    }

    #[inline]
    pub const fn south_east(self) -> Self {
        Self((self.0 >> 7) & Self::NOT_FILE_A.0)
    }

    #[inline]
    pub const fn south_west(self) -> Self {
        Self((self.0 >> 9) & Self::NOT_FILE_H.0)
    }

    pub const fn file(f: u8) -> Self {
        Self(Self::FILE_A.0 << f)
    }

    pub const fn rank(r: u8) -> Self {
        Self(Self::RANK_1.0 << (r * 8))
    }
}

impl BitAnd for Bitboard {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl Shl<u8> for Bitboard {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: u8) -> Self {
        Self(self.0 << rhs)
    }
}

impl Shr<u8> for Bitboard {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: u8) -> Self {
        Self(self.0 >> rhs)
    }
}

impl Iterator for Bitboard {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        if self.is_empty() {
            None
        } else {
            Some(self.pop_lsb())
        }
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            write!(f, "  {} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                if self.contains(sq) {
                    write!(f, "X ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "    a b c d e f g h")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_sq() {
        let bb = Bitboard::from_sq(Square::E4);
        assert!(bb.contains(Square::E4));
        assert!(!bb.contains(Square::E5));
        assert_eq!(bb.count(), 1);
    }

    #[test]
    fn test_shifts() {
        let bb = Bitboard::from_sq(Square::E4);
        assert!(bb.north().contains(Square::new(4, 4))); // e5
        assert!(bb.south().contains(Square::new(4, 2))); // e3
        assert!(bb.east().contains(Square::new(5, 3)));  // f4
        assert!(bb.west().contains(Square::new(3, 3)));  // d4
    }

    #[test]
    fn test_edge_shifts() {
        let bb = Bitboard::from_sq(Square::A1);
        assert!(bb.west().is_empty());
        assert!(bb.south().is_empty());

        let bb = Bitboard::from_sq(Square::H8);
        assert!(bb.east().is_empty());
        assert!(bb.north().is_empty());
    }

    #[test]
    fn test_iterator() {
        let bb = Bitboard::from_sq(Square::A1) | Bitboard::from_sq(Square::H8);
        let squares: Vec<_> = bb.collect();
        assert_eq!(squares.len(), 2);
        assert!(squares.contains(&Square::A1));
        assert!(squares.contains(&Square::H8));
    }

    #[test]
    fn test_pop_lsb() {
        let mut bb = Bitboard::from_sq(Square::A1) | Bitboard::from_sq(Square::H8);
        let first = bb.pop_lsb();
        assert_eq!(first, Square::A1);
        let second = bb.pop_lsb();
        assert_eq!(second, Square::H8);
        assert!(bb.is_empty());
    }
}
