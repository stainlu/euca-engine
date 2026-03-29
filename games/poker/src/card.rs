//! Card primitives: [`Suit`], [`Rank`], and [`Card`].

use serde::{Deserialize, Serialize};

/// The four suits of a standard 52-card deck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

impl Suit {
    /// All four suits in ascending order.
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
}

/// Card rank from Two (lowest) through Ace (highest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Rank {
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
    Ten = 10,
    Jack = 11,
    Queen = 12,
    King = 13,
    Ace = 14,
}

impl Rank {
    /// All thirteen ranks in ascending order.
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    /// Convert a numeric value (2..=14) to a `Rank`, returning `None` for invalid values.
    pub fn from_u8(val: u8) -> Option<Rank> {
        match val {
            2 => Some(Rank::Two),
            3 => Some(Rank::Three),
            4 => Some(Rank::Four),
            5 => Some(Rank::Five),
            6 => Some(Rank::Six),
            7 => Some(Rank::Seven),
            8 => Some(Rank::Eight),
            9 => Some(Rank::Nine),
            10 => Some(Rank::Ten),
            11 => Some(Rank::Jack),
            12 => Some(Rank::Queen),
            13 => Some(Rank::King),
            14 => Some(Rank::Ace),
            _ => None,
        }
    }

    /// Numeric value of this rank (2..=14).
    pub fn value(self) -> u8 {
        self as u8
    }
}

/// A playing card with a [`Rank`] and a [`Suit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub fn new(rank: Rank, suit: Suit) -> Self {
        Self { rank, suit }
    }
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rank = match self.rank {
            Rank::Two => "2",
            Rank::Three => "3",
            Rank::Four => "4",
            Rank::Five => "5",
            Rank::Six => "6",
            Rank::Seven => "7",
            Rank::Eight => "8",
            Rank::Nine => "9",
            Rank::Ten => "T",
            Rank::Jack => "J",
            Rank::Queen => "Q",
            Rank::King => "K",
            Rank::Ace => "A",
        };
        let suit = match self.suit {
            Suit::Clubs => "\u{2663}",
            Suit::Diamonds => "\u{2666}",
            Suit::Hearts => "\u{2665}",
            Suit::Spades => "\u{2660}",
        };
        write!(f, "{rank}{suit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn all_52_cards_have_unique_display() {
        let mut seen = HashSet::new();
        for &suit in &Suit::ALL {
            for &rank in &Rank::ALL {
                let card = Card::new(rank, suit);
                let display = card.to_string();
                assert!(
                    seen.insert(display.clone()),
                    "duplicate display string: {display}"
                );
            }
        }
        assert_eq!(seen.len(), 52);
    }

    #[test]
    fn rank_ordering_is_correct() {
        for window in Rank::ALL.windows(2) {
            assert!(
                window[0] < window[1],
                "{:?} should be less than {:?}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn rank_from_u8_roundtrip() {
        for &rank in &Rank::ALL {
            assert_eq!(Rank::from_u8(rank.value()), Some(rank));
        }
        assert_eq!(Rank::from_u8(0), None);
        assert_eq!(Rank::from_u8(1), None);
        assert_eq!(Rank::from_u8(15), None);
    }
}
