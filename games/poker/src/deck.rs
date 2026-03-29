//! Standard 52-card deck with shuffle and deal operations.

use crate::card::{Card, Rank, Suit};
use rand::seq::SliceRandom;

/// A standard 52-card deck that tracks a deal position.
#[derive(Debug, Clone)]
pub struct Deck {
    cards: Vec<Card>,
    position: usize,
}

impl Deck {
    /// Create a new unshuffled deck in canonical order.
    pub fn new() -> Self {
        let mut cards = Vec::with_capacity(52);
        for &suit in &Suit::ALL {
            for &rank in &Rank::ALL {
                cards.push(Card::new(rank, suit));
            }
        }
        Self { cards, position: 0 }
    }

    /// Shuffle the deck and reset the deal position.
    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        self.cards.shuffle(rng);
        self.position = 0;
    }

    /// Deal the next card from the deck, or `None` if exhausted.
    pub fn deal(&mut self) -> Option<Card> {
        if self.position < self.cards.len() {
            let card = self.cards[self.position];
            self.position += 1;
            Some(card)
        } else {
            None
        }
    }

    /// Number of cards remaining to be dealt.
    pub fn remaining(&self) -> usize {
        self.cards.len() - self.position
    }
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn new_deck_has_52_cards() {
        let deck = Deck::new();
        assert_eq!(deck.remaining(), 52);
    }

    #[test]
    fn all_cards_are_unique() {
        let mut deck = Deck::new();
        let mut seen = HashSet::new();
        while let Some(card) = deck.deal() {
            assert!(seen.insert(card), "duplicate card: {card}");
        }
        assert_eq!(seen.len(), 52);
    }

    #[test]
    fn shuffle_produces_different_order() {
        let unshuffled = Deck::new();

        let mut shuffled = Deck::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        shuffled.shuffle(&mut rng);

        // Collect both orderings
        let unshuffled_order: Vec<Card> = {
            let mut d = Deck::new();
            let mut v = Vec::new();
            while let Some(c) = d.deal() {
                v.push(c);
            }
            v
        };
        let shuffled_order: Vec<Card> = {
            let mut v = Vec::new();
            let mut d = shuffled;
            while let Some(c) = d.deal() {
                v.push(c);
            }
            v
        };

        // With 52 cards, the chance of an identical ordering is astronomically small.
        assert_ne!(
            unshuffled_order, shuffled_order,
            "shuffled deck should differ from unshuffled"
        );
        let _ = unshuffled;
    }

    #[test]
    fn deal_exhausts_deck() {
        let mut deck = Deck::new();
        for i in (0..52).rev() {
            assert_eq!(deck.remaining(), i + 1);
            assert!(deck.deal().is_some());
        }
        assert_eq!(deck.remaining(), 0);
        assert!(deck.deal().is_none());
    }

    #[test]
    fn shuffle_resets_position() {
        let mut deck = Deck::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        // Deal a few cards
        deck.deal();
        deck.deal();
        assert_eq!(deck.remaining(), 50);

        // Shuffle should reset
        deck.shuffle(&mut rng);
        assert_eq!(deck.remaining(), 52);
    }
}
