//! Poker hand evaluation for Texas Hold'em.
//!
//! Supports evaluating the best 5-card hand from up to 7 cards,
//! with correct ordering and tiebreaking via kickers.

use crate::card::Card;
use serde::{Deserialize, Serialize};

/// Poker hand rankings in ascending order of strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HandRank {
    HighCard,
    OnePair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
    RoyalFlush,
}

/// An evaluated hand with its ranking and kicker values for tiebreaking.
///
/// Kickers are stored highest-first and represent the relevant card values
/// that distinguish hands of the same rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatedHand {
    pub rank: HandRank,
    /// Kicker values for tiebreaking (highest first).
    pub kickers: Vec<u8>,
}

impl PartialOrd for EvaluatedHand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvaluatedHand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank.cmp(&other.rank).then_with(|| {
            // Compare kickers lexicographically (highest first).
            self.kickers.cmp(&other.kickers)
        })
    }
}

/// Evaluate the best 5-card hand from the given cards (5, 6, or 7).
///
/// For Texas Hold'em, pass 7 cards (2 hole + 5 community). The function
/// tries all C(n, 5) combinations and returns the strongest hand.
///
/// # Panics
///
/// Panics if `cards` has fewer than 5 or more than 7 elements.
pub fn evaluate_hand(cards: &[Card]) -> EvaluatedHand {
    assert!(
        cards.len() >= 5 && cards.len() <= 7,
        "expected 5..=7 cards, got {}",
        cards.len()
    );

    let n = cards.len();
    let mut best: Option<EvaluatedHand> = None;

    // Generate all C(n, 5) combinations.
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                for d in (c + 1)..n {
                    for e in (d + 1)..n {
                        let five = [cards[a], cards[b], cards[c], cards[d], cards[e]];
                        let evaluated = evaluate_five(&five);
                        best = Some(match best {
                            None => evaluated,
                            Some(prev) => {
                                if evaluated > prev {
                                    evaluated
                                } else {
                                    prev
                                }
                            }
                        });
                    }
                }
            }
        }
    }

    best.expect("at least one 5-card combination must exist")
}

/// Evaluate exactly 5 cards into a [`HandRank`] with kickers.
fn evaluate_five(cards: &[Card; 5]) -> EvaluatedHand {
    // Sort by rank descending for easier analysis.
    let mut sorted = *cards;
    sorted.sort_by(|a, b| b.rank.cmp(&a.rank));

    let values: Vec<u8> = sorted.iter().map(|c| c.rank.value()).collect();

    let is_flush = {
        let suit = sorted[0].suit;
        sorted.iter().all(|c| c.suit == suit)
    };

    let is_straight = is_straight_sequence(&values);

    // Special case: ace-low straight (wheel) — A 5 4 3 2.
    let is_wheel = values == [14, 5, 4, 3, 2];

    // Count rank frequencies.
    let mut freq: Vec<(u8, u8)> = Vec::new(); // (count, rank_value)
    {
        let mut i = 0;
        while i < values.len() {
            let val = values[i];
            let mut count = 1u8;
            while i + count as usize > 0
                && (i + count as usize) < values.len()
                && values[i + count as usize] == val
            {
                count += 1;
            }
            freq.push((count, val));
            i += count as usize;
        }
    }
    // Sort frequencies: primary by count descending, secondary by rank value descending.
    freq.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

    let counts: Vec<u8> = freq.iter().map(|&(c, _)| c).collect();
    let freq_vals: Vec<u8> = freq.iter().map(|&(_, v)| v).collect();

    // Determine hand rank and kickers.
    if (is_straight || is_wheel) && is_flush {
        if is_wheel {
            // Straight flush with ace low — kicker is 5 (the high card of the straight).
            return EvaluatedHand {
                rank: HandRank::StraightFlush,
                kickers: vec![5],
            };
        }
        if values[0] == 14 && values[1] == 13 {
            // Royal flush: A K Q J T suited.
            return EvaluatedHand {
                rank: HandRank::RoyalFlush,
                kickers: vec![14],
            };
        }
        return EvaluatedHand {
            rank: HandRank::StraightFlush,
            kickers: vec![values[0]],
        };
    }

    if counts == [4, 1] {
        // Four of a kind: kickers are [quad rank, kicker].
        return EvaluatedHand {
            rank: HandRank::FourOfAKind,
            kickers: freq_vals,
        };
    }

    if counts == [3, 2] {
        // Full house: kickers are [trips rank, pair rank].
        return EvaluatedHand {
            rank: HandRank::FullHouse,
            kickers: freq_vals,
        };
    }

    if is_flush {
        return EvaluatedHand {
            rank: HandRank::Flush,
            kickers: values,
        };
    }

    if is_straight {
        return EvaluatedHand {
            rank: HandRank::Straight,
            kickers: vec![values[0]],
        };
    }

    if is_wheel {
        return EvaluatedHand {
            rank: HandRank::Straight,
            kickers: vec![5],
        };
    }

    if counts == [3, 1, 1] {
        return EvaluatedHand {
            rank: HandRank::ThreeOfAKind,
            kickers: freq_vals,
        };
    }

    if counts == [2, 2, 1] {
        return EvaluatedHand {
            rank: HandRank::TwoPair,
            kickers: freq_vals,
        };
    }

    if counts == [2, 1, 1, 1] {
        return EvaluatedHand {
            rank: HandRank::OnePair,
            kickers: freq_vals,
        };
    }

    // High card: all five card values as kickers, highest first.
    EvaluatedHand {
        rank: HandRank::HighCard,
        kickers: values,
    }
}

/// Check if the given sorted-descending values form a consecutive straight.
fn is_straight_sequence(values: &[u8]) -> bool {
    if values.len() != 5 {
        return false;
    }
    for i in 0..4 {
        if values[i] != values[i + 1] + 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Rank, Suit};

    /// Helper to make a card quickly.
    fn c(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn test_royal_flush() {
        let cards = [
            c(Rank::Ace, Suit::Spades),
            c(Rank::King, Suit::Spades),
            c(Rank::Queen, Suit::Spades),
            c(Rank::Jack, Suit::Spades),
            c(Rank::Ten, Suit::Spades),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::RoyalFlush);
    }

    #[test]
    fn test_straight_flush() {
        let cards = [
            c(Rank::Five, Suit::Hearts),
            c(Rank::Six, Suit::Hearts),
            c(Rank::Seven, Suit::Hearts),
            c(Rank::Eight, Suit::Hearts),
            c(Rank::Nine, Suit::Hearts),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::StraightFlush);
        assert_eq!(hand.kickers, vec![9]);
    }

    #[test]
    fn test_four_of_a_kind() {
        let cards = [
            c(Rank::King, Suit::Clubs),
            c(Rank::King, Suit::Diamonds),
            c(Rank::King, Suit::Hearts),
            c(Rank::King, Suit::Spades),
            c(Rank::Two, Suit::Clubs),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::FourOfAKind);
        assert_eq!(hand.kickers, vec![13, 2]);
    }

    #[test]
    fn test_full_house() {
        let cards = [
            c(Rank::Queen, Suit::Clubs),
            c(Rank::Queen, Suit::Diamonds),
            c(Rank::Queen, Suit::Hearts),
            c(Rank::Seven, Suit::Spades),
            c(Rank::Seven, Suit::Clubs),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::FullHouse);
        assert_eq!(hand.kickers, vec![12, 7]);
    }

    #[test]
    fn test_flush() {
        let cards = [
            c(Rank::Ace, Suit::Hearts),
            c(Rank::Ten, Suit::Hearts),
            c(Rank::Seven, Suit::Hearts),
            c(Rank::Four, Suit::Hearts),
            c(Rank::Two, Suit::Hearts),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::Flush);
        assert_eq!(hand.kickers, vec![14, 10, 7, 4, 2]);
    }

    #[test]
    fn test_straight() {
        let cards = [
            c(Rank::Five, Suit::Clubs),
            c(Rank::Six, Suit::Diamonds),
            c(Rank::Seven, Suit::Hearts),
            c(Rank::Eight, Suit::Spades),
            c(Rank::Nine, Suit::Clubs),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::Straight);
        assert_eq!(hand.kickers, vec![9]);
    }

    #[test]
    fn test_three_of_a_kind() {
        let cards = [
            c(Rank::Jack, Suit::Clubs),
            c(Rank::Jack, Suit::Diamonds),
            c(Rank::Jack, Suit::Hearts),
            c(Rank::Four, Suit::Spades),
            c(Rank::Two, Suit::Clubs),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::ThreeOfAKind);
        assert_eq!(hand.kickers, vec![11, 4, 2]);
    }

    #[test]
    fn test_two_pair() {
        let cards = [
            c(Rank::Nine, Suit::Clubs),
            c(Rank::Nine, Suit::Diamonds),
            c(Rank::Five, Suit::Hearts),
            c(Rank::Five, Suit::Spades),
            c(Rank::King, Suit::Clubs),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::TwoPair);
        assert_eq!(hand.kickers, vec![9, 5, 13]);
    }

    #[test]
    fn test_one_pair() {
        let cards = [
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::Eight, Suit::Hearts),
            c(Rank::Four, Suit::Spades),
            c(Rank::Two, Suit::Clubs),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::OnePair);
        assert_eq!(hand.kickers, vec![14, 8, 4, 2]);
    }

    #[test]
    fn test_high_card() {
        let cards = [
            c(Rank::Ace, Suit::Clubs),
            c(Rank::King, Suit::Diamonds),
            c(Rank::Nine, Suit::Hearts),
            c(Rank::Five, Suit::Spades),
            c(Rank::Two, Suit::Hearts),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::HighCard);
        assert_eq!(hand.kickers, vec![14, 13, 9, 5, 2]);
    }

    #[test]
    fn test_ace_low_straight() {
        let cards = [
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Two, Suit::Diamonds),
            c(Rank::Three, Suit::Hearts),
            c(Rank::Four, Suit::Spades),
            c(Rank::Five, Suit::Clubs),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::Straight);
        assert_eq!(hand.kickers, vec![5]); // 5-high straight
    }

    #[test]
    fn test_seven_card_evaluation() {
        // Hole cards: A♠ K♠, Community: Q♠ J♠ T♠ 2♣ 3♦
        // Best hand should be Royal Flush.
        let cards = [
            c(Rank::Ace, Suit::Spades),
            c(Rank::King, Suit::Spades),
            c(Rank::Queen, Suit::Spades),
            c(Rank::Jack, Suit::Spades),
            c(Rank::Ten, Suit::Spades),
            c(Rank::Two, Suit::Clubs),
            c(Rank::Three, Suit::Diamonds),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::RoyalFlush);
    }

    #[test]
    fn test_seven_card_picks_best_hand() {
        // 7 cards where best hand is a flush, not a pair.
        let cards = [
            c(Rank::Ace, Suit::Hearts),
            c(Rank::King, Suit::Hearts),
            c(Rank::Nine, Suit::Hearts),
            c(Rank::Five, Suit::Hearts),
            c(Rank::Two, Suit::Hearts),
            c(Rank::Two, Suit::Clubs),
            c(Rank::Two, Suit::Diamonds),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::Flush);
    }

    #[test]
    fn test_hand_comparison_ordering() {
        let royal_flush = evaluate_hand(&[
            c(Rank::Ace, Suit::Spades),
            c(Rank::King, Suit::Spades),
            c(Rank::Queen, Suit::Spades),
            c(Rank::Jack, Suit::Spades),
            c(Rank::Ten, Suit::Spades),
        ]);
        let straight_flush = evaluate_hand(&[
            c(Rank::Nine, Suit::Hearts),
            c(Rank::Eight, Suit::Hearts),
            c(Rank::Seven, Suit::Hearts),
            c(Rank::Six, Suit::Hearts),
            c(Rank::Five, Suit::Hearts),
        ]);
        let four_kind = evaluate_hand(&[
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::Ace, Suit::Hearts),
            c(Rank::Ace, Suit::Spades),
            c(Rank::King, Suit::Clubs),
        ]);
        let full_house = evaluate_hand(&[
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::Ace, Suit::Hearts),
            c(Rank::King, Suit::Spades),
            c(Rank::King, Suit::Clubs),
        ]);
        let flush = evaluate_hand(&[
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::Jack, Suit::Diamonds),
            c(Rank::Nine, Suit::Diamonds),
            c(Rank::Six, Suit::Diamonds),
            c(Rank::Three, Suit::Diamonds),
        ]);
        let straight = evaluate_hand(&[
            c(Rank::Ten, Suit::Clubs),
            c(Rank::Nine, Suit::Diamonds),
            c(Rank::Eight, Suit::Hearts),
            c(Rank::Seven, Suit::Spades),
            c(Rank::Six, Suit::Clubs),
        ]);
        let three_kind = evaluate_hand(&[
            c(Rank::Seven, Suit::Clubs),
            c(Rank::Seven, Suit::Diamonds),
            c(Rank::Seven, Suit::Hearts),
            c(Rank::King, Suit::Spades),
            c(Rank::Two, Suit::Clubs),
        ]);
        let two_pair = evaluate_hand(&[
            c(Rank::King, Suit::Clubs),
            c(Rank::King, Suit::Diamonds),
            c(Rank::Three, Suit::Hearts),
            c(Rank::Three, Suit::Spades),
            c(Rank::Ace, Suit::Clubs),
        ]);
        let one_pair = evaluate_hand(&[
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::King, Suit::Hearts),
            c(Rank::Queen, Suit::Spades),
            c(Rank::Jack, Suit::Clubs),
        ]);
        let high_card = evaluate_hand(&[
            c(Rank::Ace, Suit::Clubs),
            c(Rank::King, Suit::Diamonds),
            c(Rank::Queen, Suit::Hearts),
            c(Rank::Jack, Suit::Spades),
            c(Rank::Nine, Suit::Clubs),
        ]);

        assert!(royal_flush > straight_flush);
        assert!(straight_flush > four_kind);
        assert!(four_kind > full_house);
        assert!(full_house > flush);
        assert!(flush > straight);
        assert!(straight > three_kind);
        assert!(three_kind > two_pair);
        assert!(two_pair > one_pair);
        assert!(one_pair > high_card);
    }

    #[test]
    fn test_tiebreaker_kickers() {
        // Two one-pair hands: pair of aces with different kickers.
        let hand_a = evaluate_hand(&[
            c(Rank::Ace, Suit::Clubs),
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::King, Suit::Hearts),
            c(Rank::Queen, Suit::Spades),
            c(Rank::Jack, Suit::Clubs),
        ]);
        let hand_b = evaluate_hand(&[
            c(Rank::Ace, Suit::Hearts),
            c(Rank::Ace, Suit::Spades),
            c(Rank::King, Suit::Clubs),
            c(Rank::Queen, Suit::Diamonds),
            c(Rank::Ten, Suit::Clubs),
        ]);
        assert_eq!(hand_a.rank, HandRank::OnePair);
        assert_eq!(hand_b.rank, HandRank::OnePair);
        assert!(hand_a > hand_b, "J kicker should beat T kicker");
    }

    #[test]
    fn test_tiebreaker_equal_hands() {
        // Identical hand ranks and kickers should be equal.
        let hand_a = evaluate_hand(&[
            c(Rank::Ace, Suit::Clubs),
            c(Rank::King, Suit::Diamonds),
            c(Rank::Queen, Suit::Hearts),
            c(Rank::Jack, Suit::Spades),
            c(Rank::Nine, Suit::Clubs),
        ]);
        let hand_b = evaluate_hand(&[
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::King, Suit::Hearts),
            c(Rank::Queen, Suit::Spades),
            c(Rank::Jack, Suit::Clubs),
            c(Rank::Nine, Suit::Hearts),
        ]);
        assert_eq!(hand_a, hand_b);
    }

    #[test]
    fn test_ace_low_straight_flush() {
        let cards = [
            c(Rank::Ace, Suit::Diamonds),
            c(Rank::Two, Suit::Diamonds),
            c(Rank::Three, Suit::Diamonds),
            c(Rank::Four, Suit::Diamonds),
            c(Rank::Five, Suit::Diamonds),
        ];
        let hand = evaluate_hand(&cards);
        assert_eq!(hand.rank, HandRank::StraightFlush);
        assert_eq!(hand.kickers, vec![5]);
    }
}
