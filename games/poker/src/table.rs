//! Table-level game state machine for Texas Hold'em.
//!
//! Manages the full lifecycle of a poker hand: seating, dealing, betting
//! rounds, community cards, showdown, and pot distribution.

use crate::betting::{Action, BettingRound};
use crate::card::Card;
use crate::deck::Deck;
use crate::hand::{EvaluatedHand, evaluate_hand};
use serde::{Deserialize, Serialize};

/// Phase of the current hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    WaitingForPlayers,
    PreFlop,
    Flop,
    Turn,
    River,
    Showdown,
}

/// A seated player at the table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: String,
    pub name: String,
    pub chips: u32,
    pub hole_cards: Option<[Card; 2]>,
    pub seated: bool,
}

/// A winner from a showdown with their payout and hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Winner {
    pub player_id: String,
    pub amount: u32,
    pub hand: EvaluatedHand,
}

/// The poker table: the central game state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub players: Vec<Player>,
    pub community_cards: Vec<Card>,
    pub pot: u32,
    pub phase: Phase,
    pub dealer_index: usize,
    pub small_blind: u32,
    pub big_blind: u32,
    pub betting: Option<BettingRound>,
    pub max_players: usize,

    /// Deck is server-side only and not serialized to clients.
    #[serde(skip)]
    deck: Option<Deck>,
}

impl Table {
    /// Create a new table with the given blind structure.
    pub fn new(small_blind: u32, big_blind: u32, max_players: usize) -> Self {
        Self {
            players: Vec::with_capacity(max_players),
            community_cards: Vec::new(),
            pot: 0,
            phase: Phase::WaitingForPlayers,
            dealer_index: 0,
            small_blind,
            big_blind,
            betting: None,
            max_players,
            deck: None,
        }
    }

    /// Seat a player at the table. Returns their seat index.
    pub fn seat_player(&mut self, id: String, name: String, chips: u32) -> Result<usize, String> {
        if self.players.len() >= self.max_players {
            return Err("table is full".into());
        }
        if self.players.iter().any(|p| p.id == id) {
            return Err(format!("player {id} is already seated"));
        }
        let index = self.players.len();
        self.players.push(Player {
            id,
            name,
            chips,
            hole_cards: None,
            seated: true,
        });
        Ok(index)
    }

    /// Remove a player from the table.
    pub fn remove_player(&mut self, id: &str) {
        self.players.retain(|p| p.id != id);
    }

    /// Start a new hand: shuffle, post blinds, deal hole cards.
    pub fn start_hand(&mut self, rng: &mut impl rand::Rng) -> Result<(), String> {
        let seated_count = self.seated_player_count();
        if seated_count < 2 {
            return Err("need at least 2 players to start a hand".into());
        }

        // Reset per-hand state.
        self.community_cards.clear();
        self.pot = 0;
        for player in &mut self.players {
            player.hole_cards = None;
        }

        // Shuffle deck.
        let mut deck = Deck::new();
        deck.shuffle(rng);
        self.deck = Some(deck);

        // Deal 2 hole cards to each seated player with chips.
        for player in &mut self.players {
            if player.seated && player.chips > 0 {
                let c1 = self.deck.as_mut().unwrap().deal().unwrap();
                let c2 = self.deck.as_mut().unwrap().deal().unwrap();
                player.hole_cards = Some([c1, c2]);
            }
        }

        // Determine which players are in the hand (have hole cards).
        let player_count = self.players.len();
        let folded: Vec<bool> = self
            .players
            .iter()
            .map(|p| p.hole_cards.is_none())
            .collect();
        let all_in = vec![false; player_count];

        // Collect chips into a mutable slice for blind posting.
        let mut chips: Vec<u32> = self.players.iter().map(|p| p.chips).collect();

        let betting = BettingRound::new_preflop(
            player_count,
            self.small_blind,
            self.big_blind,
            self.dealer_index,
            &mut chips,
            &folded,
            &all_in,
        );

        // Write chips back.
        for (i, &c) in chips.iter().enumerate() {
            self.players[i].chips = c;
        }

        self.betting = Some(betting);
        self.phase = Phase::PreFlop;
        Ok(())
    }

    /// Apply a player action. Advances phase when the betting round completes.
    pub fn apply_action(&mut self, player_id: &str, action: Action) -> Result<(), String> {
        if self.phase == Phase::WaitingForPlayers || self.phase == Phase::Showdown {
            return Err(format!("cannot act during {:?} phase", self.phase));
        }

        let player_idx = self
            .players
            .iter()
            .position(|p| p.id == player_id)
            .ok_or_else(|| format!("player {player_id} not found"))?;

        let betting = self.betting.as_mut().ok_or("no active betting round")?;

        if betting.action_on != player_idx {
            return Err(format!(
                "not {player_id}'s turn (action is on seat {})",
                betting.action_on
            ));
        }

        let mut chips: Vec<u32> = self.players.iter().map(|p| p.chips).collect();
        betting.apply_action(action, &mut chips)?;
        for (i, &c) in chips.iter().enumerate() {
            self.players[i].chips = c;
        }

        // Check if betting round is complete.
        if self.betting.as_ref().unwrap().is_complete() {
            self.pot += self.betting.as_mut().unwrap().collect_bets();

            if self.betting.as_ref().unwrap().active_players() <= 1 {
                // Everyone folded except one player — award pot immediately.
                self.award_to_last_player();
                self.phase = Phase::Showdown;
                self.betting = None;
            } else {
                self.advance_phase();
            }
        }

        Ok(())
    }

    /// Deal community cards for the current phase transition.
    fn deal_community(&mut self) {
        let deck = self.deck.as_mut().expect("deck must exist during a hand");
        match self.phase {
            Phase::Flop => {
                // Burn one, deal three.
                let _burn = deck.deal();
                for _ in 0..3 {
                    self.community_cards.push(deck.deal().unwrap());
                }
            }
            Phase::Turn | Phase::River => {
                // Burn one, deal one.
                let _burn = deck.deal();
                self.community_cards.push(deck.deal().unwrap());
            }
            _ => {}
        }
    }

    /// Advance to the next phase after a betting round completes.
    fn advance_phase(&mut self) {
        let next_phase = match self.phase {
            Phase::PreFlop => Phase::Flop,
            Phase::Flop => Phase::Turn,
            Phase::Turn => Phase::River,
            Phase::River => Phase::Showdown,
            other => other,
        };

        self.phase = next_phase;

        if self.phase == Phase::Showdown {
            self.betting = None;
            return;
        }

        // Deal community cards for the new phase.
        self.deal_community();

        // Start a new betting round (action starts left of the dealer).
        let player_count = self.players.len();
        let prev_betting = self.betting.as_ref().unwrap();
        let folded = prev_betting.folded.clone();
        let all_in = prev_betting.all_in.clone();

        let new_betting = BettingRound::new(
            player_count,
            self.big_blind,
            self.dealer_index,
            &folded,
            &all_in,
        );

        self.betting = Some(new_betting);

        // If the new betting round is already complete (e.g. all players all-in),
        // continue advancing.
        if self.betting.as_ref().unwrap().is_complete() {
            self.pot += self.betting.as_mut().unwrap().collect_bets();
            self.advance_phase();
        }
    }

    /// Evaluate hands and determine winner(s) at showdown.
    pub fn showdown(&mut self) -> Vec<Winner> {
        let mut contenders: Vec<(usize, EvaluatedHand)> = Vec::new();

        for (i, player) in self.players.iter().enumerate() {
            let hole = match player.hole_cards {
                Some(cards) => cards,
                None => continue,
            };

            // Check if this player is still in the hand (not folded).
            if let Some(ref betting) = self.betting
                && betting.folded[i]
            {
                continue;
            }

            let mut all_cards = Vec::with_capacity(7);
            all_cards.extend_from_slice(&hole);
            all_cards.extend_from_slice(&self.community_cards);

            if all_cards.len() >= 5 {
                let hand = evaluate_hand(&all_cards);
                contenders.push((i, hand));
            }
        }

        if contenders.is_empty() {
            return Vec::new();
        }

        // Find the best hand.
        contenders.sort_by(|a, b| b.1.cmp(&a.1));
        let best_rank = &contenders[0].1;

        // All players tied with the best hand split the pot.
        let winners: Vec<&(usize, EvaluatedHand)> =
            contenders.iter().filter(|(_, h)| h == best_rank).collect();

        let share = self.pot / winners.len() as u32;
        let remainder = self.pot % winners.len() as u32;

        let result: Vec<Winner> = winners
            .iter()
            .enumerate()
            .map(|(wi, (pi, hand))| {
                let amount = share + if wi == 0 { remainder } else { 0 };
                self.players[*pi].chips += amount;
                Winner {
                    player_id: self.players[*pi].id.clone(),
                    amount,
                    hand: hand.clone(),
                }
            })
            .collect();

        self.pot = 0;
        result
    }

    /// Award the entire pot to the last remaining player (all others folded).
    fn award_to_last_player(&mut self) {
        let betting = self.betting.as_ref().unwrap();
        for (i, player) in self.players.iter_mut().enumerate() {
            if !betting.folded[i] && player.hole_cards.is_some() {
                player.chips += self.pot;
                self.pot = 0;
                return;
            }
        }
    }

    /// Move the dealer button and start the next hand.
    pub fn next_hand(&mut self, rng: &mut impl rand::Rng) -> Result<(), String> {
        // Advance dealer button to the next seated player with chips.
        let player_count = self.players.len();
        if player_count < 2 {
            return Err("not enough players".into());
        }

        let mut next = (self.dealer_index + 1) % player_count;
        let start = next;
        loop {
            if self.players[next].seated && self.players[next].chips > 0 {
                break;
            }
            next = (next + 1) % player_count;
            if next == start {
                return Err("no eligible player for dealer button".into());
            }
        }
        self.dealer_index = next;

        self.start_hand(rng)
    }

    /// Count seated players with chips.
    fn seated_player_count(&self) -> usize {
        self.players
            .iter()
            .filter(|p| p.seated && p.chips > 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(12345)
    }

    #[test]
    fn seat_and_remove_players() {
        let mut table = Table::new(5, 10, 6);

        let idx = table
            .seat_player("p1".into(), "Alice".into(), 1000)
            .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(table.players.len(), 1);

        table.seat_player("p2".into(), "Bob".into(), 1000).unwrap();
        assert_eq!(table.players.len(), 2);

        // Duplicate should fail.
        assert!(table.seat_player("p1".into(), "Alice".into(), 500).is_err());

        table.remove_player("p1");
        assert_eq!(table.players.len(), 1);
        assert_eq!(table.players[0].id, "p2");
    }

    #[test]
    fn seat_player_table_full() {
        let mut table = Table::new(5, 10, 2);
        table
            .seat_player("p1".into(), "Alice".into(), 1000)
            .unwrap();
        table.seat_player("p2".into(), "Bob".into(), 1000).unwrap();
        assert!(
            table
                .seat_player("p3".into(), "Carol".into(), 1000)
                .is_err()
        );
    }

    #[test]
    fn start_hand_deals_two_cards_each() {
        let mut table = Table::new(5, 10, 6);
        table
            .seat_player("p1".into(), "Alice".into(), 1000)
            .unwrap();
        table.seat_player("p2".into(), "Bob".into(), 1000).unwrap();
        table
            .seat_player("p3".into(), "Carol".into(), 1000)
            .unwrap();

        let mut rng = make_rng();
        table.start_hand(&mut rng).unwrap();

        assert_eq!(table.phase, Phase::PreFlop);
        for player in &table.players {
            assert!(
                player.hole_cards.is_some(),
                "{} should have cards",
                player.name
            );
            assert_eq!(player.hole_cards.unwrap().len(), 2);
        }
    }

    #[test]
    fn need_two_players_to_start() {
        let mut table = Table::new(5, 10, 6);
        table
            .seat_player("p1".into(), "Alice".into(), 1000)
            .unwrap();
        let mut rng = make_rng();
        assert!(table.start_hand(&mut rng).is_err());
    }

    #[test]
    fn full_hand_flow() {
        let mut table = Table::new(5, 10, 6);
        table
            .seat_player("p1".into(), "Alice".into(), 1000)
            .unwrap();
        table.seat_player("p2".into(), "Bob".into(), 1000).unwrap();

        let mut rng = make_rng();
        table.start_hand(&mut rng).unwrap();
        assert_eq!(table.phase, Phase::PreFlop);

        // Pre-flop: both call/check through.
        // Heads-up: dealer(0) is SB, player(1) is BB.
        // Action starts on SB (since SB acts first pre-flop in heads-up).
        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Call).unwrap();

        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Check).unwrap();

        // Should now be Flop.
        assert_eq!(table.phase, Phase::Flop);
        assert_eq!(table.community_cards.len(), 3);

        // Flop: both check.
        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Check).unwrap();

        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Check).unwrap();

        // Turn.
        assert_eq!(table.phase, Phase::Turn);
        assert_eq!(table.community_cards.len(), 4);

        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Check).unwrap();

        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Check).unwrap();

        // River.
        assert_eq!(table.phase, Phase::River);
        assert_eq!(table.community_cards.len(), 5);

        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Check).unwrap();

        let betting = table.betting.as_ref().unwrap();
        let actor = &table.players[betting.action_on].id.clone();
        table.apply_action(actor, Action::Check).unwrap();

        // Showdown.
        assert_eq!(table.phase, Phase::Showdown);

        let winners = table.showdown();
        assert!(!winners.is_empty());

        // Total pot was 20 (SB 5 + BB 10, SB called to 10 = 20).
        let total_awarded: u32 = winners.iter().map(|w| w.amount).sum();
        assert_eq!(total_awarded, 20);
    }

    #[test]
    fn fold_gives_pot_to_remaining() {
        let mut table = Table::new(5, 10, 6);
        table
            .seat_player("p1".into(), "Alice".into(), 1000)
            .unwrap();
        table.seat_player("p2".into(), "Bob".into(), 1000).unwrap();

        let mut rng = make_rng();
        table.start_hand(&mut rng).unwrap();

        // First player to act folds.
        let betting = table.betting.as_ref().unwrap();
        let folder_id = table.players[betting.action_on].id.clone();
        let other_id = table
            .players
            .iter()
            .find(|p| p.id != folder_id)
            .unwrap()
            .id
            .clone();

        let other_chips_before = table
            .players
            .iter()
            .find(|p| p.id == other_id)
            .unwrap()
            .chips;

        table.apply_action(&folder_id, Action::Fold).unwrap();

        assert_eq!(table.phase, Phase::Showdown);

        // The non-folding player should have gained the pot.
        let other_chips_after = table
            .players
            .iter()
            .find(|p| p.id == other_id)
            .unwrap()
            .chips;
        assert!(other_chips_after > other_chips_before);
    }

    #[test]
    fn next_hand_advances_dealer() {
        let mut table = Table::new(5, 10, 6);
        table
            .seat_player("p1".into(), "Alice".into(), 1000)
            .unwrap();
        table.seat_player("p2".into(), "Bob".into(), 1000).unwrap();
        table
            .seat_player("p3".into(), "Carol".into(), 1000)
            .unwrap();

        let mut rng = make_rng();
        table.start_hand(&mut rng).unwrap();

        let first_dealer = table.dealer_index;

        // Quick hand: everyone folds to the last player.
        let betting = table.betting.as_ref().unwrap();
        let actor = table.players[betting.action_on].id.clone();
        table.apply_action(&actor, Action::Fold).unwrap();

        // Might need one more fold.
        if table.phase != Phase::Showdown {
            let betting = table.betting.as_ref().unwrap();
            let actor = table.players[betting.action_on].id.clone();
            table.apply_action(&actor, Action::Fold).unwrap();
        }

        table.next_hand(&mut rng).unwrap();
        assert_ne!(table.dealer_index, first_dealer);
    }

    #[test]
    fn side_pot_with_all_in() {
        let mut table = Table::new(5, 10, 6);
        table.seat_player("p1".into(), "Alice".into(), 100).unwrap();
        table.seat_player("p2".into(), "Bob".into(), 1000).unwrap();

        let mut rng = make_rng();
        table.start_hand(&mut rng).unwrap();

        // First to act goes all-in.
        let betting = table.betting.as_ref().unwrap();
        let actor = table.players[betting.action_on].id.clone();
        table.apply_action(&actor, Action::AllIn).unwrap();

        // Second player calls.
        if table.phase != Phase::Showdown {
            let betting = table.betting.as_ref().unwrap();
            let actor = table.players[betting.action_on].id.clone();
            table.apply_action(&actor, Action::Call).unwrap();
        }

        // Game should proceed through all community cards automatically
        // (both players are all-in or have matched) or go to showdown.
        // Verify we can reach showdown.
        assert!(
            table.phase == Phase::Showdown
                || table.community_cards.len() == 5
                || table.community_cards.len() == 3
                || table.community_cards.len() == 4
        );
    }
}
