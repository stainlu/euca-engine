//! Simplified Texas Hold'em poker table.
//!
//! This is a self-contained implementation that will later be replaced by
//! imports from the `euca-poker` game-logic crate.

use rand::seq::SliceRandom;
use rand::thread_rng;

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

/// Suit of a playing card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    Spades,
    Hearts,
    Diamonds,
    Clubs,
}

/// Rank of a playing card (2..=14 where 14 = Ace).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rank(pub u8);

impl Rank {
    #[allow(dead_code)]
    pub const TWO: Self = Self(2);
    #[allow(dead_code)]
    pub const ACE: Self = Self(14);

    pub fn label(self) -> &'static str {
        match self.0 {
            2 => "2",
            3 => "3",
            4 => "4",
            5 => "5",
            6 => "6",
            7 => "7",
            8 => "8",
            9 => "9",
            10 => "10",
            11 => "J",
            12 => "Q",
            13 => "K",
            14 => "A",
            _ => "?",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub fn display(&self) -> String {
        let suit_char = match self.suit {
            Suit::Spades => "\u{2660}",
            Suit::Hearts => "\u{2665}",
            Suit::Diamonds => "\u{2666}",
            Suit::Clubs => "\u{2663}",
        };
        format!("{}{}", self.rank.label(), suit_char)
    }
}

/// A standard 52-card deck.
#[derive(Debug, Clone)]
pub struct Deck {
    cards: Vec<Card>,
    index: usize,
}

impl Deck {
    pub fn new_shuffled() -> Self {
        let mut cards = Vec::with_capacity(52);
        for &suit in &[Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs] {
            for rank_val in 2..=14u8 {
                cards.push(Card {
                    rank: Rank(rank_val),
                    suit,
                });
            }
        }
        cards.shuffle(&mut thread_rng());
        Self { cards, index: 0 }
    }

    pub fn deal(&mut self) -> Option<Card> {
        if self.index < self.cards.len() {
            let card = self.cards[self.index];
            self.index += 1;
            Some(card)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Hand evaluation
// ---------------------------------------------------------------------------

/// Hand ranking (higher value = better hand).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

impl HandRank {
    pub fn name(self) -> &'static str {
        match self {
            Self::HighCard => "High Card",
            Self::OnePair => "One Pair",
            Self::TwoPair => "Two Pair",
            Self::ThreeOfAKind => "Three of a Kind",
            Self::Straight => "Straight",
            Self::Flush => "Flush",
            Self::FullHouse => "Full House",
            Self::FourOfAKind => "Four of a Kind",
            Self::StraightFlush => "Straight Flush",
            Self::RoyalFlush => "Royal Flush",
        }
    }
}

/// Evaluated hand: rank + kicker values for tie-breaking.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct HandValue {
    pub rank: HandRank,
    /// Tie-breaker values, ordered from most significant to least.
    pub kickers: Vec<u8>,
}

/// Evaluate the best 5-card hand from any combination of `hole` + `community`.
pub fn evaluate_hand(hole: &[Card; 2], community: &[Card]) -> HandValue {
    let mut all_cards: Vec<Card> = Vec::with_capacity(7);
    all_cards.extend_from_slice(hole);
    all_cards.extend_from_slice(community);

    // Generate all 5-card combinations and pick the best.
    let mut best: Option<HandValue> = None;
    let n = all_cards.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    for m in (l + 1)..n {
                        let five = [
                            all_cards[i],
                            all_cards[j],
                            all_cards[k],
                            all_cards[l],
                            all_cards[m],
                        ];
                        let val = eval_five(&five);
                        if best.as_ref().is_none_or(|b| val > *b) {
                            best = Some(val);
                        }
                    }
                }
            }
        }
    }
    best.unwrap_or(HandValue {
        rank: HandRank::HighCard,
        kickers: vec![],
    })
}

/// Evaluate a specific 5-card hand.
fn eval_five(cards: &[Card; 5]) -> HandValue {
    let mut ranks: Vec<u8> = cards.iter().map(|c| c.rank.0).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a)); // descending

    let is_flush = {
        let s = cards[0].suit;
        cards.iter().all(|c| c.suit == s)
    };

    let is_straight = is_straight_ranks(&ranks);

    // Ace-low straight special case: A-2-3-4-5
    let is_wheel = ranks == [14, 5, 4, 3, 2];

    // Count rank frequencies.
    let mut freq: Vec<(u8, u8)> = Vec::new(); // (count, rank)
    {
        let mut sorted_asc = ranks.clone();
        sorted_asc.sort_unstable();
        let mut i = 0;
        while i < sorted_asc.len() {
            let r = sorted_asc[i];
            let mut count = 1u8;
            while i + count as usize > 0
                && (i + count as usize) < sorted_asc.len()
                && sorted_asc[i + count as usize] == r
            {
                count += 1;
            }
            freq.push((count, r));
            i += count as usize;
        }
    }
    // Sort by (count desc, rank desc) for pattern matching.
    freq.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

    let counts: Vec<u8> = freq.iter().map(|&(c, _)| c).collect();
    let freq_ranks: Vec<u8> = freq.iter().map(|&(_, r)| r).collect();

    if is_flush && is_straight {
        if ranks[0] == 14 && ranks[1] == 13 {
            return HandValue {
                rank: HandRank::RoyalFlush,
                kickers: ranks,
            };
        }
        return HandValue {
            rank: HandRank::StraightFlush,
            kickers: if is_wheel { vec![5, 4, 3, 2, 1] } else { ranks },
        };
    }

    if counts == [4, 1] {
        return HandValue {
            rank: HandRank::FourOfAKind,
            kickers: freq_ranks,
        };
    }

    if counts == [3, 2] {
        return HandValue {
            rank: HandRank::FullHouse,
            kickers: freq_ranks,
        };
    }

    if is_flush {
        return HandValue {
            rank: HandRank::Flush,
            kickers: ranks,
        };
    }

    if is_straight {
        return HandValue {
            rank: HandRank::Straight,
            kickers: if is_wheel { vec![5, 4, 3, 2, 1] } else { ranks },
        };
    }

    if counts == [3, 1, 1] {
        return HandValue {
            rank: HandRank::ThreeOfAKind,
            kickers: freq_ranks,
        };
    }

    if counts == [2, 2, 1] {
        return HandValue {
            rank: HandRank::TwoPair,
            kickers: freq_ranks,
        };
    }

    if counts == [2, 1, 1, 1] {
        return HandValue {
            rank: HandRank::OnePair,
            kickers: freq_ranks,
        };
    }

    HandValue {
        rank: HandRank::HighCard,
        kickers: ranks,
    }
}

fn is_straight_ranks(sorted_desc: &[u8]) -> bool {
    if sorted_desc.len() != 5 {
        return false;
    }
    // Normal straight check.
    let normal = sorted_desc.windows(2).all(|w| w[0] == w[1] + 1);
    // Ace-low (wheel): A-5-4-3-2.
    let wheel = sorted_desc == [14, 5, 4, 3, 2];
    normal || wheel
}

// ---------------------------------------------------------------------------
// Poker Table
// ---------------------------------------------------------------------------

/// (seat, chips_won, hand_name)
pub type WinnerEntry = (usize, u32, String);
/// (seat, hole_cards)
pub type ShownHand = (usize, [Card; 2]);

/// Phase of a poker hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Waiting for players to ready up between hands.
    Waiting,
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Waiting => "waiting",
            Self::Preflop => "preflop",
            Self::Flop => "flop",
            Self::Turn => "turn",
            Self::River => "river",
            Self::Showdown => "showdown",
        }
    }
}

/// A player sitting at the table.
#[derive(Debug, Clone)]
pub struct Player {
    pub name: String,
    pub chips: u32,
    pub hole_cards: Option<[Card; 2]>,
    pub folded: bool,
    pub current_bet: u32,
    pub all_in: bool,
    /// Whether this player is ready for the next hand.
    pub ready: bool,
    /// Whether this seat is occupied.
    pub active: bool,
}

impl Player {
    pub fn new(name: String, chips: u32) -> Self {
        Self {
            name,
            chips,
            hole_cards: None,
            folded: false,
            current_bet: 0,
            all_in: false,
            ready: false,
            active: true,
        }
    }
}

/// Outcome of processing an action.
#[derive(Debug)]
pub enum ActionOutcome {
    /// Action accepted, hand continues.
    Continue,
    /// Only one player remains, they win the pot.
    LastPlayerWins { seat: usize },
    /// Betting round complete, advance phase. If the new phase is `Showdown`,
    /// the caller should evaluate hands.
    AdvancePhase { new_phase: Phase },
}

/// An authoritative poker table managing a single hand's state.
#[derive(Debug)]
pub struct PokerTable {
    pub players: Vec<Player>,
    pub community_cards: Vec<Card>,
    pub pot: u32,
    pub phase: Phase,
    pub dealer: usize,
    pub action_on: usize,
    pub current_bet: u32,
    pub small_blind: u32,
    pub big_blind: u32,
    pub last_raiser: Option<usize>,
    deck: Deck,
    /// Number of players who have acted this betting round.
    actions_this_round: usize,
    pub min_players: usize,
}

impl PokerTable {
    pub fn new(num_seats: usize, small_blind: u32, big_blind: u32) -> Self {
        let players = (0..num_seats)
            .map(|_| Player {
                name: String::new(),
                chips: 0,
                hole_cards: None,
                folded: false,
                current_bet: 0,
                all_in: false,
                ready: false,
                active: false,
            })
            .collect();

        Self {
            players,
            community_cards: Vec::new(),
            pot: 0,
            phase: Phase::Waiting,
            dealer: 0,
            action_on: 0,
            current_bet: 0,
            small_blind,
            big_blind,
            last_raiser: None,
            deck: Deck::new_shuffled(),
            actions_this_round: 0,
            min_players: 2,
        }
    }

    /// Seat a player at the first available seat. Returns the seat index.
    pub fn seat_player(&mut self, name: String, chips: u32) -> Option<usize> {
        for (i, p) in self.players.iter_mut().enumerate() {
            if !p.active {
                *p = Player::new(name, chips);
                return Some(i);
            }
        }
        None
    }

    /// Remove a player from a seat.
    pub fn unseat_player(&mut self, seat: usize) {
        if seat < self.players.len() {
            self.players[seat].active = false;
            self.players[seat].name.clear();
            self.players[seat].chips = 0;
        }
    }

    /// Number of active (seated) players.
    pub fn active_player_count(&self) -> usize {
        self.players.iter().filter(|p| p.active).count()
    }

    /// Number of players still in the hand (not folded, active).
    fn players_in_hand(&self) -> usize {
        self.players
            .iter()
            .filter(|p| p.active && !p.folded)
            .count()
    }

    /// Number of players who can still act (not folded, not all-in).
    fn players_who_can_act(&self) -> usize {
        self.players
            .iter()
            .filter(|p| p.active && !p.folded && !p.all_in)
            .count()
    }

    /// Mark a player as ready. Returns `true` if a new hand should start.
    pub fn set_ready(&mut self, seat: usize) -> bool {
        if seat < self.players.len() && self.players[seat].active {
            self.players[seat].ready = true;
        }
        self.should_start_hand()
    }

    fn should_start_hand(&self) -> bool {
        if self.phase != Phase::Waiting {
            return false;
        }
        let active: Vec<&Player> = self.players.iter().filter(|p| p.active).collect();
        active.len() >= self.min_players && active.iter().all(|p| p.ready)
    }

    /// Start a new hand. Resets state, shuffles, deals hole cards, posts blinds.
    pub fn start_hand(&mut self) {
        // Reset per-hand state.
        self.deck = Deck::new_shuffled();
        self.community_cards.clear();
        self.pot = 0;
        self.current_bet = 0;
        self.actions_this_round = 0;
        self.last_raiser = None;

        for p in &mut self.players {
            p.hole_cards = None;
            p.folded = false;
            p.current_bet = 0;
            p.all_in = false;
            p.ready = false;
            if !p.active {
                p.folded = true; // treat empty seats as folded
            }
        }

        // Advance dealer.
        self.dealer = self.next_active_seat(self.dealer);

        // Deal hole cards: two rounds, one card per active player per round
        // (standard dealing order, starting left of dealer).
        let seats: Vec<usize> = {
            let n = self.players.len();
            let mut s = Vec::new();
            let mut idx = self.next_active_seat(self.dealer);
            for _ in 0..self.active_player_count() {
                s.push(idx);
                idx = self.next_active_seat(idx);
            }
            // Ensure no duplicates (stop if we wrap).
            s.truncate(n);
            s
        };

        let mut dealt: Vec<(usize, Vec<Card>)> =
            seats.iter().map(|&s| (s, Vec::with_capacity(2))).collect();
        for _ in 0..2 {
            for entry in &mut dealt {
                entry.1.push(self.deck.deal().expect("deck has 52 cards"));
            }
        }
        for (seat, cards) in dealt {
            self.players[seat].hole_cards = Some([cards[0], cards[1]]);
        }

        // Post blinds.
        let sb_seat = self.next_active_seat(self.dealer);
        let bb_seat = self.next_active_seat(sb_seat);
        self.post_blind(sb_seat, self.small_blind);
        self.post_blind(bb_seat, self.big_blind);
        self.current_bet = self.big_blind;

        // Action starts left of big blind.
        self.action_on = self.next_active_seat(bb_seat);
        self.phase = Phase::Preflop;
        self.last_raiser = Some(bb_seat); // BB is the "raiser" initially.
    }

    fn post_blind(&mut self, seat: usize, amount: u32) {
        let p = &mut self.players[seat];
        let actual = amount.min(p.chips);
        p.chips -= actual;
        p.current_bet = actual;
        self.pot += actual;
        if p.chips == 0 {
            p.all_in = true;
        }
    }

    /// Apply a player action. Returns an error string if invalid.
    pub fn apply_action(
        &mut self,
        seat: usize,
        action: &str,
        amount: Option<u32>,
    ) -> Result<ActionOutcome, String> {
        if self.phase == Phase::Waiting || self.phase == Phase::Showdown {
            return Err("No active hand".to_string());
        }
        if seat != self.action_on {
            return Err(format!(
                "Not your turn (action is on seat {})",
                self.action_on
            ));
        }
        let p = &self.players[seat];
        if !p.active || p.folded || p.all_in {
            return Err("Cannot act".to_string());
        }

        match action {
            "fold" => self.do_fold(seat),
            "check" => self.do_check(seat)?,
            "call" => self.do_call(seat),
            "raise" => {
                let raise_to = amount.ok_or("Raise requires an amount")?;
                self.do_raise(seat, raise_to)?;
            }
            other => return Err(format!("Unknown action: {other}")),
        }

        self.actions_this_round += 1;

        // Check if only one player remains.
        if self.players_in_hand() == 1 {
            let winner = self
                .players
                .iter()
                .position(|p| p.active && !p.folded)
                .unwrap();
            return Ok(ActionOutcome::LastPlayerWins { seat: winner });
        }

        // Advance to next player who can act.
        let next = self.next_acting_seat(self.action_on);

        // Check if betting round is complete.
        if self.is_round_complete(next) {
            let new_phase = self.advance_phase();
            return Ok(ActionOutcome::AdvancePhase { new_phase });
        }

        self.action_on = next;
        Ok(ActionOutcome::Continue)
    }

    fn do_fold(&mut self, seat: usize) {
        self.players[seat].folded = true;
    }

    fn do_check(&mut self, seat: usize) -> Result<(), String> {
        if self.players[seat].current_bet < self.current_bet {
            return Err("Cannot check -- there is a bet to call".to_string());
        }
        Ok(())
    }

    fn do_call(&mut self, seat: usize) {
        let p = &mut self.players[seat];
        let to_call = self.current_bet.saturating_sub(p.current_bet);
        let actual = to_call.min(p.chips);
        p.chips -= actual;
        p.current_bet += actual;
        self.pot += actual;
        if p.chips == 0 {
            p.all_in = true;
        }
    }

    fn do_raise(&mut self, seat: usize, raise_to: u32) -> Result<(), String> {
        if raise_to <= self.current_bet {
            return Err(format!(
                "Raise must be above current bet ({})",
                self.current_bet
            ));
        }
        let min_raise = self.current_bet + self.big_blind;
        let p = &mut self.players[seat];

        // Allow all-in even if below minimum raise.
        let effective_raise = if raise_to < min_raise && raise_to < p.chips + p.current_bet {
            return Err(format!("Minimum raise is {min_raise}"));
        } else {
            raise_to
        };

        let additional = effective_raise.saturating_sub(p.current_bet);
        let actual = additional.min(p.chips);
        p.chips -= actual;
        p.current_bet += actual;
        self.pot += actual;
        self.current_bet = p.current_bet;
        if p.chips == 0 {
            p.all_in = true;
        }
        self.last_raiser = Some(seat);
        self.actions_this_round = 0; // reset: everyone must respond to the raise
        Ok(())
    }

    /// Check if the betting round is complete.
    fn is_round_complete(&self, next_seat: usize) -> bool {
        // If nobody can act, round is done.
        if self.players_who_can_act() == 0 {
            return true;
        }

        // Round is complete when we've gone all the way around to the last
        // raiser (or everyone has acted once and bets are even).
        if let Some(raiser) = self.last_raiser
            && next_seat == raiser
        {
            return true;
        }

        // If all active non-folded non-all-in players have matching bets
        // and everyone has had at least one chance to act:
        let all_even = self
            .players
            .iter()
            .filter(|p| p.active && !p.folded && !p.all_in)
            .all(|p| p.current_bet == self.current_bet);

        if all_even && self.actions_this_round >= self.players_who_can_act() {
            return true;
        }

        false
    }

    /// Advance to the next phase, dealing community cards as needed.
    /// Returns the new phase.
    fn advance_phase(&mut self) -> Phase {
        // Reset per-round betting state.
        for p in &mut self.players {
            p.current_bet = 0;
        }
        self.current_bet = 0;
        self.actions_this_round = 0;
        self.last_raiser = None;

        self.phase = match self.phase {
            Phase::Preflop => {
                // Burn one, deal 3.
                self.deck.deal(); // burn
                for _ in 0..3 {
                    if let Some(c) = self.deck.deal() {
                        self.community_cards.push(c);
                    }
                }
                Phase::Flop
            }
            Phase::Flop => {
                self.deck.deal(); // burn
                if let Some(c) = self.deck.deal() {
                    self.community_cards.push(c);
                }
                Phase::Turn
            }
            Phase::Turn => {
                self.deck.deal(); // burn
                if let Some(c) = self.deck.deal() {
                    self.community_cards.push(c);
                }
                Phase::River
            }
            Phase::River | Phase::Showdown | Phase::Waiting => Phase::Showdown,
        };

        // Set action to first player left of dealer.
        if self.phase != Phase::Showdown {
            self.action_on = self.next_acting_seat(self.dealer);
            // If no one can act (everyone all-in), jump straight to showdown.
            if self.players_who_can_act() == 0 {
                return self.advance_phase();
            }
        }

        self.phase
    }

    /// Evaluate hands at showdown. Returns `(winners, hands_shown)`.
    pub fn showdown(&self) -> (Vec<WinnerEntry>, Vec<ShownHand>) {
        let mut best_value: Option<HandValue> = None;
        let mut evaluations: Vec<(usize, HandValue)> = Vec::new();
        let mut hands_shown: Vec<(usize, [Card; 2])> = Vec::new();

        for (i, p) in self.players.iter().enumerate() {
            if !p.active || p.folded {
                continue;
            }
            if let Some(hole) = &p.hole_cards {
                let val = evaluate_hand(hole, &self.community_cards);
                if best_value.as_ref().is_none_or(|b| val > *b) {
                    best_value = Some(val.clone());
                }
                evaluations.push((i, val));
                hands_shown.push((i, *hole));
            }
        }

        let best = match best_value {
            Some(b) => b,
            None => {
                return (vec![], vec![]);
            }
        };

        let winner_seats: Vec<usize> = evaluations
            .iter()
            .filter(|(_, v)| *v == best)
            .map(|(seat, _)| *seat)
            .collect();

        let share = self.pot / winner_seats.len() as u32;
        let winners: Vec<(usize, u32, String)> = winner_seats
            .iter()
            .map(|&seat| (seat, share, best.rank.name().to_string()))
            .collect();

        (winners, hands_shown)
    }

    /// Award pot to a single winner (when all others fold).
    pub fn award_pot(&mut self, seat: usize) {
        self.players[seat].chips += self.pot;
        self.pot = 0;
        self.phase = Phase::Waiting;
    }

    /// Finish showdown: award chips to winners and transition to Waiting.
    pub fn finish_showdown(&mut self) {
        let (winners, _) = self.showdown();
        for (seat, amount, _) in &winners {
            self.players[*seat].chips += amount;
        }
        // Handle remainder from integer division.
        let total_awarded: u32 = winners.iter().map(|(_, a, _)| a).sum();
        if total_awarded < self.pot && !winners.is_empty() {
            self.players[winners[0].0].chips += self.pot - total_awarded;
        }
        self.pot = 0;
        self.phase = Phase::Waiting;
    }

    // -----------------------------------------------------------------------
    // Seat navigation helpers
    // -----------------------------------------------------------------------

    /// Next active seat after `from` (wrapping).
    fn next_active_seat(&self, from: usize) -> usize {
        let n = self.players.len();
        for offset in 1..=n {
            let idx = (from + offset) % n;
            if self.players[idx].active {
                return idx;
            }
        }
        from
    }

    /// Next seat that can still act (active, not folded, not all-in).
    fn next_acting_seat(&self, from: usize) -> usize {
        let n = self.players.len();
        for offset in 1..=n {
            let idx = (from + offset) % n;
            let p = &self.players[idx];
            if p.active && !p.folded && !p.all_in {
                return idx;
            }
        }
        from
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deck_deals_52() {
        let mut deck = Deck::new_shuffled();
        let mut count = 0;
        while deck.deal().is_some() {
            count += 1;
        }
        assert_eq!(count, 52);
    }

    #[test]
    fn test_hand_eval_pair() {
        let hole = [
            Card {
                rank: Rank(10),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(10),
                suit: Suit::Spades,
            },
        ];
        let community = [
            Card {
                rank: Rank(3),
                suit: Suit::Diamonds,
            },
            Card {
                rank: Rank(7),
                suit: Suit::Clubs,
            },
            Card {
                rank: Rank(9),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(2),
                suit: Suit::Spades,
            },
            Card {
                rank: Rank(5),
                suit: Suit::Diamonds,
            },
        ];
        let val = evaluate_hand(&hole, &community);
        assert_eq!(val.rank, HandRank::OnePair);
    }

    #[test]
    fn test_hand_eval_flush() {
        let hole = [
            Card {
                rank: Rank(14),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(10),
                suit: Suit::Hearts,
            },
        ];
        let community = [
            Card {
                rank: Rank(3),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(7),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(9),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(2),
                suit: Suit::Spades,
            },
            Card {
                rank: Rank(5),
                suit: Suit::Diamonds,
            },
        ];
        let val = evaluate_hand(&hole, &community);
        assert_eq!(val.rank, HandRank::Flush);
    }

    #[test]
    fn test_hand_eval_straight() {
        let hole = [
            Card {
                rank: Rank(6),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(7),
                suit: Suit::Spades,
            },
        ];
        let community = [
            Card {
                rank: Rank(8),
                suit: Suit::Diamonds,
            },
            Card {
                rank: Rank(9),
                suit: Suit::Clubs,
            },
            Card {
                rank: Rank(10),
                suit: Suit::Hearts,
            },
            Card {
                rank: Rank(2),
                suit: Suit::Spades,
            },
            Card {
                rank: Rank(3),
                suit: Suit::Diamonds,
            },
        ];
        let val = evaluate_hand(&hole, &community);
        assert_eq!(val.rank, HandRank::Straight);
    }

    #[test]
    fn test_table_seat_player() {
        let mut table = PokerTable::new(6, 5, 10);
        let seat = table.seat_player("Alice".into(), 1000);
        assert_eq!(seat, Some(0));
        assert_eq!(table.active_player_count(), 1);
    }

    #[test]
    fn test_table_start_hand() {
        let mut table = PokerTable::new(6, 5, 10);
        table.seat_player("Alice".into(), 1000);
        table.seat_player("Bob".into(), 1000);
        table.set_ready(0);
        let should_start = table.set_ready(1);
        assert!(should_start);

        table.start_hand();
        assert_eq!(table.phase, Phase::Preflop);
        // Both players should have hole cards.
        assert!(table.players[0].hole_cards.is_some());
        assert!(table.players[1].hole_cards.is_some());
        // Pot should have blinds.
        assert_eq!(table.pot, 15); // 5 + 10
    }
}
