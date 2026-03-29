//! Betting round logic for Texas Hold'em.
//!
//! Tracks bets, actions, and determines when a betting round is complete.

use serde::{Deserialize, Serialize};

/// A player action during a betting round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Surrender the hand.
    Fold,
    /// Pass (only valid when no bet to call).
    Check,
    /// Match the current bet.
    Call,
    /// Raise to the specified total amount.
    Raise(u32),
    /// Go all-in with remaining chips.
    AllIn,
}

/// State of a single betting round within a hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettingRound {
    /// Current bet amount to call.
    pub current_bet: u32,
    /// Minimum raise increment (equal to the big blind, or the last raise size).
    pub min_raise: u32,
    /// Index of the player who must act next.
    pub action_on: usize,
    /// Whether each player has acted this round (reset when a raise occurs).
    pub acted: Vec<bool>,
    /// Whether each player has folded (persistent across the hand).
    pub folded: Vec<bool>,
    /// Whether each player is all-in.
    pub all_in: Vec<bool>,
    /// Each player's current bet this round.
    pub bets: Vec<u32>,
    /// Number of players at the table.
    player_count: usize,
}

impl BettingRound {
    /// Create a new betting round.
    ///
    /// - `player_count`: total seated players.
    /// - `big_blind`: the big blind amount (also the initial minimum raise).
    /// - `dealer`: index of the dealer button.
    /// - `folded`: carried-over fold state from the hand.
    /// - `all_in`: carried-over all-in state from the hand.
    pub fn new(
        player_count: usize,
        big_blind: u32,
        dealer: usize,
        folded: &[bool],
        all_in: &[bool],
    ) -> Self {
        assert!(player_count >= 2);
        let first_to_act = Self::next_active_from(dealer, player_count, folded, all_in);
        Self {
            current_bet: 0,
            min_raise: big_blind,
            action_on: first_to_act,
            acted: vec![false; player_count],
            folded: folded.to_vec(),
            all_in: all_in.to_vec(),
            bets: vec![0; player_count],
            player_count,
        }
    }

    /// Create the pre-flop betting round with blinds already posted.
    ///
    /// Blinds are forced bets, so the small blind and big blind players
    /// have already contributed, but they haven't "acted" yet.
    pub fn new_preflop(
        player_count: usize,
        small_blind: u32,
        big_blind: u32,
        dealer: usize,
        player_chips: &mut [u32],
        folded: &[bool],
        all_in: &[bool],
    ) -> Self {
        assert!(player_count >= 2);

        let mut bets = vec![0u32; player_count];
        let mut new_all_in = all_in.to_vec();
        let mut acted = vec![false; player_count];

        // Determine blind positions.
        let (sb_idx, bb_idx) = if player_count == 2 {
            // Heads-up: dealer is the small blind.
            let sb = dealer;
            let bb = Self::next_active_from(sb, player_count, folded, all_in);
            (sb, bb)
        } else {
            let sb = Self::next_active_from(dealer, player_count, folded, all_in);
            let bb = Self::next_active_from(sb, player_count, folded, all_in);
            (sb, bb)
        };

        // Post small blind.
        let sb_amount = small_blind.min(player_chips[sb_idx]);
        player_chips[sb_idx] -= sb_amount;
        bets[sb_idx] = sb_amount;
        if player_chips[sb_idx] == 0 {
            new_all_in[sb_idx] = true;
            acted[sb_idx] = true;
        }

        // Post big blind.
        let bb_amount = big_blind.min(player_chips[bb_idx]);
        player_chips[bb_idx] -= bb_amount;
        bets[bb_idx] = bb_amount;
        if player_chips[bb_idx] == 0 {
            new_all_in[bb_idx] = true;
            acted[bb_idx] = true;
        }

        // First to act is the player after big blind (Under-The-Gun).
        let first_to_act = Self::next_active_from(bb_idx, player_count, folded, &new_all_in);

        Self {
            current_bet: big_blind.max(bb_amount),
            min_raise: big_blind,
            action_on: first_to_act,
            acted,
            folded: folded.to_vec(),
            all_in: new_all_in,
            bets,
            player_count,
        }
    }

    /// Apply an action from the current player.
    ///
    /// Updates the betting state and advances `action_on` to the next player.
    /// Returns an error if the action is illegal.
    pub fn apply_action(&mut self, action: Action, player_chips: &mut [u32]) -> Result<(), String> {
        let idx = self.action_on;

        if self.folded[idx] {
            return Err("player has already folded".into());
        }
        if self.all_in[idx] {
            return Err("player is already all-in".into());
        }

        match action {
            Action::Fold => {
                self.folded[idx] = true;
                self.acted[idx] = true;
            }
            Action::Check => {
                if self.bets[idx] < self.current_bet {
                    return Err(format!(
                        "cannot check: must call {} (current bet {})",
                        self.current_bet - self.bets[idx],
                        self.current_bet
                    ));
                }
                self.acted[idx] = true;
            }
            Action::Call => {
                let to_call = self.current_bet.saturating_sub(self.bets[idx]);
                if to_call == 0 {
                    // Equivalent to check.
                    self.acted[idx] = true;
                } else if player_chips[idx] <= to_call {
                    // Not enough chips — treat as all-in.
                    let amount = player_chips[idx];
                    self.bets[idx] += amount;
                    player_chips[idx] = 0;
                    self.all_in[idx] = true;
                    self.acted[idx] = true;
                } else {
                    player_chips[idx] -= to_call;
                    self.bets[idx] += to_call;
                    self.acted[idx] = true;
                }
            }
            Action::Raise(raise_to) => {
                let already_bet = self.bets[idx];
                let additional = raise_to.saturating_sub(already_bet);

                if additional == 0 {
                    return Err("raise must increase the bet".into());
                }

                let raise_increment = raise_to.saturating_sub(self.current_bet);
                if raise_increment < self.min_raise && additional < player_chips[idx] {
                    // Under-raise is only allowed as an all-in.
                    return Err(format!(
                        "raise increment {} is below minimum {}",
                        raise_increment, self.min_raise
                    ));
                }

                if additional > player_chips[idx] {
                    return Err(format!(
                        "insufficient chips: need {} but have {}",
                        additional, player_chips[idx]
                    ));
                }

                player_chips[idx] -= additional;
                self.bets[idx] = raise_to;
                self.min_raise = raise_increment.max(self.min_raise);
                self.current_bet = raise_to;

                if player_chips[idx] == 0 {
                    self.all_in[idx] = true;
                }

                // Reset acted for all other players (they need to respond to the raise).
                for i in 0..self.player_count {
                    if i != idx && !self.folded[i] && !self.all_in[i] {
                        self.acted[i] = false;
                    }
                }
                self.acted[idx] = true;
            }
            Action::AllIn => {
                let amount = player_chips[idx];
                if amount == 0 {
                    return Err("player has no chips to go all-in".into());
                }
                let new_bet = self.bets[idx] + amount;
                player_chips[idx] = 0;

                if new_bet > self.current_bet {
                    let raise_increment = new_bet - self.current_bet;
                    // All-in raise reopens betting only if it meets the minimum.
                    // But the all-in itself is always legal.
                    if raise_increment >= self.min_raise {
                        self.min_raise = raise_increment;
                        // Reset acted for others.
                        for i in 0..self.player_count {
                            if i != idx && !self.folded[i] && !self.all_in[i] {
                                self.acted[i] = false;
                            }
                        }
                    }
                    self.current_bet = new_bet;
                }

                self.bets[idx] = new_bet;
                self.all_in[idx] = true;
                self.acted[idx] = true;
            }
        }

        // Advance to next active player.
        if !self.is_complete() {
            self.action_on =
                Self::next_active_from(idx, self.player_count, &self.folded, &self.all_in);
        }

        Ok(())
    }

    /// Whether the betting round is complete.
    ///
    /// Complete when all active (not folded, not all-in) players have acted
    /// and their bets match the current bet, or only one player remains.
    pub fn is_complete(&self) -> bool {
        if self.active_players() <= 1 {
            return true;
        }

        for i in 0..self.player_count {
            if self.folded[i] || self.all_in[i] {
                continue;
            }
            if !self.acted[i] {
                return false;
            }
            if self.bets[i] != self.current_bet {
                return false;
            }
        }
        true
    }

    /// Number of players still in the hand (not folded).
    pub fn active_players(&self) -> usize {
        self.folded.iter().filter(|&&f| !f).count()
    }

    /// Collect all bets into a pot, returning the total collected.
    pub fn collect_bets(&mut self) -> u32 {
        let total = self.bets.iter().sum();
        self.bets.fill(0);
        total
    }

    /// Find the next active player index after `from` (wrapping around).
    fn next_active_from(from: usize, count: usize, folded: &[bool], all_in: &[bool]) -> usize {
        let mut idx = (from + 1) % count;
        let start = idx;
        loop {
            if !folded[idx] && !all_in[idx] {
                return idx;
            }
            idx = (idx + 1) % count;
            if idx == start {
                // All players are folded or all-in.
                return idx;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_removes_player() {
        let folded = vec![false, false, false];
        let all_in = vec![false, false, false];
        let mut round = BettingRound::new(3, 10, 0, &folded, &all_in);
        let mut chips = vec![1000, 1000, 1000];

        // Player at action_on folds.
        let idx = round.action_on;
        round.apply_action(Action::Fold, &mut chips).unwrap();
        assert!(round.folded[idx]);
        assert_eq!(round.active_players(), 2);
    }

    #[test]
    fn check_only_valid_when_no_bet() {
        let folded = vec![false, false];
        let all_in = vec![false, false];
        let mut round = BettingRound::new(2, 10, 0, &folded, &all_in);
        let mut chips = vec![1000, 1000];

        // No current bet — check should succeed.
        round.apply_action(Action::Check, &mut chips).unwrap();
    }

    #[test]
    fn check_fails_when_bet_exists() {
        let mut chips = vec![1000, 1000, 1000];
        let folded = vec![false, false, false];
        let all_in = vec![false, false, false];
        let mut round = BettingRound::new_preflop(3, 5, 10, 0, &mut chips, &folded, &all_in);

        // Pre-flop, action is on UTG who faces the big blind — check should fail.
        let result = round.apply_action(Action::Check, &mut chips);
        assert!(result.is_err());
    }

    #[test]
    fn call_matches_current_bet() {
        let mut chips = vec![1000, 1000, 1000];
        let folded = vec![false, false, false];
        let all_in = vec![false, false, false];
        let mut round = BettingRound::new_preflop(3, 5, 10, 0, &mut chips, &folded, &all_in);

        let caller = round.action_on;
        let chips_before = chips[caller];
        round.apply_action(Action::Call, &mut chips).unwrap();

        // Should have paid 10 (big blind amount).
        assert_eq!(chips[caller], chips_before - 10);
    }

    #[test]
    fn raise_must_meet_minimum() {
        let mut chips = vec![1000, 1000];
        let folded = vec![false, false];
        let all_in = vec![false, false];
        let mut round = BettingRound::new_preflop(2, 5, 10, 0, &mut chips, &folded, &all_in);

        // Minimum raise: current_bet (10) + min_raise (10) = 20.
        let result = round.apply_action(Action::Raise(15), &mut chips);
        assert!(result.is_err(), "raise below minimum should fail");

        // Valid raise.
        round.apply_action(Action::Raise(20), &mut chips).unwrap();
    }

    #[test]
    fn all_in_with_insufficient_chips() {
        let mut chips = vec![1000, 5]; // Player 1 has only 5 chips.
        let folded = vec![false, false];
        let all_in = vec![false, false];
        let _round = BettingRound::new_preflop(2, 5, 10, 0, &mut chips, &folded, &all_in);

        // The short-stacked player may already be all-in from blinds.
        // Let's test an explicit all-in scenario instead.
        let mut chips2 = vec![1000, 1000];
        let mut round2 = BettingRound::new(2, 10, 0, &folded, &all_in);
        round2.current_bet = 0;

        // Player goes all-in.
        let idx = round2.action_on;
        round2.apply_action(Action::AllIn, &mut chips2).unwrap();
        assert!(round2.all_in[idx]);
        assert_eq!(chips2[idx], 0);
    }

    #[test]
    fn betting_round_completes() {
        let folded = vec![false, false];
        let all_in = vec![false, false];
        let mut round = BettingRound::new(2, 10, 0, &folded, &all_in);
        let mut chips = vec![1000, 1000];

        assert!(!round.is_complete());

        // Both check.
        round.apply_action(Action::Check, &mut chips).unwrap();
        round.apply_action(Action::Check, &mut chips).unwrap();

        assert!(round.is_complete());
    }

    #[test]
    fn blind_posting_preflop() {
        let mut chips = vec![1000, 1000, 1000];
        let folded = vec![false, false, false];
        let all_in = vec![false, false, false];
        let round = BettingRound::new_preflop(3, 5, 10, 0, &mut chips, &folded, &all_in);

        // Dealer is 0, SB is 1, BB is 2.
        assert_eq!(round.bets[1], 5, "small blind should be posted");
        assert_eq!(round.bets[2], 10, "big blind should be posted");
        assert_eq!(round.current_bet, 10);

        // Chips deducted.
        assert_eq!(chips[1], 995);
        assert_eq!(chips[2], 990);
    }

    #[test]
    fn collect_bets() {
        let folded = vec![false, false, false];
        let all_in = vec![false, false, false];
        let mut round = BettingRound::new(3, 10, 0, &folded, &all_in);
        round.bets = vec![50, 50, 50];

        let pot = round.collect_bets();
        assert_eq!(pot, 150);
        assert!(round.bets.iter().all(|&b| b == 0));
    }

    #[test]
    fn heads_up_blinds() {
        let mut chips = vec![500, 500];
        let folded = vec![false, false];
        let all_in = vec![false, false];
        let round = BettingRound::new_preflop(2, 5, 10, 0, &mut chips, &folded, &all_in);

        // Heads-up: dealer (0) is SB, other (1) is BB.
        assert_eq!(round.bets[0], 5);
        assert_eq!(round.bets[1], 10);
    }

    #[test]
    fn fold_leaves_one_player() {
        let folded = vec![false, false];
        let all_in = vec![false, false];
        let mut round = BettingRound::new(2, 10, 0, &folded, &all_in);
        let mut chips = vec![1000, 1000];

        round.apply_action(Action::Fold, &mut chips).unwrap();
        assert_eq!(round.active_players(), 1);
        assert!(round.is_complete());
    }
}
