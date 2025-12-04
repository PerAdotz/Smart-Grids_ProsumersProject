class Regulator:
    """
    Simulates a regulatory body responsible for analyzing Prosumer trading behavior 
    and applying dynamic financial multipliers (Bonuses and Penalties) to incentivize 
    Peer-to-Peer (P2P) trading over reliance on the central Aggregator/Grid.
    """

    def apply_regulations(self, prosumers, current_hour , p2p_bonus_policy, grid_penalty_policy):
        """
        Analyzes Prosumer transactions from the hour just concluded and assigns new 
        Bonus and Penalty multipliers for the subsequent market round.

        Args:
            prosumers (list): A list of all Prosumer objects.
            current_hour (int): The hour of the day (0-23) that has just been processed.
            p2p_bonus_policy (dict): Dictionary defining bonus multipliers based on P2P exchange counts.
            grid_penalty_policy (dict): Dictionary defining penalty multipliers based on Aggregator exchange counts.

        Returns:
            None: Updates the 'bonus' and 'penalty' attributes of each Prosumer object directly.
        """
        print(f"[Regulator] Auditing Hour {current_hour}...")

        # Extract policy thresholds for P2P bonus
        p2p_1 = p2p_bonus_policy["1"]
        p2p_5 = p2p_bonus_policy["5"]
        p2p_10 = p2p_bonus_policy["10"]

        # Extract policy thresholds for Grid penalty
        grid_5 = grid_penalty_policy["5"]
        grid_10 = grid_penalty_policy["10"]
        
        for p in prosumers:
            # 1. Update counters based on the transactions recorded in the hour just concluded
            hourly_transactions = p.transactions.get(current_hour, [])

            # Check if any P2P transactions occurred in the last hour
            p2p_activity = any(t['type'] == 'P2P' for t in hourly_transactions)

            # Check if any Grid purchase transactions occurred in the last hour
            grid_activity = any('GRID_buy' in t['type'] for t in hourly_transactions)
            
            # Increment running totals for the Prosumer's trading history
            if p2p_activity:
                p.p2p_exchanges += 1
            if grid_activity:
                p.agg_exchanges += 1
            
            # 2. Reset base (neutral) values before calculating new multipliers
            # This ensures that multipliers are calculated based on cumulative behavior, 
            # but default to neutral (1.0) if no specific threshold is met
            p.bonus = 1.0
            p.penalty = 1.0
            
            # --- BONUS CALCULATION (P2P Encouragement) ---
            # Bonus applies if the prosumer has a history of engaging in P2P trading
            # A bonus > 1.0 provides a price advantage (discount if buying, extra profit if selling)
            if p.p2p_exchanges >= 10:
                p.bonus = p2p_10 # Highest advantage
            elif p.p2p_exchanges >= 5:
                p.bonus = p2p_5 # Medium advantage
            elif p.p2p_exchanges >= 1:
                p.bonus = p2p_1 # Base advantage
            
            # --- PENALTY CALCULATION (Grid Discouragement) ---
            # Penalty applies if the prosumer frequently relies on the Aggregator to buy power
            # A penalty > 1.0 increases the purchase price from the grid
            if p.agg_exchanges >= 10:
                p.penalty = grid_10 # Highest penalty
            elif p.agg_exchanges >= 5:
                p.penalty = grid_5 # Medium penalty