class Regulator:

    def apply_regulations(self, prosumers, current_hour , p2p_bonus_policy, grid_penalty_policy):
        """
        Analyzes behavior and assigns multipliers (Bonus/Penalty)
        for the next market round. Goal is incevintivize P2P trading
        and disincentivize grid reliance.
        """
        print(f"  [Regulator] Auditing Hour {current_hour}...")

        p2p_1 = p2p_bonus_policy["1"]
        p2p_5 = p2p_bonus_policy["5"]
        p2p_10 = p2p_bonus_policy["10"]

        grid_5 = grid_penalty_policy["5"]
        grid_10 = grid_penalty_policy["10"]
        
        for p in prosumers:
            # 1. Reset base (neutral) values at the beginning of the check
            # If the prosumer does nothing special, it resets to 1.0
            p.bonus = 1.0
            p.penalty = 1.0
            
            # --- BONUS CALCULATION (P2P) ---
            # If you are a good citizen engaging in P2P trading, you get a price advantage
            if p.p2p_exchanges >= 10:
                p.bonus = p2p_10  # 10% advantage (discount if buying, extra profit if selling)
            elif p.p2p_exchanges >= 5:
                p.bonus = p2p_5 # 5% advantage
            elif p.p2p_exchanges >= 1:
                p.bonus = p2p_1 # 2% advantage
            
            # --- PENALTY CALCULATION (Grid) ---
            # If you abuse the aggregator, a worse tariff is applied
            if p.agg_exchanges >= 10:
                p.penalty = grid_10  # You pay 10% more to the grid
            elif p.agg_exchanges >= 5:
                p.penalty = grid_5 # You pay 5% more
            
            # Update counters based on transactions from the hour just concluded
            # This way, bonuses/penalties apply to the NEXT hour
            hourly_transactions = p.transactions.get(current_hour, [])
            p2p_activity = any(t['type'] == 'P2P' for t in hourly_transactions)
            grid_activity = any('GRID_buy' in t['type'] for t in hourly_transactions)
            
            if p2p_activity:
                p.p2p_exchanges += 1
            if grid_activity:
                p.agg_exchanges += 1