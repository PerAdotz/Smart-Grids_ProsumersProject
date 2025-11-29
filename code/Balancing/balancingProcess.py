class BalancingProcess:
    def __init__(self, prosumers, neighbourhoods):
        """
        Manages the three-step balancing for all prosumers
        prosumers: list of Prosumer classes
        """
        self.prosumers = prosumers
        self.neighbourhoods = neighbourhoods
        
    def step1_self_balancing(self, hour):
        for prosumer in self.prosumers:
            prosumer.self_balance(hour)
        print(self.prosumers[1].get_stats()) #example just to see the stats after self balancing of just one prosumer

    def step2_local_market(self):
        for neighbourhood in self.neighbourhoods.values():
            # Extract prosumers in this neighbourhood
            local_prosumers = [self.prosumers[i] for i in neighbourhood]
            
            # Separate buyers and sellers
            buyers = [p for p in local_prosumers if p.imbalance < 0]
            sellers = [p for p in local_prosumers if p.imbalance > 0]
            
            # Simple matching algorithm
            for buyer in buyers:
                needed = - buyer.imbalance
                for seller in sellers:
                    if seller.imbalance <= 0:
                        continue  # seller has no surplus left
                    available = seller.imbalance
                    trade_amount = min(needed, available)
                    
                    # Execute trade
                    buyer.imbalance += trade_amount
                    seller.imbalance -= trade_amount
                    buyer.money_balance -= trade_amount * 0.1  # cost per kWh
                    seller.money_balance += trade_amount * 0.1
                    
                    # Record transactions
                    transaction = {
                        "sender": seller.id,
                        "receiver": buyer.id,
                        "amount": trade_amount,
                        "price_per_kWh": 0.1
                    }
                    buyer.transactions.append(transaction)
                    seller.transactions.append(transaction)
                    
                    needed -= trade_amount
                    if needed <= 0:
                        break  # buyer's need is fulfilled

    def step3_grid_interaction(self):
        # Placeholder for grid interaction balancing logic
        pass