class BalancingProcess:
    def __init__(self, neighbourhoods):
        """
        Manages the three-step balancing for all prosumers
        neighbourhoods: dictionary of neighbourhood_id: list of Prosumers
        """
        self.neighbourhoods = neighbourhoods
        
    def step1_self_balancing(self, hour):
        for neighbourhood , prosumers_in_neighbourhood in self.neighbourhoods.items():
            for prosumer in prosumers_in_neighbourhood:
                prosumer.self_balance(hour)

    def step2_local_market(self , current_market_price = 0.2):
        for neighbourhood , prosumers_in_neighbourhood in self.neighbourhoods.items():
            local_prosumers = prosumers_in_neighbourhood
            # Separate buyers and sellers
            buyers = [p for p in local_prosumers if p.imbalance > 0]
            sellers = [p for p in local_prosumers if p.imbalance < 0]
            
            # Sellers: Sort by Price ASC (Cheapest first)
            sellers.sort(key=lambda p: p.trading_price)
            
            # Buyers: Sort by Price DESC (Highest willingness to pay first)
            buyers.sort(key=lambda p: p.trading_price, reverse=True)

            # Match best buyer with best seller
            b_idx = 0
            s_idx = 0
            
            while b_idx < len(buyers) and s_idx < len(sellers):
                buyer = buyers[b_idx]
                seller = sellers[s_idx]
                
                # Check if trade is economically viable (Buyer willing to pay >= Seller asking)
                # Note: You can disable this check if the rule is "always trade available P2P energy"
                if buyer.trading_price >= seller.trading_price:
                    
                    # Determine trade amount (min of Buyer Need vs Seller Surplus)
                    amount = min(abs(buyer.imbalance), abs(seller.imbalance))
                    
                    # Determine trade price (e.g., average or pay-as-bid)
                    trade_price = (buyer.trading_price + seller.trading_price) / 2
                    
                    # Execute Trade
                    # Update Buyer
                    buyer.imbalance -= amount # Reduces deficit
                    buyer.money_balance -= amount * trade_price
                    
                    # Update Seller
                    seller.imbalance += amount # Reduces surplus (moves towards 0)
                    seller.money_balance += amount * trade_price
                    
                    # Record Transaction
                    transaction = {
                        "sender": seller.id,
                        "receiver": buyer.id,
                        "amount": amount,
                        "price_per_kWh": trade_price,
                        "type": "P2P"
                    }
                    buyer.transactions.append(transaction)
                    seller.transactions.append(transaction)
                    
                    # Move to next if fully satisfied/depleted
                    if abs(buyer.imbalance) < 0.001:
                        b_idx += 1
                    if abs(seller.imbalance) < 0.001:
                        s_idx += 1
                        
                else:
                    # If best buyer won't pay best seller's price, no more trades possible
                    break

    def step3_grid_interaction(self):
        # Placeholder for grid interaction balancing logic
        pass