from Blockchain.blockchain_v2 import Transaction

class BalancingProcess:
    def __init__(self, prosumers , neighbourhoods):
        """
        Manages the three-step balancing for all prosumers
        neighbourhoods: dictionary of neighbourhood_id: list of Prosumers objects
        """
        self.prosumers = prosumers
        self.neighbourhoods = neighbourhoods
        self.hour = 0
        
    def set_date_and_hour(self, date, hour):
        self.date = date
        self.hour = hour 

    def step1_self_balancing(self):
        for neighbourhood , prosumers_in_neighbourhood in self.neighbourhoods.items():
            for prosumer in prosumers_in_neighbourhood:
                prosumer.self_balance(self.date, self.hour)

    def step2_local_market(self , energy_chain):
        # for neighbourhood , prosumers_in_neighbourhood in self.neighbourhoods.items(): #we want prosumers to exhange energy even outside their neighbourhood
            # local_prosumers = prosumers_in_neighbourhood
            local_prosumers = self.prosumers
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
                    buyer.imbalance = buyer.imbalance -  amount # Reduces deficit
                    buyer.money_balance = ( buyer.money_balance - (amount * trade_price) ) * buyer.bonus
                    
                    # Update Seller
                    seller.imbalance = seller.imbalance + amount # Reduces surplus (moves towards 0)
                    seller.money_balance = ( seller.money_balance + (amount * trade_price) ) * (1 - seller.bonus)
                    
                    # Record Transaction , here we need the Blockchain part
                    transaction = {
                        "sender": seller.id,
                        "receiver": buyer.id,
                        "amount": float(amount),
                        "price_per_kWh": trade_price,
                        "type": "P2P",
                        "hour": self.hour
                    }
                    buyer.transactions[self.hour].append(transaction)
                    seller.transactions[self.hour].append(transaction)

                    tx = Transaction(sender=transaction["sender"], receiver=transaction["receiver"], 
                                     amount=transaction["amount"], price=transaction["price_per_kWh"], step=self.hour)
                    energy_chain.add_transaction(tx)

                    # Move to next if fully satisfied/depleted
                    if abs(buyer.imbalance) < 1e-5: #created an epsilon to avoid floating point issues, once it went infinite loop here
                        b_idx += 1
                    if abs(seller.imbalance) < 1e-5:
                        s_idx += 1
                        
                else:
                    # If best buyer won't pay best seller's price, no more trades possible
                    break

    def step3_grid_interaction(self, current_market_price , energy_chain, margin=0.05):
        """
        Step 3: Local Market / Grid Interaction (Aggregator)
        Any remaining imbalance is cleared with the aggregator.
        
        current_market_price: The D-1 price for this hour.
        energy_chain: The blockchain instance to record transactions.
        margin: The percentage fee/penalty for using the grid (default 5%).
        """
        # Iterate through all prosumers in all neighbourhoods
        for neighbourhood_id, prosumers_in_neighbourhood in self.neighbourhoods.items():
            for p in prosumers_in_neighbourhood:
                
                # Skip if already balanced (allowing for small floating point error)
                if abs(p.imbalance) < 1e-5:
                    continue

                transaction = None
                
                # --- CASE 1: Prosumer still needs energy (Deficit / Buyer) ---
                if p.imbalance > 0:
                    amount_needed = p.imbalance
                    
                    # Buying from grid is expensive: Price * (1 + margin)
                    grid_price = current_market_price * (1 + margin) * p.penalty
                    cost = amount_needed * grid_price
                    
                    # Execute
                    p.money_balance -= cost
                    p.imbalance = 0  # Imbalance is resolved
                    
                    transaction = {
                        "sender": "Aggregator",
                        "receiver": p.id,
                        "amount": float(amount_needed),
                        "price_per_kWh": grid_price,
                        "type": "GRID_buy",
                        "hour": self.hour
                    }

                # --- CASE 2: Prosumer has extra energy (Surplus / Seller) ---
                elif p.imbalance < 0:
                    amount_sold = abs(p.imbalance)
                    
                    # Selling to grid pays less: Price * (1 - margin)
                    grid_price = current_market_price * (1 - margin)
                    earnings = amount_sold * grid_price
                    
                    # Execute
                    p.money_balance += earnings
                    p.imbalance = 0  # Imbalance is resolved
                    
                    transaction = {
                        "sender": p.id,
                        "receiver": "Aggregator",
                        "amount": float(amount_sold),
                        "price_per_kWh": grid_price,
                        "type": "GRID_sell",
                        "hour": self.hour
                    }

                # Record transaction
                if transaction:
                    p.transactions[self.hour].append(transaction)
                    tx = Transaction(sender=transaction["sender"], receiver=transaction["receiver"], 
                                     amount=transaction["amount"], price=transaction["price_per_kWh"], step=self.hour)
                    energy_chain.add_transaction(tx)