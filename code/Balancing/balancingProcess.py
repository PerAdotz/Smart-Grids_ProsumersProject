from Blockchain.blockchain_v2 import Transaction

class BalancingProcess:
    def __init__(self, prosumers , neighbourhoods):
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
        # Usiamo tutti i prosumer insieme per massimizzare la liquidità
        local_prosumers = self.prosumers
        
        buyers = [p for p in local_prosumers if p.imbalance > 0]
        sellers = [p for p in local_prosumers if p.imbalance < 0]
        
        # Sellers: Cheapest first
        sellers.sort(key=lambda p: p.trading_price)
        # Buyers: Highest willingness to pay first
        buyers.sort(key=lambda p: p.trading_price, reverse=True)

        b_idx = 0
        s_idx = 0
        
        while b_idx < len(buyers) and s_idx < len(sellers):
            buyer = buyers[b_idx]
            seller = sellers[s_idx]
            
            # Check economic viability
            if buyer.trading_price >= seller.trading_price:
                
                amount = min(abs(buyer.imbalance), abs(seller.imbalance))
                trade_price = (buyer.trading_price + seller.trading_price) / 2
                
                transaction_value = amount * trade_price
                
                # Buyer: Il bonus reduce the cost
                # ex :  if bonus = 1.02, pay 2% less.
                if buyer.bonus > 0:
                    cost_for_buyer = transaction_value / buyer.bonus
                else:
                    cost_for_buyer = transaction_value
                buyer.imbalance -= amount
                buyer.money_balance -= cost_for_buyer
                
                # Seller: Il bonus increases the revenue
                # if bonus = 1.02, earns 2% more.
                revenue_for_seller = transaction_value * seller.bonus
                seller.imbalance += amount
                seller.money_balance += revenue_for_seller
                
                # Record Transaction
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

                if abs(buyer.imbalance) < 1e-5:
                    b_idx += 1
                if abs(seller.imbalance) < 1e-5:
                    s_idx += 1
                    
            else:
                break

    def step3_grid_interaction(self, current_market_price , energy_chain, margin=0.05):
        for neighbourhood_id, prosumers_in_neighbourhood in self.neighbourhoods.items():
            for p in prosumers_in_neighbourhood:
                
                if abs(p.imbalance) < 1e-5:
                    continue

                transaction = None
                

                if p.imbalance > 0: # Buyer (Deficit)
                    amount_needed = p.imbalance
                    
                    # Penalità aumenta il prezzo di acquisto
                    # Esempio: penalty 1.10 -> Prezzo aumenta del 10%
                    grid_price = current_market_price * (1 + margin) * p.penalty
                    cost = amount_needed * grid_price
                    
                    p.money_balance -= cost
                    p.imbalance = 0
                    
                    transaction = {
                        "sender": "Aggregator",
                        "receiver": p.id,
                        "amount": float(amount_needed),
                        "price_per_kWh": grid_price,
                        "type": "GRID_buy",
                        "hour": self.hour
                    }

                elif p.imbalance < 0: # Seller (Surplus)
                    amount_sold = abs(p.imbalance)
                    
                    # Qui la penalità non si applica solitamente alla vendita, 
                    # o se vuoi penalizzare la vendita, dovresti DIVIDERE per il fattore.
                    # Manteniamo la logica standard: vendo a prezzo base.
                    grid_price = current_market_price * (1 - margin)
                    earnings = amount_sold * grid_price
                    
                    p.money_balance += earnings
                    p.imbalance = 0
                    
                    transaction = {
                        "sender": p.id,
                        "receiver": "Aggregator",
                        "amount": float(amount_sold),
                        "price_per_kWh": grid_price,
                        "type": "GRID_sell",
                        "hour": self.hour
                    }

                if transaction:
                    p.transactions[self.hour].append(transaction)
                    tx = Transaction(sender=transaction["sender"], receiver=transaction["receiver"], 
                                     amount=transaction["amount"], price=transaction["price_per_kWh"], step=self.hour)
                    energy_chain.add_transaction(tx)