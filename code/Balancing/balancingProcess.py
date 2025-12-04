from Blockchain.blockchain_v2 import Transaction

class BalancingProcess:
    """
    Manages the hourly energy balancing and trading process for all Prosumers 
    in the community across three steps: 
    1. Self-balancing (consumption, generation, battery).
    2. Self-organized trading (P2P exchanges).
    3. Local market trading (Aggregator/Grid exchange).
    """

    def __init__(self, prosumers , neighbourhoods):
        """
        Initializes the BalancingProcess with the list of prosumers and their neighborhood grouping.

        Args:
            prosumers (list): A list of all Prosumer objects.
            neighbourhoods (dict): A dictionary mapping neighbourhoods to lists of Prosumer objects.
        """
        self.prosumers = prosumers
        self.neighbourhoods = neighbourhoods
        self.hour = 0
        
    def set_date_and_hour(self, date, hour):
        """
        Sets the temporal context for the current simulation step.

        Args:
            date (pd.Timestamp): The current date of the simulation.
            hour (int): The current hour (0-23).

        Returns:
            None
        """
        self.date = date
        self.hour = hour 

    def step1_self_balancing(self):
        """
        Executes the initial balancing step: each prosumer calculates its PV generation, 
        meets its load, and uses its battery to cover any initial residual imbalance.

        Returns:
            None
        """
        # Iterate over all prosumers, grouped by neighborhood
        for _ , prosumers_in_neighbourhood in self.neighbourhoods.items():
            for prosumer in prosumers_in_neighbourhood:
                # Prosumer calculates net load (PV - Load) and uses battery for self-consumption/storage
                prosumer.self_balance(self.date, self.hour)

    def step2_self_organized_trading(self , energy_chain):
        """
        Conducts Peer-to-Peer (P2P) trading among all prosumers to maximize local energy exchange.

        Prosumers are sorted by their declared trading price (sellers cheapest, buyers highest bid).
        Transactions are executed only if the buyer's bid meets or exceeds the seller's ask price.

        Args:
            energy_chain (Blockchain): The blockchain instance to record energy transactions.

        Returns:
            None
        """
        # Use all prosumers together to maximize liquidity in the P2P market
        local_prosumers = self.prosumers
        
        # Identify buyers (deficit, imbalance > 0) and sellers (surplus, imbalance < 0)
        buyers = [p for p in local_prosumers if p.imbalance > 0]
        sellers = [p for p in local_prosumers if p.imbalance < 0]
        
        # Sort for optimal matching:
        # - Sellers: cheapest price first
        sellers.sort(key=lambda p: p.trading_price)
        # - Buyers: highest willingness to pay first
        buyers.sort(key=lambda p: p.trading_price, reverse=True)

        b_idx = 0
        s_idx = 0
        
        # Start the P2P trading matching process
        while b_idx < len(buyers) and s_idx < len(sellers):
            buyer = buyers[b_idx]
            seller = sellers[s_idx]
            
            # Check economic viability: transaction only occurs if buyer is willing to pay >= seller's ask price
            if buyer.trading_price >= seller.trading_price:
                # Determine the traded amount (limited by the smallest imbalance)
                amount = min(abs(buyer.imbalance), abs(seller.imbalance))

                # Calculate the trading price as the average of the bid and ask prices
                trade_price = (buyer.trading_price + seller.trading_price) / 2
                
                transaction_value = amount * trade_price
                
                # --- Buyer (Deficit) Accounting ---
                # Buyer's bonus reduces the cost (e.g., bonus=1.02 means 2% discount)
                if buyer.bonus > 0:
                    cost_for_buyer = transaction_value / buyer.bonus
                else:
                    cost_for_buyer = transaction_value

                buyer.imbalance -= amount
                buyer.money_balance -= cost_for_buyer
                
                # --- Seller (Surplus) Accounting ---
                # Seller's bonus increases the revenue (e.g., bonus=1.02 means 2% gain)
                revenue_for_seller = transaction_value * seller.bonus

                seller.imbalance += amount # Surplus imbalance (negative) moves toward zero
                seller.money_balance += revenue_for_seller
                
                # --- Record and Broadcast Transaction ---
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

                # Add transaction to the energy chain (blockchain)
                tx = Transaction(sender=transaction["sender"], receiver=transaction["receiver"], 
                                amount=transaction["amount"], price=transaction["price_per_kWh"], step=self.hour)
                energy_chain.add_transaction(tx)

                # Advance pointers if the prosumer's imbalance is cleared (tolerance 1e-5)
                if abs(buyer.imbalance) < 1e-5:
                    b_idx += 1
                if abs(seller.imbalance) < 1e-5:
                    s_idx += 1
                    
            else:
                # If buyer's bid is lower than seller's ask, no more trades are economically viable
                break

    def step3_local_market(self, current_market_price , energy_chain, margin=0.05):
        """
        Clears any remaining energy imbalance by trading with the Aggregator/Grid 
        at a fixed rate plus a margin and the prosumer's penalty factor.

        Args:
            current_market_price (float): The base reference price for grid trading.
            energy_chain (Blockchain): The blockchain instance to record grid transactions.
            margin (float, optional): The aggregator's profit margin applied to the base price. Defaults to 0.05.

        Returns:
            None
        """
        # Process each prosumer's residual imbalance
        for _, prosumers_in_neighbourhood in self.neighbourhoods.items():
            for p in prosumers_in_neighbourhood:
                # Skip if the imbalance is zero (within tolerance)
                if abs(p.imbalance) < 1e-5:
                    continue

                transaction = None
                
                # If buyer (Deficit) - Must buy from the Aggregator
                if p.imbalance > 0:
                    amount_needed = p.imbalance
                    
                    # Calculate final grid purchase price: Base * (1 + margin) * Penalty factor
                    # Penalty factor increases the cost (e.g., penalty=1.10 means 10% price increase)
                    grid_price = current_market_price * (1 + margin) * p.penalty
                    cost = amount_needed * grid_price
                    
                    p.money_balance -= cost
                    p.imbalance = 0 # Imbalance is cleared
                    
                    transaction = {
                        "sender": "Aggregator",
                        "receiver": p.id,
                        "amount": float(amount_needed),
                        "price_per_kWh": grid_price,
                        "type": "GRID_buy",
                        "hour": self.hour
                    }

                # If seller (Surplus) - Must sell to the Aggregator
                elif p.imbalance < 0:
                    amount_sold = abs(p.imbalance)
                    
                    # Calculate final grid sale price: Base * (1 - margin)
                    # The penalty factor is typically NOT applied to sales revenue.
                    grid_price = current_market_price * (1 - margin)
                    earnings = amount_sold * grid_price
                    
                    p.money_balance += earnings
                    p.imbalance = 0 # Imbalance is cleared
                    
                    transaction = {
                        "sender": p.id,
                        "receiver": "Aggregator",
                        "amount": float(amount_sold),
                        "price_per_kWh": grid_price,
                        "type": "GRID_sell",
                        "hour": self.hour
                    }

                # Record transaction and add to blockchain if a transaction occurred
                if transaction:
                    p.transactions[self.hour].append(transaction)
                    tx = Transaction(sender=transaction["sender"], receiver=transaction["receiver"], 
                                    amount=transaction["amount"], price=transaction["price_per_kWh"], step=self.hour)
                    energy_chain.add_transaction(tx)