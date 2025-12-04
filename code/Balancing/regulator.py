class Regulator:
    # it' just a mock class for now doeant't do anything meaningful
    def __init__(self, objective="Maximize_P2P"):
        self.objective = objective
        self.history_log = []

    def apply_regulations(self, prosumers, current_hour):
        print(f"  [Regulator] Auditing Hour {current_hour}...")
        
        
        for p in prosumers:
            for tx in p.transactions[current_hour]:
                # Se la transazione non è di questa ora, ignorala
                if tx['hour'] != current_hour:
                    continue

                if tx['type'] == 'P2P':
                    if p.p2p_exchanges < 5:  # limit bonuses to first 5 P2P exchanges
                        p.bonus = 1.02 # 2% bonus for P2P trading, can gain a bit more when selling to the grid
                    elif p.p2p_exchanges > 5 and p.p2p_exchanges < 10:
                        p.bonus = 1.05 # 5% bonus for P2P trading, after 5 exchanges
                    elif p.p2p_exchanges >= 10:
                        p.bonus = 1.10 # 10% bonus for P2P trading, after 10 exchanges
                    p.p2p_exchanges += 1
                    
                elif tx['type'] == 'GRID_buy': # penalize only grid buys bc we do just one loop of P2P then grid interaction
                    if current_hour > 18 or current_hour < 6:
                        p.penalty = 1.10 # 10% penalty for grid buys during peak hours (6 PM to 6 AM)
                    else:
                        if p.agg_exchanges < 5:
                            p.penalty = 1.02 # 2% penalty for grid buys during off-peak hours
                        elif p.agg_exchanges >=5 and p.agg_exchanges <10:   
                            p.penalty = 1.05 # 5% penalty for grid buys during off-peak hours
                        elif p.agg_exchanges >=10:
                            p.penalty = 1.07 # 7% penalty for grid buys during off-peak hours
                    p.agg_exchanges += 1
                        

