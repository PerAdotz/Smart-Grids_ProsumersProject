class Regulator:
    def __init__(self, objective="Maximize_P2P"):
        self.objective = objective
        self.history_log = []

    def apply_regulations(self, prosumers, current_hour):
        print(f"  [Regulator] Auditing Hour {current_hour}...")
        
        total_p2p = 0
        total_grid = 0
        
        for p in prosumers:
            # CORREZIONE: Iteriamo su TUTTE le transazioni, ma agiamo solo su quelle di ORA
            for tx in p.transactions[current_hour]:
                # Se la transazione non è di questa ora, ignorala
                if tx['hour'] != current_hour:
                    continue

                # --- APPLICA REGOLE ---
                if tx['type'] == 'P2P':
                    bonus = tx['amount'] * 0.08
                    p.money_balance += bonus
                    total_p2p += tx['amount']
                    
                elif 'GRID' in tx['type']: # Grid_Buy o Grid_Sell
                    if current_hour > 18 or current_hour < 6:  
                        penalty = tx['amount'] * 0.01
                        p.money_balance -= penalty
                        total_grid += tx['amount']


        self.history_log.append({
            "hour": current_hour,
            "total_p2p": total_p2p,
            "total_grid": total_grid
        })
        print(f"  [Regulator] Hour {current_hour}: Reward P2P: {total_p2p:.2f} kWh | Penalty Grid: {total_grid:.2f} kWh")