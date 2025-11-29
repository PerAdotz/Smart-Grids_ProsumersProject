import hashlib
import json
import time
import random

# --- 1. CLASSE TRANSAZIONE --- 
class Transaction:
    def __init__(self, sender, receiver, amount, price, step):
        self.sender = sender      # ID del Prosumer che vende
        self.receiver = receiver  # ID del Prosumer che compra (o Aggregatore)
        self.amount = amount      # kWh scambiati
        self.price = price        # Prezzo per kWh
        self.step = step          # Time step della simulazione (0-23)
        self.timestamp = time.time()

    def to_dict(self):
        """Converte l'oggetto in dizionario per la serializzazione JSON"""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "price": self.price,
            "step": self.step,
            "timestamp": self.timestamp
        }

# --- 2. CLASSE BLOCCO ---
class Block:
    def __init__(self, index, transactions, previous_hash, miner_id):
        self.index = index
        self.transactions = transactions  # Lista di oggetti Transaction
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.miner_id = miner_id          # Chi ha minato il blocco
        self.nonce = 0                    # Il numero da trovare col PoW
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Crea l'hash SHA-256 del blocco.
        """
        block_string = json.dumps({
            "index": self.index,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "miner_id": self.miner_id,
            "nonce": self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        """
        Implementazione del Proof-of-Work.
        """
        target = "0" * difficulty
        start_time = time.time()
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.compute_hash()
            
        end_time = time.time()
        print(f"   [Minato] Blocco #{self.index} minato da {self.miner_id}")
        print(f"   [Hash] {self.hash}")
        print(f"   [Nonce] {self.nonce} (Tempo: {end_time - start_time:.4f}s)")

# --- 3. CLASSE BLOCKCHAIN ---
class Blockchain:
    def __init__(self, difficulty=3):
        self.chain = []
        self.pending_transactions = [] 
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, [], "0", "Genesis_System")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self, miner_address):
        if not self.pending_transactions:
            print("   Nessuna transazione da minare in questo step.")
            return

        last_block = self.chain[-1]
        
        new_block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions,
            previous_hash=last_block.hash,
            miner_id=miner_address
        )

        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.pending_transactions = []

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            if current.hash != current.compute_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
            if current.hash[:self.difficulty] != "0" * self.difficulty:
                return False
        return True

# --- 4. CLASSE MINER ---
class Miner:
    def __init__(self, miner_id):
        self.miner_id = miner_id
        # Ogni volta che creiamo il miner, avrà una potenza leggermente diversa 
        # o statica, a tua scelta. Qui la lascio randomica per simulare fluttuazioni.
        self.hash_power = random.uniform(0.1, 1.0) 
    
    def Pow_compete(self):
        """Restituisce la potenza di calcolo simulata"""
        return self.hash_power

# # --- 5. CONFIGURAZIONE SIMULAZIONE ---
# def run_simulation():
#     DIFFICULTY = 3
#     NUM_PROSUMERS = 100
#     NUM_MINERS = 10 
#     SIMULATION_STEPS = 24
    
#     energy_chain = Blockchain(difficulty=DIFFICULTY)
    
#     # Creiamo la lista di nomi dei miner UNA SOLA VOLTA
#     miners_names = [f"Miner_Node_{i}" for i in range(1, NUM_MINERS + 1)]
    
#     print(f"--- INIZIO SIMULAZIONE BLOCKCHAIN SMART GRID ---")
#     print(f"Target PoW: {DIFFICULTY} zeri iniziali")
#     print(f"Step totali: {SIMULATION_STEPS}\n")

#     for step in range(1, SIMULATION_STEPS + 1):
#         print(f"\n--- STEP TEMPORALE {step}/{SIMULATION_STEPS} ---")
        
#         # A. FASE DI GENERAZIONE TRANSAZIONI
#         num_transactions = random.randint(5, 15)
#         for _ in range(num_transactions):
#             sender = f"Prosumer_{random.randint(1, NUM_PROSUMERS)}"
#             receiver = f"Prosumer_{random.randint(1, NUM_PROSUMERS)}"
#             while receiver == sender:
#                 receiver = f"Prosumer_{random.randint(1, NUM_PROSUMERS)}"
#             amount = round(random.uniform(0.5, 5.0), 2)
#             price = round(random.uniform(0.10, 0.30), 3)
            
#             tx = Transaction(sender, receiver, amount, price, step)
#             energy_chain.add_transaction(tx)
            
#         print(f"   {num_transactions} transazioni aggiunte alla Mempool.")

#         # B. FASE DI MINING (CONSENSO) - CORRETTA
#         # Creiamo una lista temporanea per i risultati di QUESTA gara
#         current_round_competitors = []

#         for miner_name in miners_names:
#             # Istanziamo il miner
#             m = Miner(miner_name)
#             # Calcoliamo la sua potenza attuale
#             power = m.Pow_compete()
#             # Salviamo il risultato
#             current_round_competitors.append((miner_name, power))
        
#         # Troviamo chi ha il valore 'hash_power' più alto (simulazione di chi trova prima il blocco)
#         winner_name, winner_power = max(current_round_competitors, key=lambda x: x[1])
        
#         print(f"   Miner vincente: {winner_name} (Hash Power: {winner_power:.4f})")
        
#         # Il vincitore mina il blocco reale
#         energy_chain.mine_pending_transactions(winner_name)

#     # --- VERIFICA FINALE ---
#     print("\n--- FINE SIMULAZIONE ---")
#     print(f"Lunghezza Catena: {len(energy_chain.chain)} blocchi")
    
#     is_valid = energy_chain.is_chain_valid()
#     print(f"Integrità Blockchain: {'VALIDA' if is_valid else 'CORROTTA'}")
    
#     last_block = energy_chain.chain[-1]
#     print("\n[Dettaglio Ultimo Blocco]:")
#     print(json.dumps({
#         "Index": last_block.index,
#         "Hash": last_block.hash,
#         "PrevHash": last_block.previous_hash,
#         "Transactions": len(last_block.transactions)
#     }, indent=4))

# if __name__ == "__main__":
#     run_simulation()


'''Per integrare il tutto, importa la classe Blockchain nel tuo file principale della simulazione (quello

dove gestisci i prosumer e le strategie del regolatore) e istanzia un oggetto Blockchain all'inizio.

Ogni volta che due prosumer si accordano per uno scambio, chiama add_transaction.

Alla fine di ogni "Step" temporale, chiama mine_pending_transactions.'''