import hashlib
import json
import time
import random

# --- 1. TRANSACTION CLASS ---
class Transaction:
    def __init__(self, sender, receiver, amount, price, step):
        self.sender = sender      # ID of the selling Prosumer
        self.receiver = receiver  # ID of the buying Prosumer (or Aggregator)
        self.amount = amount      # Exchanged kWh
        self.price = price        # Price per kWh
        self.step = step          # Simulation time step (0-23)
        self.timestamp = time.time()

    def to_dict(self):
        """Converts transaction data to a dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "price": self.price,
            "step": self.step,
            "timestamp": self.timestamp
        }

# --- 2. BLOCK CLASS ---
class Block:
    def __init__(self, index, transactions, previous_hash, miner_id):
        self.index = index
        self.transactions = transactions  # List of Transaction objects
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.miner_id = miner_id          # Miner ID who mined the block
        self.nonce = 0                    # The number to find via PoW
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Creates the SHA-256 hash of the block content.
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
        Proof-of-Work implementation.
        """
        target = "0" * difficulty
        start_time = time.time()
        
        # Try to find a nonce that results in a hash starting with 'difficulty' zeros
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.compute_hash()
            
        end_time = time.time()
        print(f"   [Mined] Block #{self.index} mined by {self.miner_id}")
        print(f"   [Hash] {self.hash}")
        print(f"   [Nonce] {self.nonce} (Time: {end_time - start_time:.4f}s)")

# --- 3. BLOCKCHAIN CLASS ---
class Blockchain:
    def __init__(self, difficulty=3):
        self.chain = []
        self.pending_transactions = [] 
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        """Creates the first block in the chain."""
        genesis_block = Block(0, [], "0", "Genesis_System")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)

    def add_transaction(self, transaction):
        """Adds a transaction to the list of pending transactions."""
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self, miner_address):
        """
        Packages pending transactions into a block and adds it to the chain.
        """
        if not self.pending_transactions:
            print("   No transactions to mine in this step.")
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
        # Clear the pending transactions list
        self.pending_transactions = []

    def is_chain_valid(self):
        """Checks the integrity of the blockchain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check if the stored hash is correct
            if current.hash != current.compute_hash():
                return False
            # Check if the block points to the correct previous hash
            if current.previous_hash != previous.hash:
                return False
            # Check if the block meets the difficulty requirement
            if current.hash[:self.difficulty] != "0" * self.difficulty:
                return False
        return True
    
    def winner_selection(self, competitors_list , powers_list):
        winner_list = random.choices(competitors_list, weights=powers_list, k=1)
        winner_name = winner_list[0]
        
        # Retrieve the winner's power just for printing verification
        winner_idx = competitors_list.index(winner_name)
        winner_power = powers_list[winner_idx]
        return winner_name , winner_power

# --- 4. MINER CLASS ---
class Miner:
    def __init__(self, miner_id):
        self.miner_id = miner_id
        # Every time a miner is created, it has a slightly different power
        # This simulates hardware fluctuations or different stakes
        self.hash_power = random.uniform(0.1, 1.0) 
    
    def Pow_compete(self):
        """Returns the simulated computational power (or Stake amount)."""
        return self.hash_power

# --- 5. SIMULATION CONFIGURATION ---
# def run_simulation():
#     DIFFICULTY = 3
#     NUM_PROSUMERS = 100
#     NUM_MINERS = 10 
#     SIMULATION_STEPS = 24
    
#     energy_chain = Blockchain(difficulty=DIFFICULTY)
    
#     # Create miner names list ONCE
#     miners_names = [f"Miner_Node_{i}" for i in range(1, NUM_MINERS + 1)]
    
#     print(f"--- START OF SMART GRID BLOCKCHAIN SIMULATION ---")
#     print(f"PoW Target: {DIFFICULTY} leading zeros")
#     print(f"Total Steps: {SIMULATION_STEPS}\n")

#     for step in range(1, SIMULATION_STEPS + 1):
#         print(f"\n--- TIME STEP {step}/{SIMULATION_STEPS} ---")
        
#         # A. TRANSACTION GENERATION PHASE (Simulated)
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
            
#         print(f"   {num_transactions} transactions added to Mempool.")

#         # ==========================================
#         # B. MINING PHASE (CONSENSUS) - PROBABILISTIC
#         # ==========================================
        
#         competitors_names = []
#         competitors_weights = [] 

#         # 1. Each miner calculates their current power/stake
#         for miner_name in miners_names:
#             m = Miner(miner_name)
#             power = m.Pow_compete() # Returns a value between 0.1 and 1.0
            
#             competitors_names.append(miner_name)
#             competitors_weights.append(power)
        
#         # 2. PROBABILISTIC Winner Selection
#         # random.choices selects a winner based on weights.
#         # Higher weight = Higher probability, but not guaranteed win.
#         winner_list = random.choices(competitors_names, weights=competitors_weights, k=1)
#         winner_name = winner_list[0]
        
#         # Retrieve the winner's power just for printing verification
#         winner_idx = competitors_names.index(winner_name)
#         winner_val = competitors_weights[winner_idx]
        
#         print(f"   Winning Miner: {winner_name} (Hash Power/Stake: {winner_val:.4f})")
        
#         # The winner actually mines the block and adds it to the chain
#         energy_chain.mine_pending_transactions(winner_name)

#     # --- FINAL VERIFICATION ---
#     print("\n--- END OF SIMULATION ---")
#     print(f"Chain Length: {len(energy_chain.chain)} blocks")
    
#     is_valid = energy_chain.is_chain_valid()
#     status = "VALID" if is_valid else "CORRUPTED"
#     print(f"Blockchain Integrity: {status}")
    
#     last_block = energy_chain.chain[-1]
#     print("\n[Last Block Details]:")
#     print(json.dumps({
#         "Index": last_block.index,
#         "Hash": last_block.hash,
#         "PrevHash": last_block.previous_hash,
#         "Transactions": len(last_block.transactions)
#     }, indent=4))

# if __name__ == "__main__":
#     run_simulation()