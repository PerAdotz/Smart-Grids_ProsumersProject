import hashlib
import json
import time
import random

# --- 1. TRANSACTION CLASS ---
class Transaction:
    """Represents a single energy trade transaction recorded on the blockchain."""

    def __init__(self, sender, receiver, amount, price, step):
        """
        Initializes a transaction object.

        Args:
            sender (str or int): ID of the selling Prosumer (or Aggregator/Grid).
            receiver (str or int): ID of the buying Prosumer (or Aggregator/Grid).
            amount (float): Exchanged energy amount in kWh.
            price (float): Price per kWh in currency unit (€/kWh).
            step (int): Simulation time step (hour of the day, 0-23).
        """
        self.sender = sender # ID of the selling Prosumer
        self.receiver = receiver # ID of the buying Prosumer
        self.amount = amount # Exchanged kWh
        self.price = price # Price per kWh
        self.step = step # Simulation time step (0-23)
        self.timestamp = time.time() # Time when the transaction object was created

    def to_dict(self):
        """
        Converts transaction data to a dictionary.

        Returns:
            dict: Dictionary representation of the transaction.
        """
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
    """Represents a single block in the blockchain, containing a batch of transactions."""
    
    def __init__(self, index, transactions, previous_hash, miner_id):
        """
        Initializes a block object.

        Args:
            index (int): The position of the block in the chain.
            transactions (list): List of Transaction objects included in this block.
            previous_hash (str): Hash of the preceding block in the chain.
            miner_id (str): ID of the miner node that successfully mined this block.
        """
        self.index = index
        self.transactions = transactions # List of Transaction objects
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.miner_id = miner_id # Miner ID who mined the block
        self.nonce = 0 # Nonce initialized to 0 for Proof-of-Work = the number to find via PoW
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Calculates the SHA-256 hash of the block's entire serialized content.

        Returns:
            str: The SHA-256 hash string.
        """
        # Serialize the block's data (including transactions and nonce) into a JSON string
        block_string = json.dumps({
            "index": self.index,
            # Transactions must be converted to dictionary format for hashing
            "transactions": [tx.to_dict() for tx in self.transactions],
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "miner_id": self.miner_id,
            "nonce": self.nonce
        }, sort_keys=True)
        
        # Encode the string and return the SHA-256 digest
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        """
        Implements the Proof-of-Work (PoW) consensus mechanism.
        The block is mined when its hash starts with 'difficulty' number of zeros.

        Args:
            difficulty (int): The required number of leading zeros (the target).

        Returns:
            None: Updates self.nonce and self.hash upon successful mining.
        """
        target = "0" * difficulty
        start_time = time.time()
        
        # Loop until a valid hash is found (starts with 'difficulty' number of zeros)
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.compute_hash()
            
        end_time = time.time()
        print(f"   [Mined] Block #{self.index} mined by {self.miner_id}")
        print(f"   [Hash] {self.hash}")
        print(f"   [Nonce] {self.nonce} (Time: {end_time - start_time:.4f}s)")

# --- 3. BLOCKCHAIN CLASS ---
class Blockchain:
    """Manages the full chain of blocks and the consensus process."""
    
    def __init__(self, difficulty=3):
        """
        Initializes the blockchain, setting the mining difficulty and creating the genesis block.

        Args:
            difficulty (int, optional): The difficulty level for Proof-of-Work. Defaults to 3.
        """
        self.chain = [] # The main list of blocks
        self.pending_transactions = [] # Transactions waiting to be mined
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        """Creates and mines the very first block in the chain."""
        genesis_block = Block(0, [], "0", "Genesis_System")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)

    def add_transaction(self, transaction):
        """
        Adds a new Transaction object to the list of pending transactions.

        Args:
            transaction (Transaction): The transaction to be added.

        Returns:
            None
        """
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self, miner_address):
        """
        Packages pending transactions into a new block, mines it, and appends it to the chain.

        Args:
            miner_address (str): The ID of the node that won the mining competition.

        Returns:
            None: If there are no pending transactions.
        """
        if not self.pending_transactions:
            print("   No transactions to mine in this step.")
            return

        last_block = self.chain[-1]
        
        # Create a new block candidate
        new_block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions,
            previous_hash=last_block.hash,
            miner_id=miner_address
        )

        # Mine the block (execute Proof-of-Work)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        
        # Clear the pending transactions list after successful mining
        self.pending_transactions = []

    def is_chain_valid(self):
        """
        Verifies the integrity of the entire blockchain by checking hash linking and PoW requirements.

        Returns:
            bool: True if the chain is valid, False otherwise.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check 1: Ensure the current block's stored hash is consistent with its data
            if current.hash != current.compute_hash():
                print(f"   Chain validation failed at block {current.index}: hash mismatch.")
                return False
            
            # Check 2: Ensure the current block correctly points to the previous block
            if current.previous_hash != previous.hash:
                print(f"   Chain validation failed at block {current.index}: previous hash mismatch.")
                return False
            
            # Check 3: Ensure the current block meets the PoW difficulty target
            if current.hash[:self.difficulty] != "0" * self.difficulty:
                print(f"   Chain validation failed at block {current.index}: PoW requirement not met.")
                return False
            
        return True
    
    def winner_selection(self, competitors_list , powers_list):
        """
        Probabilistically selects a winner from a list of competitors based on their hash power.

        Args:
            competitors_list (list): List of miner IDs (strings).
            powers_list (list): List of corresponding hash power values (floats).

        Returns:
            tuple: The name (str) and power (float) of the selected winning miner.
        """
        # Uses weighted random selection where weights are the hash powers
        winner_list = random.choices(competitors_list, weights=powers_list, k=1)
        winner_name = winner_list[0]
        
        # Retrieve the winner's power for verification purposes
        winner_idx = competitors_list.index(winner_name)
        winner_power = powers_list[winner_idx]

        return winner_name , winner_power

# --- 4. MINER CLASS ---
class Miner:
    """Represents a mining node with a certain computational power."""
    
    def __init__(self, miner_id):
        """
        Initializes a miner with a unique ID and a randomly assigned hash power.

        Args:
            miner_id (str): Unique identifier for the miner node.
        """
        self.miner_id = miner_id

        # Assign a random hash power, simulating variability in hardware
        self.hash_power = random.uniform(0.1, 1.0) 
    
    def Pow_compete(self):
        """
        Returns the miner's current simulated computational power (hash rate).

        Returns:
            float: The miner's hash power value.
        """
        return self.hash_power