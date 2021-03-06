#Initializing our blockchain
genesis_block = {
        "previous_hash": "", 
        "index": 0, 
        "transactions": []
    }
blockchain = [genesis_block]
open_transactions = []  
owner = "Ema"
partecipants = set()

def hash_block(block):
    return "-".join([str(block[key]) for key in block]) 

def get_balance(partecipant):
    tx_sender = [[tx["amount"] for tx in block["transactions"] if tx["sender"] == partecipant] for block in blockchain]
    amount_sent = 0
    for tx in tx_sender:
        if len(tx) > 0:
            amount_sent += tx[0]
    tx_recipient = [[tx["amount"] for tx in block["transactions"] if tx["recipient"] == partecipant] for block in blockchain]
    amount_received = 0
    for tx in tx_recipient:
        if len(tx) > 0:
            amount_received += tx[0]
    return amount_received - amount_sent


def get_last_blockchain_value():
    if len(blockchain) < 1:
        return None
    return blockchain[-1]


def add_transaction(recipient, sender = owner, amount=1.0):
    """ WOWOWOWOWOWOWO """
    transaction = {"sender" : sender, "recipient" : recipient, "amount" : amount }
    open_transactions.append(transaction)
    partecipants.add(sender)
    partecipants.add(recipient)


def mine_block():
    last_block = blockchain[-1]
    hashed_block = hash_block(last_block)

    print(hashed_block)

    block = {
        "previous_hash": hashed_block, 
        "index": len(blockchain), 
        "transactions": open_transactions
    }
    blockchain.append(block)
    return True


def get_transaction_value():
    tx_recipient = input("Enter the recipient of the transaction: ")
    tx_amount = float(input("Your transaction amount please: "))
    return (tx_recipient, tx_amount)
    

def get_user_choice():
    user_input = input("Your choice: ")
    return user_input


def print_blockchain_elements():
    for block in blockchain:
        print("Outputting Block")
        print(block) 


def verify_chain():
    for (index, block) in  enumerate(blockchain): 
        if index == 0: 
            continue
        if block["previous_hash"] != hash_block(blockchain[index - 1]):
            return False
    return True


waiting_for_input = True

while waiting_for_input:
    print("Please chooose:")
    print("1: Add a new transaction value")
    print("2: Mine a new block") 
    print("3: Output the blockchain") 
    print("4: Output partecipants") 
    print("h: Manipulate the chain")
    print("q: Exit")

    user_choice = get_user_choice()
    if user_choice == "1":
        tx_data = get_transaction_value()
        recipient, amount = tx_data
        add_transaction(recipient, amount=amount)
    elif user_choice == "2": 
        if mine_block():
            open_transactions = []
    elif user_choice == "3": 
        print_blockchain_elements() 
    elif user_choice == "4":
        print(partecipants) 
    elif user_choice == "h":
        if len(blockchain) >= 1:
            blockchain[0] = {
        "previous_hash": "", 
        "index": 0, 
        "transactions": [{"sender": "Chris", "recipient": "Max", "amount": "1000"}]
    }
    elif user_choice == "q":
        waiting_for_input = False 
    else:
        print("Input was invalid")

    if not verify_chain():
        print("CHAIN IS NOT VALID")
        break 
    print(get_balance("Ema"))
    print("-" * 30)
else:
    print("User left")

print("Done")
