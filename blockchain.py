#Initializing our blockchain
blockchain = []

def get_last_blockchain_value():
    if len(blockchain) < 1:
        return None
    return blockchain[-1]


def add_transaction(transaction_amount, last_transaction=[1]):
    """ WOWOWOWOWOWOWO """
    if last_transaction == None:
        last_transaction = [1]
    blockchain.append([last_transaction, transaction_amount])


def get_transaction_value():
    user_input = float(input("Your transaction amount please: "))
    return user_input
    

def get_user_choice():
    user_input = input("Your choice: ")
    return user_input


def print_blockchain_elements():
    for block in blockchain:
        print("Outputting Block")
        print(block) 


def verify_chain(): 
    is_valid = True
    for block_index in range(len(blockchain)): 
        if block_index == 0:
            # block_index += 1
            continue
        if blockchain[block_index][0] == blockchain[block_index - 1]:
            is_valid = True
        else: 
            is_valid = False
            break
        # block_index += 1
    return is_valid


waiting_for_input = True

while waiting_for_input:
    print("Please chooose:")
    print("1: Add a new transaction value")
    print("2: Print the blockchain") 
    print("h: Manipulate the chain")
    print("q: Exit")

    user_choice = get_user_choice()
    if user_choice == "1":
        tx_amount = get_transaction_value()
        add_transaction(tx_amount, get_last_blockchain_value())
    elif user_choice == "2": 
        print_blockchain_elements()
    elif user_choice == "h":
        if len(blockchain) >= 1:
            blockchain[0] = [2]
    elif user_choice == "q":
        waiting_for_input = False 
    else:
        print("Input was invalid") 

    print("-" * 30)
    if not verify_chain():
        print("CHAIN IS NOT VALID")
        break 
else:
    print("User left")

print("Done")