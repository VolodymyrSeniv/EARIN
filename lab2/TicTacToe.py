import math

# TicTacToe board
board = [' ',' ',' ',
         ' ',' ',' ',
         ' ',' ',' ']

# Function to display the board
def display_board(board):
    print()
    print('reference:')

    print('0' + '|' + '1' + '|' + '2')
    print('-+-+-')
    print('3' + '|' + '4' + '|' + '5')
    print('-+-+-')
    print('6' + '|' + '7' + '|' + '8')
    print()
    print('board:')
    print(board[0] + '|' + board[1] + '|' + board[2])
    print('-+-+-')
    print(board[3] + '|' + board[4] + '|' + board[5])
    print('-+-+-')
    print(board[6] + '|' + board[7] + '|' + board[8])

# Function to check if the board is full
def is_board_full(board):
    return not(' ' in board)

# Function to check all win conditions
def check_for_win(board, player):
    # Check rows
    for i in range(0,9,3):
        if board[i]==player and board[i+1]==player and board[i+2]==player:
            return True
    # Check columns
    for i in range(3):
        if board[i]==player and board[i+3]==player and board[i+6]==player:
            return True
    # Check diagonals
    if board[0]==player and board[4]==player and board[8]==player:
        return True
    if board[2]==player and board[4]==player and board[6]==player:
        return True
    return False

# Function to get all available spots
def get_available_spots(board):
    return [i for i in range(len(board)) if board[i]==' ']

# Minimax function with alpha-beta pruning
# it takes as arguments board, depth(insignificant in case of TicTacToe), alpha and beta are indicators to when we can prune.
# 'alpha = -math.inf'  worst condition for maximizer. 'beta = math.inf' worst condition for minimizer
def minimax(board, depth, alpha, beta, is_maximizing):
    # Check if there is a winner
    if check_for_win(board, 'X'):
        return -1
    elif check_for_win(board, 'O'):
        return 1
    elif is_board_full(board):
        return 0

    # Maximizing player's turn
    if is_maximizing:
        best_score = -math.inf
        for spot in get_available_spots(board):
            board[spot] = 'O'
            # recursively simulating all possible moves and scoring them based on the outcome of the game.
            score = minimax(board, depth+1, alpha, beta, False)
            board[spot] = ' '
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        # return the best score for the current state of the board.
        return best_score

    # Minimizing player's turn
    else:
        best_score = math.inf
        for spot in get_available_spots(board):
            board[spot] = 'X'
            score = minimax(board, depth+1, alpha, beta, True)
            board[spot] = ' '
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score

# Function to get best move for the computer
def get_computer_move(board,mode):
    if mode == 'X' or mode == 'contX':
        # if computer plays second, he is minimizing. Same principle as described in minimax function
        best_score = -math.inf
        best_move = None
        for spot in get_available_spots(board):
            board[spot] = 'O'
            score = minimax(board, 0, -math.inf, math.inf, False)
            board[spot] = ' '
            if score > best_score:
                best_score = score
                best_move = spot
        return best_move
    elif mode == 'O' or mode == 'contO':
        # if computer plays first, he is maximizing. Same principle as described in minimax function
        best_score = math.inf
        best_move = None
        for spot in get_available_spots(board):
            board[spot] = 'X'
            score = minimax(board, 0, -math.inf, math.inf, True)
            board[spot] = ' '
            if score < best_score:
                best_score = score
                best_move = spot
        return best_move

# our main function that starts the game.
# argument 'mode' specifies who plays first and who second. 'X' - player plays first, 'O' - player plays second
def play_game(mode):
    display_board(board)
    while not is_board_full(board):
    
        # Player's turn
        if mode == 'X' or mode == 'contX':
            player_move = int(input('Your turn (enter a number from 0 to 8): '))
            # check if we put our symbol on right position
            if board[player_move] != ' ':
                print('Invalid move. Try again.')
                continue
            board[player_move] = 'X'
            display_board(board)
            # check for win or tie
            if check_for_win(board, 'X'):
                print('You win!')
                return
            if is_board_full(board):
                print('Tie!')
                return
            mode = 'contX'
        # Computer's turn
            print('Computer\'s turn')
            # determining next move for computer
            computer_move = get_computer_move(board,mode)
            board[computer_move] = 'O'
            display_board(board)
            if check_for_win(board, 'O'):
                print('You lose!')
                return

        # same principle as previously, but in this scenario computer starts first
        elif mode == 'O' or mode == 'contO':
            print('Computer\'s turn')
            computer_move = get_computer_move(board,mode)
            board[computer_move] = 'X'
            display_board(board)
            if check_for_win(board, 'X'):
                print('You lose!')
                return
            elif is_board_full(board):
                print('Tie!')
                return
            player_move = int(input('Your turn (enter a number from 0 to 8): '))
            while board[player_move] != ' ':
                print('Invalid move. Try again.')
                player_move = int(input('Your turn (enter a number from 0 to 8): '))
                if board[player_move] == ' ':
                    break
            board[player_move] = 'O'
            display_board(board)
            if check_for_win(board, 'O'):
                print('You win!')
                return
            if is_board_full(board):
                print('Tie!')
                return
            mode = 'contO'
    # when we are out of space it's a tie
    print('Tie!')


# start of program, and interaction
print('Tic Tac Toe')
print('please type X or O. X goes first, O goes second:')
mode = input()
while True:
    if mode == 'O':
        break
    elif mode == 'X':
        break
    else:
        print('wrong input')
        print('please type X or O. X goes first, O goes second:')
        mode = input()
play_game(mode)