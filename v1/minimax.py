import numpy as np

def result(board):
    if board[0] == board[1] == board[2] == 1 or \
            board[3] == board[4] == board[5] == 1 or \
            board[6] == board[7] == board[8] == 1 or \
            board[0] == board[3] == board[6] == 1 or \
            board[1] == board[4] == board[7] == 1 or \
            board[2] == board[5] == board[8] == 1 or \
            board[0] == board[4] == board[8] == 1 or \
            board[2] == board[4] == board[6] == 1:
        return 1
    if board[0] == board[1] == board[2] == 2 or \
            board[3] == board[4] == board[5] == 2 or \
            board[6] == board[7] == board[8] == 2 or \
            board[0] == board[3] == board[6] == 2 or \
            board[1] == board[4] == board[7] == 2 or \
            board[2] == board[5] == board[8] == 2 or \
            board[0] == board[4] == board[8] == 2 or \
            board[2] == board[4] == board[6] == 2:
        return 2
    if sum(board == 0) == 0: return 3
    return 0

def minimax(maxi, board):
    if result(board)!=0:
        if result(board)==1:
            return 1
        elif result(board)==2:
            return -1
        else:
            return 0

    score = []
    if maxi==0: val=1
    else: val=2
    for i in range(9):
        if board[i]!=0: continue
        board[i] = val
        score.append(minimax(not maxi, board))
        board[i] = 0

    if maxi:
        return max(score)
    else:
        return min(score)

arr = np.zeros(9)
for i in range(9):
    arr[i] = 1
    print(i, minimax(0, arr))
    arr[i] = 0