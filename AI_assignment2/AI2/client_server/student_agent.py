"""
Student Agent Implementation for River and Stones Game

This file contains the essential utilities and template for implementing your AI agent.
Your task is to complete the StudentAgent class with intelligent move selection.

Game Rules:
- Goal: Get 4 of your stones into the opponent's scoring area
- Pieces can be stones or rivers (horizontal/vertical orientation)  
- Actions: move, push, flip (stone↔river), rotate (river orientation)
- Rivers enable flow-based movement across the board

Your Task:
Implement the choose() method in the StudentAgent class to select optimal moves.
You may add any helper methods and modify the evaluation function as needed.
"""

import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2

def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"
# ==================== RIVER FLOW SIMULATION ====================

def agent_river_flow(board, rx: int, ry: int, sx: int, sy: int, player: str, 
                    rows: int, cols: int, score_cols: List[int], river_push: bool = False) -> List[Tuple[int, int]]:
    """
    Simulate river flow from a given position.
    
    Args:
        board: Current board state
        rx, ry: River entry point
        sx, sy: Source position (where piece is moving from)
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices
        river_push: Whether this is for a river push move
    
    Returns:
        List of (x, y) coordinates where the piece can end up via river flow
    """
    destinations = []
    visited = set()
    queue = [(rx, ry)]
    
    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))
        
        cell = board[y][x]
        if river_push and x == rx and y == ry:
            cell = board[sy][sx]
            
        if cell is None:
            if is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                # Block entering opponent score cell
                pass
            else:
                destinations.append((x, y))
            continue
            
        if getattr(cell, "side", "stone") != "river":
            continue
            
        # River flow directions
        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]
        
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break
                    
                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue
                    
                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue
                    
                if getattr(next_cell, "side", "stone") == "river":
                    queue.append((nx, ny))
                    break
                break
    
    # Remove duplicates
    unique_destinations = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            unique_destinations.append(d)
    
    return unique_destinations

# ==================== MOVE GENERATION HELPERS ====================
def agent_compute_valid_moves(board, sx: int, sy: int, player: str, rows: int, cols: int, score_cols: List[int]) -> Dict[str, Any]:
    """
    Compute all valid moves for a piece at position (sx, sy).
    
    Returns:
        Dictionary with 'moves' (set of coordinates) and 'pushes' (list of tuples)
    """
    if not in_bounds(sx, sy, rows, cols):
        return {'moves': set(), 'pushes': []}
        
    piece = board[sy][sx]
    if piece is None or piece.owner != player:
        return {'moves': set(), 'pushes': []}
    
    moves = set()
    pushes = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for dx, dy in directions:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue
            
        # Block moving into opponent score cell
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue
            
        target = board[ty][tx]
        
        if target is None:
            # Empty cell - direct move
            moves.add((tx, ty))
        elif getattr(target, "side", "stone") == "river":
            # River - compute flow destinations
            flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for dest in flow:
                moves.add(dest)
        else:
            # Occupied by stone - check push possibility
            if getattr(piece, "side", "stone") == "stone":
                # Stone pushing stone
                px, py = tx + dx, ty + dy
                if (in_bounds(px, py, rows, cols) and 
                    board[py][px] is None and 
                    not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                    pushes.append(((tx, ty), (px, py)))
            else:
                # River pushing - compute flow for pushed piece
                flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols, river_push=True)
                for dest in flow:
                    if not is_opponent_score_cell(dest[0], dest[1], player, rows, cols, score_cols):
                        pushes.append(((tx, ty), dest))
    
    return {'moves': moves, 'pushes': pushes}

# def get_valid_moves_for_piece(board, x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
#     """
#     Generate all valid moves for a specific piece.
    
#     Args:
#         board: Current board state
#         x, y: Piece position
#         player: Current player
#         rows, cols: Board dimensions
#         score_cols: Scoring column indices
    
#     Returns:
#         List of valid move dictionaries
#     """
#     moves = []
#     piece = board[y][x]
    
#     if piece is None or piece.owner != player:
#         return moves
    
#     directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
#     if piece.side == "stone":
#         # Stone movement
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if not in_bounds(nx, ny, rows, cols):
#                 continue
            
#             if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
#                 continue
            
#             if board[ny][nx] is None:
#                 # Simple move
#                 moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})
#             elif board[ny][nx].owner != player:
#                 # Push move
#                 px, py = nx + dx, ny + dy
#                 if (in_bounds(px, py, rows, cols) and 
#                     board[py][px] is None and 
#                     not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
#                     moves.append({"action": "push", "from": [x, y], "to": [nx, ny], "pushed_to": [px, py]})
        
#         # Stone to river flips
#         for orientation in ["horizontal", "vertical"]:
#             moves.append({"action": "flip", "from": [x, y], "orientation": orientation})
    
#     else:  # River piece
#         # River to stone flip
#         moves.append({"action": "flip", "from": [x, y]})
        
#         # River rotation
#         moves.append({"action": "rotate", "from": [x, y]})
    
#     return moves

def generate_all_moves(board, player, rows, cols, score_cols):
    moves = []
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player:
                continue

            valid = agent_compute_valid_moves(board, x, y, player, rows, cols, score_cols)
            for tx, ty in valid['moves']:
                moves.append({"action": "move", "from": [x, y], "to": [tx, ty]})
            for (pushed_from, pushed_to) in valid['pushes']:
                moves.append({"action": "push", "from": [x, y], "to": list(pushed_from), "pushed_to": list(pushed_to)})
            
            # Keep flip and rotate logic from before
            if piece.side == "stone":
                for orientation in ("horizontal", "vertical"):
                    temp = copy.deepcopy(board)
                    temp[y][x].side = "river"
                    temp[y][x].orientation = orientation
                    flow = agent_river_flow(temp, x, y, x, y, player, rows, cols, score_cols)
                    if not any(is_opponent_score_cell(dx, dy, player, rows, cols, score_cols) for dx, dy in flow):
                        moves.append({"action": "flip", "from": [x, y], "orientation": orientation})
            else:
                moves.append({"action": "flip", "from": [x, y]})
                new_orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
                temp = copy.deepcopy(board)
                temp[y][x].orientation = new_orientation
                flow = agent_river_flow(temp, x, y, x, y, player, rows, cols, score_cols)
                if not any(is_opponent_score_cell(dx, dy, player, rows, cols, score_cols) for dx, dy in flow):
                    moves.append({"action": "rotate", "from": [x, y]})
    return moves

# ==================== BOARD EVALUATION ====================

def count_stones_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    """Count how many stones a player has in their scoring area."""
    count = 0
    
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)
    
    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1
    
    return count

def basic_evaluate_board(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> float:
    """
    Advanced board evaluation function that considers:
    - Stones in scoring areas (high reward)
    - Progress toward opponent's scoring row
    - River proximity (encourages using river flow)
    - Push opportunities into rivers
    - Opponent threats near our scoring area
    """
    score = 0.0
    opponent = get_opponent(player)
    
    # 1. Scoring areas are most important
    player_scoring_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    opponent_scoring_stones = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    score += player_scoring_stones * 120
    score -= opponent_scoring_stones * 120

    if player == "circle":
        own_goal_row = rows - 3  # opponent's scoring area row
        goal_row = 2
    else:
        own_goal_row = 2
        goal_row = rows - 3

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece:
                continue

            if piece.side == "stone":
                if piece.owner == player:
                    # 2. Progress reward (closer to opponent goal row)
                    distance = abs(goal_row - y)
                    score += (rows - distance) * 0.2
                    
                    # 3. Bonus if adjacent to a friendly river (can leverage flow)
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = x + dx, y + dy
                        if in_bounds(nx, ny, rows, cols):
                            neighbor = board[ny][nx]
                            if neighbor and neighbor.owner == player and neighbor.side == "river":
                                score += 0  # incentive to use rivers
                                
                    # 4. Bonus if can push opponent into river
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = x + dx, y + dy
                        px, py = nx + dx, ny + dy
                        if in_bounds(nx, ny, rows, cols) and in_bounds(px, py, rows, cols):
                            target = board[ny][nx]
                            push_cell = board[py][px]
                            if target and target.owner == opponent and push_cell and push_cell.side == "river":
                                score += 3.0  # pushing into river is powerful
                                
                elif piece.owner == opponent:
                    # 5. Penalize opponent stones near our goal row
                    distance = abs(own_goal_row - y)
                    score -= max(0, (rows - distance) * 0.15)
                    
                    # Small penalty if opponent is near rivers that could help them
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = x + dx, y + dy
                        if in_bounds(nx, ny, rows, cols):
                            neighbor = board[ny][nx]
                            if neighbor and neighbor.owner == opponent and neighbor.side == "river":
                                score -= 1.0  # they could flow closer to our goal

            elif piece.side == "river":
                # 6. Reward rivers that are placed to help our stones advance
                if piece.owner == player:
                    if piece.orientation == "vertical":
                        score += 0.8  # vertical rivers are often better for progress
                    else:
                        score += 0.5
                else:
                    score -= 0.3  # opponent's rivers are potential threats

    return score

def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
    """
    Simulate a move on a copy of the board.
    
    Args:
        board: Current board state
        move: Move to simulate
        player: Player making the move
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        (success: bool, new_board_state or error_message)
    """
    # Import the game engine's move validation function
    try:
        from gameEngine import validate_and_apply_move
        board_copy = copy.deepcopy(board)
        success, message = validate_and_apply_move(board_copy, move, player, rows, cols, score_cols)
        return success, board_copy if success else message
    except ImportError:
        # Fallback to basic simulation if game engine not available
        return True, copy.deepcopy(board)

# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas
        
        Returns:
            Dictionary representing the chosen move, or None if no moves available
        """
        pass

# ==================== STUDENT AGENT IMPLEMENTATION ====================

class StudentAgent(BaseAgent):
    """
    Student Agent Implementation
    
    TODO: Implement your AI agent for the River and Stones game.
    The goal is to get 4 of your stones into the opponent's scoring area.
    
    You have access to these utility functions:
    - generate_all_moves(): Get all legal moves for current player
    - basic_evaluate_board(): Basic position evaluation 
    - simulate_move(): Test moves on board copy
    - count_stones_in_scoring_area(): Count stones in scoring positions
    """
    
    def __init__(self, player: str):
        super().__init__(player)
        self.last_piece_positions = {}
        # TODO: Add any initialization you need
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        moves = generate_all_moves(board, self.player, rows, cols, score_cols)
        if not moves:
            return None

        best_score = float('-inf')
        best_move = None

        for move in moves:
            piece_pos = tuple(move["from"])
            # Avoid oscillation: skip moves that return piece to its previous position
            last_pos = self.last_piece_positions.get(piece_pos)
            if last_pos and move.get("to") == last_pos:
                continue

            success, new_board = simulate_move(board, move, self.player, rows, cols, score_cols)
            if not success:
                continue

            score = basic_evaluate_board(new_board, self.player, rows, cols, score_cols)

            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            # Update last moved position for this piece
            piece_pos = tuple(best_move["from"])
            if "to" in best_move:
                self.last_piece_positions[piece_pos] = best_move["to"]

        return best_move

# ==================== TESTING HELPERS ====================

def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        
        agent = StudentAgent("circle")
        move = agent.choose(board, rows, cols, score_cols,1.0,1.0)
        
        if move:
            print("✓ Agent successfully generated a move")
        else:
            print("✗ Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("✓ StudentAgent created successfully")

if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()
