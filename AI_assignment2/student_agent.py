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

def count_reachable_in_one_move(board: List[List[Any]], player:str, rows:int, cols:int, score_cols: List[int])-> int:
    """
    m_self: count the number of player's pieces that can become a scoring stone 
    in the player's scoring area in one move
    """
    m = 0
    # Create a set of unique pieces that have already been found to be reachable
    reachable_pieces = set()

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player or (x, y) in reachable_pieces:
                continue

            # Case 1: A river piece is already in the scoring area. It can be flipped.
            if piece.side == "river" and is_own_score_cell(x, y, player, rows, cols, score_cols):
                m += 1
                reachable_pieces.add((x, y))
                continue
            
            # Case 2: A stone piece can move or be pushed into the scoring area.
            if piece.side == "stone":
                # Don't check pieces that are already scoring
                if is_own_score_cell(x, y, player, rows, cols, score_cols):
                    continue

                # Get all possible move/push destinations for this stone
                targets = agent_compute_valid_moves(board, x, y, player, rows, cols, score_cols)
                
                # Check direct moves
                for (tx, ty) in targets.get('moves', set()):
                    if is_own_score_cell(tx, ty, player, rows, cols, score_cols):
                        m += 1
                        reachable_pieces.add((x, y))
                        break 
                
                if (x, y) in reachable_pieces:
                    continue

                for _, (ptx, pty) in targets.get('pushes', []):
                    pass 
                        
    return m

def get_perimeter_squares(goal_row: int, score_cols: List[int], rows: int, cols: int) -> set:
    """
    Helper function to get the 10 key squares adjacent to a scoring zone.
    This includes the squares directly above, below, and to the sides of the 4-cell zone.
    """
    perimeter = set()
    # Squares above and below the scoring zone
    for x in score_cols:
        if in_bounds(x, goal_row - 1, rows, cols): perimeter.add((x, goal_row - 1))
        if in_bounds(x, goal_row + 1, rows, cols): perimeter.add((x, goal_row + 1))
    
    # Squares on the immediate left and right of the scoring zone
    left_x, right_x = min(score_cols) - 1, max(score_cols) + 1
    if in_bounds(left_x, goal_row, rows, cols): perimeter.add((left_x, goal_row))
    if in_bounds(right_x, goal_row, rows, cols): perimeter.add((right_x, goal_row))
    
    return perimeter

def _get_defensive_perimeter_zones(goal_row: int, score_cols: List[int], rows: int, cols: int) -> tuple[set, set]:
    """
    Splits the 10-square defensive perimeter into two distinct zones:
    1. The 8 squares directly in front of/behind the goal.
    2. The 2 squares on the immediate sides of the goal.
    Returns:
        (front_back_squares, side_squares)
    """
    front_back_squares = set()
    side_squares = set()
    
    # Zone 1: The 8 squares above and below the scoring zone
    for x in score_cols:
        if in_bounds(x, goal_row - 1, rows, cols): front_back_squares.add((x, goal_row - 1))
        if in_bounds(x, goal_row + 1, rows, cols): front_back_squares.add((x, goal_row + 1))
    
    # Zone 2: The 2 squares on the immediate left and right
    left_x, right_x = min(score_cols) - 1, max(score_cols) + 1
    if in_bounds(left_x, goal_row, rows, cols): side_squares.add((left_x, goal_row))
    if in_bounds(right_x, goal_row, rows, cols): side_squares.add((right_x, goal_row))
    
    return front_back_squares, side_squares

def basic_evaluate_board(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> float:
    """
    A corrected, optimized, and feature-rich evaluation function.
    """
    opponent = get_opponent(player)

    # --- Tunable Weights (with a more balanced suggestion for defense) ---
    W_NSCORE = 1e12
    W_MSCORE = 25.0
    W_PROGRESS = 8.5
    W_CENTER_CONTROL = 1.0
    W_FLANK_RIVER_CONTROL = 0.5 
    W_FLANK_STONE_CONTROL = 0.4
    W_RIVER_THREAT = 8.0
    W_RIVER_POS = 1.0
    W_ATTACK_STONE = 8.5
    W_ATTACK_RIVER = 2.5
    W_DEFEND_STONE = 55.0
    W_DEFEND_RIVER_CORRECT = 115.0
    W_DEFEND_RIVER_WRONG = -65.0
    W_SUPPORTED_ATTACKER = 4.0 # Set to a non-zero value to be active.
    W_DISRUPTIVE_PUSH = 6.0

    # --- Initialize All Feature Scores ---
    f_nscore_self, f_nscore_opp = 0, 0
    f_mscore_self, f_mscore_opp = 0, 0
    f_progress_self, f_progress_opp = 0, 0
    f_center_control_self, f_center_control_opp = 0, 0
    f_flank_stone_self, f_flank_river_self, f_flank_stone_opp, f_flank_river_opp = 0, 0, 0, 0
    f_river_threat_self, f_river_threat_opp = 0, 0
    f_river_pos_self, f_river_pos_opp = 0, 0
    f_attack_stone_self, f_attack_river_self, f_attack_stone_opp, f_attack_river_opp = 0, 0, 0, 0
    f_defend_stone_self, f_defend_stone_opp = 0, 0
    f_defend_river_correct_self, f_defend_river_wrong_self, f_defend_river_correct_opp, f_defend_river_wrong_opp = 0, 0, 0, 0
    f_supported_attacker_self, f_supported_attacker_opp = 0, 0
    f_disruptive_push_self, f_disruptive_push_opp = 0, 0

    # --- Pre-computation ---
    if player == "circle":
        player_goal_row, own_goal_row = top_score_row(), bottom_score_row(rows)
    else:
        player_goal_row, own_goal_row = bottom_score_row(rows), top_score_row()
    
    center_cols = set(range(cols // 2 - 2, cols // 2 + 2))
    flank_cols = {0, 1, 2, cols - 3, cols - 2, cols - 1} # Corrected to include column 0
    attack_perimeter = get_perimeter_squares(player_goal_row, score_cols, rows, cols)
    defense_perimeter = get_perimeter_squares(own_goal_row, score_cols, rows, cols)
    defense_front_back, defense_sides = _get_defensive_perimeter_zones(own_goal_row, score_cols, rows, cols)
    attack_front_back, attack_sides = _get_defensive_perimeter_zones(player_goal_row, score_cols, rows, cols)

    # --- Pre-Loop Calculations (N, M, Disruptive Pushes) ---
    f_nscore_self = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    f_nscore_opp = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    if f_nscore_self >= 4: return float('inf')
    if f_nscore_opp >= 4: return float('-inf')
    f_mscore_self = count_reachable_in_one_move(board, player, rows, cols, score_cols)
    f_mscore_opp = count_reachable_in_one_move(board, opponent, rows, cols, score_cols)
    

    # OPTIMIZATION: Calculate all disruptive pushes 
    for y_start in range(rows):
        for x_start in range(cols):
            piece_to_move = board[y_start][x_start]
            if not piece_to_move: continue
            
            targets = agent_compute_valid_moves(board, x_start, y_start, piece_to_move.owner, rows, cols, score_cols)
            for (pushed_from, pushed_to) in targets.get('pushes', []):
                pushed_piece = board[pushed_from[1]][pushed_from[0]]
                if pushed_piece and pushed_piece.owner != piece_to_move.owner:
                    if (is_own_score_cell(pushed_to[0], pushed_to[1], pushed_piece.owner, rows, cols, score_cols) or
                        pushed_to[0] == 0 or pushed_to[0] == cols - 1):
                        if piece_to_move.owner == player: f_disruptive_push_self += 1
                        else: f_disruptive_push_opp += 1

    # --- Main Board Iteration for Static Features ---
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece: continue

            is_own_piece = piece.owner == player
            pos = (x, y)

            # Center and Flank Control
            if x in center_cols:
                if is_own_piece: f_center_control_self += 1 
                else: f_center_control_opp += 1
            elif x in flank_cols:
                if piece.side == "stone":
                    if is_own_piece: f_flank_stone_self += 1 
                    else: f_flank_stone_opp += 1
                else:
                    if is_own_piece: f_flank_river_self += 1 
                    else: f_flank_river_opp += 1

            # Perimeter Control (Attack and Defense)
            if pos in attack_perimeter:
                if piece.side == "stone":
                    if is_own_piece: f_attack_stone_self += 1
                    else: f_defend_stone_opp += 1
                else:
                    if is_own_piece: f_attack_river_self += 1
            if pos in defense_perimeter:
                if piece.side == "stone":
                    if is_own_piece: f_defend_stone_self += 1
                    else: f_attack_stone_opp += 1

            # Stone-specific features
            if piece.side == "stone":
                goal_row = player_goal_row if is_own_piece else own_goal_row
                if is_own_piece: 
                    f_progress_self += (rows - abs(y - goal_row))
                    for dx, dy in [(0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if in_bounds(nx, ny, rows, cols):
                            neighbor = board[ny][nx]
                            if neighbor and neighbor.owner == player and neighbor.side == "river" and neighbor.orientation == "vertical":
                                f_supported_attacker_self += 1
                else: # Opponent's stone
                    f_progress_opp += (rows - abs(y - goal_row))
                    for dx, dy in [(0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if in_bounds(nx, ny, rows, cols):
                            neighbor = board[ny][nx]
                            if neighbor and neighbor.owner == opponent and neighbor.side == "river" and neighbor.orientation == "vertical":
                                f_supported_attacker_opp += 1
            
            # River-specific features
            elif piece.side == "river":
                if is_own_piece:
                    f_river_pos_self += (rows - abs(y - player_goal_row))
                else:
                    f_river_pos_opp += (rows - abs(y - player_goal_row))

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, rows, cols):
                        neighbor = board[ny][nx]
                        if neighbor and neighbor.side == "stone":
                            if neighbor.owner != player: f_river_threat_self += 1
                            else: f_river_threat_opp += 1
                
                if pos in defense_front_back:
                    if is_own_piece:
                        if piece.orientation == 'horizontal': f_defend_river_correct_self += 1
                        else: f_defend_river_wrong_self += 1
                elif pos in defense_sides:
                    if is_own_piece:
                        if piece.orientation == 'vertical': f_defend_river_correct_self += 1
                        else: f_defend_river_wrong_self += 1
                if pos in attack_front_back:
                    if not is_own_piece:
                        if piece.orientation == 'horizontal': f_defend_river_correct_opp += 1
                        else: f_defend_river_wrong_opp += 1
                elif pos in attack_sides:
                    if not is_own_piece:
                        if piece.orientation == 'vertical': f_defend_river_correct_opp += 1
                        else: f_defend_river_wrong_opp += 1

    # --- Final Weighted Sum Calculation ---
    score = 0
    score += W_NSCORE * (f_nscore_self - f_nscore_opp)
    score += W_MSCORE * (f_mscore_self - f_mscore_opp)
    score += W_PROGRESS * (f_progress_self - f_progress_opp)
    score += W_CENTER_CONTROL * (f_center_control_self - f_center_control_opp)
    score += W_FLANK_STONE_CONTROL * (f_flank_stone_self - f_flank_stone_opp)
    score += W_FLANK_RIVER_CONTROL * (f_flank_river_self - f_flank_river_opp)
    score += W_RIVER_THREAT * (f_river_threat_self - f_river_threat_opp)
    score += W_RIVER_POS * (f_river_pos_self - f_river_pos_opp)
    score += W_ATTACK_STONE * (f_attack_stone_self - f_attack_stone_opp)
    score += W_ATTACK_RIVER * (f_attack_river_self - f_attack_river_opp)
    score += W_DEFEND_STONE * (f_defend_stone_self - f_defend_stone_opp)
    score += W_SUPPORTED_ATTACKER * (f_supported_attacker_self - f_supported_attacker_opp)
    score += W_DISRUPTIVE_PUSH * (f_disruptive_push_self - f_disruptive_push_opp)
    score += W_DEFEND_RIVER_CORRECT * (f_defend_river_correct_self - f_defend_river_correct_opp)
    score += W_DEFEND_RIVER_WRONG * (f_defend_river_wrong_self - f_defend_river_wrong_opp)

    return score

def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
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
    Student Agent with active position repetition avoidance.
    """

    def __init__(self, player: str):
        super().__init__(player)
        self.last_move_info = {}
        self.seen_positions = []
        self.max_history = 20 

    def board_hash(self, board: List[List[Any]]) -> str:
        return str([[str(cell) if cell else '.' for cell in row] for row in board])

    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        moves = generate_all_moves(board, self.player, rows, cols, score_cols)
        if not moves:
            return None

        current_hash = self.board_hash(board)
        move_candidates = []

        for move in moves:
            success, new_board = simulate_move(board, move, self.player, rows, cols, score_cols)
            if not success:
                continue

            new_hash = self.board_hash(new_board)
            score = basic_evaluate_board(new_board, self.player, rows, cols, score_cols)
            move_candidates.append((score, move, new_hash))

        if not move_candidates:
            return None

        # Sort moves by score (best first)
        move_candidates.sort(key=lambda x: x[0], reverse=True)
        non_repeating = [(s, m, h) for s, m, h in move_candidates if h not in self.seen_positions]

        if non_repeating:
            ranked_moves = non_repeating
        else:
            ranked_moves = move_candidates

        # --- Selection with probability ---
        chosen_score, chosen_move, chosen_hash = ranked_moves[0]  # default best move
        if len(ranked_moves) > 1 and random.random() < 0.1:
            # pick randomly from next 9 best moves (excluding the top one)
            alt_moves = ranked_moves[1:10]
            if alt_moves:
                chosen_score, chosen_move, chosen_hash = random.choice(alt_moves)

        self.last_move_info.clear()
        if chosen_move:
            fr = tuple(chosen_move.get("from", []))
            to = tuple(chosen_move.get("to", []))
            if fr and to:
                self.last_move_info[to] = fr

        # Track new board hash
        self.seen_positions.append(chosen_hash)
        if len(self.seen_positions) > self.max_history:
            self.seen_positions.pop(0)

        return chosen_move

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