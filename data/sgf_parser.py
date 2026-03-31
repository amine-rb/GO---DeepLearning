"""SGF (Smart Game Format) parser for Go games.

Parses SGF files and extracts board positions with associated moves
for use as training data for neural network models.
"""

import re
from typing import Optional


BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2

# SGF column/row letter to index mapping (a=0, b=1, ..., s=18)
SGF_COORD = {c: i for i, c in enumerate("abcdefghijklmnopqrs")}


def sgf_coord_to_index(coord: str) -> Optional[int]:
    """Convert a 2-character SGF coordinate to a board index (0-360).

    Returns None for pass moves ('tt' or empty string).
    """
    if len(coord) != 2:
        return None
    col_char, row_char = coord[0], coord[1]
    if col_char not in SGF_COORD or row_char not in SGF_COORD:
        return None
    col = SGF_COORD[col_char]
    row = SGF_COORD[row_char]
    if col >= BOARD_SIZE or row >= BOARD_SIZE:
        return None
    return row * BOARD_SIZE + col


class GoBoard:
    """Minimal Go board for replaying SGF games."""

    def __init__(self):
        self.board = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
        self.ko = None  # index of ko point, or None
        self.current_player = BLACK

    def copy(self) -> "GoBoard":
        b = GoBoard()
        b.board = self.board[:]
        b.ko = self.ko
        b.current_player = self.current_player
        return b

    def _neighbors(self, idx: int):
        row, col = divmod(idx, BOARD_SIZE)
        neighbors = []
        if row > 0:
            neighbors.append((row - 1) * BOARD_SIZE + col)
        if row < BOARD_SIZE - 1:
            neighbors.append((row + 1) * BOARD_SIZE + col)
        if col > 0:
            neighbors.append(row * BOARD_SIZE + col - 1)
        if col < BOARD_SIZE - 1:
            neighbors.append(row * BOARD_SIZE + col + 1)
        return neighbors

    def _liberties(self, idx: int):
        """Count liberties of the group containing stone at idx."""
        color = self.board[idx]
        if color == EMPTY:
            return 0
        visited = set()
        stack = [idx]
        liberties = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nb in self._neighbors(cur):
                if self.board[nb] == EMPTY:
                    liberties.add(nb)
                elif self.board[nb] == color and nb not in visited:
                    stack.append(nb)
        return len(liberties)

    def _group(self, idx: int):
        """Return all stones in the group containing stone at idx."""
        color = self.board[idx]
        if color == EMPTY:
            return set()
        visited = set()
        stack = [idx]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nb in self._neighbors(cur):
                if self.board[nb] == color and nb not in visited:
                    stack.append(nb)
        return visited

    def play(self, idx: int, color: int) -> bool:
        """Place a stone and capture. Returns True if legal."""
        if self.board[idx] != EMPTY:
            return False
        if idx == self.ko:
            return False

        self.board[idx] = color
        opponent = WHITE if color == BLACK else BLACK
        captured = []
        for nb in self._neighbors(idx):
            if self.board[nb] == opponent:
                if self._liberties(nb) == 0:
                    group = self._group(nb)
                    captured.extend(group)
                    for stone in group:
                        self.board[stone] = EMPTY

        # Check suicide
        if self._liberties(idx) == 0:
            self.board[idx] = EMPTY
            return False

        # Ko rule
        if len(captured) == 1:
            self.ko = captured[0]
        else:
            self.ko = None

        return True

    def get_features(self) -> list:
        """Return feature planes as flat lists.

        Returns a list of 4 planes (each BOARD_SIZE*BOARD_SIZE):
          0: current player's stones
          1: opponent's stones
          2: ko point
          3: ones (constant plane)
        """
        current = self.current_player
        opponent = WHITE if current == BLACK else BLACK
        n = BOARD_SIZE * BOARD_SIZE
        plane_current = [1 if self.board[i] == current else 0 for i in range(n)]
        plane_opponent = [1 if self.board[i] == opponent else 0 for i in range(n)]
        plane_ko = [0] * n
        if self.ko is not None:
            plane_ko[self.ko] = 1
        plane_ones = [1] * n
        return [plane_current, plane_opponent, plane_ko, plane_ones]


def parse_sgf(sgf_text: str):
    """Parse an SGF string and yield (features, policy_target, value_target) tuples.

    features: list of 4 flat planes, each of length BOARD_SIZE*BOARD_SIZE
    policy_target: int index (0-360) of the played move
    value_target: float +1.0 for black win, -1.0 for white win, 0.0 for unknown
    """
    # Extract result
    result_match = re.search(r"RE\[([^\]]*)\]", sgf_text)
    value_target = 0.0
    if result_match:
        result = result_match.group(1).upper()
        if result.startswith("B"):
            value_target = 1.0
        elif result.startswith("W"):
            value_target = -1.0

    # Extract moves
    moves = re.findall(r";([BW])\[([a-s]{0,2})\]", sgf_text)

    board = GoBoard()
    examples = []

    for color_char, coord in moves:
        color = BLACK if color_char == "B" else WHITE

        # Set current player
        board.current_player = color

        # Skip pass moves
        if not coord or coord == "tt":
            board.current_player = WHITE if color == BLACK else BLACK
            continue

        idx = sgf_coord_to_index(coord)
        if idx is None:
            continue

        features = board.get_features()

        # Value target: from current player's perspective
        if color == BLACK:
            vt = value_target
        else:
            vt = -value_target

        examples.append((features, idx, vt))

        board.play(idx, color)
        board.current_player = WHITE if color == BLACK else BLACK

    return examples


def load_sgf_file(filepath: str):
    """Load and parse a single SGF file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return parse_sgf(content)


def load_sgf_directory(directory: str, max_games: Optional[int] = None):
    """Load all SGF files from a directory and return all examples.

    Returns a list of (features, policy_target, value_target) tuples.
    """
    import os

    all_examples = []
    games_loaded = 0

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".sgf"):
            continue
        if max_games is not None and games_loaded >= max_games:
            break
        path = os.path.join(directory, fname)
        try:
            examples = load_sgf_file(path)
            all_examples.extend(examples)
            games_loaded += 1
        except Exception:
            continue

    return all_examples
