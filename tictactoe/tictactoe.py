import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit.components.v1 as components

st.set_page_config(page_title="Tic-Tac-Toe AI Predictor", layout="centered")

X_COLOR = "#FF0000"   # fixed color for X (red)
O_COLOR = "#2196F3"   # fixed color for O (blue)

# ── Load model & columns ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("ttt_model.pkl")
    columns = joblib.load("ttt_columns.pkl")
    return model, columns

model, COLUMNS = load_model()

MOVE_COLS = [
    "Move 1-X (Row-Col)", "Move 2-O (Row-Col)", "Move 3-X (Row-Col)",
    "Move 4-O (Row-Col)", "Move 5-X (Row-Col)", "Move 6-O (Row-Col)",
    "Move 7-X (Row-Col)", "Move 8-O (Row-Col)", "Move 9-X (Row-Col)",
]
LABEL_MAP = {0: "Draw", 1: "O Wins", 2: "X Wins"}

# ── Session state ─────────────────────────────────────────────────────────────
def reset_board():
    st.session_state.board   = [[None]*3 for _ in range(3)]
    st.session_state.turn    = 'X'          # X always moves first
    st.session_state.moves   = []
    st.session_state.winner  = None
    st.session_state.proba   = None
    st.session_state.game_id = st.session_state.get("game_id", 0) + 1  # forces fresh button widgets

def start_game(human_symbol):
    st.session_state.human_symbol = human_symbol
    st.session_state.ai_symbol    = 'O' if human_symbol == 'X' else 'X'
    st.session_state.phase        = "playing"
    reset_board()

if "phase" not in st.session_state:
    st.session_state.phase = "setup"

# ── Game logic helpers ────────────────────────────────────────────────────────
def check_winner(board):
    lines = (
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],
    )
    for line in lines:
        vals = [board[r][c] for r,c in line]
        if vals[0] and vals[0] == vals[1] == vals[2]:
            return vals[0]
    if all(board[r][c] for r in range(3) for c in range(3)):
        return "Draw"
    return None

def minimax(board, is_maximizing, ai_symbol, human_symbol):
    w = check_winner(board)
    if w == ai_symbol:
        return 10
    if w == human_symbol:
        return -10
    if w == 'Draw':
        return 0
    empty = [(r,c) for r in range(3) for c in range(3) if not board[r][c]]
    if not empty:
        return 0
    if is_maximizing:
        best = -999
        for r,c in empty:
            board[r][c] = ai_symbol
            best = max(best, minimax(board, False, ai_symbol, human_symbol))
            board[r][c] = None
        return best
    else:
        best = 999
        for r,c in empty:
            board[r][c] = human_symbol
            best = min(best, minimax(board, True, ai_symbol, human_symbol))
            board[r][c] = None
        return best

def ai_move(board, ai_symbol, human_symbol):
    best_score, best_move = -999, None
    for r,c in [(r,c) for r in range(3) for c in range(3) if not board[r][c]]:
        board[r][c] = ai_symbol
        score = minimax(board, False, ai_symbol, human_symbol)
        board[r][c] = None
        if score > best_score:
            best_score, best_move = score, (r,c)
    return best_move

def build_input(moves):
    data = {}
    for i, col in enumerate(MOVE_COLS):
        data[col] = [moves[i] if i < len(moves) else "---"]
    df = pd.get_dummies(pd.DataFrame(data))
    df = df.reindex(columns=COLUMNS, fill_value=0)
    return df

def predict(moves):
    if len(moves) < 5:          # need at least 5 moves for a result
        return None
    inp   = build_input(moves)
    proba = model.predict_proba(inp)[0]
    return proba                 # [draw, O, X]

def make_move(r, c):
    player = st.session_state.turn
    st.session_state.board[r][c] = player
    st.session_state.moves.append(f"{r}-{c}")
    w = check_winner(st.session_state.board)
    if w:
        st.session_state.winner = w
    else:
        st.session_state.turn = 'O' if player == 'X' else 'X'
    st.session_state.proba = predict(st.session_state.moves)

def colorize_buttons():
    """Color the X / O text (and border) on the board buttons using the
    fixed colors. st.button() can't take colored/HTML labels, so we tweak
    the rendered buttons in the parent document with a tiny injected script.
    The nonce comment forces the script content to differ each rerun,
    otherwise Streamlit caches the iframe and never re-runs it."""
    nonce = len(st.session_state.get("moves", []))
    components.html(f"""
        <!-- nonce:{nonce} -->
        <script>
        const btns = window.parent.document.querySelectorAll('div.stButton > button');
        btns.forEach(btn => {{
            const text = btn.innerText.trim();
            if (text === 'X') {{
                btn.style.color = '{X_COLOR}';
                btn.style.borderColor = '{X_COLOR}';
            }} else if (text === 'O') {{
                btn.style.color = '{O_COLOR}';
                btn.style.borderColor = '{O_COLOR}';
            }} else {{
                btn.style.color = '';
                btn.style.borderColor = '';
            }}
        }});
        </script>
    """, height=0, width=0)

# ── Shared styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
div.stButton > button {
    width: 90px; height: 90px;
    font-size: 2rem; font-weight: bold;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("Tic-Tac-Toe — AI Outcome Predictor")

# ── Setup screen ───────────────────────────────────────────────────────────────
if st.session_state.phase == "setup":
    st.markdown("Choose your symbol before the game starts. **X always moves first.**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h3 style='color:{X_COLOR};'>Play as X</h3>", unsafe_allow_html=True)
        st.write("You go first.")
        if st.button("Start as X", use_container_width=True):
            start_game("X")
            st.rerun()
    with col2:
        st.markdown(f"<h3 style='color:{O_COLOR};'>Play as O</h3>", unsafe_allow_html=True)
        st.write("AI goes first.")
        if st.button("Start as O", use_container_width=True):
            start_game("O")
            st.rerun()
    st.stop()

# ── Playing screen ──────────────────────────────────────────────────────────────
human_symbol = st.session_state.human_symbol
ai_symbol    = st.session_state.ai_symbol

st.markdown("Play the game. After every move the XGBoost model predicts who will win.")
st.caption(f"You are **{human_symbol}**, AI is **{ai_symbol}**.")

board  = st.session_state.board
winner = st.session_state.winner

st.subheader("Board")

turn = st.session_state.turn
if not winner and turn == ai_symbol:
    r, c = ai_move(board, ai_symbol, human_symbol)
    make_move(r, c)
    st.rerun()

for r in range(3):
    cols = st.columns(3)
    for c in range(3):
        cell = board[r][c]
        label = cell if cell else " "
        disabled = bool(cell) or bool(winner)
        if cols[c].button(label, key=f"cell_{st.session_state.game_id}_{r}_{c}", disabled=disabled):
            make_move(r, c)
            st.rerun()

colorize_buttons()

# Status
move_count = len(st.session_state.moves)
if winner:
    if winner == "Draw":
        st.success("It's a Draw!")
    else:
        st.success(f"{winner} wins!")
else:
    st.info(f"Turn: **{st.session_state.turn}** &nbsp;|&nbsp; Move #{move_count + 1}")

# Prediction panel
st.divider()
st.subheader("AI Prediction")

proba = st.session_state.proba
if proba is None:
    st.write("Play at least 5 moves to see a prediction.")
else:
    draw_p, o_p, x_p = proba
    col1, col2, col3 = st.columns(3)
    col1.metric("X Wins",  f"{x_p*100:.1f}%")
    col2.metric("O Wins",  f"{o_p*100:.1f}%")
    col3.metric("Draw",    f"{draw_p*100:.1f}%")

    predicted = LABEL_MAP[int(np.argmax(proba))]
    st.markdown(f"**Model predicts → {predicted}**")

    # Bar chart
    chart_df = pd.DataFrame({
        "Outcome": ["X Wins", "O Wins", "Draw"],
        "Probability": [x_p, o_p, draw_p]
    }).set_index("Outcome")
    st.bar_chart(chart_df)

# Move log
if st.session_state.moves:
    st.divider()
    st.subheader("Move Log")
    for i, mv in enumerate(st.session_state.moves):
        player = 'X' if i % 2 == 0 else 'O'
        st.write(f"Move {i+1} ({player}): Row-Col {mv}")

# Reset / change player
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("Reset Game", use_container_width=True):
        reset_board()
        st.rerun()
with col2:
    if st.button("Change Player / New Setup", use_container_width=True):
        st.session_state.phase = "setup"
        st.rerun()