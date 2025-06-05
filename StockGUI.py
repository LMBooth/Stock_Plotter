import sys
import json
import threading
import traceback
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtCore import (
    QThread,
    pyqtSignal,
    QTimer,
    Qt,
)
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QInputDialog,
    QWidget,
    QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ─── Constants & File Paths ───
BASE_DIR    = Path(__file__).parent
STOCKS_JSON = BASE_DIR / "stocks.json"
DB_FILE     = BASE_DIR / "stock_data.db"
FETCH_INTERVAL_MS = 60 * 1000  # 60 seconds


def load_tracked_symbols():
    """
    Load the list of tracked symbols from stocks.json.
    If the file is missing or invalid, create it with default ["INTC","TSLA"].
    """
    default = ["INTC", "TSLA"]
    try:
        if STOCKS_JSON.exists():
            with open(STOCKS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(s, str) for s in data):
                    return sorted(list({ s.upper() for s in data }))
    except Exception:
        pass

    # Create default file if missing or invalid
    with open(STOCKS_JSON, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=2)
    return default.copy()


def save_tracked_symbols(symbols_list):
    """
    Overwrite stocks.json with the provided list of symbols (already uppercase, no duplicates).
    """
    with open(STOCKS_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(symbols_list), f, indent=2)


def init_db():
    """
    If stock_data.db does not exist, create it and initialize a table:
        stock_data (symbol TEXT, timestamp TEXT, price REAL, change REAL,
                    PRIMARY KEY(symbol, timestamp))
    Index on (symbol, timestamp) will greatly speed up queries.
    """
    init_needed = not DB_FILE.exists()
    conn = sqlite3.connect(str(DB_FILE))
    if init_needed:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE stock_data (
                symbol    TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price     REAL,
                change    REAL,
                PRIMARY KEY(symbol, timestamp)
            );
        """)
        cur.execute("CREATE INDEX idx_symbol_ts ON stock_data(symbol, timestamp);")
        conn.commit()
    conn.close()


class FetchThread(QThread):
    """
    Background thread that, once per minute, fetches the latest price & change
    for each symbol in `self.symbols`. Emits a dict:
        { "INTC": (43.58, +0.34), "TSLA": (650.12, -1.53), ... }
    It also inserts each fetched row into the SQLite table `stock_data`.
    """
    data_fetched = pyqtSignal(dict)

    def __init__(self, symbols, lock, parent=None):
        super().__init__(parent)
        self.symbols = symbols
        self.lock = lock
        self.interval_ms = FETCH_INTERVAL_MS

    def run(self):
        print("[FetchThread] Starting background fetch loop…")
        while True:
            result_dict = {}

            # 1) Snapshot current symbols under lock
            with self.lock:
                symbols_snapshot = sorted(self.symbols)

            # 2) Fetch each symbol’s latest price + change
            for symbol in symbols_snapshot:
                price = None
                change = None
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.fast_info or {}
                    price = info.get("last_price", None)
                    change = info.get("change", None)

                    if price is None or change is None:
                        # Fallback to .info if fast_info is empty
                        print(f"[FetchThread] fast_info missing for {symbol}, using .info…")
                        full_info = ticker.info or {}
                        price = full_info.get("regularMarketPrice", None)
                        raw_change = full_info.get("regularMarketChange", None)
                        change = raw_change if raw_change is not None else None

                    print(f"[FetchThread] {symbol} → price={price}, change={change}")

                except Exception:
                    print(f"[FetchThread] Exception fetching {symbol}:")
                    traceback.print_exc()

                result_dict[symbol] = (price, change)

                # 3) Insert into SQLite
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    conn = sqlite3.connect(str(DB_FILE))
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR IGNORE INTO stock_data(symbol, timestamp, price, change)
                        VALUES (?, ?, ?, ?);
                    """, (symbol, timestamp, price, change))
                    conn.commit()
                except Exception:
                    traceback.print_exc()
                finally:
                    conn.close()

            # 4) Emit unified signal after all symbols are fetched
            self.data_fetched.emit(result_dict)

            # 5) Sleep until next cycle
            self.msleep(self.interval_ms)


class StockWidget(QWidget):
    """
    Main GUI widget. Shows:
      1. One QLabel per tracked symbol (price & change).
      2. A ComboBox to select which symbol’s history to view.
      3. A ComboBox to select timeframe (Hour/Day/Week/Month).
      4. An embedded Matplotlib canvas plotting price vs time.
      5. Two buttons: “➕ Add Stock” and “➖ Remove Stock”.
      6. A “Last Updated” label at the bottom.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📈 Stock Tracker (Dynamic Plot + Logger)")
        self.resize(700, 600)

        # ─── 1) Load or initialize tracked symbols & SQLite DB ───
        self.symbols = load_tracked_symbols()  # e.g. ["INTC","TSLA"]
        init_db()

        # A lock to protect reads/writes to self.symbols between threads
        self.symbols_lock = threading.Lock()

        # ─── 2) Create the fetch thread & start it ───
        self.fetch_thread = FetchThread(self.symbols, self.symbols_lock)
        self.fetch_thread.data_fetched.connect(self.on_data_fetched)
        self.fetch_thread.start()

        # ─── 3) Build the UI ───
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 3a) Price labels area
        self.label_map = {}
        self.price_labels_layout = None
        self._populate_price_labels(main_layout)

        # 3b) Controls for “which symbol” and “which timeframe”
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 10, 0, 10)

        self.combo_symbol = QComboBox()
        self.combo_symbol.setToolTip("Select symbol to plot")
        self.combo_symbol.currentIndexChanged.connect(self.update_plot)

        self.combo_timescale = QComboBox()
        self.combo_timescale.addItems(["Hour", "Day", "Week", "Month"])
        self.combo_timescale.setCurrentText("Day")
        self.combo_timescale.setToolTip("Select timeframe for plot")
        self.combo_timescale.currentIndexChanged.connect(self.update_plot)

        controls_layout.addWidget(QLabel("Symbol:"))
        controls_layout.addWidget(self.combo_symbol)
        controls_layout.addSpacing(40)
        controls_layout.addWidget(QLabel("Timeframe:"))
        controls_layout.addWidget(self.combo_timescale)
        controls_layout.addStretch()

        main_layout.addLayout(controls_layout)

        # 3c) Embedded Matplotlib canvas
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, stretch=1)

        # 3d) Add/Remove buttons
        btn_layout = QHBoxLayout()
        self.btn_add    = QPushButton("➕ Add Stock")
        self.btn_remove = QPushButton("➖ Remove Stock")

        self.btn_add.clicked.connect(self.on_add_stock)
        self.btn_remove.clicked.connect(self.on_remove_stock)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 3e) “Last Updated” label
        self.last_updated_label = QLabel("Last Updated: --")
        italic_font = self.last_updated_label.font()
        italic_font.setItalic(True)
        self.last_updated_label.setFont(italic_font)
        self.last_updated_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.last_updated_label)

        # Final: populate combo box choices and draw initial plot
        self._refresh_symbol_combo()
        self.update_plot()


    def _populate_price_labels(self, parent_layout):
        """
        At startup (or when symbols change), create one QLabel per symbol,
        showing “SYMBOL: Fetching…” initially. These live at the top of the UI.
        """
        # Remove any existing label layout and widgets
        if self.price_labels_layout is not None:
            while self.price_labels_layout.count() > 0:
                item = self.price_labels_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            parent_layout.removeItem(self.price_labels_layout)
            self.price_labels_layout.deleteLater()
            self.price_labels_layout = None

        self.label_map.clear()

        # Create a new horizontal layout to hold them
        row_layout = QHBoxLayout()
        self.price_labels_layout = row_layout
        with self.symbols_lock:
            symbols_snapshot = sorted(self.symbols)

        for sym in symbols_snapshot:
            lbl = QLabel(f"{sym}: Fetching…")
            font = lbl.font()
            font.setPointSize(12)
            font.setBold(True)
            lbl.setFont(font)
            lbl.setMargin(6)
            row_layout.addWidget(lbl)
            self.label_map[sym] = lbl

        row_layout.addStretch()
        parent_layout.addLayout(row_layout)


    def _refresh_symbol_combo(self):
        """
        Rebuilds the combo box listing all tracked symbols. Keeps the same current index
        if possible; otherwise selects the first symbol by default.
        """
        with self.symbols_lock:
            symbols_snapshot = sorted(self.symbols)

        current_symbol = self.combo_symbol.currentText()
        self.combo_symbol.blockSignals(True)
        self.combo_symbol.clear()
        self.combo_symbol.addItems(symbols_snapshot)
        # Try restoring previous selection if still present
        if current_symbol in symbols_snapshot:
            self.combo_symbol.setCurrentText(current_symbol)
        self.combo_symbol.blockSignals(False)


    def on_data_fetched(self, data_dict):
        """
        Slot called once per minute with a dict like:
            { "INTC": (43.58, 0.34), "TSLA": (650.12, -1.53), ... }
        Updates each QLabel at the top, and then calls update_plot() so the embedded
        chart will incorporate the latest point (if it falls in the selected timeframe).
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.symbols_lock:
            symbols_snapshot = sorted(self.symbols)

        for sym in symbols_snapshot:
            price, change = data_dict.get(sym, (None, None))
            lbl = self.label_map.get(sym)
            if lbl is None:
                continue  # symbol was removed just before data arrived

            if price is None or change is None:
                lbl.setText(f"{sym}:    — fetch failed —")
            else:
                arrow = "▲" if change >= 0 else "▼"
                color = "green" if change >= 0 else "red"
                lbl.setText(
                    f'<span>{sym}: </span>'
                    f'<span style="font-size:14pt;">${price:,.2f}</span> '
                    f'<span style="color:{color}; font-size:10pt;">'
                    f'{arrow} {change:+.2f}</span>'
                )

        # Update timestamp
        self.last_updated_label.setText(f"Last Updated: {now}")

        # Redraw the plot so the newest point shows (if it’s in the selected window)
        self.update_plot()


    def on_add_stock(self):
        """
        Prompt the user for a new ticker. Validate by fetching once.
        If valid and not already tracked, add to self.symbols, save JSON,
        rebuild labels & combo box, and refresh the plot.
        """
        text, ok = QInputDialog.getText(self, "Add Stock", "Enter ticker symbol (e.g. AAPL):")
        if not ok or not text.strip():
            return

        new_sym = text.strip().upper()
        with self.symbols_lock:
            if new_sym in self.symbols:
                QMessageBox.information(self, "Already Tracked", f"{new_sym} is already in the list.")
                return

        # Quick validation via yfinance
        try:
            info = yf.Ticker(new_sym).fast_info or {}
            price = info.get("last_price", None)
            change = info.get("change", None)
            if price is None or change is None:
                # Fallback to .info if fast_info is empty
                full = yf.Ticker(new_sym).info or {}
                price = full.get("regularMarketPrice", None)
                change = full.get("regularMarketChange", None)

            if price is None or change is None:
                raise ValueError()
        except Exception:
            QMessageBox.warning(
                self,
                "Invalid Symbol",
                f"Could not fetch data for “{new_sym}”.\nMake sure it’s a valid ticker."
            )
            return

        # Valid symbol; add to tracked list
        with self.symbols_lock:
            self.symbols.append(new_sym)
            self.symbols[:] = sorted(set(self.symbols))
            save_tracked_symbols(self.symbols)

        # Rebuild price labels & combo box
        self._populate_price_labels(self.layout())
        self._refresh_symbol_combo()
        self.update_plot()


    def on_remove_stock(self):
        """
        Show a dialog with a dropdown of currently tracked symbols.
        After selecting one, remove it from the list, save JSON,
        rebuild labels & combo box, and refresh the plot.
        """
        with self.symbols_lock:
            current_symbols = sorted(self.symbols)
        if not current_symbols:
            QMessageBox.information(self, "No Symbols", "No symbols to remove.")
            return

        choice, ok = QInputDialog.getItem(
            self,
            "Remove Stock",
            "Select a symbol to remove:",
            current_symbols,
            editable=False,
        )
        if not ok:
            return

        with self.symbols_lock:
            if choice in self.symbols:
                self.symbols.remove(choice)
                save_tracked_symbols(self.symbols)

        # Rebuild price labels & combo box
        self._populate_price_labels(self.layout())
        self._refresh_symbol_combo()
        self.update_plot()


    def _ensure_historical_data(self, symbol: str, cutoff: pd.Timestamp):
        """
        Ensure that the SQLite table `stock_data` contains every timestamp for
        `symbol` from cutoff→now. If missing, download the missing portion via
        yfinance.download (using Python datetime objects to avoid parsing errors),
        then insert those rows into SQLite.
        """
        try:
            # 1) Connect to SQLite and query the earliest timestamp for this symbol
            conn = sqlite3.connect(str(DB_FILE))
            cur  = conn.cursor()
            cur.execute("""
                SELECT MIN(timestamp) FROM stock_data WHERE symbol = ?;
            """, (symbol,))
            row = cur.fetchone()
            conn.close()

            earliest_str = row[0]  # ISO string "YYYY-MM-DD HH:MM:SS" or None
        except Exception:
            # If DB is missing or corrupted, bail out; yfinance.history will fill everything later
            return

        if earliest_str is None:
            # No data at all for this symbol → download everything from cutoff→now
            start_dt = cutoff.to_pydatetime()
            end_dt   = (pd.Timestamp.now() + pd.Timedelta(minutes=1)).to_pydatetime()

            hist = yf.download(
                tickers      = symbol,
                start        = start_dt,
                end          = end_dt,
                interval     = "1m",
                progress     = False,
                auto_adjust  = False,
                actions      = False,
            )
            if hist is None or hist.empty:
                return

            hist = hist.reset_index()[["Datetime", "Close"]]
            hist.rename(columns={"Datetime": "timestamp", "Close": "price"}, inplace=True)
            hist["symbol"] = symbol
            hist["change"] = hist["price"].diff().fillna(0)

            # Insert all rows into SQLite
            conn = sqlite3.connect(str(DB_FILE))
            cur  = conn.cursor()
            for ts, pr, ch in zip(hist["timestamp"], hist["price"], hist["change"]):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    cur.execute("""
                        INSERT OR IGNORE INTO stock_data(symbol, timestamp, price, change)
                        VALUES (?, ?, ?, ?);
                    """, (symbol, ts_str, pr, ch))
                except Exception:
                    pass
            conn.commit()
            conn.close()
            return

        # We do have some data; parse earliest timestamp string into a pd.Timestamp
        earliest_local = pd.to_datetime(earliest_str)

        if earliest_local <= cutoff:
            # Already covered from cutoff→now; nothing missing
            return

        # Missing [cutoff → earliest_local). Download exactly that slice
        missing_start = cutoff.to_pydatetime()
        missing_end   = (earliest_local + pd.Timedelta(seconds=1)).to_pydatetime()

        hist = yf.download(
            tickers      = symbol,
            start        = missing_start,
            end          = missing_end,
            interval     = "1m",
            progress     = False,
            auto_adjust  = False,
            actions      = False,
        )
        if hist is None or hist.empty:
            return

        hist = hist.reset_index()[["Datetime", "Close"]]
        hist.rename(columns={"Datetime": "timestamp", "Close": "price"}, inplace=True)
        hist["symbol"] = symbol
        hist["change"] = hist["price"].diff().fillna(0)

        # Insert only the missing slice into SQLite
        conn = sqlite3.connect(str(DB_FILE))
        cur  = conn.cursor()
        for ts, pr, ch in zip(hist["timestamp"], hist["price"], hist["change"]):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO stock_data(symbol, timestamp, price, change)
                    VALUES (?, ?, ?, ?);
                """, (symbol, ts_str, pr, ch))
            except Exception:
                pass
        conn.commit()
        conn.close()


    def update_plot(self):
        """
        Reads from SQLite + any missing history → to the selected symbol/timeframe,
        and redraws the embedded Matplotlib canvas accordingly.  If no data in that 
        timeframe, fall back to plotting the last 30 points.
        """
        # 1) Clear the figure entirely so nothing “lingers”
        self.figure.clear()

        symbol = self.combo_symbol.currentText()
        timeframe = self.combo_timescale.currentText()

        if not symbol:
            # No symbols to plot
            ax = self.figure.subplots()
            ax.text(0.5, 0.5, "No symbol selected.", va="center", ha="center", fontsize=12, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        # 2) Determine cutoff time
        now = pd.Timestamp.now()
        if timeframe == "Hour":
            cutoff      = now - pd.Timedelta(hours=1)
            yf_interval = "1m"
        elif timeframe == "Day":
            cutoff      = now - pd.Timedelta(days=1)
            yf_interval = "5m"
        elif timeframe == "Week":
            cutoff      = now - pd.Timedelta(weeks=1)
            yf_interval = "30m"
        elif timeframe == "Month":
            cutoff      = now - pd.Timedelta(days=30)
            yf_interval = "1h"
        else:
            cutoff      = now - pd.Timedelta(days=1)
            yf_interval = "5m"

        # 3) Ensure the SQLite table has every data point from cutoff→now for this symbol
        self._ensure_historical_data(symbol, cutoff)

        # 4) Now query SQLite, filter to symbol/timeframe, and plot
        try:
            conn = sqlite3.connect(str(DB_FILE))
            df = pd.read_sql_query(
                """
                SELECT timestamp, price 
                FROM stock_data 
                WHERE symbol = ? 
                  AND timestamp >= ? 
                ORDER BY timestamp ASC;
                """, 
                conn, 
                params=(symbol, cutoff.strftime("%Y-%m-%d %H:%M:%S")),
                parse_dates=["timestamp"]
            )
            conn.close()
        except Exception as e:
            ax = self.figure.subplots()
            ax.text(0.5, 0.5, f"Error querying DB:\n{e}", va="center", ha="center", fontsize=12, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        if df.empty:
            # No data in that timeframe—fallback to last 30 points
            try:
                conn = sqlite3.connect(str(DB_FILE))
                df_fallback = pd.read_sql_query(
                    """
                    SELECT timestamp, price 
                    FROM (
                        SELECT timestamp, price 
                        FROM stock_data 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 30
                    ) ORDER BY timestamp ASC;
                    """, 
                    conn, 
                    params=(symbol,),
                    parse_dates=["timestamp"]
                )
                conn.close()
            except Exception:
                df_fallback = pd.DataFrame(columns=["timestamp", "price"])

            ax = self.figure.subplots()
            if df_fallback.empty:
                ax.text(0.5, 0.5, f"No data for {symbol}.", va="center", ha="center", fontsize=12, alpha=0.6)
            else:
                # Sanitize price column to ensure numeric values
                df_fallback["price"] = pd.to_numeric(df_fallback["price"], errors="coerce")
                df_fallback.dropna(subset=["price"], inplace=True)

                if df_fallback.empty:
                    ax.text(0.5, 0.5, "No data", va="center", ha="center", fontsize=12, alpha=0.6)
                else:
                    # Convert to Python datetime + float arrays
                    x_vals = df_fallback["timestamp"].dt.to_pydatetime()
                    y_vals = df_fallback["price"].to_numpy(float)
                    ax.plot(x_vals, y_vals, marker="o", linestyle="-", alpha=0.7)
                    ax.set_title(
                        f"No data in last {timeframe.lower()}. Showing last {len(df_fallback)} points."
                    )
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (USD)")
            ax.tick_params(axis="x", rotation=45)
            self.figure.tight_layout()
            self.canvas.draw()
            return

        ax = self.figure.subplots()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df.dropna(subset=["price"], inplace=True)

        if df.empty:
            ax.text(0.5, 0.5, "No data", va="center", ha="center", fontsize=12, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        # Convert to Python datetime + float arrays
        x_vals = df["timestamp"].dt.to_pydatetime()
        y_vals = df["price"].to_numpy(float)
        ax.plot(x_vals, y_vals, marker="o", linestyle="-", alpha=0.8)
        ax.set_title(f"{symbol} price over last {timeframe.lower()}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        ax.tick_params(axis="x", rotation=45)
        self.figure.tight_layout()
        self.canvas.draw()


    def closeEvent(self, event):
        """
        Cleanly stop the thread when the main window is closed.
        """
        try:
            self.fetch_thread.terminate()
            self.fetch_thread.wait(1000)
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = StockWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
