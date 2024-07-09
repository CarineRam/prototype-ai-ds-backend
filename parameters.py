import sqlite3

def connect_db():
    return sqlite3.connect('saved_parameters.db')

def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            model TEXT,
            temperature REAL,
            max_tokens INTEGER,
            stop_sequences TEXT,
            save_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_parameters(name, model, temperature, max_tokens, stop_sequences):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO saved_parameters (name, model, temperature, max_tokens, stop_sequences)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, model, temperature, max_tokens, stop_sequences))
    conn.commit()
    conn.close()

create_table()

def print_all_parameters():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM saved_parameters')
    rows = cursor.fetchall()
    conn.close()

    # for row in rows:
    #     print(f"Row: {row}")

    return rows

