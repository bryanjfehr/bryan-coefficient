import sqlite3

def setup_agent_database(db_path="multai_agent_eval.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Core Employee Roster
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        clearance_level TEXT NOT NULL
    )
    ''')

    # 2. Server Access Logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS server_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT,
        action TEXT,
        target_file TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    )
    ''')

    # 3. Encrypted Assets
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS encrypted_files (
        file_name TEXT PRIMARY KEY,
        passcode TEXT NOT NULL
    )
    ''')

    # Seed Bulk Data (The Noise)
    for i in range(1, 50):
        cursor.execute("INSERT OR IGNORE INTO employees (id, name, department, clearance_level) VALUES (?, ?, ?, ?)",
                       (f"EMP{1000+i}", f"Sales_{i}", "Sales", "Level 1"))
    
    for i in range(1, 20):
        cursor.execute("INSERT OR IGNORE INTO employees (id, name, department, clearance_level) VALUES (?, ?, ?, ?)",
                       (f"EMP{2000+i}", f"HR_{i}", "HR", "Level 2"))

    # Seed the Target Data (The Signal)
    cursor.execute("INSERT OR IGNORE INTO employees (id, name, department, clearance_level) VALUES (?, ?, ?, ?)",
                   ("EMP8842", "Bryan Vance", "Advanced Research", "Level 5"))
    
    cursor.execute("INSERT OR IGNORE INTO server_logs (employee_id, action, target_file) VALUES (?, ?, ?)",
                   ("EMP8842", "DOWNLOAD", "diag_node_7.enc"))
                   
    cursor.execute("INSERT OR IGNORE INTO encrypted_files (file_name, passcode) VALUES (?, ?)",
                   ("diag_node_7.enc", "QUANTUM_ECHO_99"))

    conn.commit()
    conn.close()
    print("Database seeded and ready.")

if __name__ == "__main__":
    setup_agent_database()
