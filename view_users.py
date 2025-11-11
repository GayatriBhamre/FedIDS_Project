# view_users.py
import sqlite3, os
db_path = os.path.join(os.path.dirname(__file__), 'backend', 'users.db')
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT id, username, email, role, created_at, first_name, last_name FROM users")
for row in cur.fetchall():
    print(row)
conn.close()