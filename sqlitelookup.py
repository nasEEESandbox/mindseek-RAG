"""
This script is used to look up the patient criteria table in the database.
Only for debugging purposes.
"""

import sqlite3

conn = sqlite3.connect('diagnosis.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM patient_criteria")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
