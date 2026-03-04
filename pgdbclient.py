import psycopg2

conn = psycopg2.connect(
    host="aiserver.anudina.com",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="postgres"  # update if different
)

cur = conn.cursor()

# Server version
cur.execute("SELECT version();")
print("Server version:", cur.fetchone()[0])

# List databases
cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
print("\nDatabases:")
for row in cur.fetchall():
    print(" -", row[0])

cur.close()
conn.close()
