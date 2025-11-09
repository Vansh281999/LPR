import os
import sqlite3
import argparse

DB_PATH = os.getenv('LPR_DB_PATH', 'society_vehicles.db')


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS owners (
            plate_number TEXT PRIMARY KEY,
            email TEXT,
            whatsapp TEXT
        )
        """
    )
    conn.commit()
    return conn


def upsert_owner(plate: str, email: str, whatsapp: str):
    conn = init_db(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO owners (plate_number, email, whatsapp) VALUES (?, ?, ?)\n"
        "ON CONFLICT(plate_number) DO UPDATE SET email=excluded.email, whatsapp=excluded.whatsapp",
        (plate, email or None, whatsapp or None),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register or update an owner for a plate number")
    parser.add_argument("plate", help="License plate number, exact as detected (e.g., DL8CAF5030)")
    parser.add_argument("--email", default="", help="Owner email or comma-separated list")
    parser.add_argument("--whatsapp", default="", help="WhatsApp number in international format, e.g., +919876543210")
    args = parser.parse_args()

    upsert_owner(args.plate.strip(), args.email.strip(), args.whatsapp.strip())
    print(f"Registered owner for {args.plate}")
