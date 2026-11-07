"""
Create a dashboard admin account.

Usage: python create_admin.py admin@example.com "Admin Name"
"""
import sys
from getpass import getpass

from sqlalchemy import text

from app.auth.security import hash_password
from app.dashboard.repository import email_exists
from app.database.connection import engine
from app.database.migrations import run_migrations


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())

    email, name = sys.argv[1].strip(), sys.argv[2].strip()

    run_migrations()

    if email_exists(email):
        sys.exit(f"A user with email {email} already exists")

    password = getpass("Password (8+ characters): ")

    if len(password) < 8:
        sys.exit("Password must be at least 8 characters")

    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO users (client_id, name, email, password_hash, role)
                VALUES (NULL, :name, :email, :password_hash, 'admin')
            """),
            {"name": name, "email": email, "password_hash": hash_password(password)},
        )

    print(f"Admin {email} created")


if __name__ == "__main__":
    main()
