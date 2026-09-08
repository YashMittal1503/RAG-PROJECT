"""Add storing status and is_tabular flag for Text-to-SQL pipeline

Revision ID: 002
Revises: 001
Create Date: 2026-09-06 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add 'storing' to the document_status enum
    # PostgreSQL requires ALTER TYPE to add a new value
    op.execute("ALTER TYPE document_status ADD VALUE IF NOT EXISTS 'storing' AFTER 'embedding'")

    # Add is_tabular boolean column to documents table
    op.add_column(
        "documents",
        sa.Column("is_tabular", sa.Boolean, nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    # Remove the is_tabular column
    op.drop_column("documents", "is_tabular")

    # Note: PostgreSQL does not support removing enum values easily.
    # The 'storing' enum value will remain but won't be used.
