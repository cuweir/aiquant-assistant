import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# --- MODIFICATION 1: Load environment variables ---
from dotenv import load_dotenv

# Add the project root to the path and load the .env file
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()
# --- END OF MODIFICATION 1 ---

# Import your Base model from your application
from app.db.base_class import Base
# You also need to import the models themselves so SQLAlchemy registers them
from app.db import models

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# --- MODIFICATION 2: Set sqlalchemy.url from environment variable ---
# This line will read the DATABASE_URL from your .env file
# and set it as the configuration for alembic.
# This completely overrides the `sqlalchemy.url` in alembic.ini
db_url_from_env = os.getenv("DATABASE_URL")
if db_url_from_env:
    config.set_main_option("sqlalchemy.url", db_url_from_env)
else:
    # Optional: Raise an error if DATABASE_URL is not set, to avoid confusion
    raise ValueError("DATABASE_URL environment variable not set. Please create a .env file.")
# --- END OF MODIFICATION 2 ---

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
