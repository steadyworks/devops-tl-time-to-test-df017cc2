import os
from typing import (
    AsyncGenerator,
    Callable,
    Protocol,
    TypeVar,
    cast,
    overload,
)

import pytest
import pytest_asyncio
from sqlalchemy import (
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

F = TypeVar("F", bound=Callable[..., object])


class _AsyncFixtureDecorator(Protocol):
    @overload
    def __call__(self, func: F, /) -> F: ...
    @overload
    def __call__(
        self,
        *,
        scope: str | None = ...,
        autouse: bool | None = ...,
        name: str | None = ...,
    ) -> Callable[[F], F]: ...


# Tell the type checker that pytest_asyncio.fixture matches our decorator protocol
async_fixture = cast("_AsyncFixtureDecorator", pytest_asyncio.fixture)


@async_fixture(scope="session")
async def pg_engine() -> AsyncGenerator[AsyncEngine, None]:
    db_url = os.environ.get("TEST_DATABASE_URL")
    if not db_url:
        pytest.skip(
            "Set TEST_DATABASE_URL=postgresql+asyncpg://... to run PG integration tests without Docker"
        )

    engine = create_async_engine(db_url, echo=False, future=True)

    async with engine.begin() as conn:
        # optional: make sure pgcrypto is available (safe if extension is missing)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
        except Exception:
            # not fatal for tests since SQLModel defaults use uuid4()
            pass

        # clean slate for the test database
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

        # production-expected unique indexes used by our logic
        await conn.execute(
            text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_recipient_share_by_user
            ON shares (photobook_id, recipient_user_id)
            WHERE (kind = 'recipient' AND recipient_user_id IS NOT NULL);
        """)
        )

        await conn.execute(
            text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_share_channel_dest_per_photobook
            ON share_channels (photobook_id, channel_type, destination);
        """)
        )

        await conn.execute(
            text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_outbox_idempotency
            ON notification_outbox (share_channel_id, notification_type, idempotency_key)
            WHERE idempotency_key IS NOT NULL;
        """)
        )

        await conn.execute(
            text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = 'public' AND indexname = 'shares_share_slug_key'
                ) THEN
                    CREATE UNIQUE INDEX shares_share_slug_key ON shares (share_slug);
                END IF;
            END $$;
        """)
        )

        await conn.run_sync(SQLModel.metadata.create_all)

        # Ensure the session defaults to UTC for deterministic tests
        await conn.execute(text("SET TIME ZONE 'UTC'"))

        # Make the outbox timestamps tz-aware like prod
        await conn.execute(
            text("""
            ALTER TABLE notification_outbox
            ALTER COLUMN scheduled_for TYPE timestamptz USING scheduled_for AT TIME ZONE 'UTC',
            ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC',
            ALTER COLUMN updated_at TYPE timestamptz USING updated_at AT TIME ZONE 'UTC',
            ALTER COLUMN dispatch_claimed_at TYPE timestamptz USING dispatch_claimed_at AT TIME ZONE 'UTC',
            ALTER COLUMN dispatch_lease_expires_at TYPE timestamptz USING dispatch_lease_expires_at AT TIME ZONE 'UTC',
            ALTER COLUMN canceled_at TYPE timestamptz USING canceled_at AT TIME ZONE 'UTC',
            ALTER COLUMN last_scheduled_at TYPE timestamptz USING last_scheduled_at AT TIME ZONE 'UTC';
        """)
        )

        # (Optional) if your tests ever assert share/channel timestamps too:
        await conn.execute(
            text("""
            ALTER TABLE shares
            ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC',
            ALTER COLUMN updated_at TYPE timestamptz USING updated_at AT TIME ZONE 'UTC';
        """)
        )
        await conn.execute(
            text("""
            ALTER TABLE share_channels
            ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC',
            ALTER COLUMN updated_at TYPE timestamptz USING updated_at AT TIME ZONE 'UTC';
        """)
        )

    try:
        yield engine
    finally:
        await engine.dispose()


@async_fixture(scope="function")
async def db_session(pg_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    Session = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        yield s


@async_fixture(autouse=True)
async def clean_share_tables(db_session: AsyncSession) -> None:
    # Order matters due to FKs; TRUNCATE … CASCADE is simplest/fastest.
    await db_session.execute(
        text("""
        TRUNCATE
            notification_delivery_attempts,
            notification_outbox,
            share_channels,
            shares
        RESTART IDENTITY CASCADE
        """)
    )
    await db_session.commit()
