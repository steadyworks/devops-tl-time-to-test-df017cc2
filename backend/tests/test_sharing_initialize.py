# tests/test_sharing_initialize.py

from datetime import timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import (
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from backend.db.dal import safe_transaction
from backend.db.data_models import (
    DAONotificationDeliveryAttempts,
    DAONotificationOutbox,
    DAOPhotobooks,
    DAOShareChannels,
    DAOShares,
    DAOUsers,
    NotificationDeliveryEvent,
    ShareChannelStatus,
    ShareChannelType,
    ShareProvider,
)
from backend.lib.notifs.dispatch_service import (
    claim_and_enqueue_one_outbox,
    claim_and_enqueue_ready_batch,
    reclaim_and_enqueue_expired_leases,
)
from backend.lib.sharing.schemas import (
    ShareChannelSpec,
    ShareCreateRequest,
    ShareRecipientSpec,
)
from backend.lib.sharing.service import initialize_shares_and_channels
from backend.lib.utils.common import utcnow
from backend.worker.job_processor.remote_deliver_notification import (
    RemoteDeliverNotificationJobProcessor,
)
from backend.worker.job_processor.types import (
    DeliverNotificationInputPayload,
    JobInputPayload,
    JobType,
)

from .conftest import async_fixture

if TYPE_CHECKING:
    from backend.db.session.factory import AsyncSessionFactory
    from backend.lib.asset_manager.base import AssetManager
    from backend.worker.process.types import RemoteIOBoundWorkerProcessResources


# -------------------------
# Seed helpers
# -------------------------


@async_fixture
async def owner_user(db_session: AsyncSession) -> DAOUsers:
    user = DAOUsers(id=uuid4(), email="owner@example.com", name="Owner")
    db_session.add(user)
    await db_session.commit()
    return user


@async_fixture
async def photobook(db_session: AsyncSession, owner_user: DAOUsers) -> DAOPhotobooks:
    pb = DAOPhotobooks(
        id=uuid4(),
        title="Test Photobook",
        user_id=owner_user.id,
        status=None,
        status_last_edited_by=None,
    )
    db_session.add(pb)
    await db_session.commit()
    return pb


# -------------------------
# Test utilities
# -------------------------


def _email_recipient(
    email: str, *, idempotency_key: str | None = None
) -> ShareRecipientSpec:
    return ShareRecipientSpec(
        channels=[
            ShareChannelSpec(
                channel_type=ShareChannelType.EMAIL,
                destination=email,
                idempotency_key=idempotency_key,
            )
        ],
        recipient_display_name="Friend",
    )


async def _count(session: AsyncSession, model: Any) -> int:
    res = await session.execute(select(model))
    return len(res.scalars().all())


# -------------------------
# A1. Create send-now
# -------------------------


@pytest.mark.asyncio
async def test_A1_create_send_now(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    req = ShareCreateRequest(
        recipients=[_email_recipient("friend@example.com")],
        sender_display_name="Owner",
        scheduled_for=None,
    )

    resp = await initialize_shares_and_channels(
        session=db_session,
        user_id=owner_user.id,
        photobook_id=photobook.id,
        req=req,
    )

    # shares: 1
    assert await _count(db_session, DAOShares) == 1
    # share_channels: 1
    assert await _count(db_session, DAOShareChannels) == 1
    # outbox: 1, status=PENDING
    assert await _count(db_session, DAONotificationOutbox) == 1

    outbox_id = resp.recipients[0].outbox_results[0].outbox_id
    row = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert row.status == ShareChannelStatus.PENDING
    assert row.scheduled_for is None
    assert row.created_by_user_id == owner_user.id


# -------------------------
# A2. Schedule future -> SCHEDULED
# -------------------------


@pytest.mark.asyncio
async def test_A2_schedule_future(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    scheduled_for = utcnow() + timedelta(hours=1)
    req = ShareCreateRequest(
        recipients=[_email_recipient("friend@example.com")],
        sender_display_name="Owner",
        scheduled_for=scheduled_for,
    )

    resp = await initialize_shares_and_channels(
        session=db_session, user_id=owner_user.id, photobook_id=photobook.id, req=req
    )

    outbox_id = resp.recipients[0].outbox_results[0].outbox_id
    row = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()

    assert row.status == ShareChannelStatus.SCHEDULED
    assert row.scheduled_for == scheduled_for
    assert row.scheduled_by_user_id == owner_user.id
    assert row.last_scheduled_at is not None


# -------------------------
# A3. Idempotency key -> upsert (one row)
# -------------------------


@pytest.mark.asyncio
async def test_A3_idempotency_key_upsert(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    email = "friend@example.com"
    key = "idem-123"

    req1 = ShareCreateRequest(
        recipients=[_email_recipient(email, idempotency_key=key)],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    resp1 = await initialize_shares_and_channels(
        session=db_session, user_id=owner_user.id, photobook_id=photobook.id, req=req1
    )
    ob1 = resp1.recipients[0].outbox_results[0].outbox_id

    # second call same channel + same idempotency key → should NOT create another row
    req2 = ShareCreateRequest(
        recipients=[_email_recipient(email, idempotency_key=key)],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    resp2 = await initialize_shares_and_channels(
        session=db_session, user_id=owner_user.id, photobook_id=photobook.id, req=req2
    )
    ob2 = resp2.recipients[0].outbox_results[0].outbox_id

    assert ob1 == ob2  # same row
    # Ensure only one outbox row exists
    assert await _count(db_session, DAONotificationOutbox) == 1


# -------------------------
# A4. No key: live outbox dedupe (pending/scheduled/sending)
# -------------------------


@pytest.mark.asyncio
async def test_A4_live_outbox_dedupe_without_key(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    email = "friend@example.com"

    req1 = ShareCreateRequest(
        recipients=[_email_recipient(email)],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    resp1 = await initialize_shares_and_channels(
        session=db_session, user_id=owner_user.id, photobook_id=photobook.id, req=req1
    )
    ob1 = resp1.recipients[0].outbox_results[0].outbox_id

    # Call again WITHOUT idempotency key — should reuse the 'live' one (status=PENDING)
    req2 = ShareCreateRequest(
        recipients=[_email_recipient(email)],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    resp2 = await initialize_shares_and_channels(
        session=db_session, user_id=owner_user.id, photobook_id=photobook.id, req=req2
    )
    ob2 = resp2.recipients[0].outbox_results[0].outbox_id

    assert ob2 == ob1
    assert await _count(db_session, DAONotificationOutbox) == 1


# -------------------------
# A5. After terminal outbox -> new outbox next time
# -------------------------


@pytest.mark.asyncio
async def test_A5_after_terminal_status_inserts_new_outbox(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    email = "friend@example.com"

    # First time, create PENDING
    req1 = ShareCreateRequest(
        recipients=[_email_recipient(email)],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    resp1 = await initialize_shares_and_channels(
        session=db_session, user_id=owner_user.id, photobook_id=photobook.id, req=req1
    )
    ob1 = resp1.recipients[0].outbox_results[0].outbox_id

    # Mark it SENT (terminal)
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == ob1)
        .values(status=ShareChannelStatus.SENT)
    )
    await db_session.commit()

    # Second call, no idempotency key — should create a NEW row now
    req2 = ShareCreateRequest(
        recipients=[_email_recipient(email)],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    resp2 = await initialize_shares_and_channels(
        session=db_session, user_id=owner_user.id, photobook_id=photobook.id, req=req2
    )
    ob2 = resp2.recipients[0].outbox_results[0].outbox_id

    assert ob2 != ob1
    assert await _count(db_session, DAONotificationOutbox) == 2


# -------------------------
# B. Claim & Enqueue (single)
# -------------------------


class FakeJobManager:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.enqueued: list[tuple[JobType, JobInputPayload]] = []

    async def enqueue(
        self,
        job_type: JobType,
        job_payload: JobInputPayload,
        max_retries: int,
        db_session: AsyncSession,
    ) -> UUID:
        if self.should_fail:
            raise RuntimeError("enqueue fail (test)")
        self.enqueued.append((job_type, job_payload))
        return uuid4()

    async def poll(self, timeout: int) -> Optional[UUID]:
        return None

    async def claim(
        self, job_id: UUID, db_session: AsyncSession
    ) -> tuple[JobType, JobInputPayload]:
        raise NotImplementedError


class FakeJobManagerBatch:
    """Deterministic batch-capable fake with optional failure on specific enqueue indexes."""

    def __init__(self, *, fail_on_index: set[int] | None = None) -> None:
        self.enqueued: list[tuple[JobType, JobInputPayload]] = []
        self._counter = 0
        self._fail_on_index = fail_on_index or set()

    async def enqueue(
        self,
        job_type: JobType,
        job_payload: JobInputPayload,
        max_retries: int,
        db_session: AsyncSession,
    ) -> UUID:
        self._counter += 1
        if self._counter in self._fail_on_index:
            raise RuntimeError(f"enqueue fail at index {self._counter}")
        self.enqueued.append((job_type, job_payload))
        return uuid4()

    async def poll(self, timeout: int) -> Optional[UUID]:
        return None

    async def claim(
        self, job_id: UUID, db_session: AsyncSession
    ) -> tuple[JobType, JobInputPayload]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_B1_claim_and_enqueue_success(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # 1) Seed: create one PENDING, due-now outbox via initialize
    req = ShareCreateRequest(
        recipients=[_email_recipient("friend@example.com")],
        sender_display_name="Owner",
        scheduled_for=None,  # due now
    )
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=req,
        )
        outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # 2) Claim + enqueue
    jm = FakeJobManager()
    job_id = await claim_and_enqueue_one_outbox(
        session=db_session,
        job_manager=jm,
        outbox_id=outbox_id,
        user_id=owner_user.id,
        worker_id="test-worker",
        lease_seconds=300,
    )
    assert job_id is not None

    # 3) DB assertions: SENDING + token + worker + claimed_at
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.SENDING
    assert outbox.dispatch_token is not None
    assert outbox.dispatch_claimed_at is not None
    assert outbox.dispatch_worker_id == "test-worker"

    # 4) Enqueue payload correctness (expected_dispatch_token)
    assert len(jm.enqueued) == 1
    (jt, payload) = jm.enqueued[0]
    assert jt == JobType.REMOTE_DELIVER_NOTIFICATION
    assert isinstance(payload, DeliverNotificationInputPayload)
    assert payload.notification_outbox_id == outbox_id
    assert payload.expected_dispatch_token == outbox.dispatch_token
    assert payload.originating_photobook_id == outbox.photobook_id
    assert payload.user_id == owner_user.id


@pytest.mark.asyncio
async def test_B2_not_due_returns_none(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed a future-scheduled outbox
    req = ShareCreateRequest(
        recipients=[_email_recipient("future@example.com")],
        sender_display_name="Owner",
        scheduled_for=utcnow() + timedelta(hours=1),
    )
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=req,
        )
        outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    jm = FakeJobManager()
    job_id = await claim_and_enqueue_one_outbox(
        session=db_session,
        job_manager=jm,
        outbox_id=outbox_id,
        user_id=owner_user.id,
        worker_id="test-worker",
    )
    assert job_id is None

    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.SCHEDULED
    assert outbox.dispatch_claimed_at is None
    assert outbox.dispatch_token is None
    assert len(jm.enqueued) == 0


@pytest.mark.asyncio
async def test_B3_already_claimed_returns_none(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed a due-now outbox
    req = ShareCreateRequest(
        recipients=[_email_recipient("alreadyclaimed@example.com")],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=req,
        )
        outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # Manually mark as already claimed
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == outbox_id)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_claimed_at=utcnow(),
            dispatch_token=uuid4(),
            dispatch_worker_id="someone-else",
        )
    )
    await db_session.commit()

    jm = FakeJobManager()
    job_id = await claim_and_enqueue_one_outbox(
        session=db_session,
        job_manager=jm,
        outbox_id=outbox_id,
        user_id=owner_user.id,
        worker_id="test-worker",
    )
    assert job_id is None
    assert len(jm.enqueued) == 0


@pytest.mark.asyncio
async def test_B4_revoked_share_cannot_be_claimed(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed due-now outbox
    req = ShareCreateRequest(
        recipients=[_email_recipient("revoked@example.com")],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=req,
        )
        outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # Find and revoke its share
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    await db_session.execute(
        update(DAOShares)
        .where(getattr(DAOShares, "id") == outbox.share_id)
        .values(access_policy="revoked")  # enum cast via string ok in SQLAlchemy
    )
    await db_session.commit()

    jm = FakeJobManager()
    job_id = await claim_and_enqueue_one_outbox(
        session=db_session,
        job_manager=jm,
        outbox_id=outbox_id,
        user_id=owner_user.id,
        worker_id="test-worker",
    )
    assert job_id is None

    # Still unclaimed, still pending (or scheduled), not sending
    outbox_after = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox_after.dispatch_claimed_at is None
    assert outbox_after.status in (
        ShareChannelStatus.PENDING,
        ShareChannelStatus.SCHEDULED,
    )
    assert len(jm.enqueued) == 0


@pytest.mark.asyncio
async def test_B5_enqueue_failure_propagates_and_claim_is_kept(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # NOTE: This test reflects CURRENT behavior of claim function (no release-on-failure).
    # If you adopt the "release on enqueue failure" improvement later, flip the assertions below.

    # Seed due-now outbox
    req = ShareCreateRequest(
        recipients=[_email_recipient("fail-enqueue@example.com")],
        sender_display_name="Owner",
        scheduled_for=None,
    )
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=req,
        )
        outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    jm = FakeJobManager(should_fail=True)

    with pytest.raises(RuntimeError, match="enqueue fail"):
        await claim_and_enqueue_one_outbox(
            session=db_session,
            job_manager=jm,
            outbox_id=outbox_id,
            user_id=owner_user.id,
            worker_id="test-worker",
        )

    # Current semantics: claim transaction committed; row remains SENDING with token
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.SENDING
    assert outbox.dispatch_token is not None
    assert outbox.dispatch_claimed_at is not None


# -------------------------
# C. Batch claim & enqueue
# -------------------------


@pytest.mark.asyncio
async def test_C1_claim_ready_batch_respects_limit_and_due_only(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed: 3 due-now + 1 future-scheduled
    emails = ["a@example.com", "b@example.com", "c@example.com", "later@example.com"]
    reqs = [
        ShareCreateRequest(
            recipients=[_email_recipient(emails[i])],
            sender_display_name="Owner",
            scheduled_for=None if i < 3 else utcnow() + timedelta(hours=1),
        )
        for i in range(4)
    ]
    async with safe_transaction(db_session):
        for r in reqs:
            await initialize_shares_and_channels(
                session=db_session,
                user_id=owner_user.id,
                photobook_id=photobook.id,
                req=r,
            )

    jm = FakeJobManagerBatch()
    job_ids = await claim_and_enqueue_ready_batch(
        session=db_session,
        job_manager=jm,
        user_id=owner_user.id,
        worker_id="batch-worker",
        limit=2,  # only pick two of the three due rows
    )
    assert len(job_ids) == 2
    assert len(jm.enqueued) == 2

    # Verify exactly 2 rows are SENDING (claimed), 1 still PENDING, and the future one SCHEDULED
    rows = (await db_session.execute(select(DAONotificationOutbox))).scalars().all()

    sending = [r for r in rows if r.status == ShareChannelStatus.SENDING]
    pending = [r for r in rows if r.status == ShareChannelStatus.PENDING]
    scheduled = [r for r in rows if r.status == ShareChannelStatus.SCHEDULED]

    assert len(sending) == 2
    assert len(pending) == 1
    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_C2_batch_skips_already_claimed_rows(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed: 2 due-now
    async with safe_transaction(db_session):
        for email in ["x@example.com", "y@example.com"]:
            await initialize_shares_and_channels(
                session=db_session,
                user_id=owner_user.id,
                photobook_id=photobook.id,
                req=ShareCreateRequest(
                    recipients=[_email_recipient(email)],
                    sender_display_name="Owner",
                    scheduled_for=None,
                ),
            )

    # Manually pre-claim one outbox
    first = (
        (
            await db_session.execute(
                select(DAONotificationOutbox).order_by(
                    getattr(DAONotificationOutbox, "created_at").asc()
                )
            )
        )
        .scalars()
        .first()
    )
    assert first is not None
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == first.id)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_claimed_at=utcnow(),
            dispatch_token=uuid4(),
            dispatch_worker_id="other-worker",
        )
    )
    await db_session.commit()

    jm = FakeJobManagerBatch()
    job_ids = await claim_and_enqueue_ready_batch(
        session=db_session,
        job_manager=jm,
        user_id=owner_user.id,
        worker_id="batch-worker",
        limit=10,
    )
    # Only the unclaimed one should be taken
    assert len(job_ids) == 1
    assert len(jm.enqueued) == 1


@pytest.mark.asyncio
async def test_C3_partial_enqueue_fail_releases_only_failed_claims(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed: 3 due-now
    async with safe_transaction(db_session):
        for email in ["p@example.com", "q@example.com", "r@example.com"]:
            await initialize_shares_and_channels(
                session=db_session,
                user_id=owner_user.id,
                photobook_id=photobook.id,
                req=ShareCreateRequest(
                    recipients=[_email_recipient(email)],
                    sender_display_name="Owner",
                    scheduled_for=None,
                ),
            )

    # Fail on the 2nd enqueue call
    jm = FakeJobManagerBatch(fail_on_index={2})
    job_ids = await claim_and_enqueue_ready_batch(
        session=db_session,
        job_manager=jm,
        user_id=owner_user.id,
        worker_id="batch-worker",
        limit=3,
    )
    # Two should succeed (1st and 3rd), middle fails and should be released
    assert len(job_ids) == 2
    # Re-fetch rows
    rows = (await db_session.execute(select(DAONotificationOutbox))).scalars().all()
    sending = [r for r in rows if r.status == ShareChannelStatus.SENDING]
    pending = [r for r in rows if r.status == ShareChannelStatus.PENDING]

    assert len(sending) == 2
    assert (
        len(pending) == 1
    )  # the failed one is released back to PENDING (token cleared)


@pytest.mark.asyncio
async def test_C4_revoked_are_never_claimed(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed a due-now row
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[_email_recipient("revoke_me@example.com")],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # Revoke the share
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    await db_session.execute(
        update(DAOShares)
        .where(getattr(DAOShares, "id") == outbox.share_id)
        .values(access_policy="revoked")
    )
    await db_session.commit()

    jm = FakeJobManagerBatch()
    job_ids = await claim_and_enqueue_ready_batch(
        session=db_session,
        job_manager=jm,
        user_id=owner_user.id,
        worker_id="batch-worker",
        limit=10,
    )
    assert job_ids == []


# -------------------------
# D. Lease reclaimer
# -------------------------


@pytest.mark.asyncio
async def test_D1_reclaim_expired_leases_enqueues_and_refreshes_token_and_times(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed due-now
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[_email_recipient("expired1@example.com")],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # Pretend it was claimed long ago and expired
    old_token = uuid4()
    old_claim_time = utcnow() - timedelta(hours=2)
    old_expiry = utcnow() - timedelta(hours=1)
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == outbox_id)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_token=old_token,
            dispatch_worker_id="stale-worker",
            dispatch_claimed_at=old_claim_time,
            dispatch_lease_expires_at=old_expiry,
        )
    )
    await db_session.commit()

    jm = FakeJobManagerBatch()
    job_ids = await reclaim_and_enqueue_expired_leases(
        session=db_session,
        job_manager=jm,
        user_id=owner_user.id,
        worker_id="reclaimer",
        limit=10,
        lease_seconds=600,
    )
    assert len(job_ids) == 1
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.SENDING
    assert outbox.dispatch_worker_id == "reclaimer"
    assert outbox.dispatch_claimed_at is not None
    assert outbox.dispatch_lease_expires_at is not None
    assert outbox.dispatch_token is not None and outbox.dispatch_token != old_token


@pytest.mark.asyncio
async def test_D2_reclaimer_respects_limit_and_skips_revoked(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed 3 rows; all expired
    rows: list[UUID] = []
    async with safe_transaction(db_session):
        for email in ["exp_a@example.com", "exp_b@example.com", "exp_c@example.com"]:
            resp = await initialize_shares_and_channels(
                session=db_session,
                user_id=owner_user.id,
                photobook_id=photobook.id,
                req=ShareCreateRequest(
                    recipients=[_email_recipient(email)],
                    sender_display_name="Owner",
                    scheduled_for=None,
                ),
            )
            rows.append(resp.recipients[0].outbox_results[0].outbox_id)

    # Mark expired + SENDING
    for oid in rows:
        await db_session.execute(
            update(DAONotificationOutbox)
            .where(getattr(DAONotificationOutbox, "id") == oid)
            .values(
                status=ShareChannelStatus.SENDING,
                dispatch_token=uuid4(),
                dispatch_worker_id="old",
                dispatch_claimed_at=utcnow() - timedelta(hours=2),
                dispatch_lease_expires_at=utcnow() - timedelta(hours=1),
            )
        )
    await db_session.commit()

    # Revoke the share of the middle one
    mid = rows[1]
    mid_row = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == mid
            )
        )
    ).scalar_one()
    await db_session.execute(
        update(DAOShares)
        .where(getattr(DAOShares, "id") == mid_row.share_id)
        .values(access_policy="revoked")
    )
    await db_session.commit()

    jm = FakeJobManagerBatch()
    job_ids = await reclaim_and_enqueue_expired_leases(
        session=db_session,
        job_manager=jm,
        user_id=owner_user.id,
        worker_id="reclaimer",
        limit=2,  # should only reclaim 2 (but one is revoked -> result should be 2 because 3rd is available)
        lease_seconds=600,
    )
    assert len(job_ids) == 2  # 2 reclaimed, revoked one skipped


@pytest.mark.asyncio
async def test_D3_reclaimer_enqueue_failure_releases_claim(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed 1 expired
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[_email_recipient("exp_fail@example.com")],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    oid = resp.recipients[0].outbox_results[0].outbox_id
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == oid)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_token=uuid4(),
            dispatch_worker_id="old",
            dispatch_claimed_at=utcnow() - timedelta(hours=2),
            dispatch_lease_expires_at=utcnow() - timedelta(hours=1),
        )
    )
    await db_session.commit()

    jm = FakeJobManagerBatch(fail_on_index={1})
    job_ids = await reclaim_and_enqueue_expired_leases(
        session=db_session,
        job_manager=jm,
        user_id=owner_user.id,
        worker_id="reclaimer",
        limit=10,
        lease_seconds=600,
    )
    assert job_ids == []  # single one failed

    # Claim should be released back to PENDING
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == oid
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.PENDING
    assert outbox.dispatch_token is None
    assert outbox.dispatch_claimed_at is None


# --- ctor dependency stubs for RemoteDeliverNotificationJobProcessor ---


class _DummyAssetManager:
    pass


class _DummySessionFactory:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def new_session(self) -> Any:
        s = self._session

        class _Ctx:
            async def __aenter__(self) -> AsyncSession:
                return s

            async def __aexit__(
                self,
                exc_type: Optional[type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[TracebackType],
            ) -> None:
                return None

        return _Ctx()


class _FakeEmailProviderClient:
    def get_share_provider(self) -> ShareProvider:
        return ShareProvider.RESEND  # import ShareProvider from your data_models


class _DummyRemoteResources:
    def __init__(self, email_client: _FakeEmailProviderClient) -> None:
        self.email_provider_client = email_client


# -------------------------
# E. Processor behavior (ctor-friendly)
# -------------------------


@pytest.mark.asyncio
async def test_E1_email_success_marks_sent_and_logs_attempts(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed 1 email outbox (due-now)
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[_email_recipient("proc_success@example.com")],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # Claim it (simulate dispatcher)
    token = uuid4()
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == outbox_id)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_token=token,
            dispatch_claimed_at=utcnow(),
            dispatch_worker_id="proc",
        )
    )
    await db_session.commit()

    proc = RemoteDeliverNotificationJobProcessor(
        job_id=uuid4(),
        asset_manager=cast("AssetManager", _DummyAssetManager()),
        db_session_factory=cast(
            "AsyncSessionFactory", _DummySessionFactory(db_session)
        ),
        worker_process_resources=cast(
            "RemoteIOBoundWorkerProcessResources",
            _DummyRemoteResources(_FakeEmailProviderClient()),
        ),
    )

    payload = DeliverNotificationInputPayload(
        user_id=owner_user.id,
        originating_photobook_id=photobook.id,
        notification_outbox_id=outbox_id,
        expected_dispatch_token=token,
    )

    await proc.process(payload)

    # Outbox should be SENT with provider message id and cleared token
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.SENT
    assert outbox.last_provider_message_id == "123"  # from EmailSendResult in processor
    assert outbox.provider == ShareProvider.RESEND
    assert outbox.dispatch_token is None

    # Attempts: PROCESSING + SENT
    attempts = (
        (
            await db_session.execute(
                select(DAONotificationDeliveryAttempts).where(
                    getattr(DAONotificationDeliveryAttempts, "notification_outbox_id")
                    == outbox_id
                )
            )
        )
        .scalars()
        .all()
    )
    events = [a.event for a in attempts]
    assert NotificationDeliveryEvent.PROCESSING in events
    assert NotificationDeliveryEvent.SENT in events
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_E2_wrong_token_noop(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[_email_recipient("proc_wrongtoken@example.com")],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # Set token T1, but payload will carry T2
    token_db = uuid4()
    token_payload = uuid4()
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == outbox_id)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_token=token_db,
            dispatch_claimed_at=utcnow(),
            dispatch_worker_id="proc",
        )
    )
    await db_session.commit()

    proc = RemoteDeliverNotificationJobProcessor(
        job_id=uuid4(),
        asset_manager=cast("AssetManager", _DummyAssetManager()),
        db_session_factory=cast(
            "AsyncSessionFactory", _DummySessionFactory(db_session)
        ),
        worker_process_resources=cast(
            "RemoteIOBoundWorkerProcessResources",
            _DummyRemoteResources(_FakeEmailProviderClient()),
        ),
    )

    payload = DeliverNotificationInputPayload(
        user_id=owner_user.id,
        originating_photobook_id=photobook.id,
        notification_outbox_id=outbox_id,
        expected_dispatch_token=token_payload,
    )
    await proc.process(payload)

    # Should remain SENDING with original token, and no attempts were written
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.SENDING
    assert outbox.dispatch_token == token_db

    attempts = (
        (
            await db_session.execute(
                select(DAONotificationDeliveryAttempts).where(
                    getattr(DAONotificationDeliveryAttempts, "notification_outbox_id")
                    == outbox_id
                )
            )
        )
        .scalars()
        .all()
    )
    assert attempts == []


@pytest.mark.asyncio
async def test_E3_lease_expired_noop(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed outbox
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[_email_recipient("lease_expired@example.com")],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    outbox_id = resp.recipients[0].outbox_results[0].outbox_id

    # Claim but set lease to past
    token = uuid4()
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == outbox_id)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_token=token,
            dispatch_claimed_at=utcnow() - timedelta(hours=2),
            dispatch_lease_expires_at=utcnow() - timedelta(minutes=1),
            dispatch_worker_id="proc",
        )
    )
    await db_session.commit()

    proc = RemoteDeliverNotificationJobProcessor(
        job_id=uuid4(),
        asset_manager=cast("AssetManager", _DummyAssetManager()),
        db_session_factory=cast(
            "AsyncSessionFactory", _DummySessionFactory(db_session)
        ),
        worker_process_resources=cast(
            "RemoteIOBoundWorkerProcessResources",
            _DummyRemoteResources(_FakeEmailProviderClient()),
        ),
    )

    payload = DeliverNotificationInputPayload(
        user_id=owner_user.id,
        originating_photobook_id=photobook.id,
        notification_outbox_id=outbox_id,
        expected_dispatch_token=token,
    )
    await proc.process(payload)

    # No changes; still SENDING with same token; no attempts written
    outbox = (
        await db_session.execute(
            select(DAONotificationOutbox).where(
                getattr(DAONotificationOutbox, "id") == outbox_id
            )
        )
    ).scalar_one()
    assert outbox.status == ShareChannelStatus.SENDING
    assert outbox.dispatch_token == token
    attempts = (
        (
            await db_session.execute(
                select(DAONotificationDeliveryAttempts).where(
                    getattr(DAONotificationDeliveryAttempts, "notification_outbox_id")
                    == outbox_id
                )
            )
        )
        .scalars()
        .all()
    )
    assert attempts == []


@pytest.mark.asyncio
async def test_E4_terminal_status_short_circuit(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    async with safe_transaction(db_session):
        resp = await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[_email_recipient("already_sent@example.com")],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    outbox_id = resp.recipients[0].outbox_results[0].outbox_id
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == outbox_id)
        .values(status=ShareChannelStatus.SENT)
    )
    await db_session.commit()

    proc = RemoteDeliverNotificationJobProcessor(
        job_id=uuid4(),
        asset_manager=cast("AssetManager", _DummyAssetManager()),
        db_session_factory=cast(
            "AsyncSessionFactory", _DummySessionFactory(db_session)
        ),
        worker_process_resources=cast(
            "RemoteIOBoundWorkerProcessResources",
            _DummyRemoteResources(_FakeEmailProviderClient()),
        ),
    )

    payload = DeliverNotificationInputPayload(
        user_id=owner_user.id,
        originating_photobook_id=photobook.id,
        notification_outbox_id=outbox_id,
        expected_dispatch_token=uuid4(),
    )
    await proc.process(payload)

    # Still SENT, and no new attempts written
    attempts = (
        (
            await db_session.execute(
                select(DAONotificationDeliveryAttempts).where(
                    getattr(DAONotificationDeliveryAttempts, "notification_outbox_id")
                    == outbox_id
                )
            )
        )
        .scalars()
        .all()
    )
    assert attempts == []


@pytest.mark.asyncio
async def test_E5_sms_raises_not_implemented_but_logs_processing_first(
    db_session: AsyncSession, owner_user: DAOUsers, photobook: DAOPhotobooks
) -> None:
    # Seed SMS channel
    async with safe_transaction(db_session):
        await initialize_shares_and_channels(
            session=db_session,
            user_id=owner_user.id,
            photobook_id=photobook.id,
            req=ShareCreateRequest(
                recipients=[
                    ShareRecipientSpec(
                        channels=[
                            ShareChannelSpec(
                                channel_type=ShareChannelType.SMS,
                                destination="+15551234567",
                            )
                        ],
                        recipient_display_name="SMS Friend",
                    )
                ],
                sender_display_name="Owner",
                scheduled_for=None,
            ),
        )
    # Grab the outbox (only one)
    outbox = (await db_session.execute(select(DAONotificationOutbox))).scalars().one()
    # Claim it
    token = uuid4()
    await db_session.execute(
        update(DAONotificationOutbox)
        .where(getattr(DAONotificationOutbox, "id") == outbox.id)
        .values(
            status=ShareChannelStatus.SENDING,
            dispatch_token=token,
            dispatch_claimed_at=utcnow(),
            dispatch_worker_id="proc",
        )
    )
    await db_session.commit()

    proc = RemoteDeliverNotificationJobProcessor(
        job_id=uuid4(),
        asset_manager=cast("AssetManager", _DummyAssetManager()),
        db_session_factory=cast(
            "AsyncSessionFactory", _DummySessionFactory(db_session)
        ),
        worker_process_resources=cast(
            "RemoteIOBoundWorkerProcessResources",
            _DummyRemoteResources(_FakeEmailProviderClient()),
        ),
    )

    payload = DeliverNotificationInputPayload(
        user_id=owner_user.id,
        originating_photobook_id=photobook.id,
        notification_outbox_id=outbox.id,
        expected_dispatch_token=token,
    )
    with pytest.raises(NotImplementedError):
        await proc.process(payload)

    # PROCESSING attempt should exist even though channel isn't implemented
    attempts = (
        (
            await db_session.execute(
                select(DAONotificationDeliveryAttempts).where(
                    getattr(DAONotificationDeliveryAttempts, "notification_outbox_id")
                    == outbox.id
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(attempts) == 1
    assert attempts[0].event == NotificationDeliveryEvent.PROCESSING
