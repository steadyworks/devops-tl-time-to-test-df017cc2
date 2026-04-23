# backend/lib/notifs/dispatch_service.py

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dal import safe_transaction
from backend.db.data_models import (
    DAONotificationOutbox,
    DAOShares,
    ShareAccessPolicy,
    ShareChannelStatus,
)
from backend.lib.utils.common import utcnow
from backend.worker.job_processor.types import (
    DeliverNotificationInputPayload,
    JobType,
)

if TYPE_CHECKING:
    from backend.lib.job_manager.protocol import JobManagerProtocol


async def claim_and_enqueue_one_outbox(
    *,
    session: AsyncSession,
    job_manager: JobManagerProtocol,  # exposes .enqueue(JobType, payload, max_retries=..., db_session=...)
    outbox_id: UUID,
    user_id: UUID,  # <-- NEW: required for JobInputPayload.user_id
    worker_id: str,
    lease_seconds: int = 600,
) -> Optional[UUID]:
    """
    Atomically claim a due notification_outbox row and enqueue a delivery job.

    Returns job_id if enqueued, else None if the row was not claimable
    (not due, already claimed, revoked, or already terminal).

    Transaction contract: This function opens its own DB transaction and must not be called
    while a transaction is active on the provided AsyncSession.

    Callers should commit/close any open transaction first.
    """
    if session.in_transaction():
        raise RuntimeError(
            "[claim_and_enqueue_*] Must be called with no active transaction on the session. "
            "Commit/close the caller transaction first."
        )

    now_ts = utcnow()
    lease_expires = now_ts + timedelta(seconds=lease_seconds)
    new_token = uuid4()

    o = DAONotificationOutbox
    s = DAOShares

    o_id = getattr(o, "id")
    o_status = getattr(o, "status")
    o_scheduled_for = getattr(o, "scheduled_for")
    o_dispatch_claimed_at = getattr(o, "dispatch_claimed_at")
    o_dispatch_token = getattr(o, "dispatch_token")
    o_photobook_id = getattr(o, "photobook_id")
    o_share_id = getattr(o, "share_id")

    s_id = getattr(s, "id")
    s_access_policy = getattr(s, "access_policy")

    async with safe_transaction(session, "claim_and_enqueue_one", raise_on_fail=True):
        revoked_exists = exists().where(
            and_(
                s_id == o_share_id,
                s_access_policy == ShareAccessPolicy.REVOKED,
            )
        )

        claim_stmt = (
            update(o)
            .where(
                and_(
                    o_id == outbox_id,
                    o_status.in_(
                        [ShareChannelStatus.PENDING, ShareChannelStatus.SCHEDULED]
                    ),
                    or_(o_scheduled_for.is_(None), o_scheduled_for <= now_ts),
                    o_dispatch_claimed_at.is_(None),
                    ~revoked_exists,
                )
            )
            .values(
                status=ShareChannelStatus.SENDING,  # reserve; processor will re-check token
                dispatch_token=new_token,
                dispatch_claimed_at=now_ts,
                dispatch_lease_expires_at=lease_expires,
                dispatch_worker_id=worker_id,
                updated_at=now_ts,
            )
            .returning(o_id, o_photobook_id, o_dispatch_token)
        )

        result = await session.execute(claim_stmt)
        row = result.first()
        if not row:
            return None

        claimed_outbox_id: UUID = row[0]
        originating_photobook_id: UUID = row[1]
        claimed_token: UUID = row[2]

    job_id: UUID = await job_manager.enqueue(
        JobType.REMOTE_DELIVER_NOTIFICATION,
        DeliverNotificationInputPayload(
            user_id=user_id,
            originating_photobook_id=originating_photobook_id,
            notification_outbox_id=claimed_outbox_id,
            expected_dispatch_token=claimed_token,
        ),
        max_retries=0,
        db_session=session,
    )
    return job_id


# -------------------------------
# helpers
# -------------------------------


async def _release_claim_best_effort(
    session: AsyncSession,
    *,
    outbox_id: UUID,
    expected_token: UUID,
) -> None:
    """Return the row to PENDING only if we still hold this claim."""
    o = DAONotificationOutbox
    o_id = getattr(o, "id")
    o_status = getattr(o, "status")
    o_dispatch_token = getattr(o, "dispatch_token")
    o_dispatch_claimed_at = getattr(o, "dispatch_claimed_at")
    o_dispatch_lease_expires_at = getattr(o, "dispatch_lease_expires_at")
    o_dispatch_worker_id = getattr(o, "dispatch_worker_id")
    o_updated_at = getattr(o, "updated_at")

    try:
        async with safe_transaction(
            session, "release_claim_best_effort", raise_on_fail=False
        ):
            await session.execute(
                update(o)
                .where(and_(o_id == outbox_id, o_dispatch_token == expected_token))
                .values(
                    **{
                        o_status.key: ShareChannelStatus.PENDING,
                        o_dispatch_token.key: None,
                        o_dispatch_claimed_at.key: None,
                        o_dispatch_lease_expires_at.key: None,
                        o_dispatch_worker_id.key: None,
                        o_updated_at.key: utcnow(),
                    }
                )
            )
    except Exception:
        # swallow — a sweeper can still reclaim by lease expiry later
        await session.rollback()


# -------------------------------
# batch sweeper
# -------------------------------


async def claim_and_enqueue_ready_batch(
    *,
    session: AsyncSession,
    job_manager: JobManagerProtocol,
    user_id: UUID,
    worker_id: str,
    limit: int = 100,
    lease_seconds: int = 600,
) -> list[UUID]:
    """
    Claim up to `limit` due outbox rows (PENDING/SCHEDULED and due now),
    and enqueue one job per claimed row.

    Returns list of job_ids enqueued.

    Transaction contract: This function opens its own DB transaction and must not be called
    while a transaction is active on the provided AsyncSession.

    Callers should commit/close any open transaction first.
    """
    if session.in_transaction():
        raise RuntimeError(
            "[claim_and_enqueue_*] Must be called with no active transaction on the session. "
            "Commit/close the caller transaction first."
        )

    now_ts = utcnow()
    lease_expires = now_ts + timedelta(seconds=lease_seconds)

    o = DAONotificationOutbox
    s = DAOShares

    # columns (getattr appeases linters)
    o_id = getattr(o, "id")
    o_status = getattr(o, "status")
    o_scheduled_for = getattr(o, "scheduled_for")
    o_dispatch_claimed_at = getattr(o, "dispatch_claimed_at")
    o_dispatch_token = getattr(o, "dispatch_token")
    o_dispatch_lease_expires_at = getattr(o, "dispatch_lease_expires_at")
    o_dispatch_worker_id = getattr(o, "dispatch_worker_id")
    o_updated_at = getattr(o, "updated_at")
    o_created_at = getattr(o, "created_at")
    o_photobook_id = getattr(o, "photobook_id")
    o_share_id = getattr(o, "share_id")

    s_id = getattr(s, "id")
    s_access_policy = getattr(s, "access_policy")

    revoked_exists = exists().where(
        and_(s_id == o_share_id, s_access_policy == ShareAccessPolicy.REVOKED)
    )

    # 1) SELECT ids we will claim, with row-level locks and SKIP LOCKED
    due_cte = (
        select(o_id.label("id"), o_photobook_id.label("photobook_id"))
        .where(
            and_(
                o_status.in_(
                    [ShareChannelStatus.PENDING, ShareChannelStatus.SCHEDULED]
                ),
                or_(o_scheduled_for.is_(None), o_scheduled_for <= now_ts),
                o_dispatch_claimed_at.is_(None),
                ~revoked_exists,
            )
        )
        .order_by(o_scheduled_for.asc().nullsfirst(), o_created_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True)
        .cte("due")
    )

    # 2) UPDATE ... FROM due CTE to atomically claim and generate per-row tokens
    claim_stmt = (
        update(o)
        .where(o_id.in_(select(due_cte.c.id)))
        .values(
            **{
                o_status.key: ShareChannelStatus.SENDING,
                o_dispatch_token.key: func.gen_random_uuid(),  # per-row token from DB
                o_dispatch_claimed_at.key: now_ts,
                o_dispatch_lease_expires_at.key: lease_expires,
                o_dispatch_worker_id.key: worker_id,
                o_updated_at.key: now_ts,
            }
        )
        .returning(o_id, o_photobook_id, o_dispatch_token)
    )

    claimed_rows: list[tuple[UUID, UUID, UUID]] = []
    async with safe_transaction(session, "claim_ready_batch", raise_on_fail=True):
        result = await session.execute(claim_stmt)
        claimed_rows = [(row[0], row[1], row[2]) for row in result.fetchall()]

    job_ids: list[UUID] = []
    # 3) Enqueue one job per claimed row. If enqueue for a row fails, release that claim.
    for outbox_id, originating_photobook_id, token in claimed_rows:
        try:
            job_id: UUID = await job_manager.enqueue(
                JobType.REMOTE_DELIVER_NOTIFICATION,
                DeliverNotificationInputPayload(
                    user_id=user_id,
                    originating_photobook_id=originating_photobook_id,
                    notification_outbox_id=outbox_id,
                    expected_dispatch_token=token,
                ),
                max_retries=0,
                db_session=session,
            )
            job_ids.append(job_id)
        except Exception:
            await _release_claim_best_effort(
                session, outbox_id=outbox_id, expected_token=token
            )

    return job_ids


# -------------------------------
# lease reclaimer
# -------------------------------


async def reclaim_and_enqueue_expired_leases(
    *,
    session: AsyncSession,
    job_manager: JobManagerProtocol,
    user_id: UUID,
    worker_id: str,
    limit: int = 100,
    lease_seconds: int = 600,
) -> list[UUID]:
    """
    Reclaim up to `limit` rows that are SENDING but whose lease has expired,
    refreshing token/lease and enqueueing jobs.

    Transaction contract: This function opens its own DB transaction and must not be called
    while a transaction is active on the provided AsyncSession.

    Callers should commit/close any open transaction first.
    """
    if session.in_transaction():
        raise RuntimeError(
            "[claim_and_enqueue_*] Must be called with no active transaction on the session. "
            "Commit/close the caller transaction first."
        )

    now_ts = utcnow()
    lease_expires = now_ts + timedelta(seconds=lease_seconds)

    o = DAONotificationOutbox
    s = DAOShares

    o_id = getattr(o, "id")
    o_status = getattr(o, "status")
    o_dispatch_claimed_at = getattr(o, "dispatch_claimed_at")
    o_dispatch_token = getattr(o, "dispatch_token")
    o_dispatch_lease_expires_at = getattr(o, "dispatch_lease_expires_at")
    o_dispatch_worker_id = getattr(o, "dispatch_worker_id")
    o_updated_at = getattr(o, "updated_at")
    o_photobook_id = getattr(o, "photobook_id")
    o_share_id = getattr(o, "share_id")

    s_id = getattr(s, "id")
    s_access_policy = getattr(s, "access_policy")

    revoked_exists = exists().where(
        and_(s_id == o_share_id, s_access_policy == ShareAccessPolicy.REVOKED)
    )

    expired_cte = (
        select(o_id.label("id"), o_photobook_id.label("photobook_id"))
        .where(
            and_(
                o_status == ShareChannelStatus.SENDING,
                o_dispatch_claimed_at.is_not(None),
                o_dispatch_lease_expires_at <= now_ts,
                ~revoked_exists,
            )
        )
        .order_by(o_dispatch_claimed_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True)
        .cte("expired")
    )

    reclaim_stmt = (
        update(o)
        .where(o_id.in_(select(expired_cte.c.id)))
        .values(
            **{
                o_status.key: ShareChannelStatus.SENDING,  # keep as sending; we're re-claiming
                o_dispatch_token.key: func.gen_random_uuid(),
                o_dispatch_claimed_at.key: now_ts,
                o_dispatch_lease_expires_at.key: lease_expires,
                o_dispatch_worker_id.key: worker_id,
                o_updated_at.key: now_ts,
            }
        )
        .returning(o_id, o_photobook_id, o_dispatch_token)
    )

    reclaimed_rows: list[tuple[UUID, UUID, UUID]] = []
    async with safe_transaction(session, "reclaim_expired_leases", raise_on_fail=True):
        result = await session.execute(reclaim_stmt)
        reclaimed_rows = [(row[0], row[1], row[2]) for row in result.fetchall()]

    job_ids: list[UUID] = []
    for outbox_id, originating_photobook_id, token in reclaimed_rows:
        try:
            job_id: UUID = await job_manager.enqueue(
                JobType.REMOTE_DELIVER_NOTIFICATION,
                DeliverNotificationInputPayload(
                    user_id=user_id,
                    originating_photobook_id=originating_photobook_id,
                    notification_outbox_id=outbox_id,
                    expected_dispatch_token=token,
                ),
                max_retries=0,
                db_session=session,
            )
            job_ids.append(job_id)
        except Exception:
            await _release_claim_best_effort(
                session, outbox_id=outbox_id, expected_token=token
            )

    return job_ids
