-- migrate:up

-- Shares: revocation fields + index
ALTER TABLE public.shares
  ADD COLUMN revoked_at timestamptz,
  ADD COLUMN revoked_by_user_id uuid,
  ADD COLUMN revoked_reason text;

CREATE INDEX idx_shares_revoked_at
  ON public.shares (revoked_at)
  WHERE revoked_at IS NOT NULL;

-- Outbox: idempotency key + uniqueness
ALTER TABLE public.notification_outbox
  ADD COLUMN idempotency_key text;

CREATE UNIQUE INDEX uq_outbox_idempotency
  ON public.notification_outbox (share_channel_id, notification_type, idempotency_key)
  WHERE idempotency_key IS NOT NULL;

-- Outbox: dispatch lease fields + ready index
ALTER TABLE public.notification_outbox
  ADD COLUMN dispatch_lease_expires_at timestamptz,
  ADD COLUMN dispatch_worker_id text;

CREATE INDEX idx_outbox_ready
  ON public.notification_outbox (status, scheduled_for, created_at)
  WHERE status IN ('pending','scheduled');

-- Outbox: cancel/scheduling audit fields
ALTER TABLE public.notification_outbox
  ADD COLUMN canceled_at timestamptz,
  ADD COLUMN canceled_by_user_id uuid,
  ADD COLUMN scheduled_by_user_id uuid,
  ADD COLUMN last_scheduled_at timestamptz;

-- migrate:down

-- Drop indexes that depend on the new columns first
DROP INDEX IF EXISTS idx_shares_revoked_at;
DROP INDEX IF EXISTS uq_outbox_idempotency;
DROP INDEX IF EXISTS idx_outbox_ready;

-- Then drop the columns

-- Shares
ALTER TABLE public.shares
  DROP COLUMN IF EXISTS revoked_at,
  DROP COLUMN IF EXISTS revoked_by_user_id,
  DROP COLUMN IF EXISTS revoked_reason;

-- Outbox (idempotency)
ALTER TABLE public.notification_outbox
  DROP COLUMN IF EXISTS idempotency_key;

-- Outbox (dispatch lease)
ALTER TABLE public.notification_outbox
  DROP COLUMN IF EXISTS dispatch_lease_expires_at,
  DROP COLUMN IF EXISTS dispatch_worker_id;

-- Outbox (cancel/scheduling audit)
ALTER TABLE public.notification_outbox
  DROP COLUMN IF EXISTS canceled_at,
  DROP COLUMN IF EXISTS canceled_by_user_id,
  DROP COLUMN IF EXISTS scheduled_by_user_id,
  DROP COLUMN IF EXISTS last_scheduled_at;
