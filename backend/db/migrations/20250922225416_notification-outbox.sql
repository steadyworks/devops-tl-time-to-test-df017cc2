-- migrate:up

-- 1) New table: notification_outbox
-- One row represents a pending/scheduled/in-flight or terminal send for a specific share_channel.
-- You can create multiple rows over time per channel (e.g., re-send), but we enforce at most one *active* row
-- (pending|scheduled|sending) per share_channel via a partial unique index.
ALTER TYPE public.share_channel_status ADD VALUE IF NOT EXISTS 'canceled';

CREATE TABLE public.notification_outbox (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Denormalized pointers for easy filters/joins:
    photobook_id uuid NOT NULL,
    share_id uuid NOT NULL,
    share_channel_id uuid NOT NULL,
    -- What/how we plan to send:
    channel_type public.share_channel_type NOT NULL,
    provider public.share_provider,
    -- Centralized delivery state:
    status public.share_channel_status NOT NULL DEFAULT 'pending',
    scheduled_for timestamptz,
    last_error text,
    last_provider_message_id text,
    -- Timestamps:
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    notification_type text,
    job_id uuid,
    created_by_user_id uuid
);

COMMENT ON TABLE public.notification_outbox IS
'Authoritative, per-channel notification lifecycle: pending/scheduled/sending/sent/failed/canceled.';

-- FKs (keep referential integrity and cascade deletes)
ALTER TABLE public.notification_outbox
    ADD CONSTRAINT notification_outbox_photobook_id_fkey
    FOREIGN KEY (photobook_id) REFERENCES public.photobooks(id) ON DELETE CASCADE;

ALTER TABLE public.notification_outbox
    ADD CONSTRAINT notification_outbox_share_id_fkey
    FOREIGN KEY (share_id) REFERENCES public.shares(id) ON DELETE CASCADE;

ALTER TABLE public.notification_outbox
    ADD CONSTRAINT notification_outbox_share_channel_id_fkey
    FOREIGN KEY (share_channel_id) REFERENCES public.share_channels(id) ON DELETE CASCADE;

-- Indexes for common scheduler queries
CREATE INDEX idx_notification_outbox_status           ON public.notification_outbox (status);
CREATE INDEX idx_notification_outbox_scheduled_for    ON public.notification_outbox (scheduled_for);
CREATE INDEX idx_notification_outbox_share_channel    ON public.notification_outbox (share_channel_id);
CREATE INDEX idx_notification_outbox_photobook_id     ON public.notification_outbox (photobook_id);
ALTER TABLE public.notification_outbox
    ADD CONSTRAINT notification_outbox_created_by_user_fkey
    FOREIGN KEY (created_by_user_id) REFERENCES public.users(id) ON DELETE SET NULL;

CREATE INDEX idx_notification_outbox_job_id ON public.notification_outbox (job_id);
CREATE INDEX idx_notification_outbox_created_by ON public.notification_outbox (created_by_user_id);



-- Enforce "only one active outbox row per channel" (pending|scheduled|sending)
CREATE UNIQUE INDEX uq_notification_outbox_active_per_channel
    ON public.notification_outbox (share_channel_id)
    WHERE status IN ('pending','scheduled','sending');


-- 2) Link attempts table forward to outbox (nullable so you can switch writers when ready)
ALTER TABLE public.notification_delivery_attempts
    ADD COLUMN notification_outbox_id uuid;

ALTER TABLE public.notification_delivery_attempts
    ADD CONSTRAINT notification_delivery_attempts_outbox_id_fkey
    FOREIGN KEY (notification_outbox_id) REFERENCES public.notification_outbox(id) ON DELETE SET NULL;

-- 3) Remove delivery-state columns from share_channels (they live in outbox now)
--    Drop dependent indexes first.
DROP INDEX IF EXISTS public.idx_share_channels_status;
DROP INDEX IF EXISTS public.idx_share_channels_scheduled_for;

ALTER TABLE public.share_channels
    DROP COLUMN IF EXISTS status,
    DROP COLUMN IF EXISTS last_error,
    DROP COLUMN IF EXISTS last_provider_message_id,
    DROP COLUMN IF EXISTS scheduled_for;

COMMENT ON TABLE public.share_channels IS
'One row per delivery avenue under a RECIPIENT share. No delivery status here; see notification_outbox.';


-- migrate:down

-- Add back columns on share_channels
ALTER TABLE public.share_channels
    ADD COLUMN status public.share_channel_status NOT NULL DEFAULT 'pending',
    ADD COLUMN last_error text,
    ADD COLUMN last_provider_message_id text,
    ADD COLUMN scheduled_for timestamptz;

-- Recreate the indexes we removed
CREATE INDEX IF NOT EXISTS idx_share_channels_status        ON public.share_channels (status);
CREATE INDEX IF NOT EXISTS idx_share_channels_scheduled_for ON public.share_channels (scheduled_for);

-- Remove FK column from attempts
ALTER TABLE public.notification_delivery_attempts
    DROP CONSTRAINT IF EXISTS notification_delivery_attempts_outbox_id_fkey;
ALTER TABLE public.notification_delivery_attempts
    DROP COLUMN IF EXISTS notification_outbox_id;

-- Drop outbox + trigger + function

DROP INDEX IF EXISTS public.uq_notification_outbox_active_per_channel;
DROP INDEX IF EXISTS public.idx_notification_outbox_photobook_id;
DROP INDEX IF EXISTS public.idx_notification_outbox_share_channel;
DROP INDEX IF EXISTS public.idx_notification_outbox_scheduled_for;
DROP INDEX IF EXISTS public.idx_notification_outbox_status;
DROP TABLE IF EXISTS public.notification_outbox;
DROP INDEX IF EXISTS public.idx_notification_outbox_job_id;
DROP INDEX IF EXISTS public.idx_notification_outbox_created_by;
