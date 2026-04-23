-- migrate:up
CREATE UNIQUE INDEX uq_outbox_live_per_channel_type
ON public.notification_outbox (share_channel_id, notification_type)
WHERE status IN ('pending','scheduled','sending');


-- migrate:down

DROP INDEX IF EXISTS uq_outbox_live_per_channel_type;