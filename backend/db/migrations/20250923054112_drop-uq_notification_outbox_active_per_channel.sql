-- migrate:up
DROP INDEX IF EXISTS uq_notification_outbox_active_per_channel;

-- migrate:down

CREATE UNIQUE INDEX uq_notification_outbox_active_per_channel ON public.notification_outbox USING btree (share_channel_id) WHERE (status = ANY (ARRAY['pending'::public.share_channel_status, 'scheduled'::public.share_channel_status, 'sending'::public.share_channel_status]));
