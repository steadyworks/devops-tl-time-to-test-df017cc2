-- migrate:up
ALTER TABLE public.notification_outbox
    RENAME COLUMN claimed_at TO dispatch_claimed_at;

-- migrate:down
ALTER TABLE public.notification_outbox
    RENAME COLUMN dispatch_claimed_at TO claimed_at;
