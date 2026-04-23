-- migrate:up
ALTER TABLE public.notification_outbox
  RENAME COLUMN job_id TO dispatch_token;

DROP INDEX IF EXISTS public.idx_notification_outbox_job_id;
CREATE INDEX idx_notification_outbox_dispatch_token ON public.notification_outbox (dispatch_token);

ALTER TABLE public.notification_outbox
  ADD COLUMN claimed_at timestamptz;

CREATE INDEX idx_notification_outbox_claimed_at ON public.notification_outbox (claimed_at);


-- migrate:down
ALTER TABLE public.notification_outbox
  RENAME COLUMN dispatch_token TO job_id;

DROP INDEX IF EXISTS public.idx_notification_outbox_dispatch_token;
CREATE INDEX idx_notification_outbox_job_id ON public.notification_outbox (job_id);

ALTER TABLE public.notification_outbox
  DROP COLUMN IF EXISTS claimed_at;
DROP INDEX IF EXISTS public.idx_notification_outbox_claimed_at;
