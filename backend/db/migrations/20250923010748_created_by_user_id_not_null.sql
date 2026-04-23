-- migrate:up

-- First, backfill any NULLs with a placeholder (if needed).
-- Replace `00000000-0000-0000-0000-000000000000` with a real system user id
-- if you have one, otherwise this step may fail if rows exist with NULLs.
UPDATE public.notification_outbox
SET created_by_user_id = '00000000-0000-0000-0000-000000000000'
WHERE created_by_user_id IS NULL;

-- Then enforce non-null constraint
ALTER TABLE public.notification_outbox
ALTER COLUMN created_by_user_id SET NOT NULL;

-- migrate:down
ALTER TABLE public.notification_outbox
ALTER COLUMN created_by_user_id DROP NOT NULL;
