-- migrate:up

-- A dedicated enum for delivery-oriented notifications (separate from public.notification_type).
CREATE TYPE public.share_notification_type AS ENUM (
  'shared_with_you'
);

COMMENT ON TYPE public.share_notification_type IS
'Delivery-oriented notification kinds used by notification_outbox and notification_delivery_attempts.';

-- Convert columns from text -> enum (NULL-safe; empty string -> NULL).
ALTER TABLE public.notification_outbox
  ALTER COLUMN notification_type TYPE public.share_notification_type
  USING CASE
    WHEN notification_type IS NULL OR notification_type = '' OR notification_type = 'photobook_sharing_with_you_initial' THEN NULL
    ELSE notification_type::public.share_notification_type
  END;

ALTER TABLE public.notification_delivery_attempts
  ALTER COLUMN notification_type TYPE public.share_notification_type
  USING CASE
    WHEN notification_type IS NULL OR notification_type = '' OR notification_type = 'photobook_sharing_with_you_initial' THEN NULL
    ELSE notification_type::public.share_notification_type
  END;



-- Ensure every row has a value, then enforce NOT NULL and a default going forward.
-- Assumes public.share_notification_type already exists.

-- 1) Set a safe default on the column (applies to future inserts)
ALTER TABLE public.notification_outbox
  ALTER COLUMN notification_type SET DEFAULT 'shared_with_you'::public.share_notification_type;

-- 2) Backfill existing NULLs
UPDATE public.notification_outbox
SET notification_type = 'shared_with_you'::public.share_notification_type
WHERE notification_type IS NULL;

-- 3) Enforce NOT NULL
ALTER TABLE public.notification_outbox
  ALTER COLUMN notification_type SET NOT NULL;


-- migrate:down

-- Revert enum columns back to text, then drop the enum.
ALTER TABLE public.notification_outbox
  ALTER COLUMN notification_type TYPE text
  USING notification_type::text;

ALTER TABLE public.notification_delivery_attempts
  ALTER COLUMN notification_type TYPE text
  USING notification_type::text;

DROP TYPE IF EXISTS public.share_notification_type;

-- Allow NULLs again and drop the default (reversible)
ALTER TABLE public.notification_outbox
  ALTER COLUMN notification_type DROP NOT NULL;

ALTER TABLE public.notification_outbox
  ALTER COLUMN notification_type DROP DEFAULT;
