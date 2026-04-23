-- migrate:up

-- rename published -> shared
ALTER TYPE public.photobook_status RENAME VALUE 'published' TO 'shared';

-- add scheduled
ALTER TYPE public.photobook_status ADD VALUE 'scheduled';



-- migrate:down

-- NOTE: Postgres doesn’t support removing enum values directly.
-- To roll back safely, we need to recreate the enum.

-- 1. Rename the existing type
ALTER TYPE public.photobook_status RENAME TO photobook_status_old;

-- 2. Create the old type again
CREATE TYPE public.photobook_status AS ENUM (
    'draft',
    'pending',
    'deleted',
    'permanently_deleted',
    'published'
);

-- 3. Update columns using the old type back to the recreated type
ALTER TABLE photobooks
    ALTER COLUMN status TYPE public.photobook_status
    USING status::text::public.photobook_status;

-- 4. Drop the old type
DROP TYPE public.photobook_status_old;
