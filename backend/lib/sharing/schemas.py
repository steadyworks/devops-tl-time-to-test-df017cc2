from datetime import datetime  # noqa: TCH001
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from backend.db.data_models import (
    ShareChannelType,
)


class ShareChannelSpec(BaseModel):
    channel_type: ShareChannelType
    destination: str
    # If provided, ensures idempotent creation of this outbox row.
    idempotency_key: Optional[str] = None


class ShareRecipientSpec(BaseModel):
    # One of (recipient_user_id) or (share_slug) must be present to make the share deterministically addressable.
    recipient_user_id: Optional[UUID] = None
    # Optional display metadata
    recipient_display_name: Optional[str] = None
    notes: Optional[str] = None

    # Per-recipient channels
    channels: list[ShareChannelSpec] = Field(default_factory=list[ShareChannelSpec])


class ShareCreateRequest(BaseModel):
    # One or more recipients (recipient shares)
    recipients: list[ShareRecipientSpec] = Field(
        default_factory=list[ShareRecipientSpec]
    )
    sender_display_name: Optional[str] = None
    # If provided, we schedule; if omitted and send_now==False, we default to pending with no schedule.
    scheduled_for: Optional[datetime] = None


class ShareChannelResult(BaseModel):
    share_channel_id: UUID
    channel_type: ShareChannelType
    destination: str


class ShareOutboxResult(BaseModel):
    outbox_id: UUID
    share_channel_id: UUID


class ShareRecipientResult(BaseModel):
    share_id: UUID
    share_slug: str
    share_channel_results: list[ShareChannelResult] = Field(
        default_factory=list[ShareChannelResult]
    )
    outbox_results: list[ShareOutboxResult] = Field(
        default_factory=list[ShareOutboxResult]
    )


class ShareCreateResponse(BaseModel):
    photobook_id: UUID
    recipients: list[ShareRecipientResult]
