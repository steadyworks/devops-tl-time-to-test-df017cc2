# backend/worker/job_processor/remote_deliver_notification.py

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dal import (
    DALNotificationDeliveryAttempts,
    DALNotificationOutbox,
    DALPhotobooks,
    DALShareChannels,
    DALShares,
    DAONotificationDeliveryAttemptsCreate,
    DAONotificationOutboxUpdate,
    locked_row_by_id,
    safe_commit,
    safe_transaction,
)
from backend.db.data_models import (
    DAONotificationOutbox,
    DAOPhotobooks,
    DAOShareChannels,
    DAOShares,
    NotificationDeliveryEvent,
    ShareAccessPolicy,
    ShareChannelStatus,
    ShareChannelType,
)
from backend.lib.notifs.email.types import EmailAddress, EmailMessage, EmailSendResult
from backend.lib.utils.common import none_throws, utcnow
from backend.worker.process.types import RemoteIOBoundWorkerProcessResources

from .remote import RemoteJobProcessor
from .types import DeliverNotificationInputPayload, DeliverNotificationOutputPayload


class RemoteDeliverNotificationJobProcessor(
    RemoteJobProcessor[
        DeliverNotificationInputPayload,
        DeliverNotificationOutputPayload,
        RemoteIOBoundWorkerProcessResources,
    ]
):
    async def process(
        self, input_payload: DeliverNotificationInputPayload
    ) -> DeliverNotificationOutputPayload:
        now_ts = utcnow()
        async with self.db_session_factory.new_session() as db_session:
            # 1) Look up the outbox row (authoritative), plus its related records
            async with safe_transaction(db_session, "initial_check"):
                outbox = none_throws(
                    await DALNotificationOutbox.get_by_id(
                        db_session, input_payload.notification_outbox_id
                    ),
                    f"Invalid notification_outbox_id: {input_payload.notification_outbox_id}",
                )

                # Short-circuit if terminal (idempotency)
                if outbox.status in (
                    ShareChannelStatus.SENT,
                    ShareChannelStatus.CANCELED,
                ):
                    return DeliverNotificationOutputPayload(job_id=self.job_id)

                share_channel = none_throws(
                    await DALShareChannels.get_by_id(
                        db_session, outbox.share_channel_id
                    ),
                    f"Invalid share channel: {outbox.share_channel_id}",
                )
                share = none_throws(
                    await DALShares.get_by_id(db_session, outbox.share_id),
                    f"Invalid share: {outbox.share_id}",
                )
                photobook = none_throws(
                    await DALPhotobooks.get_by_id(db_session, outbox.photobook_id),
                    f"Invalid photobook: {outbox.photobook_id}",
                )

            # 2) Mark SENDING + log PROCESSING under lock
            #    (If another worker somehow got here, dispatch_token should have prevented it,
            #     but we still guard with a tiny transactional update.)
            email_provider = self.worker_process_resources.email_provider_client

            async with safe_transaction(db_session, "notif start -> SENDING"):
                async with locked_row_by_id(
                    db_session, DAONotificationOutbox, outbox.id
                ) as locked_outbox:
                    if locked_outbox.status in (
                        ShareChannelStatus.SENT,
                        ShareChannelStatus.CANCELED,
                    ):
                        # Someone else finished or it was canceled between reads
                        return DeliverNotificationOutputPayload(job_id=self.job_id)

                    # defensive: if not claimed, do nothing (shouldn’t happen if enqueue only after claim)
                    if locked_outbox.dispatch_token is None:
                        return DeliverNotificationOutputPayload(job_id=self.job_id)

                    if (
                        locked_outbox.dispatch_token
                        != input_payload.expected_dispatch_token
                    ):
                        return DeliverNotificationOutputPayload(job_id=self.job_id)

                    if (
                        locked_outbox.dispatch_lease_expires_at
                        and locked_outbox.dispatch_lease_expires_at <= now_ts
                    ):
                        return DeliverNotificationOutputPayload(job_id=self.job_id)

                    # bail if revoked post-claim
                    if share.access_policy == ShareAccessPolicy.REVOKED:
                        return DeliverNotificationOutputPayload(job_id=self.job_id)

                    # Move to SENDING and (optionally) record provider being used
                    await DALNotificationOutbox.update_by_id(
                        db_session,
                        locked_outbox.id,
                        DAONotificationOutboxUpdate(
                            status=ShareChannelStatus.SENDING,
                            provider=email_provider.get_share_provider(),
                            last_error=None,
                        ),
                    )

                    # Append processing attempt (with outbox linkage)
                    await DALNotificationDeliveryAttempts.create(
                        db_session,
                        DAONotificationDeliveryAttemptsCreate(
                            notification_outbox_id=locked_outbox.id,
                            share_channel_id=share_channel.id,
                            notification_type=locked_outbox.notification_type,
                            channel_type=locked_outbox.channel_type,
                            provider=email_provider.get_share_provider(),
                            event=NotificationDeliveryEvent.PROCESSING,
                        ),
                    )

            # 3) Deliver based on channel type
            if outbox.channel_type == ShareChannelType.EMAIL:
                await self._process_email_notif(
                    db_session=db_session,
                    outbox=outbox,
                    share_channel=share_channel,
                    share=share,
                    photobook=photobook,
                )
            elif outbox.channel_type == ShareChannelType.SMS:
                # TODO: implement SMS provider send + similar logging
                raise NotImplementedError("SMS delivery not yet implemented")
            elif outbox.channel_type == ShareChannelType.APNS:
                # TODO: implement APNS provider send + similar logging
                raise NotImplementedError("APNS delivery not yet implemented")
            else:
                raise RuntimeError(f"Unrecognized channel type: {outbox.channel_type}")

            return DeliverNotificationOutputPayload(job_id=self.job_id)

    async def _process_email_notif(
        self,
        *,
        db_session: AsyncSession,
        outbox: DAONotificationOutbox,
        share_channel: DAOShareChannels,
        share: DAOShares,
        photobook: DAOPhotobooks,
    ) -> None:
        provider_client = self.worker_process_resources.email_provider_client
        provider = provider_client.get_share_provider()

        # Build message (keep idempotency key stable per (outbox, channel))
        msg = EmailMessage(
            subject=f"{photobook.title} from {share.sender_display_name or ''}",
            from_=EmailAddress(
                email="hello@snapgifts.app", name=share.sender_display_name
            ),
            to_=[
                EmailAddress(
                    email=share_channel.destination,
                    name=share.recipient_display_name,
                )
            ],
            html=(
                f"<p>You received a gift from {share.sender_display_name}!"
                f" Check it out at https://snapgifts.app/share/{share.share_slug}</p>"
            ),
            idempotency_key=f"{outbox.notification_type.value}_{outbox.share_channel_id}_{outbox.id}",
        )

        try:
            # send_result = await provider_client.send(msg)

            send_result = EmailSendResult(message_id="123", idempotency_key="456")
            print(msg)

            # On success: mark SENT + store provider message id; append attempt=SENT
            async with safe_commit(db_session, "notif email SENT", raise_on_fail=True):
                await DALNotificationOutbox.update_by_id(
                    db_session,
                    outbox.id,
                    DAONotificationOutboxUpdate(
                        status=ShareChannelStatus.SENT,
                        last_provider_message_id=send_result.message_id,
                        provider=provider,
                        dispatch_token=None,
                    ),
                )
                await DALNotificationDeliveryAttempts.create(
                    db_session,
                    DAONotificationDeliveryAttemptsCreate(
                        notification_outbox_id=outbox.id,
                        share_channel_id=share_channel.id,
                        notification_type=outbox.notification_type,
                        channel_type=outbox.channel_type,
                        provider=provider,
                        event=NotificationDeliveryEvent.SENT,
                        payload={"result": send_result.model_dump(mode="json")},
                    ),
                )

        except Exception as e:
            # On failure: mark FAILED + record error; append attempt=FAILED
            async with safe_commit(
                db_session, "notif email FAILED", raise_on_fail=False
            ):
                await DALNotificationOutbox.update_by_id(
                    db_session,
                    outbox.id,
                    DAONotificationOutboxUpdate(
                        status=ShareChannelStatus.FAILED,
                        last_error=str(e),
                        provider=provider,
                        dispatch_token=None,
                    ),
                )
                await DALNotificationDeliveryAttempts.create(
                    db_session,
                    DAONotificationDeliveryAttemptsCreate(
                        notification_outbox_id=outbox.id,
                        share_channel_id=share_channel.id,
                        notification_type=outbox.notification_type,
                        channel_type=outbox.channel_type,
                        provider=provider,
                        event=NotificationDeliveryEvent.FAILED,
                        payload={"error": str(e)},
                    ),
                )
            # Re-raise so the job framework can mark the job attempt as failed
            raise
