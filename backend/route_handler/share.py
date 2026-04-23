# backend/route_handler/share.py

from uuid import UUID

from fastapi import Request

from backend.db.dal import safe_transaction
from backend.lib.notifs.dispatch_service import claim_and_enqueue_one_outbox
from backend.lib.sharing.schemas import ShareCreateRequest, ShareCreateResponse
from backend.lib.sharing.service import initialize_shares_and_channels
from backend.route_handler.base import (
    RouteHandler,
    enforce_response_model,
)


class ShareAPIHandler(RouteHandler):
    def register_routes(self) -> None:
        self.route(
            "/api/share/{photobook_id}/initialize-share",
            "share_photobook_initialize",
            methods=["POST"],
        )

    @enforce_response_model
    async def share_photobook_initialize(
        self,
        photobook_id: UUID,
        payload: ShareCreateRequest,
        request: Request,
    ) -> ShareCreateResponse:
        # Acquire session (adapt to your DI pattern)
        async with self.app.db_session_factory.new_session() as db_session:
            request_context = await self.get_request_context(request)

            async with safe_transaction(
                db_session, "verify_photobook_ownership", raise_on_fail=True
            ):
                await self.get_photobook_assert_owned_by(
                    db_session, photobook_id, request_context.user_id
                )

            async with safe_transaction(
                db_session, "share_photobook_initialize", raise_on_fail=True
            ):
                resp = await initialize_shares_and_channels(
                    session=db_session,
                    user_id=request_context.user_id,
                    photobook_id=photobook_id,
                    req=payload,
                )

            job_ids: list[UUID] = []
            for r in resp.recipients:
                for ob in r.outbox_results:
                    job_id = await claim_and_enqueue_one_outbox(
                        session=db_session,
                        job_manager=self.app.remote_job_manager_io_bound,
                        outbox_id=ob.outbox_id,
                        worker_id="api-share-init",  # use a stable actor id / hostname
                        lease_seconds=600,
                        user_id=request_context.user_id,
                    )
                    if job_id is not None:
                        job_ids.append(job_id)

            return resp
