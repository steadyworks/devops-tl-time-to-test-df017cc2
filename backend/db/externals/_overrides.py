# pyright: reportPrivateUsage=false

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable, Optional, Self

from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dal import (
    DALAssets,
    DALPhotobookComments,
    DALShareChannels,
    DALShares,
    FilterOp,
)
from backend.db.data_models import (
    CommentStatus,
    DAOAssets,
    DAOPages,
    DAOPhotobooks,
    DAOShareChannels,
    DAOShares,
)
from backend.db.data_models.types import (
    MessageOption,
    PageSchema,
    PhotobookSchema,
    SharedWithUserAvatar,
)
from backend.db.utils.common import retrieve_available_asset_key_in_order_of
from backend.lib.asset_manager.base import AssetManager
from backend.route_handler.share_v0 import SharedData, ShareV0APIHandler

from ._generated_DO_NOT_USE import (
    APIResponseModelConvertibleFromDAOMixin,
    ShareChannelsOverviewResponse,
    SharesOverviewResponse,
    _AssetsOverviewResponse,
    _PagesOverviewResponse,
    _PhotobooksOverviewResponse,
)

if TYPE_CHECKING:
    from uuid import UUID


class AssetsOverviewResponse(_AssetsOverviewResponse):
    asset_key_original: Optional[str] = Field(default=None, exclude=True)
    asset_key_display: Optional[str] = Field(default=None, exclude=True)
    asset_key_llm: Optional[str] = Field(default=None, exclude=True)
    asset_key_thumbnail: Optional[str] = Field(default=None, exclude=True)

    signed_asset_url: str
    signed_asset_url_thumbnail: str

    @classmethod
    async def rendered_from_daos(
        cls,
        daos: list[DAOAssets],
        asset_manager: AssetManager,
    ) -> list[Self]:
        uuid_asset_keys_map_display = {
            dao.id: retrieve_available_asset_key_in_order_of(
                dao,
                [
                    "asset_key_display",
                    "asset_key_original",
                    "asset_key_llm",
                ],
            )
            for dao in daos
        }
        uuid_asset_keys_map_thumbnail = {
            dao.id: retrieve_available_asset_key_in_order_of(
                dao,
                [
                    "asset_key_thumbnail",
                    "asset_key_llm",
                    "asset_key_display",
                    "asset_key_original",
                ],
            )
            for dao in daos
        }
        signed_urls = await asset_manager.generate_signed_urls_batched(
            list(uuid_asset_keys_map_display.values())
            + list(uuid_asset_keys_map_thumbnail.values())
        )
        resps: list[Self] = []
        for dao in daos:
            signed_asset_url_or_exception = signed_urls.get(
                uuid_asset_keys_map_display[dao.id]
            )
            signed_asset_url_thumbnail_or_exception = signed_urls.get(
                uuid_asset_keys_map_thumbnail[dao.id]
            )
            resps.append(
                cls(
                    **dao.model_dump(),
                    signed_asset_url=(
                        signed_asset_url_or_exception
                        if isinstance(signed_asset_url_or_exception, str)
                        else ""
                    ),
                    signed_asset_url_thumbnail=(
                        signed_asset_url_thumbnail_or_exception
                        if isinstance(signed_asset_url_thumbnail_or_exception, str)
                        else ""
                    ),
                )
            )

        return resps


class PhotobooksOverviewResponse(_PhotobooksOverviewResponse):
    thumbnail_asset_signed_url: Optional[str]
    thumbnail_asset_blur_data_url: Optional[str]
    num_comments: int
    shared_with: list[SharedWithUserAvatar]
    suggested_overall_gift_message_alternative_options: Optional[dict[str, Any]] = (
        Field(default=None, exclude=True)
    )
    suggested_overall_gift_message_alternative_options_parsed: Optional[
        list[MessageOption]
    ] = None
    shares: list[SharesOverviewResponse]
    share_channels: list[ShareChannelsOverviewResponse]

    @classmethod
    async def rendered_from_daos(
        cls: type[Self],
        daos: Iterable[DAOPhotobooks],
        db_session: AsyncSession,
        asset_manager: AssetManager,
    ) -> list[Self]:
        # Step 4: Collect all asset_ids used
        thumbnail_asset_ids = [
            dao.thumbnail_asset_id for dao in daos if dao.thumbnail_asset_id is not None
        ]
        thumbnail_asset_list = await DALAssets.get_by_ids(
            db_session, thumbnail_asset_ids
        )
        thumbnail_assets_by_ids = {asset.id: asset for asset in thumbnail_asset_list}

        # Step 5: Generate signed URLs for original asset keys
        uuid_asset_keys_map = {
            asset.id: retrieve_available_asset_key_in_order_of(
                asset,
                [
                    "asset_key_llm",
                    "asset_key_display",
                    "asset_key_original",
                ],
            )
            for asset in thumbnail_asset_list
        }
        signed_urls = await asset_manager.generate_signed_urls_batched(
            list(uuid_asset_keys_map.values())
        )

        dao_ids = [dao.id for dao in daos]
        share_daos, share_channel_daos = await asyncio.gather(
            DALShares.list_all(db_session, {"photobook_id": (FilterOp.IN, dao_ids)}),
            DALShareChannels.list_all(
                db_session, {"photobook_id": (FilterOp.IN, dao_ids)}
            ),
        )
        share_dao_map: dict[UUID, list[DAOShares]] = defaultdict(list)
        share_channel_dao_map: dict[UUID, list[DAOShareChannels]] = defaultdict(list)

        for share_dao in share_daos:
            share_dao_map[share_dao.photobook_id].append(share_dao)
        for share_channel_dao in share_channel_daos:
            share_channel_dao_map[share_channel_dao.photobook_id].append(
                share_channel_dao
            )

        rendered_resps: list[Self] = []
        for dao in daos:
            thumbnail_signed_url, thumbnail_asset_blur_data_url = None, None
            if dao.thumbnail_asset_id is not None:
                thumbnail_asset = thumbnail_assets_by_ids.get(dao.thumbnail_asset_id)
                if thumbnail_asset is not None:
                    thumbnail_asset_blur_data_url = thumbnail_asset.blur_data_url
                    thumbnail_signed_url_or_exception = signed_urls.get(
                        uuid_asset_keys_map[thumbnail_asset.id]
                    )
                    if isinstance(thumbnail_signed_url_or_exception, str):
                        thumbnail_signed_url = thumbnail_signed_url_or_exception

            comment_count = await DALPhotobookComments.count(
                db_session,
                filters={
                    "photobook_id": (FilterOp.EQ, dao.id),
                    "status": (FilterOp.EQ, CommentStatus.VISIBLE),
                },
            )

            # Get Current Photobook shares V0
            # current photobook shares
            shared_data: SharedData = await ShareV0APIHandler.find_photobook_shares(
                db_session=db_session,
                photobook_id=dao.id,
            )
            shared_with_list: list[SharedWithUserAvatar] = []
            for user in shared_data.already_shared_users:
                shared_with_list.append(
                    SharedWithUserAvatar(
                        email=user.email,
                        avatar_url="",
                        username=user.username,
                    )
                )
            for email in shared_data.already_shared_emails:
                shared_with_list.append(
                    SharedWithUserAvatar(
                        email=email,
                        avatar_url=None,
                        username=None,
                    )
                )

            resp = cls(
                **dao.model_dump(),
                thumbnail_asset_signed_url=thumbnail_signed_url,
                thumbnail_asset_blur_data_url=thumbnail_asset_blur_data_url,
                num_comments=comment_count,
                shared_with=shared_with_list,
                suggested_overall_gift_message_alternative_options_parsed=PhotobookSchema.deserialize_overall_gift_message_alternatives(
                    dao.suggested_overall_gift_message_alternative_options
                ),
                shares=SharesOverviewResponse.from_daos(share_dao_map.get(dao.id, [])),
                share_channels=ShareChannelsOverviewResponse.from_daos(
                    share_channel_dao_map.get(dao.id, [])
                ),
            )
            rendered_resps.append(resp)
        return rendered_resps


class PagesOverviewResponse(
    _PagesOverviewResponse, APIResponseModelConvertibleFromDAOMixin[DAOPages]
):
    user_message_alternative_options: Optional[dict[str, Any]] = Field(
        default=None, exclude=True
    )
    user_message_alternative_options_parsed: Optional[list[MessageOption]] = None

    @classmethod
    def from_dao(cls, dao: DAOPages) -> Self:
        return cls(
            **dao.model_dump(),
            user_message_alternative_options_parsed=PageSchema.deserialize_page_message_alternatives(
                dao.user_message_alternative_options
            ),
        )
