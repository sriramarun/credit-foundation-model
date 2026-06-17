# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Data stage API routes.
"""

from fastapi import APIRouter

router = APIRouter(prefix='/api/data', tags=['data'])


@router.get('/status')
def status():
    return {'stage': 'data', 'status': 'not_implemented'}
