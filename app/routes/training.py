# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Training stage API routes.
"""

from fastapi import APIRouter

router = APIRouter(prefix='/api/training', tags=['training'])


@router.get('/status')
def status():
    return {'stage': 'training', 'status': 'not_implemented'}
