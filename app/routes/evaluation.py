# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Evaluation stage API routes.
"""

from fastapi import APIRouter

router = APIRouter(prefix='/api/evaluation', tags=['evaluation'])


@router.get('/status')
def status():
    return {'stage': 'evaluation', 'status': 'not_implemented'}
