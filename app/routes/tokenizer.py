# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Tokenizer stage API routes.
"""

from fastapi import APIRouter

router = APIRouter(prefix='/api/tokenizer', tags=['tokenizer'])


@router.get('/status')
def status():
    return {'stage': 'tokenizer', 'status': 'not_implemented'}
