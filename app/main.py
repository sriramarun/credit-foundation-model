# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""FastAPI dashboard entry. Surfaces the four pipeline stages with a working demo
on the Dutch mortgages reference. Lifts orchestration UI from the existing skeleton.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import data, tokenizer, training, evaluation

app = FastAPI(title='Credit Foundation Model Dashboard')
app.include_router(data.router)
app.include_router(tokenizer.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.mount('/', StaticFiles(directory='app/static', html=True), name='static')
