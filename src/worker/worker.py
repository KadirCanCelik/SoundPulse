import os

from celery import Celery
from kombu import Queue

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

app = Celery(
    "worker",
    broker = CELERY_BROKER_URL,
    backend = CELERY_RESULT_BACKEND,
    include = ["src.worker.tasks"]
)


app.conf.task_queues = (

    Queue('gpu_queue', routing_key='gpu.#'),
    Queue('cpu_queue', routing_key='cpu.#'),
    Queue('celery', routing_key='celery.#')
)

app.conf.task_route = {

    'src.worker.tasks.transcribe_audio_task':{
        'queue': 'gpu_queue',
        'routing_key': 'gpu.transcribe'
    },

    'src.worker.tasks.analyze_and_report_task':{
        'queue':'cpu_queue',
        'routing_key':'cpu_analyze'
    },

    '*': {
        'queue': 'cpu_queue',
        'routing_key': 'cpu.default'
    }

}

app.conf.update(

    tasks_acks_late = True,
    worker_prefetch_multiplier = 1,
    result_expires=86400,
    task_track_started=True,
    result_extended=True,

    )