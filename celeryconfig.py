import datetime

BROKER_URL = "librabbitmq://guest:guest@kjm248-g1:5672//"

CELERY_RESULT_BACKEND = 'database'

CELERY_RESULT_DBURI = 'postgresql://kmatzen@kjm248-g1:9999/detector'

CELERYD_CONCURRENCY = 1

CELERY_ACKS_LATE = True

CELERY_IMPORTS = ('detect', 'geo_rescore', 'nms')

CELERY_TASK_RESULT_EXPIRES = datetime.timedelta(hours=1)

CELERY_SEND_TASK_ERROR_EMAILS = True

CELERY_RESULT_SERIALIZER = 'json'

#CELERYD_MAX_TASKS_PER_CHILD = 1

CELERYD_FORCE_EXECV = True
