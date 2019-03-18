from datetime import datetime, timedelta

import pytz
from croniter import croniter
from pytz import BaseTzInfo


central_tz = pytz.timezone('US/Central')  # Central timezone
base = central_tz.localize(datetime(2019, 3, 10, 1, 50))

print(f"Base time in ISO format: {base.isoformat()}")

cron = croniter('*/10 * * * *', base)
for _ in range(5):
    print(f"Following cron schedule: {cron.get_next(datetime).isoformat()}")
    # fetches the following time in cron schedule

'''
Outputs:
>>Base time in ISO format: 2019-03-10T01:50:00-06:00
>>Following cron schedule: 2019-03-10T03:00:00-05:00
>>Following cron schedule: 2019-03-10T03:10:00-05:00
>>Following cron schedule: 2019-03-10T03:20:00-05:00
>>Following cron schedule: 2019-03-10T03:30:00DstTzInfo-05:00
>>Following cron schedule: 2019-03-10T03:40:00-05:00
'''


base = central_tz.localize(datetime(2019, 3, 8, 1, 50))

cron = croniter('0 2 * * *', base)  # 2 AM every day
for _ in range(10):
    print(f"Following cron schedule: {cron.get_next(datetime).isoformat()}")
    # fetches the following time in cron schedule

'''
Outputs:
>>Following cron schedule: 2019-03-08T02:00:00-06:00
>>Following cron schedule: 2019-03-09T02:00:00-06:00
>>Following cron schedule: 2019-03-10T02:00:00-05:00  # doesn't exist in Central timezone
>>Following cron schedule: 2019-03-10T03:00:00-05:00  # Executes an hour after the previous schedule
>>Following cron schedule: 2019-03-11T01:00:00-05:00  # Executes itself on 1 A.M
>>Following cron schedule: 2019-03-11T02:00:00-05:00  # Executes an hour after the previous schedule
>>Following cron schedule: 2019-03-12T02:00:00-05:00  # Comes back to the correct schedule here!!
>>Following cron schedule: 2019-03-13T02:00:00-05:00
>>Following cron schedule: 2019-03-14T02:00:00-05:00
>>Following cron schedule: 2019-03-15T02:00:00-05:00
'''


base = central_tz.localize(datetime(2019, 3, 9, 2, 0))  # and by know you are familiar with the drill
cron = croniter('0 2 * * *', base)
# but wait!!
print(f"Time delta in hours between subsequent schedules across time shifts:"
      f" {(cron.get_next(datetime) - base).total_seconds()/3600}")

'''
Outputs:
Time delta in hours between subsequent schedules across time shifts: 23
'''


base = central_tz.localize(datetime(2019, 3, 8, 1, 50))


def generate_next_schedule(base_time: datetime, tz: BaseTzInfo):
    naive_dt_time = datetime(
        base_time.year, base_time.month, base_time.day, base_time.hour, base_time.minute  # assumption is DST shift
        # happens on minute/hour
    )
    cron = croniter('0 2 * * *', naive_dt_time)
    delta = cron.get_next(datetime) - naive_dt_time
    return tz.normalize(base_time + delta)


for _ in range(10):
    base = generate_next_schedule(base, central_tz)
    print(f"Following schedule: {base.isoformat()}")

'''
Outputs:
>>Following schedule: 2019-03-08T02:00:00-06:00
>>Following schedule: 2019-03-09T02:00:00-06:00
>>Following schedule: 2019-03-10T03:00:00-05:00  # Correct transition across daylight time shift
>>Following schedule: 2019-03-11T02:00:00-05:00  # Correct schedule follows as usual with changed time zone
>>Following schedule: 2019-03-12T02:00:00-05:00
>>Following schedule: 2019-03-13T02:00:00-05:00
>>Following schedule: 2019-03-14T02:00:00-05:00
>>Following schedule: 2019-03-15T02:00:00-05:00
>>Following schedule: 2019-03-16T02:00:00-05:00
>>Following schedule: 2019-03-17T02:00:00-05:00
'''
