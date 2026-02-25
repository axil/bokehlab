# numeric labels
plot([[1,2,3],[4,5,6]], label=np.array([1, 2]), legend_loc='outside')

# legend_title
plot([1,2,4], label='asdf', legend_title='qwer')

# datetime
from zoneinfo import ZoneInfo
from datetime import datetime
times_ny = [
    datetime(2024, 3, 15, 8, 0, tzinfo=ZoneInfo("America/New_York")),
    datetime(2024, 3, 15, 12, 0, tzinfo=ZoneInfo("America/New_York")),
    datetime(2024, 3, 15, 18, 0, tzinfo=ZoneInfo("America/New_York")),
]
idx = pd.to_datetime(times_ny)
s = pd.Series(times_ny)
ts = [pd.Timestamp(q) for q in times_ny]
val = [10, 20, 15]

plot(times_ny, val)
plot(idx, val)
plot([idx], [val])
plot(s, val)
plot(ts, val)

df = pd.DataFrame({'val': val}, index=times_ny)
plot(df.val)