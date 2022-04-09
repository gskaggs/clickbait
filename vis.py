import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

percent_click_bait, click_bait_count, headline_count = [], [], []

with open('percent_click_bait.pkl', 'rb') as handle:
    percent_click_bait, click_bait_count, headline_count = pickle.load(handle)

num_weeks = len(percent_click_bait)
num_months = int(math.ceil(num_weeks / 4))

click_bait_count_by_month = [0] * num_months
headline_count_by_month = [0] * num_months
percent_click_bait_by_month = [0] * num_months

for i in range(num_weeks):
    month = i // 4
    click_bait_count_by_month[month] += click_bait_count[i]
    headline_count_by_month[month] += headline_count[i]

[x for x in range(num_months) if headline_count_by_month[x] != 0]
print(num_weeks, num_months, num_months // 12)

for i in range(num_months):
    if headline_count_by_month[i] == 0:
        percent_click_bait_by_month[i] = -1
        continue
    percent_click_bait_by_month[i] = 100 * click_bait_count_by_month[i] / headline_count_by_month[i]

percent_click_bait_by_month = [x for x in percent_click_bait_by_month if x != -1][:-2]


# Add 20 months to a given datetime object
def month(n):
    given_date = '24/1/2013'

    date_format = '%d/%m/%Y'
    dtObj = datetime.strptime(given_date, date_format)
    future_date = dtObj + relativedelta(months=n)
    return future_date

dates = [month(i) for i in range(len(percent_click_bait_by_month))]


plt.title('Percent of Headlines Clickbait by Month')
plt.ylabel('Percent Clickbait')
plt.xlabel('Month')
plt.xticks(rotation=45)

plt.rcParams.update({'font.size': 14})

plt.plot(dates, percent_click_bait_by_month)
plt.show()

