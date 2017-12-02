import pandas as pd
import numpy as np

data = pd.read_csv("fake.csv", encoding='utf-8')

# drop articles without text or not in english
for index, row in data.iterrows():
  if(row["text"] is np.nan):
    data.drop(index, axis=0, inplace=True)
  elif(row["language"] != 'english'):
    data.drop(index, axis=0, inplace=True)

# clean data matrix
data.drop("uuid", axis=1, inplace=True)
data.drop("ord_in_thread", axis=1, inplace=True)
data.drop("language", axis=1, inplace=True)
data.drop("crawled", axis=1, inplace=True)
data.drop("country", axis=1, inplace=True)
data.drop("domain_rank", axis=1, inplace=True)
data.drop("thread_title", axis=1, inplace=True)
data.drop("spam_score", axis=1, inplace=True)
data.drop("main_img_url", axis=1, inplace=True)
data.drop("replies_count", axis=1, inplace=True)
data.drop("participants_count", axis=1, inplace=True)
data.drop("likes", axis=1, inplace=True)
data.drop("comments", axis=1, inplace=True)
data.drop("shares", axis=1, inplace=True)
data.drop("type", axis=1, inplace=True)

data.columns = ['authors', 'date', 'title', 'text', 'domain']

# format dates (day-month-year)
for index, row in data.iterrows():
  date = row["date"][:10]
  date_segments = date.split('-')
  date_segments.reverse()
  date = '-'.join(date_segments)

  data.set_value(index, "date", date)

data.to_csv("fake_clean.csv", index=False, encoding='utf-8')