# StepMiner implementation in Python

import numpy as np

def fitStep(data, start=None, end=None, times=None):

  if start is None:
    start = 0
    
  if end is None:
    end = len(data) - 1
    
  count = end - start + 1
  if count <= 0:
    return [0, 0, 0, 0, 0, 0, 0, 0];

  if times is None:
    times = range(len(data))
  else:
    sk = sorted(range(start, end+1), key=lambda x: times[x])
    data = [data[x] for x in sk];
    times = [times[x] for x in sk];
    start = 0;
    end = len(data) - 1;

  sdata = data[start:end+1]
  sseArray = np.zeros(count)
  sum0 = np.sum(sdata)
  mean = np.nanmean(sdata)
  sstot = np.nanvar(sdata)
  sum1 = 0.0;
  count1 = 0;
  m1 = 0.0;
  sum2 = sum0;
  count2 = count;
  m2 = (sum0/count);
  sum1sq = 0.0;
  sum2sq = sstot;
  sse = sum1sq + sum2sq;
  for i in range(count):
    entry = data[i + start];
    if entry is None:
      sseArray[i] = sse;
      continue
    count1 += 1
    count2 -= 1
    if count2 == 0:
      sseArray[i] = sstot;
      continue

    tmp = (mean - (entry + sum1)/count1);
    sum1sq = sum1sq + (entry - mean) * (entry - mean) - \
      tmp * tmp * count1 + (count1 - 1) * (mean - m1) * (mean - m1);
    tmp = (mean - (sum2 - entry)/count2);
    sum2sq = sum2sq - (entry - mean) * (entry - mean) - \
      tmp * tmp * count2 + (count2 + 1) * (mean - m2) * (mean - m2);
    sum1 += entry;
    sum2 -= entry;
    m1 = sum1/count1;
    m2 = sum2/count2;
    sse = sum1sq + sum2sq;
    sseArray[i] = sse;

  bestSse = None
  bestIndex = 0;
  dof = 0;
  for i in range(count):
    if i < (count - 1):
       if times[i+start] == times[i+start+1]:
            continue

    dof += 1
    index = i + start;
    if bestSse is None:
      bestSse = sseArray[i];
      bestIndex = index;

    if (sseArray[i] < bestSse):
      bestSse = sseArray[i];
      bestIndex = index;

  m1 = np.nanmean(data[start:bestIndex+1])
  m2 = np.nanmean(data[bestIndex + 1:count])
  thr = (m1 + m2)/2.0;

  label = 0;
  if (m1 < m2):
    label = 1;
  else:
    label = 2;

  statistic = 0;
  if (bestSse > 0):
    if (dof > 4):
      statistic = (sstot - bestSse)/3/(bestSse/(dof - 4));
    else:
      statistic = (sstot - bestSse)/2/bestSse;

  return [bestIndex, bestSse, sstot, statistic, m1, m2, thr, label];

def fitStepSimple(data):

  data = np.array(data)
  count = len(data)
  sseArray = np.zeros(count)
  sum0 = np.sum(data)
  mean = np.nanmean(data)
  sstot = np.nanvar(data)
  sum1 = 0.0;
  count1 = 0;
  m1 = 0.0;
  sum2 = sum0;
  count2 = count;
  m2 = (sum0/count);
  sum1sq = 0.0;
  sum2sq = sstot;
  sse = sum1sq + sum2sq;
  for i in range(count):
    entry = data[i];
    if entry is None:
      sseArray[i] = sse;
      continue
    count1 += 1
    count2 -= 1
    if count2 == 0:
      sseArray[i] = sstot;
      continue

    tmp = (mean - (entry + sum1)/count1);
    sum1sq = sum1sq + (entry - mean) * (entry - mean) - \
      tmp * tmp * count1 + (count1 - 1) * (mean - m1) * (mean - m1);
    tmp = (mean - (sum2 - entry)/count2);
    sum2sq = sum2sq - (entry - mean) * (entry - mean) - \
      tmp * tmp * count2 + (count2 + 1) * (mean - m2) * (mean - m2);
    sum1 += entry;
    sum2 -= entry;
    m1 = sum1/count1;
    m2 = sum2/count2;
    sse = sum1sq + sum2sq;
    sseArray[i] = sse;

  bestSse = None
  bestIndex = 0;
  dof = 0;
  for i in range(count):
    dof += 1
    index = i
    if bestSse is None:
      bestSse = sseArray[i];
      bestIndex = index;

    if (sseArray[i] < bestSse):
      bestSse = sseArray[i];
      bestIndex = index;

  m1 = np.nanmean(data[0:bestIndex+1])
  m2 = np.nanmean(data[bestIndex + 1:count])
  thr = (m1 + m2)/2.0;

  label = 0;
  if (m1 < m2):
    label = 1;
  else:
    label = 2;

  statistic = 0;
  if (bestSse > 0):
    if (dof > 4):
      statistic = (sstot - bestSse)/3/(bestSse/(dof - 4));
    else:
      statistic = (sstot - bestSse)/2/bestSse;

  return [bestIndex, bestSse, sstot, statistic, m1, m2, thr, label];

