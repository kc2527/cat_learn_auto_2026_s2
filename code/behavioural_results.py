import numpy as np
import scipy
import pingouin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from util_func_dbm import *

dir_data = '../at_home_data'
dir_data_lab = '../ii_task'

df_lab_rec = []
df_train_rec = []

sns.set_palette('rocket')

for fd in os.listdir(dir_data_lab):
    dir_data_lab_fd = os.path.join(dir_data_lab, fd)
    if os.path.isdir(dir_data_lab_fd):
        for fs in os.listdir(dir_data_lab_fd):
            f_full_path = os.path.join(dir_data_lab_fd, fs)
            if os.path.isfile(f_full_path) and fs.endswith('.csv'):

                    df = pd.read_csv(f_full_path)
                    df['f_name'] = fs
                    df_lab_rec.append(df)

# not reading in cp task
for fd in os.listdir(dir_data):
    dir_data_fd = os.path.join(dir_data, fd)
    if os.path.isdir(dir_data_fd):
        for fs in os.listdir(dir_data_fd):
            f_full_path = os.path.join(dir_data_fd, fs)
            if os.path.isfile(f_full_path) and 'task_cp_' not in fs:
                
                df = pd.read_csv(f_full_path)
                df['f_name'] = fs
                session = df['session_num'].unique()
                df_train_rec.append(df)

d_lab = pd.concat(df_lab_rec, ignore_index=True)
d_home = pd.concat(df_train_rec, ignore_index=True)

# NOTE: create dfs
block_size = 25

d_lab = d_lab.sort_values(['subject_id', 'session_num', 'session_part',
                             'trial']).reset_index(drop=True)
d_lab['acc'] = (d_lab['cat'] == d_lab['resp']).astype(int)
d_lab['trial'] = d_lab.groupby(['subject_id', 'session_num']).cumcount()
d_lab['n_trials'] = d_lab.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_lab['block'] = d_lab.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
d_lab.loc[d_lab['session_num'] == 5, 'session_num'] = 21
d_lab['session_type'] = 'Lab'

d_home = d_home.sort_values(['subject_id', 'session_num', 'session_part',
                               'trial']).reset_index(drop=True)
d_home['acc'] = (d_home['cat'] == d_home['resp']).astype(int)
d_home['trial'] = d_home.groupby(['subject_id', 'session_num']).cumcount()
d_home['n_trials'] = d_home.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_home['block'] = d_home.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
d_home['session_num'] = d_home['session_num'] + 4
d_home['session_type'] = 'Training'

# # NOTE: create a numpy array of the intersection of subjects across all dataframes
# all_subs = np.unique(np.concatenate([d_home.subject_id.unique(),
#                                      d_lab.subject_id.unique()]))
# 
# subs_to_keep = np.intersect1d(all_subs, d_home.subject_id.unique())
# subs_to_keep = np.intersect1d(subs_to_keep, d_lab.subject_id.unique())

# merge all dataframes inserting np.nan into columns that don't exist in a particular dataframe
d_all = pd.concat([d_home, d_lab], ignore_index=True, sort=False)
d_all['session_num'] = d_all.groupby('subject_id')['session_num'].rank(method='dense').astype(int)

# # NOTE: exclude subjects not in all three dataframes (i.e., who did not complete
# # the task correctly)
# d_all = d_all[d_all['subject_id'].isin(subs_to_keep)].reset_index(drop=True)

# aggregate data for upcoming figures
dd_all = (d_all.groupby(['subject_id', 'session_num', 'session_type', 'phase', 'probe_condition'],
          as_index=False)[['acc', 'rt']].mean())

# NOTE: aggregate data for upcoming figures 
pal = sns.color_palette('rocket', 6)
mid3 = pal[1:4]
back2 = pal[4:6]

# NOTE: Figure --- accuracy across all session types
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))

sns.pointplot(data=dd_all[dd_all['phase']=='train'], 
              x='session_num', 
              y='acc',
              hue='session_type', 
              errorbar=('se'), 
              palette=mid3,
              ax=ax[0, 0])

sns.pointplot(data=dd_all[dd_all['phase']=='test'], 
              x='session_num', 
              y='acc',
              hue='probe_condition', 
              errorbar=('se'), 
              dodge=0.25,
              palette=back2,
              ax=ax[0, 0])

[x.set_xticks(np.arange(0, dd_all['session_num'].max(), 1)) for x in ax.flatten()]
ax[0 ,0].set_title('Mean Accuracy Across Days per Session Type', fontsize=16)
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Accuracy (Proportion Correct)')
ax[0, 0].legend(loc='upper left')
plt.show()

#plt.savefig('../figures/accuracy_across_days.png', dpi=300)
#plt.close()

# NOTE: Figure --- reaction time across all session types
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))

sns.pointplot(data=dd_all[dd_all['phase']=='train'],
              x='session_num', 
              y='rt', 
              hue='session_type',
              errorbar=('se'), 
              palette=mid3, 
              ax=ax[0, 0])

sns.pointplot(data=dd_all[dd_all['phase']=='test'], 
              x='session_num', 
              y='rt',
              hue='probe_condition', 
              errorbar=('se'), 
              dodge=0.25,
              palette=back2,
              ax=ax[0, 0])

[x.set_xticks(np.arange(0, dd_all['session_num'].max(), 1)) for x in ax.flatten()]
ax[0 ,0].set_title('Mean Reaction Times Across Days per Session Type', fontsize=16)
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Reaction Time (ms)')
ax[0, 0].legend(loc='upper right')
plt.show()
#plt.savefig('../figures/rts_across_days.png', dpi=300)
#plt.close()

# NOTE: Figure -- accuracy across all lab days (blocks)
d_lab_all = d_all[d_all['session_type'] == 'Lab'].copy()
d_lab_all['block_cont'] = ((d_lab_all['session_num'] - 1) * 26) + d_lab_all['block'] + 1

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8,8))
sns.pointplot(data=d_lab_all, x='block_cont', y='acc', hue='probe_condition',
              errorbar='se', scale=0.75, ax=ax[0,0])
plt.tight_layout()
plt.show()

# NOTE: Stats -- anova across all days: does accuracy improve across days?
d_anova = d_all[~d_all['session_num'].isin(d_all[d_all['session_num']==22])]

res_anova = pg.rm_anova(data=d_anova,
                        dv='acc_plot',
                        within='session_num',
                        subject='subject_id',
                        correction=True)

print('ANOVA \n', res_anova)

# NOTE: Figure -- calculating + plotting cost for accuracy and reaction time
# calibrated with block size of 25
test_start = (d_all['block'].where(d_all['phase'].eq('test'))
              .groupby([d_all['subject_id'], d_all['session_num']]).transform('min'))

keep = (d_all['phase'].eq('train') & d_all['block'].eq(test_start - 2) |
        d_all['phase'].eq('train') & d_all['block'].eq(test_start - 1) |
        d_all['phase'].eq('test') & d_all['block'].eq(test_start, test_start + 1))

d_cost = d_all.loc[keep].copy()

dd = (d_cost.groupby(['subject_id', 'session_num', 'phase', 'probe_condition'])
      [['acc', 'rt']].mean().reset_index())

# accuracy
dd_wide_acc = (
  dd.pivot_table(
      index=['subject_id', 'session_num', 'probe_condition'],
      columns='phase',
      values='acc',
      aggfunc='mean'
  )
  .reset_index()
)

dd_wide_acc['diff_score'] = dd_wide_acc['train'] - dd_wide_acc['test']
dd_wide_acc['probe_condition'] = dd_wide_acc['probe_condition'].astype('category')
dd_wide_acc['subject_id'] = dd_wide_acc['subject_id'].astype('category')

# plot accuracy cost
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide_acc,
              x='session_num',
              y='diff_score',
              hue='probe_condition',
              errorbar='se',
              linestyle='none',
              palette=mid3,
              dodge=True
)
plt.show()

# reaction times
dd_wide_rt = (
  dd.pivot_table(
      index=['subject_id', 'session_num', 'probe_condition'],
      columns='phase',
      values='rt',
      aggfunc='mean'
  )
  .reset_index()
)

# making it test - train to make +ve values
dd_wide_rt['diff_score'] = dd_wide_rt['test'] - dd_wide_rt['train']
dd_wide_rt['probe_condition'] = dd_wide_rt['probe_condition'].astype('category')
dd_wide_rt['subject_id'] = dd_wide_rt['subject_id'].astype('category')

# plot reaction time cost
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide_rt,
              x='session_num',
              y='diff_score',
              hue='probe_condition',
              errorbar='se',
              linestyle='none',
              palette=mid3,
              dodge=True
)
plt.show()

# plot accuracy cost for each subject 
fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
sns.lineplot(data=dd_wide_acc[dd_wide_acc['probe_condition'] == 90],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             ax=ax[0, 0]
)
sns.lineplot(data=dd_wide_acc[dd_wide_acc['probe_condition'] == 180],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             ax=ax[0, 1]
)
sns.move_legend(ax[0, 0], 'upper left', bbox_to_anchor=(1, 1))
sns.move_legend(ax[0, 1], 'upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
