import pandas as pd
import numpy as np
import os 

def comb_min(user_job, job_user):
    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    merged[['score_user_job', 'score_job_user']] = merged[['score_user_job', 'score_job_user']].fillna(0)

    merged['score'] = merged[['score_user_job', 'score_job_user']].min(axis=1)
    return merged[['U_ID', 'J_ID', 'score']]

def comb_max(user_job, job_user):
    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    merged[['score_user_job', 'score_job_user']] = merged[['score_user_job', 'score_job_user']].fillna(0)

    merged['score'] = merged[['score_user_job', 'score_job_user']].max(axis=1)
    return merged[['U_ID', 'J_ID', 'score']]

def comb_med(user_job, job_user):
    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    merged[['score_user_job', 'score_job_user']] = merged[['score_user_job', 'score_job_user']].fillna(0)

    merged['score'] = merged[['score_user_job', 'score_job_user']].median(axis=1)
    return merged[['U_ID', 'J_ID', 'score']]

def comb_sum(user_job, job_user):
    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    merged[['score_user_job', 'score_job_user']] = merged[['score_user_job', 'score_job_user']].fillna(0)

    merged['score'] = merged['score_user_job'] + merged['score_job_user']
    return merged[['U_ID', 'J_ID', 'score']]

def weighted_sum(user_job, job_user, w1=0.5):
    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    merged[['score_user_job', 'score_job_user']] = merged[['score_user_job', 'score_job_user']].fillna(0)

    merged['score'] = w1 * merged['score_user_job'] + (1 - w1) * merged['score_job_user']
    return merged[['U_ID', 'J_ID', 'score']]

def compute_ranks(df, ascending=False):
    """Helper function to compute ranks for a dataframe"""
    df['rank'] = df.groupby('U_ID')['score'].rank(method="dense", ascending=ascending)
    return df

def borda_fuse(user_job, job_user, w1=0.5):
    """Borda Count Fusion: Assigns points based on ranks and sums them."""
    user_job = compute_ranks(user_job, ascending=False)
    job_user = compute_ranks(job_user, ascending=False)
    
    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    MAX_RANK = merged[['rank_user_job', 'rank_job_user']].max().max()

    merged[['rank_user_job', 'rank_job_user']] = merged[['rank_user_job', 'rank_job_user']].fillna(MAX_RANK + 1)
    
    M_map = merged.groupby('U_ID')['rank_user_job'].transform("max")
    merged['M'] = merged['U_ID'].map(M_map)

    merged['score'] = w1 * (merged['M'] - merged['rank_user_job'] + 1) + (1 - w1) * (merged['M'] - merged['rank_job_user'] + 1)
    
    return merged[['U_ID', 'J_ID', 'score']]

def isr(user_job, job_user, w1=0.5):
    """Inverse Square Rank (ISR): 1 / rank^2"""
    user_job = compute_ranks(user_job, ascending=False)
    job_user = compute_ranks(job_user, ascending=False)

    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    MAX_RANK = merged[['rank_user_job', 'rank_job_user']].max().max()
    merged[['rank_user_job', 'rank_job_user']] = merged[['rank_user_job', 'rank_job_user']].fillna(MAX_RANK + 1)
    merged['score'] = w1 * (1 / (merged['rank_user_job']**2)) + (1 - w1) * (1 / (merged['rank_job_user']**2))
    return merged[['U_ID', 'J_ID', 'score']]


def rrf(user_job, job_user, k=60, w1=0.5):
    """Reciprocal Rank Fusion (RRF): 1 / (k + rank)"""
    user_job = compute_ranks(user_job, ascending=False)
    job_user = compute_ranks(job_user, ascending=False)

    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    MAX_RANK = merged[['rank_user_job', 'rank_job_user']].max().max()
    merged[['rank_user_job', 'rank_job_user']] = merged[['rank_user_job', 'rank_job_user']].fillna(MAX_RANK + 1)

    merged['score'] = w1 * (1 / (k + merged['rank_user_job'])) + (1-w1) * (1 / (k + merged['rank_job_user']))
    
    return merged[['U_ID', 'J_ID', 'score']]




method_dict = {
    'comb_min': comb_min,
    'comb_max': comb_max,
    'comb_med': comb_med,
    'comb_sum': comb_sum,
    'weighted_sum': weighted_sum,
    'borda_fuse': borda_fuse,
    'isr': isr,
    'rrf': rrf
}