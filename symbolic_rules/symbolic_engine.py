import pandas as pd

def sym_score_df(df_in):
    s = pd.Series(0.0, index=df_in.index)
    s += (df_in['sbytes'] > 1e4)       * 0.4
    s += (df_in['dbytes'] > 5e4)       * 0.2
    s += (df_in['rate']   > 100)       * 0.3
    s += (df_in['trans_depth'] > 10)   * 0.2
    s += (df_in['response_body_len']>1e5)*0.1
    s += (df_in['ct_state_ttl'] < 5)   * 0.3
    s += (df_in['spkts'] > df_in['dpkts']*5) * 0.2
    s += (df_in['is_sm_ips_ports'] == 1) * 0.2
    return s.values