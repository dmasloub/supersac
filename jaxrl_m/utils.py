import argparse
import functools
from functools import partial

import jax
import jax.numpy as jnp

import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def none_or_str(value):
    if value == 'None':
        return None
    return value

def compute_q(anc_agent,obs,actor_params,critic_params):

    actions = anc_agent.actor(obs, params=actor_params)
    q = anc_agent.critic(obs, actions,False,params=critic_params)
   
    return q

def estimate_return(acq_rollout,
                    anc_agent,anc_critic_params,anc_return):
    
    acq_obs = acq_rollout.observations
    acq_masks = acq_rollout.disc_masks
    acq_return = acq_rollout.policy_return
  
    anc_actor_params = anc_agent.actor.params
    acq_actor_params = acq_rollout.policy_params
    
    anc_q = compute_q(anc_agent,acq_obs,anc_actor_params,anc_critic_params)
    acq_q = compute_q(anc_agent,acq_obs,acq_actor_params,anc_critic_params)
    
    acq_adv = ((acq_q - anc_q)*acq_masks).sum()/acq_rollout.num_rollouts
    acq_return_pred = anc_return + acq_adv
  
    
    return acq_return_pred,acq_adv,acq_return


def evaluate_critic(anc_critic_params,anc_agent,
                    anc_return,policy_rollouts):

    
    tmp =  partial(estimate_return,
                   anc_agent=anc_agent,
                   anc_critic_params =anc_critic_params,
                   anc_return = anc_return)
    y_pred,adv,y = jax.vmap(tmp)(policy_rollouts)
    a2 = ((y-y_pred)**2).sum()
    b2=jnp.clip(((y-y.mean())**2).sum(),1e-8)
    R2 = 1-(a2/b2)  
    bias = (y_pred-y).mean()
    ### Upperbound for offline RL ###
    anc_return_e = (y-adv).mean()
    tmp =  partial(estimate_return,
                   anc_agent=anc_agent,
                   anc_critic_params =anc_critic_params,
                   anc_return = anc_return_e)
    y_pred,_,y = jax.vmap(tmp)(policy_rollouts)
    a2 = jnp.clip(((y-y_pred)**2),a_min=1e-4).sum()
    b2=((y-y.mean())**2).sum()
    R2_bound = 1-(a2/b2)  

    return R2,bias,R2_bound,anc_return_e

@jax.jit
def train_evaluation(anc_agent,anc_return,policy_rollouts):
    
    anc_critic_params = anc_agent.critic.params
    R2,bias,R2_bound,anc_return_e = jax.vmap(evaluate_critic,in_axes=(0,None,None,None))(anc_critic_params,anc_agent,anc_return,policy_rollouts)
    
    return R2,bias,R2_bound,anc_return_e


@jax.jit
def test_evaluation(anc_agent,anc_critic_params,
                    anc_return,anc_return_e,
                    policy_rollouts):
    
    R2_test_bound,_,_,_ = jax.vmap(evaluate_critic,in_axes=(0,None,0,None))(anc_critic_params,anc_agent,anc_return_e,policy_rollouts)
    R2_test,_,_,_ = jax.vmap(evaluate_critic,in_axes=(0,None,None,None))(anc_critic_params,anc_agent,anc_return,policy_rollouts)
    
    return R2_test,R2_test_bound




def merge(x,y):

    return jax.tree.map(lambda x,y : jnp.vstack([x,y]),x,y)

def flatten_rollouts(policy_rollouts):
    
    n_policies = len(policy_rollouts)
    merged_rollouts = functools.reduce(merge, policy_rollouts)
    merged_rollouts = jax.tree.map(lambda x:jnp.stack(jnp.split(x,n_policies,axis=0)),merged_rollouts)
    
    def reshape_tree(tree, reference_tree,n_policies):
        def reshape_fn(x, reference_x):
            return jnp.reshape(x, (n_policies,*reference_x.shape))
        
        return jax.tree.map(reshape_fn, tree, reference_tree)
    
    merged_rollouts = reshape_tree(merged_rollouts,policy_rollouts[0],n_policies)
    
    return merged_rollouts

def split_rollouts(flattened_rollouts,MAX_SIZE):
    
    key = jax.random.PRNGKey(0)
    max = flattened_rollouts.policy_return.shape[0]
    size = jnp.minimum(max,MAX_SIZE)

    idxs = jax.random.choice(key,a=max, shape=(size,), replace=False)   
    train_idxs = idxs[:int(0.8*size)]
    test_idxs = idxs[int(0.8*size):]
    train_rollouts = jax.tree.map(lambda x : x[train_idxs],flattened_rollouts)
    test_rollouts = jax.tree.map(lambda x : x[test_idxs],flattened_rollouts)
    
    return train_rollouts,test_rollouts


def measure_action_distance(agent,new_params,old_params,observations):
    
    a_new = agent.actor(observations,params=new_params)
    a_old = agent.actor(observations,params=old_params)
    
    return jnp.linalg.norm(a_new-a_old,axis=-1).mean()


def make_balanced_recent_jax(transitions,
                             current_policy_id: int,
                             M: int,
                             min_total: int = 250,
                             key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    pids = transitions['policy_id']              
    N = pids.shape[0]

    start_pid = max(0, current_policy_id - (M - 1))
    recent_ids_py = np.arange(start_pid, current_policy_id + 1, dtype=np.int32)

    counts = jax.vmap(lambda rid: (pids == rid).sum())(jnp.asarray(recent_ids_py))
    counts_py = np.asarray(counts)               


    present_mask = counts_py > 0
    if not present_mask.any():
        return transitions                      

    present_ids_py = recent_ids_py[present_mask]
    counts_present = counts_py[present_mask]
    B_eff = int(present_ids_py.shape[0])
    
    per_id0 = int(counts_present.min())                         
    if per_id0 * B_eff < min_total:
        per_id = int(np.ceil(min_total / B_eff))
    else:
        per_id = per_id0

    keys = jax.random.split(key, B_eff)
    present_ids = jnp.asarray(present_ids_py, dtype=pids.dtype)

    def sample_one(rid, k):
        mask  = (pids == rid)
        count = mask.sum()
        prob  = jnp.where(mask, 1.0, 0.0)
        prob  = prob / jnp.maximum(count.astype(prob.dtype), 1.0)

        def with_replacement(k_):
            return jax.random.choice(k_, a=N, shape=(per_id,), p=prob, replace=True)
        def without_replacement(k_):
            return jax.random.choice(k_, a=N, shape=(per_id,), p=prob, replace=False)

        return jax.lax.cond(count < per_id, with_replacement, without_replacement, k)

    idxs = jax.vmap(sample_one)(present_ids, keys).reshape(-1)   
    return jax.tree.map(lambda x: x[idxs], transitions)



def select_allowed_policies(agent, transitions, current_pid, alpha_log=0.7):
    pids = np.asarray(transitions['policy_id'])
    uniq = np.unique(pids)
    
    obs  = jax.device_get(transitions['observations'])
    prea = jax.device_get(transitions['pre_actions'])
    mu_logp = np.asarray(transitions['log_probs'])

    dist_k   = agent.actor.apply_fn({'params': agent.actor.params}, obs)
    k_pre_lp = np.asarray(dist_k.log_prob(prea))
    
    tanh_corr = np.sum(2*(np.log(2) - prea - np.log1p(np.exp(-2*prea))), axis=-1)
    k_logp = k_pre_lp - tanh_corr

    d = np.abs(k_logp - mu_logp)

    allowed = set([int(current_pid)])
    for pid in uniq:
        idx = (pids == pid)
        if idx.any():
            med = np.median(d[idx])  
            if med <= alpha_log:
                allowed.add(int(pid))
    return allowed


def add_meta(transitions, current_pid: int, eps_base: float):
    n = transitions['observations'].shape[0]
    return {
        **transitions,
        'current_policy_id': jnp.full((n,), current_pid, dtype=jnp.int32),
        'eps_base': jnp.full((n,), eps_base, dtype=transitions['rewards'].dtype),
    }