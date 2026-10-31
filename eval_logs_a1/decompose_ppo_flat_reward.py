import sys,numpy as np,random,collections
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper
from stable_baselines3 import PPO
random.seed(0);np.random.seed(0)
S=sys.argv[1]
m=PPO.load(sys.argv[2],device='cpu')  # ppo_cyborg_fixedflat_shaped_vulnerable.zip
c=CybORG(S,'sim',agents={'Red':B_lineAgent});t=BlueTableWrapper(env=c,output_mode='vector');e=EnumActionWrapper(env=t);f=FixedFlatWrapper(env=c,agent='Blue')
e.action_space_change(c.get_action_space('Blue'));A=e.possible_actions
for ep in range(3):
    c.reset();rs=[];reds=collections.Counter()
    for k in range(100):
        o=np.array(f.get_observation('Blue'),dtype=np.float32).flatten()
        a,_=m.predict(o,deterministic=True);r=c.step(agent='Blue',action=A[int(a)])
        rs.append(r.reward); reds[str(c.get_last_action('Red')).split(' ')[0]]+=1
    rs=np.array(rs);nonact=rs+1
    print('ep',ep,'total',round(rs.sum(),1),'restore cost',-100,'breach+outage',round(nonact.sum(),1),'min step',round(rs.min(),1),'steps<=-11',int((rs<=-11).sum()),'red',dict(reds))
