"""Bounded interface checks only: no full seasons, training runs or score gates."""
import unittest,math,copy
import torch
from route_rl.local.policy import Policy
from route_rl.local.runtime import NativePolicy
from route_rl.local.encoding import CONTEXT_SIZE,CANDIDATE_SIZE
from route_rl.local.runner import compare,summarize
from route_rl.local.settings import Settings
from route_rl.local.adapter import LocalEconomy

class LocalPipelineTests(unittest.TestCase):
    def test_initial_gate_mass_does_not_depend_on_alternative_count(self):
        torch.set_num_threads(1)
        model=Policy(torch,.03)
        for n in (2,6):
            lp,_=model(torch.zeros(1,CONTEXT_SIZE),torch.zeros(1,n,CANDIDATE_SIZE),torch.ones(1,n,dtype=torch.bool))
            self.assertAlmostEqual(float(lp[0,0].exp().detach()),.97,places=6)
            self.assertAlmostEqual(float(lp[0,1:].exp().sum().detach()),.03,places=6)
        lp,_=model(torch.zeros(1,CONTEXT_SIZE),torch.zeros(1,1,CANDIDATE_SIZE),torch.ones(1,1,dtype=torch.bool))
        self.assertEqual(float(lp[0,0].detach()),0.)

    def test_runtime_and_training_policy_probabilities_match(self):
        torch.manual_seed(12);model=Policy(torch,.15)
        weights={k:v.tolist() for k,v in model.module.state_dict().items() if not k.startswith("value.")}
        native=NativePolicy(weights)
        context=torch.randn(1,CONTEXT_SIZE);features=torch.randn(1,6,CANDIDATE_SIZE)
        lp,_=model(context,features,torch.ones(1,6,dtype=torch.bool))
        actual=torch.tensor(native.probabilities(context[0].tolist(),features[0].tolist()))
        self.assertTrue(torch.allclose(actual,lp[0],atol=2e-6))
        (-lp[0,2]).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.module.parameters()))

    def test_keep_hook_calls_original_without_changing_action(self):
        # Test the actual hook, without executing a simulator or complete baseline.
        controller=object.__new__(LocalEconomy)
        from collections import Counter
        controller.contracts={};controller.stats=Counter();controller.settings=Settings(max_changes=0)
        controller.original_plan=lambda obs,action,st:action["market"].append(["BUY_SEED","WHEAT",2])
        controller.error=None
        action=dict(farmer=["HARVEST"],hands=[],market=[])
        controller._plan_hook(dict(day=10),action,dict(sites={}))
        self.assertEqual(action,dict(farmer=["HARVEST"],hands=[],market=[["BUY_SEED","WHEAT",2]]))
        self.assertIsNone(controller.error)

    def test_paired_reward_is_not_just_profit_or_task_count(self):
        policy=dict(seed=1,seat=0,own_cash=10000,opponent_cash=9000,stats={})
        reference=dict(own_cash=11000,opponent_cash=13000)
        result=compare(policy,reference)
        self.assertEqual(result["delta_cash"],-1000)
        self.assertEqual(result["delta_margin"],3000)
        summary=summarize([result,dict(result,seat=1)])
        self.assertEqual(summary["seeds"],1)
        self.assertIsNone(summary["paired_seed_standard_error"])

if __name__=="__main__":unittest.main()
