(base) [mhossen1@dt-login04 CRAFT-CL]$ ARMS="learned_gate" bash explore/Routing_vs_learned_gate/submit_routing_gate_trace8.sh
Submitting router_method=learned_gate  as job CRAFT_CL_routergate_learned_gate_llama32_1b_trace8
Submitted batch job 21315235

Submitted routing-vs-gate arms on TRACE-8 (Llama-1B).
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_vs_learned_gate/logs/trace8_llama1b
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_vs_learned_gate/outputs/CRAFT_CL_routergate_*
(base) [mhossen1@dt-login04 CRAFT-CL]$ ARMS="overwrite" bash Merge_rule_ablation/submit_merge_ablation_gemma_trace8.sh
bash: Merge_rule_ablation/submit_merge_ablation_gemma_trace8.sh: No such file or directory
(base) [mhossen1@dt-login04 CRAFT-CL]$ ARMS="overwrite" bash explore/Merge_rule_ablation/submit_merge_ablation_gemma_trace8.sh
Submitting merge_rule=overwrite  as job CRAFT_CL_merge_overwrite_gemma2b_trace8
Submitted batch job 21315250
Submitting merge_rule=ema  as job CRAFT_CL_merge_ema_gemma2b_trace8
Submitted batch job 21315251
Submitting merge_rule=fisher  as job CRAFT_CL_merge_fisher_gemma2b_trace8
Submitted batch job 21315252
Submitting merge_rule=prototype  as job CRAFT_CL_merge_prototype_gemma2b_trace8
Submitted batch job 21315253

Submitted 4 merge-ablation jobs on TRACE-8 (Gemma-2B-it).
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/Merge_rule_ablation/logs/gemma_trace8
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/Merge_rule_ablation/outputs/CRAFT_CL_merge_*
(base) [mhossen1@dt-login04 CRAFT-CL]$ bash explore/Merge_rule_ablation/submit_merge_ablation_llama15.sh
Submitting merge_rule=overwrite  as job CRAFT_CL_merge_overwrite_llama32_1b_15task
Submitted batch job 21315259
Submitting merge_rule=ema  as job CRAFT_CL_merge_ema_llama32_1b_15task
Submitted batch job 21315260
Submitting merge_rule=fisher  as job CRAFT_CL_merge_fisher_llama32_1b_15task
Submitted batch job 21315261
Submitting merge_rule=prototype  as job CRAFT_CL_merge_prototype_llama32_1b_15task
Submitted batch job 21315262

Submitted 4 merge-ablation jobs on the 15-task stream (Llama-1B).
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/Merge_rule_ablation/logs/llama15
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/Merge_rule_ablation/outputs/CRAFT_CL_merge_*

(base) [mhossen1@dt-login04 CRAFT-CL]$ bash explore/Merge_rule_ablation/submit_merge_ablation_gemma_trace8.sh
Submitting merge_rule=overwrite  as job CRAFT_CL_merge_overwrite_gemma2b_trace8
Submitted batch job 21315285
Submitting merge_rule=ema  as job CRAFT_CL_merge_ema_gemma2b_trace8
Submitted batch job 21315286
Submitting merge_rule=fisher  as job CRAFT_CL_merge_fisher_gemma2b_trace8
Submitted batch job 21315287
Submitting merge_rule=prototype  as job CRAFT_CL_merge_prototype_gemma2b_trace8
Submitted batch job 21315288

Submitted 4 merge-ablation jobs on TRACE-8 (Gemma-2B-it).
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/Merge_rule_ablation/logs/gemma_trace8
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/Merge_rule_ablation/outputs/CRAFT_CL_merge_*


























