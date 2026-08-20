[mhossen1@dt-login03 CRAFT-CL]$ bash explore/Eviction_demonstration/submit_eviction_demo.sh
Submitting arm=no_evict  eviction_enabled=0  severity=1.3  abs=1.0  as job CRAFT_CL_evict_no_evict_llama32_1b_fomc_scienceqa
Submitted batch job 21301587
Submitting arm=evict_default  eviction_enabled=1  severity=1.3  abs=1.0  as job CRAFT_CL_evict_evict_default_llama32_1b_fomc_scienceqa
Submitted batch job 21301588
Submitting arm=evict_sens  eviction_enabled=1  severity=1.1  abs=0.3  as job CRAFT_CL_evict_evict_sens_llama32_1b_fomc_scienceqa
Submitted batch job 21301589
Submitting arm=evict_vsens  eviction_enabled=1  severity=1.05  abs=0.1  as job CRAFT_CL_evict_evict_vsens_llama32_1b_fomc_scienceqa
Submitted batch job 21301590

Submitted eviction-demonstration arms (forced FOMC->ScienceQA mis-routing).
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Eviction_demonstration/logs/fomc_scienceqa
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Eviction_demonstration/outputs/CRAFT_CL_evict_*
[mhossen1@dt-login03 CRAFT-CL]$ bash explore/Routing_KL_truncation/submit_duplicate_unit_test.sh
Submitting arm=k64_p16  capture_top_k=64  top_k=64  probe_size=16  as job CRAFT_CL_trunc_dup_k64_p16_llama32_1b
Submitted batch job 21301593
Submitting arm=k128_p16  capture_top_k=128  top_k=128  probe_size=16  as job CRAFT_CL_trunc_dup_k128_p16_llama32_1b
Submitted batch job 21301594
Submitting arm=k256_p16  capture_top_k=256  top_k=256  probe_size=16  as job CRAFT_CL_trunc_dup_k256_p16_llama32_1b
Submitted batch job 21301595
Submitting arm=k512_p16  capture_top_k=512  top_k=512  probe_size=16  as job CRAFT_CL_trunc_dup_k512_p16_llama32_1b
Submitted batch job 21301596
Submitting arm=k64_p8  capture_top_k=64  top_k=64  probe_size=8  as job CRAFT_CL_trunc_dup_k64_p8_llama32_1b
Submitted batch job 21301597
Submitting arm=k64_p64  capture_top_k=64  top_k=64  probe_size=64  as job CRAFT_CL_trunc_dup_k64_p64_llama32_1b
Submitted batch job 21301598

Submitted ScienceQA-A/B duplicate unit test (Llama-1B), diagnostic-only.
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_KL_truncation/logs/scienceqa_dup_test
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_KL_truncation/outputs/CRAFT_CL_trunc_*
[mhossen1@dt-login03 CRAFT-CL]$ bash explore/Routing_KL_truncation/submit_truncation_sweep_trace8.sh
Submitting arm=k64_p16  capture_top_k=64  top_k=64  probe_size=16  as job CRAFT_CL_trunc_k64_p16_llama32_1b_trace8
Submitted batch job 21301614
Submitting arm=k128_p16  capture_top_k=128  top_k=128  probe_size=16  as job CRAFT_CL_trunc_k128_p16_llama32_1b_trace8
Submitted batch job 21301615
Submitting arm=k256_p16  capture_top_k=256  top_k=256  probe_size=16  as job CRAFT_CL_trunc_k256_p16_llama32_1b_trace8
Submitted batch job 21301616
Submitting arm=k512_p16  capture_top_k=512  top_k=512  probe_size=16  as job CRAFT_CL_trunc_k512_p16_llama32_1b_trace8
Submitted batch job 21301617
Submitting arm=k64_p8  capture_top_k=64  top_k=64  probe_size=8  as job CRAFT_CL_trunc_k64_p8_llama32_1b_trace8
Submitted batch job 21301618
Submitting arm=k64_p64  capture_top_k=64  top_k=64  probe_size=64  as job CRAFT_CL_trunc_k64_p64_llama32_1b_trace8
Submitted batch job 21301619

Submitted routing-truncation sweep on TRACE-8 (Llama-1B), diagnostic-only.
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_KL_truncation/logs/trace8_sweep
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_KL_truncation/outputs/CRAFT_CL_trunc_*
[mhossen1@dt-login03 CRAFT-CL]$  bash explore/Routing_vs_learned_gate/submit_routing_gate_trace8.sh
Submitting router_method=kl_routing  as job CRAFT_CL_routergate_kl_routing_llama32_1b_trace8
Submitted batch job 21301620
Submitting router_method=learned_gate  as job CRAFT_CL_routergate_learned_gate_llama32_1b_trace8
Submitted batch job 21301621

Submitted routing-vs-gate arms on TRACE-8 (Llama-1B).
Monitor:        squeue -u $USER
stdout logs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_vs_learned_gate/logs/trace8_llama1b
Run outputs:    /projects/bfwa/mhossen1/CRAFT-CL/explore/Routing_vs_learned_gate/outputs/CRAFT_CL_routergate_*
[mhossen1@dt-login03 CRAFT-CL]$ 


[mhossen1@dt-login03 CRAFT-CL]$ squeue -u mhossen1 --Format=JobID,Account:16,Name:60,State,TimeUsed
JOBID               ACCOUNT         NAME                                                        STATE               TIME                
21301595            bdpm-delta-gpu  CRAFT_CL_trunc_dup_k256_p16_llama32_1b                      PENDING             0:00                
21301596            bdpm-delta-gpu  CRAFT_CL_trunc_dup_k512_p16_llama32_1b                      PENDING             0:00                
21301597            bdpm-delta-gpu  CRAFT_CL_trunc_dup_k64_p8_llama32_1b                        PENDING             0:00                
21301575            bdpm-delta-gpu  CRAFT_CL_stream_l30_llama32_1b_trace8                       PENDING             0:00                
21301598            bdpm-delta-gpu  CRAFT_CL_trunc_dup_k64_p64_llama32_1b                       PENDING             0:00                
21301594            bdpm-delta-gpu  CRAFT_CL_trunc_dup_k128_p16_llama32_1b                      PENDING             0:00                
21301576            bdpm-delta-gpu  CRAFT_CL_stream_f5l25_llama32_1b_trace8                     PENDING             0:00                
21301574            bdpm-delta-gpu  CRAFT_CL_stream_f30_llama32_1b_trace8                       PENDING             0:00                
21301593            bdpm-delta-gpu  CRAFT_CL_trunc_dup_k64_p16_llama32_1b                       PENDING             0:00                
21301590            bdpm-delta-gpu  CRAFT_CL_evict_evict_vsens_llama32_1b_fomc_scienceqa        PENDING             0:00                
21301589            bdpm-delta-gpu  CRAFT_CL_evict_evict_sens_llama32_1b_fomc_scienceqa         PENDING             0:00                
21301573            bdpm-delta-gpu  CRAFT_CL_stream_f15l15_llama32_1b_trace8                    PENDING             0:00                
21301588            bdpm-delta-gpu  CRAFT_CL_evict_evict_default_llama32_1b_fomc_scienceqa      PENDING             0:00                
21301587            bdpm-delta-gpu  CRAFT_CL_evict_no_evict_llama32_1b_fomc_scienceqa           PENDING             0:00                
21301620            bhot-delta-gpu  CRAFT_CL_routergate_kl_routing_llama32_1b_trace8            PENDING             0:00                
21301621            bhot-delta-gpu  CRAFT_CL_routergate_learned_gate_llama32_1b_trace8          PENDING             0:00                
21301619            bhot-delta-gpu  CRAFT_CL_trunc_k64_p64_llama32_1b_trace8                    PENDING             0:00                
21301618            bhot-delta-gpu  CRAFT_CL_trunc_k64_p8_llama32_1b_trace8                     PENDING             0:00                
21301617            bhot-delta-gpu  CRAFT_CL_trunc_k512_p16_llama32_1b_trace8                   PENDING             0:00                
21301616            bhot-delta-gpu  CRAFT_CL_trunc_k256_p16_llama32_1b_trace8                   PENDING             0:00                
21301615            bhot-delta-gpu  CRAFT_CL_trunc_k128_p16_llama32_1b_trace8                   PENDING             0:00                
21301614            bhot-delta-gpu  CRAFT_CL_trunc_k64_p16_llama32_1b_trace8                    PENDING             0:00                
21299210            bhot-delta-gpu  CRAFT_CL_merge_fisher_gemma2b_trace8_r2                     RUNNING             4:04:39             
21299652            bhot-delta-gpu  CRAFT_CL_KLroute_R3_evict_beta0p3_gemma2b_oracle            RUNNING             3:21:34             
21299657            bhot-delta-gpu  CRAFT_CL_KLroute_R3_evict_beta0p3_gemma2b_prototype         RUNNING             2:57:35             
21300983            bdpm-delta-gpu  CRAFT_CL_KLroute_R3_evict_beta0p3_gemma2b_perplexity        RUNNING             56:06               
21300981            bdpm-delta-gpu  CRAFT_CL_KLroute_R3_evict_beta0p3_gemma2b_subspace_energy   RUNNING             1:03:53             
[mhossen1@dt-login03 CRAFT-CL]$ 







