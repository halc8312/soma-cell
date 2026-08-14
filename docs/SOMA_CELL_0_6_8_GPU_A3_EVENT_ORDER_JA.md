# SOMA-CELL 0.6.8-GPU A3 event order

この文書は A3 scheduler の順序契約である。コード上の convenience order ではなく、0.6.6 の実際の MRO と virtual dispatch を追跡した結果を固定する。

## 1 world step

1. `Formal066World.step`: 直前 motor sign を保存し、chemostat を適用する。
2. 0.6.1 mechanism fault（期限到来時）を inherited step より前に適用する。
3. P2 environment maintenance と uptake snapshot。
4. dynamic pre-P2 chain:
   - 0.6.4 rule schedule
   - 0.6.3 sentinel/evidence/trigger/precursor transfer
   - Formal06 formal prepare と tissue pre-step
5. P1 patch maintenance / pre-neural。
6. 0.5 corpse step、続いて eDNA diffusion/decay。
7. 0.4 interaction pass 1: living cell ごとに `sense_environment → apply_effectors → surface_exchange`。
   - 0.6.6 grammar ATP/wear は `apply_effectors` 内で支払う。A3 metabolism へ移して二重課金しない。
8. 0.5 interaction pass 2: living cell ごとに corpse contact と eDNA/HGT。
   - HGT は同 step の translation には反映し得るが、その step の grammar control/cost には遡及しない。
9. living cell ごとに environmental damage、続いて一回だけ cell metabolism。
10. 全 cell の metabolism 後に collision、death/division handler、field compact。
11. world age を進め、0.2 novel path と 0.4 patch distance を記録する。
12. P1/P2 post、cue toxicity、tissue post、formal observe、0.6.3 lifecycle update。
13. 0.6.6 daughter recast、HGT grammar trace、washout、rule-sign/finiteness accounting。

washout は同 step で release と remove を完了するため、washout 後の dead scan を追加しない。

## 1 cell metabolism

実 entry は `Formal066ProtoCell.metabolism` であり、0.5→0.4→0.3→0.2 を包む。0.1 metabolism は呼ばれない。A3 path はこの親 monolith と追加 GPU phase を併用せず、以下を一つの authoritative chain として実行する。

1. alive guard、position snapshot、damage viability defer。
2. gene cache refresh と protein pool sync。
3. generic gene reactions（reaction 0, 1, 2, 3 の順）。
4. gene-coded ATP generation と membrane/transporter/nucleotide precursor synthesis。
5. paid maintenance。
6. CPU-authoritative translation（0.3 misfolded protein を含む）。
7. CPU-authoritative replication（0.6.6 dt scale を経て 0.4 copier）。
8. paid membrane/transporter surface assembly。
9. damage generation の固定内部順:
   - active protein → damaged protein
   - damaged protein → aggregate
   - waste → reactive
   - membrane oxidation と severe hydrolysis
   - genome lesion gain
   - CPU RNG による重度 lesion symbol hydrolysis
10. transporter channel smoothing、続いて membrane smoothing。
11. ordinary membrane/transporter decay、ATP/trace decay。
12. A2 waste export。
13. A2 leak（CPU-authoritative particle emission/RNG を含む）。
14. A2 radius relaxation。
15. A2 motion（CPU-authoritative Brownian RNG を含む）。
16. CPU division-state update、base/information viability、cell age increment。
17. repair の固定順:
   - antioxidant
   - chaperone
   - protease/recycle（damaged dict order、次に aggregate）
   - genome lesion repair
   - membrane chemical repair/replacement
18. damage viability、functional-age state。
19. 0.4 sensorimotor learning。
20. 0.5 mobile-element export。
21. 0.6.6 mineral→membrane-precursor paid route。

修復と mobile export、0.6.6 supplemental route は cell age increment 後だが world age increment 前である。

## division / death

- septum planning は metabolism 内、actual split は後の world handler。
- damage segregation plan は world capacity が actual split を受理した場合に限り、一回だけ計算・支払い・消費する。
- `DamageProtoCell.split` より外側の wrapper を迂回しない。neural matter return、controller reset、damage partition、daughter grammar mutation を保持する。
- death hook は lifecycle cleanup、numerical tissue cleanup、neural matter return、corpse/eDNA release の継承順を保持する。
- mobile export は HGT pass より後なので、同 step では摂取されない。

## scheduler event names

world events:

`chemostat`, `pre_p2`, `corpse_edna`, `interaction_surface`, `interaction_hgt`, `cell_metabolism_loop`, `collision_death_division`, `post_p2`, `washout`

cell events:

`surface_exchange`, `gene_refresh`, `generic_reactions`, `precursor_synthesis`, `maintenance`, `translation_cpu`, `replication_cpu`, `surface_assembly`, `protein_damage`, `aggregation`, `reactive_byproduct`, `membrane_oxidation`, `genome_lesion_gain`, `genome_hydrolysis_cpu_rng`, `transporter_smoothing`, `membrane_smoothing`, `ordinary_decay`, `waste_export`, `leak`, `radius`, `motion`, `division_update_cpu`, `base_viability`, `cell_age`, `repair_antioxidant`, `repair_chaperone`, `repair_protease`, `repair_genome`, `repair_membrane`, `damage_viability`, `sensorimotor_learning_cpu`, `mobile_export_cpu`, `formal066_supplemental`, `segregation_plan`, `actual_split_cpu`, `death_release_cpu`

`status` は canonical operation の invocation を表し、生成・移送物量そのものは表さない。canonical site で pure plan、backend、CPU-authoritative method を呼んだ場合は、feature disabled または物量 0 により内部が no-work return しても `executed` とする。cell dead、phase 到達前死亡、同 step daughter、非 admitted split など operation 自体を呼ばなかった場合だけ `skipped` とする。`enabled`、`work_performed`、`amount` は metadata に分離し、例えば `transport=false` でも passive gap exchange を含む A2 surface backend を呼ぶため `surface_exchange.status=executed`、`metadata.enabled=false` となる。重複を no-op で隠さず例外にする。
