| ladder | config | this run GHS | 18c nominal GHS | diff % |
|---|---|---:|---:|---:|
| phase5_ladder_lv2_sourced_mean.csv | oracle | 198,440.41 | 198,440.41 | +0.0000 |
| phase5_ladder_lv2_sourced_mean.csv | baseline | 281,647.09 | 279,630.14 | +0.7213 |
| phase5_ladder_lv2_sourced_mean.csv | roll24 | 237,679.33 | 237,679.33 | +0.0000 |
| phase5_ladder_lv2_sourced_mean.csv | roll24p | 202,372.00 | 202,372.00 | +0.0000 |
| mix12/phase5_ladder_mix12.csv | oracle | 278,039.20 | 278,039.20 | +0.0000 |
| mix12/phase5_ladder_mix12.csv | baseline | 376,515.73 | 364,113.22 | +3.4062 |
| mix12/phase5_ladder_mix12.csv | roll24 | 316,949.85 | 316,949.85 | +0.0000 |
| v1/phase5_ladder_loadv1.csv | baseline | 266,862.81 | 269,064.05 | -0.8181 |
| v1/phase5_ladder_loadv1.csv | roll24 | 229,375.96 | 229,375.96 | +0.0000 |
| phase5_ladder_lv2_generac.csv | oracle | 189,867.86 | 189,867.86 | +0.0000 |
| phase5_ladder_lv2_generac.csv | baseline | 259,440.59 | 242,623.94 | +6.9312 |
| phase5_ladder_lv2_generac.csv | roll24 | 225,720.41 | 225,720.41 | +0.0000 |
| phase5_ladder_lv2_cummins.csv | oracle | 198,184.27 | 198,184.27 | +0.0000 |
| phase5_ladder_lv2_cummins.csv | baseline | 280,949.45 | 278,448.14 | +0.8983 |
| phase5_ladder_lv2_cummins.csv | roll24 | 236,443.79 | 236,443.79 | +0.0000 |
| phase5_ladder_lv2_kohler.csv | oracle | 206,632.96 | 206,632.96 | +0.0000 |
| phase5_ladder_lv2_kohler.csv | baseline | 304,596.75 | 317,856.36 | -4.1716 |
| phase5_ladder_lv2_kohler.csv | roll24 | 249,929.11 | 249,929.11 | +0.0000 |

Largest baseline difference: 6.9312%. EXPECTED to be non-zero: this run's baseline obeys the UC rules, 18c's did not. MILP rows (oracle, roll24, roll24p) should agree within solver noise. Heuristic rows not listed.
