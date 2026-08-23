# STEM-EBIC-Simulation-2

TEM 시편에서 EBIC / SEEBIC 시뮬레이션을 이미지 + SIMS 데이터 + 재료 물성표로 수행하는 Jupyter notebook.

## Files

| File | Purpose |
|---|---|
| `ebic_sim.ipynb` | 전체 파이프라인 (이미지 로드 → 스케일바 감지 → 자동 세그멘테이션 → 재료·SIMS 매핑 → E-field / 공핍층 / 밴드 다이어그램 / EBIC / SEEBIC) |
| `Material_table_for_ebic_cal.csv` | 재료 물성 (semi/metal 구분, work function, χ, Eg, permittivity, 유효질량). `"Cal"` 값은 노트북이 BGN + 도핑으로 계산 |
| `SIMSNdata.csv`, `SIMSPdata.csv` | 깊이(nm) vs. 도핑 농도(cm⁻³) 프로파일 |
| `image.tif` | TEM 단면 이미지 (스케일바 포함) |
| `requirements.txt` | Python 의존성 |

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook ebic_sim.ipynb
```

1. `Section 1 (CONFIG)`에서 파일 경로, 기판 타입/농도, 전자빔 keV, 회로 단자 설정.
2. 노트북을 위에서부터 실행. `Section 3` 세그멘테이션 오버레이가 뜨면 각 region_id → Material_name 매핑을 `CONFIG['region_assignment']`에 채우고, `Section 1`부터 다시 실행.
3. `"SIMS"`로 라벨된 region은 SIMS N/P 프로파일로 자동 N/P 결정 (SIMS 영역 bounding box의 위쪽 변이 y=0 표면).
4. 나머지 셀에서 자동으로 도핑맵, E-field, 공핍층, 밴드 다이어그램, EBIC/SEEBIC 맵이 생성됨.

## Physics highlights

- **Bandgap narrowing** (Slotboom form): `|N| ≥ 1e18 cm⁻³`일 때만 적용
- **Arora 1982** 이동도 + SRH 소수 캐리어 수명 → 확산 길이
- **Poisson 적분** (`cumulative_trapezoid`)으로 E-field / 공핍폭 계산 (단순 평균 아님)
- **Collection probability**: 공핍층 내부 1, 외부 `exp(-d/L)`, 강한 필드에서 drift enhancement 결합
- **Kanaya-Okayama** 범위로 generation bulb 크기 결정, TEM foil 두께로 클리핑
- **접촉 분류**: 금속-반도체 work function 차이로 Schottky/Ohmic 자동 판단
