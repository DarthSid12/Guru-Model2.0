"""Add source-labelled human references to the editable report; retain raw outputs."""
import csv,hashlib,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
RESULTS=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'completed-sdsc/results'
HUMAN_SOURCE=ROOT/'HUMAN_SOURCE.json'
SOURCE_METADATA=json.loads((ROOT/'HUMAN_SOURCE_METADATA.json').read_text())
PDF=Path('/Users/darthsid/Downloads/Yin1969.pdf')
CONDITIONS=('UU','UI','II','IU')
PROTOCOLS=('single_uu','study_only_fitted','two_cross')
STIMULI={'faces':'Photographic faces','houses':'Houses','objects':'Airplanes'}
DESIGNS={
 'faces_only':('40 faces',{'faces':40}),
 'houses_only':('40 houses',{'houses':40}),
 'objects_only':('40 objects',{'objects':40}),
 'faces_houses':('20 faces + 20 houses',{'faces':20,'houses':20}),
 'faces_objects':('20 faces + 20 objects',{'faces':20,'objects':20}),
 'houses_objects':('20 houses + 20 objects',{'houses':20,'objects':20}),
 'faces_houses_objects':('14 faces + 13 houses + 13 objects',{'faces':14,'houses':13,'objects':13}),
}
LABELS={'single_uu':'Single noise','study_only_fitted':'Fitted study-only noise','two_cross':'Upright/inverted noise on both phases'}
MAP={'Upright study / upright test':'UU','Upright study / inverted test':'UI','Inverted study / inverted test':'II','Inverted study / upright test':'IU'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 source=json.loads(HUMAN_SOURCE.read_text());summary_path=RESULTS/'SUMMARY.json';original_hash=sha(summary_path)
 summary=json.loads(summary_path.read_text());human={};rows=[]
 for category,stimulus in STIMULI.items():
  for row in source['yin_conditions']:
   if row['stimulus_set']!=stimulus or row['experiment'] not in ['I','II']:continue
   condition=MAP[row['condition']];n=row['test_decisions_per_category_per_condition'];assert n==12
   assert row['study_images_per_category_per_block']==20
   mean=100*(1-row['mean_errors_reported']/n);sd=100*row['sd_errors_reported']/n
   human[category,condition]={
    'category_reference_for':category,'condition':condition,'stimulus_set':stimulus,
    'mean_pct':mean,'sd_pct':sd,'sem_pct':sd/math.sqrt(row['n_participants']),
    'n_participants':row['n_participants'],'experiment':row['experiment'],
    'study_images_per_category':20,'memory_items':40,'test_pairs_per_category':n,
    'mean_errors_reported':row['mean_errors_reported'],'sd_errors_reported':row['sd_errors_reported'],
    'source_location':row['source_location'],
    'reference_type':'historical_airplane_reference_for_wordbank_objects' if category=='objects' else 'historical_reference_with_different_stimuli',
   }
 assert len(human)==12
 assert all(human[c,k]['n_participants']==(26 if k in ['UU','II'] else 21) for c in STIMULI for k in CONDITIONS)
 human_effects={cat:{'mean_pct':human[cat,'UU']['mean_pct']-human[cat,'II']['mean_pct'],'sd_pct':None,'sem_pct':None,'n_participants':26,'source':'Yin (1969), Experiment I, Table 1','note':'Difference-score SD/SEM cannot be recovered from marginal SDs without paired covariance.'} for cat in STIMULI}
 for cell in summary['cells']:
  h=human[cell['category'],cell['condition']]
  rows.append({
   'model':cell['model'],'design':cell['design'],'category':cell['category'],'protocol':cell['protocol'],'condition':cell['condition'],
   'model_mean_pct':cell['mean_pct'],'model_sd_pct':cell['sd_pct'],'model_sem_pct':cell['sem_pct'],'model_simulation_n':cell['n'],
   'model_study_items_in_category':DESIGNS[cell['design']][1][cell['category']],'model_test_pairs_in_category':cell['n_trials'],
   'p1':cell['p1'],'p2':cell['p2'],
   'human_stimulus_set':h['stimulus_set'],'human_mean_pct':h['mean_pct'],'human_sd_pct':h['sd_pct'],'human_sem_pct':h['sem_pct'],
   'human_n_participants':h['n_participants'],'human_experiment':h['experiment'],
   'human_study_items_in_category':20,'human_test_pairs_in_category':12,
   'human_reference_type':h['reference_type'],'human_source_location':h['source_location'],
  })
 assert len(rows)==len(summary['cells']) and len({tuple(r[k] for k in ['model','design','category','protocol','condition']) for r in rows})==len(summary['cells'])
 extra={
  'source_model_summary_sha256':original_hash,'source_human_extraction_sha256':sha(HUMAN_SOURCE),
  'source_pdf':str(PDF),'source_pdf_sha256':SOURCE_METADATA['source_pdf_sha256'],
  'source_citation':'Yin, R. K. (1969). Looking at upside-down faces. Journal of Experimental Psychology, 81(1), 141–145. Tables 1–2.',
  'human_conditions':list(human.values()),'human_inversion_effect_UU_minus_II':human_effects,
  'comparisons':rows,'model_inversion_effect_UU_minus_II':summary['inversion_effect_UU_minus_II'],
  'notes':[
   'Human references are the same published observations repeated for comparison, not new independent groups for each model, design or noise protocol.',
   'Human Experiments I/II studied40 mixed pictures,20/category, and tested12 old/new pairs/category. Model single-category designs study40/category; the triple design studies14/13/13. Human values are not measured results for these model designs.',
   'DalHouses residential photographs, CFD faces and Wordbank objects differ from original human stimuli. Airplane references are not matched human Wordbank object data.',
   'Displayed uncertainty is participant SD for humans and sample SD across50 simulation seeds for models. SEM is SD/sqrt(n), with n=26/21 participants for humans and50 simulation seeds for models.',
   'Human UU/II are from Experiment I,n26; UI/IU are from Experiment II,n21.',
   'Human accuracy=100*(1-mean_errors/12); human SD=100*error_SD/12. Derived from rounded source errors/SDs.',
   'Human UU–II mean is shown; no human difference-score SD or SEM is inferred without paired covariance.',
   'Model face UU is a calibration anchor in all protocols; face II is also fitted in the two-noise protocols. These fitted face conditions are not held-out validation.',
  ],
 }
 (RESULTS/'SUMMARY_WITH_HUMANS.json').write_text(json.dumps(extra,indent=2,allow_nan=False)+'\n')
 with (RESULTS/'human_model_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=[
  '# ResNet-18 Yin results: unseen DalHouses with human references','',
  'Each study/category group places the published human reference beside the Delta LP model. Values are **accuracy % ± SD**; UU–II is an inversion drop in percentage points.','',
  '**Human reference:** Yin (1969), Table 1 (p. 142): UU/II, n = 26; Table 2 (p. 143): UI/IU, n = 21. Humans studied 40 mixed pictures, **20 per category**, with **12 test pairs per category**. Accuracy = 100 × (1 − errors/12); SD = 100 × error SD/12.','',
  'Human SD describes participants; model SD describes 50 simulation seeds. Human references are repeated for comparison across model designs and noise protocols; they are the same published data each time.','',
  '**Comparison limits:** model stimuli differ from Yin’s photographs, and human values are not measurements of the model’s 40-per-category or three-category designs. **Human airplane values are a historical object reference, not matched Wordbank object data.** Houses are64 separately screened DalHouses residential identities, with no identified overlap with model training; the identity audit is frozen with this run.','',
  'Human UU–II SD cannot be recovered from the published marginal SDs and is not estimated. Face UU is fitted in every model protocol; face II is also fitted in the two-noise protocols.','',
  f'[Source paper]({PDF}) · [Full statistics with human SD and SEM]({RESULTS}/SUMMARY_WITH_HUMANS.json) · [Comparison CSV]({RESULTS}/human_model_comparison.csv)','',
 ]
 cell_index={tuple(c[k] for k in ['model','design','category','protocol','condition']):c for c in summary['cells']}
 contrast_index={tuple(c[k] for k in ['model','design','category','protocol']):c for c in summary['inversion_effect_UU_minus_II']}
 def fmt(c):return f"{c['mean_pct']:.2f} ± {c['sd_pct']:.2f}"
 for protocol in PROTOCOLS:
  lines.extend([f'## {LABELS[protocol]} (`{protocol}`)',''])
  if protocol=='single_uu':lines.append('Study/test noise = p1/p1 in every condition.')
  elif protocol=='study_only_fitted':lines.append('UU/UI use p1/p1; II/IU use p2/p1 (study/test noise).')
  else:lines.append('UU: p1/p1; UI: p1/p2; II: p2/p2; IU: p2/p1 (study/test noise).')
  lines.extend(['','| Study / outcome | Source | p1, p2 | UU | UI | II | IU | UU–II |','|---|---|---|---:|---:|---:|---:|---:|'])
  for design,(label,cats) in DESIGNS.items():
   for category in cats:
    context=f'{label} / {category}'
    hl='Human airplanes (object reference)' if category=='objects' else 'Human '+category+' (reference)'
    hv=' | '.join(fmt(human[category,c]) for c in CONDITIONS)
    lines.append(f"| {context} | **{hl}** | — | {hv} | {human_effects[category]['mean_pct']:.2f} (SD unavailable) |")
    for model in summary['models']:
     values=[cell_index[model['model'],design,category,protocol,c] for c in CONDITIONS]
     contrast=contrast_index[model['model'],design,category,protocol]
     vals=' | '.join(fmt(v) for v in values)
     lines.append(f"| {context} | {model['label']} | {values[0]['p1']:.2f}, {values[0]['p2']:.2f} | {vals} | {fmt(contrast)} |")
  lines.append('')
 dest=RESULTS/'SUMMARY.md';backup=RESULTS/'SUMMARY_MODELS_ONLY.md'
 if not backup.exists():backup.write_bytes(dest.read_bytes())
 dest.write_text('\n'.join(lines)+'\n')
 assert sha(summary_path)==original_hash,'Keep validated model JSON unchanged'
 assert dest.read_text().count('| **Human ')==36
 assert '92.58 ± 9.08' in dest.read_text() and '81.42 ± 13.33' in dest.read_text() and '69.58 ± 14.08' in dest.read_text()
 validated=json.loads((RESULTS/'VALIDATED.json').read_text());assert validated['summary_sha256']==sha(summary_path) and validated['csv_sha256']==sha(RESULTS/'all_trials.csv')
 verification={'passed':True,'model_cells':len(summary['cells']),'human_condition_references':12,'human_rows_displayed':36,'model_json_unchanged':True,'raw_model_csv_unchanged':True,'report_sha256':sha(dest),'comparison_json_sha256':sha(RESULTS/'SUMMARY_WITH_HUMANS.json'),'comparison_csv_sha256':sha(RESULTS/'human_model_comparison.csv')}
 (ROOT/'HUMAN_COMPARISON_VERIFIED.json').write_text(json.dumps(verification,indent=2)+'\n')
 print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
