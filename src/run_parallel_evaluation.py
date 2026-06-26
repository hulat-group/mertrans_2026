"""
run_parallel_evaluation.py - Evaluación batch con checkpoints y trazas.

Uso:
    python run_parallel_evaluation.py --corpus constitucion --traces --n 15
    python run_parallel_evaluation.py --corpus clinicos --resume
    python run_parallel_evaluation.py --corpus test
"""
import os, sys, argparse, contextlib, gc, csv, json, time, warnings, logging
from datetime import datetime

warnings.filterwarnings('ignore')
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from bert_score import score as bert_score_func
import textstat, transformers

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph import setup_system, run_simplification
from src.signals import normalize_signals, fernandez_huerta, sari


CORPUS_PATHS = {
    "constitucion": "data/constitucion_articulos_test.csv",
    "clinicos": "data/casos_clinicos.csv",
    "test": "data/es_test_data.csv",
    "test_mini": "data/es_test_data.csv",
}

FIELDNAMES = [
    'id','document_id','original','referencia','salida_final','estrategia_final',
    'action_final','MeaningBERT','SARI','BERTScore','BLEU','Fernandez_Huerta',
    'avg_sentence_len','split_ratio','lexical_simplification_ratio',
    'difficult_word_ratio','glossary_hit_unresolved','unsupported_content_ratio',
    'retries_usados','motivo','tiempo_seg'
]

def load_corpus(name, max_n=None, start_id=None, end_id=None):
    path = CORPUS_PATHS.get(name)
    if not path or not os.path.exists(path):
        print(f"[FAIL] Corpus '{name}' no encontrado"); sys.exit(1)
    data = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row.get('original_text','').strip(): continue
            
            # Filtro por ID si se proporciona (Constitución)
            if start_id is not None and end_id is not None:
                try:
                    # Buscamos el ID en 'original_sentence_id' o 'id'
                    sid = row.get('original_sentence_id') or row.get('id')
                    if sid:
                        curr_id = int(sid)
                        if not (start_id <= curr_id <= end_id):
                            continue
                except: continue
                
            data.append(row)
    return data[:max_n] if max_n else data

def compute_external_metrics(original, output, reference):
    sari_val = bleu_val = bs_val = 0.0
    if not output or not reference: return sari_val, bleu_val, bs_val
    
    # SARI 
    try: sari_val = sari(original, output, [reference])
    except: pass
    
    # BLEU 
    try:
        from sacrebleu.metrics import BLEU
        bleu_val = BLEU(effective_order=True).sentence_score(output, [reference]).score / 100
    except:
        try:
            import evaluate
            bleu_val = evaluate.load('bleu').compute(predictions=[output], references=[reference])['bleu']
        except: pass
        
    # BERTScore
    try:
        P, R, F1 = bert_score_func([output], [reference], model_type='bert-base-multilingual-cased', lang='es')
        bs_val = F1.mean().item()
    except: pass
        
    return sari_val, bleu_val, bs_val

# =========================================================
# CHECKPOINT HELPERS
# =========================================================
def count_partial(filepath):
    if not os.path.exists(filepath): return 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in csv.DictReader(f))
    except: return 0

def load_processed_ids(filepath):
    ids = set()
    if not os.path.exists(filepath): return ids
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                ids.add((row.get('document_id',''), str(row.get('id',''))))
    except: pass
    return ids

# =========================================================
# REPORT GENERATION
# =========================================================
def generate_reports(df, output_dir, corpus, traces, total):
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV completo
    csv_path = f"{output_dir}/resultados_{corpus}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV: {csv_path}")

    # Trazas
    if traces:
        traces_path = f"{output_dir}/trazas_{corpus}.json"
        with open(traces_path, 'w', encoding='utf-8') as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        
        traces_txt = f"{output_dir}/trazas_{corpus}.txt"
        with open(traces_txt, 'w', encoding='utf-8') as f:
            for t in traces:
                f.write(f"{'='*70}\nID: {t['id']}\nORIGINAL: {t['original']}\n\n")
                f.write(f"CANDIDATO A: {t.get('candidate_a','')}\n")
                f.write(f"CANDIDATO B: {t.get('candidate_b','')}\n")
                f.write(f"CANDIDATO C: {t.get('candidate_c','')}\n\n")
                for k in ['a','b','c']:
                    ev = t.get('evaluations',{}).get(k,{})
                    if ev:
                        f.write(f"  [{k.upper()}] {ev.get('action','')} -> {ev.get('reason','')}\n")
                        f.write(f"       MB={ev.get('meaningbert',0):.3f} | FH={ev.get('fernandez_huerta',0):.3f}\n")
                f.write(f"\nESTRATEGIA: {t.get('strategy','')}\n")
                f.write(f"SALIDA FINAL: {t.get('salida_final','')}\n")
                f.write(f"DECISIÓN: {t.get('decision_final','')} -> {t.get('motivo','')}\n{'='*70}\n\n")
        print(f"[OK] Trazas: {traces_txt}")

    # Resumen cuantitativo
    report_path = f"{output_dir}/resumen_{corpus}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\nRESUMEN - {corpus.upper()} | {datetime.now():%Y-%m-%d %H:%M}\nTotal: {total}\n{'='*70}\n\n")
        f.write("ESTRATEGIAS\n" + "-"*30 + "\n")
        for s in ['V1','V2','V3','RETRY']:
            c = len(df[df['estrategia_final']==s])
            f.write(f"  {s}: {c} ({c/total*100:.1f}%)\n") if total else None
        f.write("\nACCIONES FINALES\n" + "-"*30 + "\n")
        for a in df['action_final'].unique():
            c = len(df[df['action_final']==a])
            f.write(f"  {a}: {c} ({c/total*100:.1f}%)\n") if total else None
        f.write("\nMÉTRICAS\n" + "-"*30 + "\n")
        for m in ['MeaningBERT','SARI','BERTScore','BLEU','Fernandez_Huerta','avg_sentence_len','difficult_word_ratio','unsupported_content_ratio']:
            if m in df.columns: f.write(f"  {m}: {df[m].mean():.4f}\n")
        f.write(f"\nRETRIES: media={df['retries_usados'].mean():.2f}, con_retry={len(df[df['retries_usados']>0])}\n")
    print(f"OK Resumen: {report_path}")

    # Muestra cualitativa
    sample_path = f"{output_dir}/muestra_cualitativa_{corpus}.txt"
    n_s = 10 if total >= 50 else 5
    with open(sample_path, 'w', encoding='utf-8') as f:
        def ws(title, subset):
            f.write(f"\n{'='*70}\n{title} (top {n_s})\n{'='*70}\n")
            for _, r in subset.head(n_s).iterrows():
                f.write(f"\n  ID: {r['id']} | Doc: {r['document_id']}\n")
                f.write(f"  Orig: {str(r['original'])[:200]}\n  Sal:  {str(r['salida_final'])[:200]}\n")
                f.write(f"  Ref:  {str(r['referencia'])[:200]}\n")
                f.write(f"  {r['estrategia_final']} | {r['action_final']} | MB={r['MeaningBERT']:.3f} SARI={r['SARI']:.3f} FH={r['Fernandez_Huerta']:.3f}\n")
        df['_c'] = df['MeaningBERT']*0.4 + df['SARI']*0.3 + df['BERTScore']*0.3
        ws("--- MEJORES", df.sort_values('_c', ascending=False))
        ws("--- PEORES", df.sort_values('_c'))
        ws("--- DUDOSOS", df[df['action_final'].str.contains('WARNING|REVIEW', case=False, na=False)])
        ws("--- V3", df[df['estrategia_final']=='V3'])
        df.drop(columns=['_c'], inplace=True)
    print(f"OK Cualitativa: {sample_path}")

# =========================================================
# MAIN
# =========================================================
def run_evaluation(args):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: print("[!] GOOGLE_API_KEY no encontrada"); return

    corpus = args.corpus
    
    start_id = end_id = None
    if args.range:
        try:
            start_id, end_id = map(int, args.range.split('-'))
            print(f"  Filtro por rango de IDs: {start_id} - {end_id}")
        except:
            print("[!] Formato de rango incorrecto. Use '127-141'.")

    max_n = args.n if args.n else (20 if corpus == "test_mini" else None)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"output/{corpus}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    import sys
    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")
        def write(self, message):
     
            try:
                self.terminal.write(message)
            except UnicodeEncodeError:
                self.terminal.write(message.encode('ascii', 'replace').decode('ascii'))
            
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def isatty(self):
            return self.terminal.isatty()

    sys.stdout = TeeLogger(f"{output_dir}/terminal_log.txt")
    sys.stderr = sys.stdout

    # Checkpoint files
    results_partial = f"output/{corpus}_results.partial.csv"
    traces_partial = f"output/{corpus}_traces.partial.json"

    print(f"\n{'='*70}")
    print(f"  EVALUACION BATCH - {corpus.upper()} | Trazas: {'Si' if args.traces else 'No'}")
    print(f"{'='*70}\n")

    # Resume logic
    processed_ids = set()
    resume_mode = args.resume

    if count_partial(results_partial) > 0:
        count = count_partial(results_partial)
        if not args.resume:
            print(f"[WARNING]  Progreso previo detectado: {count} ejemplo{'s' if count != 1 else ''} en checkpoint.")
            while True:
                choice = input("   ¿[R]eanudar, [S]obrescribir o [Q]salir? (R/S/Q): ").strip().lower()
                if choice == 'r': resume_mode = True; break
                elif choice == 's':
                    for f in [results_partial, traces_partial]:
                        if os.path.exists(f): os.remove(f)
                    resume_mode = False; break
                elif choice == 'q': print("Saliendo."); return
                else: print("[!] Opcion no valida.")
        else:
            resume_mode = True

    if resume_mode:
        processed_ids = load_processed_ids(results_partial)
        print(f"  Reanudando: {len(processed_ids)} ejemplo{'s' if len(processed_ids) != 1 else ''} ya procesado{'s' if len(processed_ids) != 1 else ''}.\n")

    # Load existing traces
    existing_traces = []
    if resume_mode and os.path.exists(traces_partial):
        try:
            with open(traces_partial, 'r', encoding='utf-8') as f:
                existing_traces = json.load(f)
        except: pass

    # Setup models
    print("  Conectando con Gemini 2.5 Flash...")
    gemini_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key, temperature=0.1)

    print("  Preparando RigoChat 7B...")
    import torch
    has_gpu = torch.cuda.is_available()
    gpu_layers = -1 if has_gpu else 0
    n_threads = 4 if has_gpu else 8

    print(f"  Hardware detectado: {'GPU NVIDIA' if has_gpu else 'SOLO CPU'}")
    gpu_label = "todas las" if gpu_layers == -1 else f"{gpu_layers} capa{'s' if gpu_layers != 1 else ''}"
    print(f"  Cargando RigoChat GGUF ({n_threads} hilo{'s' if n_threads != 1 else ''}, {gpu_label} GPU)...")

    try:
        model_path = hf_hub_download(repo_id="IIC/RigoChat-7b-v2-GGUF",
            filename="rigochat-7b-v2-Q4_K_M.gguf", cache_dir="./models")
        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            rigochat_llm = LlamaCpp(
                model_path=model_path, 
                n_ctx=2048,
                n_threads=n_threads, 
                n_gpu_layers=gpu_layers,
                temperature=0.2, 
                verbose=False
            )
    except Exception as e:
        print(f"[!] Error cargando RigoChat: {e}"); return

    app, glossary, lexical_resources = setup_system(gemini_llm, rigochat_llm)
    data = load_corpus(corpus, max_n, start_id, end_id)
    print(f"  Corpus: {len(data)} ejemplo{'s' if len(data) != 1 else ''} | Pendientes: {len(data) - len(processed_ids)}\n")

    # Processing loop with checkpoint
    mode = 'a' if resume_mode and os.path.exists(results_partial) else 'w'
    traces_list = list(existing_traces)

    with open(results_partial, mode, encoding='utf-8', newline='') as f_res:
        writer = csv.DictWriter(f_res, fieldnames=FIELDNAMES)
        if mode == 'w': writer.writeheader()

        for i, row in enumerate(data):
            doc_id = row.get('document_id', '')
            sent_id = row.get('original_sentence_id', str(i))
            original = row['original_text']
            reference = row.get('adaptado', row.get('simplified_text', ''))

            if (doc_id, sent_id) in processed_ids:
                continue

            print(f"\n{'-'*70}")
            print(f"  [{i+1}/{len(data)}] {doc_id} (ID {sent_id})")

            try:
                t0 = time.time()
                state = run_simplification(app, glossary, lexical_resources, gemini_llm, original, reference)
                elapsed = time.time() - t0
                output = state.get('final_output', '')

                sari_val, bleu_val, bs_val = compute_external_metrics(original, output, reference)

                # Legibilidad estandarizada (0-1)
                fh_ext = fernandez_huerta(output) if output else 0.0
                fs = state.get('final_signals', {})

                result = {
                    'id': sent_id, 'document_id': doc_id,
                    'original': original, 'referencia': reference,
                    'salida_final': output,
                    'estrategia_final': state.get('strategy', ''),
                    'action_final': state.get('final_decision', ''),
                    'MeaningBERT': round(state.get('final_meaningbert', 0.0), 4),
                    'SARI': round(sari_val, 4), 
                    'BERTScore': round(bs_val, 4),
                    'BLEU': round(bleu_val, 4), 
                    'Fernandez_Huerta': round(fh_ext, 4),
                    'avg_sentence_len': round(fs.get('avg_sentence_len', 0.0), 2),
                    'split_ratio': round(fs.get('split_ratio', 0.0), 2),
                    'lexical_simplification_ratio': round(fs.get('lexical_simplification_ratio', 0.0), 4),
                    'difficult_word_ratio': round(fs.get('difficult_word_ratio', 0.0), 4),
                    'glossary_hit_unresolved': fs.get('glossary_hit_unresolved', 0),
                    'unsupported_content_ratio': round(fs.get('unsupported_content_ratio', 0.0), 4),
                    'retries_usados': state.get('retry_count', 0),
                    'motivo': state.get('final_reason', ''),
                    'tiempo_seg': round(elapsed, 2),
                }
                writer.writerow(result)
                f_res.flush()  # Checkpoint: flush inmediato

                # Trace
                if args.traces:
                    trace = {
                        'id': sent_id, 'original': original,
                        'candidate_a': state.get('candidate_a', ''),
                        'candidate_b': state.get('candidate_b', ''),
                        'candidate_c': state.get('candidate_c', ''),
                        'evaluations': {}, 'strategy': state.get('strategy', ''),
                        'salida_final': output,
                        'decision_final': state.get('final_decision', ''),
                        'motivo': state.get('final_reason', ''),
                    }
                    for k in ['a','b','c']:
                        ev = state.get('evaluations',{}).get(k)
                        if ev:
                            trace['evaluations'][k] = {
                                'action': ev.get('action',''), 'reason': ev.get('reason',''),
                                'meaningbert': ev.get('meaningbert',0),
                                'fernandez_huerta': ev.get('fernandez_huerta',0),
                            }
                    traces_list.append(trace)
                    # Save traces checkpoint
                    with open(traces_partial, 'w', encoding='utf-8') as ft:
                        json.dump(traces_list, ft, ensure_ascii=False, indent=2)

                print(f"   [OK] {state.get('strategy','')} | MB={result['MeaningBERT']:.3f} SARI={sari_val:.3f} BS={bs_val:.3f} FH={fh_ext:.3f} | {elapsed:.1f}s")

            except Exception as e:
                print(f"   [FAIL] Error: {e}")

            gc.collect()

    # Generate final reports
    print(f"\n{'='*70}\n[METRIC] Generando informes...\n{'='*70}")
    df = pd.read_csv(results_partial)
    generate_reports(df, output_dir, corpus, traces_list if args.traces else None, len(df))

    # Terminal summary
    print(f"\n{'='*70}")
    print(f"  RESUMEN - {corpus.upper()} ({len(df)} ejemplo{'s' if len(df) != 1 else ''})")
    print(f"{'='*70}")
    for m in ['MeaningBERT','SARI','BERTScore','BLEU','Fernandez_Huerta']:
        print(f"  {m:<20}: {df[m].mean():.4f}")
    print(f"\n  Estrategias: {dict(df['estrategia_final'].value_counts())}")
    print(f"  Acciones:    {dict(df['action_final'].value_counts())}")
    print(f"  Informes en: {output_dir}/")
    print(f"  Checkpoint: {results_partial}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluación batch con checkpoints")
    p.add_argument("--corpus", required=True, choices=["constitucion","clinicos","test","test_mini"])
    p.add_argument("--n", type=int, default=None, help="Max ejemplos")
    p.add_argument("--traces", action="store_true", help="Generar trazas completas")
    p.add_argument("--resume", action="store_true", help="Reanudar desde checkpoint")
    p.add_argument("--range", type=str, default=None, help="Rango de IDs (ej: 127-141)")
    run_evaluation(p.parse_args())
