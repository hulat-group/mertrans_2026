import os
import sys
import contextlib
import gc
import csv
import warnings
warnings.filterwarnings('ignore')
import logging
import pandas as pd

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from bert_score import score as bert_score_func
import evaluate
import transformers
import textstat

# Silenciar avisos técnicos
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph import setup_system, run_simplification
from src.signals import normalize_signals, sari, fernandez_huerta


def compute_external_metrics(original, output, reference):
    sari_val = bleu_val = bs_val = 0.0
    if not output or not reference: return sari_val, bleu_val, bs_val
    
    # 1. SARI (Estandardizado en src.signals)
    try: sari_val = sari(original, output, [reference])
    except: pass
    
    # 2. BLEU (Normalizado 0-1)
    try:
        from sacrebleu.metrics import BLEU
        bleu_val = BLEU(effective_order=True).sentence_score(output, [reference]).score / 100
    except:
        try:
            import evaluate
            bleu_val = evaluate.load('bleu').compute(predictions=[output], references=[reference])['bleu']
        except: pass
        
    # 3. BERTScore (Normalizado 0-1)
    try:
        P, R, F1 = bert_score_func([output], [reference], model_type='bert-base-multilingual-cased', lang='es')
        bs_val = F1.mean().item()
    except: pass
        
    return sari_val, bleu_val, bs_val

def main():
    # 1. Cargar configuración
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[FAIL] Error: No se ha encontrado la GOOGLE_API_KEY en el archivo .env")
        return

    print("\n  INICIANDO EVALUACION DEL TRIAL iDEM")
    print("-" * 50)

    # 2. Preparar modelos
    print("  Conectando con Gemini 2.5 Flash...")
    gemini_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key, temperature=0.1)
    
    print("  Preparando RigoChat 7B (Inferencia local)...")
    import torch
    has_gpu = torch.cuda.is_available()
    gpu_layers = -1 if has_gpu else 0
    n_threads = 4 if has_gpu else 8

    print(f"  Hardware detectado: {'GPU NVIDIA' if has_gpu else 'SOLO CPU'}")
    gpu_label = "todas las" if gpu_layers == -1 else f"{gpu_layers} capa{'s' if gpu_layers != 1 else ''}"
    print(f"  Cargando RigoChat GGUF ({n_threads} hilo{'s' if n_threads != 1 else ''}, {gpu_label} GPU)...")
    try:
        model_path = hf_hub_download(
            repo_id="IIC/RigoChat-7b-v2-GGUF",
            filename="rigochat-7b-v2-Q4_K_M.gguf",
            cache_dir="./models"
        )
        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            rigochat_llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                n_threads=n_threads,
                n_gpu_layers=gpu_layers,
                temperature=0.2,
                verbose=False
        )
        print(f"✓ RigoChat listo ({'GPU' if has_gpu else 'CPU'})")
    except Exception as e:
        print(f"[FAIL] Error cargando RigoChat: {e}")
        return

    # 3. Inicializar Sistema y Métricas
    app, glossary, lexical_resources = setup_system(gemini_llm, rigochat_llm)
    
    print("[METRIC] Cargando evaluadores de métricas...")
    # sari_scorer = evaluate.load('sari', trust_remote_code=True) # Usamos versión de la profe
    bleu_scorer = evaluate.load('bleu')
    try:
        print("   -> Cargando BERTScore (esto puede tardar un poco)...")
        bertscore_scorer = evaluate.load('bertscore')
    except Exception as e:
        print(f"   [WARNING] Warning BERTScore: {e}")
        bertscore_scorer = None

    # 4. Cargar Trial CSV
    trial_data = []
    csv_path = 'data/es_trial_document.csv'
    try:
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('original_text'):
                    trial_data.append(row)
    except Exception as e:
        print(f"[FAIL] Error leyendo el CSV: {e}")
        return

    print(f"  Trial cargado: {len(trial_data)} ejemplo{'s' if len(trial_data) != 1 else ''} para procesar.\n")

    results = []
    for i, row in enumerate(trial_data):
        print(f"\r--- Procesando ejemplo {i+1}/{len(trial_data)} ---", end="", flush=True)
        original = row['original_text']
        reference = row.get('simplified_text', "")
        
        # Ejecutar Pipeline
        state = run_simplification(app, glossary, lexical_resources, gemini_llm, original, reference)
        output = state.get('final_output', '')
        
        # Calcular Métricas Externas (SARI, BLEU, BERTScore)
        sari_cur, bleu_cur, bs_cur = compute_external_metrics(original, output, reference)

        # Calcular Legibilidad (Fernandez-Huerta estandarizada 0-1)
        leg_norm = fernandez_huerta(output) if output else 0.0

        # El resumen detallado ya lo imprime run_simplification()
        # Aquí solo añadimos las métricas específicas de evaluación externa para el log
        print(f"[METRIC] Métricas Externas: SARI={sari_cur:.3f} | BLEU={bleu_cur:.3f} | BERTScore={bs_cur:.3f}")
        print("-"*65 + "\n")

        results.append({
            'id': row.get('original_sentence_id', i),
            'strategy': state.get('strategy'),
            'decision': state.get('final_decision'),
            'meaningbert': state.get('final_meaningbert'),
            'bertscore': round(bs_cur, 3),
            'sari': round(sari_cur, 3),
            'bleu': round(bleu_cur, 3),
            'legibilidad': round(leg_norm, 3),
            'original': original,
            'output': output
        })
        
        # Limpieza profunda tras cada ejemplo
        del state
        del output
        gc.collect()

    # 5. Mostrar Resultados
    df = pd.DataFrame(results)
    
    print("\n\n" + "="*85)
    print("🏆 RESUMEN DE RESULTADOS - TRIAL iDEM")
    print("="*85)
    cols = ['id', 'strategy', 'decision', 'sari', 'bleu', 'meaningbert', 'legibilidad']
    print(df[cols].to_string(index=False))
    
    print("\n" + "-"*30)
    print("  PROMEDIOS NORMALIZADOS:")
    print(f"SARI medio:        {df['sari'].mean():.3f}")
    print(f"BLEU medio:        {df['bleu'].mean():.3f}")
    print(f"BERTScore medio:   {df['bertscore'].mean():.3f}")
    if 'meaningbert' in df.columns:
        print(f"MeaningBERT medio: {df['meaningbert'].mean():.3f}")
    print(f"Legibilidad media: {df['legibilidad'].mean():.3f}")
    print("="*70)

    # Guardar a CSV para que el usuario pueda descargarlo si quiere
    df.to_csv("resultados_trial_completo.csv", index=False)
    print("\nOK Resultados guardados en 'resultados_trial_completo.csv'")

if __name__ == "__main__":
    # Robustez Unicode para terminales Windows
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

    output_dir = "output/trial_logs"
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = TeeLogger(f"{output_dir}/trial_terminal_log.txt")
    sys.stderr = sys.stdout

    main()
