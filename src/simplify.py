# -*- coding: utf-8 -*-
import os
import sys
import contextlib
import gc
import csv
import warnings
warnings.filterwarnings('ignore')
import logging
import argparse
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from bert_score import score as bert_score_func
import evaluate
import transformers
import textstat

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

# Asegurar raíz en el path
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
    parser = argparse.ArgumentParser(description='Simplificador On-Demand MER-TRANS')
    parser.add_argument('--input', type=str, required=True, help='Archivo .txt con el formato *Texto a simplificar: ...')
    args = parser.parse_args()

    # 1. Parsear archivo con asteriscos
    original_text = ""
    reference_text = None
    
    if not os.path.exists(args.input):
        print(f"[FAIL] Error: El archivo '{args.input}' no existe.")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("*Texto a simplificar:"):
                original_text = line.replace("*Texto a simplificar:", "").strip()
            elif line.startswith("*Texto de referencia:"):
                ref = line.replace("*Texto de referencia:", "").strip()
                if ref: reference_text = ref

    if not original_text:
        print("[FAIL] Error: No se ha encontrado el marcador '*Texto a simplificar:' en el archivo.")
        return

    # 2. Inicializar sistema
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[FAIL] Error: No se ha encontrado la GOOGLE_API_KEY en el archivo .env")
        return

    print("\n" + "="*55)
    print("  INICIANDO SIMPLIFICACION BAJO DEMANDA - MER-TRANS")
    print("="*55 + "\n")
    
    gc.collect()

    print("  Conectando con Gemini 2.5 Flash para razonamiento lingüístico...")
    gemini_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key, temperature=0.1)
    
    import torch
    has_gpu = torch.cuda.is_available()
    gpu_layers = -1 if has_gpu else 0
    n_threads = 4 if has_gpu else 8

    print(f"  Hardware detectado: {'GPU NVIDIA' if has_gpu else 'SOLO CPU'}")
    gpu_label = "todas las" if gpu_layers == -1 else f"{gpu_layers} capa{'s' if gpu_layers != 1 else ''}"
    print(f"  Cargando RigoChat GGUF ({n_threads} hilo{'s' if n_threads != 1 else ''}, {gpu_label} GPU)...")
    try:
        model_path = hf_hub_download(repo_id="IIC/RigoChat-7b-v2-GGUF", filename="rigochat-7b-v2-Q4_K_M.gguf", cache_dir="./models")
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
        print(f"[FAIL] Error al cargar RigoChat: {e}")
        return

    # Carga de recursos adicionales
    app, glossary, lexical_resources = setup_system(gemini_llm, rigochat_llm)
    bleu_scorer = evaluate.load('bleu')

    # 3. Procesamiento
    print("[BRAIN] Propagando texto a través del pipeline multi-agente...")
    state = run_simplification(app, glossary, lexical_resources, gemini_llm, original_text, reference_text if reference_text else "")
    output = state.get('final_output', '')
    meaningbert = state.get('final_meaningbert', 0.0)

    # 4. Métricas y Normalización
    print("[METRIC] Calculando métricas de calidad y normalizando puntuaciones...")
    raw_signals = state.get('final_signals', {})
    norm = normalize_signals(raw_signals)
    
    # Legibilidad estandarizada (0-1)
    legibilidad_norm = fernandez_huerta(output) if output else 0.0
    
    # Métricas externas
    sari_val, bleu, bs_cur = compute_external_metrics(original_text, output, reference_text if reference_text else "")

    # 5. Salida en .txt
    out_file = "resultado_simplificacion.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("="*75 + "\n")
        f.write("      INFORME DE SIMPLIFICACIÓN - PIPELINE MULTI-AGENTE MER-TRANS\n")
        f.write("="*75 + "\n\n")
        f.write(f"TEXTO ORIGINAL:\n{original_text}\n\n")
        f.write(f"TEXTO SIMPLIFICADO:\n{output}\n\n")
        if reference_text:
            f.write(f"REFERENCIA HUMANA:\n{reference_text}\n\n")
        
        f.write("-" * 75 + "\n")
        f.write("MÉTRICAS NORMALIZADAS (Escala 0-1, donde 1.0 es la máxima calidad):\n")
        f.write(f"  - Fidelidad Semántica (MeaningBERT): {meaningbert:.3f}\n")
        f.write(f"  - Legibilidad (F. Huerta):           {legibilidad_norm:.3f}\n")
        f.write(f"  - Adecuación Longitud de Frase:      {norm.get('longitud_frase', 0):.3f}\n")
        f.write(f"  - Claridad de la Sintaxis:           {norm.get('sintaxis_clara', 0):.3f}\n")
        f.write(f"  - Sencillez del Vocabulario:         {norm.get('sencillez_lexica', 0):.3f}\n")
        
        if reference_text:
            f.write(f"\nMÉTRICAS DE REFERENCIA (Comparación académica):\n")
            f.write(f"  - SARI (Xu et al. 2016): {sari_val:.3f}\n")
            f.write(f"  - BLEU:                  {bleu:.3f}\n")
            f.write(f"  - BERTScore:             {bs_cur:.3f}\n")
        else:
            f.write("\n  [Métricas de referencia omitidas por falta de texto humano comparativo]\n")
        
        f.write("-" * 75 + "\n")
        f.write(f"Estrategia Empleada: {state.get('strategy', 'N/A')}\n")
        f.write(f"Estado de Decisión:  {state.get('final_decision', 'N/A')}\n")
        f.write("="*75 + "\n")

    print(f"\nOK ¡Éxito! Informe detallado generado en '{out_file}'\n")

    # 6. Salida de Traza al estilo run_parallel
    trace_file = "traza_simplificacion.txt"
    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\nID: Individual\nORIGINAL: {original_text}\n\n")
        f.write(f"CANDIDATO A: {state.get('candidate_a','')}\n")
        f.write(f"CANDIDATO B: {state.get('candidate_b','')}\n")
        f.write(f"CANDIDATO C: {state.get('candidate_c','')}\n\n")
        for k in ['a','b','c']:
            ev = state.get('evaluations',{}).get(k,{})
            if ev:
                f.write(f"  [{k.upper()}] {ev.get('action','')} -> {ev.get('reason','')}\n")
                f.write(f"       MB={ev.get('meaningbert',0):.3f} | FH={ev.get('fernandez_huerta',0):.3f}\n")
        f.write(f"\nESTRATEGIA: {state.get('strategy','')}\n")
        f.write(f"SALIDA FINAL: {state.get('final_output','')}\n")
        f.write(f"DECISIÓN: {state.get('final_decision','')} -> {state.get('final_reason','')}\n{'='*70}\n\n")
    print(f"OK ¡Éxito! Traza de ejecución generada en '{trace_file}'\n")

    # Limpieza final
    del state
    gc.collect()

if __name__ == '__main__':
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

    output_dir = "output/simplification_logs"
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = TeeLogger(f"{output_dir}/simplify_terminal_log.txt")
    sys.stderr = sys.stdout

    main()
